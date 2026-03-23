"""
goal_processor.py — Autonomous goal processing background task.

Periodically scans for active goals with no plan steps, proposes a plan
to the user via notifier, waits for user approval, then executes steps
serially — respecting step ownership (model vs human).

Lifecycle:
  1. Scanner finds active goals with auto_process_status IS NULL and no plan steps
  2. Decomposer creates concept + task steps via plan-decomposer LLM
  3. Proposal sent to user via notifier (goal_plan_proposed event)
  4. User responds: approve / defer / reject / modify
  5. Executor runs model-owned steps serially; pauses on human steps
  6. Completion cascades via existing plan_engine logic

Config (plugins-enabled.json → plugin_config.goal_processor):
    enabled:                bool   — master switch (default false)
    interval_m:             int    — minutes between scans (default 30)
    model:                  str    — decomposer model key (default "plan-decomposer")
    max_goals_per_cycle:    int    — max goals to propose per cycle (default 3)
    max_exec_steps_per_cycle: int  — max steps to execute per cycle (default 10)
    defer_cooldown_hours:   int    — hours before re-proposing deferred goals (default 24)

Runtime control:
    get_stats()       → dict of counters + last-run info
    trigger_now()     → wake sleeping loop immediately
    run_goal_processor() → manual single-cycle run
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta

log = logging.getLogger("goal_processor")

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_stats: dict = {
    "runs":                 0,
    "goals_scanned":        0,
    "plans_proposed":       0,
    "steps_executed":       0,
    "last_run_at":          None,
    "last_run_duration_s":  None,
    "last_error":           None,
}

_wake_event: asyncio.Event | None = None

# Maps goal_id → client_id that approved the goal (for direct progress notifications).
# Populated by register_initiator(), cleared on completion/error. Lost on restart — acceptable.
_initiator_sessions: dict[int, str] = {}

_PLUGINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")


def get_stats() -> dict:
    return dict(_stats)


def register_initiator(goal_id: int, client_id: str) -> None:
    """Record which session approved a goal so progress goes back to them."""
    _initiator_sessions[goal_id] = client_id


def trigger_now() -> None:
    if _wake_event:
        _wake_event.set()


async def _notify_initiator(goal_id: int, msg: str, fallback_event: str = "") -> None:
    """Send a progress message to the session that approved this goal.

    Falls back to fire_event (broadcast) if no initiator is registered.
    """
    client_id = _initiator_sessions.get(goal_id)
    if client_id:
        try:
            from state import push_notif
            await push_notif(client_id, msg)
            return
        except Exception as e:
            log.warning(f"goal_processor: push to initiator {client_id} failed: {e}")
    # Fallback: broadcast via notifier
    if fallback_event:
        from notifier import fire_event
        await fire_event(fallback_event, msg)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    try:
        with open(_PLUGINS_PATH) as f:
            raw = json.load(f).get("plugin_config", {}).get("goal_processor", {})
    except Exception:
        raw = {}
    return {
        "enabled":                  raw.get("enabled", False),
        "interval_m":               int(raw.get("interval_m", 30)),
        "model":                    raw.get("model", ""),  # override; empty = use model_roles
        "max_goals_per_cycle":      int(raw.get("max_goals_per_cycle", 3)),
        "max_exec_steps_per_cycle": int(raw.get("max_exec_steps_per_cycle", 10)),
        "defer_cooldown_hours":     int(raw.get("defer_cooldown_hours", 24)),
    }


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _GOALS():
    from memory import _GOALS as _G
    return _G()

def _PLANS():
    from memory import _PLANS as _P
    return _P()


# ---------------------------------------------------------------------------
# Scanner — find active goals with no plan steps
# ---------------------------------------------------------------------------

async def _scan_unplanned_goals(max_goals: int) -> list[dict]:
    """Find active goals eligible for autonomous processing."""
    from database import fetch_dicts

    rows = await fetch_dicts(
        f"SELECT g.* FROM {_GOALS()} g "
        f"WHERE g.status = 'active' "
        f"AND g.auto_process_status IS NULL "
        f"AND NOT EXISTS ("
        f"  SELECT 1 FROM {_PLANS()} p WHERE p.goal_id = g.id"
        f") "
        f"ORDER BY g.importance DESC, g.created_at ASC "
        f"LIMIT {max_goals}"
    )
    return rows or []


async def _scan_deferred_goals() -> list[dict]:
    """Find deferred goals whose cooldown has expired."""
    from database import fetch_dicts

    rows = await fetch_dicts(
        f"SELECT * FROM {_GOALS()} "
        f"WHERE status = 'active' "
        f"AND auto_process_status = 'deferred' "
        f"AND (defer_until IS NULL OR defer_until <= NOW()) "
        f"ORDER BY importance DESC "
        f"LIMIT 5"
    )
    return rows or []


# ---------------------------------------------------------------------------
# Proposer — decompose goal and notify user
# ---------------------------------------------------------------------------

async def _propose_plan(goal: dict, model_key: str) -> bool:
    """Create concept step, decompose into tasks, notify user. Returns True on success."""
    import plan_engine
    from database import execute_sql
    from notifier import fire_event

    goal_id = goal["id"]
    title = goal.get("title", f"goal-{goal_id}")

    try:
        # Create a single concept step from the goal description
        concept_id = await plan_engine.create_concept_step(
            description=goal.get("description") or title,
            goal_id=goal_id,
            step_order=1,
            source="assistant",
            target="model",
            approval="proposed",
        )

        # Decompose into task steps (not auto-approved — user must review)
        tasks = await plan_engine.decompose_concept_step(
            concept_id, model_key=model_key, auto_approve=False
        )

        # Mark goal as proposed
        await execute_sql(
            f"UPDATE {_GOALS()} SET auto_process_status = 'proposed' "
            f"WHERE id = {goal_id}"
        )

        # Build notification summary
        step_lines = []
        for t in tasks:
            owner = "you" if t.get("target") == "human" else "assistant"
            step_lines.append(f"  {t.get('step_order', '?')}. [{owner}] {t.get('description', '')}")
        steps_text = "\n".join(step_lines)

        model_count = sum(1 for t in tasks if t.get("target") == "model")
        human_count = sum(1 for t in tasks if t.get("target") == "human")

        summary = (
            f"Goal {goal_id}: {title}\n"
            f"{len(tasks)} steps ({model_count} auto, {human_count} user):\n"
            f"{steps_text}\n\n"
            f"!plan auto approve {goal_id}  |  defer {goal_id}  |  reject {goal_id}"
        )

        await fire_event("goal_plan_proposed", summary)
        log.info(f"goal_processor: proposed plan for goal {goal_id} ({len(tasks)} steps)")
        return True

    except Exception as e:
        log.error(f"goal_processor: propose failed for goal {goal_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Executor — run approved goal steps serially
# ---------------------------------------------------------------------------

async def _execute_goal_serial(goal_id: int, max_steps: int) -> dict:
    """
    Execute steps for one goal serially, respecting ownership.
    Returns summary dict.
    """
    from database import fetch_dicts, execute_sql
    import plan_engine

    # Resolve goal title once for notifications
    goal_rows = await fetch_dicts(
        f"SELECT title FROM {_GOALS()} WHERE id = {goal_id}"
    )
    goal_title = goal_rows[0]["title"] if goal_rows else f"goal-{goal_id}"

    # Get all task steps ordered by step_order, not yet done
    steps = await fetch_dicts(
        f"SELECT * FROM {_PLANS()} "
        f"WHERE goal_id = {goal_id} AND step_type = 'task' "
        f"AND status NOT IN ('done', 'skipped') "
        f"ORDER BY step_order "
        f"LIMIT {max_steps}"
    )

    # Count total tasks for progress fraction
    total_row = await fetch_dicts(
        f"SELECT COUNT(*) as total, "
        f"SUM(CASE WHEN status IN ('done','skipped') THEN 1 ELSE 0 END) as completed "
        f"FROM {_PLANS()} WHERE goal_id = {goal_id} AND step_type = 'task'"
    )
    total_tasks = total_row[0]["total"] if total_row else 0
    done_tasks = total_row[0]["completed"] if total_row else 0

    if not steps:
        # Check for pending concept steps FIRST — they may produce new tasks
        pending_concepts = await fetch_dicts(
            f"SELECT id, description FROM {_PLANS()} "
            f"WHERE goal_id = {goal_id} AND step_type = 'concept' "
            f"AND status = 'pending' AND approval = 'approved' "
            f"ORDER BY step_order LIMIT 5"
        )
        if pending_concepts:
            # Notify: decomposition starting
            concept_descs = [c.get("description", "")[:60] for c in pending_concepts]
            await _notify_initiator(
                goal_id,
                f"Goal {goal_id} ({goal_title}): decomposing "
                f"{len(pending_concepts)} concept step(s) into tasks...\n"
                f"  {'; '.join(concept_descs)}"
            )

            decomposed_any = False
            total_new_tasks = 0
            for concept in pending_concepts:
                try:
                    cfg = _cfg()
                    model_key = cfg.get("model") or ""
                    if not model_key:
                        from config import LLM_REGISTRY
                        model_key = LLM_REGISTRY.get("model_roles", {}).get("plan_decomposer", "plan-decomposer")
                    tasks = await plan_engine.decompose_concept_step(
                        concept["id"], model_key=model_key, auto_approve=True
                    )
                    if tasks:
                        decomposed_any = True
                        total_new_tasks += len(tasks)
                        log.info(f"goal_processor: auto-decomposed concept {concept['id']} "
                                 f"into {len(tasks)} tasks for goal {goal_id}")
                except Exception as e:
                    log.error(f"goal_processor: failed to decompose concept {concept['id']}: {e}")
            if decomposed_any:
                # Notify: decomposition complete, starting execution
                await _notify_initiator(
                    goal_id,
                    f"Goal {goal_id} ({goal_title}): decomposed into "
                    f"{total_new_tasks} task(s). Executing..."
                )
                # Re-enter execution now that task steps exist
                return await _execute_goal_serial(goal_id, max_steps)

        # No pending concepts — check if all tasks are done (goal completable)
        if total_tasks > 0 and total_tasks == done_tasks:
            await execute_sql(
                f"UPDATE {_GOALS()} SET auto_process_status = 'completed' "
                f"WHERE id = {goal_id}"
            )
            return {"completed": True, "goal_id": goal_id}

        return {"no_steps": True, "goal_id": goal_id}

    executed = 0
    for step in steps:
        step_id = step["id"]
        target = step.get("target", "model")

        # Human-owned step — pause and notify
        if target == "human":
            await execute_sql(
                f"UPDATE {_GOALS()} SET auto_process_status = 'paused_user' "
                f"WHERE id = {goal_id}"
            )
            await _notify_initiator(
                goal_id,
                f"Goal {goal_id} ({goal_title}) waiting on you:\n"
                f"  Step [{step_id}]: {step.get('description', '')}\n\n"
                f"When done: !plan auto done {goal_id} {step_id}",
                fallback_event="goal_step_waiting_user",
            )
            log.info(f"goal_processor: goal {goal_id} paused at human step {step_id}")
            return {"paused_at": step_id, "reason": "user_step", "goal_id": goal_id}

        # Investigate step — pause (can't auto-execute)
        if target == "investigate":
            await execute_sql(
                f"UPDATE {_GOALS()} SET auto_process_status = 'paused_user' "
                f"WHERE id = {goal_id}"
            )
            await _notify_initiator(
                goal_id,
                f"Goal {goal_id} ({goal_title}) has an unresolved step:\n"
                f"  Step [{step_id}]: {step.get('description', '')}\n"
                f"  (target=investigate — needs manual resolution)",
                fallback_event="goal_step_waiting_user",
            )
            return {"paused_at": step_id, "reason": "investigate", "goal_id": goal_id}

        # Not approved — stop
        if step.get("approval") != "approved":
            log.debug(f"goal_processor: step {step_id} not approved, stopping")
            return {"stopped_at": step_id, "reason": "not_approved", "goal_id": goal_id}

        # Model-owned step — execute
        try:
            result = await plan_engine.execute_task_step(step_id)
            executed += 1
            done_tasks += 1
            _stats["steps_executed"] += 1
            log.info(f"goal_processor: executed step {step_id} for goal {goal_id}")

            # Per-step progress notification
            desc = step.get("description", "")[:80]
            await _notify_initiator(
                goal_id,
                f"Goal {goal_id}: step {done_tasks}/{total_tasks} done — {desc}"
            )

            # Check if execution failed (plan_engine marks step done even on error results)
            if result and ("ERROR" in result.upper() or "failed" in result.lower()):
                log.warning(f"goal_processor: step {step_id} may have failed: {result[:200]}")
                # Revert step to pending and mark goal as paused for user intervention
                await execute_sql(
                    f"UPDATE {_PLANS()} SET status = 'pending' WHERE id = {step_id}"
                )
                await execute_sql(
                    f"UPDATE {_GOALS()} SET auto_process_status = 'paused_user' "
                    f"WHERE id = {goal_id}"
                )
                await _notify_initiator(
                    goal_id,
                    f"Goal {goal_id} ({goal_title}) step [{step_id}] failed:\n"
                    f"  {result[:200]}\n\n"
                    f"Fix the issue or skip: !plan auto done {goal_id} {step_id}",
                    fallback_event="goal_step_waiting_user",
                )
                return {"failed_at": step_id, "result": result[:500], "goal_id": goal_id}

        except Exception as e:
            log.error(f"goal_processor: step {step_id} execution error: {e}")
            await execute_sql(
                f"UPDATE {_GOALS()} SET auto_process_status = 'paused_user' "
                f"WHERE id = {goal_id}"
            )
            await _notify_initiator(
                goal_id,
                f"Goal {goal_id} ({goal_title}) step [{step_id}] error: {e}",
                fallback_event="goal_step_waiting_user",
            )
            return {"error_at": step_id, "error": str(e), "goal_id": goal_id}

    # All remaining steps executed — check completion
    remaining = await fetch_dicts(
        f"SELECT COUNT(*) as cnt FROM {_PLANS()} "
        f"WHERE goal_id = {goal_id} AND step_type = 'task' "
        f"AND status NOT IN ('done', 'skipped')"
    )
    if remaining and remaining[0]["cnt"] == 0:
        await execute_sql(
            f"UPDATE {_GOALS()} SET auto_process_status = 'completed' "
            f"WHERE id = {goal_id}"
        )
        # Let plan_engine cascade handle goal status → done
        concept_rows = await fetch_dicts(
            f"SELECT id FROM {_PLANS()} WHERE goal_id = {goal_id} AND step_type = 'concept'"
        )
        for cr in (concept_rows or []):
            await plan_engine._check_parent_completion(cr["id"])

        # Completion notification with result summary
        done_steps = await fetch_dicts(
            f"SELECT description, result FROM {_PLANS()} "
            f"WHERE goal_id = {goal_id} AND step_type = 'task' AND status = 'done' "
            f"ORDER BY step_order"
        )
        result_lines = []
        for ds in (done_steps or []):
            r = ds.get("result", "") or ""
            if r:
                result_lines.append(f"  - {r[:120]}")
        result_summary = "\n".join(result_lines[-5:]) if result_lines else "  (no result details)"

        await _notify_initiator(
            goal_id,
            f"Goal {goal_id} ({goal_title}) completed — {done_tasks} task(s) done.\n"
            f"Results:\n{result_summary}",
            fallback_event="goal_completed",
        )
        # Clean up initiator tracking
        _initiator_sessions.pop(goal_id, None)
        return {"completed": True, "executed": executed, "goal_id": goal_id}

    return {"executed": executed, "goal_id": goal_id}


# ---------------------------------------------------------------------------
# Core run logic — one cycle
# ---------------------------------------------------------------------------

async def run_goal_processor() -> dict:
    """Run one goal_processor cycle. Safe to call manually."""
    cfg = _cfg()
    max_goals = cfg["max_goals_per_cycle"]
    max_exec = cfg["max_exec_steps_per_cycle"]

    # Resolve decomposer model: config override → model_roles → fallback
    model_key = cfg["model"]
    if not model_key:
        from config import get_model_role
        try:
            model_key = get_model_role("plan_decomposer")
        except KeyError:
            model_key = "plan-decomposer"

    from database import set_db_override, list_managed_databases

    t_start = time.monotonic()
    summary = {
        "scanned": 0, "proposed": 0, "executed_goals": 0,
        "deferred_requeued": 0, "error": None,
    }

    for db_name in list_managed_databases():
        set_db_override(db_name)
        try:
            # Phase 1: Scan for unplanned goals and propose
            unplanned = await _scan_unplanned_goals(max_goals)
            summary["scanned"] += len(unplanned)
            _stats["goals_scanned"] += len(unplanned)

            for goal in unplanned:
                ok = await _propose_plan(goal, model_key)
                if ok:
                    summary["proposed"] += 1
                    _stats["plans_proposed"] += 1

            # Phase 2: Check deferred goals whose cooldown expired
            deferred = await _scan_deferred_goals()
            for goal in deferred:
                from database import execute_sql
                await execute_sql(
                    f"UPDATE {_GOALS()} SET auto_process_status = NULL, defer_until = NULL "
                    f"WHERE id = {goal['id']}"
                )
                summary["deferred_requeued"] += 1
                log.info(f"goal_processor[{db_name}]: re-queued deferred goal {goal['id']}")

            # Phase 3: Execute steps for approved/executing goals
            from database import fetch_dicts
            exec_goals = await fetch_dicts(
                f"SELECT * FROM {_GOALS()} "
                f"WHERE status = 'active' "
                f"AND auto_process_status IN ('approved', 'executing') "
                f"ORDER BY importance DESC"
            ) or []

            exec_budget = max_exec
            for goal in exec_goals:
                if exec_budget <= 0:
                    break
                from database import execute_sql
                await execute_sql(
                    f"UPDATE {_GOALS()} SET auto_process_status = 'executing' "
                    f"WHERE id = {goal['id']}"
                )
                result = await _execute_goal_serial(goal["id"], exec_budget)
                summary["executed_goals"] += 1
                exec_budget -= result.get("executed", 0)

        except Exception as e:
            log.error(f"goal_processor[{db_name}]: run error: {e}")
            summary["error"] = str(e)
            _stats["last_error"] = str(e)
    set_db_override("")

    duration = time.monotonic() - t_start
    _stats["runs"] += 1
    _stats["last_run_at"] = datetime.now(timezone.utc).isoformat()
    _stats["last_run_duration_s"] = round(duration, 2)

    log.info(
        f"goal_processor: cycle done — scanned={summary['scanned']} "
        f"proposed={summary['proposed']} executed={summary['executed_goals']} "
        f"dur={duration:.1f}s"
    )
    return summary


# ---------------------------------------------------------------------------
# Background task entry point
# ---------------------------------------------------------------------------

async def goal_processor_task() -> None:
    """Long-running asyncio task. Loops every interval_m minutes."""
    from timer_registry import register_timer, timer_sleep, timer_start, timer_end

    register_timer("goal_processor", "config")

    global _wake_event
    _wake_event = asyncio.Event()

    # Startup delay — let system stabilize before scanning goals
    await asyncio.sleep(30)

    while True:
        cfg = _cfg()

        if not cfg["enabled"]:
            _wake_event.clear()
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=300)
                _wake_event.clear()
            except asyncio.TimeoutError:
                pass
            continue

        interval_m = cfg["interval_m"]
        if interval_m <= 0:
            await asyncio.sleep(3600)
            continue

        register_timer("goal_processor", f"{interval_m}m")
        t0 = timer_start("goal_processor")
        try:
            await run_goal_processor()
            timer_end("goal_processor", t0)
        except Exception as e:
            log.warning(f"goal_processor_task: unhandled error: {e}")
            _stats["last_error"] = str(e)
            timer_end("goal_processor", t0, error=str(e))

        sleep_sec = interval_m * 60
        timer_sleep("goal_processor", sleep_sec)
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            log.info("goal_processor_task: woken early by trigger")
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass
