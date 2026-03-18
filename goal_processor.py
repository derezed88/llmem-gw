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

_PLUGINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")


def get_stats() -> dict:
    return dict(_stats)


def trigger_now() -> None:
    if _wake_event:
        _wake_event.set()


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
    from notifier import fire_event

    # Get all task steps ordered by step_order, not yet done
    steps = await fetch_dicts(
        f"SELECT * FROM {_PLANS()} "
        f"WHERE goal_id = {goal_id} AND step_type = 'task' "
        f"AND status NOT IN ('done', 'skipped') "
        f"ORDER BY step_order "
        f"LIMIT {max_steps}"
    )

    if not steps:
        # Check if all steps are done — goal might be completable
        all_steps = await fetch_dicts(
            f"SELECT COUNT(*) as total, "
            f"SUM(CASE WHEN status IN ('done','skipped') THEN 1 ELSE 0 END) as completed "
            f"FROM {_PLANS()} WHERE goal_id = {goal_id} AND step_type = 'task'"
        )
        if all_steps and all_steps[0]["total"] > 0 and all_steps[0]["total"] == all_steps[0]["completed"]:
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
            goal_rows = await fetch_dicts(
                f"SELECT title FROM {_GOALS()} WHERE id = {goal_id}"
            )
            goal_title = goal_rows[0]["title"] if goal_rows else f"goal-{goal_id}"
            await fire_event(
                "goal_step_waiting_user",
                f"Goal {goal_id} ({goal_title}) waiting on you:\n"
                f"  Step [{step_id}]: {step.get('description', '')}\n\n"
                f"When done: !plan auto done {goal_id} {step_id}"
            )
            log.info(f"goal_processor: goal {goal_id} paused at human step {step_id}")
            return {"paused_at": step_id, "reason": "user_step", "goal_id": goal_id}

        # Investigate step — pause (can't auto-execute)
        if target == "investigate":
            await execute_sql(
                f"UPDATE {_GOALS()} SET auto_process_status = 'paused_user' "
                f"WHERE id = {goal_id}"
            )
            await fire_event(
                "goal_step_waiting_user",
                f"Goal {goal_id} has an unresolved step:\n"
                f"  Step [{step_id}]: {step.get('description', '')}\n"
                f"  (target=investigate — needs manual resolution)"
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
            _stats["steps_executed"] += 1
            log.info(f"goal_processor: executed step {step_id} for goal {goal_id}")

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
                await fire_event(
                    "goal_step_waiting_user",
                    f"Goal {goal_id} step [{step_id}] failed:\n"
                    f"  {result[:200]}\n\n"
                    f"Fix the issue or skip: !plan auto done {goal_id} {step_id}"
                )
                return {"failed_at": step_id, "result": result[:500], "goal_id": goal_id}

        except Exception as e:
            log.error(f"goal_processor: step {step_id} execution error: {e}")
            await execute_sql(
                f"UPDATE {_GOALS()} SET auto_process_status = 'paused_user' "
                f"WHERE id = {goal_id}"
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

        await fire_event(
            "goal_completed",
            f"Goal {goal_id} auto-completed all steps."
        )
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
