"""
reflection.py — Periodic reflection loop.

Runs as a background asyncio task. On each cycle it:
  1. Pulls recent conversation turns from samaritan_memory_shortterm
     (source IN ('user','assistant'), ordered by created_at DESC, up to
     reflection_turn_limit rows).
  2. Calls reflection_model with a structured prompt:
       "What did I learn? What should I follow up on? What patterns do I notice?"
  3. Parses the JSON response — an array of {topic, content, importance}
     objects — and saves each to the cognition table (origin='reflection')
     via save_cognition().
  4. Goal health outputs (replan/abandon) also go to cognition (origin='goal_health').

Cognition rows are stored in their own table, separate from conversation
memory (ST/LT), so they are not affected by !memreview topic renames or
memory aging cycles. The fuzzy-dedup gate prevents redundant saves between cycles.

Config (plugins-enabled.json → plugin_config.proactive_cognition):
    enabled:                    bool   — master switch
    reflection_enabled:         bool   — this loop (default true when master on)
    reflection_interval_m:      int    — minutes between runs (default 60)
    reflection_model:           str    — model key (default "summarizer-gemini")
    reflection_turn_limit:      int    — max ST rows to pull per cycle (default 40)
    reflection_min_turns:       int    — skip if fewer recent turns than this (default 5)
    reflection_max_memories:    int    — max memory rows to save per cycle (default 6)

Runtime control:
    get_reflection_stats()      → dict of counters + last-run info
    trigger_now()               → wake sleeping loop immediately
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

log = logging.getLogger("reflection")

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_stats: dict = {
    "runs":              0,
    "turns_processed":   0,
    "memories_saved":    0,
    "memories_skipped":  0,
    "last_run_at":       None,
    "last_run_duration_s": None,
    "last_run_saved":    0,
    "last_error":        None,
    "last_feedback":     None,   # last feedback verdict dict
}

# Self-summary refresh every N reflection cycles
_SELF_SUMMARY_EVERY_N = 5

_wake_event: asyncio.Event | None = None

_PLUGINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")


def get_reflection_stats() -> dict:
    return dict(_stats)


def trigger_now() -> None:
    if _wake_event:
        _wake_event.set()


# ---------------------------------------------------------------------------
# Config helper — shares runtime overrides from contradiction.py
# ---------------------------------------------------------------------------

def _rcogn_cfg() -> dict:
    try:
        with open(_PLUGINS_PATH) as f:
            raw = json.load(f).get("plugin_config", {}).get("proactive_cognition", {})
    except Exception:
        raw = {}

    try:
        from contradiction import get_runtime_overrides
        ovr = get_runtime_overrides()
    except ImportError:
        ovr = {}

    base = {
        "enabled":               raw.get("enabled",               False),
        "reflection_enabled":    raw.get("reflection_enabled",    True),
        "reflection_interval_m": int(raw.get("reflection_interval_m", 60)),
        "reflection_model":      raw.get("reflection_model",      ""),
        "reflection_turn_limit": int(raw.get("reflection_turn_limit",  40)),
        "reflection_min_turns":  int(raw.get("reflection_min_turns",   5)),
        "reflection_max_memories": int(raw.get("reflection_max_memories", 6)),
        # Goal health pass — quit logic thresholds
        "goal_health_enabled":          raw.get("goal_health_enabled", True),
        "goal_health_failure_replan":    int(raw.get("goal_health_failure_replan", 3)),
        "goal_health_failure_abandon":   int(raw.get("goal_health_failure_abandon", 5)),
        "goal_health_autonomy_threshold": float(raw.get("goal_health_autonomy_threshold", 0.6)),
        "goal_health_model":            raw.get("goal_health_model", ""),
    }
    base.update(ovr)
    return base


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a reflective memory assistant. You analyse recent conversation turns "
    "and extract durable insights: things learned, patterns noticed, follow-up items. "
    "Be concise. Do not repeat what was said verbatim — synthesise it.\n\n"
    "You also detect behavioral patterns about the AI agent itself. When you notice "
    "repeated successes, failures, or preferences in how the agent operates, emit "
    "rows with prefixed topics:\n"
    "  - self-capability-<slug>: something the agent does well\n"
    "  - self-failure-<slug>: something the agent struggles with or gets wrong repeatedly\n"
    "  - self-preference-<slug>: a stylistic or behavioral tendency the agent exhibits\n"
    "Keep these terse — 1-2 sentences each. Use type='self_model' for these rows.\n\n"
    "You also detect goal completions. If active goals are provided and the conversation "
    "clearly shows one has been achieved, emit a goal_done entry."
)

_USER_PROMPT_TMPL = (
    "Below are recent conversation turns (newest last). "
    "Identify up to {max_memories} distinct insights worth remembering.\n\n"
    "Return a JSON object with two keys:\n"
    '  "memories": array of memory rows\n'
    '  "goals_done": array of goal IDs (integers) clearly completed in these turns\n\n'
    "Each memory row:\n"
    '  {{"topic": "slug-label", "content": "one sentence", '
    '"importance": 1-10, "type": "context|belief|semantic|episodic|procedural|self_model"}}\n\n'
    "Memory rules:\n"
    "- topic: a short dash-separated slug (reuse existing topics when possible)\n"
    "- importance: 7-9 for concrete follow-up items; 5-6 for general observations\n"
    "- type: use 'belief' for inferred facts about user/world; 'semantic' for general "
    "knowledge; 'procedural' for how-to; 'episodic' for specific events; 'context' otherwise\n"
    "- For self-model patterns: use topic prefix 'self-capability-', 'self-failure-', or "
    "'self-preference-' and type='self_model'. Only emit these when a clear pattern is evident.\n"
    "- Skip anything already obvious from the raw text or too ephemeral to be useful\n\n"
    "Goal completion rules:\n"
    "- Only mark a goal done if the conversation clearly shows the objective was achieved\n"
    "- Do NOT mark done based on plans or intent — only on evidence of completion\n"
    "- Return empty array [] for goals_done if no goals were clearly completed\n\n"
    "{active_goals_block}"
    "TURNS:\n{turns_text}"
)


# ---------------------------------------------------------------------------
# Fetch recent turns from ST
# ---------------------------------------------------------------------------

async def _fetch_recent_turns(limit: int) -> list[dict]:
    from database import fetch_dicts
    from memory import _ST
    try:
        return await fetch_dicts(
            f"SELECT id, topic, content, source, created_at "
            f"FROM {_ST()} "
            f"WHERE source IN ('user', 'assistant') "
            f"ORDER BY created_at DESC LIMIT {limit}"
        ) or []
    except Exception as e:
        log.warning(f"reflection: fetch_recent_turns failed: {e}")
        return []


def _format_turns(rows: list[dict]) -> str:
    # Rows come back newest-first; reverse to chronological for the prompt
    lines = []
    for r in reversed(rows):
        src = r.get("source", "?").upper()
        content = (r.get("content") or "").strip()[:400]
        topic = r.get("topic", "")
        lines.append(f"[{src}|{topic}] {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_llm(model_key: str, turns_text: str, max_memories: int,
                   active_goals: list[dict] | None = None) -> tuple[list[dict], list[int]]:
    """
    Returns (memory_items, goal_ids_done).
    memory_items: list of memory row dicts to save.
    goal_ids_done: list of goal IDs the LLM identified as completed.
    """
    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage

    if active_goals:
        goals_lines = "\n".join(
            f"  [id={g['id']}] {g.get('title','')} — {g.get('description','')}"
            for g in active_goals
        )
        goals_block = f"ACTIVE GOALS (check for completion):\n{goals_lines}\n\n"
    else:
        goals_block = ""

    prompt = _USER_PROMPT_TMPL.format(
        turns_text=turns_text,
        max_memories=max_memories,
        active_goals_block=goals_block,
    )
    try:
        if model_key not in LLM_REGISTRY:
            log.warning(f"reflection: unknown model {model_key!r}")
            return [], []
        cfg = LLM_REGISTRY[model_key]
        timeout = cfg.get("llm_call_timeout", 90)
        llm = _build_lc_llm(model_key)
        msgs = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)
    except Exception as e:
        log.warning(f"reflection: LLM call failed: {e}")
        return [], []

    if not raw:
        return [], []

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        parsed = json.loads(cleaned)
        # New format: {"memories": [...], "goals_done": [...]}
        if isinstance(parsed, dict):
            memories = [x for x in parsed.get("memories", []) if isinstance(x, dict)]
            goals_done = [int(x) for x in parsed.get("goals_done", []) if str(x).isdigit()]
            return memories, goals_done
        # Legacy format: bare array (backward compat)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)], []
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"reflection: JSON parse failed: {e}. raw={raw[:200]}")
    return [], []


# ---------------------------------------------------------------------------
# Second-pass goal completion scan (reasoning model)
# ---------------------------------------------------------------------------

_GOAL_SCAN_SYSTEM = (
    "You check whether active goals have been completed based on conversation evidence. "
    "For each goal, determine if the conversation shows the objective was ACTUALLY ACHIEVED "
    "(not just planned, discussed, or intended). Be strict — only mark done when there is "
    "clear evidence the goal's deliverable exists or the action was performed."
)

_GOAL_SCAN_USER = (
    "ACTIVE GOALS:\n{goals_block}\n\n"
    "RECENT CONVERSATION:\n{turns_text}\n\n"
    "Return ONLY a JSON object: {{\"goals_done\": [<id>, ...]}}\n"
    "Return {{\"goals_done\": []}} if no goals were clearly completed.\n"
    "Do NOT include goals that were merely discussed or planned."
)

_GOAL_SCAN_MODEL = "samaritan-reasoning"


async def _scan_goal_completions(turns_text: str,
                                  active_goals: list[dict]) -> list[int]:
    """
    Second-pass goal completion scan using the reasoning model.
    Separated from memory extraction for higher-quality judgment.
    """
    if not active_goals:
        return []

    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage

    model_key = _GOAL_SCAN_MODEL
    if model_key not in LLM_REGISTRY:
        log.warning(f"reflection: goal-scan model {model_key!r} not in registry, skipping")
        return []

    goals_block = "\n".join(
        f"  [id={g['id']}] {g.get('title', '')} — {g.get('description', '')}"
        for g in active_goals
    )
    prompt = _GOAL_SCAN_USER.format(goals_block=goals_block, turns_text=turns_text)

    try:
        cfg = LLM_REGISTRY[model_key]
        timeout = cfg.get("llm_call_timeout", 90)
        llm = _build_lc_llm(model_key)
        msgs = [SystemMessage(content=_GOAL_SCAN_SYSTEM), HumanMessage(content=prompt)]
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)
    except Exception as e:
        log.warning(f"reflection: goal-scan LLM call failed: {e}")
        return []

    if not raw:
        return []

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return [int(x) for x in parsed.get("goals_done", []) if str(x).isdigit()]
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"reflection: goal-scan JSON parse failed: {e}. raw={raw[:200]}")
    return []


# ---------------------------------------------------------------------------
# Goal health pass — failure escalation + proposal + quit logic
# ---------------------------------------------------------------------------
#
# Runs after memory extraction + goal completion in each reflection cycle.
# Three responsibilities:
#
# 1. FAILURE COUNTING — For each active goal, count self-failure-* rows whose
#    content mentions the goal's title/keywords. Increment the goal's
#    failure_count column.
#
# 2. ESCALATION LADDER:
#    - failure_count >= replan_threshold (default 3): ask reasoning model
#      to propose new plan steps (replan). Writes self-failure-* summary.
#    - failure_count >= abandon_threshold (default 5): auto-abandon the goal
#      with a reason, write self-failure-* lesson learned.
#
# 3. GOAL PROPOSALS — After evaluating health, ask reasoning model to propose
#    new goals based on patterns (self-failure remediation, curiosity threads).
#    Proposals are gated by the autonomy drive:
#    - autonomy >= threshold: auto-create with source='assistant', status='active'
#    - autonomy < threshold: save as status='proposed' for user review via !cogn goals
#
# 4. ABANDON GUARD — Proposals are checked against abandoned goals via semantic
#    similarity. If a proposal resembles an abandoned goal, it's blocked.
#    (Implemented in _set_goal_exec as a code-level check.)

_GOAL_HEALTH_SYSTEM = (
    "You evaluate the health of active goals and propose actions.\n\n"
    "For each goal, you receive its title, description, failure_count, and "
    "any recent self-failure rows mentioning this goal.\n\n"
    "Return a JSON object with two keys:\n"
    '  "actions": array of action objects\n'
    '  "proposals": array of new goal proposals\n\n'
    "Action types:\n"
    '  {"goal_id": N, "action": "replan", "reason": "why current approach fails", '
    '"new_steps": ["step1", "step2"]}\n'
    '  {"goal_id": N, "action": "abandon", "reason": "why this is unachievable"}\n'
    '  {"goal_id": N, "action": "ok"} — goal is healthy, no action needed\n\n'
    "Proposal format:\n"
    '  {"title": "short title", "description": "what and why", '
    '"trigger": "self-failure|curiosity|pattern", "importance": 5-8}\n\n'
    "Rules:\n"
    "- Only propose 'abandon' when failure is structural (missing capability, "
    "external dependency, repeated identical failure pattern)\n"
    "- Only propose 'replan' when the approach is clearly wrong but the goal is achievable\n"
    "- Proposals should address gaps you see — remediate failures, explore curiosity threads\n"
    "- Max 2 proposals per cycle. Quality over quantity.\n"
    "- Do NOT propose goals that duplicate active or recently abandoned goals.\n"
    '- Return {"actions": [], "proposals": []} if everything is healthy.'
)

_GOAL_HEALTH_USER = (
    "ACTIVE GOALS:\n{goals_block}\n\n"
    "RECENT SELF-FAILURE PATTERNS:\n{failures_block}\n\n"
    "ABANDONED GOALS (do NOT re-propose):\n{abandoned_block}\n\n"
    "CURRENT DRIVES:\n{drives_block}\n\n"
    "Evaluate each goal's health and propose up to 2 new goals if warranted."
)


async def _run_goal_health(
    active_goals: list[dict],
    turns_text: str,
) -> dict:
    """
    Goal health pass. Returns summary dict with actions taken and proposals.
    """
    cfg = _rcogn_cfg()
    if not cfg.get("goal_health_enabled", True):
        return {"skipped": "goal_health_enabled=false"}
    if not active_goals:
        return {"skipped": "no active goals"}

    replan_thresh = cfg["goal_health_failure_replan"]
    abandon_thresh = cfg["goal_health_failure_abandon"]
    autonomy_thresh = cfg["goal_health_autonomy_threshold"]
    model_key = cfg["goal_health_model"]
    if not model_key:
        from config import get_model_role
        try:
            model_key = get_model_role("goal_health")
        except KeyError:
            model_key = "samaritan-reasoning"

    from database import fetch_dicts, execute_sql, execute_insert
    from memory import _GOALS, _ST, _typed_metric_write, _COGNITION, save_cognition
    from memory import load_drives

    summary = {
        "replanned": [], "abandoned": [], "proposed": [],
        "auto_created": [], "pending_review": [],
    }

    # ── 1. Failure counting ─────────────────────────────────────────────
    # Count failure rows from cognition table
    try:
        failure_rows = await fetch_dicts(
            f"SELECT id, topic, content FROM {_COGNITION()} "
            f"WHERE origin IN ('tool_failure', 'goal_health') "
            f"ORDER BY created_at DESC LIMIT 100"
        ) or []
    except Exception as e:
        log.warning(f"goal_health: failure rows fetch failed: {e}")
        failure_rows = []

    # Match failures to goals by keyword overlap
    goal_failures: dict[int, list[dict]] = {g["id"]: [] for g in active_goals}
    for g in active_goals:
        keywords = set(
            w.lower() for w in
            (g.get("title", "") + " " + g.get("description", "")).split()
            if len(w) > 3  # skip short words
        )
        for fr in failure_rows:
            content_lower = (fr.get("content") or "").lower()
            # Match if ≥2 keywords appear in the failure content
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches >= 2:
                goal_failures[g["id"]].append(fr)

    # Update failure_count in DB
    for g in active_goals:
        new_fc = len(goal_failures.get(g["id"], []))
        old_fc = int(g.get("failure_count", 0))
        if new_fc != old_fc:
            try:
                await execute_sql(
                    f"UPDATE {_GOALS()} SET failure_count={new_fc} "
                    f"WHERE id={g['id']}"
                )
            except Exception as e:
                log.debug(f"goal_health: failure_count update failed id={g['id']}: {e}")
        g["failure_count"] = new_fc  # update in-memory for LLM prompt

    # ── 2. Fetch context for LLM ────────────────────────────────────────
    # Abandoned goals (for dedup guard)
    try:
        abandoned = await fetch_dicts(
            f"SELECT id, title, description, abandon_reason FROM {_GOALS()} "
            f"WHERE status='abandoned' ORDER BY updated_at DESC LIMIT 20"
        ) or []
    except Exception:
        abandoned = []

    # Current drives — convert list[dict] to name→value dict
    try:
        _drive_rows = await load_drives()
        drives = {d["name"]: float(d.get("value", 0.5)) for d in _drive_rows}
    except Exception:
        drives = {}

    # ── 3. Build prompt and call LLM ────────────────────────────────────
    goals_lines = []
    for g in active_goals:
        fc = g.get("failure_count", 0)
        ac = g.get("attempt_count", 0)
        goals_lines.append(
            f"  [id={g['id']} failures={fc} attempts={ac}] "
            f"{g.get('title', '')} — {g.get('description', '')}"
        )
    goals_block = "\n".join(goals_lines) if goals_lines else "(none)"

    fail_lines = []
    for gid, frs in goal_failures.items():
        if frs:
            gtitle = next((g["title"] for g in active_goals if g["id"] == gid), "?")
            for fr in frs[:3]:  # max 3 per goal in prompt
                fail_lines.append(f"  [{gtitle}] {fr.get('content', '')[:200]}")
    failures_block = "\n".join(fail_lines) if fail_lines else "(none)"

    abandoned_lines = [
        f"  [id={a['id']}] {a.get('title') or ''} — reason: {a.get('abandon_reason') or 'n/a'}"
        for a in abandoned
    ]
    abandoned_block = "\n".join(abandoned_lines) if abandoned_lines else "(none)"

    drives_block = "\n".join(
        f"  {name}: {val:.2f}" for name, val in drives.items()
    ) if drives else "(no drives loaded)"

    prompt = _GOAL_HEALTH_USER.format(
        goals_block=goals_block,
        failures_block=failures_block,
        abandoned_block=abandoned_block,
        drives_block=drives_block,
    )

    # Call reasoning model
    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage

    if model_key not in LLM_REGISTRY:
        log.warning(f"goal_health: model {model_key!r} not in registry")
        return summary

    try:
        llm = _build_lc_llm(model_key)
        timeout = LLM_REGISTRY[model_key].get("llm_call_timeout", 90)
        msgs = [SystemMessage(content=_GOAL_HEALTH_SYSTEM), HumanMessage(content=prompt)]
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)
    except Exception as e:
        log.warning(f"goal_health: LLM call failed: {e}")
        return summary

    if not raw:
        return summary

    # Parse response
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"goal_health: JSON parse failed: {e}. raw={raw[:300]}")
        return summary

    if not isinstance(parsed, dict):
        return summary

    # ── 4. Process actions ──────────────────────────────────────────────
    valid_ids = {g["id"] for g in active_goals}
    actions = parsed.get("actions", [])
    for act in actions:
        if not isinstance(act, dict):
            continue
        gid = act.get("goal_id")
        action = act.get("action", "")
        reason = str(act.get("reason", ""))[:500]

        if gid not in valid_ids:
            continue

        goal = next((g for g in active_goals if g["id"] == gid), None)
        if not goal:
            continue
        fc = goal.get("failure_count", 0)

        if action == "abandon" and fc >= abandon_thresh:
            # Auto-abandon
            try:
                await execute_sql(
                    f"UPDATE {_GOALS()} SET status='abandoned', "
                    f"abandon_reason='{reason.replace(chr(39), chr(39)*2)}' "
                    f"WHERE id={gid} AND status='active'"
                )
                _typed_metric_write(_GOALS())
                summary["abandoned"].append(gid)
                log.info(
                    f"goal_health: auto-abandoned goal id={gid} "
                    f"(failures={fc} >= {abandon_thresh}): {reason}"
                )
                # Write lesson learned
                await save_cognition(
                    origin="goal_health",
                    topic=goal.get("title", f"goal-{gid}"),
                    content=f"Goal '{goal.get('title', '')}' abandoned after {fc} failures. "
                            f"Reason: {reason}",
                    importance=8,
                )
            except Exception as e:
                log.warning(f"goal_health: abandon failed id={gid}: {e}")

        elif action == "replan" and fc >= replan_thresh:
            # Log replan suggestion — the reasoning model proposed new steps
            new_steps = act.get("new_steps", [])
            steps_str = "; ".join(str(s) for s in new_steps[:5])
            try:
                await save_cognition(
                    origin="goal_health",
                    topic=goal.get("title", f"goal-{gid}"),
                    content=f"Goal '{goal.get('title', '')}' needs replanning "
                            f"(failures={fc}): {reason}. Suggested steps: {steps_str}",
                    importance=7,
                )
                summary["replanned"].append(gid)
                log.info(
                    f"goal_health: replan suggested for goal id={gid} "
                    f"(failures={fc} >= {replan_thresh})"
                )
            except Exception as e:
                log.warning(f"goal_health: replan save failed id={gid}: {e}")

    # ── 5. Process proposals — drive-gated ──────────────────────────────
    proposals = parsed.get("proposals", [])[:2]  # max 2

    autonomy_val = float(drives.get("autonomy", 0.4))
    auto_create = autonomy_val >= autonomy_thresh

    for prop in proposals:
        if not isinstance(prop, dict):
            continue
        title = str(prop.get("title", ""))[:255].strip()
        desc = str(prop.get("description", ""))[:1000].strip()
        imp = max(5, min(8, int(prop.get("importance", 6))))

        if not title:
            continue

        # Abandon guard: check if this resembles an abandoned goal
        if _proposal_resembles_abandoned(title, desc, abandoned):
            log.info(f"goal_health: proposal blocked (resembles abandoned): {title}")
            continue

        # Duplicate guard: check active goals
        if _proposal_resembles_active(title, active_goals):
            log.info(f"goal_health: proposal blocked (resembles active): {title}")
            continue

        if auto_create:
            # Autonomy high enough — create directly
            t = title.replace("'", "''")
            d = desc.replace("'", "''")
            try:
                row_id = await execute_insert(
                    f"INSERT INTO {_GOALS()} "
                    f"(title, description, status, importance, source, session_id) "
                    f"VALUES ('{t}', '{d}', 'active', {imp}, 'assistant', 'reflection')"
                )
                _typed_metric_write(_GOALS())
                summary["auto_created"].append({"id": row_id, "title": title})
                log.info(
                    f"goal_health: auto-created goal id={row_id} "
                    f"(autonomy={autonomy_val:.2f} >= {autonomy_thresh}): {title}"
                )
            except Exception as e:
                log.warning(f"goal_health: auto-create failed: {e}")
        else:
            # Autonomy too low — save as proposed for user review
            t = title.replace("'", "''")
            d = desc.replace("'", "''")
            try:
                row_id = await execute_insert(
                    f"INSERT INTO {_GOALS()} "
                    f"(title, description, status, importance, source, session_id) "
                    f"VALUES ('{t}', '{d}', 'active', {imp}, 'assistant', 'reflection-proposed')"
                )
                _typed_metric_write(_GOALS())
                # Mark as proposed by setting a low importance so it doesn't dominate
                # and session_id='reflection-proposed' for easy querying
                summary["pending_review"].append({"id": row_id, "title": title})
                log.info(
                    f"goal_health: proposed goal id={row_id} "
                    f"(autonomy={autonomy_val:.2f} < {autonomy_thresh}, needs review): {title}"
                )
            except Exception as e:
                log.warning(f"goal_health: proposal save failed: {e}")

    return summary


def _proposal_resembles_abandoned(
    title: str, desc: str, abandoned: list[dict], threshold: float = 0.5
) -> bool:
    """
    Keyword overlap check: does the proposal share enough words with an
    abandoned goal to be considered a retry?
    """
    prop_words = set(
        w.lower() for w in (title + " " + desc).split() if len(w) > 3
    )
    if not prop_words:
        return False
    for ab in abandoned:
        ab_words = set(
            w.lower() for w in
            (ab.get("title", "") + " " + ab.get("description", "")).split()
            if len(w) > 3
        )
        if not ab_words:
            continue
        overlap = len(prop_words & ab_words)
        ratio = overlap / min(len(prop_words), len(ab_words))
        if ratio >= threshold:
            return True
    return False


def _proposal_resembles_active(title: str, active_goals: list[dict]) -> bool:
    """Check if a proposal duplicates an active goal by title similarity."""
    title_lower = title.lower()
    for g in active_goals:
        gt = g.get("title", "").lower()
        # Exact substring match or very high word overlap
        if title_lower in gt or gt in title_lower:
            return True
        gt_words = set(w for w in gt.split() if len(w) > 3)
        title_words = set(w for w in title_lower.split() if len(w) > 3)
        if gt_words and title_words:
            overlap = len(gt_words & title_words)
            if overlap / min(len(gt_words), len(title_words)) >= 0.7:
                return True
    return False


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

async def run_reflection() -> dict:
    """
    Run one reflection pass. Safe to call manually (e.g. from !cogn reflection run).
    Returns summary dict.
    """
    cfg = _rcogn_cfg()
    model_key    = cfg["reflection_model"]
    if not model_key:
        from config import get_model_role
        try:
            model_key = get_model_role("reflection")
        except KeyError:
            model_key = "summarizer-gemini"
    turn_limit   = cfg["reflection_turn_limit"]
    min_turns    = cfg["reflection_min_turns"]
    max_memories = cfg["reflection_max_memories"]

    from database import set_db_override, list_managed_databases, fetch_dicts as _fd_refl

    t_start = time.monotonic()
    summary = {"turns": 0, "saved": 0, "skipped": 0, "error": None}
    all_stale = True  # track if every DB was stale (skip entire cycle)

    for db_name in list_managed_databases():
        set_db_override(db_name)

        # Staleness gate: skip LLM call if no new ST entries since last interval
        try:
            from memory import _ST
            staleness_minutes = cfg["reflection_interval_m"] * 1.3
            latest = await _fd_refl(
                f"SELECT MAX(created_at) AS latest FROM {_ST()} "
                f"WHERE source IN ('user', 'assistant')"
            )
            if latest and latest[0].get("latest"):
                from datetime import timedelta
                age = datetime.now() - latest[0]["latest"]
                if age > timedelta(minutes=staleness_minutes):
                    log.debug(
                        f"reflection[{db_name}]: newest ST entry is {age.total_seconds()/60:.0f}m old "
                        f"(threshold {staleness_minutes:.0f}m) — skipping"
                    )
                    continue
        except Exception as e:
            log.debug(f"reflection[{db_name}]: staleness check failed ({e}), proceeding anyway")

        all_stale = False

        try:
            rows = await _fetch_recent_turns(turn_limit)
            summary["turns"] += len(rows)
            _stats["turns_processed"] += len(rows)

            if len(rows) < min_turns:
                log.debug(f"reflection[{db_name}]: only {len(rows)} turns < min={min_turns}, skipping")
                continue

            turns_text = _format_turns(rows)

            # Fetch active goals to pass for completion detection
            active_goals: list[dict] = []
            try:
                from memory import _GOALS
                from database import fetch_dicts as _fd
                active_goals = await _fd(
                    f"SELECT id, title, description, attempt_count, failure_count "
                    f"FROM {_GOALS()} WHERE status = 'active'"
                ) or []
            except Exception as _ge:
                log.debug(f"reflection[{db_name}]: goals fetch failed: {_ge}")

            items, goals_done = await _call_llm(model_key, turns_text, max_memories, active_goals)

            # Second-pass: dedicated goal completion scan using reasoning model
            if active_goals:
                try:
                    goals_done_2 = await _scan_goal_completions(turns_text, active_goals)
                    goals_done = list(set(goals_done) | set(goals_done_2))
                    if goals_done_2:
                        log.info(f"reflection[{db_name}]: goal-scan (reasoning) detected completions: {goals_done_2}")
                except Exception as e:
                    log.warning(f"reflection[{db_name}]: goal-scan second pass failed: {e}")

            from memory import save_cognition as _save_cogn
            for item in items:
                if not isinstance(item, dict):
                    continue
                topic   = str(item.get("topic", "reflection"))[:255]
                content = str(item.get("content", ""))[:2000]
                imp     = max(1, min(10, int(item.get("importance", 6))))

                if not topic or not content:
                    continue

                new_id = await _save_cogn(
                    origin="reflection",
                    topic=topic,
                    content=content,
                    importance=imp,
                )
                if new_id:
                    summary["saved"] += 1
                    _stats["memories_saved"] += 1
                else:
                    summary["skipped"] += 1
                    _stats["memories_skipped"] += 1

            # Process goal completions detected by LLM
            if goals_done:
                from memory import _GOALS
                from database import execute_sql as _exec_sql
                valid_ids = {g["id"] for g in active_goals}
                marked = []
                for gid in goals_done:
                    if gid not in valid_ids:
                        log.debug(f"reflection[{db_name}]: goal_done id={gid} not in active goals — skipped")
                        continue
                    try:
                        await _exec_sql(
                            f"UPDATE {_GOALS()} SET status='done' WHERE id={gid} AND status='active'"
                        )
                        marked.append(gid)
                        log.info(f"reflection[{db_name}]: marked goal id={gid} as done (detected in conversation)")
                    except Exception as _ge2:
                        log.warning(f"reflection[{db_name}]: goal mark-done failed id={gid}: {_ge2}")
                if marked:
                    summary.setdefault("goals_marked_done", []).extend(marked)

            # Goal health pass — failure escalation, replanning, proposals
            if active_goals:
                try:
                    gh_summary = await _run_goal_health(active_goals, turns_text)
                    summary["goal_health"] = gh_summary
                    if gh_summary.get("abandoned"):
                        log.info(f"reflection[{db_name}]: goal_health abandoned: {gh_summary['abandoned']}")
                    if gh_summary.get("auto_created"):
                        log.info(f"reflection[{db_name}]: goal_health auto-created: {gh_summary['auto_created']}")
                    if gh_summary.get("pending_review"):
                        log.info(f"reflection[{db_name}]: goal_health pending review: {gh_summary['pending_review']}")
                except Exception as e:
                    log.warning(f"reflection[{db_name}]: goal_health pass failed: {e}")

        except Exception as e:
            log.error(f"reflection[{db_name}]: run error: {e}")
            summary["error"] = str(e)
            _stats["last_error"] = str(e)

    set_db_override("")

    if all_stale:
        summary["skipped_reason"] = "all databases stale"
        return summary

    duration = time.monotonic() - t_start
    _stats["runs"]             += 1
    _stats["last_run_at"]       = datetime.now(timezone.utc).isoformat()
    _stats["last_run_duration_s"] = round(duration, 2)
    _stats["last_run_saved"]    = summary["saved"]

    log.info(
        f"reflection: run done — turns={summary['turns']} saved={summary['saved']} "
        f"skipped={summary['skipped']} dur={duration:.1f}s"
    )

    # Drive decay and goal-based nudge (runs across all managed DBs)
    for db_name in list_managed_databases():
        set_db_override(db_name)
        try:
            from memory import update_drives_from_goals
            drive_summary = await update_drives_from_goals()
            summary["drives_updated"] = summary.get("drives_updated", 0) + drive_summary.get("drives_updated", 0)
            log.info(
                f"reflection[{db_name}]: drives — updated={drive_summary.get('drives_updated', 0)} "
                f"goals_done={drive_summary.get('goals_done', 0)} "
                f"goals_blocked={drive_summary.get('goals_blocked', 0)}"
            )
        except Exception as e:
            log.warning(f"reflection[{db_name}]: drive update failed: {e}")
    set_db_override("")

    # Feedback evaluation — update watermark first so evaluator sees rows written this cycle
    try:
        from cogn_feedback import evaluate, LOOP_REFLECTION
        fb = await evaluate(LOOP_REFLECTION, summary)
        _stats["last_feedback"] = fb
        if fb.get("verdict") not in (None, "insufficient_data", "neutral", "useful"):
            log.info(f"reflection: feedback verdict={fb.get('verdict')} strength={fb.get('strength')} streak={fb.get('streak')}")
    except Exception as e:
        log.warning(f"reflection: feedback evaluation failed: {e}")

    # Refresh self-summary every N cycles
    if _stats["runs"] % _SELF_SUMMARY_EVERY_N == 0:
        try:
            from memory import refresh_self_summary
            from agents import _call_llm_text
            from config import LLM_REGISTRY
            _summarizer_key = cfg.get("reflection_model") or ""
            if not _summarizer_key:
                from config import get_model_role
                try:
                    _summarizer_key = get_model_role("reflection")
                except KeyError:
                    _summarizer_key = "summarizer-gemini"
            # Fall back to any available summarizer
            if _summarizer_key not in LLM_REGISTRY:
                _summarizer_key = next(
                    (k for k in LLM_REGISTRY if "summarizer" in k), _summarizer_key
                )

            async def _self_llm_fn(prompt: str) -> str:
                return await _call_llm_text(_summarizer_key, prompt)

            _self_summary = await refresh_self_summary(llm_call_fn=_self_llm_fn)
            if _self_summary:
                log.info(f"reflection: self-summary refreshed ({len(_self_summary)} chars)")
                summary["self_summary_refreshed"] = True
        except Exception as e:
            log.warning(f"reflection: self-summary refresh failed: {e}")

    return summary


# ---------------------------------------------------------------------------
# Background task entry point
# ---------------------------------------------------------------------------

async def reflection_task() -> None:
    """
    Long-running asyncio task. Loops every reflection_interval_m minutes.
    Wakes early if trigger_now() is called.
    """
    from timer_registry import register_timer, timer_sleep
    register_timer("reflection", "cogn")

    global _wake_event
    _wake_event = asyncio.Event()

    while True:
        cfg = _rcogn_cfg()

        if not cfg["enabled"] or not cfg["reflection_enabled"]:
            _wake_event.clear()
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=300)
                _wake_event.clear()
            except asyncio.TimeoutError:
                pass
            continue

        interval_m = cfg["reflection_interval_m"]
        if interval_m <= 0:
            await asyncio.sleep(3600)
            continue

        try:
            await run_reflection()
        except Exception as e:
            log.warning(f"reflection_task: unhandled error: {e}")
            _stats["last_error"] = str(e)

        # Backoff: jump to 24h when both contradiction (cap 60m) and
        # prospective (cap 120m) have hit their caps
        from state import backoff_interval, idle_seconds, fmt_interval
        try:
            with open(_PLUGINS_PATH) as _f:
                _pcog = json.load(_f).get("plugin_config", {}).get("proactive_cognition", {})
        except Exception:
            _pcog = {}
        contra_base = int(_pcog.get("contradiction_interval_m", 2))
        prosp_base = int(_pcog.get("prospective_interval_m", 1))
        contra_eff = backoff_interval(contra_base, 60)
        prosp_eff = backoff_interval(prosp_base, 120)
        if contra_eff >= 60 and prosp_eff >= 120:
            effective_m = 1440  # 24 hours
            log.info(f"reflection: backoff {interval_m}m → 1440m (siblings at cap, idle {idle_seconds()/60:.0f}m)")
        else:
            effective_m = interval_m
        sleep_sec = effective_m * 60
        timer_sleep("reflection", sleep_sec, interval_desc=fmt_interval(effective_m))
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            log.info("reflection_task: woken early by trigger")
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass
