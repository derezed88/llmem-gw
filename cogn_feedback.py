"""
cogn_feedback.py — Shared feedback evaluator for proactive cognition loops.

Called at the end of each loop cycle (reflection, prospective, contradiction).
Measures whether the loop's output was actually used in conversation, then
reinforces or extinguishes a conditioned behavior row accordingly.

Each loop has one conditioned row, identified by topic slug:
  reflection    → topic='cogn-feedback-reflection'
  prospective   → topic='cogn-feedback-prospective'
  contradiction → topic='cogn-feedback-contradiction'

Conditioning model (operant, not classical):
  - Output is "used" if rows produced by this loop have last_accessed > created_at
    (i.e. they were retrieved into at least one prompt since being written).
  - access_ratio = accessed_rows / total_rows_from_loop_in_ST
  - Consecutive low-ratio runs increment strength toward extinction.
  - A high-ratio run decrements strength (loop recovering its usefulness).
  - At strength >= STRENGTH_THROTTLE  → double the loop's interval (runtime override).
  - At strength >= STRENGTH_EXTINGUISH → disable the loop (runtime override).
  - Recovery: when strength drops back below STRENGTH_THROTTLE, restore interval.

Strength thresholds (tunable via plugins-enabled.json → proactive_cognition):
  feedback_strength_throttle:   int  default 7  — start doubling interval
  feedback_strength_extinguish: int  default 10 — disable loop
  feedback_low_ratio:          float default 0.2 — below this = "not useful"
  feedback_high_ratio:         float default 0.5 — above this = "useful"
  feedback_min_rows:           int  default 3   — skip eval if fewer rows to check

Per-loop topic prefix used to find output rows in ST:
  reflection    → source='assistant' AND topic NOT LIKE 'prospective-%'
                  AND topic NOT LIKE 'contradiction-%' AND topic NOT LIKE 'cogn-%'
                  (reflection writes to arbitrary topic slugs)
                  → identified via session watermark: rows with id > last_watermark
  prospective   → topic='prospective-reminder'
  contradiction → topic='contradiction-flag' in beliefs table (not ST)
                  → proxy: flags_written > 0 in a cycle = "did something"

Because reflection writes to arbitrary topics, it uses an id watermark stored
in _feedback_state to track which ST rows were written since the last run.
Prospective and contradiction use their own output tables directly.

Public API:
    evaluate(loop_name, cycle_outcome)  → dict (feedback summary)
    get_feedback_state()                → dict (all loop states)
    reset_feedback_state(loop_name)     → clear loop's streak and conditioned row
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timezone

log = logging.getLogger("cogn_feedback")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOP_REFLECTION    = "reflection"
LOOP_PROSPECTIVE   = "prospective"
LOOP_CONTRADICTION = "contradiction"

_TOPIC = {
    LOOP_REFLECTION:    "cogn-feedback-reflection",
    LOOP_PROSPECTIVE:   "cogn-feedback-prospective",
    LOOP_CONTRADICTION: "cogn-feedback-contradiction",
}

# Trigger / reaction text stored in the conditioned row — model reads these
_TRIGGER = {
    LOOP_REFLECTION:    "reflection loop produces memories that are not accessed in conversation",
    LOOP_PROSPECTIVE:   "prospective reminders fire but are not acted on in conversation",
    LOOP_CONTRADICTION: "contradiction scanner runs but flags are ignored or never resolved",
}
_REACTION = {
    LOOP_REFLECTION:    "reduce reflection frequency; fewer insights are better than noise",
    LOOP_PROSPECTIVE:   "reduce prospective check frequency; reminders are not being used",
    LOOP_CONTRADICTION: "reduce contradiction scan frequency; flags are not being resolved",
}

# ---------------------------------------------------------------------------
# Per-loop in-memory state (survives within one server run)
# ---------------------------------------------------------------------------

_feedback_state: dict = {
    LOOP_REFLECTION:    {
        "consecutive_low": 0,
        "watermarks":      {},  # db_name → max ST row id seen at end of last run
        "conditioned_ids": {},  # db_name → row id in <db>_conditioned (0 = not yet created)
        "current_strength": 0,
        "last_eval_at":    None,
        "last_ratio":      None,
    },
    LOOP_PROSPECTIVE:   {
        "consecutive_low": 0,
        "conditioned_ids": {},
        "current_strength": 0,
        "last_eval_at":    None,
        "last_ratio":      None,
    },
    LOOP_CONTRADICTION: {
        "consecutive_low": 0,
        "conditioned_ids": {},
        "current_strength": 0,
        "last_eval_at":    None,
        "last_ratio":      None,
    },
}


def get_feedback_state() -> dict:
    import copy
    return copy.deepcopy(_feedback_state)


def reset_feedback_state(loop_name: str) -> None:
    if loop_name in _feedback_state:
        s = _feedback_state[loop_name]
        s["consecutive_low"] = 0
        s["current_strength"] = 0
        s["last_ratio"] = None
        s["conditioned_ids"] = {}
        if "watermarks" in s:
            s["watermarks"] = {}
        log.info(f"cogn_feedback: reset state for {loop_name}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PLUGINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")


def _fb_cfg() -> dict:
    try:
        with open(_PLUGINS_PATH) as f:
            raw = json.load(f).get("plugin_config", {}).get("proactive_cognition", {})
    except Exception:
        raw = {}
    return {
        "feedback_strength_throttle":   int(raw.get("feedback_strength_throttle",   7)),
        "feedback_strength_extinguish": int(raw.get("feedback_strength_extinguish", 10)),
        "feedback_low_ratio":           float(raw.get("feedback_low_ratio",          0.20)),
        "feedback_high_ratio":          float(raw.get("feedback_high_ratio",         0.50)),
        "feedback_min_rows":            int(raw.get("feedback_min_rows",             3)),
    }


# ---------------------------------------------------------------------------
# Conditioned row upsert
# ---------------------------------------------------------------------------

async def _upsert_conditioned(loop_name: str, db_name: str, strength: int, status: str = "active") -> int:
    """
    Create or update the conditioned row for this loop in the currently-active DB.
    Returns the row id.
    """
    from memory import _CONDITIONED, _typed_metric_write
    from database import execute_sql, execute_insert

    state = _feedback_state[loop_name]
    topic   = _TOPIC[loop_name].replace("'", "''")
    trigger = _TRIGGER[loop_name].replace("'", "''")
    reaction = _REACTION[loop_name].replace("'", "''")
    strength = max(1, min(10, strength))
    status = status if status in ("active", "extinguished") else "active"
    tbl = _CONDITIONED()

    existing_id = state["conditioned_ids"].get(db_name, 0)

    # If we don't have a cached id, check DB by topic
    if not existing_id:
        try:
            from database import fetch_dicts
            rows = await fetch_dicts(
                f"SELECT id, strength FROM {tbl} WHERE topic='{topic}' LIMIT 1"
            )
            if rows:
                existing_id = rows[0]["id"]
                state["conditioned_ids"][db_name] = existing_id
                state["current_strength"] = rows[0].get("strength", 0)
        except Exception as e:
            log.warning(f"cogn_feedback: conditioned lookup failed for {loop_name}[{db_name}]: {e}")

    if existing_id:
        try:
            await execute_sql(
                f"UPDATE {tbl} SET strength={strength}, status='{status}' WHERE id={existing_id}"
            )
            _typed_metric_write(tbl)
            state["current_strength"] = strength
            log.info(f"cogn_feedback: updated conditioned id={existing_id} [{loop_name}][{db_name}] strength={strength} status={status}")
            return existing_id
        except Exception as e:
            log.warning(f"cogn_feedback: conditioned update failed: {e}")
            return existing_id
    else:
        try:
            row_id = await execute_insert(
                f"INSERT INTO {tbl} "
                f"(topic, `trigger`, `reaction`, strength, status, source) "
                f"VALUES ('{topic}', '{trigger}', '{reaction}', {strength}, '{status}', 'assistant')"
            )
            _typed_metric_write(tbl)
            state["conditioned_ids"][db_name] = row_id
            state["current_strength"] = strength
            log.info(f"cogn_feedback: created conditioned id={row_id} [{loop_name}][{db_name}] strength={strength}")
            return row_id
        except Exception as e:
            log.warning(f"cogn_feedback: conditioned insert failed: {e}")
            return 0


# ---------------------------------------------------------------------------
# Access ratio computation — per loop
# ---------------------------------------------------------------------------

async def _reflection_ratio(watermark: int, min_rows: int) -> tuple[float | None, int]:
    """
    Check cognition rows written since watermark (id > watermark, origin='reflection').
    Returns (ratio, new_watermark). ratio=None means too few rows to judge.
    """
    from memory import _COGNITION
    from database import fetch_dicts, execute_sql
    try:
        rows = await fetch_dicts(
            f"SELECT id, last_accessed, created_at FROM {_COGNITION()} "
            f"WHERE id > {watermark} AND origin = 'reflection' "
            f"ORDER BY id ASC"
        )
    except Exception as e:
        log.warning(f"cogn_feedback: reflection_ratio query failed: {e}")
        return None, watermark

    if not rows:
        return None, watermark

    new_watermark = max(r["id"] for r in rows)

    if len(rows) < min_rows:
        return None, new_watermark

    accessed = sum(
        1 for r in rows
        if r.get("last_accessed") and r.get("created_at")
        and r["last_accessed"] > r["created_at"]
    )
    ratio = accessed / len(rows)
    log.debug(f"cogn_feedback: reflection ratio={ratio:.2f} ({accessed}/{len(rows)}) watermark={new_watermark}")
    return ratio, new_watermark


async def _prospective_ratio(min_rows: int) -> float | None:
    """
    Check prospective rows in cognition table.
    A reminder is "used" if last_accessed > created_at (injected into a prompt after firing).
    Returns ratio or None if too few rows.
    """
    from memory import _COGNITION
    from database import fetch_dicts
    try:
        rows = await fetch_dicts(
            f"SELECT last_accessed, created_at FROM {_COGNITION()} "
            f"WHERE origin = 'prospective' "
            f"ORDER BY created_at DESC LIMIT 50"
        )
    except Exception as e:
        log.warning(f"cogn_feedback: prospective_ratio query failed: {e}")
        return None

    if len(rows) < min_rows:
        return None

    accessed = sum(
        1 for r in rows
        if r.get("last_accessed") and r.get("created_at")
        and r["last_accessed"] > r["created_at"]
    )
    ratio = accessed / len(rows)
    log.debug(f"cogn_feedback: prospective ratio={ratio:.2f} ({accessed}/{len(rows)})")
    return ratio


async def _contradiction_ratio(cycle_flags_written: int, cycle_pairs: int) -> float | None:
    """
    Contradiction proxy: flags_written / max(pairs_evaluated, 1) is output volume,
    but "usefulness" is whether existing contradiction-flag beliefs have been retracted
    (resolved by model) vs. piling up ignored.

    ratio = retracted_flags / (retracted_flags + active_flags)
    A high active pile with no retractions → low ratio → not useful.
    Returns None if total flags < min_rows.
    """
    from memory import _BELIEFS
    from database import fetch_dicts
    try:
        rows = await fetch_dicts(
            f"SELECT status FROM {_BELIEFS()} WHERE topic='contradiction-flag'"
        )
    except Exception as e:
        log.warning(f"cogn_feedback: contradiction_ratio query failed: {e}")
        return None

    if len(rows) < 3:
        return None

    retracted = sum(1 for r in rows if r.get("status") == "retracted")
    active    = sum(1 for r in rows if r.get("status") == "active")
    total = retracted + active
    if total == 0:
        return None

    ratio = retracted / total
    log.debug(f"cogn_feedback: contradiction ratio={ratio:.2f} ({retracted}/{total})")
    return ratio


# ---------------------------------------------------------------------------
# Apply outcome — adjust strength and runtime overrides
# ---------------------------------------------------------------------------

def _apply_outcome(loop_name: str, ratio: float, cfg: dict) -> str:
    """
    Given a usage ratio, update consecutive_low streak and return verdict:
    'useful' | 'neutral' | 'low' | 'throttle' | 'extinguish' | 'recover'
    """
    state = _feedback_state[loop_name]
    low_thresh  = cfg["feedback_low_ratio"]
    high_thresh = cfg["feedback_high_ratio"]
    thr_strength = cfg["feedback_strength_throttle"]
    ext_strength = cfg["feedback_strength_extinguish"]

    state["last_ratio"] = round(ratio, 3)

    if ratio >= high_thresh:
        # Useful — decay streak
        if state["consecutive_low"] > 0:
            state["consecutive_low"] = max(0, state["consecutive_low"] - 1)
        return "useful"

    if ratio >= low_thresh:
        # Neutral — no change
        return "neutral"

    # Low — increment streak
    state["consecutive_low"] += 1
    streak = state["consecutive_low"]

    if streak * 1 >= ext_strength:  # strength = streak capped at 10
        return "extinguish"
    if streak * 1 >= thr_strength:
        return "throttle"
    return "low"


def _apply_interval_override(loop_name: str, verdict: str, cfg: dict) -> None:
    """
    Write runtime overrides to contradiction module's shared _overrides dict
    based on verdict. Recovers overrides when verdict improves.
    """
    try:
        from contradiction import set_runtime_override, get_runtime_overrides, _cogn_cfg
    except ImportError:
        return

    import json as _json
    try:
        with open(_PLUGINS_PATH) as f:
            raw = _json.load(f).get("plugin_config", {}).get("proactive_cognition", {})
    except Exception:
        raw = {}

    _interval_key = {
        LOOP_REFLECTION:    "reflection_interval_m",
        LOOP_PROSPECTIVE:   "prospective_interval_m",
        LOOP_CONTRADICTION: "contradiction_interval_m",
    }
    _enabled_key = {
        LOOP_REFLECTION:    "reflection_enabled",
        LOOP_PROSPECTIVE:   "prospective_enabled",
        LOOP_CONTRADICTION: "contradiction_enabled",
    }
    _base_interval = {
        LOOP_REFLECTION:    int(raw.get("reflection_interval_m", 60)),
        LOOP_PROSPECTIVE:   int(raw.get("prospective_interval_m", 5)),
        LOOP_CONTRADICTION: int(raw.get("contradiction_interval_m", 60)),
    }

    ikey = _interval_key[loop_name]
    ekey = _enabled_key[loop_name]
    base = _base_interval[loop_name]
    ovr  = get_runtime_overrides()

    if verdict == "extinguish":
        set_runtime_override(ekey, False)
        log.warning(
            f"cogn_feedback: {loop_name} EXTINGUISHED — too many low-ratio cycles. "
            f"Use !cogn reset or !cogn {loop_name} on to restore."
        )
    elif verdict == "throttle":
        doubled = base * 2
        current_ovr = ovr.get(ikey)
        if current_ovr != doubled:
            set_runtime_override(ikey, doubled)
            log.info(f"cogn_feedback: {loop_name} throttled — interval doubled to {doubled}")
    elif verdict == "useful":
        # Restore interval override if it was throttled (but not extinguished)
        if not ovr.get(ekey) is False:  # not extinguished
            if ikey in ovr:
                from contradiction import clear_runtime_overrides, get_runtime_overrides
                # Only remove the interval key, not all overrides
                current = dict(get_runtime_overrides())
                current.pop(ikey, None)
                clear_runtime_overrides()
                for k, v in current.items():
                    set_runtime_override(k, v)
                log.info(f"cogn_feedback: {loop_name} recovering — interval override cleared")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def evaluate(loop_name: str, cycle_outcome: dict) -> dict:
    """
    Evaluate loop effectiveness after one cycle and update conditioned behavior
    across all managed databases.

    cycle_outcome keys (all optional, loop-specific):
        saved          int  — rows saved (reflection)
        skipped        int  — rows deduped (reflection)
        fired          int  — reminders fired (prospective)
        flags          int  — flags written (contradiction)
        pairs          int  — pairs evaluated (contradiction)
        error          str  — if set, skip eval

    Returns summary dict with keys: ratio, verdict, strength, action.
    """
    if loop_name not in _feedback_state:
        return {"error": f"unknown loop: {loop_name}"}

    if cycle_outcome.get("error"):
        return {"skipped": "cycle had error"}

    from database import set_db_override, list_managed_databases

    cfg   = _fb_cfg()
    state = _feedback_state[loop_name]
    min_rows = cfg["feedback_min_rows"]

    result = {"loop": loop_name, "ratio": None, "verdict": None,
              "strength": state["current_strength"], "action": "none"}

    # --- Compute ratio across all managed DBs, weighted by row count ---
    total_accessed = 0
    total_rows     = 0
    any_data       = False

    for db_name in list_managed_databases():
        set_db_override(db_name)
        try:
            if loop_name == LOOP_REFLECTION:
                wm = state["watermarks"].get(db_name, 0)
                from memory import _COGNITION
                from database import fetch_dicts
                try:
                    rows = await fetch_dicts(
                        f"SELECT id, last_accessed, created_at FROM {_COGNITION()} "
                        f"WHERE id > {wm} AND origin = 'reflection' ORDER BY id ASC"
                    )
                    if rows:
                        state["watermarks"][db_name] = max(r["id"] for r in rows)
                    if len(rows) >= min_rows:
                        accessed = sum(
                            1 for r in rows
                            if r.get("last_accessed") and r.get("created_at")
                            and r["last_accessed"] > r["created_at"]
                        )
                        total_accessed += accessed
                        total_rows     += len(rows)
                        any_data = True
                except Exception as e:
                    log.warning(f"cogn_feedback: reflection_ratio query failed [{db_name}]: {e}")

            elif loop_name == LOOP_PROSPECTIVE:
                from memory import _COGNITION
                from database import fetch_dicts
                try:
                    rows = await fetch_dicts(
                        f"SELECT last_accessed, created_at FROM {_COGNITION()} "
                        f"WHERE origin = 'prospective' ORDER BY created_at DESC LIMIT 50"
                    )
                    if len(rows) >= min_rows:
                        accessed = sum(
                            1 for r in rows
                            if r.get("last_accessed") and r.get("created_at")
                            and r["last_accessed"] > r["created_at"]
                        )
                        total_accessed += accessed
                        total_rows     += len(rows)
                        any_data = True
                except Exception as e:
                    log.warning(f"cogn_feedback: prospective_ratio query failed [{db_name}]: {e}")

            elif loop_name == LOOP_CONTRADICTION:
                from memory import _BELIEFS
                from database import fetch_dicts
                try:
                    rows = await fetch_dicts(
                        f"SELECT status FROM {_BELIEFS()} WHERE topic='contradiction-flag'"
                    )
                    if len(rows) >= 3:
                        retracted = sum(1 for r in rows if r.get("status") == "retracted")
                        active    = sum(1 for r in rows if r.get("status") == "active")
                        total_accessed += retracted
                        total_rows     += retracted + active
                        any_data = True
                except Exception as e:
                    log.warning(f"cogn_feedback: contradiction_ratio query failed [{db_name}]: {e}")

        except Exception as e:
            log.warning(f"cogn_feedback: ratio computation failed for {loop_name}[{db_name}]: {e}")
        finally:
            set_db_override("")

    state["last_eval_at"] = datetime.now(timezone.utc).isoformat()

    if not any_data or total_rows < min_rows:
        result["verdict"] = "insufficient_data"
        result["action"] = "none"
        return result

    ratio = total_accessed / total_rows
    result["ratio"] = ratio

    # --- Apply verdict ---
    verdict = _apply_outcome(loop_name, ratio, cfg)
    result["verdict"] = verdict
    streak = state["consecutive_low"]

    # Strength = consecutive low streak, capped at 10
    new_strength = min(10, streak)

    # --- Update conditioned row in each DB if strength changed or verdict is notable ---
    old_strength = state["current_strength"]
    if new_strength != old_strength or verdict in ("throttle", "extinguish", "useful"):
        status = "extinguished" if verdict == "extinguish" else "active"
        for db_name in list_managed_databases():
            set_db_override(db_name)
            try:
                await _upsert_conditioned(loop_name, db_name, new_strength if new_strength > 0 else 1, status)
            except Exception as e:
                log.warning(f"cogn_feedback: conditioned upsert failed [{loop_name}][{db_name}]: {e}")
            finally:
                set_db_override("")

    # --- Apply interval/enable overrides ---
    _apply_interval_override(loop_name, verdict, cfg)

    result["strength"] = state["current_strength"]
    result["streak"]   = streak
    result["action"]   = verdict

    log.info(
        f"cogn_feedback: {loop_name} ratio={ratio:.2f} verdict={verdict} "
        f"streak={streak} strength={new_strength}"
    )
    return result
