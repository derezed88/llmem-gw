"""
contradiction.py — Proactive contradiction scanner for the beliefs table.

Runs as a background asyncio task (see llmem-gw.py). On each cycle it:
  1. Groups active belief rows by topic (same-topic SQL JOIN) to find candidate pairs.
  2. Sends batches of pairs to a cheap LLM for contradiction detection.
  3. Writes flag rows back to samaritan_beliefs (status='active', topic='contradiction-flag')
     so the model sees them in context and can resolve them conversationally.
  4. Never auto-retracts — destruction requires human or model confirmation.

Config (plugins-enabled.json → plugin_config.proactive_cognition):
    enabled:                  bool   — master switch (default false)
    contradiction_enabled:    bool   — this loop specifically (default true when master on)
    contradiction_interval_m: int    — minutes between scans (default 60)
    contradiction_model:      str    — model key to use (default "summarizer-gemini")
    contradiction_min_beliefs:int    — skip if fewer active beliefs than this (default 5)
    contradiction_max_pairs:  int    — max belief pairs per LLM batch (default 10)
    contradiction_auto_retract: bool — auto-retract lower-confidence belief on match (default false)

Runtime control (no restart needed):
    get_contradiction_stats()  → dict of counters + last-run info
    get_runtime_overrides()    → dict of session overrides
    set_runtime_override(k, v) → set a runtime override (survives until restart)
    clear_runtime_overrides()  → reset to config-file values
    trigger_now()              → wake the sleeping task immediately
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

log = logging.getLogger("contradiction")


class _SkipScan(Exception):
    """Raised to skip a scan cycle while still recording stats."""

# ---------------------------------------------------------------------------
# Runtime state — shared between background task and !cogn command
# ---------------------------------------------------------------------------

_stats: dict = {
    "scans_run":          0,
    "pairs_evaluated":    0,
    "contradictions_found": 0,
    "flags_written":      0,
    "auto_retracted":     0,
    "last_scan_at":       None,   # ISO string
    "last_scan_duration_s": None, # float
    "last_scan_pairs":    0,
    "last_scan_flags":    0,
    "last_error":         None,
    "last_feedback":      None,
}

# Runtime overrides — set via !cogn; survive until restart
_overrides: dict = {}

# Event used to wake the sleeping task early (e.g. !cogn run)
_wake_event: asyncio.Event | None = None


def get_contradiction_stats() -> dict:
    return dict(_stats)


def get_runtime_overrides() -> dict:
    return dict(_overrides)


def set_runtime_override(key: str, value) -> None:
    _overrides[key] = value


def clear_runtime_overrides() -> None:
    _overrides.clear()


def trigger_now() -> None:
    """Wake the background task immediately to run a scan."""
    if _wake_event:
        _wake_event.set()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_PLUGINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")


def _cogn_cfg() -> dict:
    """
    Return proactive_cognition config with defaults.
    Runtime overrides shadow file values — checked at call time so live.
    """
    try:
        with open(_PLUGINS_PATH) as f:
            raw = json.load(f).get("plugin_config", {}).get("proactive_cognition", {})
    except Exception:
        raw = {}

    base = {
        "enabled":                    raw.get("enabled",                    False),
        # contradiction loop
        "contradiction_enabled":      raw.get("contradiction_enabled",      True),
        "contradiction_interval_m":   int(raw.get("contradiction_interval_m",     60)),
        "contradiction_model":        raw.get("contradiction_model",        ""),
        "contradiction_min_beliefs":  int(raw.get("contradiction_min_beliefs",  5)),
        "contradiction_max_pairs":    int(raw.get("contradiction_max_pairs",    10)),
        "contradiction_auto_retract": raw.get("contradiction_auto_retract", False),
        # prospective loop
        "prospective_enabled":        raw.get("prospective_enabled",        True),
        "prospective_interval_m":     int(raw.get("prospective_interval_m",     5)),
        "prospective_model":          raw.get("prospective_model",          ""),
        "prospective_reminder_imp":   int(raw.get("prospective_reminder_imp",   9)),
        # reflection loop
        "reflection_enabled":         raw.get("reflection_enabled",         True),
        "reflection_interval_m":      int(raw.get("reflection_interval_m",      60)),
        "reflection_model":           raw.get("reflection_model",           ""),
        "reflection_turn_limit":      int(raw.get("reflection_turn_limit",      40)),
        "reflection_min_turns":       int(raw.get("reflection_min_turns",        5)),
        "reflection_max_memories":    int(raw.get("reflection_max_memories",     6)),
        # feedback evaluator
        "feedback_low_ratio":         float(raw.get("feedback_low_ratio",    0.2)),
        "feedback_high_ratio":        float(raw.get("feedback_high_ratio",   0.5)),
        "feedback_min_rows":          int(raw.get("feedback_min_rows",           3)),
        "feedback_strength_throttle": int(raw.get("feedback_strength_throttle",  7)),
        "feedback_strength_extinguish": int(raw.get("feedback_strength_extinguish", 10)),
    }

    # Apply runtime overrides last
    base.update(_overrides)
    return base


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------

async def _fetch_beliefs() -> list[dict]:
    """Return all active beliefs as list of dicts {id, topic, content, confidence}."""
    from database import fetch_dicts
    from memory import _BELIEFS
    try:
        rows = await fetch_dicts(
            f"SELECT id, topic, content, confidence "
            f"FROM {_BELIEFS()} WHERE status = 'active' ORDER BY confidence DESC"
        )
        return rows or []
    except Exception as e:
        log.warning(f"contradiction: fetch_beliefs failed: {e}")
        return []


def _build_pairs(beliefs: list[dict], max_pairs: int) -> list[tuple[dict, dict]]:
    """
    Group beliefs by topic; emit pairs within each topic group.
    Cross-topic pairs are skipped in phase 1 (Qdrant-assisted cross-topic is phase 2).
    Returns at most max_pairs pairs.
    """
    by_topic: dict[str, list[dict]] = {}
    for b in beliefs:
        t = (b.get("topic") or "general").strip().lower()
        by_topic.setdefault(t, []).append(b)

    pairs: list[tuple[dict, dict]] = []
    for topic, group in by_topic.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append((group[i], group[j]))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _format_batch(pairs: list[tuple[dict, dict]]) -> str:
    lines = []
    for a, b in pairs:
        lines.append(
            f"PAIR [{a['id']} vs {b['id']}]  topic={a.get('topic','?')}\n"
            f"  A [id={a['id']} conf={a.get('confidence',5)}]: {a.get('content','')}\n"
            f"  B [id={b['id']} conf={b.get('confidence',5)}]: {b.get('content','')}"
        )
    return "\n\n".join(lines)


_SYSTEM_PROMPT = (
    "You detect logical contradictions in a belief store. "
    "A contradiction means belief A and belief B assert incompatible facts — "
    "they cannot both be true at the same time. "
    "Differences in nuance, emphasis, or completeness are NOT contradictions. "
    "Be conservative: only flag clear logical negations."
)

_USER_PROMPT_TMPL = (
    "Check each pair below for logical contradiction. "
    "Return a JSON array — one object per contradiction found, empty array [] if none.\n"
    "Each object: {{\"belief_id_a\": N, \"belief_id_b\": N, \"conflict\": \"one sentence\", "
    "\"resolution\": \"retract_a|retract_b|merge|flag\", \"confidence\": 0.0-1.0}}\n\n"
    "Only include pairs where confidence >= 0.7.\n\n"
    "PAIRS:\n{pairs_text}"
)


async def _call_llm(model_key: str, pairs_text: str) -> list[dict]:
    """Call the contradiction-check model; return list of contradiction dicts."""
    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage
    import asyncio as _asyncio

    prompt = _USER_PROMPT_TMPL.format(pairs_text=pairs_text)
    try:
        if model_key not in LLM_REGISTRY:
            log.warning(f"contradiction: unknown model {model_key!r}, falling back to first available")
            model_key = next(iter(LLM_REGISTRY))
        cfg = LLM_REGISTRY[model_key]
        timeout = cfg.get("llm_call_timeout", 60)
        llm = _build_lc_llm(model_key)
        msgs = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        response = await _asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)
    except Exception as e:
        log.warning(f"contradiction: LLM call failed: {e}")
        return []

    if not raw:
        return []

    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        items = json.loads(cleaned)
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"contradiction: JSON parse failed: {e}. raw={raw[:200]}")
    return []


async def _write_flag(item: dict, auto_retract: bool) -> None:
    """
    Write a contradiction-flag belief row and optionally retract one side.
    Flag row: topic='contradiction-flag', confidence=9, source='assistant'.
    """
    from database import execute_sql
    from memory import _BELIEFS

    id_a = int(item.get("belief_id_a", 0))
    id_b = int(item.get("belief_id_b", 0))
    conflict = str(item.get("conflict", ""))[:500].replace("'", "''")
    resolution = str(item.get("resolution", "flag"))
    conf_score = float(item.get("confidence", 0.7))

    flag_content = (
        f"CONTRADICTION DETECTED (confidence={conf_score:.2f}): "
        f"belief {id_a} vs belief {id_b} — {conflict} "
        f"(suggested resolution: {resolution})"
    ).replace("'", "''")

    tbl = _BELIEFS()

    # Avoid duplicate flags for the same pair
    try:
        dup_check = await execute_sql(
            f"SELECT COUNT(*) FROM {tbl} "
            f"WHERE topic='contradiction-flag' AND status='active' "
            f"AND content LIKE '%belief {id_a} vs belief {id_b}%'"
        )
        for line in dup_check.strip().splitlines():
            line = line.strip()
            if line.isdigit() and int(line) > 0:
                log.debug(f"contradiction: flag already exists for {id_a} vs {id_b}, skipping")
                return
            parts = line.split()
            if parts and parts[-1].isdigit() and int(parts[-1]) > 0:
                return
    except Exception:
        pass  # proceed even if dup check fails

    try:
        await execute_sql(
            f"INSERT INTO {tbl} (topic, content, confidence, status, source) "
            f"VALUES ('contradiction-flag', '{flag_content}', 9, 'active', 'assistant')"
        )
        _stats["flags_written"] += 1
        log.info(f"contradiction: flag written for beliefs {id_a} vs {id_b}")
    except Exception as e:
        log.warning(f"contradiction: flag insert failed: {e}")
        return

    if auto_retract and resolution in ("retract_a", "retract_b"):
        retract_id = id_a if resolution == "retract_a" else id_b
        try:
            await execute_sql(
                f"UPDATE {tbl} SET status='retracted' WHERE id={retract_id}"
            )
            _stats["auto_retracted"] += 1
            log.info(f"contradiction: auto-retracted belief {retract_id}")
        except Exception as e:
            log.warning(f"contradiction: auto-retract failed for {retract_id}: {e}")


async def run_scan() -> dict:
    """
    Run one full contradiction scan. Returns summary dict.
    Safe to call manually (e.g. from !cogn run).
    """
    cfg = _cogn_cfg()
    model_key     = cfg["contradiction_model"]
    if not model_key:
        from config import get_model_role
        try:
            model_key = get_model_role("contradiction")
        except KeyError:
            model_key = "summarizer-gemini"
    min_beliefs   = cfg["contradiction_min_beliefs"]
    max_pairs     = cfg["contradiction_max_pairs"]
    auto_retract  = cfg["contradiction_auto_retract"]

    from database import set_db_override, list_managed_databases

    t_start = time.monotonic()
    summary = {"pairs": 0, "contradictions": 0, "flags": 0, "error": None}

    flags_total_before = _stats["flags_written"]
    for db_name in list_managed_databases():
        set_db_override(db_name)
        try:
            beliefs = await _fetch_beliefs()
            if len(beliefs) < min_beliefs:
                log.debug(f"contradiction[{db_name}]: only {len(beliefs)} beliefs < min={min_beliefs}, skipping")
                continue

            pairs = _build_pairs(beliefs, max_pairs)
            summary["pairs"] += len(pairs)
            _stats["pairs_evaluated"] += len(pairs)

            if not pairs:
                continue

            pairs_text = _format_batch(pairs)
            results = await _call_llm(model_key, pairs_text)

            summary["contradictions"] += len(results)
            _stats["contradictions_found"] += len(results)

            for item in results:
                await _write_flag(item, auto_retract)
        except Exception as e:
            log.error(f"contradiction[{db_name}]: scan error: {e}")
            summary["error"] = str(e)
            _stats["last_error"] = str(e)
    set_db_override("")

    summary["flags"] = _stats["flags_written"] - flags_total_before
    if summary["flags"] > 0:
        import asyncio as _asyncio
        try:
            import notifier as _notifier
            _asyncio.ensure_future(_notifier.fire_event(
                "contradiction_detected",
                f"{summary['contradictions']} contradiction(s) detected",
                f"flags written: {summary['flags']}",
            ))
        except Exception:
            pass

    duration = time.monotonic() - t_start
    _stats["scans_run"]           += 1
    _stats["last_scan_at"]         = datetime.now(timezone.utc).isoformat()
    _stats["last_scan_duration_s"] = round(duration, 2)
    _stats["last_scan_pairs"]      = summary["pairs"]
    _stats["last_scan_flags"]      = summary["flags"]

    log.info(
        f"contradiction: scan done — pairs={summary['pairs']} "
        f"found={summary['contradictions']} flags={summary['flags']} "
        f"dur={duration:.1f}s"
    )

    # Feedback evaluation
    try:
        from cogn_feedback import evaluate, LOOP_CONTRADICTION
        fb = await evaluate(LOOP_CONTRADICTION, summary)
        _stats["last_feedback"] = fb
        if fb.get("verdict") not in (None, "insufficient_data", "neutral", "useful"):
            log.info(f"contradiction: feedback verdict={fb.get('verdict')} strength={fb.get('strength')}")
    except Exception as e:
        log.warning(f"contradiction: feedback evaluation failed: {e}")

    return summary


# ---------------------------------------------------------------------------
# Background task entry point — called from llmem-gw.py
# ---------------------------------------------------------------------------

async def contradiction_task() -> None:
    """
    Long-running asyncio task. Loops forever:
      - reads config each iteration (live toggle support)
      - sleeps contradiction_interval_m minutes between scans
      - wakes early if trigger_now() is called
    """
    from timer_registry import register_timer, timer_sleep
    register_timer("contradiction", "cogn")

    global _wake_event
    _wake_event = asyncio.Event()

    while True:
        cfg = _cogn_cfg()

        if not cfg["enabled"] or not cfg["contradiction_enabled"]:
            # Disabled — sleep 5 min and re-check (same pattern as aging tasks)
            _wake_event.clear()
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=300)
                _wake_event.clear()
            except asyncio.TimeoutError:
                pass
            continue

        interval_m = cfg["contradiction_interval_m"]
        if interval_m <= 0:
            await asyncio.sleep(3600)
            continue

        # Run the scan
        try:
            await run_scan()
        except Exception as e:
            log.warning(f"contradiction_task: unhandled error: {e}")
            _stats["last_error"] = str(e)

        # Backoff: double interval for every 10 min of inactivity, cap at 60 min
        from state import backoff_interval, idle_seconds, fmt_interval
        effective_m = backoff_interval(interval_m, 60)
        sleep_sec = effective_m * 60
        if effective_m != interval_m:
            log.info(f"contradiction: backoff {interval_m}m → {effective_m:.0f}m (idle {idle_seconds()/60:.0f}m)")
        timer_sleep("contradiction", sleep_sec, interval_desc=fmt_interval(effective_m))
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            log.info("contradiction_task: woken early by trigger")
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass
