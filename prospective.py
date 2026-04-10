"""
prospective.py — Prospective memory loop.

Runs as a background asyncio task. On each cycle it:
  1. Queries samaritan_prospective for status='pending' rows.
  2. For each row, tries to parse due_at as a datetime or LLM-evaluate it as a
     natural-language trigger (e.g. "next Monday", "when user asks about X").
     Datetime-based: fires if now >= due_at (UTC).
     Natural-language: passes the phrase + current datetime to a cheap LLM for a
       yes/no "is this overdue?" judgment. Defaults to "not yet" on ambiguity.
  3. For every overdue row, injects a high-importance reminder row into
     samaritan_memory_shortterm (source='assistant', type='prospective',
     topic='prospective-reminder') so it appears in the next context injection.
  4. Marks fired rows as status='done' in samaritan_prospective.

Config (plugins-enabled.json → plugin_config.proactive_cognition):
    enabled:                    bool   — master switch
    prospective_enabled:        bool   — this loop (default true when master on)
    prospective_interval_m:     int    — minutes between checks (default 5)
    prospective_model:          str    — model for NL due_at evaluation (default "summarizer-gemini")
    prospective_reminder_imp:   int    — importance of injected reminder rows (default 9)

Runtime control:
    get_prospective_stats()     → dict of counters + last-run info
    set_runtime_override(k, v)  → shadow config (same _overrides dict as contradiction.py)
    trigger_now()               → wake sleeping loop immediately
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta

log = logging.getLogger("prospective")

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_stats: dict = {
    "checks_run":       0,
    "rows_evaluated":   0,
    "rows_fired":       0,
    "reminders_written":0,
    "last_check_at":    None,
    "last_check_duration_s": None,
    "last_error":       None,
    "last_feedback":    None,
}

_wake_event: asyncio.Event | None = None

_PLUGINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")


def get_prospective_stats() -> dict:
    return dict(_stats)


def trigger_now() -> None:
    if _wake_event:
        _wake_event.set()


# ---------------------------------------------------------------------------
# Config helper — imports _overrides from contradiction to share runtime state
# ---------------------------------------------------------------------------

def _pcogn_cfg() -> dict:
    try:
        with open(_PLUGINS_PATH) as f:
            raw = json.load(f).get("plugin_config", {}).get("proactive_cognition", {})
    except Exception:
        raw = {}

    # Import shared overrides dict from contradiction module
    try:
        from contradiction import get_runtime_overrides
        ovr = get_runtime_overrides()
    except ImportError:
        ovr = {}

    base = {
        "enabled":                  raw.get("enabled",                  False),
        "prospective_enabled":      raw.get("prospective_enabled",      True),
        "prospective_interval_m":   int(raw.get("prospective_interval_m",   5)),
        "prospective_model":        raw.get("prospective_model",        ""),
        "prospective_reminder_imp": int(raw.get("prospective_reminder_imp", 9)),
    }
    base.update(ovr)
    return base


# ---------------------------------------------------------------------------
# due_at parsing
# ---------------------------------------------------------------------------

def _try_parse_datetime(due_at: str) -> datetime | None:
    """
    Attempt to parse due_at as an ISO/common datetime string.
    Returns UTC-aware datetime or None if unparseable.
    """
    if not due_at:
        return None
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    s = due_at.strip()
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


async def _nl_is_overdue(due_at_text: str, model_key: str) -> bool:
    """
    Ask a cheap LLM whether a natural-language due_at phrase is overdue right now.
    Returns False on any error or ambiguity.

    Uses JSON output format ({"overdue": true/false}) for reliable parsing
    across both large and small models.
    """
    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage
    import json as _json, re as _re

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt = (
        f"Current date: {now_str}\n"
        f"Due date: {due_at_text}\n"
        "Is the due date before the current date?"
    )
    system = (
        "You compare dates. Given a current date and a due date, determine if "
        "the due date is in the past. Return ONLY a JSON object: "
        '{"overdue": true} or {"overdue": false}. No other text. '
        "If the due date is not a date (e.g. an event description), return "
        '{"overdue": false}.'
    )
    try:
        if model_key not in LLM_REGISTRY:
            return False
        cfg = LLM_REGISTRY[model_key]
        timeout = cfg.get("llm_call_timeout", 30)
        llm = _build_lc_llm(model_key)
        msgs = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        # Log per-call cost
        try:
            from cost_events import log_cost_event, _estimate_cost_for_model
            _usage = getattr(response, "usage_metadata", None) or {}
            _ti = _usage.get("input_tokens", 0) or 0
            _to = _usage.get("output_tokens", 0) or 0
            _cost = _estimate_cost_for_model(cfg, _ti, _to)
            if _cost is not None:
                asyncio.ensure_future(log_cost_event(
                    provider=cfg.get("host", "unknown").split("//")[-1].split("/")[0],
                    service=cfg.get("model_id", model_key),
                    tool_name="cogn-prospective",
                    model_key=model_key,
                    client_id="cogn-prospective",
                    cost_usd=_cost, tokens_in=_ti, tokens_out=_to, unit="tokens",
                ))
        except Exception:
            pass
        raw = _content_to_str(response.content).strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        raw = raw.strip()

        # Try JSON parse
        try:
            obj = _json.loads(raw)
            return bool(obj.get("overdue", False))
        except (_json.JSONDecodeError, ValueError):
            pass

        # Fallback: regex for {"overdue": true/false}
        m = _re.search(r'"overdue"\s*:\s*(true|false)', raw, _re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"

        # Legacy fallback: YES/NO (for models that still use that format)
        upper = raw.upper()
        if upper.startswith("YES"):
            return True
        return False
    except Exception as e:
        log.debug(f"prospective: NL due_at check failed: {e}")
        return False


async def _is_overdue(row: dict, model_key: str) -> bool:
    """Return True if a prospective row should fire now."""
    due_at = (row.get("due_at") or "").strip()
    if not due_at:
        # No due_at — never auto-fire (would need explicit model trigger)
        return False

    dt = _try_parse_datetime(due_at)
    if dt is not None:
        return datetime.now(timezone.utc) >= dt

    # Natural-language fallback
    return await _nl_is_overdue(due_at, model_key)


# ---------------------------------------------------------------------------
# Core check logic
# ---------------------------------------------------------------------------

async def _fetch_pending() -> list[dict]:
    from database import fetch_dicts
    from memory import _PROSPECTIVE
    try:
        return await fetch_dicts(
            f"SELECT id, topic, content, due_at, importance "
            f"FROM {_PROSPECTIVE()} WHERE status = 'pending' ORDER BY importance DESC"
        ) or []
    except Exception as e:
        log.warning(f"prospective: fetch_pending failed: {e}")
        return []


async def _inject_reminder(row: dict, reminder_imp: int) -> bool:
    """Write a high-importance reminder row into cognition table. Returns True on success."""
    from memory import save_cognition
    due_note = f" (was due: {row['due_at']})" if row.get("due_at") else ""
    content = f"[PROSPECTIVE REMINDER]{due_note} {row.get('content', '')}".strip()
    try:
        rid = await save_cognition(
            origin="prospective",
            topic=row.get("topic", "prospective-reminder"),
            content=content,
            importance=reminder_imp,
        )
        return rid > 0
    except Exception as e:
        log.warning(f"prospective: inject_reminder failed: {e}")
        return False


async def _mark_done(row_id: int) -> None:
    from database import execute_sql
    from memory import _PROSPECTIVE
    try:
        await execute_sql(
            f"UPDATE {_PROSPECTIVE()} SET status='done' WHERE id={row_id}"
        )
    except Exception as e:
        log.warning(f"prospective: mark_done failed for id={row_id}: {e}")


async def run_check() -> dict:
    """
    Run one prospective memory check pass. Safe to call manually.
    Returns summary dict.
    """
    cfg = _pcogn_cfg()
    model_key   = cfg["prospective_model"]
    if not model_key:
        from config import get_model_role
        try:
            model_key = get_model_role("prospective")
        except KeyError:
            model_key = "summarizer-gemini"
    reminder_imp = cfg["prospective_reminder_imp"]

    from database import set_db_override, list_managed_databases

    t_start = time.monotonic()
    summary = {"evaluated": 0, "fired": 0, "reminders": 0, "error": None}

    for db_name in list_managed_databases():
        set_db_override(db_name)
        try:
            rows = await _fetch_pending()
            summary["evaluated"] += len(rows)
            _stats["rows_evaluated"] += len(rows)

            for row in rows:
                overdue = await _is_overdue(row, model_key)
                if not overdue:
                    continue

                _stats["rows_fired"] += 1
                summary["fired"] += 1

                wrote = await _inject_reminder(row, reminder_imp)
                if wrote:
                    _stats["reminders_written"] += 1
                    summary["reminders"] += 1
                    import asyncio as _asyncio
                    try:
                        import notifier as _notifier
                        _asyncio.ensure_future(_notifier.fire_event(
                            "prospective_reminder",
                            f"topic={row.get('topic')!r}  due_at={row.get('due_at') or 'now'}",
                            (row.get("content") or "")[:200],
                        ))
                    except Exception:
                        pass

                # Auto-create cognition step for FIRE: directives
                raw_content = (row.get("content") or "").strip()
                if raw_content.upper().startswith("FIRE:"):
                    step_desc = raw_content[len("FIRE:"):].strip()
                    if step_desc:
                        try:
                            from plugin_mcp_direct import _queue_cogn_step
                            _asyncio.ensure_future(_queue_cogn_step(step_desc))
                            log.info(
                                f"prospective: auto-created cogn step for "
                                f"id={row['id']} topic={row.get('topic')!r}"
                            )
                        except Exception as e:
                            log.warning(f"prospective: auto-create cogn step failed: {e}")

                await _mark_done(row["id"])
                log.info(
                    f"prospective[{db_name}]: fired id={row['id']} topic={row.get('topic')!r} "
                    f"due_at={row.get('due_at')!r}"
                )

        except Exception as e:
            log.error(f"prospective[{db_name}]: check error: {e}")
            summary["error"] = str(e)
            _stats["last_error"] = str(e)
    set_db_override("")

    _stats["checks_run"] += 1
    _stats["last_check_at"] = datetime.now(timezone.utc).isoformat()
    _stats["last_check_duration_s"] = round(time.monotonic() - t_start, 2)

    # Feedback evaluation — run every Nth check to avoid thrashing on 5-min cycles
    # Only evaluate when at least one reminder has been written (something to judge)
    if _stats["reminders_written"] > 0 and _stats["checks_run"] % 12 == 0:
        try:
            from cogn_feedback import evaluate, LOOP_PROSPECTIVE
            fb = await evaluate(LOOP_PROSPECTIVE, summary)
            _stats["last_feedback"] = fb
            if fb.get("verdict") not in (None, "insufficient_data", "neutral", "useful"):
                log.info(f"prospective: feedback verdict={fb.get('verdict')} strength={fb.get('strength')}")
        except Exception as e:
            log.warning(f"prospective: feedback evaluation failed: {e}")

    return summary


# ---------------------------------------------------------------------------
# Background task entry point
# ---------------------------------------------------------------------------

async def prospective_task() -> None:
    """
    Long-running asyncio task. Loops every prospective_interval_m minutes.
    Wakes early if trigger_now() is called.
    """
    from timer_registry import register_timer, timer_sleep
    from state import fmt_interval as _fmt_iv
    register_timer("prospective", _fmt_iv(_pcogn_cfg().get("prospective_interval_m", 5)))

    global _wake_event
    _wake_event = asyncio.Event()

    # Skip immediate run on startup — wait before first run
    log.info("prospective_task: startup delay 5m before first run")
    try:
        await asyncio.wait_for(_wake_event.wait(), timeout=300)
        _wake_event.clear()
    except asyncio.TimeoutError:
        pass

    while True:
        cfg = _pcogn_cfg()

        if not cfg["enabled"] or not cfg["prospective_enabled"]:
            _wake_event.clear()
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=300)
                _wake_event.clear()
            except asyncio.TimeoutError:
                pass
            continue

        interval_m = cfg["prospective_interval_m"]
        if interval_m <= 0:
            await asyncio.sleep(300)
            continue

        try:
            await run_check()
        except Exception as e:
            log.warning(f"prospective_task: unhandled error: {e}")
            _stats["last_error"] = str(e)

        # Backoff: double interval for every 10 min of inactivity, cap at 120 min
        from state import backoff_interval, idle_seconds, fmt_interval
        effective_m = backoff_interval(interval_m, 120)
        sleep_sec = effective_m * 60
        if effective_m != interval_m:
            log.info(f"prospective: backoff {interval_m}m → {effective_m:.0f}m (idle {idle_seconds()/60:.0f}m)")
        timer_sleep("prospective", sleep_sec, interval_desc=fmt_interval(effective_m))
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            log.info("prospective_task: woken early by trigger")
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass
