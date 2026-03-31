"""
Automatic memory review — background task that periodically runs
!memreview (topics, types, classify) with auto-accept.

Config (plugins-enabled.json → plugin_config.memory):
    auto_review_enabled:    bool   — master toggle (default false)
    auto_review_interval_h: int    — hours between runs (default 6)
    auto_review_modes:      list   — subset of ["topics", "types", "classify"]
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("agent")

_PLUGINS_PATH = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
_CLIENT_ID = "__memreview_auto__"

_wake_event: asyncio.Event | None = None

_stats: dict = {
    "runs": 0,
    "topics_applied": 0,
    "types_applied": 0,
    "classify_applied": 0,
    "last_run_at": None,
    "last_run_duration_s": None,
    "last_error": None,
}


def _cfg() -> dict:
    try:
        with open(_PLUGINS_PATH) as f:
            raw = json.load(f).get("plugin_config", {}).get("memory", {})
    except Exception:
        raw = {}
    return {
        "enabled": raw.get("auto_review_enabled", False),
        "interval_h": int(raw.get("auto_review_interval_h", 6)),
        "modes": raw.get("auto_review_modes", ["topics", "types", "classify"]),
    }


_runtime_enabled: bool | None = None  # None = use config, True/False = override

def set_auto_enabled(enabled: bool | None) -> None:
    """Runtime override for auto_review_enabled. Pass None to revert to config."""
    global _runtime_enabled
    _runtime_enabled = enabled
    # Update timer registry immediately so !timers reflects the change
    from timer_registry import timer_disabled as _td
    if enabled is False:
        _td("memreview_auto")
    elif enabled is True:
        trigger_now()  # wake the sleeping task to resume
    elif enabled is None:
        # Reverted to config — if config says disabled, mark it
        if not _cfg()["enabled"]:
            _td("memreview_auto")

def is_auto_enabled() -> bool:
    """Return effective enabled state (runtime override wins over config)."""
    if _runtime_enabled is not None:
        return _runtime_enabled
    return _cfg()["enabled"]

def trigger_now() -> None:
    """Wake the sleeping task immediately."""
    if _wake_event:
        _wake_event.set()


def get_memreview_auto_stats() -> dict:
    return dict(_stats)


async def _run_cycle() -> dict:
    """Run all configured review modes with auto_accept=True across every database."""
    from routes import cmd_memreview, _pending_reviews, _pending_type_reviews, _pending_classify_reviews
    from database import set_db_override, list_managed_databases

    cfg = _cfg()
    modes = cfg["modes"]
    counts = {"topics": 0, "types": 0, "classify": 0}

    databases = list_managed_databases()
    for db_name in databases:
        set_db_override(db_name)
        log.info(f"memreview_auto: reviewing database {db_name}")

        try:
            if "topics" in modes:
                await cmd_memreview(_CLIENT_ID, arg="", model_key="", auto_accept=True)

            if "types" in modes:
                await cmd_memreview(_CLIENT_ID, arg="types", model_key="", auto_accept=True)

            if "classify" in modes:
                await cmd_memreview(_CLIENT_ID, arg="classify", model_key="", auto_accept=True)
        except Exception as e:
            log.warning(f"memreview_auto: error on db {db_name}: {e}")
        finally:
            # Clean up pending state between databases
            _pending_reviews.pop(_CLIENT_ID, None)
            _pending_type_reviews.pop(_CLIENT_ID, None)
            _pending_classify_reviews.pop(_CLIENT_ID, None)

    # Clear db override when done
    set_db_override("")

    # Clean up the SSE queue we created
    try:
        from state import sse_queues
        sse_queues.pop(_CLIENT_ID, None)
    except Exception:
        pass

    return counts


async def memreview_auto_task() -> None:
    """Long-running asyncio task. Runs auto-review every auto_review_interval_h hours."""
    import time as _time
    from timer_registry import register_timer, timer_start, timer_end, timer_sleep

    global _wake_event
    _wake_event = asyncio.Event()

    cfg = _cfg()
    initial_sleep = max(300, cfg["interval_h"] * 3600)

    if not is_auto_enabled():
        from timer_registry import timer_disabled
        timer_disabled("memreview_auto")
    else:
        register_timer("memreview_auto", f"{cfg['interval_h']}h")

    # Defer first run to the full interval — avoids expensive LLM calls on every restart
    timer_sleep("memreview_auto", initial_sleep)
    _wake_event.clear()
    try:
        await asyncio.wait_for(_wake_event.wait(), timeout=initial_sleep)
        _wake_event.clear()
    except asyncio.TimeoutError:
        pass

    while True:
        t0 = None
        try:
            cfg = _cfg()
            if not is_auto_enabled():
                from timer_registry import timer_disabled
                timer_disabled("memreview_auto")
                await asyncio.sleep(300)
                continue

            interval_h = cfg["interval_h"]
            if interval_h <= 0:
                from timer_registry import timer_disabled
                timer_disabled("memreview_auto")
                await asyncio.sleep(3600)
                continue

            register_timer("memreview_auto", f"{interval_h}h")
            t0 = timer_start("memreview_auto")

            counts = await _run_cycle()

            _stats["runs"] += 1
            _stats["topics_applied"] += counts.get("topics", 0)
            _stats["types_applied"] += counts.get("types", 0)
            _stats["classify_applied"] += counts.get("classify", 0)
            _stats["last_run_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _stats["last_run_duration_s"] = round(_time.monotonic() - t0, 1) if t0 else None
            _stats["last_error"] = None

            timer_end("memreview_auto", t0)
            log.info(
                f"memreview_auto cycle complete: "
                f"topics={counts['topics']}, types={counts['types']}, "
                f"classify={counts['classify']}"
            )
        except Exception as e:
            log.error(f"memreview_auto_task error: {e}")
            _stats["last_error"] = str(e)[:200]
            if t0 is not None:
                timer_end("memreview_auto", t0, error=str(e))

        # Sleep with backoff — jump to 24h when siblings at cap
        try:
            cfg = _cfg()
            sleep_sec = max(300, cfg["interval_h"] * 3600)
        except Exception:
            sleep_sec = 21600  # 6h fallback

        try:
            from state import backoff_interval, idle_seconds, fmt_interval
            with open(_PLUGINS_PATH) as _f:
                _pcog = json.load(_f).get("plugin_config", {}).get("proactive_cognition", {})
            contra_eff = backoff_interval(int(_pcog.get("contradiction_interval_m", 2)), 60)
            prosp_eff = backoff_interval(int(_pcog.get("prospective_interval_m", 1)), 120)
            if contra_eff >= 60 and prosp_eff >= 120:
                sleep_sec = 86400  # 24 hours
                log.info(f"memreview_auto: backoff → 24h (siblings at cap, idle {idle_seconds()/60:.0f}m)")
        except Exception:
            pass  # keep original sleep_sec on any error

        from state import fmt_interval as _fmt
        timer_sleep("memreview_auto", sleep_sec, interval_desc=_fmt(sleep_sec / 60))
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            log.info("memreview_auto_task: woken early by trigger")
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass
