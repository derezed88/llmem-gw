"""timer_registry.py — Central registry for all background async timers.

Each timer calls:
  register_timer(name, interval_desc)   — once at startup
  t0 = timer_start(name)               — before doing work
  timer_end(name, t0, error=...)        — after work (pass error string on exception)
  timer_sleep(name, sleep_sec)          — before asyncio.sleep()
  timer_disabled(name)                  — when config disables the timer

Routes can call get_all_timers() to render the !timers dashboard.
"""

import time
import logging
from datetime import datetime, timezone, timedelta

log = logging.getLogger("timer_registry")

# name → stat dict
_registry: dict[str, dict] = {}


def register_timer(name: str, interval_desc: str) -> None:
    """Register a timer. Safe to call multiple times — only updates interval_desc on repeat."""
    if name not in _registry:
        _registry[name] = {
            "interval_desc": interval_desc,
            "status": "starting",
            "run_count": 0,
            "last_run_at": None,
            "next_run_at": None,
            "last_duration_s": None,
            "last_error": None,
        }
    else:
        _registry[name]["interval_desc"] = interval_desc


def timer_start(name: str) -> float:
    """Mark timer as running. Returns monotonic t0 for duration tracking."""
    if name in _registry:
        _registry[name]["status"] = "running"
        _registry[name]["next_run_at"] = None
        _registry[name]["last_error"] = None
    return time.monotonic()


def timer_end(name: str, t0: float, error: str | None = None) -> None:
    """Record completed run. Pass error string if the cycle raised an exception."""
    dur = round(time.monotonic() - t0, 2)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if name in _registry:
        _registry[name]["last_run_at"] = now_iso
        _registry[name]["last_duration_s"] = dur
        _registry[name]["run_count"] = _registry[name].get("run_count", 0) + 1
        if error:
            _registry[name]["last_error"] = str(error)[:200]
            _registry[name]["status"] = "error"
        # status transitions to "sleeping" or "disabled" via subsequent calls


def timer_sleep(name: str, sleep_sec: float, interval_desc: str | None = None) -> None:
    """Record sleep start and expected wake time.
    Optionally update interval_desc to reflect actual effective interval."""
    wake = datetime.now(timezone.utc) + timedelta(seconds=sleep_sec)
    if name in _registry:
        _registry[name]["next_run_at"] = wake.strftime("%Y-%m-%dT%H:%M:%SZ")
        _registry[name]["status"] = "sleeping"
        if interval_desc is not None:
            _registry[name]["interval_desc"] = interval_desc


def timer_disabled(name: str) -> None:
    """Mark timer as disabled (config-driven)."""
    if name in _registry:
        _registry[name]["status"] = "disabled"
        _registry[name]["next_run_at"] = None


def get_all_timers() -> dict:
    return {k: dict(v) for k, v in _registry.items()}


def get_timer(name: str) -> dict | None:
    return dict(_registry[name]) if name in _registry else None
