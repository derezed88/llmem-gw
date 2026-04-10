"""
notifier.py — Async event notification system.

Pushes SSE "notif" events to registered sessions when system events fire.
Events are suppressed when the target session was recently active (quiet period)
to avoid interrupting active users.

When a target session is offline, events are queued in memory and delivered
the next time that session connects (endpoint_stream drain).

Delivery hooks: non-SSE clients (e.g. Slack) register a custom delivery
function via register_delivery_hook(prefix, handler). When a target session's
client_id matches the prefix, the hook is called instead of push_notif, so
notifications bypass the SSE queue and go directly to Slack (or any other
transport that doesn't have a persistent queue consumer).

Commands (dispatched via !notifier):
  add <id> [event1,event2,...]   — register a session for notifications
  list                           — show all targets and pending counts
  delete <id>                    — remove a target
  clear                          — remove all targets
  events <id> <event1,event2,..> — update event subscriptions for a session
  events                         — show available event names
  quiet <id> <minutes>           — set quiet period (default 10)

Config: notifier.json (auto-created in cwd)
Format: {"targets": [{"session_id": 105, "events": [...], "quiet_minutes": 10}]}

Available event types:
  goal_created, goal_updated, goal_completed, goal_blocked, goal_abandoned
  task_created, task_updated, task_completed
  belief_saved, contradiction_detected
  prospective_reminder, temporal_pattern_inferred
  tool_called
"""

import json
import logging
import os
import time
from typing import Callable, Awaitable, Optional

log = logging.getLogger("notifier")

_CONFIG_FILE = "notifier.json"

ALL_EVENTS: list[str] = [
    "goal_created",
    "goal_updated",
    "goal_completed",
    "goal_blocked",
    "goal_abandoned",
    "task_created",
    "task_updated",
    "task_completed",
    "belief_saved",
    "contradiction_detected",
    "interrupt",
    "prospective_reminder",
    "temporal_pattern_inferred",
    "tool_called",
    "goal_plan_proposed",
    "goal_step_waiting_user",
    "sms_received",
    "email_important",
]

_targets: list[dict] = []
_pending: dict[int, list[dict]] = {}  # shorthand_id -> queued notifications for offline delivery

# Delivery hooks for non-SSE clients (e.g. Slack).
# Key = client_id prefix, Value = async callable(client_id, msg) -> bool
_delivery_hooks: dict[str, Callable[..., Awaitable[bool]]] = {}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load() -> None:
    global _targets
    if not os.path.exists(_CONFIG_FILE):
        return
    try:
        with open(_CONFIG_FILE) as f:
            data = json.load(f)
        _targets = [
            {
                "session_id": int(t["session_id"]),
                "events": set(t.get("events", ALL_EVENTS)),
                "quiet_minutes": int(t.get("quiet_minutes", 10)),
                "client_id_prefix": t.get("client_id_prefix", ""),
            }
            for t in data.get("targets", [])
        ]
        log.info(f"notifier: loaded {len(_targets)} target(s)")
    except Exception as e:
        log.warning(f"notifier: config load failed: {e}")


def _save() -> None:
    try:
        data = {
            "targets": [
                {
                    "session_id": t["session_id"],
                    "events": sorted(t["events"]),
                    "quiet_minutes": t["quiet_minutes"],
                    **({"client_id_prefix": t["client_id_prefix"]} if t.get("client_id_prefix") else {}),
                }
                for t in _targets
            ]
        }
        with open(_CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning(f"notifier: config save failed: {e}")


def _is_enabled() -> bool:
    try:
        with open("plugins-enabled.json") as f:
            cfg = json.load(f)
        return bool(cfg.get("plugin_config", {}).get("notifier", {}).get("enabled", True))
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Delivery hook registration (for non-SSE clients)
# ---------------------------------------------------------------------------

def register_delivery_hook(
    prefix: str,
    handler: Callable[..., Awaitable[bool]],
) -> None:
    """Register an async delivery function for client_ids starting with prefix.

    handler signature: async def handler(client_id: str, msg: str) -> bool
    Returns True if delivery succeeded, False to fall back to pending queue.
    """
    _delivery_hooks[prefix] = handler
    log.info(f"notifier: registered delivery hook for prefix={prefix!r}")


def _find_delivery_hook(client_id: str) -> Optional[Callable[..., Awaitable[bool]]]:
    """Find the first matching delivery hook for a client_id."""
    for prefix, handler in _delivery_hooks.items():
        if client_id.startswith(prefix):
            return handler
    return None


# ---------------------------------------------------------------------------
# Target management (called from !notifier command handlers)
# ---------------------------------------------------------------------------

def add_target(session_id: int, events: list[str] | None = None, quiet_minutes: int = 10) -> str:
    for t in _targets:
        if t["session_id"] == session_id:
            return (
                f"Session {session_id} already registered. "
                f"Use !notifier events {session_id} <list> to update subscriptions."
            )
    evt = set(events) & set(ALL_EVENTS) if events else set(ALL_EVENTS)
    _targets.append({"session_id": session_id, "events": evt, "quiet_minutes": quiet_minutes})
    _save()
    ev_str = "all events" if evt == set(ALL_EVENTS) else f"{len(evt)} events"
    return f"Notifier: session {session_id} added ({ev_str}, quiet={quiet_minutes}m)"


def add_target_prefix(prefix: str, events: list[str] | None = None, quiet_minutes: int = 10) -> str:
    """Register a client_id prefix as a notification target.

    All sessions whose client_id starts with prefix will receive notifications.
    If a target with this prefix already exists, its event list is updated.
    Persisted to notifier.json so it survives restarts.
    """
    # Use a synthetic session_id derived from prefix hash (stable across restarts)
    synthetic_id = abs(hash(prefix)) % 900000 + 100000
    evt = set(events) & set(ALL_EVENTS) if events else set(ALL_EVENTS)

    for t in _targets:
        if t.get("client_id_prefix") == prefix:
            t["events"] = evt
            t["quiet_minutes"] = quiet_minutes
            _save()
            return f"Notifier: prefix '{prefix}' updated ({len(evt)} events, quiet={quiet_minutes}m)"

    _targets.append({
        "session_id": synthetic_id,
        "events": evt,
        "quiet_minutes": quiet_minutes,
        "client_id_prefix": prefix,
    })
    _save()
    return f"Notifier: prefix '{prefix}' registered (id={synthetic_id}, {len(evt)} events, quiet={quiet_minutes}m)"


def remove_target_prefix(prefix: str) -> str:
    """Remove a prefix-based notification target."""
    for t in _targets[:]:
        if t.get("client_id_prefix") == prefix:
            _targets.remove(t)
            _save()
            return f"Notifier: prefix '{prefix}' removed"
    return f"Notifier: prefix '{prefix}' not found"


def remove_target(session_id: int) -> str:
    for t in _targets[:]:
        if t["session_id"] == session_id:
            _targets.remove(t)
            _pending.pop(session_id, None)
            _save()
            return f"Notifier: session {session_id} removed"
    return f"Notifier: session {session_id} not found"


def clear_targets() -> str:
    count = len(_targets)
    _targets.clear()
    _pending.clear()
    _save()
    return f"Notifier: cleared {count} target(s)"


def list_targets() -> str:
    if not _targets:
        return "Notifier: no targets configured\nUse !notifier add <session_id> to register one."
    lines = [f"Notifier targets (enabled={_is_enabled()}):"]
    for t in _targets:
        ev_count = len(t["events"])
        if ev_count == len(ALL_EVENTS):
            ev_str = f"all ({ev_count})"
        else:
            ev_str = ", ".join(sorted(t["events"]))
        pending_count = len(_pending.get(t["session_id"], []))
        pending_str = f"  [{pending_count} pending]" if pending_count else ""
        lines.append(
            f"  [{t['session_id']}] events=[{ev_str}]  quiet={t['quiet_minutes']}m{pending_str}"
        )
    return "\n".join(lines)


def update_events(session_id: int, events: list[str]) -> str:
    for t in _targets:
        if t["session_id"] == session_id:
            valid = set(e for e in events if e in ALL_EVENTS)
            invalid = set(events) - valid
            t["events"] = valid
            _save()
            msg = f"Notifier: session {session_id} subscribed to: {', '.join(sorted(valid))}"
            if invalid:
                msg += f"\n  Ignored unknown events: {', '.join(sorted(invalid))}"
            return msg
    return f"Notifier: session {session_id} not found. Use !notifier add {session_id} first."


def update_quiet(session_id: int, minutes: int) -> str:
    for t in _targets:
        if t["session_id"] == session_id:
            t["quiet_minutes"] = minutes
            _save()
            return f"Notifier: session {session_id} quiet period = {minutes}m"
    return f"Notifier: session {session_id} not found"


def show_events() -> str:
    lines = ["Available notification events:"]
    for ev in ALL_EVENTS:
        lines.append(f"  {ev}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pending delivery — for sessions that were offline when an event fired
# ---------------------------------------------------------------------------

def drain_pending(shorthand_id: int) -> list[dict]:
    """Return and remove all pending notifications for a session. Called on SSE connect."""
    return _pending.pop(shorthand_id, [])


# ---------------------------------------------------------------------------
# Event firing — called from system hooks
# ---------------------------------------------------------------------------

async def fire_event(event_type: str, summary: str, detail: str = "") -> None:
    """
    Fire a notification to all sessions subscribed to event_type.
    Skips sessions that were active within their quiet_minutes window.
    Queues notifications for sessions that are currently offline.

    For sessions with a registered delivery hook (e.g. Slack), the hook is
    called directly — bypassing the SSE queue which has no persistent consumer.
    """
    if not _targets or not _is_enabled():
        return

    from state import sessions, get_session_by_shorthand, push_notif  # type: ignore

    ts = time.strftime("%H:%M:%S")
    lines = [f"[NOTIFY {ts}] {event_type.upper()}: {summary}"]
    if detail:
        lines.append(f"  {detail}")
    msg = "\n".join(lines)

    for target in _targets:
        if event_type not in target["events"]:
            continue

        shorthand_id: int = target["session_id"]
        quiet_seconds: int = target["quiet_minutes"] * 60

        # Resolve client_id: prefer client_id_prefix match, fall back to shorthand_id
        client_id = None
        prefix = target.get("client_id_prefix", "")
        if prefix:
            for cid in sessions:
                if cid.startswith(prefix):
                    client_id = cid
                    break
        if not client_id:
            client_id = get_session_by_shorthand(shorthand_id)
        if not client_id or client_id not in sessions:
            _pending.setdefault(shorthand_id, []).append(
                {"event_type": event_type, "summary": summary, "detail": detail, "ts": ts}
            )
            log.debug(f"notifier: session {shorthand_id} offline, queued {event_type}")
            continue

        session = sessions[client_id]
        idle_seconds = time.time() - session.get("last_active", 0)
        if idle_seconds < quiet_seconds:
            log.debug(
                f"notifier: session {shorthand_id} active "
                f"({idle_seconds:.0f}s < {quiet_seconds}s), skip {event_type}"
            )
            continue

        # Check for a delivery hook (e.g. Slack direct post)
        hook = _find_delivery_hook(client_id)
        if hook:
            try:
                delivered = await hook(client_id, msg)
                if delivered:
                    log.info(f"notifier: → session {shorthand_id} (hook): {event_type}")
                    continue
                # Hook returned False — fall through to pending
            except Exception as e:
                log.warning(f"notifier: delivery hook failed for {shorthand_id}: {e}")
            # Hook failed or returned False — queue as pending
            _pending.setdefault(shorthand_id, []).append(
                {"event_type": event_type, "summary": summary, "detail": detail, "ts": ts}
            )
            continue

        # Default: push via SSE queue
        try:
            await push_notif(client_id, msg)
            log.info(f"notifier: → session {shorthand_id}: {event_type}")
        except Exception as e:
            log.warning(f"notifier: push failed for {shorthand_id}: {e}")


_load()
