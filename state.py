import asyncio
import json
import logging
import os
import re
from contextvars import ContextVar

_log = logging.getLogger("agent")

# Directory for persisted session histories (relative to this file)
SESSION_HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session-history")


def _safe_filename(session_id: str) -> str:
    """Convert a session_id to a safe filename component."""
    # Replace any character that isn't alphanumeric, hyphen, underscore, or dot
    return re.sub(r"[^A-Za-z0-9._-]", "_", session_id)


def save_history(session_id: str, history: list) -> None:
    """Persist a session's history to disk before reaping."""
    if not history:
        return
    try:
        os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)
        path = os.path.join(SESSION_HISTORY_DIR, f"{_safe_filename(session_id)}.history")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        _log.info(f"Session history saved ({len(history)} messages): {path}")
    except Exception as e:
        _log.warning(f"Failed to save session history for {session_id}: {e}")


def load_history(session_id: str) -> list:
    """Load a session's persisted history from disk, if it exists."""
    path = os.path.join(SESSION_HISTORY_DIR, f"{_safe_filename(session_id)}.history")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if isinstance(history, list):
            _log.info(f"Session history loaded ({len(history)} messages): {path}")
            return history
    except Exception as e:
        _log.warning(f"Failed to load session history for {session_id}: {e}")
    return []


def delete_history(session_id: str) -> bool:
    """Delete the persisted history file for a session, if it exists. Returns True if deleted."""
    path = os.path.join(SESSION_HISTORY_DIR, f"{_safe_filename(session_id)}.history")
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
        _log.info(f"Session history file deleted: {path}")
        return True
    except Exception as e:
        _log.warning(f"Failed to delete session history for {session_id}: {e}")
        return False


# Keys from the session dict that are user-configurable via !config and should
# survive a reap/reconnect cycle.
SESSION_CONFIG_KEYS = ("agent_call_stream", "tool_preview_length", "tool_suppress", "memory_scan_suppress", "stream_level", "memory_enabled", "auto_enrich", "model", "database")


def save_session_config(session_id: str, session: dict) -> None:
    """Persist user-configurable session settings to disk."""
    cfg = {k: session[k] for k in SESSION_CONFIG_KEYS if k in session}
    if not cfg:
        return
    try:
        os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)
        path = os.path.join(SESSION_HISTORY_DIR, f"{_safe_filename(session_id)}.config")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)
        _log.info(f"Session config saved: {path}")
    except Exception as e:
        _log.warning(f"Failed to save session config for {session_id}: {e}")


def load_session_config(session_id: str) -> dict:
    """Load persisted session config from disk. Returns {} if none exists."""
    path = os.path.join(SESSION_HISTORY_DIR, f"{_safe_filename(session_id)}.config")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            _log.info(f"Session config loaded: {path}")
            return cfg
    except Exception as e:
        _log.warning(f"Failed to load session config for {session_id}: {e}")
    return {}

# ---------------------------------------------------------------------------
# History size estimation
# ---------------------------------------------------------------------------

def estimate_history_size(history: list) -> dict:
    """
    Estimate the byte/token footprint of a session history list.

    Returns a dict with:
      char_count  - raw character count across all message content strings
      token_est   - rough token estimate (chars / 4, the widely-used rule of thumb)
    """
    chars = 0
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            # LangChain multi-part content blocks
            for part in content:
                if isinstance(part, dict):
                    chars += len(part.get("text", ""))
                elif isinstance(part, str):
                    chars += len(part)
    return {"char_count": chars, "token_est": chars // 4}


def update_session_token_stats(session: dict, usage_metadata: dict) -> None:
    """
    Accumulate real token counts from a LangChain usage_metadata dict.

    Expected keys (all optional — not every backend provides all of them):
      input_tokens, output_tokens, total_tokens

    Session keys updated:
      tokens_in_total   — cumulative input tokens (lifetime of session)
      tokens_out_total  — cumulative output tokens
      tokens_in_last    — input tokens from most recent LLM call
      tokens_out_last   — output tokens from most recent LLM call
    """
    if not usage_metadata:
        return
    inp = usage_metadata.get("input_tokens", 0) or 0
    out = usage_metadata.get("output_tokens", 0) or 0
    if inp == 0 and out == 0:
        return
    session["tokens_in_total"] = session.get("tokens_in_total", 0) + inp
    session["tokens_out_total"] = session.get("tokens_out_total", 0) + out
    session["tokens_in_last"] = inp
    session["tokens_out_last"] = out


def format_session_token_line(session: dict) -> str:
    """
    Return a one-line summary of token usage for display in !session.

    Includes: last-call counts, cumulative totals, and per-hour rates
    (computed from session creation time stored in 'created_at').
    """
    import time as _time

    in_total = session.get("tokens_in_total", 0)
    out_total = session.get("tokens_out_total", 0)
    in_last = session.get("tokens_in_last")
    out_last = session.get("tokens_out_last")

    # Per-hour rate: use session creation time; fall back to last_active if absent
    created = session.get("created_at") or session.get("last_active")
    if created:
        elapsed_hours = max((_time.time() - created) / 3600.0, 1 / 3600.0)
        in_rate = int(in_total / elapsed_hours)
        out_rate = int(out_total / elapsed_hours)
        rate_str = f", rate: in={_fmt_k(in_rate)}/hr out={_fmt_k(out_rate)}/hr"
    else:
        rate_str = ""

    if in_total == 0 and out_total == 0:
        return "  tokens: no LLM calls yet (or provider doesn't report usage)"

    last_str = ""
    if in_last is not None:
        last_str = f"last: in={_fmt_k(in_last)} out={_fmt_k(out_last or 0)} | "

    return (
        f"  tokens: {last_str}"
        f"total: in={_fmt_k(in_total)} out={_fmt_k(out_total)}"
        f"{rate_str}"
    )


def _fmt_k(n: int) -> str:
    """Format an integer compactly: 1234 -> '1.2k', 123456 -> '123k', 999 -> '999'."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


# Current client ID context variable — set in execute_tool() so executors can
# read it without needing it as an explicit parameter.
current_client_id: ContextVar[str] = ContextVar("current_client_id", default="")

# Per-client SSE queues
sse_queues: dict[str, asyncio.Queue] = {}
queue_lock = asyncio.Lock()

# Session store
sessions: dict[str, dict] = {}

# Global chat activity tracker — used by background tasks to backoff when idle
import time as _time
_last_chat_ts: float = _time.time()  # seed with startup time

def update_chat_activity() -> None:
    """Call on every incoming user message to reset backoff clocks.
    Also wakes backed-off cognitive tasks so they snap back to base interval.
    """
    global _last_chat_ts
    was_idle = (_time.time() - _last_chat_ts) > 600  # was idle > 10 min
    _last_chat_ts = _time.time()
    if was_idle:
        # Wake backed-off tasks so they resume at base interval
        for _mod, _fn in [("contradiction", "trigger_now"),
                          ("prospective", "trigger_now"),
                          ("reflection", "trigger_now"),
                          ("temporal_inference", "trigger_now"),
                          ("memreview_auto", "trigger_now")]:
            try:
                import importlib
                m = importlib.import_module(_mod)
                getattr(m, _fn)()
            except Exception:
                pass

def idle_seconds() -> float:
    """Seconds since last chat activity across any session."""
    return _time.time() - _last_chat_ts

def backoff_interval(base_m: float, cap_m: float) -> float:
    """Compute backoff interval in minutes given idle time.
    Doubles base every 10 min of inactivity, capped at cap_m.
    Returns base_m when chat is active (idle < 10 min).
    """
    idle = idle_seconds()
    if idle < 600:
        return base_m
    doublings = int(idle // 600)
    return min(base_m * (2 ** doublings), cap_m)

def fmt_interval(minutes: float) -> str:
    """Format interval in minutes to human-readable string (e.g. '2m', '1h', '24h')."""
    if minutes >= 60:
        h = minutes / 60
        return f"{h:.0f}h" if h == int(h) else f"{h:.1f}h"
    return f"{minutes:.0f}m" if minutes == int(minutes) else f"{minutes:.1f}m"

# Session reaper wake event — set when a session is created so the reaper
# can resume from disabled/long-sleep immediately.
_reaper_wake: asyncio.Event | None = None

def init_reaper_wake() -> asyncio.Event:
    """Create the reaper wake event (called once from the reaper task)."""
    global _reaper_wake
    _reaper_wake = asyncio.Event()
    return _reaper_wake

def wake_reaper() -> None:
    """Signal the session reaper that a session now exists."""
    if _reaper_wake is not None:
        _reaper_wake.set()

# Session ID management - shorthand IDs for user convenience
_session_id_counter = 100  # Start at 100 for readability
session_id_to_shorthand: dict[str, int] = {}  # full_session_id -> shorthand_id
shorthand_to_session_id: dict[int, str] = {}  # shorthand_id -> full_session_id

def get_or_create_shorthand_id(session_id: str) -> int:
    """Get existing shorthand ID or create a new one."""
    global _session_id_counter
    if session_id not in session_id_to_shorthand:
        _session_id_counter += 1
        shorthand_id = _session_id_counter
        session_id_to_shorthand[session_id] = shorthand_id
        shorthand_to_session_id[shorthand_id] = session_id
    return session_id_to_shorthand[session_id]

def get_session_by_shorthand(shorthand_id: int) -> str | None:
    """Look up full session ID by shorthand ID."""
    return shorthand_to_session_id.get(shorthand_id)

def remove_shorthand_mapping(session_id: str):
    """Remove shorthand mappings when session is deleted."""
    if session_id in session_id_to_shorthand:
        shorthand_id = session_id_to_shorthand[session_id]
        del session_id_to_shorthand[session_id]
        del shorthand_to_session_id[shorthand_id]

# Active request tasks — one per client_id. Cancelled when a new request arrives
# or when the model is switched while a request is in flight.
active_tasks: dict[str, asyncio.Task] = {}

async def get_queue(client_id: str) -> asyncio.Queue:
    async with queue_lock:
        if client_id not in sse_queues:
            sse_queues[client_id] = asyncio.Queue()
        return sse_queues[client_id]

async def drain_queue(client_id: str) -> int:
    """Discard all pending items in the client's SSE queue. Returns count drained."""
    if client_id not in sse_queues:
        return 0
    q = sse_queues[client_id]
    count = 0
    while not q.empty():
        try:
            q.get_nowait()
            count += 1
        except asyncio.QueueEmpty:
            break
    return count

async def push_tok(client_id: str, text: str):
    (await get_queue(client_id)).put_nowait({"t": "tok", "d": text.replace("\n", "\\n")})

async def push_done(client_id: str):
    (await get_queue(client_id)).put_nowait({"t": "done"})

async def push_flush(client_id: str):
    """Intermediate flush: shell.py treats this like a soft done (clears reply buffer);
    api_client ignores it and keeps streaming.  Use for tool-result round trips that
    are followed by more LLM turns."""
    (await get_queue(client_id)).put_nowait({"t": "flush"})

async def push_err(client_id: str, msg: str):
    (await get_queue(client_id)).put_nowait({"t": "err", "d": msg})
    (await get_queue(client_id)).put_nowait({"t": "done"})

async def push_model(client_id: str, model_key: str):
    (await get_queue(client_id)).put_nowait({"t": "model", "d": model_key})

async def push_notif(client_id: str, text: str):
    """Push an async notification event to a session's SSE queue."""
    (await get_queue(client_id)).put_nowait({"t": "notif", "d": text.replace("\n", "\\n")})

async def push_close(client_id: str):
    """Signal the SSE generator to terminate, then remove the queue.

    Used by the session reaper and explicit session deletion so the
    server-side EventSourceResponse generator breaks out of its loop
    instead of spinning forever after the session dict is gone.
    """
    if client_id in sse_queues:
        sse_queues[client_id].put_nowait({"t": "close"})
        # Don't delete the queue yet — the generator needs to read the
        # sentinel first.  It will be garbage-collected when the generator
        # exits and no more references remain.

# ---------------------------------------------------------------------------
# Gate infrastructure
# Per-client pending gate requests: client_id -> asyncio.Future[bool]
# ---------------------------------------------------------------------------
_gate_futures: dict[str, asyncio.Future] = {}


def has_pending_gate(client_id: str) -> bool:
    """True if there is an unanswered gate request for this client."""
    return client_id in _gate_futures


async def wait_for_gate(client_id: str, timeout: float = 120.0) -> bool:
    """
    Create a Future for the given client and wait up to `timeout` seconds
    for the user to answer Y/N via resolve_gate().

    Returns True (allow) or False (deny). Auto-denies on timeout.
    """
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    _gate_futures[client_id] = fut
    try:
        return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
    except asyncio.TimeoutError:
        return False
    finally:
        _gate_futures.pop(client_id, None)


def resolve_gate(client_id: str, approved: bool) -> bool:
    """
    Answer the pending gate Future for the given client.

    Returns True if a pending gate was resolved, False if there was none.
    """
    fut = _gate_futures.get(client_id)
    if fut is None or fut.done():
        return False
    fut.set_result(approved)
    return True


# ---------------------------------------------------------------------------
# Gate state — used by gate.py
# ---------------------------------------------------------------------------

# Per-client pending gate events: gate_id -> {"event": asyncio.Event, "decision": str|None}
pending_gates: dict[str, dict] = {}

# Per-table db gate permissions: table_name -> {"read": bool, "write": bool}
# Populated by !autoAIdb commands in routes.py
auto_aidb_state: dict[str, dict] = {}

# Per-tool gate permissions: tool_name -> {"read": bool, "write": bool}
# Populated by !autogate commands in routes.py
tool_gate_state: dict[str, dict] = {}

# Per-client active gate data (for shell.py display): client_id -> gate_data dict
_client_active_gates: dict[str, dict] = {}


async def push_gate(client_id: str, gate_data: dict) -> None:
    """Push a gate event to the client's SSE queue."""
    import json as _json
    q = await get_queue(client_id)
    q.put_nowait({"t": "gate", "d": _json.dumps(gate_data)})
    _client_active_gates[client_id] = gate_data


def clear_client_gate(client_id: str) -> None:
    """Remove the active gate record for a client after it is resolved."""
    _client_active_gates.pop(client_id, None)


async def cancel_active_task(client_id: str) -> bool:
    """
    Cancel any in-flight request task for this client and drain its output queue.

    Returns True if a task was cancelled, False if there was nothing running.
    Called before spawning a new request task, and on !model switch.
    """
    task = active_tasks.get(client_id)
    # Don't cancel ourselves — the current task is the one processing !model
    if task and not task.done() and task is not asyncio.current_task():
        task.cancel()
        try:
            # Wait up to 5s for the task to acknowledge cancellation.
            # If it's stuck in a non-cancellable I/O call we don't block forever.
            done, _ = await asyncio.wait({task}, timeout=5)
            if not done:
                # Task didn't finish in time — log and move on
                import logging
                logging.getLogger("agent").warning(
                    f"cancel_active_task: task for '{client_id}' didn't finish in 5s after cancel"
                )
        except Exception:
            pass
        active_tasks.pop(client_id, None)
        # Drain any stale tokens the cancelled task already queued
        if client_id in sse_queues:
            q = sse_queues[client_id]
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        return True
    active_tasks.pop(client_id, None)
    return False