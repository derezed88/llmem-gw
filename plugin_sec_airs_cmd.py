"""
plugin_sec_airs_cmd.py — !airs shell/Slack command for AIRS scan results.

NOT a history chain plugin — named outside plugin_history_* intentionally so
llmemctl.py history-list does not discover it.  Loaded at startup by agent-mcp.py
via the regular plugin manifest.
It reads violation data written by plugin_history_sec_async into the session dict
and presents it on demand from any client (shell.py, Slack, open-webui, etc.).

This demonstrates that scan results can be pulled by any interface — the data
lives in the session and can equally be forwarded to SIEM, ticketing systems,
or other consumers.

Commands:
    !airs                           — show last 10 violations (all sessions, from disk)
    !airs log [N] [filter]          — last N from disk, optional client_id filter
    !airs status <ID>               — blocked flag + counts for a specific session
    !airs violations <ID>           — in-memory violations for a specific session
    !airs unblock <ID>              — clear block flag (admin)

    <ID> can be a shorthand ID (e.g. 101) or full session ID prefix.
    Use !session to list active sessions.

Dependencies:
    plugin_history_sec_async or plugin_history_sec_sync must be in the history
    chain (they write the violation data into the session).

    Activation: add to plugin-manifest.json and plugins-enabled.json as a
    regular data_tool plugin, or import it directly in agent-mcp.py startup.
"""

import json
import logging
import os
import threading

log = logging.getLogger(__name__)

NAME = "sec_airs_cmd"

_DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "airs-violations.log")
_log_lock    = threading.Lock()


def _append_log_record(record: dict) -> None:
    """Append a JSON line to the shared violation log (thread-safe)."""
    try:
        with _log_lock:
            with open(_DEFAULT_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.error(f"plugin_sec_airs_cmd: failed to write log: {exc}")


# ---------------------------------------------------------------------------
# Session resolver — supports shorthand IDs and full/prefix session IDs
# ---------------------------------------------------------------------------

def _resolve_session(target: str):
    """
    Resolve a target string to (client_id, session).
    Accepts: shorthand integer ID (e.g. "101"), full session ID, or prefix.
    Returns (None, None) if not found.
    """
    from state import sessions, get_session_by_shorthand

    # Try shorthand integer ID
    try:
        full_id = get_session_by_shorthand(int(target))
        if full_id and full_id in sessions:
            return full_id, sessions[full_id]
        return None, None
    except (ValueError, TypeError):
        pass

    # Exact or prefix match on full session ID
    matches = [k for k in sessions if k == target or k.startswith(target)]
    if len(matches) == 1:
        return matches[0], sessions[matches[0]]
    return None, None


# ---------------------------------------------------------------------------
# !airs command handler
# ---------------------------------------------------------------------------

async def _cmd_airs(args: str) -> str:
    """
    !airs [violations|status|unblock] [<session_id>]

    (no args)             — show last 10 entries from disk log (all sessions)
    violations <ID>       — list in-memory violations for a session
    status     <ID>       — show blocked flag + violation count for a session
    unblock    <ID>       — clear the airs_blocked flag (admin action)
    log [N] [filter]      — last N entries from disk log (default N=10)

    <session_id> can be a shorthand ID (e.g. 101) or full session ID prefix.
    Omit to query the current session.
    """
    from state import current_client_id, sessions, get_or_create_shorthand_id

    caller_client_id = current_client_id.get("")
    if not caller_client_id:
        return "ERROR: !airs could not determine session (no client_id in context)"

    # Parse sub-command and optional target session ID
    # e.g. "status 101"  "violations slack-C0AE..."  "unblock 103"  "101"
    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else "violations"
    target_arg = parts[1].strip() if len(parts) > 1 else ""

    # If sub is not a known keyword, treat it as a session ID with default sub
    if sub not in ("violations", "status", "unblock", "log"):
        target_arg = sub  # the whole arg is the session ID
        sub = "violations"

    # ------------------------------------------------------------------
    # log: always reads from disk, no session needed
    # Also: no-arg defaults to log (current session status is useless —
    # if blocked, !airs can't even run; if not blocked, nothing to show)
    if sub == "log" or (sub in ("violations", "status") and not target_arg):
        n = 10
        filter_str = ""
        if target_arg:
            log_parts = target_arg.split(None, 1)
            try:
                n = int(log_parts[0])
                filter_str = log_parts[1].strip() if len(log_parts) > 1 else ""
            except ValueError:
                filter_str = target_arg
        return _read_violation_log(n, filter_str)

    # Session-specific subcommands require a target session ID
    target_id, session = _resolve_session(target_arg)
    if session is None:
        return f"ERROR: session '{target_arg}' not found. Use !session to list active sessions."

    shorthand = get_or_create_shorthand_id(target_id)

    # ------------------------------------------------------------------
    if sub == "unblock":
        was_blocked = session.pop("airs_blocked", False)
        block_report = session.pop("airs_block_report_id", None)
        if was_blocked:
            log.info(f"!airs unblock: [{shorthand}] {target_id} unblocked by {caller_client_id}")
            # Append resolution record to disk log
            import time as _time
            _append_log_record({
                "timestamp":  _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "source":     "unblock",
                "client_id":  target_id,
                "report_id":  block_report or "",
                "unblocked_by": caller_client_id,
            })
            return f"Session [{shorthand}] unblocked. AIRS block flag cleared."
        return f"Session [{shorthand}] was not blocked."

    # ------------------------------------------------------------------
    if sub == "status":
        blocked     = session.get("airs_blocked", False)
        report_id   = session.get("airs_block_report_id", "")
        sync_count  = len(session.get("airs_sync_violations", []))
        async_count = len(session.get("airs_async_violations", []))
        lines = [f"AIRS status for session [{shorthand}] {target_id}:"]
        lines.append(f"  Blocked:          {'YES' if blocked else 'no'}")
        if blocked and report_id:
            lines.append(f"  Block report ID:  {report_id}")
        lines.append(f"  Sync violations:  {sync_count}  (response was blocked before delivery)")
        lines.append(f"  Async violations: {async_count}  (response was already delivered)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # sub == "violations" (default) — show both sync and async
    sync_v  = session.get("airs_sync_violations", [])
    async_v = session.get("airs_async_violations", [])
    all_violations = [("sync", v) for v in sync_v] + [("async", v) for v in async_v]

    if not all_violations:
        return f"No AIRS violations recorded for session [{shorthand}] {target_id}."

    lines = [f"AIRS violations for session [{shorthand}] {target_id} ({len(all_violations)} total):"]
    for i, (source, v) in enumerate(all_violations, 1):
        pd = v.get("prompt_detected") or {}
        rd = v.get("response_detected") or {}
        delivered = "response already delivered" if source == "async" else "response was BLOCKED"
        lines.append(
            f"\n  [{i}] {v.get('timestamp', 'unknown time')}  [{source}]  ({delivered})"
            f"\n      Category:   {v.get('category', '?')}"
            f"\n      Report ID:  {v.get('report_id', '?')}"
            f"\n      Scan ID:    {v.get('scan_id', '?')}"
            f"\n      Prompt:     url_cats={pd.get('url_cats','?')}  "
            f"injection={pd.get('injection','?')}  dlp={pd.get('dlp','?')}"
            f"\n      Response:   url_cats={rd.get('url_cats','?')}  "
            f"dlp={rd.get('dlp','?')}"
        )
    return "\n".join(lines)


def _read_violation_log(n: int, filter_client: str = "") -> str:
    """
    Read the last N entries from airs-violations.log.
    If filter_client is set, only return entries whose client_id contains it.
    Survives agent restarts — reads from disk every time.
    """
    import os as _os
    log_path = _os.path.join(_os.path.dirname(__file__), "airs-violations.log")
    if not _os.path.exists(log_path):
        return f"Violation log not found: {log_path}"

    try:
        with open(log_path) as f:
            raw_lines = [l.strip() for l in f if l.strip()]
    except Exception as exc:
        return f"ERROR reading violation log: {exc}"

    if not raw_lines:
        return "Violation log is empty — no violations recorded yet."

    records = []
    for line in raw_lines:
        try:
            records.append(json.loads(line))
        except Exception:
            pass  # skip malformed lines

    if filter_client:
        records = [r for r in records if filter_client in r.get("client_id", "")]

    records = records[-n:]  # last N

    if not records:
        return f"No violations found matching '{filter_client}' in log."

    # Build set of report_ids that were unblocked (for status annotation)
    unblocked = {}  # report_id -> {timestamp, unblocked_by}
    for r in records:
        if r.get("source") == "unblock" and r.get("report_id"):
            unblocked[r["report_id"]] = {
                "timestamp":    r.get("timestamp", "?"),
                "unblocked_by": r.get("unblocked_by", "?"),
            }

    lines = [f"AIRS violation log — last {len(records)} entries"
             + (f" (filtered: {filter_client!r})" if filter_client else "")
             + f"  [{log_path}]:"]
    for i, v in enumerate(records, 1):
        source = v.get("source", "?")
        cid    = v.get("client_id", "")

        # Unblock records get their own format
        if source == "unblock":
            lines.append(
                f"\n  [{i}] {v.get('timestamp', '?')}  [UNBLOCKED]"
                f"\n      Session:      {cid or '(unknown)'}"
                f"\n      Report ID:    {v.get('report_id', '?')}"
                f"\n      Unblocked by: {v.get('unblocked_by', '?')}"
            )
            continue

        # Violation records
        pd = v.get("prompt_detected") or {}
        rd = v.get("response_detected") or {}
        rid = v.get("report_id", "")
        if source == "async":
            status = "delivered then flagged"
        elif rid in unblocked:
            ub = unblocked[rid]
            status = f"BLOCKED → unblocked {ub['timestamp']} by {ub['unblocked_by']}"
        else:
            status = "BLOCKED"

        lines.append(
            f"\n  [{i}] {v.get('timestamp', '?')}  [{source}]  ({status})"
            f"\n      Session:    {cid or '(unknown)'}"
            f"\n      Category:   {v.get('category', '?')}"
            f"\n      Report ID:  {rid or '?'}"
            f"\n      Scan ID:    {v.get('scan_id', '?')}"
            f"\n      Prompt:     url_cats={pd.get('url_cats','?')}  "
            f"injection={pd.get('injection','?')}  dlp={pd.get('dlp','?')}"
            f"\n      Response:   url_cats={rd.get('url_cats','?')}  "
            f"dlp={rd.get('dlp','?')}"
        )
    return "\n".join(lines)


# This needs to be reachable from _cmd_airs — import json at module level


_AIRS_HELP = (
    "  !airs                         — show last 10 violations from disk log (all sessions)\n"
    "  !airs log [N] [filter]        — last N entries from disk log (default N=10)\n"
    "                                   filter = session ID substring to narrow results\n"
    "  !airs status <ID>             — blocked flag + violation count for a session\n"
    "  !airs violations <ID>         — list in-memory violations for a session\n"
    "  !airs unblock <ID>            — clear session block flag (admin)\n"
    "  <ID> = shorthand (e.g. 101) or full session ID prefix\n"
)


# ---------------------------------------------------------------------------
# Register !airs at import time
# tools.py is always imported before history plugins load (via routes.py)
# ---------------------------------------------------------------------------

try:
    from tools import register_plugin_commands
    register_plugin_commands(
        "plugin_sec_airs_cmd",
        {"airs": _cmd_airs},
        _AIRS_HELP,
    )
except Exception as _e:
    log.warning(f"plugin_sec_airs_cmd: could not register !airs command: {_e}")
