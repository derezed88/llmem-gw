"""
SMS Proxy Plugin for llmem-gw

Server-side component that bridges incoming SMS (from macOS sms_proxy.py client)
to the notifier system and provides !sms commands for replying.

Architecture:
- POST /sms/inbound   — macOS proxy pushes incoming SMS here
- GET  /sms/outbound  — macOS proxy polls for reply messages to send
- POST /sms/ack       — macOS proxy acknowledges sent replies

Notification delivery (push, no polling):
- Routes to most recently active session matching notify_models patterns via push_notif
- Also fires "sms_received" notifier event for explicitly subscribed sessions (e.g. Slack)

Commands (via !sms):
  !sms                      — show recent messages and status
  !sms reply <phone> <msg>  — queue a reply to be sent via macOS proxy
  !sms history [count]      — show last N messages (default 10)
  !sms enable               — enable SMS relay (runtime)
  !sms disable              — disable SMS relay (runtime)
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import List, Dict, Optional
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from plugin_loader import BasePlugin
from config import log

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    """Load plugin_sms_proxy config from plugins-enabled.json."""
    try:
        with open("plugins-enabled.json") as f:
            cfg = json.load(f)
        return cfg.get("plugin_config", {}).get("plugin_sms_proxy", {})
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

# Incoming SMS log (ring buffer)
_MAX_INBOX = 50
_inbox: list[dict] = []  # [{phone, text, timestamp, id}, ...]
_inbox_counter = 0

# Outbound reply queue (macOS proxy polls this)
_outbound: list[dict] = []  # [{id, phone, text, timestamp}, ...]
_outbound_counter = 0

# SSE subscribers for outbound push (one asyncio.Queue per connected macOS proxy)
_outbound_subscribers: set[asyncio.Queue] = set()

# Track when the macOS proxy last contacted us (SSE connect = heartbeat)
_PROXY_TIMEOUT = 30  # seconds
_last_proxy_contact: float = 0.0

# Runtime enable/disable (overrides plugin_config)
_runtime_enabled: Optional[bool] = None


def _is_enabled() -> bool:
    """Check if SMS relay is enabled (runtime override > plugin_config)."""
    if _runtime_enabled is not None:
        return _runtime_enabled
    return bool(_load_plugin_config().get("relay_enabled", True))


def _get_notify_models() -> list[str]:
    """Return model name patterns for auto-notify (from plugin_config)."""
    return _load_plugin_config().get("notify_models", [])


async def _resolve_phone_name(phone: str) -> str:
    """Look up a phone number in the person table. Returns nickname, full_name, or the phone number."""
    import asyncio
    def _lookup():
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host="localhost",
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASS"),
                database="mymcp",
            )
            cur = conn.cursor()
            try:
                cur.execute("SELECT full_name, nickname FROM person WHERE phone = %s LIMIT 1", (phone,))
                row = cur.fetchone()
                if row:
                    return row[1] or row[0]  # nickname first, else full_name
            finally:
                cur.close()
                conn.close()
        except Exception as e:
            log.warning(f"SMS phone lookup failed for {phone}: {e}")
        return None
    result = await asyncio.to_thread(_lookup)
    return result or phone


async def _auto_notify_sessions(phone: str, text: str) -> int:
    """
    Push SMS notification to the most recently active frontend with an open
    SSE connection. Falls back through all sessions in last_active order until
    one succeeds.

    "Active" means the session already has an entry in sse_queues — i.e. it has
    an open SSE stream consuming from its queue. Sessions without an existing
    queue entry are skipped (push_notif would silently enqueue into the void).

    Returns 1 if a session was notified, 0 if no live session was found.
    """
    from state import sessions, push_notif, sse_queues

    sender = await _resolve_phone_name(phone)
    msg = f"SMS from {sender}. {text}"

    # Sort all sessions by last_active descending (most recently active first)
    sorted_sessions = sorted(
        sessions.items(),
        key=lambda kv: kv[1].get("last_active", 0.0),
        reverse=True,
    )

    for client_id, session in sorted_sessions:
        # Only push to sessions with an existing SSE queue (live connection).
        # get_queue() creates a new queue for any client_id — checking first
        # prevents silently dropping into a queue nobody is reading.
        if client_id not in sse_queues:
            continue
        try:
            await push_notif(client_id, msg)
            log.info(
                f"SMS auto-notify → {client_id} "
                f"model={session.get('model', '?')} "
                f"last_active={session.get('last_active', 0):.0f}"
            )
            return 1
        except Exception as e:
            log.warning(f"SMS auto-notify failed for {client_id}: {e}")
            continue

    log.info("SMS auto-notify: no live SSE session found")
    return 0


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

async def endpoint_sms_inbound(request: Request) -> JSONResponse:
    """
    Receive an incoming SMS from the macOS proxy.

    Body (JSON):
        phone : str   — sender phone number
        text  : str   — message body
    """
    global _inbox_counter

    if not _is_enabled():
        return JSONResponse({"status": "disabled"}, status_code=503)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    phone = payload.get("phone", "").strip()
    text = payload.get("text", "").strip()
    if not phone or not text:
        return JSONResponse({"error": "Missing phone or text"}, status_code=400)

    _inbox_counter += 1
    msg = {
        "id": _inbox_counter,
        "phone": phone,
        "text": text,
        "timestamp": time.time(),
        "ts_str": time.strftime("%H:%M:%S"),
    }
    _inbox.append(msg)
    if len(_inbox) > _MAX_INBOX:
        _inbox.pop(0)

    log.info(f"SMS inbound from {phone}: {text[:80]}")

    # Auto-notify matching model sessions (direct push, no registration needed)
    auto_count = await _auto_notify_sessions(phone, text)

    # Also fire notifier event for explicitly subscribed sessions
    try:
        import notifier
        sender = await _resolve_phone_name(phone)
        preview = text[:120] + ("..." if len(text) > 120 else "")
        await notifier.fire_event(
            "sms_received",
            f"SMS from {sender}",
            f"{preview}\n  Reply: !sms reply {phone} <your message>",
        )
    except Exception as e:
        log.warning(f"SMS notifier fire failed: {e}")

    return JSONResponse({"status": "ok", "id": _inbox_counter, "auto_notified": auto_count})


async def endpoint_sms_outbound(request: Request) -> JSONResponse:
    """
    Poll for outbound replies. macOS proxy calls this periodically.

    Returns list of pending reply messages, leaves them in queue until ACKed.
    """
    if not _is_enabled():
        return JSONResponse({"status": "disabled", "messages": []})

    return JSONResponse({
        "status": "ok",
        "messages": list(_outbound),
    })


async def endpoint_sms_outbound_stream(request: Request) -> StreamingResponse:
    """
    SSE stream for outbound replies. macOS proxy holds this connection open.
    Server pushes reply messages as they are queued; proxy sends and ACKs.
    Connection is always macOS → llmem-gw (NAT/firewall safe).
    """
    global _last_proxy_contact
    _last_proxy_contact = time.time()

    if not _is_enabled():
        return JSONResponse({"status": "disabled"}, status_code=503)

    q: asyncio.Queue = asyncio.Queue()
    _outbound_subscribers.add(q)
    log.info("SMS outbound SSE: proxy stream connected")

    async def event_stream():
        global _last_proxy_contact
        try:
            # Replay any currently unacked messages on reconnect
            for msg in list(_outbound):
                yield f"data: {json.dumps(msg)}\n\n"

            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                    _last_proxy_contact = time.time()
                except asyncio.TimeoutError:
                    # Heartbeat — keeps connection alive through proxies/firewalls
                    yield ": ping\n\n"
                    _last_proxy_contact = time.time()
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            _outbound_subscribers.discard(q)
            log.info("SMS outbound SSE: proxy stream disconnected")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def endpoint_sms_ack(request: Request) -> JSONResponse:
    """
    Acknowledge that outbound messages were sent.

    Body (JSON):
        ids : list[int]  — message IDs that were successfully sent
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    ack_ids = set(payload.get("ids", []))
    before = len(_outbound)
    _outbound[:] = [m for m in _outbound if m["id"] not in ack_ids]
    removed = before - len(_outbound)

    log.info(f"SMS outbound ACK: {removed} message(s) cleared")
    return JSONResponse({"status": "ok", "cleared": removed})


async def endpoint_sms_health(request: Request) -> JSONResponse:
    """Health check for the SMS proxy plugin."""
    proxy_connected = len(_outbound_subscribers) > 0
    return JSONResponse({
        "status": "ok",
        "enabled": _is_enabled(),
        "proxy_connected": proxy_connected,
        "inbox_count": len(_inbox),
        "outbound_pending": len(_outbound),
    })


# ---------------------------------------------------------------------------
# !sms command handler
# ---------------------------------------------------------------------------

async def cmd_sms(args: str) -> str:
    """Handle !sms commands. Handler signature: async (args: str) -> str."""
    global _runtime_enabled, _outbound_counter

    from state import current_client_id as _ccid
    client_id = _ccid.get("")

    parts = args.strip().split(None, 1) if args.strip() else []
    subcmd = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    if subcmd == "enable":
        _runtime_enabled = True
        return "SMS relay enabled (runtime override)"

    if subcmd == "disable":
        _runtime_enabled = False
        return "SMS relay disabled (runtime override)"

    if subcmd == "reply":
        if not _is_enabled():
            return "SMS relay is disabled. Use !sms enable first."
        # Parse: !sms reply +14155551234 Hello there
        reply_parts = rest.strip().split(None, 1)
        if len(reply_parts) < 2:
            return "Usage: !sms reply <phone> <message>"
        phone = reply_parts[0]
        message = reply_parts[1]

        _outbound_counter += 1
        msg_data = {
            "id": _outbound_counter,
            "phone": phone,
            "text": message,
            "timestamp": time.time(),
            "from_session": client_id,
        }
        _outbound.append(msg_data)
        for _q in list(_outbound_subscribers):
            _q.put_nowait(msg_data)
        log.info(f"SMS reply queued → {phone}: {message[:80]} (from {client_id})")
        return f"SMS reply queued → {phone} ({len(message)} chars)"

    if subcmd == "history":
        count = 10
        if rest.strip().isdigit():
            count = int(rest.strip())
        return _format_inbox(count)

    # Default: show status + recent messages
    status = "enabled" if _is_enabled() else "DISABLED"
    lines = [
        f"SMS Proxy ({status})",
        f"  Inbox: {len(_inbox)} messages, Outbound pending: {len(_outbound)}",
        "",
    ]
    if _inbox:
        lines.append("Recent messages:")
        lines.append(_format_inbox(5))
    else:
        lines.append("No messages received yet.")
    lines.append("")
    lines.append("Commands:")
    lines.append("  !sms reply <phone> <msg>  — send a reply")
    lines.append("  !sms history [N]          — show last N messages")
    lines.append("  !sms enable / disable     — toggle relay")
    return "\n".join(lines)


def _format_inbox(count: int) -> str:
    """Format recent inbox messages."""
    recent = _inbox[-count:]
    if not recent:
        return "  (no messages)"
    lines = []
    for m in reversed(recent):
        age = int(time.time() - m["timestamp"])
        if age < 60:
            age_str = f"{age}s ago"
        elif age < 3600:
            age_str = f"{age // 60}m ago"
        else:
            age_str = f"{age // 3600}h ago"
        preview = m["text"][:80] + ("..." if len(m["text"]) > 80 else "")
        lines.append(f"  [{m['ts_str']}] {m['phone']}: {preview}  ({age_str})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class SmsProxyPlugin(BasePlugin):
    """SMS Proxy server-side plugin."""

    PLUGIN_NAME = "plugin_sms_proxy"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "client_interface"
    DESCRIPTION = "SMS relay via macOS Messages proxy — notifier integration and !sms commands"
    DEPENDENCIES = []
    ENV_VARS = []

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize SMS proxy plugin."""
        self.enabled = True
        log.info("SMS proxy plugin initialized")
        log.info(f"  relay_enabled: {_is_enabled()}")
        return True

    def shutdown(self) -> None:
        """Cleanup."""
        self.enabled = False
        log.info("SMS proxy plugin shutdown")

    def get_config(self) -> dict:
        """No dedicated port — routes are mounted on the shared Starlette app."""
        return {"port": None, "name": "SMS Proxy"}

    def get_routes(self) -> List[Route]:
        """Return HTTP routes for SMS proxy."""
        return [
            Route("/sms/health", endpoint_sms_health, methods=["GET"]),
            Route("/sms/inbound", endpoint_sms_inbound, methods=["POST"]),
            Route("/sms/outbound/stream", endpoint_sms_outbound_stream, methods=["GET"]),
            Route("/sms/outbound", endpoint_sms_outbound, methods=["GET"]),
            Route("/sms/ack", endpoint_sms_ack, methods=["POST"]),
        ]

    def get_commands(self) -> Dict[str, any]:
        """Return !sms command handler."""
        return {"sms": cmd_sms}

    def get_help(self) -> str:
        return (
            "\n--- SMS Proxy ---\n"
            "  !sms                      — status and recent messages\n"
            "  !sms reply <phone> <msg>  — send SMS reply via macOS proxy\n"
            "  !sms history [N]          — show last N messages\n"
            "  !sms enable / disable     — toggle SMS relay\n"
        )
