"""
API Client Interface Plugin for MCP Agent

Exposes a JSON/SSE HTTP API on port 8767 for programmatic access
and agent-to-agent (swarm) communication.

Endpoints:
- POST /api/v1/submit        - Submit message or command
- GET  /api/v1/stream/{id}   - SSE stream of JSON events
- GET  /api/v1/sessions      - List active sessions
- GET  /api/v1/health        - Health check
- DELETE /api/v1/session/{sid} - Delete a session

SSE event types (all data is JSON):
  event: tok        data: {"text": "..."}
  event: done       data: {}
  event: error      data: {"message": "..."}
  event: keepalive  (comment, no data)
"""

import os
import asyncio
import json
import uuid

from typing import List, AsyncGenerator
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from plugin_loader import BasePlugin
from state import get_queue, push_done, sessions, sse_queues, remove_shorthand_mapping, get_or_create_shorthand_id, drain_queue
from routes import process_request, cancel_active_task
from state import active_tasks

# Optional API key — if set, all /api/v1/* requests must carry Authorization: Bearer <key>
_API_KEY: str = os.getenv("API_KEY", "")


def _check_auth(request: Request) -> bool:
    """Return True if auth passes (or no API_KEY configured)."""
    if not _API_KEY:
        return True
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {_API_KEY}"


def _auth_error() -> JSONResponse:
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


async def endpoint_api_submit(request: Request) -> JSONResponse:
    """
    Submit a message or command.

    Body (JSON):
        client_id  : str  (optional; auto-generated api-{8hex} if omitted)
        text       : str  (required)
        wait       : bool (optional, default false — sync mode waits for completion)
        timeout    : int  (optional, default 60 — max wait in seconds for sync mode)

    Response:
        {"client_id": "api-...", "status": "accepted"}
        or for wait=true: {"client_id": "...", "status": "complete", "text": "..."}
    """
    if not _check_auth(request):
        return _auth_error()

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Missing 'text' field"}, status_code=400)

    client_id = payload.get("client_id", "").strip()
    if not client_id:
        client_id = f"api-{uuid.uuid4().hex[:8]}"

    wait = bool(payload.get("wait", False))
    timeout = int(payload.get("timeout", 60))
    peer_ip = request.client.host if request.client else None

    await cancel_active_task(client_id)
    await drain_queue(client_id)

    if wait:
        # Sync mode: accumulate all tokens until done, then return
        q = await get_queue(client_id)
        task = asyncio.create_task(process_request(client_id, text, payload, peer_ip=peer_ip))
        active_tasks[client_id] = task

        accumulated = []
        try:
            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    task.cancel()
                    return JSONResponse({
                        "client_id": client_id,
                        "status": "timeout",
                        "text": "".join(accumulated),
                    })
                try:
                    item = await asyncio.wait_for(q.get(), timeout=min(remaining, 30.0))
                except asyncio.TimeoutError:
                    # Keepalive interval — check deadline
                    continue

                t = item.get("t")
                if t == "tok":
                    # push_tok encodes newlines as \n literal; restore them
                    accumulated.append(item["d"].replace("\\n", "\n"))
                elif t == "err":
                    return JSONResponse({
                        "client_id": client_id,
                        "status": "error",
                        "text": item["d"].replace("\\n", "\n"),
                    })
                elif t == "done":
                    return JSONResponse({
                        "client_id": client_id,
                        "status": "complete",
                        "text": "".join(accumulated),
                    })
        except asyncio.CancelledError:
            return JSONResponse({"client_id": client_id, "status": "cancelled", "text": "".join(accumulated)})
    else:
        # Async mode: fire-and-forget, client streams via /api/v1/stream/{client_id}
        task = asyncio.create_task(process_request(client_id, text, payload, peer_ip=peer_ip))
        active_tasks[client_id] = task
        return JSONResponse({"client_id": client_id, "status": "accepted"})


async def endpoint_api_stream(request: Request):
    """
    SSE stream of JSON events for a client_id.

    Connect before or after submitting — the queue persists.
    Events:
        event: tok      data: {"text": "..."}
        event: gate     data: {"gate_id": "...", "tool_name": "...", ...}
        event: done     data: {}
        event: error    data: {"message": "..."}
        keepalive comment every 25s when idle
    """
    if not _check_auth(request):
        return _auth_error()

    client_id = request.path_params.get("client_id", "")
    if not client_id:
        return JSONResponse({"error": "Missing client_id"}, status_code=400)

    q = await get_queue(client_id)

    async def generator() -> AsyncGenerator[dict, None]:
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(q.get(), timeout=25.0)
            except asyncio.TimeoutError:
                yield {"comment": "keepalive"}
                continue

            t = item.get("t")
            if t == "tok":
                yield {"event": "tok", "data": json.dumps({"text": item["d"].replace("\\n", "\n")})}
            elif t == "done":
                yield {"event": "done", "data": "{}"}
            elif t == "flush":
                # Intermediate flush: shell.py uses this to display tool results mid-turn.
                # api_client ignores it (only stops on "done") so the stream stays open.
                yield {"event": "flush", "data": "{}"}
            elif t == "progress":
                yield {"event": "progress", "data": json.dumps({"text": item["d"].replace("\\n", "\n")})}
            elif t == "err":
                yield {"event": "error", "data": json.dumps({"message": item["d"].replace("\\n", "\n")})}

    return EventSourceResponse(generator())



async def endpoint_api_sessions(request: Request) -> JSONResponse:
    """List all active sessions with metadata."""
    if not _check_auth(request):
        return _auth_error()

    session_list = []
    for cid, data in sessions.items():
        shorthand = get_or_create_shorthand_id(cid)
        session_list.append({
            "client_id": cid,
            "shorthand_id": shorthand,
            "model": data.get("model", "unknown"),
            "history_length": len(data.get("history", [])),
        })
    return JSONResponse({"sessions": session_list})


async def endpoint_api_health(request: Request) -> JSONResponse:
    """Health check."""
    if not _check_auth(request):
        return _auth_error()

    from config import LLM_REGISTRY
    return JSONResponse({
        "status": "ok",
        "sessions": len(sessions),
        "models": list(LLM_REGISTRY.keys()),
    })


async def endpoint_api_delete_session(request: Request) -> JSONResponse:
    """Delete a session by full client_id or shorthand integer ID."""
    if not _check_auth(request):
        return _auth_error()

    sid = request.path_params.get("sid", "")
    if not sid:
        return JSONResponse({"error": "Missing session ID"}, status_code=400)

    # Resolve shorthand integer ID
    from state import get_session_by_shorthand
    target_id = sid
    try:
        shorthand = int(sid)
        resolved = get_session_by_shorthand(shorthand)
        if resolved:
            target_id = resolved
    except ValueError:
        pass  # Not an integer — use as-is

    if target_id not in sessions:
        return JSONResponse({"error": f"Session '{sid}' not found"}, status_code=404)

    remove_shorthand_mapping(target_id)
    sessions.pop(target_id, None)
    if target_id in sse_queues:
        sse_queues.pop(target_id, None)

    return JSONResponse({"status": "deleted", "client_id": target_id})


async def endpoint_api_stop(request: Request) -> JSONResponse:
    """Cancel the active LLM job for a client without starting a new one."""
    if not _check_auth(request):
        return _auth_error()
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    client_id = payload.get("client_id", "").strip()
    if not client_id:
        return JSONResponse({"error": "Missing 'client_id'"}, status_code=400)
    cancelled = await cancel_active_task(client_id)
    if cancelled:
        await push_done(client_id)
    return JSONResponse({"status": "OK", "cancelled": cancelled})


class ApiClientPlugin(BasePlugin):
    """Programmatic API client interface plugin."""

    PLUGIN_NAME = "plugin_client_api"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "client_interface"
    DESCRIPTION = "Programmatic HTTP/SSE API for agent access and swarm coordination"
    DEPENDENCIES = ["sse-starlette"]
    ENV_VARS = []

    def __init__(self):
        self.enabled = False
        self.api_port = 8766
        self.api_host = "0.0.0.0"

    def init(self, config: dict) -> bool:
        try:
            self.api_port = config.get("api_port", int(os.getenv("API_PORT", 8767)))
            self.api_host = config.get("api_host", "0.0.0.0")
            import sse_starlette  # noqa: F401 — verify dep
            self.enabled = True
            return True
        except Exception as e:
            print(f"API client plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_routes(self) -> List[Route]:
        return [
            Route("/api/v1/submit", endpoint_api_submit, methods=["POST"]),
            Route("/api/v1/stream/{client_id}", endpoint_api_stream, methods=["GET"]),
            Route("/api/v1/stop", endpoint_api_stop, methods=["POST"]),
            Route("/api/v1/sessions", endpoint_api_sessions, methods=["GET"]),
            Route("/api/v1/health", endpoint_api_health, methods=["GET"]),
            Route("/api/v1/session/{sid}", endpoint_api_delete_session, methods=["DELETE"]),
        ]

    def get_config(self) -> dict:
        return {
            "port": self.api_port,
            "host": self.api_host,
            "name": "MCP API service",
        }
