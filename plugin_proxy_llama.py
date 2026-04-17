"""
Llama Proxy Client Interface Plugin for MCP Agent

Provides OpenAI/Ollama compatible API endpoints for various chat clients:
- /v1/chat/completions - OpenAI format
- /api/generate, /api/chat - Ollama format
- /v1/models, /api/tags - Model listing
- Supports open-webui, Enchanted, and other compatible clients
"""

import json
import asyncio
import time
from typing import List, Optional, Dict, Any
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from plugin_loader import BasePlugin
from config import log, DEFAULT_MODEL
from state import sessions, sse_queues, queue_lock
from routes import process_request


class LlamaProxyConfig:
    """Configuration for llama proxy mode"""
    def __init__(self):
        self.enabled = False

llama_config = LlamaProxyConfig()


def _detect_client_type(request: Request, path: str) -> str:
    """
    Detect the client type based on User-Agent header and request path.
    Returns: 'ollama' or 'openai'
    """
    user_agent = request.headers.get('user-agent', '').lower()

    if 'ollama' in user_agent:
        return 'ollama'
    elif 'enchanted' in user_agent:
        return 'ollama'
    elif 'lm studio' in user_agent or 'lmstudio' in user_agent:
        return 'lmstudio'
    elif 'open-webui' in user_agent or 'openwebui' in user_agent:
        return 'openai'

    if path.startswith('api/') or 'api/' in path:
        return 'ollama'
    elif path.startswith('v1/'):
        return 'openai'

    return 'openai'


def _extract_prompt_from_request(request_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract prompt text from various request formats.
    Handles both /api/generate (prompt field) and /v1/chat/completions (messages array)
    """
    if 'prompt' in request_data:
        return request_data['prompt']

    if 'messages' in request_data and isinstance(request_data['messages'], list):
        messages = request_data['messages']
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                return item.get('text', '')

    return None


async def handle_llama_request_streaming(
    request_data: Dict[str, Any],
    client_id: str,
    path: str,
    client_type: str = 'openai'
) -> StreamingResponse:
    """Handle streaming requests from llama proxy clients."""
    prompt = _extract_prompt_from_request(request_data)

    if not prompt:
        return StreamingResponse(
            iter([json.dumps({"error": "No prompt found"})]),
            media_type='application/json',
            status_code=400
        )

    log.info(f"Llama proxy streaming request from {client_id}: {prompt[:100]}")

    if client_id not in sessions:
        sessions[client_id] = {
            "model": DEFAULT_MODEL,
            "history": []
        }
        log.info(f"[{client_id}] Created new session with model: {DEFAULT_MODEL}")

    response_queue = asyncio.Queue()

    async with queue_lock:
        old_queue = sse_queues.get(client_id)
        sse_queues[client_id] = response_queue

    asyncio.create_task(process_request(client_id, prompt, {"default_model": sessions[client_id]["model"]}))
    log.debug(f"[{client_id}] process_request task started for streaming")

    async def stream_generator():
        try:
            max_wait_time = 120.0
            start_time = asyncio.get_event_loop().time()
            current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            first_chunk_sent = False

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > max_wait_time:
                    log.warning(f"[{client_id}] Stream timeout after {elapsed:.1f}s")
                    break

                try:
                    item = await asyncio.wait_for(response_queue.get(), timeout=5.0)
                    t = item.get("t")

                    if t == "tok":
                        text = item["d"].replace("\\n", "\n")
                        chunk_size = 10
                        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] if len(text) > chunk_size else [text]

                        for text_piece in text_chunks:
                            if client_type == 'ollama':
                                chunk = {
                                    "model": request_data.get("model", "mcp"),
                                    "created_at": current_timestamp,
                                    "response": text_piece,
                                    "done": False
                                }
                                yield f"{json.dumps(chunk)}\n"
                            else:
                                if not first_chunk_sent:
                                    delta_content = {"role": "assistant", "content": text_piece}
                                    first_chunk_sent = True
                                else:
                                    delta_content = {"content": text_piece}

                                chunk = {
                                    "id": "chatcmpl-mcp",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request_data.get("model", "mcp"),
                                    "choices": [{
                                        "index": 0,
                                        "delta": delta_content,
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"

                            await asyncio.sleep(0.02)

                    elif t == "done":
                        log.info(f"[{client_id}] Stream complete")

                        if client_type == 'ollama':
                            done_chunk = {
                                "model": request_data.get("model", "mcp"),
                                "created_at": current_timestamp,
                                "response": "",
                                "done": True,
                                "done_reason": "stop"
                            }
                            yield f"{json.dumps(done_chunk)}\n"
                        else:
                            done_chunk = {
                                "id": "chatcmpl-mcp",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request_data.get("model", "mcp"),
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(done_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                        break

                    elif t == "err":
                        error_msg = item['d']
                        log.error(f"[{client_id}] Error: {error_msg}")

                        if client_type == 'ollama':
                            yield json.dumps({"error": error_msg}) + "\n"
                        else:
                            yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        break

                    elif t == "gate":
                        gate_data = item.get("d", {})
                        tool_name = gate_data.get("tool_name", "unknown")
                        gate_msg = (f"[Human approval required for {tool_name}. "
                                   f"Use shell.py client to approve/reject this request.]")

                        if client_type == 'ollama':
                            chunk = {
                                "model": request_data.get("model", "mcp"),
                                "created_at": current_timestamp,
                                "response": gate_msg,
                                "done": True,
                                "done_reason": "stop"
                            }
                            yield f"{json.dumps(chunk)}\n"
                        else:
                            chunk = {
                                "id": "chatcmpl-mcp",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request_data.get("model", "mcp"),
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": gate_msg},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                        break

                except asyncio.TimeoutError:
                    log.debug(f"[{client_id}] Stream keepalive (no data for 5s)")
                    continue

        finally:
            async with queue_lock:
                if old_queue is not None:
                    sse_queues[client_id] = old_queue
                else:
                    sse_queues.pop(client_id, None)

    media_type = 'application/x-ndjson' if client_type == 'ollama' else 'text/event-stream'
    connection_header = 'keep-alive' if client_type == 'ollama' else 'close'

    return StreamingResponse(
        stream_generator(),
        media_type=media_type,
        headers={
            'Cache-Control': 'no-cache',
            'Connection': connection_header,
            'X-Accel-Buffering': 'no'
        }
    )


async def handle_llama_request_non_streaming(
    request_data: Dict[str, Any],
    client_id: str,
    path: str,
    client_type: str
) -> Response:
    """Handle non-streaming requests from llama proxy clients."""
    prompt = _extract_prompt_from_request(request_data)

    if not prompt:
        return Response(
            content=json.dumps({"error": "No prompt found"}),
            media_type='application/json',
            status_code=400
        )

    log.info(f"Llama proxy non-streaming request from {client_id}: {prompt[:100]}")

    if client_id not in sessions:
        sessions[client_id] = {
            "model": DEFAULT_MODEL,
            "history": []
        }
        log.info(f"[{client_id}] Created new session with model: {DEFAULT_MODEL}")

    response_queue = asyncio.Queue()

    async with queue_lock:
        old_queue = sse_queues.get(client_id)
        sse_queues[client_id] = response_queue

    asyncio.create_task(process_request(client_id, prompt, {"default_model": sessions[client_id]["model"]}))
    log.debug(f"[{client_id}] process_request task started for non-streaming")

    full_response = ""
    max_wait_time = 120.0
    start_time = asyncio.get_event_loop().time()

    try:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait_time:
                log.warning(f"[{client_id}] Non-streaming timeout after {elapsed:.1f}s")
                break

            try:
                item = await asyncio.wait_for(response_queue.get(), timeout=5.0)
                t = item.get("t")

                if t == "tok":
                    full_response += item["d"].replace("\\n", "\n")

                elif t == "done":
                    log.info(f"[{client_id}] Non-streaming complete")
                    break

                elif t == "err":
                    error_msg = item['d']
                    log.error(f"[{client_id}] Error: {error_msg}")
                    return Response(
                        content=json.dumps({"error": error_msg}),
                        media_type='application/json',
                        status_code=500
                    )

                elif t == "gate":
                    gate_data = item.get("d", {})
                    tool_name = gate_data.get("tool_name", "unknown")
                    full_response += f"\n[Human approval required for {tool_name}. Use shell.py client to approve/reject.]\n"

            except asyncio.TimeoutError:
                log.debug(f"[{client_id}] Waiting for response data...")
                continue

    finally:
        async with queue_lock:
            if old_queue is not None:
                sse_queues[client_id] = old_queue
            else:
                sse_queues.pop(client_id, None)

    if client_type == 'ollama' or 'api/' in path:
        response_data = {
            "model": request_data.get("model", "mcp"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "response": full_response,
            "done": True,
            "done_reason": "stop"
        }
    else:
        response_data = {
            "id": "chatcmpl-mcp",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mcp"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    return Response(
        content=json.dumps(response_data),
        media_type='application/json',
        status_code=200,
        headers={
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=60, max=100'
        }
    )


OAUTH_DISCOVERY_PATHS = frozenset((
    '.well-known/oauth-authorization-server',
    '.well-known/oauth-authorization-server/mcp',
    'mcp/.well-known/oauth-authorization-server',
    '.well-known/oauth-protected-resource',
    '.well-known/oauth-protected-resource/mcp',
    'mcp/.well-known/oauth-protected-resource',
))


async def llama_proxy_handler(request: Request) -> Response:
    """
    Main llama proxy handler - processes all requests through MCP service.
    Handles both model listing and normal LLM requests.
    """
    path = request.path_params.get('path', '')
    method = request.method
    client_ip = request.client.host if request.client else "unknown"

    # MCP clients probe OAuth discovery well-known paths per the authorization spec
    # (2025-03-26 / 2025-06-18) before opening a session. This server doesn't require
    # OAuth — a 404 tells the client to proceed unauthenticated. Short-circuit here so
    # the probes don't generate INFO/WARNING log noise on every MCP connection.
    if path in OAUTH_DISCOVERY_PATHS:
        log.debug(f"[LLAMA] OAuth discovery probe: {method} /{path} → 404 (no auth configured)")
        return Response(
            content=json.dumps({"error": "no authorization server configured"}),
            media_type='application/json',
            status_code=404
        )

    log.info(f"[LLAMA] {method} /{path} from {client_ip}")

    if not llama_config.enabled:
        log.error(f"[LLAMA] Proxy not enabled, rejecting request")
        return Response(
            content=json.dumps({"error": "Llama proxy not enabled"}),
            media_type='application/json',
            status_code=503
        )

    try:
        body = await request.body()
        log.info(f"[LLAMA] Body length: {len(body) if body else 0} bytes")

        if path in ['api/generate', 'api/chat', 'v1/chat/completions', 'chat/completions', 'v1/api/generate', 'v1/api/chat'] and method == 'POST':
            try:
                request_data = json.loads(body) if body else {}
                client_type = _detect_client_type(request, path)
                log.info(f"[LLAMA] {path} - client: {client_type}, model: {request_data.get('model', 'none')}, stream: {request_data.get('stream', False)}")

                client_id = f"llama-{request.client.host}"
                is_streaming = request_data.get('stream', True)

                if is_streaming:
                    return await handle_llama_request_streaming(request_data, client_id, path, client_type)
                else:
                    return await handle_llama_request_non_streaming(request_data, client_id, path, client_type)

            except json.JSONDecodeError as e:
                log.error(f"Failed to parse request body as JSON: {e}")
                return Response(
                    content=json.dumps({"error": "Invalid JSON"}),
                    media_type='application/json',
                    status_code=400
                )
            except Exception as e:
                log.error(f"Error in llama endpoint: {e}", exc_info=True)
                return Response(
                    content=json.dumps({"error": str(e)}),
                    media_type='application/json',
                    status_code=500
                )

        else:
            if path in ('v1/', 'v1', ''):
                return Response(
                    content=json.dumps({"status": "ok", "version": "0.1.0-mcp"}),
                    media_type='application/json',
                    status_code=200
                )

            elif path in ('v1/models', 'models'):
                from config import LLM_REGISTRY
                current_time = int(time.time())
                models_data = [
                    {"id": key, "object": "model", "created": current_time, "owned_by": "mcp-server"}
                    for key in LLM_REGISTRY.keys()
                ]
                log.info(f"[LLAMA] Returning {len(models_data)} models in OpenAI format")
                return Response(
                    content=json.dumps({"object": "list", "data": models_data}),
                    media_type='application/json',
                    status_code=200
                )

            elif path in ('v1/api/tags', 'api/tags'):
                from config import LLM_REGISTRY
                models = [{"name": key} for key in LLM_REGISTRY.keys()]
                log.info(f"[LLAMA] Returning {len(models)} models in Ollama format")
                return Response(
                    content=json.dumps({"models": models}),
                    media_type='application/json',
                    status_code=200
                )

            elif path == 'api/version':
                return Response(
                    content=json.dumps({"version": "0.1.0-mcp"}),
                    media_type='application/json',
                    status_code=200
                )

            elif path.startswith('v1/models/') or path.startswith('models/'):
                from config import LLM_REGISTRY
                model_id = path.replace('v1/models/', '').replace('models/', '')
                if model_id in LLM_REGISTRY:
                    return Response(
                        content=json.dumps({"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "mcp-server"}),
                        media_type='application/json',
                        status_code=200
                    )
                else:
                    return Response(
                        content=json.dumps({"error": {"message": f"Model '{model_id}' not found", "type": "invalid_request_error", "param": None, "code": "model_not_found"}}),
                        media_type='application/json',
                        status_code=404
                    )

            else:
                log.warning(f"[LLAMA] Unsupported endpoint: {path}")
                return Response(
                    content=json.dumps({"error": f"Endpoint not supported: {path}"}),
                    media_type='application/json',
                    status_code=404
                )

    except Exception as e:
        log.error(f"Error handling request: {e}", exc_info=True)
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type='application/json'
        )


class LlamaProxyPlugin(BasePlugin):
    """Llama proxy client interface plugin."""

    PLUGIN_NAME = "plugin_proxy_llama"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "client_interface"
    DESCRIPTION = "Llama proxy for OpenAI/Ollama compatible clients (open-webui, Enchanted, etc.)"
    DEPENDENCIES = []
    ENV_VARS = []

    def __init__(self):
        self.enabled = False
        self.llama_port = 11434
        self.llama_host = "0.0.0.0"

    def init(self, config: dict) -> bool:
        """Initialize llama proxy plugin."""
        try:
            self.llama_port = config.get('llama_port', 11434)
            self.llama_host = config.get('llama_host', '0.0.0.0')
            llama_config.enabled = True
            self.enabled = True
            return True
        except Exception as e:
            print(f"Llama proxy plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup llama proxy resources."""
        llama_config.enabled = False
        self.enabled = False

    def get_routes(self) -> List[Route]:
        """Return Starlette routes for llama proxy."""
        return [
            Route("/{path:path}", llama_proxy_handler, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
        ]

    def get_config(self) -> dict:
        """Return plugin configuration for server startup."""
        return {
            "port": self.llama_port,
            "host": self.llama_host,
            "name": "Llama proxy"
        }
