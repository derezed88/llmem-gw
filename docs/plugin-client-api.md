# plugin_client_api — API Client Interface

Exposes a JSON/SSE HTTP API for programmatic access and agent-to-agent (swarm) communication. This is the transport layer used internally by the `agent_call` tool.

**Default port:** 8767
**Plugin type:** `client_interface`
**Default enabled:** yes

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/submit` | Submit a message or command |
| `GET` | `/api/v1/stream/{client_id}` | SSE stream of JSON events |
| `POST` | `/api/v1/gate/{gate_id}` | Respond to a gate request |
| `GET` | `/api/v1/sessions` | List active sessions |
| `GET` | `/api/v1/health` | Health check |
| `DELETE` | `/api/v1/session/{sid}` | Delete a session |

---

## SSE Event Types

All event data is JSON:

| Event | Data |
|---|---|
| `tok` | `{"text": "..."}` — response token |
| `gate` | `{"gate_id": "...", "tool_name": "...", "tool_args": {...}, "tables": [...]}` |
| `done` | `{}` — response complete |
| `error` | `{"message": "..."}` |
| `keepalive` | SSE comment (no data) — sent every 25s when idle |

---

## Submit Request

```bash
curl -X POST http://localhost:8767/api/v1/submit \
  -H "Content-Type: application/json" \
  -d '{"client_id": "my-client", "text": "!model"}'
```

**Request body:**

| Field | Required | Default | Description |
|---|---|---|---|
| `text` | yes | — | Message or `!command` to process |
| `client_id` | no | `api-{8 hex chars}` | Session identifier — omit to auto-generate |
| `wait` | no | `false` | If `true`, block until done and return full response |
| `timeout` | no | `30` | Max seconds to wait when `wait=true` |

**Response:**
```json
{"client_id": "my-client", "status": "accepted"}
```

---

## Streaming Example

```bash
# Subscribe first (keep connection open)
curl -N http://localhost:8767/api/v1/stream/my-client &

# Then submit
curl -X POST http://localhost:8767/api/v1/submit \
  -H "Content-Type: application/json" \
  -d '{"client_id": "my-client", "text": "hello"}'
```

---

## Gate Handling

When the LLM makes a tool call that requires human approval, the server:

1. Pushes a `gate` event to the client's SSE stream
2. Waits `API_GATE_TIMEOUT` seconds (default 2s) for the client to respond
3. Auto-rejects if no response arrives

To approve:
```bash
curl -X POST http://localhost:8767/api/v1/gate/{gate_id} \
  -H "Content-Type: application/json" \
  -d '{"approve": true}'
```

`AgentClient` with `auto_approve_gates` configured responds within milliseconds automatically.

---

## Authentication

Set `API_KEY` in `.env` to require Bearer token auth on all endpoints:

```
API_KEY=your-secret-key
```

```bash
curl -H "Authorization: Bearer your-secret-key" http://localhost:8767/api/v1/health
```

If `API_KEY` is not set, no authentication is required (suitable for local use).

---

## Python Client: `api_client.py`

`api_client.py` is a standalone async client library for this plugin.

### Basic usage

```python
import asyncio
from api_client import AgentClient

async def main():
    client = AgentClient("http://localhost:8767")

    # One-shot: submit and wait for complete response
    result = await client.send("What time is it?")
    print(result)

    # Streaming: yield tokens as they arrive
    async for token in client.stream("Summarise the person table"):
        print(token, end="", flush=True)

asyncio.run(main())
```

### Gate policy

```python
# Reject all gates (default — safe for automated use)
client = AgentClient("http://localhost:8767")

# Approve all gates (use with care)
client = AgentClient("http://localhost:8767", auto_approve_gates=True)

# Approve by tool name
client = AgentClient(
    "http://localhost:8767",
    auto_approve_gates={"db_query": True, "google_drive": False}
)

# Approve by operation type
client = AgentClient(
    "http://localhost:8767",
    auto_approve_gates={"read": True, "write": False}
)
```

### Session management

```python
sessions = await client.sessions()          # list all sessions
await client.delete_session("api-abc12345") # delete by client_id or shorthand
health = await client.health()              # {"status": "ok", ...}
```

---

## Port Configuration

```bash
python llmemctl.py port-list
python llmemctl.py port-set plugin_client_api 8777
```

---

## Swarm / Agent-to-Agent

The `agent_call` tool (available to all LLMs) uses this plugin as its transport. See [ARCHITECTURE.md](ARCHITECTURE.md#swarm-architecture) for the full flow.

Key properties of swarm sessions:

- **client_id format:** `api-swarm-{md5(calling_session:agent_url)[:8]}` — stable across repeated calls
- **Gate handling:** auto-reject (non-interactive, same as llama proxy)
- **Depth guard:** swarm sessions cannot initiate further `agent_call` calls (1-hop limit)
- **Discovery:** no automatic discovery — targets must be specified by URL

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | no | Bearer token for auth. If unset, all requests are accepted. |
