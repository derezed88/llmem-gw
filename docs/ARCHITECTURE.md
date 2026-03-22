# Architecture

## Overview

The MCP Agent is a multi-client AI agent server. It maintains persistent sessions with conversation history, routes all client requests through a central LLM dispatch loop, controls tool access via per-model toolsets, and exposes a modular plugin system for data tools and client interfaces.

```
Clients                 Server                          Backends
───────                 ──────                          ────────
shell.py     ──SSE──►  llmem-gw.py                    OpenAI API (gpt, local)
open-webui    ─HTTP─►  ┌──────────────────────┐        xAI API (grok) + Responses API
LM Studio app ─HTTP─►  │  routes.py           │        Gemini API
                       │                      │        Local llama.cpp / Ollama
Slack  ─Socket Mode──►  │  (process_request)   │
       ◄─Web API(bot)─  │                      │
                       │    │                 │
                       │    ▼                 │
                       │  agents.py           │
                       │  (dispatch_llm)      │
                       │  LangChain agentic   │
                       │  loop               │
                       │    │                 │
                       │    ▼                 │
                       │  execute_tool()      │
                       │    │                 │
                       │    ▼                 │
                       │  Plugin executors    │──► MySQL
                       │  (db, drive, search) │──► Google Drive
                       └──────────────────────┘──► Web search APIs
```

## Source Files

| File | Responsibility |
|---|---|
| `llmem-gw.py` | Entry point. Initialises plugins, builds Starlette app, starts uvicorn |
| `config.py` | LLM registry (loaded from `llm-models.json`), environment loading, rate limit config |
| `state.py` | In-memory session store, SSE queues, context vars |
| `routes.py` | HTTP endpoints, `!command` routing, `@model` switching, `process_request()` |
| `agents.py` | LangChain-based LLM dispatch loop (`agentic_lc`), `execute_tool()`, rate limiter |
| `tools.py` | Core tool definitions as LangChain `StructuredTool` objects; plugin tool registry; per-model toolset filtering |
| `prompt.py` | Recursive system prompt loader, section tree, `apply_prompt_operation()` |
| `plugin_loader.py` | `BasePlugin` ABC, dynamic plugin loading from manifest |
| `agents_xai.py` | Responses API dispatch for xAI and OpenAI models; `agentic_responses_api()` |
| `database.py` | MySQL connection and SQL execution helpers |

## Request Flow

```
Client sends message
       │
       ▼
process_request()  ──── stripped.startswith("!") ──► cmd_* handler ──► push_tok ──► done
       │
       ├── multi-command batch? ──► process each !cmd sequentially
       │
       ├── stripped.startswith("@") ──► @model temp switch
       │       sets session["model"] + session["_temp_model_active"]
       │
       ▼
session["history"].append(user message)
       │
       ▼
dispatch_llm(model, history, client_id)
       │
       └── agentic_lc()  ← single loop for all model types (OpenAI + Gemini)
               │
               │  _build_lc_llm(model_key)  ← ChatOpenAI or ChatGoogleGenerativeAI
               │  llm.bind_tools(_CURRENT_LC_TOOLS)
               │
               ▼ (loop, max MAX_TOOL_ITERATIONS)
        LLM API call via LangChain ainvoke()
               │
        tool_calls present?  ── No ──► try_force_tool_calls() (bare text fallback)
               │                             │
               │                      forced calls? ──► execute_tool() ──► inject as HumanMessage
               │
               ▼ (Yes — native tool calls)
        execute_tool(tool_name, tool_args, client_id)
               │
               ├── check_rate_limit()
               │
               └── executor(**tool_args) ──► result ──► ToolMessage ──► back to LLM context
               │
        LLM produces final text response
               │
       ▼
session["history"].append(assistant message)
push_tok(response text) ──► SSE queue ──► client
```

## LangChain Integration

The agent uses LangChain as its LLM abstraction and tool-calling layer.

### LLM abstraction (`agents.py`)

`_build_lc_llm(model_key)` creates a LangChain chat model from the registry:

```python
# OPENAI type → ChatOpenAI (covers OpenAI, xAI, local llama.cpp, Ollama)
ChatOpenAI(model=..., base_url=..., api_key=..., streaming=True, timeout=...)

# GEMINI type → ChatGoogleGenerativeAI
ChatGoogleGenerativeAI(model=..., google_api_key=...)
```

Both return the same `ainvoke()` / `bind_tools()` interface — the rest of `agentic_lc()` is model-agnostic.

### Responses API path (`agents_xai.py`)

Models flagged with `xai_responses_api: true` or `openai_responses_api: true` in `llm-models.json` bypass LangChain and use the native Responses API via `agentic_responses_api()`. This enables direct access to provider-specific features (e.g., extended thinking for xAI/Grok). The dispatch logic in `dispatch_llm()` checks for these flags before falling back to the standard `agentic_lc()` loop.

### Tool schema format (`tools.py`)

All tools — core and plugin — are defined as LangChain `StructuredTool` objects with Pydantic `BaseModel` argument schemas. This is the **single source of truth**:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class _MyToolArgs(BaseModel):
    query: str = Field(description="Search query")

tool = StructuredTool.from_function(
    coroutine=my_executor,      # async executor function
    name="my_tool",
    description="Description shown to the LLM.",
    args_schema=_MyToolArgs,
)
```

`_lc_tool_to_openai_dict()` converts StructuredTool to OpenAI dict format on the fly for:
- `try_force_tool_calls()` — bare/XML tool call fallback for local models

No separate OpenAI or Gemini tool definitions are maintained.

### Content normalisation

`_content_to_str(content)` normalises `AIMessage.content` to a plain string.
Gemini returns content as `list[dict]` (content blocks); OpenAI returns `str`.
All `.content` accesses in `agents.py` go through this helper.

### Bare tool call fallback

Local models (Qwen, Hermes) sometimes output tool calls as raw text or XML rather than using the native tool-calling API. `try_force_tool_calls()` parses these and injects results as `HumanMessage` objects back into the context. Tool names are validated against the live `StructuredTool` registry.

## Slack Client Transport

The Slack plugin uses **asymmetric** transports — inbound and outbound are different mechanisms:

| Direction | Transport | Credential |
|---|---|---|
| Inbound (Slack → agent) | Socket Mode WebSocket (persistent connection) | `SLACK_APP_TOKEN` (starts with `xapp-`) |
| Outbound (agent → Slack) | Slack Web API `chat.postMessage` | `SLACK_BOT_TOKEN` (starts with `xoxb-`) |

**Required `.env` variables:**

```
SLACK_BOT_TOKEN=xoxb-...    # Web API calls (chat.postMessage for replies)
SLACK_APP_TOKEN=xapp-...    # Socket Mode connection (inbound events)
```

`SLACK_WEBHOOK_URL` is **not used** — do not add it. Webhook URLs cannot target specific threads, so `chat.postMessage` is used instead to maintain Slack thread context.

**Flow:**
1. Slack sends event over Socket Mode WebSocket → `_handle_socket_mode_request()`
2. Socket Mode acknowledgement sent immediately (required by Slack within 3 seconds)
3. Message dispatched to `process_request()` → LLM → response accumulated in queue
4. Accumulated response sent via `chat.postMessage` in the originating thread

## Session Model

Sessions are keyed by `client_id`:

| Client type | client_id format |
|---|---|
| shell.py | read from `.aiops_session_id` |
| llama proxy | `llama-<client-ip>` |
| Slack | `slack-<channel_id>-<thread_ts>` |

Each session stores: `model`, `history`, `tool_preview_length`, `_temp_model_active`.

Sessions persist in memory until explicitly deleted (`!session <ID> delete`) or server restart.

Shorthand IDs (101, 102, ...) are assigned sequentially and map to full client IDs for convenience.

## Plugin System

Plugins are declared in `plugin-manifest.json` and enabled/disabled in `plugins-enabled.json`.

### Plugin types

**`client_interface`** — adds HTTP/WebSocket endpoints to the server:
- `plugin_client_shellpy` — SSE streaming endpoint for shell.py (port 8765)
- `plugin_proxy_llama` — OpenAI/Ollama-compatible proxy (configurable port, default 11434)
- `plugin_client_slack` — Slack client: inbound via Socket Mode WebSocket, outbound via Web API (`chat.postMessage`)

**`data_tool`** — registers tools callable by the LLM:
- `plugin_database_mysql` — `db_query` tool
- `plugin_storage_googledrive` — `google_drive` tool
- `plugin_search_ddgs` — `search_ddgs` tool
- `plugin_search_tavily` — `search_tavily` tool
- `plugin_search_xai` — `search_xai` tool
- `plugin_search_google` — `search_google` tool
- `plugin_search_perplexity` — `perplexity_search`, `sonar_answer` tools
- `plugin_urlextract_tavily` — `url_extract` tool
- `plugin_calendar_google` — `calendar_google` tool
- `plugin_geocode_google` — `geocode_google` tool
- `plugin_places_google` — `places_google` tool
- `plugin_weather_google` — `weather_google` tool
- `plugin_sms_proxy` — `sms_proxy` tool

### Plugin loading sequence (llmem-gw.py startup)

1. Read `plugin-manifest.json` — all known plugins and their metadata
2. Read `plugins-enabled.json` — which plugins are active
3. Import each enabled plugin module dynamically
4. Call `plugin.init(config)` — connects to DB, authenticates, opens ports
5. Call `plugin.get_tools()` — returns `{'lc': [StructuredTool, ...]}` list
6. Register plugin routes with Starlette
7. Call `agents_module.update_tool_definitions()` — rebuilds `_CURRENT_LC_TOOLS` from core + all plugins

### BasePlugin contract

```python
class BasePlugin(ABC):
    PLUGIN_NAME: str       # e.g. "plugin_database_mysql"
    PLUGIN_TYPE: str       # "client_interface" | "data_tool"

    def init(self, config: dict) -> bool: ...    # return False to abort load
    def shutdown(self) -> None: ...
    def get_tools(self) -> dict: ...             # {"lc": [StructuredTool, ...]}
    def get_routes(self) -> list[Route]: ...     # client_interface only
```

## Toolset Architecture

Tool access is controlled per-model via the `llm_tools` field in `llm-models.json`. Each model declares which tools it may use. The server filters the tool list before each LLM invocation so the model only sees its permitted tools.

**`llm_tools` field:**
- `"all"` — model sees every registered tool (core + plugin)
- `["tool_a", "tool_b"]` — model sees only the listed tools
- `[]` (empty list) — model sees no tools (text-only)

**Runtime management via unified resource tools:**

The 24 individual tool/gate/config commands have been replaced by 5 unified resource tools:

| Tool | Purpose | Example |
|---|---|---|
| `llm_tools` | View/edit per-model tool access lists | `!llm_tools list`, `!llm_tools read gemini25f` |
| `model_cfg` | View/edit model configuration fields | `!model_cfg read gemini25f` |
| `sysprompt_cfg` | View/edit system prompt sections | `!sysprompt_cfg read` |
| `config_cfg` | View/edit server configuration | `!config_cfg read` |
| `limits_cfg` | View/edit depth and rate limits | `!limits_cfg read` |

These tools are also available as LLM tool calls with the same names.

## Swarm / Multi-Agent Coordination

The `agent_call` tool allows any LLM session to contact another llmem-gw instance
and return its response. This enables multi-agent workflows: delegation, verification,
parallel perspectives, or fan-out across specialised nodes.

### How it works

```
Primary node (human session)          Remote node (any llmem-gw instance)
────────────────────────────          ─────────────────────────────────────
agentic_lc() calls agent_call()  ──► POST /api/v1/submit  (plugin_client_api)
                                       process_request() → LLM → tool calls
AgentClient.stream() drains SSE  ◄──  push_tok() / push_done()
each chunk relayed via push_tok()
push_done() after each turn
Slack consumer posts turn to Slack
```

Remote session identity is derived as `api-swarm-{md5(calling_client:agent_url)[:8]}`,
so repeated calls from the same human session to the same URL reuse the same remote
session (history is preserved). Pass `target_client_id` to override.

### One-hop depth guard

**Why it exists:** Without cycle detection, a full-mesh topology can produce infinite
loops. If A calls B and B's LLM tries to call A (or any node), the chain recurses
until it hits a timeout or resource limit.

**How it works:** Every swarm call uses a `client_id` prefixed `api-swarm-`. When
`agent_call()` runs, it reads `current_client_id` (a `contextvars.ContextVar` set by
`execute_tool()`). If that ID starts with `api-swarm-`, the call is rejected immediately:

```python
# agents.py — agent_call()
if calling_client.startswith("api-swarm-"):
    return "[agent_call] Max swarm depth reached (1 hop). Call rejected to prevent recursion."
```

This means: a node that was *itself* called as a remote agent cannot make further
`agent_call` invocations. The primary orchestrates; remotes only respond.

**Topology supported today:**
- Star (one primary → N remotes): fully supported
- Full mesh (any node → any other node, human-initiated): fully supported
- Chaining (A → B → C) and loops (A → B → A): blocked by the guard

**Removing or relaxing the guard:** The single line above in `agents.py:agent_call()`
is the only enforcement point. To support deeper chains or configurable depth, replace
the prefix check with a hop-count header passed through `AgentClient` and propagated
in the `/api/v1/submit` payload. The remote would increment the counter and reject when
it exceeds the configured maximum. Cycle detection requires a visited-node set (e.g. a
list of node URLs passed in the request header) so a node can refuse if it sees its own
URL already in the chain.

### Queue drain on new submission

When a new request arrives for a session (`POST /api/v1/submit`), the server:
1. Cancels any active LLM task for that session (`cancel_active_task`)
2. Drains all pending items from the session's SSE queue (`drain_queue`)

This ensures stale responses from prior conversations — including orphaned tokens
from streams that disconnected before fully draining — never appear as the response
to a new request. Implemented in `plugin_client_api.py:endpoint_api_submit()`.

## System Prompt Structure

The system prompt is a recursive tree of section files:

```
.system_prompt                    ← root: main paragraph + [SECTIONS] list
  .system_prompt_memory-hierarchy
  .system_prompt_tool-guardrails
  .system_prompt_tools            ← container: [SECTIONS] list only
    .system_prompt_tool-db-query
    .system_prompt_tool-url-extract
    ...
  .system_prompt_behavior
```

Loop detection prevents circular references at load time. Duplicate section names across branches are also caught.

All sections at all depths are addressable by name or index:
- `read_system_prompt("tool-url-extract")` — returns just that tool's definition
- `update_system_prompt("behavior", "append", "...")` — edits any leaf section

Container sections (those with `[SECTIONS]`) cannot be directly edited — edit their children instead.

## LLM Model Registry

Models are registered in `llm-models.json` and loaded at startup by `config.py`. Two types are supported:

| Type | LangChain class | Examples |
|---|---|---|
| `OPENAI` | `ChatOpenAI` | grok-4, gpt-5.2, local llama.cpp, Ollama |
| `GEMINI` | `ChatGoogleGenerativeAI` | gemini-2.5-flash |

Each model entry:

| Field | Description |
|---|---|
| `model_id` | Model name passed to the API |
| `type` | `OPENAI` or `GEMINI` |
| `host` | API base URL (`null` for official Gemini endpoint) |
| `env_key` | `.env` key holding the API key (`null` for keyless local models) |
| `max_context` | Max messages retained in session history |
| `enabled` | `true`/`false` — disabled models are excluded from the registry entirely |
| `llm_tools` | `"all"`, or a list of tool names the model may use (e.g. `["ddgs_search", "db_query"]`) |
| `llm_call_timeout` | Timeout in seconds for delegation calls |
| `description` | Human-readable label shown in `!model` |

### Adding a new model

1. Add an entry to `llm-models.json`
2. Add the API key to `.env` if required
3. Restart the server — `config.py` loads `llm-models.json` at import time

No code changes are needed for models that use the standard `OPENAI` or `GEMINI` type.

## LLM Delegation Tools

The unified `llm_call` tool delegates sub-tasks to other registered models:

| Tool | Context sent | Tools available | Use case |
|---|---|---|---|
| `llm_call(model, prompt, mode='text')` | Prompt only — no context, no tools | None (text only) | Summarization, analysis of embedded data |
| `llm_call(model, prompt, mode='tool', tool=...)` | Tool def only | One named tool | Isolated tool call via target model |
| `llm_call(model, prompt, mode='agent')` | Full history and tool context | Target model's `llm_tools` set | Sub-agent with full capabilities |
| `@model <prompt>` (user-initiated) | Full session | Model's `llm_tools` set | Full turn delegation to free/local model |

`backup_models` can be configured per-model in `llm-models.json` for automatic fallback when the primary delegation target fails.

Tool access for delegation targets is controlled by the target model's `llm_tools` field in `llm-models.json`. Manage with `!llm_tools read <model>` and `!llm_tools write <model> <tools>`.

## Rate Limiting

Universal rate limiter in `agents.py` runs before tool execution. Configured per tool type in `plugins-enabled.json`:

```json
"rate_limits": {
  "llm_call": {"calls": 3, "window_seconds": 20, "auto_disable": true},
  "search":   {"calls": 5, "window_seconds": 10, "auto_disable": false},
  "extract":  {"calls": 5, "window_seconds": 30, "auto_disable": false},
  "drive":    {"calls": 10, "window_seconds": 60, "auto_disable": false},
  "db":       {"calls": 20, "window_seconds": 60, "auto_disable": false}
}
```

`auto_disable: true` disables all tools of that type for the session when the limit is exceeded.

## Client Protocol Detection

The llama proxy (`plugin_proxy_llama`) auto-detects the client format:

| Signal | Format |
|---|---|
| User-Agent contains "ollama" OR path starts with `/api/` | Ollama (NDJSON) |
| User-Agent contains "open-webui" | OpenAI (SSE) |
| Path starts with `/v1/api/` | Enchanted hybrid → Ollama response |
| Default | OpenAI (SSE) |

All formats route to the same `process_request()` — protocol differences are only in request parsing and response serialization.
