# llmem-gw

A multi-client AI agent server with a plugin architecture. Maintains persistent conversation sessions, routes all requests through a unified LangChain-based LLM dispatch loop, enforces human-approval gates on tool calls, and exposes a modular plugin system for data tools and client interfaces.

---

## Why Start Here Instead of From Scratch

Building a multi-LLM agent from scratch means solving: async session management, streaming SSE, multi-interface adapters (OpenAI vs. Ollama wire format alone is a week of work), per-tool gate/permission systems, rate limiting with auto-disable, tiered system prompt assembly, swarm coordination with depth guards, and a plugin discovery mechanism. This codebase has all of that working and tested across production use.

You inherit ~25,000 lines solving infrastructure so you can write the 100 lines that make your agent unique.

This system serves two distinct developer profiles. Both get a foundation rather than a blank page.

---

### Audience 1: Agent System Designers

You want to deploy a capable multi-LLM agent and shape its behavior — without writing Python.

**You don't write code. You design behavior.**

#### 5-Minute Start

```bash
git clone https://github.com/derezed88/llmem-gw.git
cd llmem-gw
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add at least one API key (e.g. GEMINI_API_KEY)
python llmem-gw.py           # server is running
python shell.py               # connect and start chatting
```

That's a working multi-tool LLM agent with MySQL, Google Drive, web search, and Slack support available as plugins.

#### Pick Your LLM — Or Use Several

`llm-models.json` is the model registry. Add an entry, drop the API key in `.env`, restart. No code. Out of the box:

- **Local**: Qwen3.5-9B via Ollama (no API cost)
- **OpenAI**: gpt-4o-mini, gpt-5-mini, gpt-5-nano
- **Google**: Gemini 2.5 Flash / Flash-Lite / 2.0 Flash
- **xAI**: Grok 4 Fast (reasoning and non-reasoning)
- **Anthropic**: Claude Sonnet / Haiku (via OpenAI-compatible endpoint)

Switch the active model at runtime: `!model gemini25f` — persisted to disk immediately, no restart.

#### Control What the LLM Can Touch

Gates are per-tool, per-table read/write permissions. The pattern is `!<toolname>_gate_read` / `!<toolname>_gate_write`. Set defaults in `gate-defaults.json` or change live in chat:

```
!google_drive_gate_read false      # auto-allow Drive reads (gate off)
!db_query_gate_write * false       # auto-allow DB writes for all tables
!sysprompt_gate_write false        # auto-allow system prompt updates
!search_ddgs_gate_read false       # auto-allow DuckDuckGo searches
```

`true` = gated (human must approve each call), `false` = auto-allow. Non-interactive clients (open-webui, Slack) auto-reject gated calls rather than hanging — the LLM is told why and asks for alternatives. Use `!gate_list` to see live status of every gate.

#### Tune Agent Behavior via Text Files

The system prompt is a set of modular section files — not a monolithic blob. Edit any section, add new ones, or assign a different folder to each model:

```
system_prompt/
├── 000_default/         ← section files (behavior, memory chain, tool rules...)
├── 001_blank/           ← minimal alternative for specialized deployments
├── 004_reasoning/       ← reasoning-arm prompt (Grok / Claude)
├── 004_voice/           ← voice-frontend prompt
└── 007_judge/           ← judge/evaluator prompt
```

Give a local low-power model a stripped prompt; give frontier models the full PDDS memory chain. Each model's folder is set in `llm-models.json` — one field, no code.

#### A Self-Learning, Self-Evolving System

The combination of persistent memory, writable system prompt, and LLM access to the admin command surface enables something beyond simple configuration: **a system that can learn from interactions and evolve its own behavior over time.**

The pieces that make this possible:

- **Tiered persistent memory** — facts, preferences, and outcomes are stored in a two-tier hierarchy: short-term (hot context, injected every request) and long-term (aged-out summaries, retrieved on demand). Memory ages automatically — short-term rows are summarised by the LLM and promoted to long-term without operator involvement.
- **Context auto-enrichment** — before each LLM call, `auto-enrich.json` rules are evaluated against the user's message. Matching rules execute SQL queries and inject the results as grounded context — no tool call required from the LLM. Useful for identity tables, deployment config, or any data that should be available without asking. Individual rules can be disabled with `"enabled": false`; the entire feature can be suppressed for a session with `!config write auto_enrich false`.
- **Semantic retrieval via vector search** — each turn, the agent embeds the current topic and queries a local Qdrant vector store to pull the most relevant long-term memories into context. Retrieval is score-gated and tier-aware: only memories that match the current topic surface, regardless of when they were stored.
- **Topic-aware memory routing** — the agent tags every response with a `<<topic-slug>>` label. The system extracts this tag, uses it as the topic key for memory storage, and feeds it as the Qdrant query seed next turn. Memory stays coherent across long sessions without manual tagging.
- **Writable, modular system prompt** — the LLM can append, replace, or delete sections of its own operating rules. A rule learned in one conversation becomes part of how it behaves in all future conversations.
- **Full admin command access** — gate permissions, model selection, context limits, and rate limits are all tools the LLM can invoke. Tell it "use the local model for searches from now on" and it switches and remembers.

The result: an agent that gets better at your specific workflows the more you use it. It learns your terminology, stores your preferences, tightens its own behavior rules, and adapts its tool usage — without you writing code or editing config files. The operator retains control through gates and the system prompt's keyword guards, but within those bounds the agent is free to evolve.

This is the foundation for building an agent that isn't just a stateless tool — it's an accumulating, adapting system.

#### Runtime Admin Without Code

All configuration is JSON + commands. Nothing requires a restart:

```
!model gemini25f                       # switch LLM
!search_ddgs_gate_read false           # auto-allow DuckDuckGo searches
!maxctx 50                             # set history window
!session                               # list active sessions with shorthand IDs
!session 102 delete                    # drop a session
!limit_set max_agent_call_depth 2      # raise swarm recursion limit
```

Rate limits, session timeouts, tool permissions, model timeouts — all configurable live via `!commands` or `llmemctl.py` CLI.

**The LLM has access to the same command surface you do.** Gate commands, model switching, session inspection, system prompt edits, and limit adjustments are all available as tools the LLM can call. You can instruct the agent in natural language ("switch to gemini", "allow drive reads from now on") and it executes the corresponding command itself, without you typing it.

#### llmemctl.py — Offline System Configuration

`llmemctl.py` is the operator's configuration tool, run while the server is stopped (or to set persistent defaults before first start). It edits `plugins-enabled.json`, `llm-models.json`, and `gate-defaults.json` directly. It also has an interactive menu mode (`python llmemctl.py` with no arguments).

**Plugins:**
```bash
python llmemctl.py list                          # show all plugins and enabled status
python llmemctl.py enable plugin_tmux            # enable a plugin
python llmemctl.py disable plugin_client_slack   # disable a plugin
python llmemctl.py info plugin_database_mysql    # show deps, env vars, tools
```

**Models:**
```bash
python llmemctl.py models                        # list all models
python llmemctl.py model gemini25f               # set default model
python llmemctl.py model-add                     # interactive: add a new model
python llmemctl.py model-enable gpt4om           # enable a disabled model
python llmemctl.py model-disable gpt4om          # disable without removing
python llmemctl.py model-context gemini25f 200   # set context window
python llmemctl.py model-llmcall gemini25f true  # allow model to call other LLMs
python llmemctl.py model-timeout gemini25f 120   # set LLM delegation timeout (s)
```

**Gate defaults** (persisted to `gate-defaults.json`, loaded at every startup):
```bash
python llmemctl.py gate-list                          # show all gate defaults
python llmemctl.py llm-allow google_drive read false   # auto-allow Drive reads by default
python llmemctl.py llm-allow db * write false          # auto-allow all DB writes by default
python llmemctl.py gate-reset                         # restore factory defaults
```

**Rate limits:**
```bash
python llmemctl.py ratelimit-list                     # show current limits
python llmemctl.py ratelimit-set search 10 30         # 10 calls per 30s for search tools
python llmemctl.py ratelimit-autodisable llm_call true  # auto-disable on breach
```

**Ports, limits, session defaults:**
```bash
python llmemctl.py port-list                          # show configured ports
python llmemctl.py port-set llama_port 11435          # change llama proxy port
python llmemctl.py limit-set max_agent_call_depth 2   # raise swarm recursion limit
python llmemctl.py history-maxctx 100                 # set global history window
python llmemctl.py max-users 20                       # max simultaneous sessions
python llmemctl.py session-timeout 120                # idle session timeout (minutes)
```

**History chain** (for custom history backends):
```bash
python llmemctl.py history-list                          # show chain and plugins
python llmemctl.py history-chain-add plugin_history_vec  # append custom plugin
python llmemctl.py history-chain-remove plugin_history_vec
```

---

### Audience 2: Code-Level Developers

You're building custom integrations, new tools, or specialized agent behaviors on top of a working foundation.

**You write the 100 lines that make your agent unique. The other 24,900 are already here.**

#### 5-Minute Start for a New Plugin

The smallest working example in the codebase is [`plugin_search_ddgs.py`](plugin_search_ddgs.py) — 111 lines, fully functional:

```python
class SearchDdgsPlugin(BasePlugin):
    PLUGIN_NAME    = "plugin_search_ddgs"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE    = "data_tool"
    DESCRIPTION    = "Web search via DuckDuckGo (no API key required)"
    DEPENDENCIES   = ["ddgs"]

    def get_tools(self) -> dict:
        return {"lc": [StructuredTool.from_function(
            coroutine=search_ddgs_executor,
            name="search_ddgs",
            args_schema=_DdgsSearchArgs,
        )]}
```

That's it. [`plugin_loader.py`](plugin_loader.py) discovers it, wires it into the gate system, rate limiter, and LLM tool dispatch automatically. Declare the schema with Pydantic, return the tool, done.

#### The Tool Ecosystem Is Already Populated

[`tools.py`](tools.py) (1,561 lines) defines the core tool layer. Plugins extend it. What's already working:

| Category | Tools |
|---|---|
| Memory / storage | `db_query` (MySQL), `google_drive` (list/read/write/search), `memory_save`, `memory_recall`, `memory_update`, `memory_age` |
| Search | `search_ddgs` (no API key), `search_google`, `search_tavily`, `search_xai` |
| Web | `url_extract` (Tavily content extraction), `file_extract` (Gemini file extraction) |
| System | `get_system_info`, `sysprompt_cfg` (read/write/delete sections) |
| LLM delegation | `llm_call` (clean context or tool call), `agent_call` (swarm) |
| Config / admin | `model_cfg`, `config_cfg`, `limits_cfg`, `llm_tools`, `judge_configure` |
| Inspection | `tool_list`, `session`, `llm_list`, `help` |
| Terminal | `tmux_new`, `tmux_exec`, `tmux_ls`, `tmux_kill_session`, `tmux_kill_server`, `tmux_history`, `tmux_history_limit` (via [`plugin_tmux.py`](plugin_tmux.py)) |

New tools you add are immediately available to all connected LLMs — no restart, because registration happens at load time.

#### History Is a Swappable Chain

[`plugin_history_default.py`](plugin_history_default.py) provides a sliding-window implementation. The contract is two methods:

```python
def process(history, session, model_cfg) -> list[dict]:
    """Called per-request. Returns trimmed history."""

def on_model_switch(session, old_model, new_model, ...) -> list[dict]:
    """Called immediately on !model switch."""
```

Swap it for Redis, SQLite, or a vector store by implementing a new `plugin_history_*.py`. The chain is configured in `plugins-enabled.json`:

```json
"chain": ["plugin_history_default", "plugin_history_custom"]
```

Each plugin in the chain receives the output of the previous one — composable history processing.

#### Swarm Architecture Is Already Wired

[`plugin_client_api.py`](plugin_client_api.py) exposes:

```
POST /api/v1/submit              ← submit message or command
GET  /api/v1/stream/{id}         ← SSE stream (tok, gate, done, error events)
POST /api/v1/gate/{gate_id}      ← respond to a gate programmatically
GET  /api/v1/sessions            ← list active sessions
```

`agent_call(agent_url, message)` lets any LLM call any other llmem-gw instance. Depth guards (`max_at_llm_depth`, `max_agent_call_depth`) prevent recursive runaway. Swarm session IDs are deterministic — repeated calls from the same session to the same remote agent reuse the same remote session, preserving history across calls.

Drive a session programmatically in ~10 lines via [`api_client.py`](api_client.py):

```python
from api_client import AgentClient
client = AgentClient("http://localhost:8767")
response = await client.send("!model")
async for token in client.stream("summarise my drive files"):
    print(token, end="", flush=True)
```

#### What You'd Have to Build Without This

The infrastructure already solved here, that you would otherwise spend weeks on:

- Async session management with per-client queues (`state.py`)
- SSE streaming with keepalives across all four client types
- OpenAI vs. Ollama wire format detection and translation (`plugin_proxy_llama.py`)
- Per-tool gate UI with interactive shell approval and auto-reject for non-interactive clients
- Rate limiting by tool type with auto-disable on breach (`agents.py:check_rate_limit`)
- Bare-JSON tool call extraction for local models that don't use the native function-calling API
- Deterministic swarm session ID derivation and one-hop depth guards
- Modular system prompt assembly with per-model folder assignment
- Plugin dependency validation, env-var checking, and priority-ordered load

#### The Plugin Contract Reference

```python
class BasePlugin(ABC):
    PLUGIN_NAME: str       # unique, matches filename
    PLUGIN_VERSION: str
    PLUGIN_TYPE: str       # "data_tool" or "client_interface"
    DESCRIPTION: str
    DEPENDENCIES: List[str]  # pip package names — validated at load time

    def init(self, config: dict) -> bool: ...      # connect, validate env
    def shutdown(self) -> None: ...
    def get_tools(self) -> dict: ...               # {"lc": [StructuredTool, ...]}
    def get_gate_tools(self) -> dict: ...          # declares gate types per tool
    def get_routes(self) -> List[Route]: ...       # client_interface only
    def get_commands(self) -> dict: ...            # optional !command handlers
    def get_help(self) -> str: ...                 # optional help text
```

See [`plugin_search_ddgs.py`](plugin_search_ddgs.py) (101 lines) for the minimal working pattern. See [`plugin_client_api.py`](plugin_client_api.py) (~300 lines) for a full client interface with SSE, gates, and session management.

---

## Technical Overview

### LangChain LLM Abstraction

All LLM backends are unified under a single `agentic_lc()` loop using LangChain:

- **`ChatOpenAI`** — covers OpenAI, xAI/Grok, and any local llama.cpp or Ollama server (all speak the OpenAI chat completions API)
- **`ChatGoogleGenerativeAI`** — covers Gemini models via the Google GenAI SDK

`_build_lc_llm(model_key)` constructs the correct LangChain chat model from `llm-models.json` at call time. Adding a new LLM backend requires only a JSON entry — no code changes.

The loop calls `llm.bind_tools(tools)` once and `llm.ainvoke(messages)` each iteration. Tool calls, results, and conversation history are exchanged as typed LangChain message objects (`AIMessage`, `ToolMessage`, `HumanMessage`). A `_content_to_str()` helper normalises model responses — Gemini returns content as a list of typed blocks; OpenAI returns a plain string.

### Tool Definition — LangChain StructuredTool

All tools — core and plugin — are defined as LangChain `StructuredTool` objects with Pydantic `BaseModel` argument schemas. This is the single source of truth for every backend:

```python
class _SearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: Optional[int] = Field(default=10, description="Max results")

StructuredTool.from_function(
    coroutine=search_executor,
    name="search_ddgs",
    description="Search the web via DuckDuckGo.",
    args_schema=_SearchArgs,
)
```

`_lc_tool_to_openai_dict()` converts StructuredTool to OpenAI dict format on the fly for the bare-text tool call fallback used by local models. No separate OpenAI or Gemini tool declarations are maintained anywhere.

### Bare Tool Call Fallback

Local models (Qwen, Hermes) sometimes output tool calls as raw JSON or XML rather than using the native function-calling API. `try_force_tool_calls()` extracts these from the model's text output and re-injects them as proper tool calls, with results fed back as `HumanMessage` objects. Tool names are validated against the live StructuredTool registry.

### Plugin Architecture

Plugins are loaded dynamically from `plugin-manifest.json` at startup. Each plugin is a Python file with a single `BasePlugin` subclass:

```python
class MyPlugin(BasePlugin):
    PLUGIN_TYPE = "data_tool"          # or "client_interface"

    def init(self, config) -> bool: ...     # connect, validate env
    def shutdown(self) -> None: ...
    def get_tools(self) -> dict: ...        # returns {"lc": [StructuredTool, ...]}
    def get_gate_tools(self) -> dict: ...   # declares gate types per tool
```

Executors are auto-extracted from `StructuredTool.coroutine` — no separate executor registry needed.

### Client Interfaces

| Client | Transport | Protocol |
|---|---|---|
| `shell.py` terminal | SSE (port 8765) | Custom SSE streaming + gate approval UI |
| OpenAI-compatible chat apps (open-webui, LM Studio) | HTTP (`llama_port`, default 11434) | OpenAI chat completions (streaming + non-streaming) |
| Ollama-compatible apps | HTTP (`llama_port`, default 11434) | Ollama NDJSON |
| Slack | Socket Mode WebSocket (inbound) + Web API (outbound) | Slack Events API / `chat.postMessage` |
| Programmatic / swarm | SSE (port 8767) | JSON/SSE via `api_client.py` or HTTP directly |

The llama proxy auto-detects client format from User-Agent and path prefix, then routes all formats to the same `process_request()` pipeline.

### Human Approval Gate System

Every tool call passes through `check_human_gate()` before execution. Gates are registered per tool type by plugins at startup — no hardcoded tool names in gate logic.

| Gate type | Command | Granularity |
|---|---|---|
| `search` tools | `!search_ddgs_gate_read true/false` | Per search engine |
| `url_extract` | `!url_extract_gate_read true/false` | Read gate |
| `google_drive` | `!google_drive_gate_read/write true/false` | Separate read and write |
| `db_query` | `!db_query_gate_read/write [table\|*] true/false` | Per-table, per-operation |
| `sysprompt_write` | `!sysprompt_gate_write true/false` | System prompt writes |
| `session` / `model` / `reset` | `!session_gate_read/write true/false` etc. | Per operation |

Gate defaults persist across restarts via `gate-defaults.json` (managed with `llmemctl.py llm-allow`).

Non-interactive clients (llama proxy, Slack) auto-reject gated calls immediately with an instructive message to the LLM. API clients get a 2-second window for programmatic approval.

### LLM Delegation

The session LLM can delegate sub-tasks to other registered models:

| Tool | What is sent | Use case |
|---|---|---|
| `llm_call(model, prompt, mode='text')` | Prompt only — no context, no tools | Summarisation, analysis |
| `llm_call(model, prompt, mode='tool', tool=...)` | Tool definition only | Isolated tool call via a second model |
| `llm_call(model, prompt, mode='agent')` | Full history and tool context | Sub-agent with full capabilities |

Enable delegation per model with `!llm_call <model> true`. Rate-limited by default (4 calls / 5 s, auto-disables on breach).

### Swarm / Multi-Agent Communication

The `plugin_client_api` plugin exposes a JSON/SSE HTTP API (port 8767 by default) for programmatic and agent-to-agent access. Combined with the `agent_call` tool, any LLM on any instance can reach any other instance that has the API plugin enabled.

**`agent_call(agent_url, message)`** — core swarm tool:
- Sends `message` to a remote llmem-gw instance at `agent_url`
- The remote agent processes the message through its full stack (LLM, tools, gates)
- Returns the complete text response to the calling LLM
- Session persistence: the remote session is derived deterministically from the calling session + target URL, so repeated calls from the same session to the same agent reuse the same remote session (history preserved across calls)
- Depth guard: calls from a swarm-originated session are rejected with an error to prevent unbounded recursion (max 1 hop)

**Programmatic access via `api_client.py`:**

```python
from api_client import AgentClient

client = AgentClient("http://localhost:8767")
response = await client.send("!model")          # sync — returns full text
async for token in client.stream("hello"):      # streaming — yields tokens
    print(token, end="", flush=True)
```

The API plugin and `api_client.py` are also the transport layer used internally by `agent_call`.

> **Note:** A scheme for agents to discover each other automatically is not yet implemented. Swarm targets must be specified explicitly by URL.

### Modular System Prompt

The system prompt is assembled at runtime from section files stored in `system_prompt/<folder>/`. Each model can have its own folder, configured via `system_prompt_folder` in `llm-models.json`. The default folder is `system_prompt/000_default/`.

Sections are editable by the LLM via `sysprompt_write` / `sysprompt_delete` tools or by the operator via `!sysprompt_*` commands. The full prompt or any section can be read with `!sysprompt_read <model> [section]`.

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/derezed88/llmem-gw.git
cd llmem-gw
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add at least one LLM API key (e.g. GEMINI_API_KEY)
```

Edit `llm-models.json` to match your models and endpoints. At minimum, make sure one model is `"enabled": true` with a valid `env_key` pointing to your `.env` variable.

### 3. Start the server

```bash
source venv/bin/activate
python llmem-gw.py
```

### 4. Connect with shell.py (in a second terminal)

```bash
source venv/bin/activate
python shell.py
```

Type `!help` to see all commands. Some useful ones to start:

```
!model                          list available LLMs (* = current)
!model <name>                   switch active model
!search_ddgs_gate_read true     auto-allow DuckDuckGo searches (no gate pop-ups)
!db_query_gate_read * true      auto-allow all DB reads
!reset                          clear conversation history
!session                        list all active sessions
```

---

## Architecture

```
Clients                 Server                          Backends
───────                 ──────                          ────────
shell.py     ──SSE──►  llmem-gw.py                     OpenAI API
open-webui   ─HTTP──►  ┌──────────────────────────┐    Gemini API
LM Studio    ─HTTP──►  │ routes.py                │    xAI API
Slack ─Socket Mode──►  │ agents.py (agentic_lc)   │    FriendliAI serverless
api_client   ─HTTP──►  │   LangChain bind_tools() │    llama.cpp / Ollama
Agent B ──────HTTP──►  │   ChatOpenAI             │──► MySQL
                       │   ChatGoogleGenerativeAI │──► Google Drive
                       │ plugin_*.py              │──► Web search APIs
                       └──────────────────────────┘
                                    │ agent_call tool
                                    ▼
                             Agent B / Agent C  (other llmem-gw instances)
```

- **`llmem-gw.py`** — entry point; loads plugins, builds Starlette app, starts uvicorn servers
- **`agents.py`** — LangChain dispatch loop, tool execution, rate limiting, LLM delegation
- **`tools.py`** — StructuredTool registry; single source of truth for all tool schemas
- **`shell.py`** — interactive terminal client with gate approval UI
- **`plugin_*.py`** — pluggable data tools and client interfaces
- **`llmemctl.py`** — CLI tool for managing plugins and models

---

## Plugins

**Client interfaces** — how users and other agents connect:

| Plugin | Port | What it adds |
|---|---|---|
| `plugin_client_shellpy` | 8765 | Interactive terminal (`shell.py`) with streaming and gate approval UI |
| `plugin_proxy_llama` | 11434 | OpenAI/Ollama-compatible proxy for open-webui, LM Studio, Enchanted, etc. |
| `plugin_client_slack` | — | Slack bidirectional bot via Socket Mode; per-thread sessions |
| `plugin_client_api` | 8767 | JSON/SSE HTTP API for programmatic access and agent-to-agent (swarm) calls |

**Data tools** — what the LLM can read and write:

| Plugin | Tool(s) | What it adds |
|---|---|---|
| `plugin_database_mysql` | `db_query` | SQL against MySQL; per-table read/write gates |
| `plugin_storage_googledrive` | `google_drive` | List, read, write, search files within an authorised Drive folder |
| `plugin_search_ddgs` | `search_ddgs` | DuckDuckGo web search — no API key required |
| `plugin_search_tavily` | `search_tavily` | Tavily AI-curated search results |
| `plugin_search_xai` | `search_xai` | xAI Grok search (web + X/Twitter) |
| `plugin_search_google` | `search_google` | Google search via Gemini grounding |
| `plugin_urlextract_tavily` | `url_extract` | Extract full text content from any URL via Tavily |
| `plugin_extract_gemini` | `file_extract` | Extract text/data from local files and Google Drive documents via Gemini file API; supports Drive Workspace export (Docs, Sheets, Slides) |
| `plugin_tmux` | `tmux_new`, `tmux_exec`, `tmux_ls`, `tmux_kill_session`, `tmux_kill_server`, `tmux_history`, `tmux_history_limit` | Persistent PTY shell sessions — LLM can run shell commands and read output; whitelist/blacklist configurable |
| `plugin_memory_vector_qdrant` | _(infrastructure)_ | Qdrant vector index for semantic memory retrieval. Embeds memory rows via a local embedding model and provides scored nearest-neighbour lookup by topic. MySQL remains the source of truth; Qdrant is the retrieval index. Requires a running Qdrant instance and a compatible embedding endpoint (e.g. nomic-embed-text via llama.cpp). |
| `plugin_claude_vscode_sessions` | _(client interface)_ | Exposes Claude Code session exports stored in Google Drive as a readable resource for the LLM |

**History** — how conversation context is managed:

| Plugin | What it adds |
|---|---|
| `plugin_history_default` | Sliding-window history trimming: keeps last N messages where N = min(agent_max_ctx, model.max_context). Always first in the chain. Additional `plugin_history_*.py` plugins can be appended to the chain for compression, vector retrieval, or custom strategies. |
| `plugin_history_judge` | Optional second chain member: passes each turn through a judge model to score or filter content before it enters the history window. |

Manage plugins:

```bash
python llmemctl.py list
python llmemctl.py enable <plugin_name>
python llmemctl.py disable <plugin_name>
```

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/QUICK_START.md](docs/QUICK_START.md) | Essential commands and first steps |
| [docs/ADMINISTRATION.md](docs/ADMINISTRATION.md) | Full plugin/model/gate/session management reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System internals, LangChain integration, request flow |
| [docs/PLUGIN_DEVELOPMENT.md](docs/PLUGIN_DEVELOPMENT.md) | How to write new plugins and add new models |
| [docs/setup_services.md](docs/setup_services.md) | systemd, tmux, screen, and tunnel deployment |
| [docs/plugin-client-api.md](docs/plugin-client-api.md) | API plugin — programmatic access and swarm setup |
| [docs/SWARMDESIGN.md](docs/SWARMDESIGN.md) | Swarm foundation and discovery design options |
| [docs/JUDGEMODEL.md](docs/JUDGEMODEL.md) | Judge model architecture — per-turn scoring, memory review, history filtering |
| [docs/plugin-*.md](docs/) | Per-plugin setup and configuration |

---

## Configuration Files

| File | Purpose |
|---|---|
| `.env` | API keys and credentials (never commit) |
| `llm-models.json` | Model registry — `type`, `host`, `env_key`, `enabled`, `tool_call_available`, `system_prompt_folder` |
| `plugins-enabled.json` | Active plugins, rate limits, per-plugin config |
| `gate-defaults.json` | Gate auto-allow defaults loaded at startup (managed via `llmemctl.py llm-allow`) |
| `system_prompt/<folder>/` | Modular system prompt sections; `000_default/` ships with the repo |
| `auto-enrich.json` | Context auto-enrichment rules (gitignored — instance-specific) |
