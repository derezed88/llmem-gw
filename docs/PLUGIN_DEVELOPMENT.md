# MCP Agent Plugin Specification

This document defines the complete contract for writing plugins that integrate
with the MCP agent system. Any plugin that follows this spec will be automatically
discovered, loaded, and surfaced in `!help` — with no changes required to
mainline code (`llmem-gw.py`, `routes.py`).

---

## Plugin Types

| Type               | PLUGIN_TYPE value    | Must implement          | Loaded by          |
|--------------------|----------------------|-------------------------|--------------------|
| Data / AI tool     | `"data_tool"`        | `get_tools()`           | `llmem-gw.py`     |
| Client interface   | `"client_interface"` | `get_routes()`          | `llmem-gw.py`     |

Both types share the same base class and lifecycle methods.

---

## File Naming Convention

```
plugin_<type>_<name>.py
```

Examples:
- `plugin_search_ddgs.py`         — data_tool, search category
- `plugin_storage_googledrive.py` — data_tool, storage category
- `plugin_database_mysql.py`      — data_tool, database category
- `plugin_client_shellpy.py`      — client_interface
- `plugin_client_slack.py`        — client_interface
- `plugin_proxy_llama.py`         — client_interface

The class name inside the file must be a subclass of `BasePlugin`. The loader
finds it automatically by scanning for any `BasePlugin` subclass in the module.

---

## The Two Config Files

Before adding a plugin, understand the roles of the two JSON files that govern
the plugin system:

| File | Role | Who edits it |
|------|------|--------------|
| `plugin-manifest.json` | **Static registry** — declares that a plugin *exists*: its file, type, dependencies, required env vars, and load priority. Never read at agent startup for enable/disable decisions — only for validation metadata. | Plugin author (once, when adding the plugin) |
| `plugins-enabled.json` | **Runtime config** — which plugins to actually load, per-plugin config overrides (port, host, `enabled` flag), default model, and rate limits. This is the operator control panel. | `llmemctl.py` or direct edit |

**When you add a new plugin you touch both files:**
1. Add an entry to `plugin-manifest.json` so the system knows the plugin exists and can validate its dependencies.
2. Add the plugin name to the `enabled_plugins` list in `plugins-enabled.json` so the loader picks it up.  Add a `plugin_config` block only if you need non-default settings (port, host, or to start it disabled with `"enabled": false`).

**The `enabled: false` pattern** lets you keep a plugin in `enabled_plugins` (so its config is preserved) without actually starting it.  This is how `plugin_proxy_llama` and `plugin_client_slack` ship — configured but off until the operator flips the flag.  Use `python llmemctl.py enable <plugin>` or set it directly in `plugins-enabled.json`.

---

## Manifest Entry (`plugin-manifest.json`)

Every plugin must have an entry in `plugin-manifest.json`:

```json
"plugin_search_example": {
  "type": "data_tool",
  "file": "plugin_search_example.py",
  "description": "One-line human-readable description",
  "dependencies": ["package-name>=1.0"],
  "env_vars": ["EXAMPLE_API_KEY"],
  "config_files": [],
  "priority": 340,
  "tools": ["example_search"]
}
```

**Fields:**
- `type` — `"data_tool"` or `"client_interface"`
- `file` — filename relative to the llmem-gw working directory
- `description` — shown in `python llmemctl.py list`
- `dependencies` — pip package names (validated at startup; use `>=` for minimum version)
- `env_vars` — environment variable names required from `.env`; validated at startup
- `config_files` — additional files that must exist (e.g., `credentials.json`)
- `priority` — load order (lower = loaded first); use these ranges:
  - 10–99: client interfaces
  - 100–199: database plugins
  - 200–299: storage plugins
  - 300–399: search plugins
- `tools` — list of tool names this plugin provides (for reference / plugin-manager display)

---

## Base Class Contract (`BasePlugin`)

```python
from plugin_loader import BasePlugin
from typing import Dict, Any

class ExamplePlugin(BasePlugin):

    # ── Required class-level metadata ──────────────────────────────────────
    PLUGIN_NAME    = "plugin_search_example"   # must match manifest key
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE    = "data_tool"               # "data_tool" or "client_interface"
    DESCRIPTION    = "One-line description"
    DEPENDENCIES   = ["example-package"]       # pip install names
    ENV_VARS       = ["EXAMPLE_API_KEY"]       # required .env keys

    def __init__(self):
        self.enabled = False
        # store any clients/connections here

    # ── Required lifecycle methods ──────────────────────────────────────────

    def init(self, config: dict) -> bool:
        """
        Called once at startup with plugin-specific config from plugins-enabled.json.
        Return True on success, False to abort loading (plugin skipped).
        Validate env vars and connect here.
        """
        ...

    def shutdown(self) -> None:
        """Called on server shutdown. Close connections, set self.enabled = False."""
        ...
```

---

## `data_tool` Plugin — Additional Methods

### `get_tools()` — **Required**

Returns a dict with a single `"lc"` key containing a list of LangChain
`StructuredTool` objects. This is the **only** format — no OpenAI JSON dicts
or Gemini declarations needed. The same definition works for all LLM backends.

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import Optional

# 1. Define argument schema as a Pydantic model
class _ExampleSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: Optional[int] = Field(
        default=10,
        description="Maximum number of results (default: 10)"
    )

# 2. Define the async executor at module level
async def example_search_executor(query: str, max_results: int = 10) -> str:
    return await _run_example_search(query, max_results)

# 3. Return the StructuredTool from get_tools()
def get_tools(self) -> Dict[str, Any]:
    return {
        "lc": [
            StructuredTool.from_function(
                coroutine=example_search_executor,
                name="example_search",
                description="Search the web using Example API. Returns titles, URLs, and snippets.",
                args_schema=_ExampleSearchArgs,
            )
        ]
    }
```

**Rules:**
- `name` must be a valid Python identifier and must match the executor function's
  logical name (used by `execute_tool()` and gate checks).
- `coroutine` must be an `async def` function returning `str`.
- `args_schema` must be a `pydantic.BaseModel` subclass. Always provide one —
  even for no-argument tools (`class _NoArgs(BaseModel): pass`) — to prevent
  LangChain from leaking function docstrings into the schema.
- If the underlying library is synchronous, wrap it with
  `await asyncio.get_event_loop().run_in_executor(None, _sync_fn)`.
- If the executor needs instance state (e.g., an API client), make it a closure
  inside `get_tools()`. Put the actual I/O work in a module-level `_run_*()` for
  testability.
- Executors are auto-extracted from the `StructuredTool.coroutine` attribute —
  no separate `"executors"` key is needed.

**Schema tips for LLM compatibility:**
- Enum constraints (`Literal["basic", "advanced"]`) improve model accuracy.
- Keep descriptions short but precise — they are sent to the LLM on every call.
- `Optional` fields with sensible defaults reduce required arguments.

---

## `client_interface` Plugin — Additional Methods

### `get_routes()` — **Required**

Returns Starlette `Route` objects that are mounted on the HTTP server.

```python
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse

def get_routes(self) -> list:
    return [
        Route("/example/health", self._handle_health, methods=["GET"]),
        Route("/example/submit", self._handle_submit, methods=["POST"]),
    ]
```

### `get_config()` — **Required**

Returns server binding config. The loader starts a separate uvicorn server per
client plugin.

```python
def get_config(self) -> dict:
    return {
        "port": 8767,         # TCP port to bind
        "host": "0.0.0.0",    # bind address
        "name": "Example client"  # displayed in startup log
    }
```

**Rules:**
- Each client plugin binds its own port. Avoid conflicts with other running plugins.
  Known defaults (all configurable in `plugins-enabled.json`):
  - 8765: shellpy (MCP service)
  - 11434: llama proxy default (`llama_port` — change to any free port)
  - 8767+: suggested range for new client plugins
- Client plugins must **not** define `get_tools()` — they consume the agent,
  not extend its tool set.
- To push responses to the user, use `push_tok(client_id, text)` and
  `push_done(client_id)` from `state.py`.
- To submit user input to the agent, call `process_request(client_id, text, payload)`
  from `routes.py` as an `asyncio.create_task`.

---

## `get_commands()` — Optional (either plugin type)

Plugins may contribute `!command` handlers. Return a dict of command name →
async handler. The handler signature is `async def handler(args: str) -> str`.
The dispatcher in `routes.py` calls the handler and pushes the returned string
to the client — the plugin does not call `push_tok` directly.

```python
def get_commands(self) -> Dict[str, Any]:
    async def my_cmd(args: str) -> str:
        return f"mycommand received: {args}"

    return {"mycommand": my_cmd}
```

Pair with `get_help()` to add a section to `!help`:

```python
def get_help(self) -> str:
    return (
        "My Plugin:\n"
        "  !mycommand <args>   - do something\n"
    )
```

Commands are auto-registered at startup by `llmem-gw.py` via `register_plugin_commands()`
and dispatched generically by `cmd_plugin_command()` in `routes.py`.

---

## Startup Flow (what `llmem-gw.py` does)

```
for each plugin in plugins-enabled.json:
    validate(plugin)              # check file, env vars, dependencies
    load_plugin(plugin)           # dynamic import, find BasePlugin subclass
    plugin.init(config)           # plugin connects / initialises

    if PLUGIN_TYPE == "data_tool":
        tools_module.register_plugin_tools(name, plugin.get_tools())
        # → extends _PLUGIN_TOOLS_LC with the plugin's StructuredTool list
        # → auto-extracts executors from StructuredTool.coroutine

    elif PLUGIN_TYPE == "client_interface":
        mount plugin.get_routes() on Starlette app
        start uvicorn server on plugin.get_config()["port"]

agents_module.update_tool_definitions()
# → builds CORE_LC_TOOLS via _make_core_lc_tools()
# → sets _CURRENT_LC_TOOLS = core tools + all plugin tools
# → sets _CURRENT_OPENAI_TOOLS (derived, for bare-call fallback)
```

---

## Adding a New Model

Models live in `llm-models.json`. No code changes are needed for standard backends.

```json
"mymodel": {
  "model_id": "my-model-name",
  "type": "OPENAI",
  "host": "https://api.example.com/v1",
  "env_key": "EXAMPLE_API_KEY",
  "max_context": 100,
  "enabled": true,
  "description": "My custom model",
  "llm_tools": "all",
  "llm_call_timeout": 60
}
```

**Supported `type` values:**

| `type` | LangChain class | When to use |
|---|---|---|
| `OPENAI` | `ChatOpenAI` | Any OpenAI-compatible endpoint: OpenAI, xAI, local llama.cpp, Ollama |
| `GEMINI` | `ChatGoogleGenerativeAI` | Google Gemini models |

**Steps:**
1. Add the entry to `llm-models.json`
2. Add `EXAMPLE_API_KEY=...` to `.env` (if required)
3. Restart the server

The model is immediately available as `!model mymodel`. Set `llm_tools` to `"all"` for
full tool access, or provide a specific list of tool names the model should use.

**Local models (llama.cpp / Ollama):**
- Set `type: "OPENAI"`, `host` to the local server URL, `env_key: null`
- Set `llm_tools` to only the tools the model reliably handles
- Increase `llm_call_timeout` for slow hardware (e.g., 120s)

---

## Adding a New Plugin — Checklist

1. **Create** `plugin_<type>_<name>.py` — subclass `BasePlugin`, implement all
   required methods (see template below).
2. **Add entry** to `plugin-manifest.json` — file, type, description, dependencies,
   env_vars, config_files, priority, tools.
3. **Add to `plugins-enabled.json`** — append the plugin name to the `enabled_plugins`
   list.  Optionally add a `plugin_config` block for non-default port/host settings,
   or to ship it disabled (`"enabled": false`) until credentials are ready.
4. **Add env vars** to `.env` if required.
5. **Restart** `python llmem-gw.py`.
6. **Verify** with `python llmemctl.py list` — plugin should show `✓` (green,
   all deps and env vars present) or `–` (yellow, if you shipped it with `enabled: false`).
7. **Test** with `!help` in the client — new tool should appear automatically.
8. **Grant access** — add the new tool name to the `llm_tools` list for models that should use it in `llm-models.json`, or set `llm_tools` to `"all"`.

No changes to `routes.py` or `llmem-gw.py` are needed for
standard `data_tool` search/drive/storage plugins.

---

## Example: Minimal Read-Only Search Plugin

```python
"""
Example Search Plugin — minimal template for a read-only web search tool.
"""

import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin


# ── Argument schema ──────────────────────────────────────────────────────────

class _ExampleSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: Optional[int] = Field(
        default=10,
        description="Maximum number of results to return (default: 10)"
    )


# ── Executor (module-level so it can be tested independently) ────────────────

async def example_search_executor(query: str, max_results: int = 10) -> str:
    """Execute example search."""
    return await _run_example_search(query, max_results)


# ── Plugin class ─────────────────────────────────────────────────────────────

class SearchExamplePlugin(BasePlugin):

    PLUGIN_NAME    = "plugin_search_example"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE    = "data_tool"
    DESCRIPTION    = "Web search via Example API"
    DEPENDENCIES   = ["example-sdk"]
    ENV_VARS       = ["EXAMPLE_API_KEY"]

    def __init__(self):
        self.enabled  = False
        self._api_key = None

    def init(self, config: dict) -> bool:
        import os
        self._api_key = os.getenv("EXAMPLE_API_KEY")
        if not self._api_key:
            print("plugin_search_example: EXAMPLE_API_KEY not set")
            return False
        self.enabled = True
        return True

    def shutdown(self) -> None:
        self._api_key = None
        self.enabled  = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=example_search_executor,
                    name="example_search",
                    description=(
                        "Search the web using Example API. "
                        "Returns titles, URLs, and snippets for top results."
                    ),
                    args_schema=_ExampleSearchArgs,
                )
            ]
        }


# ── Implementation ───────────────────────────────────────────────────────────

async def _run_example_search(query: str, max_results: int = 10) -> str:
    """Actual search logic — synchronous SDK wrapped in run_in_executor."""
    def _sync():
        # Replace with real SDK call, e.g.:
        # results = ExampleSDK().search(query, max_results=max_results)
        return f"Results for: {query}"

    try:
        return await asyncio.get_event_loop().run_in_executor(None, _sync)
    except Exception as e:
        return f"example_search error: {e}"
```

**Notes on this template:**
- The `_ExampleSearchArgs` Pydantic schema is defined at module level, not inside `get_tools()`.
- The executor `example_search_executor` is at module level so it can be imported and tested independently.
- `_run_example_search` contains the actual I/O — the executor is just a thin bridge.
- If the plugin needs to capture instance state (e.g., `self._client`) in the executor, make the executor a closure inside `get_tools()` instead:

```python
def get_tools(self) -> Dict[str, Any]:
    client = self._client  # capture at registration time

    async def example_search_executor(query: str, max_results: int = 10) -> str:
        return await _run_example_search(client, query, max_results)

    return {
        "lc": [
            StructuredTool.from_function(
                coroutine=example_search_executor,
                name="example_search",
                description="...",
                args_schema=_ExampleSearchArgs,
            )
        ]
    }
```

---

## Key Files Reference

| File                     | Role                                                          |
|--------------------------|---------------------------------------------------------------|
| `plugin_loader.py`       | `BasePlugin` base class and `PluginLoader`                   |
| `plugin-manifest.json`   | Plugin metadata registry                                      |
| `plugins-enabled.json`   | Which plugins are active; per-plugin config; rate limits      |
| `llm-models.json`        | LLM model registry (add new models here)                      |
| `llmem-gw.py`           | Startup orchestration; registers tools                        |
| `tools.py`               | `StructuredTool` registry; `register_plugin_tools()`; per-model toolset filtering |
| `agents.py`              | `agentic_lc()` loop; `_build_lc_llm()`; `execute_tool()`; `_content_to_str()` |
| `routes.py`              | `!help`, unified `!command` dispatch                          |
| `llmemctl.py`      | CLI tool for system administration (plugins, models, config)  |
| `.env`                   | API keys and credentials                                      |
| `.system_prompt_tools`   | LLM-facing tool documentation (update manually for new tools) |
