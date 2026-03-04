import datetime
import platform
from mcp.server.fastmcp import FastMCP
from langchain_core.tools import StructuredTool

from config import log
from database import execute_sql
from drive import run_drive_op
from search import run_google_search
from prompt import get_section

mcp_server = FastMCP("AIOps-DB-Tools")


@mcp_server.tool()
async def db_query(sql: str) -> str:
    """Execute SQL against mymcp MySQL database."""
    return await execute_sql(sql)


@mcp_server.tool()
async def get_system_info() -> dict:
    """Return current date/time and status."""
    from state import current_client_id, sessions, estimate_history_size
    result: dict = {
        "local_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "PST",
        "status": "connected",
        "platform": platform.system(),
    }
    cid = current_client_id.get("")
    if cid and cid in sessions:
        sess = sessions[cid]
        history = sess.get("history", [])
        size = estimate_history_size(history)
        result["session_context"] = {
            "history_messages": len(history),
            "history_chars": size["char_count"],
            "history_token_est": size["token_est"],
        }
    return result


@mcp_server.tool()
async def google_search(query: str) -> str:
    """Search web using Gemini grounding."""
    return await run_google_search(query)


@mcp_server.tool()
async def google_drive(
    operation: str,
    file_id: str = "",
    file_name: str = "",
    content: str = "",
    folder_id: str = "",
) -> str:
    """
    Perform CRUD operations on Google Drive within a SPECIFIC authorized folder.

    IMPORTANT: This tool only accesses files in a pre-configured folder (FOLDER_ID from .env).
    Do NOT pass folder_id="root" or attempt to access the entire Drive.
    Leave folder_id empty to use the configured folder.
    """
    return await run_drive_op(
        operation,
        file_id or None,
        file_name or None,
        content or None,
        folder_id or None,
    )


# ---------------------------------------------------------------------------
# Dynamic Tool Registry
# ---------------------------------------------------------------------------

# Plugin tool storage — all plugins now use LangChain StructuredTool format (Step 2b)
_PLUGIN_TOOLS_LC: list = []
_PLUGIN_TOOL_EXECUTORS: dict = {}

# Plugin command registry: command_name -> async handler(subcommand_or_args: str) -> str
# Handlers have signature: async (args: str) -> str
# Populated at startup by register_plugin_commands(); queried by routes.py dispatch.
# Also stores optional help text per plugin: _PLUGIN_HELP[plugin_name] -> str
_PLUGIN_COMMANDS: dict = {}   # cmd_name -> async handler
_PLUGIN_HELP: dict = {}       # plugin_name -> help string (from get_help())


def register_plugin_commands(plugin_name: str, commands: dict, help_text: str = ""):
    """
    Register !command handlers contributed by a plugin.

    Args:
        plugin_name: Plugin identifier (for logging)
        commands: Dict mapping command_name -> async handler(args: str) -> str
        help_text: Optional help section string from plugin.get_help()
    """
    global _PLUGIN_COMMANDS, _PLUGIN_HELP
    _PLUGIN_COMMANDS.update(commands)
    if help_text:
        _PLUGIN_HELP[plugin_name] = help_text
    log.info(f"Registered {len(commands)} command(s) from {plugin_name}: {list(commands.keys())}")


def get_plugin_command(cmd_name: str):
    """Return the handler for a plugin-registered command, or None if not found."""
    return _PLUGIN_COMMANDS.get(cmd_name)


def get_plugin_help_sections() -> list[str]:
    """Return all plugin help section strings, in registration order."""
    return list(_PLUGIN_HELP.values())


def register_plugin_tools(plugin_name: str, tool_defs: dict):
    """
    Register tools from a plugin.

    Expected format: {'lc': [StructuredTool, ...]}

    Executors are extracted automatically from the coroutine attribute of each
    StructuredTool, so no 'executors' key is needed.
    """
    global _PLUGIN_TOOLS_LC, _PLUGIN_TOOL_EXECUTORS

    lc_tools = tool_defs.get('lc', [])
    _PLUGIN_TOOLS_LC.extend(lc_tools)
    for lc_tool in lc_tools:
        if lc_tool.coroutine and lc_tool.name not in _PLUGIN_TOOL_EXECUTORS:
            _PLUGIN_TOOL_EXECUTORS[lc_tool.name] = lc_tool.coroutine
    log.info(f"Registered {len(lc_tools)} LC tools from {plugin_name}")


def get_section_for_tool(tool_name: str) -> str:
    """
    Return the system prompt section body for a named tool.
    Tries 'tool-<name>' (hyphenated) and 'tool_<name>' (underscored) variants.
    Used by llm_call(mode='tool') to build the target model's system prompt.
    """
    # Try hyphenated form first (canonical: tool-url-extract)
    hyphenated = "tool-" + tool_name.replace("_", "-")
    section = get_section(hyphenated)
    if section:
        return section
    # Try underscore form (tool_url_extract)
    underscored = "tool_" + tool_name
    section = get_section(underscored)
    if section:
        return section
    return ""


def get_openai_tool_schema(tool_name: str) -> dict | None:
    """
    Return the OpenAI tool schema dict for a named tool.
    Searches core and plugin LC tools, converting on the fly.
    """
    for lc_tool in CORE_LC_TOOLS + _PLUGIN_TOOLS_LC:
        if lc_tool.name == tool_name:
            return _lc_tool_to_openai_dict(lc_tool)
    return None


def get_core_tools():
    """Return core (always-enabled) tool definitions."""
    import agents as _agents
    return {
        'lc': CORE_LC_TOOLS,
        'executors': {
            'get_system_info':       get_system_info,
            'llm_call':              _agents.llm_call,
            'llm_list':              _agents.llm_list,
            'agent_call':            _agents.agent_call,
            'session':               _session_exec,
            'model':                 _model_exec,
            'reset':                 _reset_exec,
            'help':                  _help_exec,
            'sleep':                 _sleep_exec,
            'llm_tools':             _llm_tools_exec,
            'model_cfg':             _model_cfg_exec,
            'sysprompt_cfg':         _sysprompt_cfg_exec,
            'config_cfg':            _config_cfg_exec,
            'limits_cfg':            _limits_cfg_exec,
        }
    }


def get_all_lc_tools() -> list:
    """Get all LangChain StructuredTool objects (core + plugins)."""
    return list(CORE_LC_TOOLS) + list(_PLUGIN_TOOLS_LC)


def get_all_openai_tools() -> list:
    """
    Get all tool definitions in OpenAI dict format (core + plugins).

    Used by try_force_tool_calls() for tool name validation and by
    get_openai_tool_schema() for llm_call(mode='tool'). Derived from CORE_LC_TOOLS
    and _PLUGIN_TOOLS_LC so there is a single source of truth.
    """
    return [_lc_tool_to_openai_dict(t) for t in CORE_LC_TOOLS + _PLUGIN_TOOLS_LC]


# ---------------------------------------------------------------------------
# LangChain StructuredTool ↔ OpenAI dict helpers
# ---------------------------------------------------------------------------

def _lc_tool_to_openai_dict(tool: StructuredTool) -> dict:
    """
    Convert a LangChain StructuredTool to the OpenAI function-calling dict format.

    Used by get_all_openai_tools() so try_force_tool_calls() and
    get_openai_tool_schema() have a single source of truth (CORE_LC_TOOLS).
    """
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {"type": "object", "properties": {}, "required": []}
    # Pydantic v2 puts $defs at top level — flatten for OpenAI compat
    schema.pop("$defs", None)
    schema.pop("title", None)
    # Strip 'title' from each property — Pydantic adds these but they confuse local models
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        },
    }


# Tool type mapping for core (always-enabled) tools.
# Used by execute_tool() for universal rate limiting.
_CORE_TOOL_TYPES: dict[str, str] = {
    "get_system_info":       "system",
    "llm_call":              "llm_call",
    "llm_list":              "system",
    "agent_call":            "agent_call",
    "session":               "system",
    "model":                 "system",
    "reset":                 "system",
    "help":                  "system",
    "sleep":                 "system",
    "llm_tools":             "system",
    "model_cfg":             "system",
    "sysprompt_cfg":         "system",
    "config_cfg":            "system",
    "limits_cfg":            "system",
}


def get_tool_type(tool_name: str) -> str:
    """
    Return the tool type string for rate-limit bucketing.

    Core tools use _CORE_TOOL_TYPES.
    Falls back to 'system' (unlimited by default) for unknown tools.
    """
    return _CORE_TOOL_TYPES.get(tool_name, "system")


def get_tool_executor(tool_name: str):
    """Get executor function for a tool."""
    # Lazy import to avoid circular dependency (agents imports tools, tools needs agents funcs)
    import agents as _agents

    core_executors = {
        'get_system_info':       get_system_info,
        'llm_call':              _agents.llm_call,
        'llm_list':              _agents.llm_list,
        'agent_call':            _agents.agent_call,
        'session':               _session_exec,
        'model':                 _model_exec,
        'reset':                 _reset_exec,
        'help':                  _help_exec,
        'sleep':                 _sleep_exec,
        'llm_tools':             _llm_tools_exec,
        'model_cfg':             _model_cfg_exec,
        'sysprompt_cfg':         _sysprompt_cfg_exec,
        'config_cfg':            _config_cfg_exec,
        'limits_cfg':            _limits_cfg_exec,
        'memory_save':           _memory_save_exec,
        'memory_recall':         _memory_recall_exec,
        'memory_update':         _memory_update_exec,
        'memory_age':            _memory_age_exec,
    }

    if tool_name in core_executors:
        return core_executors[tool_name]

    # Check plugin tools
    return _PLUGIN_TOOL_EXECUTORS.get(tool_name)


# ---------------------------------------------------------------------------
# Core Tool Definitions — LangChain StructuredTool (single source of truth)
#
# Descriptions are the LLM-facing text and are preserved verbatim from the
# former CORE_OPENAI_TOOLS / CORE_GEMINI_TOOL dual-format definitions.
# get_all_openai_tools() derives OpenAI dicts from these on the fly via
# _lc_tool_to_openai_dict() so there is no longer a second copy to maintain.
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import Literal


class _GetSystemInfoArgs(BaseModel):
    pass  # No arguments — explicit schema prevents LangChain from leaking the docstring into parameters


class _LlmCallArgs(BaseModel):
    model: str = Field(description="Target model key. Use llm_list() to see valid names.")
    prompt: str = Field(description="The prompt / user message to send to the target model.")
    mode: str = Field(
        default="text",
        description=(
            "'text' — return the target model's raw text response (default). "
            "'tool' — delegate a single tool call; requires the 'tool' argument."
        ),
    )
    sys_prompt: str = Field(
        default="none",
        description=(
            "System prompt to send to the target model. "
            "'none'   — no system prompt (clean call). "
            "'caller' — inject the calling model's current assembled system prompt. "
            "'target' — load the target model's own system_prompt_folder data."
        ),
    )
    history: str = Field(
        default="none",
        description=(
            "Chat history to prepend. "
            "'none'   — no history; clean single-turn call. "
            "'caller' — prepend the calling session's full chat history."
        ),
    )
    tool: str = Field(
        default="",
        description="Exact tool name to delegate. Required when mode='tool'.",
    )


# ---------------------------------------------------------------------------
# Session / model / reset / help tool arg schemas
# ---------------------------------------------------------------------------

class _SessionArgs(BaseModel):
    action: Literal["list", "delete"] = Field(
        description="'list' to show all sessions, 'delete' to remove a session."
    )
    session_id: str = Field(
        default="",
        description="Session shorthand ID (e.g., '101') or full session ID. Required for 'delete'.",
    )


class _ModelArgs(BaseModel):
    action: Literal["list", "set"] = Field(
        description="'list' to show available models, 'set' to switch the active model."
    )
    model_key: str = Field(
        default="",
        description="Model key to switch to. Required for 'set' action.",
    )


class _ResetArgs(BaseModel):
    pass  # No arguments


class _HelpArgs(BaseModel):
    pass  # No arguments


class _AgentCallArgs(BaseModel):
    agent_url: str = Field(
        description="Base URL of the target agent-mcp instance, e.g. 'http://localhost:8767'. "
                    "The target must have the API client plugin (plugin_client_api) enabled."
    )
    message: str = Field(
        description=(
            "The message or command to send to the target agent. "
            "Can be any text, !command, or @model prefix. "
            "The full response from the target agent is returned.\n\n"
            "CRITICAL: You are the ORCHESTRATOR. Send ONE direct conversational prompt per call. "
            "The remote agent only RESPONDS — it cannot itself call agent_call (depth guard blocks it). "
            "NEVER embed multi-turn instructions in the message (e.g. 'have a 3-turn conversation with me'). "
            "That causes Max swarm depth errors. "
            "For N-turn conversations: make N separate agent_call invocations, each with a single question."
        )
    )
    target_client_id: str = Field(
        default="",
        description="Optional: session name to use on the target agent. "
                    "Omit to auto-generate an isolated swarm session."
    )
    stream: bool = Field(
        default=True,
        description="If True (default), relay the remote agent's tokens in real-time as they "
                    "arrive so Slack and other clients see per-turn progress. "
                    "Set to False to suppress streaming and return only the final result — "
                    "useful when the intermediate tokens would be noisy or are not needed."
    )


# ---------------------------------------------------------------------------
# Session / model / reset / help tool executors
# ---------------------------------------------------------------------------

async def _session_exec(action: str, session_id: str = "") -> str:
    from state import sessions, get_or_create_shorthand_id, get_session_by_shorthand, remove_shorthand_mapping, current_client_id
    cid = current_client_id.get("")

    if action == "list":
        if not sessions:
            return "No active sessions."
        lines = ["Active sessions:"]
        for sid, data in sessions.items():
            marker = " (current)" if sid == cid else ""
            model = data.get("model", "unknown")
            history_len = len(data.get("history", []))
            shorthand_id = get_or_create_shorthand_id(sid)
            peer_ip = data.get("peer_ip")
            ip_str = f", ip={peer_ip}" if peer_ip else ""
            lines.append(f"  ID [{shorthand_id}] {sid}: model={model}, history={history_len} messages{ip_str}{marker}")
        return "\n".join(lines)

    if action == "delete":
        if not session_id:
            return "ERROR: session_id required for 'delete' action."
        # Try shorthand ID
        target_sid = None
        try:
            shorthand_id = int(session_id)
            target_sid = get_session_by_shorthand(shorthand_id)
            if not target_sid:
                return f"Session ID [{shorthand_id}] not found."
        except ValueError:
            target_sid = session_id
        if target_sid in sessions:
            shorthand_id = get_or_create_shorthand_id(target_sid)
            del sessions[target_sid]
            remove_shorthand_mapping(target_sid)
            return f"Deleted session ID [{shorthand_id}]: {target_sid}"
        return f"Session not found: {target_sid}"

    return f"Unknown action '{action}'. Valid: list, delete"


async def _model_exec(action: str, model_key: str = "") -> str:
    from config import LLM_REGISTRY
    from state import current_client_id, sessions, cancel_active_task

    if action == "list":
        lines = ["Available models:"]
        cid = current_client_id.get("")
        current = sessions.get(cid, {}).get("model", "") if cid else ""
        for key, meta in LLM_REGISTRY.items():
            model_id = meta.get("model_id", key)
            marker = " (current)" if key == current else ""
            lines.append(f"  {key:<12} {model_id}{marker}")
        return "\n".join(lines)

    if action == "set":
        if not model_key:
            return "ERROR: model_key required for 'set' action."
        cid = current_client_id.get("")
        if not cid:
            return "ERROR: No active session context for model switch."
        if model_key not in LLM_REGISTRY:
            available = ", ".join(LLM_REGISTRY.keys())
            return f"ERROR: Unknown model '{model_key}'\nAvailable: {available}"
        await cancel_active_task(cid)
        sessions[cid]["model"] = model_key
        return f"Model set to '{model_key}'."

    return f"Unknown action '{action}'. Valid: list, set"


async def _reset_exec() -> str:
    from state import current_client_id, sessions, delete_history
    cid = current_client_id.get("")
    if not cid or cid not in sessions:
        return "ERROR: No active session context."
    history_len = len(sessions[cid].get("history", []))
    sessions[cid]["history"] = []
    delete_history(cid)
    return f"Conversation history cleared ({history_len} messages removed)."


async def _help_exec() -> str:
    return (
        "Available commands: Use !help in the chat interface for full command list.\n"
        "Key tool calls (LLM-invocable):\n"
        "  llm_tools(action, ...)        - manage named toolsets (list/read/write/delete/add)\n"
        "  model_cfg(action, ...)        - unified model management (list/read/write/copy/delete/enable/disable)\n"
        "  sysprompt_cfg(action, ...)    - unified system prompt management (list_dir/list/read/write/delete/copy_dir/set_dir)\n"
        "  config_cfg(action, ...)       - manage session/server config (stream/tool_preview_length/tool_suppress/default_model)\n"
        "  limits_cfg(action, ...)       - manage depth/iteration/rate limits\n"
        "  session(action, ...)          - list or delete sessions\n"
        "  model(action, ...)            - list or switch models\n"
        "  reset()                       - clear conversation history\n"
        "  db_query(sql)                 - run SQL\n"
        "  search_ddgs/search_tavily/search_xai/search_google(query) - web search\n"
        "  url_extract(method, url)      - extract web page content\n"
        "  google_drive(operation, ...) - Google Drive CRUD\n"
        "  get_system_info()             - date/time/status\n"
        "  llm_list()                    - list LLM models\n"
        "  llm_call(model, prompt, ...)  - call LLM (mode/sys_prompt/history/tool params)\n"
        "  agent_call(agent_url, message) - call remote agent\n"
        "  sleep(seconds)                - sleep 1-300 seconds\n"
    )


class _SleepArgs(BaseModel):
    seconds: int = Field(description="Number of seconds to sleep (1–300).")


async def _sleep_exec(seconds: int) -> str:
    """Sleep for the given number of seconds (1–300)."""
    import asyncio as _asyncio
    if not isinstance(seconds, int) or seconds < 1:
        return "ERROR: seconds must be a positive integer."
    if seconds > 300:
        return "ERROR: maximum sleep is 300 seconds."
    await _asyncio.sleep(seconds)
    return f"Slept for {seconds} second(s)."


# ---------------------------------------------------------------------------
# Memory tools: memory_save, memory_recall, memory_age
# ---------------------------------------------------------------------------

class _MemorySaveArgs(BaseModel):
    topic: str = Field(description="Short topic label, e.g. 'user-preferences', 'project-status', 'tasks'.")
    content: str = Field(description="One concise sentence describing the fact to remember.")
    importance: int = Field(default=5, description="Importance 1 (low) to 10 (critical). Default 5.")
    source: str = Field(default="user", description="Source: 'user', 'session', or 'directive'.")

class _MemoryRecallArgs(BaseModel):
    topic: str = Field(default="", description="Keyword to search — matches topic label OR content text. Use words from the 'Known topics' list in Active Memory when possible.")
    tier: str = Field(default="short", description="'short' (default, searches topic+content): active memories. 'long': aged-out facts (also searches content).")
    limit: int = Field(default=20, description="Max rows to return.")

class _MemoryUpdateArgs(BaseModel):
    id: int = Field(description="Row id to update (from memory_recall or !memory list).")
    tier: str = Field(default="short", description="'short' (default) or 'long'.")
    importance: int = Field(default=0, description="New importance 1-10. 0 = leave unchanged.")
    content: str = Field(default="", description="New content text. Empty = leave unchanged.")
    topic: str = Field(default="", description="New topic label. Empty = leave unchanged.")

class _MemoryAgeArgs(BaseModel):
    older_than_hours: int = Field(default=48, description="Move rows older than this many hours to long-term.")
    max_rows: int = Field(default=100, description="Max rows to age per call.")


async def _memory_save_exec(topic: str, content: str, importance: int = 5, source: str = "user") -> str:
    from memory import save_memory
    from state import current_client_id
    session_id = current_client_id.get("") or ""
    row_id = await save_memory(
        topic=topic, content=content,
        importance=importance, source=source,
        session_id=session_id,
    )
    if row_id == 0:
        return f"Memory duplicate skipped (already in memory): [{topic}] {content}"
    return f"Memory saved (id={row_id}): [{topic}] {content} (importance={importance})"


async def _memory_recall_exec(topic: str = "", tier: str = "short", limit: int = 20) -> str:
    from memory import load_short_term, load_long_term, _parse_table
    if tier == "long":
        rows = await load_long_term(limit=limit, topic=topic)
    else:
        rows = await load_short_term(limit=limit, min_importance=1)
        if topic:
            rows = [r for r in rows if topic.lower() in r.get("topic", "").lower() or topic.lower() in r.get("content", "").lower()]
    if not rows:
        return f"No memories found (tier={tier}, topic='{topic}')."
    lines = [f"Memories (tier={tier}, {len(rows)} rows):"]
    for r in rows:
        lines.append(f"  [{r.get('topic','')}] imp={r.get('importance','')} — {r.get('content','')}")
    return "\n".join(lines)


async def _memory_age_exec(older_than_hours: int = 48, max_rows: int = 100) -> str:
    from memory import age_to_longterm
    moved = await age_to_longterm(older_than_hours=older_than_hours, max_rows=max_rows)
    return f"Aged {moved} memories from short-term to long-term (threshold: {older_than_hours}h)."


async def _memory_update_exec(id: int, tier: str = "short", importance: int = 0,
                               content: str = "", topic: str = "") -> str:
    from memory import update_memory
    return await update_memory(
        row_id=id,
        tier=tier,
        importance=importance if importance > 0 else None,
        content=content if content else None,
        topic=topic if topic else None,
    )


# ---------------------------------------------------------------------------
# Unified resource tools: llm_tools, model_cfg, sysprompt_cfg, config_cfg, limits_cfg
# These consolidate multiple individual tools into single CRUD-style resources.
# ---------------------------------------------------------------------------

class _LlmToolsArgs(BaseModel):
    action: Literal["list", "read", "write", "delete", "add"] = Field(
        description=(
            "'list' — show all toolset names and their tool lists.\n"
            "'read' — show a single toolset (requires name).\n"
            "'write' — overwrite a toolset (requires name and tools).\n"
            "'delete' — remove a toolset (requires name).\n"
            "'add' — add tool names to an existing toolset (requires name and tools)."
        )
    )
    name: str = Field(
        default="",
        description="Toolset name (e.g. 'core', 'db', 'search'). Required for read/write/delete/add.",
    )
    tools: str = Field(
        default="",
        description="Comma-separated tool names for write/add actions (e.g. 'db_query,search_ddgs').",
    )


async def _llm_tools_exec(action: str, name: str = "", tools: str = "") -> str:
    from config import LLM_TOOLSETS, LLM_REGISTRY, save_llm_toolset, delete_llm_toolset

    if action == "list":
        if not LLM_TOOLSETS:
            return "No toolsets defined in llm-tools.json."
        lines = ["Toolsets (llm-tools.json):"]
        for ts_name in sorted(LLM_TOOLSETS.keys()):
            tool_list = LLM_TOOLSETS[ts_name]
            lines.append(f"  {ts_name}: {', '.join(tool_list)} ({len(tool_list)} tools)")
        # Also show per-model assignments
        lines.append("\nModel → toolsets:")
        for model_name in sorted(LLM_REGISTRY.keys()):
            ts = LLM_REGISTRY[model_name].get("llm_tools", [])
            lines.append(f"  {model_name}: {', '.join(ts) if ts else '(none)'}")
        return "\n".join(lines)

    if action == "read":
        if not name:
            return "ERROR: 'name' required for action='read'."
        ts = LLM_TOOLSETS.get(name)
        if ts is None:
            return f"ERROR: Toolset '{name}' not found. Use llm_tools(action='list') to see available toolsets."
        return f"Toolset '{name}': {', '.join(ts)} ({len(ts)} tools)"

    if action == "write":
        if not name:
            return "ERROR: 'name' required for action='write'."
        if not tools:
            return "ERROR: 'tools' required for action='write'. Provide comma-separated tool names."
        tool_list = [t.strip() for t in tools.split(",") if t.strip()]
        if save_llm_toolset(name, tool_list):
            LLM_TOOLSETS[name] = tool_list
            return f"Toolset '{name}' written: {', '.join(tool_list)} ({len(tool_list)} tools). Persisted to llm-tools.json."
        return f"ERROR: Failed to save toolset '{name}'."

    if action == "delete":
        if not name:
            return "ERROR: 'name' required for action='delete'."
        ok, msg = delete_llm_toolset(name)
        if ok:
            LLM_TOOLSETS.pop(name, None)
        return msg

    if action == "add":
        if not name:
            return "ERROR: 'name' required for action='add'."
        if not tools:
            return "ERROR: 'tools' required for action='add'. Provide comma-separated tool names to add."
        existing = LLM_TOOLSETS.get(name, [])
        new_tools = [t.strip() for t in tools.split(",") if t.strip()]
        merged = list(dict.fromkeys(existing + new_tools))  # preserve order, dedupe
        if save_llm_toolset(name, merged):
            LLM_TOOLSETS[name] = merged
            added = [t for t in new_tools if t not in existing]
            return f"Toolset '{name}' updated: added {', '.join(added) if added else '(no new tools)'}. Now {len(merged)} tools."
        return f"ERROR: Failed to update toolset '{name}'."

    return f"Unknown action '{action}'. Valid: list, read, write, delete, add"


class _ModelCfgArgs(BaseModel):
    action: Literal["list", "read", "write", "copy", "delete", "enable", "disable"] = Field(
        description=(
            "'list' — show all models.\n"
            "'read' — show full config for a model (requires name).\n"
            "'write' — set a model field (requires name, field, value).\n"
            "'copy' — copy a model (requires name=source, value=new_name).\n"
            "'delete' — delete a model (requires name).\n"
            "'enable' — enable a disabled model (requires name).\n"
            "'disable' — disable a model (requires name)."
        )
    )
    name: str = Field(default="", description="Model key (e.g. 'gemini25f').")
    field: str = Field(
        default="",
        description=(
            "Field to update for action='write'. "
            "Valid fields: llm_tools, llm_tools_gates, llm_call_timeout, "
            "temperature, top_p, top_k, token_selection_setting, system_prompt_folder, max_context."
        ),
    )
    value: str = Field(default="", description="New value as string. For llm_tools, comma-separated toolset names.")


async def _model_cfg_exec(action: str, name: str = "", field: str = "", value: str = "") -> str:
    from config import (
        LLM_REGISTRY, save_llm_model_field, copy_llm_model, delete_llm_model,
        enable_llm_model, disable_llm_model,
    )
    from state import current_client_id, sessions, cancel_active_task

    if action == "list":
        if not LLM_REGISTRY:
            return "No models registered."
        lines = ["Registered models:"]
        cid = current_client_id.get("")
        current = sessions.get(cid, {}).get("model", "") if cid else ""
        for key in sorted(LLM_REGISTRY.keys()):
            cfg = LLM_REGISTRY[key]
            marker = " (current)" if key == current else ""
            ts = cfg.get("llm_tools", [])
            lines.append(
                f"  {key:<16} {cfg.get('model_id',''):<30} "
                f"tools=[{','.join(ts)}]{marker}"
            )
        return "\n".join(lines)

    if action == "read":
        if not name:
            return "ERROR: 'name' required for action='read'."
        cfg = LLM_REGISTRY.get(name)
        if not cfg:
            return f"ERROR: Model '{name}' not found. Available: {', '.join(sorted(LLM_REGISTRY.keys()))}"
        lines = [f"Model: {name}"]
        for k, v in sorted(cfg.items()):
            if k == "key":
                lines.append(f"  {k}: {'***' if v else '(none)'}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    if action == "write":
        if not name:
            return "ERROR: 'name' required for action='write'."
        if not field:
            return "ERROR: 'field' required for action='write'."
        if name not in LLM_REGISTRY:
            return f"ERROR: Model '{name}' not found."

        # Type coercion for known fields
        coerced_value = value
        if field in ("llm_tools", "llm_tools_gates"):
            # Accept comma-separated list; empty string clears the list
            coerced_value = [t.strip() for t in value.split(",") if t.strip()]
        elif field in ("llm_call_timeout", "max_context"):
            try:
                coerced_value = int(value)
            except ValueError:
                return f"ERROR: '{field}' must be an integer."
        elif field in ("temperature", "top_p"):
            try:
                coerced_value = float(value)
            except ValueError:
                return f"ERROR: '{field}' must be a number."
        elif field == "top_k":
            if value.lower() == "null" or value.lower() == "none":
                coerced_value = None
            else:
                try:
                    coerced_value = int(value)
                except ValueError:
                    return f"ERROR: 'top_k' must be an integer or null."

        old = LLM_REGISTRY[name].get(field, "(unset)")
        LLM_REGISTRY[name][field] = coerced_value
        if save_llm_model_field(name, field, coerced_value):
            return f"{name}.{field}: {old} → {coerced_value} (persisted to llm-models.json)"
        return f"{name}.{field}: {old} → {coerced_value} (runtime only — JSON write failed)"

    if action == "copy":
        if not name:
            return "ERROR: 'name' (source model) required for action='copy'."
        if not value:
            return "ERROR: 'value' (new model name) required for action='copy'."
        ok, msg = copy_llm_model(name, value)
        return msg

    if action == "delete":
        if not name:
            return "ERROR: 'name' required for action='delete'."
        cid = current_client_id.get("")
        if cid and sessions.get(cid, {}).get("model") == name:
            return f"ERROR: Cannot delete '{name}' — it is the active model. Switch first."
        ok, msg = delete_llm_model(name)
        return msg

    if action == "enable":
        if not name:
            return "ERROR: 'name' required for action='enable'."
        ok, msg = enable_llm_model(name)
        return msg

    if action == "disable":
        if not name:
            return "ERROR: 'name' required for action='disable'."
        ok, msg = disable_llm_model(name)
        return msg

    return f"Unknown action '{action}'. Valid: list, read, write, copy, delete, enable, disable"


class _SyspromptCfgArgs(BaseModel):
    action: Literal["list_dir", "list", "read", "write", "delete", "copy_dir", "set_dir"] = Field(
        description=(
            "'list_dir' — list all system prompt directories.\n"
            "'list' — list files in a model's prompt dir (requires model).\n"
            "'read' — read a prompt file or full assembled prompt (requires model; file optional).\n"
            "'write' — write a prompt file (requires model, file, content).\n"
            "'delete' — delete a file or entire dir (requires model; file optional).\n"
            "'copy_dir' — copy prompt dir to new name (requires model, newdir).\n"
            "'set_dir' — assign prompt dir to model (requires model, newdir)."
        )
    )
    model: str = Field(default="", description="Model key or 'self' for current model.")
    file: str = Field(default="", description="Section name or filename (e.g. 'behavior', '.system_prompt').")
    content: str = Field(default="", description="Content to write (for action='write').")
    newdir: str = Field(default="", description="New directory name (for copy_dir/set_dir).")


async def _sysprompt_cfg_exec(
    action: str, model: str = "", file: str = "", content: str = "", newdir: str = ""
) -> str:
    from prompt import (
        sp_list_directories, sp_list_files, sp_read_prompt, sp_read_file,
        sp_write_file, sp_delete_file, sp_delete_directory,
        sp_copy_directory, sp_set_directory, sp_resolve_model,
    )
    from config import LLM_REGISTRY
    from state import current_client_id, sessions

    cid = current_client_id.get("")
    current_model = sessions.get(cid, {}).get("model", "") if cid else ""

    if action == "list_dir":
        return sp_list_directories()

    if action == "list":
        if not model:
            return "ERROR: 'model' required for action='list'."
        resolved = sp_resolve_model(model, current_model)
        return sp_list_files(resolved, LLM_REGISTRY)

    if action == "read":
        if not model:
            return "ERROR: 'model' required for action='read'."
        resolved = sp_resolve_model(model, current_model)
        if file:
            return sp_read_file(resolved, file, LLM_REGISTRY)
        return sp_read_prompt(resolved, LLM_REGISTRY)

    if action == "write":
        if not model:
            return "ERROR: 'model' required for action='write'."
        if not file:
            return "ERROR: 'file' required for action='write'."
        if not content:
            return "ERROR: 'content' required for action='write'."
        resolved = sp_resolve_model(model, current_model)
        return sp_write_file(resolved, file, content, LLM_REGISTRY)

    if action == "delete":
        if not model:
            return "ERROR: 'model' required for action='delete'."
        resolved = sp_resolve_model(model, current_model)
        if file:
            return sp_delete_file(resolved, file, LLM_REGISTRY)
        return sp_delete_directory(resolved, LLM_REGISTRY)

    if action == "copy_dir":
        if not model:
            return "ERROR: 'model' required for action='copy_dir'."
        if not newdir:
            return "ERROR: 'newdir' required for action='copy_dir'."
        resolved = sp_resolve_model(model, current_model)
        return sp_copy_directory(resolved, newdir, LLM_REGISTRY)

    if action == "set_dir":
        if not model:
            return "ERROR: 'model' required for action='set_dir'."
        if not newdir:
            return "ERROR: 'newdir' required for action='set_dir'."
        resolved = sp_resolve_model(model, current_model)
        return sp_set_directory(resolved, newdir)

    return f"Unknown action '{action}'. Valid: list_dir, list, read, write, delete, copy_dir, set_dir"


class _ConfigCfgArgs(BaseModel):
    action: Literal["list", "read", "write"] = Field(
        description=(
            "'list' — show all session config keys and values.\n"
            "'read' — show a single config key (requires key).\n"
            "'write' — set a config key (requires key and value)."
        )
    )
    key: str = Field(
        default="",
        description=(
            "Config key. Valid keys: stream, tool_preview_length, tool_suppress, "
            "default_model, outbound_agent_filters."
        ),
    )
    value: str = Field(default="", description="New value as string.")


async def _config_cfg_exec(action: str, key: str = "", value: str = "") -> str:
    from state import current_client_id, sessions
    from config import DEFAULT_MODEL, save_default_model, LLM_REGISTRY
    import config as _config_mod

    cid = current_client_id.get("")
    sess = sessions.get(cid, {}) if cid else {}

    _SESSION_KEYS = {
        "stream":              ("agent_call_stream",  "bool",  True),
        "tool_preview_length": ("tool_preview_length", "int",  500),
        "tool_suppress":       ("tool_suppress",       "bool", False),
    }

    if action == "list":
        lines = ["Session config:"]
        for display_key, (sess_key, typ, default) in _SESSION_KEYS.items():
            val = sess.get(sess_key, default)
            lines.append(f"  {display_key}: {val}")
        lines.append(f"\nServer config:")
        lines.append(f"  default_model: {_config_mod.DEFAULT_MODEL}")
        return "\n".join(lines)

    if action == "read":
        if not key:
            return "ERROR: 'key' required for action='read'."
        if key in _SESSION_KEYS:
            sess_key, typ, default = _SESSION_KEYS[key]
            val = sess.get(sess_key, default)
            return f"{key}: {val}"
        if key == "default_model":
            return f"default_model: {_config_mod.DEFAULT_MODEL}"
        if key == "outbound_agent_filters":
            from agents import _outbound_agent_allowed, _outbound_agent_blocked
            return f"allowed: {_outbound_agent_allowed}\nblocked: {_outbound_agent_blocked}"
        return f"ERROR: Unknown config key '{key}'."

    if action == "write":
        if not key:
            return "ERROR: 'key' required for action='write'."
        if not value and key != "tool_suppress":
            return "ERROR: 'value' required for action='write'."

        if key in _SESSION_KEYS:
            sess_key, typ, default = _SESSION_KEYS[key]
            if not cid or cid not in sessions:
                return "ERROR: No active session context."
            if typ == "bool":
                coerced = value.lower() in ("true", "1", "yes")
            elif typ == "int":
                try:
                    coerced = int(value)
                except ValueError:
                    return f"ERROR: '{key}' must be an integer."
            else:
                coerced = value
            old = sess.get(sess_key, default)
            sessions[cid][sess_key] = coerced
            from state import save_session_config
            save_session_config(cid, sessions[cid])
            return f"{key}: {old} → {coerced}"

        if key == "default_model":
            if value not in LLM_REGISTRY:
                return f"ERROR: Model '{value}' not found."
            old = _config_mod.DEFAULT_MODEL
            _config_mod.DEFAULT_MODEL = value
            save_default_model(value)
            return f"default_model: {old} → {value} (persisted)"

        return f"ERROR: Unknown config key '{key}'."

    return f"Unknown action '{action}'. Valid: list, read, write"


class _LimitsCfgArgs(BaseModel):
    action: Literal["list", "read", "write"] = Field(
        description=(
            "'list' — show all limits.\n"
            "'read' — show a single limit (requires key).\n"
            "'write' — set a limit (requires key and value)."
        )
    )
    key: str = Field(
        default="",
        description=(
            "Limit key. Valid keys: max_at_llm_depth, max_agent_call_depth, "
            "max_tool_iterations, rate_<type>_calls, rate_<type>_window."
        ),
    )
    value: str = Field(default="", description="New integer value.")


async def _limits_cfg_exec(action: str, key: str = "", value: str = "") -> str:
    from config import LIVE_LIMITS, RATE_LIMITS, save_limit_field, save_rate_limit

    _DEPTH_KEYS = {
        "max_at_llm_depth": "Max nested llm_call(history=caller) hops",
        "max_agent_call_depth": "Max nested agent_call hops",
        "max_tool_iterations": "Max LLM↔tool round-trips per request",
    }

    if action == "list":
        lines = ["Depth / iteration limits:"]
        for k, desc in sorted(_DEPTH_KEYS.items()):
            val = LIVE_LIMITS.get(k, 1)
            lines.append(f"  {k}: {val} — {desc}")
        lines.append("\nRate limits:")
        for tool_type in sorted(RATE_LIMITS.keys()):
            cfg = RATE_LIMITS[tool_type]
            calls = cfg.get("calls", 0)
            window = cfg.get("window_seconds", 0)
            auto = cfg.get("auto_disable", False)
            lines.append(
                f"  {tool_type}: {calls} calls / {window}s"
                f"{' (auto-disable)' if auto else ''}"
            )
        return "\n".join(lines)

    if action == "read":
        if not key:
            return "ERROR: 'key' required for action='read'."
        if key in _DEPTH_KEYS:
            val = LIVE_LIMITS.get(key, 1)
            return f"{key}: {val} — {_DEPTH_KEYS[key]}"
        # Rate limit keys: rate_<type>_calls or rate_<type>_window
        if key.startswith("rate_"):
            parts = key.split("_")
            if len(parts) >= 3 and parts[-1] in ("calls", "window"):
                tool_type = "_".join(parts[1:-1])
                field = parts[-1] if parts[-1] == "calls" else "window_seconds"
                cfg = RATE_LIMITS.get(tool_type, {})
                val = cfg.get(field, 0)
                return f"{key}: {val}"
        return f"ERROR: Unknown limit key '{key}'."

    if action == "write":
        if not key:
            return "ERROR: 'key' required for action='write'."
        if not value:
            return "ERROR: 'value' required for action='write'."
        try:
            int_val = int(value)
        except ValueError:
            return "ERROR: value must be an integer."

        if key in _DEPTH_KEYS:
            if int_val < 0:
                return "ERROR: Value must be >= 0."
            old = LIVE_LIMITS.get(key, 1)
            LIVE_LIMITS[key] = int_val
            save_limit_field(key, int_val)
            return f"{key}: {old} → {int_val} (persisted)"

        # Rate limit keys
        if key.startswith("rate_"):
            parts = key.split("_")
            if len(parts) >= 3 and parts[-1] in ("calls", "window"):
                tool_type = "_".join(parts[1:-1])
                field = parts[-1] if parts[-1] == "calls" else "window_seconds"
                if tool_type not in RATE_LIMITS:
                    RATE_LIMITS[tool_type] = {}
                old = RATE_LIMITS[tool_type].get(field, 0)
                RATE_LIMITS[tool_type][field] = int_val
                save_rate_limit(tool_type, field, int_val)
                return f"{key}: {old} → {int_val} (persisted)"

        return f"ERROR: Unknown limit key '{key}'."

    return f"Unknown action '{action}'. Valid: list, read, write"


def _make_core_lc_tools() -> list:
    """Build CORE_LC_TOOLS after agents module is available (avoids circular import)."""
    import agents as _agents
    return [
        StructuredTool.from_function(
            coroutine=get_system_info,
            name="get_system_info",
            description="Returns current local date, time, and system status.",
            args_schema=_GetSystemInfoArgs,
        ),
        StructuredTool.from_function(
            coroutine=_agents.llm_call,
            name="llm_call",
            description=(
                "Call a target LLM model with full control over system prompt and history.\n\n"
                "Parameters:\n"
                "  model      — target model key (use llm_list() to see valid names)\n"
                "  prompt     — the user message to send\n"
                "  mode       — 'text' (default): return raw text response\n"
                "               'tool': delegate a single tool call; requires 'tool' argument\n"
                "  sys_prompt — 'none' (default): no system prompt sent\n"
                "               'caller': inject the calling model's assembled system prompt\n"
                "               'target': load the target model's own system_prompt_folder\n"
                "  history    — 'none' (default): clean single-turn call\n"
                "               'caller': prepend the calling session's full chat history\n"
                "  tool       — exact tool name; required when mode='tool'\n\n"
                "Common patterns:\n"
                "  Summarize text cleanly:     mode=text sys_prompt=none  history=none\n"
                "  Delegate with full context: mode=text sys_prompt=target history=caller\n"
                "  Offload a tool call:        mode=tool sys_prompt=none  history=none  tool=url_extract\n"
                "  Expert tool call:           mode=tool sys_prompt=target history=none  tool=db_query\n\n"
                "Rate limited (default: 3 calls per 20 seconds per session)."
            ),
            args_schema=_LlmCallArgs,
        ),
        StructuredTool.from_function(
            coroutine=_agents.llm_list,
            name="llm_list",
            description=(
                "List all registered LLM models with their details: type, model_id, host, "
                "max_context, timeout, and description. "
                "Use this before calling llm_call to identify a suitable target model."
            ),
        ),
        StructuredTool.from_function(
            coroutine=_agents.agent_call,
            name="agent_call",
            description=(
                "Send a single direct message to another agent-mcp instance and return its response. "
                "Use for multi-agent coordination (swarm): delegate tasks, verify answers, or "
                "gather perspectives across agent instances.\n\n"
                "ORCHESTRATION MODEL: YOU are the orchestrator. YOU make repeated agent_call "
                "invocations — one per conversation turn. The remote agent ONLY RESPONDS to the "
                "single message you send; it does NOT itself call agent_call (depth guard blocks "
                "recursion at 1 hop). For an N-turn conversation, make N separate agent_call "
                "calls, then synthesize all responses. NEVER embed multi-turn orchestration in "
                "the message field (e.g. 'have a 5-turn conversation') — that causes immediate "
                "Max swarm depth errors.\n\n"
                "Rate limited: 5 calls per 60 seconds per session. "
                "By default (stream=True) remote tokens are relayed in real-time for live Slack progress. "
                "Set stream=False to suppress streaming and return only the final result."
            ),
            args_schema=_AgentCallArgs,
        ),
        # --- Session / model / reset / help tools ---
        StructuredTool.from_function(
            coroutine=_session_exec,
            name="session",
            description=(
                "Manage agent sessions. action='list' shows all active sessions. "
                "action='delete' removes a session (requires session_id, can be shorthand integer or full ID)."
            ),
            args_schema=_SessionArgs,
        ),
        StructuredTool.from_function(
            coroutine=_model_exec,
            name="model",
            description=(
                "Manage the active LLM model. action='list' shows all available models. "
                "action='set' switches the active model for this session (requires model_key)."
            ),
            args_schema=_ModelArgs,
        ),
        StructuredTool.from_function(
            coroutine=_reset_exec,
            name="reset",
            description=(
                "Clear conversation history for the current session."
            ),
            args_schema=_ResetArgs,
        ),
        StructuredTool.from_function(
            coroutine=_help_exec,
            name="help",
            description="Return a summary of available commands and tool calls.",
            args_schema=_HelpArgs,
        ),
        # --- Utility tools ---
        StructuredTool.from_function(
            coroutine=_sleep_exec,
            name="sleep",
            description=(
                "Sleep (pause execution) for a specified number of seconds (1–300). "
                "Useful for rate-limiting, polling loops, or staged sequences."
            ),
            args_schema=_SleepArgs,
        ),
        # --- Unified resource tools ---
        StructuredTool.from_function(
            coroutine=_llm_tools_exec,
            name="llm_tools",
            description=(
                "Manage named toolsets that control which tools each model can use. "
                "Each model's llm_tools array references toolset names (e.g. ['core','db','search']). "
                "Actions: list (all toolsets + model assignments), read (one toolset), "
                "write (overwrite toolset), delete (remove toolset), add (append tools to toolset). "
                "Changes persist to llm-tools.json."
            ),
            args_schema=_LlmToolsArgs,
        ),
        StructuredTool.from_function(
            coroutine=_model_cfg_exec,
            name="model_cfg",
            description=(
                "Unified model management: list, read, write fields, copy, delete, enable, disable. "
                "Use action='write' with field and value to set any model parameter "
                "(llm_tools, llm_tools_gates, llm_call_timeout, temperature, top_p, top_k, "
                "token_selection_setting, system_prompt_folder, max_context, etc.). "
                "llm_tools_gates: comma-separated gate entries (e.g. 'db_query,model_cfg write') "
                "that require human approval before the tool call is executed."
            ),
            args_schema=_ModelCfgArgs,
        ),
        StructuredTool.from_function(
            coroutine=_sysprompt_cfg_exec,
            name="sysprompt_cfg",
            description=(
                "Unified system prompt management. Actions: list_dir, list, read, write, delete, copy_dir, set_dir. "
                "Use model='self' for the current model."
            ),
            args_schema=_SyspromptCfgArgs,
        ),
        StructuredTool.from_function(
            coroutine=_config_cfg_exec,
            name="config_cfg",
            description=(
                "Manage session and server configuration. "
                "Actions: list (all settings), read (single key), write (set key). "
                "Session keys: stream, tool_preview_length, tool_suppress. "
                "Server keys: default_model."
            ),
            args_schema=_ConfigCfgArgs,
        ),
        StructuredTool.from_function(
            coroutine=_limits_cfg_exec,
            name="limits_cfg",
            description=(
                "Manage depth, iteration, and rate limits. "
                "Actions: list (all limits), read (single key), write (set key). "
                "Depth keys: max_at_llm_depth, max_agent_call_depth, max_tool_iterations. "
                "Rate keys: rate_<type>_calls, rate_<type>_window (e.g. rate_llm_call_calls). "
                "Changes persist to JSON."
            ),
            args_schema=_LimitsCfgArgs,
        ),
        # --- Memory tools ---
        StructuredTool.from_function(
            coroutine=_memory_save_exec,
            name="memory_save",
            description=(
                "Save a fact to short-term memory. "
                "Use topic labels like 'user-preferences', 'project-status', 'tasks', 'technical-decisions'. "
                "importance: 1=low, 5=medium, 10=critical. "
                "Short-term memories are auto-injected into every future request."
            ),
            args_schema=_MemorySaveArgs,
        ),
        StructuredTool.from_function(
            coroutine=_memory_recall_exec,
            name="memory_recall",
            description=(
                "Recall memories from storage. "
                "ALWAYS check Active Memory context block first — it is pre-injected every request. "
                "Only call this if the answer isn't already in the injected ## Active Memory block. "
                "tier='short' (default): searches topic AND content. tier='long': aged-out facts, also searches content. "
                "topic: keyword matched against both topic label and content text."
            ),
            args_schema=_MemoryRecallArgs,
        ),
        StructuredTool.from_function(
            coroutine=_memory_update_exec,
            name="memory_update",
            description=(
                "Update an existing memory row. Use to correct importance, fix content, or retopic a fact. "
                "id: row id (from memory_recall output). tier: 'short' or 'long'. "
                "importance: new value 1-10 (0 = unchanged). content/topic: new text (empty = unchanged)."
            ),
            args_schema=_MemoryUpdateArgs,
        ),
        StructuredTool.from_function(
            coroutine=_memory_age_exec,
            name="memory_age",
            description=(
                "Age old short-term memories into long-term storage. "
                "Moves rows older than older_than_hours to long-term memory storage. "
                "Run periodically to keep short-term context lean."
            ),
            args_schema=_MemoryAgeArgs,
        ),
    ]


# Populated by agent-mcp.py after plugin registration via update_tool_definitions()
CORE_LC_TOOLS: list = []


def get_all_gate_tools() -> dict:
    """
    Return a merged dict of all gate-declared tools from all loaded plugins.
    Each entry: tool_name -> {type, operations, description}
    """
    try:
        from plugin_loader import loaded_plugins
        result = {}
        for plugin in loaded_plugins:
            try:
                result.update(plugin.get_gate_tools())
            except Exception:
                pass
        return result
    except Exception:
        return {}


def get_gate_tools_by_type(gate_type: str) -> set:
    """Return the set of tool names that have the given gate type (e.g. 'tmux', 'search')."""
    return {name for name, meta in get_all_gate_tools().items() if meta.get("type") == gate_type}
