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
    from config import now_display, display_tz_label
    _now = now_display()
    result: dict = {
        "local_time": _now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": display_tz_label(),
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


def get_section_for_tool(tool_name: str, folder: str | None = None) -> str:
    """
    Return the system prompt section body for a named tool.
    Tries 'tool-<name>' (hyphenated) and 'tool_<name>' (underscored) variants.
    Used by llm_call(mode='tool') to build the target model's system prompt.

    If folder is provided, reads directly from that folder without touching the
    global prompt cache (avoids root .system_prompt fallback).
    """
    if folder:
        import os
        from prompt import load_prompt_for_folder
        # Read the specific section file directly — no need to assemble the whole prompt
        for variant in (
            ".system_prompt_tool-" + tool_name.replace("_", "-"),
            ".system_prompt_tool_" + tool_name,
        ):
            fpath = os.path.join(folder, variant)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        raw = fh.read()
                    # Strip leading ## header line if present
                    lines = raw.split("\n", 1)
                    return lines[1] if len(lines) > 1 and lines[0].startswith("## ") else raw
                except Exception:
                    pass
        return ""

    # Fallback: global cache (only hit if no folder provided)
    hyphenated = "tool-" + tool_name.replace("_", "-")
    section = get_section(hyphenated)
    if section:
        return section
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
            'tool_list':             _tool_list_exec,
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
    "judge_configure":       "system",
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
        'judge_configure':       _judge_configure_exec,
        'memory_save':           _memory_save_exec,
        'memory_recall':         _memory_recall_exec,
        'memory_update':         _memory_update_exec,
        'memory_age':            _memory_age_exec,
        'recall_temporal':       _recall_temporal_exec,
        'set_goal':              _set_goal_exec,
        'set_plan':              _set_plan_exec,
        'assert_belief':         _assert_belief_exec,
        'set_conditioned':       _set_conditioned_exec,
        'save_memory_typed':     _save_memory_typed_exec,
        'procedure_save':        _procedure_save_exec,
        'procedure_recall':      _procedure_recall_exec,
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
        "  (system info auto-injected — no tool call needed)\n"
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
    source: str = Field(default="assistant", description="Source: 'assistant' (your own conclusions/recommendations), 'user' (facts the user stated), 'session', or 'directive'. Default is 'assistant'.")

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

class _RecallTemporalArgs(BaseModel):
    query: str = Field(default="", description="Free-text keyword to filter memories by content or topic (e.g. 'Lee', 'walk', 'gym'). Empty = all memories.")
    group_by: str = Field(default="day_of_week", description="How to aggregate: 'hour' (time-of-day buckets), 'day_of_week' (Mon-Sun), 'date' (calendar date), 'week', 'month'.")
    day_of_week: str = Field(default="", description="Optional day filter: 'Monday', 'Tuesday', etc. Comma-separated for multiple.")
    time_range: str = Field(default="", description="Optional time filter: 'HH:MM-HH:MM' (e.g. '09:00-12:00'), or 'morning' (06-12), 'afternoon' (12-17), 'evening' (17-22), 'now' (±90 min from current time).")
    lookback_days: int = Field(default=30, description="How many days back to search. Default 30.")
    limit: int = Field(default=50, description="Max raw rows to return alongside the aggregated pattern summary.")
    new: bool = Field(default=False, description="Force a fresh query, bypassing the temporal cache. Default False = return cached result if available.")


# ---------------------------------------------------------------------------
# Typed memory tools: set_goal, set_plan, assert_belief
# ---------------------------------------------------------------------------

class _SetGoalArgs(BaseModel):
    title: str = Field(default="", description="Short title for the goal.")
    description: str = Field(default="", description="Full description of the objective.")
    importance: int = Field(default=9, description="Importance 1-10. Goals default to 9.")
    status: str = Field(default="", description="Set goal status: 'active', 'done', 'blocked', or 'abandoned'. Omit to leave status unchanged.")
    id: int = Field(default=0, description="Existing goal id to update. 0 = create new.")
    childof: str = Field(default="", description="JSON array of parent goal IDs, e.g. '[1,2]'. Empty = none.")
    parentof: str = Field(default="", description="JSON array of child goal IDs. Empty = none.")
    memory_link: str = Field(default="", description="JSON array of ST/LT memory row IDs that support this goal.")

class _SetPlanArgs(BaseModel):
    goal_id: int = Field(description="ID of the parent goal this step belongs to. 0 = ad-hoc plan (no goal).")
    step_order: int = Field(default=1, description="Step number (ascending order).")
    description: str = Field(description="What this step involves.")
    status: str = Field(default="pending", description="'pending', 'in_progress', 'done', or 'skipped'.")
    id: int = Field(default=0, description="Existing plan step id to update. 0 = create new.")
    memory_link: str = Field(default="", description="JSON array of memory row IDs related to this step.")
    step_type: str = Field(default="concept", description="'concept' (human-readable intent) or 'task' (executable atom).")
    target: str = Field(default="model", description="'model' (auto-executable), 'human' (requires person), or 'investigate' (needs analysis).")
    approval: str = Field(default="proposed", description="'proposed' (needs review), 'approved', 'rejected', or 'auto'.")

class _AssertBeliefArgs(BaseModel):
    topic: str = Field(description="Short topic label for this belief.")
    content: str = Field(description="The asserted world-state fact.")
    confidence: int = Field(default=7, description="Confidence 1-10. How certain is this belief?")
    status: str = Field(default="active", description="'active' or 'retracted'.")
    id: int = Field(default=0, description="Existing belief id to update. 0 = create new.")
    memory_link: str = Field(default="", description="JSON array of ST/LT memory row IDs that support this belief, e.g. '[42,57]'. Populate whenever you know which rows the belief is derived from.")

class _SetConditionedArgs(BaseModel):
    topic: str = Field(description="Short topic label (e.g. 'proactive-schema-gap').")
    trigger: str = Field(description="Stimulus pattern or condition that activates this behavior.")
    reaction: str = Field(description="The learned response or behavior to apply when triggered.")
    strength: int = Field(default=7, description="Reinforcement strength 1-10. Higher = stronger bias.")
    status: str = Field(default="active", description="'active' or 'extinguished'.")
    source: str = Field(default="assistant", description="'session', 'user', 'directive', or 'assistant'.")
    id: int = Field(default=0, description="Existing conditioned id to update. 0 = create new.")
    memory_link: str = Field(default="", description="JSON array of memory row IDs that support this conditioning.")


class _ProcedureSaveArgs(BaseModel):
    task_type: str = Field(
        description="Short machine-readable task category slug, e.g. 'deploy-docker', 'db-schema-change', 'git-push-pr'. Lowercase, hyphens."
    )
    topic: str = Field(description="Human-readable title, e.g. 'Deploy Docker container to nuc11'.")
    steps: str = Field(
        description=(
            'JSON array of ordered step objects. Each: {"step": N, "action": "verb phrase", "tool": "tool_name_or_null", "note": "optional caveat"}. '
            'Example: [{"step":1,"action":"SSH to host","tool":null,"note":"check port conflicts first"}]'
        )
    )
    outcome: str = Field(default="unknown", description="'success', 'partial', 'failure', or 'unknown'. Record actual outcome of this run.")
    notes: str = Field(default="", description="Lessons learned, caveats, edge cases discovered during execution.")
    importance: int = Field(default=7, description="Importance 1-10. Use 8+ to enable pre-injection into task-start context.")
    id: int = Field(default=0, description="Existing procedure row id to update (records another run). 0 = create new.")


class _ProcedureRecallArgs(BaseModel):
    query: str = Field(description="Natural language description of the task you are about to perform. Used for semantic similarity search.")
    task_type: str = Field(default="", description="Optional exact task_type slug to filter results, e.g. 'deploy-docker'. Leave empty for broad semantic search.")
    top_k: int = Field(default=5, description="Max procedures to return.")


async def _procedure_save_exec(
    task_type: str, topic: str, steps: str,
    outcome: str = "unknown", notes: str = "",
    importance: int = 7, id: int = 0,
) -> str:
    import json as _json
    from memory import save_procedure
    from state import current_client_id
    session_id = current_client_id.get("") or ""
    outcome = outcome if outcome in ("success", "partial", "failure", "unknown") else "unknown"
    importance = max(1, min(10, int(importance)))
    try:
        steps_list = _json.loads(steps) if isinstance(steps, str) else steps
        if not isinstance(steps_list, list):
            return "procedure_save: 'steps' must be a JSON array."
    except Exception as e:
        return f"procedure_save: steps JSON parse failed: {e}"
    row_id = await save_procedure(
        topic=topic, task_type=task_type, steps=steps_list,
        outcome=outcome, notes=notes, importance=importance,
        source="assistant", session_id=session_id, id=id,
    )
    if not row_id:
        return "procedure_save: failed (row not found for update)."
    if id and id > 0:
        return f"Procedure id={row_id} updated: outcome={outcome} runs+1"
    return f"Procedure saved (id={row_id}): [{task_type}] {topic} outcome={outcome}"


async def _procedure_recall_exec(
    query: str, task_type: str = "", top_k: int = 5,
) -> str:
    import json as _json
    from memory import recall_procedures
    results = await recall_procedures(query=query, task_type=task_type, top_k=top_k)
    if not results:
        return "No relevant procedures found."
    lines = [f"Found {len(results)} procedure(s):\n"]
    for p in results:
        steps_list = p.get("steps") or []
        step_text = "\n".join(
            f"  {s.get('step','?')}. {s.get('action','')} "
            f"{'[' + s['tool'] + ']' if s.get('tool') else ''}"
            f"{' // ' + s['note'] if s.get('note') else ''}"
            for s in steps_list
        )
        score_str = f" score={p['score']}" if p.get("score") else ""
        lines.append(
            f"[id={p['id']} task_type={p.get('task_type','')} "
            f"outcome={p.get('outcome','?')} "
            f"runs={p.get('success_count','?')}/{p.get('run_count','?')}{score_str}]\n"
            f"Title: {p.get('topic','')}\n"
            f"Steps:\n{step_text}\n"
            f"Notes: {p.get('notes','') or '(none)'}"
        )
    return "\n---\n".join(lines)


class _SaveMemoryTypedArgs(BaseModel):
    memory_type: str = Field(
        description=(
            "The experiential memory type to write: "
            "'episodic' (specific events/experiences), "
            "'semantic' (facts, concepts, world knowledge), "
            "'procedural' (skills, habits, task steps), "
            "'autobiographical' (identity-defining facts about self or key relationships), "
            "'prospective' (planned future intentions — include due_at if time-sensitive)."
        )
    )
    topic: str = Field(description="Short topic label, e.g. 'family', 'coding-style', 'self-identity'.")
    content: str = Field(description="The memory content to store.")
    importance: int = Field(default=5, description="Importance 1-10. autobiographical defaults to 7, prospective to 7.")
    source: str = Field(default="assistant", description="'assistant', 'user', 'directive', or 'session'.")
    due_at: str = Field(default="", description="For prospective only: when to act (timestamp or free-text, e.g. 'next Monday').")
    status: str = Field(default="pending", description="For prospective only: 'pending', 'done', or 'missed'. Ignored for other types.")
    id: int = Field(default=0, description="Existing row id to update. 0 = create new.")
    memory_link: str = Field(default="", description="JSON array of ST/LT memory row IDs that support this entry.")


async def _save_memory_typed_exec(
    memory_type: str, topic: str, content: str, importance: int = 5,
    source: str = "assistant", due_at: str = "", status: str = "pending",
    id: int = 0, memory_link: str = "",
) -> str:
    from memory import (
        _EPISODIC, _SEMANTIC, _PROCEDURAL, _AUTOBIOGRAPHICAL, _PROSPECTIVE,
        _typed_metric_write,
    )
    from database import execute_sql, execute_insert
    from state import current_client_id

    _type_map = {
        "episodic":        _EPISODIC,
        "semantic":        _SEMANTIC,
        "procedural":      _PROCEDURAL,
        "autobiographical": _AUTOBIOGRAPHICAL,
        "prospective":     _PROSPECTIVE,
    }
    if memory_type not in _type_map:
        return f"save_memory_typed: unknown memory_type '{memory_type}'. Valid: {', '.join(_type_map)}."

    table = _type_map[memory_type]()
    session_id = current_client_id.get("") or ""
    source = source if source in ("session", "user", "directive", "assistant") else "assistant"
    importance = max(1, min(10, int(importance)))
    ml_val = memory_link.replace("'", "''") if memory_link else "NULL"

    if id and id > 0:
        # Update
        parts = [f"importance = {importance}"]
        if content:
            parts.append(f"content = '{content.replace(chr(39), chr(39)*2)}'")
        if topic:
            parts.append(f"topic = '{topic.replace(chr(39), chr(39)*2)}'")
        if memory_link:
            parts.append(f"memory_link = '{ml_val}'")
        if memory_type == "prospective":
            _st = status if status in ("pending", "done", "missed") else "pending"
            parts.append(f"status = '{_st}'")
            if due_at:
                parts.append(f"due_at = '{due_at.replace(chr(39), chr(39)*2)}'")
        sql = f"UPDATE {table} SET {', '.join(parts)} WHERE id = {id}"
        try:
            await execute_sql(sql)
            _typed_metric_write(table)
            return f"{memory_type} id={id} updated."
        except Exception as e:
            return f"save_memory_typed update failed: {e}"
    else:
        # Insert
        if not topic or not content:
            return "save_memory_typed: topic and content are required."
        t = topic.replace("'", "''")
        c = content.replace("'", "''")
        _ml = "NULL" if not memory_link else f"'{ml_val}'"

        if memory_type == "prospective":
            _st = status if status in ("pending", "done", "missed") else "pending"
            _due = "NULL" if not due_at else f"'{due_at.replace(chr(39), chr(39)*2)}'"
            sql = (
                f"INSERT INTO {table} "
                f"(topic, content, due_at, status, importance, source, session_id, memory_link) "
                f"VALUES ('{t}', '{c}', {_due}, '{_st}', {importance}, '{source}', '{session_id}', {_ml})"
            )
        else:
            sql = (
                f"INSERT INTO {table} "
                f"(topic, content, importance, source, session_id, memory_link) "
                f"VALUES ('{t}', '{c}', {importance}, '{source}', '{session_id}', {_ml})"
            )
        try:
            row_id = await execute_insert(sql)
            _typed_metric_write(table)
            return f"{memory_type} memory created (id={row_id}): [{topic}] {content}"
        except Exception as e:
            return f"save_memory_typed insert failed: {e}"


async def _set_goal_exec(
    title: str = "", description: str = "", importance: int = 9,
    status: str = "", id: int = 0, childof: str = "",
    parentof: str = "", memory_link: str = "",
) -> str:
    from memory import _GOALS, _typed_metric_write
    from database import execute_sql, execute_insert, fetch_dicts
    from state import current_client_id
    session_id = current_client_id.get("") or ""
    # Empty string = caller did not pass status; treat as "leave unchanged" on updates
    status_explicit = status.strip().lower() if status.strip() else None
    if status_explicit and status_explicit not in ("active", "done", "blocked", "abandoned"):
        status_explicit = None
    importance = max(1, min(10, int(importance)))
    childof_val = childof.replace("'", "''") if childof else "NULL"
    parentof_val = parentof.replace("'", "''") if parentof else "NULL"
    ml_val = memory_link.replace("'", "''") if memory_link else "NULL"

    if id and id > 0:
        # Update existing — only touch status when explicitly provided
        parts = [f"importance = {importance}"]
        if status_explicit:
            parts.insert(0, f"status = '{status_explicit}'")
        if title:
            parts.append(f"title = '{title.replace(chr(39), chr(39)*2)}'")
        if description:
            parts.append(f"description = '{description.replace(chr(39), chr(39)*2)}'")
        if childof:
            parts.append(f"childof = '{childof_val}'")
        if parentof:
            parts.append(f"parentof = '{parentof_val}'")
        if memory_link:
            parts.append(f"memory_link = '{ml_val}'")
        sql = f"UPDATE {_GOALS()} SET {', '.join(parts)} WHERE id = {id}"
        try:
            await execute_sql(sql)
            _typed_metric_write(_GOALS())
            # Read back actual DB state — prevents hallucination on partial updates
            rows = await fetch_dicts(f"SELECT status FROM {_GOALS()} WHERE id = {id} LIMIT 1")
            actual_status = rows[0]["status"] if rows else (status_explicit or "unknown")
            msg = f"Goal id={id} updated: status={actual_status}"
            if status_explicit and actual_status != status_explicit:
                msg += f" (WARNING: requested {status_explicit} but DB shows {actual_status})"
            # Fire notification
            import asyncio as _asyncio
            try:
                import notifier as _notifier
                _evt = {"done": "goal_completed", "blocked": "goal_blocked",
                        "abandoned": "goal_abandoned"}.get(actual_status, "goal_updated")
                _asyncio.ensure_future(_notifier.fire_event(_evt, msg))
            except Exception:
                pass
            return msg
        except Exception as e:
            return f"set_goal update failed: {e}"
    else:
        # Insert new
        if not title:
            return "set_goal: title is required for new goals."
        # Block goals that require unavailable infrastructure
        _blocked_keywords = [
            "timer", "cron", "scheduled", "swarm", "multi-prompt",
            "multi_prompt", "DDL", "schema migration", "create table",
            "alter table", "add column",
        ]
        _combined = (title + " " + description).lower()
        for _kw in _blocked_keywords:
            if _kw.lower() in _combined:
                return (
                    f"set_goal BLOCKED: goal title/description contains '{_kw}' which requires "
                    f"infrastructure marked NOT AVAILABLE (timers, swarms, DDL). "
                    f"Do not retry this goal. Consult the capability map and select a goal "
                    f"that is fully executable with current tools."
                )
        # Abandon guard: reject goals that resemble previously abandoned ones
        try:
            _abandoned = await fetch_dicts(
                f"SELECT id, title, description, abandon_reason FROM {_GOALS()} "
                f"WHERE status='abandoned' ORDER BY updated_at DESC LIMIT 20"
            ) or []
            if _abandoned:
                _prop_words = set(
                    w.lower() for w in (title + " " + description).split() if len(w) > 3
                )
                for _ab in _abandoned:
                    _ab_words = set(
                        w.lower() for w in
                        (_ab.get("title", "") + " " + _ab.get("description", "")).split()
                        if len(w) > 3
                    )
                    if _prop_words and _ab_words:
                        _overlap = len(_prop_words & _ab_words)
                        _ratio = _overlap / min(len(_prop_words), len(_ab_words))
                        if _ratio >= 0.5:
                            return (
                                f"set_goal BLOCKED: this goal resembles abandoned goal "
                                f"id={_ab['id']} ('{_ab.get('title', '')}') which was abandoned "
                                f"because: {_ab.get('abandon_reason', 'persistent failure')}. "
                                f"Do not retry abandoned goals."
                            )
        except Exception:
            pass  # fail-open: if abandon check fails, allow creation

        t = title.replace("'", "''")
        d = description.replace("'", "''")
        _co = "NULL" if not childof else f"'{childof_val}'"
        _po = "NULL" if not parentof else f"'{parentof_val}'"
        _ml = "NULL" if not memory_link else f"'{ml_val}'"
        insert_status = status_explicit or "active"
        sql = (
            f"INSERT INTO {_GOALS()} "
            f"(title, description, status, importance, source, session_id, childof, parentof, memory_link) "
            f"VALUES ('{t}', '{d}', '{insert_status}', {importance}, 'assistant', '{session_id}', "
            f"{_co}, {_po}, {_ml})"
        )
        try:
            row_id = await execute_insert(sql)
            _typed_metric_write(_GOALS())
            _created_msg = f"Goal created (id={row_id}): {title} [status={insert_status} imp={importance}]"
            # Fire notification
            import asyncio as _asyncio
            try:
                import notifier as _notifier
                _asyncio.ensure_future(_notifier.fire_event(
                    "goal_created", _created_msg
                ))
            except Exception:
                pass
            return _created_msg
        except Exception as e:
            return f"set_goal insert failed: {e}"


async def _set_plan_exec(
    goal_id: int = 0, step_order: int = 1, description: str = "",
    status: str = "pending", id: int = 0, memory_link: str = "",
    step_type: str = "concept", target: str = "model", approval: str = "proposed",
) -> str:
    from memory import _PLANS, _typed_metric_write
    from database import execute_sql, execute_insert
    from state import current_client_id
    session_id = current_client_id.get("") or ""
    status = status if status in ("pending", "in_progress", "done", "skipped") else "pending"
    step_type = step_type if step_type in ("concept", "task") else "concept"
    target = target if target in ("model", "human", "investigate") else "model"
    approval = approval if approval in ("proposed", "approved", "rejected", "auto") else "proposed"
    ml_val = memory_link.replace("'", "''") if memory_link else "NULL"

    if id and id > 0:
        parts = [f"status = '{status}'"]
        if description:
            parts.append(f"description = '{description.replace(chr(39), chr(39)*2)}'")
        if memory_link:
            parts.append(f"memory_link = '{ml_val}'")
        # Allow updating target and approval on existing steps
        parts.append(f"target = '{target}'")
        parts.append(f"approval = '{approval}'")
        sql = f"UPDATE {_PLANS()} SET {', '.join(parts)} WHERE id = {id}"
        try:
            from database import fetch_dicts as _fd
            await execute_sql(sql)
            _typed_metric_write(_PLANS())
            rows = await _fd(f"SELECT status, step_type, target, approval FROM {_PLANS()} WHERE id = {id} LIMIT 1")
            actual_status = rows[0]["status"] if rows else status
            actual_type = rows[0].get("step_type", "?") if rows else step_type
            msg = f"Plan step id={id} updated: status={actual_status} type={actual_type} target={target} approval={approval}"
            if actual_status != status:
                msg += f" (WARNING: requested {status} but DB shows {actual_status})"
            # Fire notification
            import asyncio as _asyncio
            try:
                import notifier as _notifier
                _tevt = "task_completed" if actual_status == "done" else "task_updated"
                _asyncio.ensure_future(_notifier.fire_event(_tevt, msg))
            except Exception:
                pass
            return msg
        except Exception as e:
            return f"set_plan update failed: {e}"
    else:
        if not description:
            return "set_plan: description is required for new steps."
        d = description.replace("'", "''")
        _ml = "NULL" if not memory_link else f"'{ml_val}'"
        sql = (
            f"INSERT INTO {_PLANS()} "
            f"(goal_id, step_order, description, status, step_type, target, approval, source, session_id, memory_link) "
            f"VALUES ({goal_id}, {step_order}, '{d}', '{status}', '{step_type}', '{target}', '{approval}', 'assistant', '{session_id}', {_ml})"
        )
        try:
            row_id = await execute_insert(sql)
            _typed_metric_write(_PLANS())
            _plan_msg = (
                f"Plan step created (id={row_id}): goal={goal_id} step={step_order} "
                f"[{status}] type={step_type} target={target} approval={approval} {description}"
            )
            # Fire notification
            import asyncio as _asyncio
            try:
                import notifier as _notifier
                _asyncio.ensure_future(_notifier.fire_event("task_created", _plan_msg))
            except Exception:
                pass
            return _plan_msg
        except Exception as e:
            return f"set_plan insert failed: {e}"


async def _assert_belief_exec(
    topic: str, content: str, confidence: int = 7,
    status: str = "active", id: int = 0, memory_link: str = "",
) -> str:
    from memory import _BELIEFS, _typed_metric_write
    from database import execute_sql, execute_insert
    from state import current_client_id
    session_id = current_client_id.get("") or ""
    status = status if status in ("active", "retracted") else "active"
    confidence = max(1, min(10, int(confidence)))
    ml_val = memory_link.replace("'", "''") if memory_link else "NULL"

    if id and id > 0:
        parts = [f"status = '{status}'", f"confidence = {confidence}"]
        if content:
            parts.append(f"content = '{content.replace(chr(39), chr(39)*2)}'")
        if memory_link:
            parts.append(f"memory_link = '{ml_val}'")
        sql = f"UPDATE {_BELIEFS()} SET {', '.join(parts)} WHERE id = {id}"
        try:
            await execute_sql(sql)
            _typed_metric_write(_BELIEFS())
            return f"Belief id={id} updated: status={status} confidence={confidence}"
        except Exception as e:
            return f"assert_belief update failed: {e}"
    else:
        t = topic.replace("'", "''")
        c = content.replace("'", "''")
        _ml = "NULL" if not memory_link else f"'{ml_val}'"
        sql = (
            f"INSERT INTO {_BELIEFS()} "
            f"(topic, content, confidence, status, source, session_id, memory_link) "
            f"VALUES ('{t}', '{c}', {confidence}, '{status}', 'assistant', '{session_id}', {_ml})"
        )
        try:
            row_id = await execute_insert(sql)
            _typed_metric_write(_BELIEFS())
            return f"Belief asserted (id={row_id}): [{topic}] {content} (confidence={confidence})"
        except Exception as e:
            return f"assert_belief insert failed: {e}"


async def _set_conditioned_exec(
    topic: str, trigger: str = "", reaction: str = "",
    strength: int = 7, status: str = "active", source: str = "assistant",
    id: int = 0, memory_link: str = "",
) -> str:
    from memory import _CONDITIONED, _typed_metric_write
    from database import execute_sql, execute_insert
    from state import current_client_id
    session_id = current_client_id.get("") or ""
    strength = max(1, min(10, int(strength)))
    status = status if status in ("active", "extinguished") else "active"
    source = source if source in ("session", "user", "directive", "assistant") else "assistant"
    ml_val = memory_link.replace("'", "''") if memory_link else "NULL"

    if id and id > 0:
        parts = [f"status = '{status}'", f"strength = {strength}"]
        if trigger:
            parts.append(f"`trigger` = '{trigger.replace(chr(39), chr(39)*2)}'")
        if reaction:
            parts.append(f"`reaction` = '{reaction.replace(chr(39), chr(39)*2)}'")
        if memory_link:
            parts.append(f"memory_link = '{ml_val}'")
        sql = f"UPDATE {_CONDITIONED()} SET {', '.join(parts)} WHERE id = {id}"
        try:
            await execute_sql(sql)
            _typed_metric_write(_CONDITIONED())
            return f"Conditioned id={id} updated: status={status} strength={strength}"
        except Exception as e:
            return f"set_conditioned update failed: {e}"
    else:
        if not topic or not trigger or not reaction:
            return "set_conditioned: topic, trigger, and reaction are required for new entries."
        _to = topic.replace("'", "''")
        _tr = trigger.replace("'", "''")
        _re = reaction.replace("'", "''")
        _ml = "NULL" if not memory_link else f"'{ml_val}'"
        sql = (
            f"INSERT INTO {_CONDITIONED()} "
            f"(topic, `trigger`, `reaction`, strength, status, source, session_id, memory_link) "
            f"VALUES ('{_to}', '{_tr}', '{_re}', {strength}, '{status}', '{source}', '{session_id}', {_ml})"
        )
        try:
            row_id = await execute_insert(sql)
            _typed_metric_write(_CONDITIONED())
            return f"Conditioned entry created (id={row_id}): [{topic}] strength={strength} {trigger} → {reaction}"
        except Exception as e:
            return f"set_conditioned insert failed: {e}"


# ---------------------------------------------------------------------------
# Judge configure tool
# ---------------------------------------------------------------------------

class _JudgeConfigureArgs(BaseModel):
    action: str = Field(
        description=(
            "Action to perform. One of: "
            "status (show current config), "
            "list (all models), "
            "on (enable gate), "
            "off (disable gate), "
            "set_model (set judge model for session), "
            "set_mode (block or warn), "
            "set_threshold (0.0-1.0), "
            "reset (clear session overrides), "
            "test (evaluate text), "
            "persist (save field to llm-models.json)."
        )
    )
    gate: str = Field(default="", description="Gate name: prompt, response, tool, memory, or all. Used with on/off.")
    model: str = Field(default="", description="Judge model name. Used with set_model or persist action.")
    mode: str = Field(default="", description="block or warn. Used with set_mode or persist.")
    threshold: float = Field(default=-1.0, description="Score threshold 0.0-1.0. Used with set_threshold or persist.")
    text: str = Field(default="", description="Text to evaluate. Used with test action.")
    target_model: str = Field(default="", description="Model to persist config to. Used with persist action.")
    field: str = Field(default="", description="Field to persist: model, mode, threshold, gates. Used with persist.")
    gates: str = Field(default="", description="Comma-separated gates list. Used with persist when field=gates.")


async def _judge_configure_exec(
    action: str,
    gate: str = "",
    model: str = "",
    mode: str = "",
    threshold: float = -1.0,
    text: str = "",
    target_model: str = "",
    field: str = "",
    gates: str = "",
) -> str:
    from state import current_client_id, sessions
    from judge import cmd_judge as _judge_cmd
    client_id = current_client_id.get("") or ""
    session = sessions.get(client_id, {})

    # Map structured args to the !judge command string format
    if action == "status":
        arg = "status"
    elif action == "list":
        arg = "list"
    elif action == "on":
        if not gate:
            return "ERROR: gate required for action='on'."
        arg = f"on {gate}"
    elif action == "off":
        if not gate:
            return "ERROR: gate required for action='off'."
        arg = f"off {gate}"
    elif action == "set_model":
        if not model:
            return "ERROR: model required for action='set_model'."
        arg = f"model {model}"
    elif action == "set_mode":
        if mode not in ("block", "warn"):
            return "ERROR: mode must be 'block' or 'warn'."
        arg = f"mode {mode}"
    elif action == "set_threshold":
        if threshold < 0.0 or threshold > 1.0:
            return "ERROR: threshold must be between 0.0 and 1.0."
        arg = f"threshold {threshold}"
    elif action == "reset":
        arg = "reset"
    elif action == "test":
        if not text:
            return "ERROR: text required for action='test'."
        arg = f"test {text}"
    elif action == "persist":
        if not target_model or not field:
            return "ERROR: target_model and field required for action='persist'."
        if field == "gates":
            value = gates
        elif field == "model":
            value = model
        elif field == "mode":
            value = mode
        elif field == "threshold":
            value = str(threshold) if threshold >= 0 else ""
        else:
            return f"ERROR: unknown field '{field}'."
        if not value:
            return f"ERROR: value required for field='{field}'."
        arg = f"set {target_model} {field} {value}"
    else:
        return f"ERROR: unknown action '{action}'."

    return await _judge_cmd(client_id, arg, session)


async def _memory_save_exec(topic: str, content: str, importance: int = 5, source: str = "assistant") -> str:
    from memory import save_memory
    from state import current_client_id, sessions
    session_id = current_client_id.get("") or ""
    _sess_mem = sessions.get(session_id, {}).get("memory_enabled", None)
    if _sess_mem is False:
        return "Memory logging is disabled for this session."
    row_id = await save_memory(
        topic=topic, content=content,
        importance=importance, source=source,
        session_id=session_id,
    )
    if row_id == 0:
        return f"Memory duplicate skipped (already in memory): [{topic}] {content}"
    return f"Memory saved (id={row_id}): [{topic}] {content} (importance={importance})"


async def _memory_recall_exec(topic: str = "", tier: str = "short", limit: int = 20) -> str:
    from memory import load_short_term, load_long_term
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


def _temporal_query_key(query: str, group_by: str, day_of_week: str, time_range: str) -> str:
    """Build a normalized cache key for temporal queries.
    Excludes lookback_days and limit since those don't change the semantic query.
    'now' time_range is normalized to the current hour bucket for reasonable cache hits.
    """
    import datetime as _dt
    tr = time_range.lower().strip() if time_range else ""
    if tr == "now":
        # Normalize 'now' to current hour bucket in display timezone
        from config import now_display
        tr = f"now-h{now_display().hour}"
    return f"{query.lower().strip()}|{group_by}|{day_of_week.lower().strip()}|{tr}"


def _temporal_table() -> str:
    """Return the temporal cache table name for the active model context."""
    from database import get_tables_for_model
    return get_tables_for_model().get("temporal", "samaritan_temporal")


async def _temporal_cache_lookup(query_key: str) -> str | None:
    """Check samaritan_temporal for a cached result matching this query_key.
    Returns the cached result string, or None if no match.
    """
    from database import execute_sql
    tbl = _temporal_table()
    escaped_key = query_key.replace("'", "''").replace("\\", "\\\\")
    # Look for a cache entry from the last 24 hours
    row = await execute_sql(
        f"SELECT id, result FROM {tbl} "
        f"WHERE query_key = '{escaped_key}' "
        f"AND created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR) "
        f"ORDER BY created_at DESC LIMIT 1"
    )
    if row and "(no rows)" not in row:
        # Bump hit count
        # Extract id from the formatted result (first data line, first column)
        lines = row.strip().split("\n")
        if len(lines) >= 3:  # header + sep + data
            row_id = lines[2].split("|")[0].strip()
            if row_id.isdigit():
                await execute_sql(
                    f"UPDATE {tbl} SET hit_count = hit_count + 1 WHERE id = {row_id}"
                )
                # Result is the second column
                result_text = "|".join(lines[2].split("|")[1:]).strip()
                return result_text
    return None


async def _temporal_cache_store(
    query_key: str, query_params: dict, result: str, source: str = "explicit"
) -> int:
    """Store a temporal query result in the cache table. Returns the new row id."""
    from database import execute_sql
    import json as _json
    tbl = _temporal_table()
    esc_key = query_key.replace("'", "''").replace("\\", "\\\\")
    esc_result = result.replace("'", "''").replace("\\", "\\\\")
    esc_params = _json.dumps(query_params).replace("'", "''").replace("\\", "\\\\")
    insert_sql = (
        f"INSERT INTO {tbl} (source, query_key, query_params, result) "
        f"VALUES ('{source}', '{esc_key}', '{esc_params}', '{esc_result}')"
    )
    await execute_sql(insert_sql)
    # Get inserted id
    id_result = await execute_sql(f"SELECT LAST_INSERT_ID() AS id")
    lines = id_result.strip().split("\n")
    if len(lines) >= 3:
        row_id = lines[2].strip()
        return int(row_id) if row_id.isdigit() else 0
    return 0


async def _recall_temporal_exec(
    query: str = "", group_by: str = "day_of_week", day_of_week: str = "",
    time_range: str = "", lookback_days: int = 30, limit: int = 50,
    new: bool = False, source: str = "explicit",
) -> str:
    """Search memories by temporal patterns across both short-term and long-term.
    Checks the temporal cache first; use new=True to force a fresh query.
    """
    from database import execute_sql
    from memory import _ST, _LT
    import datetime as _dt

    # --- Cache lookup (unless new=True) ---
    qkey = _temporal_query_key(query, group_by, day_of_week, time_range)
    if not new:
        cached = await _temporal_cache_lookup(qkey)
        if cached:
            return f"[cached result — use new=True to refresh]\n{cached}"

    st_table = _ST()
    lt_table = _LT()

    # --- Build WHERE clauses ---
    where_parts = [f"created_at >= DATE_SUB(NOW(), INTERVAL {int(lookback_days)} DAY)"]

    # Content/topic keyword filter
    if query:
        escaped = query.replace("'", "''").replace("\\", "\\\\")
        where_parts.append(
            f"(content LIKE '%{escaped}%' OR topic LIKE '%{escaped}%')"
        )

    # Day-of-week filter
    if day_of_week:
        days = [d.strip().capitalize() for d in day_of_week.split(",")]
        valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        days = [d for d in days if d in valid_days]
        if days:
            day_list = ", ".join(f"'{d}'" for d in days)
            where_parts.append(f"DAYNAME(created_at) IN ({day_list})")

    # Time range filter
    if time_range:
        _named = {
            "morning": ("06:00", "12:00"),
            "afternoon": ("12:00", "17:00"),
            "evening": ("17:00", "22:00"),
            "night": ("22:00", "06:00"),
        }
        if time_range.lower() in _named:
            t_start, t_end = _named[time_range.lower()]
        elif time_range.lower() == "now":
            from config import now_display
            now = now_display()
            t_start = (now - _dt.timedelta(minutes=90)).strftime("%H:%M")
            t_end = (now + _dt.timedelta(minutes=90)).strftime("%H:%M")
        elif "-" in time_range:
            parts = time_range.split("-", 1)
            t_start, t_end = parts[0].strip(), parts[1].strip()
        else:
            t_start, t_end = None, None

        if t_start and t_end:
            if t_start <= t_end:
                where_parts.append(f"TIME(created_at) BETWEEN '{t_start}' AND '{t_end}'")
            else:
                # Wraps midnight (e.g. 22:00-06:00)
                where_parts.append(
                    f"(TIME(created_at) >= '{t_start}' OR TIME(created_at) <= '{t_end}')"
                )

    where_clause = " AND ".join(where_parts)

    # --- Aggregation SQL ---
    _group_expr = {
        "hour": "HOUR(created_at)",
        "day_of_week": "DAYNAME(created_at)",
        "date": "DATE(created_at)",
        "week": "YEARWEEK(created_at, 1)",
        "month": "DATE_FORMAT(created_at, '%Y-%m')",
    }
    group_col = _group_expr.get(group_by, "DAYNAME(created_at)")
    group_label = group_by

    # Source filter: only user/assistant content (skip session/tool-call noise)
    content_filter = "source IN ('user', 'assistant')"

    # Run aggregation query across both tiers
    agg_sql = (
        f"SELECT {group_col} AS `{group_label}`, "
        f"COUNT(*) AS occurrences, "
        f"GROUP_CONCAT(DISTINCT topic ORDER BY topic SEPARATOR ', ') AS topics, "
        f"MIN(created_at) AS earliest, MAX(created_at) AS latest "
        f"FROM ("
        f"  SELECT topic, content, created_at, source FROM {st_table} WHERE {where_clause} AND {content_filter}"
        f"  UNION ALL"
        f"  SELECT topic, content, created_at, 'assistant' AS source FROM {lt_table} WHERE {where_clause}"
        f") combined "
        f"GROUP BY {group_col} "
        f"ORDER BY occurrences DESC"
    )

    # Run raw sample query (most recent matches)
    raw_sql = (
        f"SELECT tier, topic, LEFT(content, 150) AS content, "
        f"DATE(created_at) AS date, TIME(created_at) AS time, "
        f"DAYNAME(created_at) AS day_of_week "
        f"FROM ("
        f"  SELECT 'short' AS tier, topic, content, created_at, source FROM {st_table} WHERE {where_clause} AND {content_filter}"
        f"  UNION ALL"
        f"  SELECT 'long' AS tier, topic, content, created_at, 'assistant' AS source FROM {lt_table} WHERE {where_clause}"
        f") combined "
        f"ORDER BY created_at DESC "
        f"LIMIT {int(limit)}"
    )

    agg_result = await execute_sql(agg_sql)
    raw_result = await execute_sql(raw_sql)

    # Full result returned to caller includes raw samples for immediate use
    result = (
        f"## Temporal Pattern Summary (group_by={group_by}, lookback={lookback_days}d)\n"
        f"{agg_result}\n\n"
        f"## Recent Matching Memories ({limit} max)\n"
        f"{raw_result}"
    )

    # Cache only the aggregation summary — raw samples are bulky and can be
    # re-queried live; caching them wastes storage with ASCII table formatting.
    cache_result = (
        f"## Temporal Pattern Summary (group_by={group_by}, lookback={lookback_days}d)\n"
        f"{agg_result}"
    )

    # --- Store in cache ---
    query_params = {
        "query": query, "group_by": group_by, "day_of_week": day_of_week,
        "time_range": time_range, "lookback_days": lookback_days,
    }
    await _temporal_cache_store(qkey, query_params, cache_result, source=source)

    return result


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
    from config import LLM_TOOLSETS, LLM_TOOLSET_META, LLM_REGISTRY, save_llm_toolset, delete_llm_toolset

    if action == "list":
        if not LLM_TOOLSETS:
            return "No toolsets defined in llm-tools.json."
        lines = ["Toolsets (llm-tools.json):"]
        for ts_name in sorted(LLM_TOOLSETS.keys()):
            tool_list = LLM_TOOLSETS[ts_name]
            meta = LLM_TOOLSET_META.get(ts_name, {})
            always = "always" if meta.get("always_active", True) else "hot/cold"
            lines.append(f"  {ts_name} [{always}]: {', '.join(tool_list)} ({len(tool_list)} tools)")
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
        meta = LLM_TOOLSET_META.get(name, {})
        always = "always_active" if meta.get("always_active", True) else f"hot/cold heat_curve={meta.get('heat_curve')}"
        return f"Toolset '{name}' [{always}]: {', '.join(ts)} ({len(ts)} tools)"

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
            LLM_TOOLSET_META.pop(name, None)
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
        "stream":               ("agent_call_stream",   "bool", True),
        "tool_preview_length":  ("tool_preview_length", "int",  500),
        "tool_suppress":        ("tool_suppress",        "bool", False),
        "memory_scan_suppress": ("memory_scan_suppress", "bool", False),
        "auto_enrich":          ("auto_enrich",          "bool", True),
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
        "max_tool_iterations": "Max LLM↔tool round-trips per request (-1 = unlimited)",
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


# ---------------------------------------------------------------------------
# tool_list — always-active capability discovery tool
# ---------------------------------------------------------------------------

class _ToolListArgs(BaseModel):
    action: str = Field(
        default="list",
        description="'list' — show all authorized tools with hot/cold status. 'describe' — full description for one tool.",
    )
    tool: str = Field(default="", description="Tool name for action='describe'.")


async def _tool_list_exec(action: str = "list", tool: str = "") -> str:
    """Discover authorized tools and their hot/cold status for the current session."""
    from config import LLM_TOOLSETS, LLM_TOOLSET_META, LLM_REGISTRY
    from agents import _compute_active_tools, _get_cold_tool_names, _CURRENT_LC_TOOLS
    from state import sessions, current_client_id

    client_id = current_client_id.get(None)
    session = sessions.get(client_id, {}) if client_id else {}
    model_key = session.get("model", "")

    if action == "list":
        cfg = LLM_REGISTRY.get(model_key, {})
        authorized_toolsets = cfg.get("llm_tools", [])
        if not authorized_toolsets:
            return "No tools authorized for this model."

        active = _compute_active_tools(model_key, client_id) if client_id else set()
        subs = session.get("tool_subscriptions", {})

        lines = ["Authorized tools (hot = active in current schema, cold = available on demand):"]
        for ts_name in authorized_toolsets:
            meta = LLM_TOOLSET_META.get(ts_name, {})
            tools_in_set = LLM_TOOLSETS.get(ts_name, [ts_name])
            if meta.get("always_active", True):
                status = "always"
            else:
                sub = subs.get(ts_name, {})
                heat = sub.get("heat", 0)
                status = f"hot(heat={heat})" if heat > 0 else "cold"
            for t in tools_in_set:
                lines.append(f"  {t}: [{status}]")
        lines.append("\nCall tool_list(action='describe', tool='<name>') for full usage details on any tool.")
        return "\n".join(lines)

    if action == "describe":
        if not tool:
            return "ERROR: 'tool' required for action='describe'."
        for lc_tool in _CURRENT_LC_TOOLS:
            if lc_tool.name == tool:
                schema = lc_tool.args_schema.model_json_schema() if lc_tool.args_schema else {}
                params = schema.get("properties", {})
                param_lines = [f"  {k}: {v.get('description', '')}" for k, v in params.items()]
                param_str = "\n".join(param_lines) if param_lines else "  (no parameters)"
                return f"Tool: {tool}\n{lc_tool.description}\n\nParameters:\n{param_str}"
        return f"Tool '{tool}' not found or not loaded."

    return "Unknown action. Valid: list, describe"


def _make_core_lc_tools() -> list:
    """Build CORE_LC_TOOLS after agents module is available (avoids circular import)."""
    import agents as _agents
    return [
        StructuredTool.from_function(
            coroutine=_tool_list_exec,
            name="tool_list",
            description=(
                "Discover authorized tools and their current hot/cold status. "
                "action='list': show all tools available to this model, indicating which are "
                "currently active (hot) and which are available on demand (cold). "
                "action='describe': get full usage details for a specific tool by name. "
                "Use this when you are unsure what tools are available."
            ),
            args_schema=_ToolListArgs,
        ),
        # get_system_info removed as LLM-callable tool — now auto-injected
        # via auto_enrich_context() system-info line. Models no longer need to
        # call a tool for date/time.
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
        StructuredTool.from_function(
            coroutine=_judge_configure_exec,
            name="judge_configure",
            description=(
                "Configure and control the LLM-as-judge enforcement layer. "
                "Actions: status (show current config), list (all models), "
                "on/off (enable/disable a gate: prompt, response, tool, memory, all), "
                "set_model (set judge model for this session), "
                "set_mode (block=deny on fail, warn=log+allow), "
                "set_threshold (score floor 0.0–1.0), "
                "reset (clear session overrides), "
                "test (evaluate text with the judge), "
                "persist (save a judge_config field to llm-models.json permanently). "
                "Session changes are temporary; use persist to make them permanent."
            ),
            args_schema=_JudgeConfigureArgs,
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
        StructuredTool.from_function(
            coroutine=_recall_temporal_exec,
            name="recall_temporal",
            description=(
                "Discover time-based patterns in memory. Checks the temporal cache first; "
                "returns cached result if available (use new=True to force a fresh query). "
                "Searches BOTH short-term and long-term memory "
                "and returns an aggregated pattern summary plus recent matching rows. "
                "Use for questions like 'what do I usually do at this time?', 'what happens on Tuesdays?', "
                "'what patterns exist around Lee?', 'what happens monthly?'. "
                "query: keyword filter (e.g. 'Lee', 'walk'). "
                "group_by: 'hour', 'day_of_week', 'date', 'week', 'month'. "
                "day_of_week: optional day filter. time_range: 'HH:MM-HH:MM', 'morning', 'afternoon', 'evening', 'now'. "
                "lookback_days: how far back (default 30). "
                "new: True to bypass cache and run a fresh query."
            ),
            args_schema=_RecallTemporalArgs,
        ),
        # --- Typed memory tools ---
        StructuredTool.from_function(
            coroutine=_set_goal_exec,
            name="set_goal",
            description=(
                "Create or update an active goal in the goals table. "
                "Goals persist across sessions and are always injected into context. "
                "id=0 creates a new goal; id>0 updates an existing one (use to change status to 'done'/'blocked'). "
                "childof/parentof: JSON arrays of related goal IDs for goal hierarchies."
            ),
            args_schema=_SetGoalArgs,
        ),
        StructuredTool.from_function(
            coroutine=_set_plan_exec,
            name="set_plan",
            description=(
                "Create or update a plan step linked to a goal. "
                "Each step is one row; use step_order to sequence them. "
                "id=0 creates new; id>0 updates status of an existing step. "
                "Status: pending → in_progress → done/skipped."
            ),
            args_schema=_SetPlanArgs,
        ),
        StructuredTool.from_function(
            coroutine=_assert_belief_exec,
            name="assert_belief",
            description=(
                "Assert or update a world-state belief. "
                "Beliefs are always injected into context (confidence-ordered). "
                "id=0 creates new; id>0 updates an existing belief (use status='retracted' to withdraw). "
                "confidence: 1=speculative, 7=reasonably certain, 10=ground truth. "
                "IMPORTANT: content must be a synthesized conclusion in your own words — not a copy of source text. "
                "One concise assertional sentence, e.g. 'Admin prefers terse, direct responses over verbose explanations.' "
                "If the belief is derived from specific ST/LT memory rows, pass their IDs as memory_link (JSON array, e.g. '[42,57]'). "
                "memory_link is evidence provenance — populate it whenever you know which rows support the assertion."
            ),
            args_schema=_AssertBeliefArgs,
        ),
        StructuredTool.from_function(
            coroutine=_set_conditioned_exec,
            name="set_conditioned",
            description=(
                "Record or update a conditioned behavior — a learned trigger→reaction pattern. "
                "Active entries are always injected into context (strength-ordered) as behavioral biases. "
                "id=0 creates new; id>0 updates (use status='extinguished' to suppress). "
                "strength: 1=weak hint, 7=strong bias, 10=near-mandatory. "
                "source='assistant' for self-identified patterns; source='user' or 'directive' for instructed ones."
            ),
            args_schema=_SetConditionedArgs,
        ),
        StructuredTool.from_function(
            coroutine=_save_memory_typed_exec,
            name="save_memory_typed",
            description=(
                "Save an experiential memory to a dedicated typed table. "
                "Use this for rich, structured memories beyond conv_log: "
                "episodic (specific events/experiences), "
                "semantic (facts, concepts, world knowledge), "
                "procedural (skills, habits, task steps), "
                "autobiographical (identity-defining facts — always injected every turn), "
                "prospective (planned future intentions — always injected until done; set due_at if time-sensitive). "
                "id=0 creates new; id>0 updates an existing entry. "
                "For prospective: use status='done' or 'missed' to retire the intention."
            ),
            args_schema=_SaveMemoryTypedArgs,
        ),
        # --- Structured procedural memory tools ---
        StructuredTool.from_function(
            coroutine=_procedure_save_exec,
            name="procedure_save",
            description=(
                "Save or update a structured procedure — a reusable multi-step task record. "
                "Call this AFTER completing a multi-step task to encode the exact steps and outcome. "
                "id=0 creates a new procedure; id>0 records another run on an existing one (increments run_count, updates outcome). "
                "task_type: machine-readable slug (e.g. 'git-push-pr', 'db-schema-change', 'docker-deploy'). "
                "steps: JSON array [{\"step\":N,\"action\":\"...\",\"tool\":\"...\",\"note\":\"...\"}]. "
                "outcome: 'success', 'partial', or 'failure'. "
                "importance >= 8 causes this procedure to be injected into context at task-start time. "
                "success_count/run_count ratio builds over time — high ratio = battle-tested procedure."
            ),
            args_schema=_ProcedureSaveArgs,
        ),
        StructuredTool.from_function(
            coroutine=_procedure_recall_exec,
            name="procedure_recall",
            description=(
                "Retrieve stored procedures semantically relevant to a task you are about to perform. "
                "Call this at the START of any multi-step task before beginning. "
                "If a matching procedure exists with high success_count, follow its steps unless you have reason to deviate. "
                "query: natural language description of what you are about to do. "
                "task_type: optional exact slug to narrow results."
            ),
            args_schema=_ProcedureRecallArgs,
        ),
    ]


# Populated by llmem-gw.py after plugin registration via update_tool_definitions()
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
