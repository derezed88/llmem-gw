import asyncio
import json
import os
import re
import uuid

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
)

#from .config import log, MAX_TOOL_ITERATIONS, LLM_REGISTRY
#from .state import push_tok, push_done, push_err
#from .prompt import get_current_prompt
import time

from config import log, MAX_TOOL_ITERATIONS, LLM_REGISTRY, RATE_LIMITS, LIVE_LIMITS, LLM_TOOLSETS, LLM_TOOLSET_META, TOOL_CALL_LOG_DEFAULT, save_llm_model_field
from state import push_tok, push_done, push_flush, push_err, current_client_id, sessions, wait_for_gate, resolve_gate, has_pending_gate, update_session_token_stats
from prompt import load_prompt_for_folder
from database import execute_sql, set_model_context
from tools import (
    get_system_info,
    get_all_lc_tools, get_all_openai_tools, get_tool_executor,
    get_tool_type,
)

# ---------------------------------------------------------------------------
# Outbound agent message filters
# Loaded once at startup from plugins-enabled.json
# plugin_config.plugin_client_api.OUTBOUND_AGENT_ALLOWED_COMMANDS
# plugin_config.plugin_client_api.OUTBOUND_AGENT_BLOCKED_COMMANDS
#
# These filter the *message* text sent outbound via agent_call — not tool names.
# ALLOWED (non-empty): message must start with one of the listed prefixes.
#          Empty [] = all messages permitted (no check performed).
# BLOCKED: message must not start with any of the listed prefixes.
#          Always checked when non-empty; empty [] = nothing blocked.
# Both lists are lowercased prefix strings.
# Default is empty for both — all agent-to-agent messages are permitted.
# ---------------------------------------------------------------------------

_outbound_agent_allowed: list[str] = []
_outbound_agent_blocked: list[str] = []


def _load_outbound_agent_filters() -> None:
    """Load OUTBOUND_AGENT_ALLOWED/BLOCKED_COMMANDS from plugins-enabled.json."""
    global _outbound_agent_allowed, _outbound_agent_blocked
    try:
        path = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
        with open(path, "r") as f:
            cfg = json.load(f)
        api_cfg = cfg.get("plugin_config", {}).get("plugin_client_api", {})
        raw_allow = api_cfg.get("OUTBOUND_AGENT_ALLOWED_COMMANDS", [])
        raw_block = api_cfg.get("OUTBOUND_AGENT_BLOCKED_COMMANDS", [])
        _outbound_agent_allowed = [s.strip().lower() for s in raw_allow if s.strip()]
        _outbound_agent_blocked = [s.strip().lower() for s in raw_block if s.strip()]
    except Exception:
        pass   # no filter file = no restrictions


_load_outbound_agent_filters()


def _match_outbound_pattern(msg_lower: str, pattern: str) -> bool:
    """
    Match a lowercased, stripped message against a filter pattern.

    Special commands have multiple forms:
      !reset                       (no args)
      !model / !model <name>        (optional args after space)
      !tmux new foo / !tmux ls     (subcommand + optional args)

    Rules:
    - If pattern ends with a space: raw prefix match (caller controls boundary).
    - Otherwise: match if msg == pattern OR msg starts with pattern + " ".
      This prevents "!mod" from matching "!model" while still matching
      "!model", "!model <name>", and "!model list".
    - Non-! patterns (plain text prefixes) use the same boundary logic.
    """
    if pattern.endswith(" "):
        return msg_lower.startswith(pattern)
    return msg_lower == pattern or msg_lower.startswith(pattern + " ")


def _check_outbound_agent_message(message: str) -> str | None:
    """
    Apply outbound agent message filters.
    Returns None if permitted, or an error string if blocked.

    Patterns are lowercased at load time. Matching is word-boundary aware:
    pattern '!model' matches '!model' and '!model <name>' but NOT '!modelx'.
    To match any prefix including mid-word, end the pattern with a space.
    """
    msg_lower = message.strip().lower()
    if _outbound_agent_allowed:
        if not any(_match_outbound_pattern(msg_lower, p) for p in _outbound_agent_allowed):
            return (
                f"BLOCKED by OUTBOUND_AGENT_ALLOWED_COMMANDS: message does not match "
                f"any allowed prefix. Allowed: {', '.join(_outbound_agent_allowed)}"
            )
    for pattern in _outbound_agent_blocked:
        if _match_outbound_pattern(msg_lower, pattern):
            return (
                f"BLOCKED by OUTBOUND_AGENT_BLOCKED_COMMANDS: message matches "
                f"blocked pattern '{pattern}'."
            )
    return None


# ---------------------------------------------------------------------------
# LangChain LLM Factory
# ---------------------------------------------------------------------------

def _build_lc_llm(model_key: str, use_cache: bool = False):
    """
    Build a LangChain chat model from LLM_REGISTRY config.

    When token_selection_setting == "custom", passes temperature/top_p/top_k
    to the constructor.  When "default", no sampling parameters are passed so
    the API backend uses its own defaults.

    use_cache=True (stream_level >= 1): return cached client if available,
    otherwise build and store. ChatOpenAI/ChatGoogleGenerativeAI are stateless
    so sharing across sessions/requests is safe.

    Returns a ChatOpenAI or ChatGoogleGenerativeAI instance.
    Both expose the same .ainvoke() / .astream() interface.
    """
    if use_cache and model_key in _llm_client_cache:
        return _llm_client_cache[model_key]

    cfg = LLM_REGISTRY[model_key]
    use_custom = cfg.get("token_selection_setting", "default") == "custom"

    if cfg["type"] == "OPENAI":
        kwargs = dict(
            model=cfg["model_id"],
            base_url=cfg.get("host"),
            api_key=cfg.get("key") or "no-key-required",
            streaming=True,
            stream_usage=True,
            timeout=cfg.get("llm_call_timeout", 60),
        )
        if use_custom:
            kwargs["temperature"] = cfg.get("temperature", 1.0)
            kwargs["top_p"]       = cfg.get("top_p", 1.0)
            top_k = cfg.get("top_k")
            if top_k is not None:
                # extra_body forwards top_k as a top-level JSON field in the request.
                # llama.cpp accepts it; real OpenAI API ignores unknown extra_body fields.
                kwargs["extra_body"] = {"top_k": int(top_k)}
        max_tokens = cfg.get("max_tokens")
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        client = ChatOpenAI(**kwargs)
        if use_cache:
            _llm_client_cache[model_key] = client
        return client

    if cfg["type"] == "GEMINI":
        kwargs = dict(
            model=cfg["model_id"],
            google_api_key=cfg.get("key"),
            request_timeout=cfg.get("llm_call_timeout", 60),
        )
        if use_custom:
            kwargs["temperature"] = cfg.get("temperature", 1.0)
            kwargs["top_p"]       = cfg.get("top_p", 0.95)
            top_k = cfg.get("top_k")
            if top_k is not None:
                kwargs["top_k"] = int(top_k)
        max_tokens = cfg.get("max_tokens")
        if max_tokens is not None:
            kwargs["max_output_tokens"] = int(max_tokens)
        client = ChatGoogleGenerativeAI(**kwargs)
        if use_cache:
            _llm_client_cache[model_key] = client
        return client

    raise ValueError(f"Unsupported model type '{cfg['type']}' for model '{model_key}'")


def _content_to_str(content) -> str:
    """
    Normalise AIMessage.content to a plain string.

    LangChain models can return content as:
      - str  (OpenAI, most models)
      - list of dicts (Gemini multimodal, Anthropic content blocks)
        e.g. [{'type': 'text', 'text': '...'}, {'type': 'tool_use', ...}]
      - list of str  (Gemini 2.5 Flash sometimes returns this)

    Extracts all 'text' blocks and joins them. Returns "" for non-text only.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts).strip()
    return ""


def _to_lc_messages(system_prompt: str, messages: list[dict]) -> list[BaseMessage]:
    """
    Convert the internal message format (list of role/content dicts) to
    LangChain BaseMessage objects.

    Internal format:  [{"role": "user"|"assistant"|"system"|"tool", "content": "..."}]
    Tool messages also carry "tool_call_id" and optionally "name".
    """
    lc_msgs: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "user":
            lc_msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            # Preserve tool_calls if present (for history replay)
            tool_calls = m.get("tool_calls")
            if tool_calls:
                lc_msgs.append(AIMessage(content=content, tool_calls=[
                    {"id": tc["id"], "name": tc["function"]["name"],
                     "args": json.loads(tc["function"]["arguments"])}
                    for tc in tool_calls
                ]))
            else:
                lc_msgs.append(AIMessage(content=content))
        elif role == "tool":
            lc_msgs.append(ToolMessage(
                content=content,
                tool_call_id=m.get("tool_call_id", ""),
            ))
    return lc_msgs


# ---------------------------------------------------------------------------
# Global variables for tool definitions (updated dynamically)
# ---------------------------------------------------------------------------

_CURRENT_LC_TOOLS: list = []   # StructuredTool objects — passed to bind_tools()
_CURRENT_OPENAI_TOOLS: list = []  # OpenAI dicts — used by try_force_tool_calls()


def update_tool_definitions():
    """
    Populate tool globals after all plugins are registered.

    Called once by llmem-gw.py after plugin loading completes.
    Also triggers CORE_LC_TOOLS construction (which needs agents imported).
    """
    global _CURRENT_LC_TOOLS, _CURRENT_OPENAI_TOOLS
    import tools as _tools_module
    # Build core LC tools now (agents is fully imported, no circular issue)
    _tools_module.CORE_LC_TOOLS = _tools_module._make_core_lc_tools()
    _CURRENT_LC_TOOLS = get_all_lc_tools()
    _CURRENT_OPENAI_TOOLS = get_all_openai_tools()


def _get_heat_value(heat_curve: list | None, call_count: int) -> int:
    """Return the heat value for a given call count from a heat_curve list.
    Index = call_count - 1 (0-based). Last entry is the cap."""
    if not heat_curve:
        return 3  # default heat if no curve defined
    idx = min(call_count - 1, len(heat_curve) - 1)
    return heat_curve[max(0, idx)]


def _subscribe_toolset(session: dict, ts_name: str, call_count: int | None = None) -> None:
    """Subscribe a toolset in the session, setting heat from its heat_curve.
    call_count=None means first call (call_count=1).
    Updates call_count if already subscribed."""
    subs = session.setdefault("tool_subscriptions", {})
    meta = LLM_TOOLSET_META.get(ts_name, {})
    heat_curve = meta.get("heat_curve")
    existing = subs.get(ts_name)
    if existing is None:
        new_count = call_count if call_count is not None else 1
        subs[ts_name] = {
            "heat": _get_heat_value(heat_curve, new_count),
            "call_count": new_count,
        }
    else:
        new_count = (existing["call_count"] + 1) if call_count is None else call_count
        subs[ts_name] = {
            "heat": _get_heat_value(heat_curve, new_count),
            "call_count": new_count,
        }


def _compute_active_tools(model_key: str, client_id: str) -> set[str]:
    """
    Compute the set of individual tool names that should be active for this invocation.

    Active tools = always_active toolsets + toolsets with heat > 0 in session subscriptions.
    Only considers toolsets the model is authorized to use (in its llm_tools config).
    """
    cfg = LLM_REGISTRY.get(model_key, {})
    authorized_toolsets = cfg.get("llm_tools", [])
    session = sessions.get(client_id, {})
    subs = session.get("tool_subscriptions", {})

    active_tool_names: set[str] = set()
    for ts_name in authorized_toolsets:
        if ts_name not in LLM_TOOLSETS:
            # Literal tool name — treat as always active
            active_tool_names.add(ts_name)
            continue
        meta = LLM_TOOLSET_META.get(ts_name, {})
        if meta.get("always_active", True):
            active_tool_names.update(LLM_TOOLSETS[ts_name])
        else:
            sub = subs.get(ts_name)
            if sub and sub.get("heat", 0) > 0:
                active_tool_names.update(LLM_TOOLSETS[ts_name])
    return active_tool_names


def _get_cold_tool_names(model_key: str, client_id: str) -> list[str]:
    """Return names of individual tools that are authorized but currently cold."""
    cfg = LLM_REGISTRY.get(model_key, {})
    authorized_toolsets = cfg.get("llm_tools", [])
    session = sessions.get(client_id, {})
    subs = session.get("tool_subscriptions", {})

    cold: list[str] = []
    for ts_name in authorized_toolsets:
        if ts_name not in LLM_TOOLSETS:
            continue
        meta = LLM_TOOLSET_META.get(ts_name, {})
        if meta.get("always_active", True):
            continue
        sub = subs.get(ts_name)
        if not sub or sub.get("heat", 0) <= 0:
            cold.extend(LLM_TOOLSETS[ts_name])
    return cold


def _toolset_for_tool(tool_name: str) -> str | None:
    """Return the toolset name that contains the given tool, or None."""
    for ts_name, tools in LLM_TOOLSETS.items():
        if tool_name in tools:
            return ts_name
    return None


def _resolve_model_tools(model_key: str, active_tools: set[str] | None = None) -> list:
    """
    Resolve a model's llm_tools list into StructuredTool objects.

    Each entry in llm_tools is either:
      - A group name (key in LLM_TOOLSETS, e.g. "core", "admin") → expanded to all tools in that group
      - A literal tool name (e.g. "sysprompt_cfg") → included directly

    If active_tools is provided, only tools in that set are returned (hot/cold filtering).
    If active_tools is None, all authorized tools are returned (legacy behaviour for llm_call).
    Returns [] if the model has no llm_tools configured (no tools bound).
    """
    cfg = LLM_REGISTRY.get(model_key, {})
    toolset_names = cfg.get("llm_tools", [])
    if not toolset_names:
        return []

    # Expand toolset names to individual tool names.
    allowed_names: set[str] = set()
    for ts_name in toolset_names:
        if ts_name in LLM_TOOLSETS:
            allowed_names.update(LLM_TOOLSETS[ts_name])
        else:
            allowed_names.add(ts_name)

    # Apply hot/cold filter if active_tools provided
    if active_tools is not None:
        allowed_names &= active_tools

    return [t for t in _CURRENT_LC_TOOLS if t.name in allowed_names]


# --- Universal Rate Limiter ---

# Sliding-window call timestamps: key = "client_id:tool_type" -> [timestamps]
_rate_timestamps: dict[str, list[float]] = {}

# LLM client cache (Level 1 stream optimization): model_key -> ChatOpenAI | ChatGoogleGenerativeAI
# ChatOpenAI/ChatGoogleGenerativeAI are stateless — safe to share across sessions.
_llm_client_cache: dict[str, object] = {}

# Sentence boundary regex for Level 3 astream() chunking.
# Splits after sentence-ending punctuation followed by whitespace.
_SENT_RE = re.compile(r'(?<=[.!?])\s+')


async def check_rate_limit(client_id: str, tool_name: str, tool_type: str) -> tuple[bool, str]:
    """
    Check whether a tool call is within its rate limit.

    Returns (allowed: bool, error_msg: str).
    error_msg is empty when allowed=True.

    When auto_disable=True and the limit is breached for an llm_call tool,
    the rate limit message is returned (but no model state is mutated).
    """
    cfg = RATE_LIMITS.get(tool_type, {})
    max_calls = cfg.get("calls", 0)
    window = cfg.get("window_seconds", 0)

    if max_calls == 0 or window == 0:
        return True, ""  # unlimited

    key = f"{client_id}:{tool_type}"
    now = time.monotonic()

    # Prune timestamps outside the window
    timestamps = _rate_timestamps.get(key, [])
    timestamps = [t for t in timestamps if now - t < window]

    if len(timestamps) >= max_calls:
        auto_disable = cfg.get("auto_disable", False)
        auto_disable_msg = ""

        if auto_disable and tool_type == "llm_call":
            auto_disable_msg = " Rate limit auto-disable triggered for llm_call."

        error_msg = (
            f"RATE LIMIT EXCEEDED: {tool_name} ({tool_type}) — "
            f"limit is {max_calls} calls in {window}s for this session."
            f"{auto_disable_msg}"
        )
        log.warning(f"Rate limit exceeded: client={client_id} tool={tool_name} type={tool_type}")
        return False, error_msg

    timestamps.append(now)
    _rate_timestamps[key] = timestamps
    return True, ""


# --- Gate helpers ---

def _is_llama_client(client_id: str) -> bool:
    """True for llama-proxy and OpenAI-proxy clients that cannot respond to gate prompts."""
    return (
        client_id.startswith("llama-")
        or client_id.startswith("api-swarm-")
    )


def _is_slack_client(client_id: str) -> bool:
    """True for Slack clients that cannot respond to gate prompts."""
    return client_id.startswith("slack-")


def _gate_matches(tool_name: str, tool_args: dict, gate_entry: str) -> bool:
    """
    Check whether a tool call matches a gate entry.

    Gate entry syntax:
      "<tool_name>"               — matches any call to that tool
      "<tool_name> <subcommand>"  — matches only when tool's 'action' arg == subcommand

    Examples:
      "db_query"          matches db_query(sql="SELECT ...")
      "model_cfg write"   matches model_cfg(action="write", ...)
      "model_cfg read"    does NOT match model_cfg(action="write", ...)
    """
    parts = gate_entry.strip().split(None, 1)
    if not parts:
        return False
    gate_tool = parts[0]
    if gate_tool != tool_name:
        return False
    if len(parts) == 1:
        return True  # tool name only — all calls match
    gate_sub = parts[1].strip()
    # Check 'action' field (unified tools) or first positional
    actual_sub = tool_args.get("action") or tool_args.get("operation") or ""
    return actual_sub == gate_sub


async def check_gate(client_id: str, model_key: str, tool_name: str, tool_args: dict) -> tuple[bool, str]:
    """
    Check whether a tool call requires human gate approval.

    Returns (allowed: bool, reason: str).
    reason is empty when allowed=True.

    Gate flow:
    1. Look up llm_tools_gates for the session's current model.
    2. If any gate entry matches the call, prompt the user.
    3. For llama/slack/swarm clients — auto-deny (cannot interactively respond).
    4. For shell.py clients — send a gate prompt event; wait up to 120s.
    """
    cfg = LLM_REGISTRY.get(model_key, {})
    gates = cfg.get("llm_tools_gates", [])
    if not gates:
        return True, ""

    # Check if any gate pattern matches this tool call
    matched_gate = None
    for entry in gates:
        if _gate_matches(tool_name, tool_args, entry):
            matched_gate = entry
            break

    if matched_gate is None:
        return True, ""

    # Auto-deny for non-interactive clients
    if _is_llama_client(client_id) or _is_slack_client(client_id):
        reason = (
            f"GATE DENIED (auto): tool '{tool_name}' is gated for model '{model_key}' "
            f"(gate='{matched_gate}'). This client cannot respond to gate requests. "
            f"Use a shell.py session to approve gated tool calls. Do NOT retry."
        )
        log.info(f"Gate auto-denied: client={client_id} tool={tool_name} gate='{matched_gate}'")
        return False, reason

    # Send gate prompt to user via SSE "gate" event
    args_preview = ", ".join(
        f"{k}={repr(v)[:60]}" for k, v in list(tool_args.items())[:4]
    )
    gate_msg = (
        f"[GATE] Model '{model_key}' wants to call: {tool_name}({args_preview})\n"
        f"Gate rule: '{matched_gate}'\n"
        f"Allow? (y/yes to allow, anything else to deny) [120s timeout]"
    )
    from state import sse_queues, get_queue
    q = await get_queue(client_id)
    q.put_nowait({"t": "gate", "d": gate_msg})

    log.info(f"Gate pending: client={client_id} model={model_key} tool={tool_name} gate='{matched_gate}'")
    approved = await wait_for_gate(client_id, timeout=120.0)

    if approved:
        log.info(f"Gate approved: client={client_id} tool={tool_name}")
        return True, ""
    else:
        reason = (
            f"GATE DENIED: tool call '{tool_name}' was denied by the user (or timed out). "
            f"Do NOT retry the same call. Acknowledge the denial and continue without it."
        )
        log.info(f"Gate denied: client={client_id} tool={tool_name}")
        return False, reason


# --- Tool Execution ---

async def execute_tool(client_id: str, tool_name: str, tool_args: dict) -> str:
    # Set context var so executors can read client_id without it being a parameter
    current_client_id.set(client_id)

    # Set DB routing context for per-model database scoping
    _et_model = sessions.get(client_id, {}).get("model", "")
    set_model_context(_et_model)

    # Universal rate limit check
    tool_type = get_tool_type(tool_name)
    rate_ok, rate_err = await check_rate_limit(client_id, tool_name, tool_type)
    if not rate_ok:
        await push_tok(client_id, f"\n[RATE LIMITED] {tool_name}: {rate_err}\n")
        return rate_err

    # Gate check — requires human approval if tool matches model's llm_tools_gates
    sess_model = sessions.get(client_id, {}).get("model", "")
    gate_ok, gate_err = await check_gate(client_id, sess_model, tool_name, tool_args)
    if not gate_ok:
        await push_tok(client_id, f"\n[GATE DENIED] {tool_name}\n")
        return gate_err

    # Judge tool gate — LLM-as-judge evaluation (no-op when plugin not loaded)
    try:
        import judge as _judge_mod
        _judge_sess = sessions.get(client_id, {})
        _judge_ok, _judge_err = await _judge_mod.check_tool_gate(
            client_id, sess_model, _judge_sess, tool_name, tool_args
        )
        if not _judge_ok:
            await push_tok(client_id, f"\n[JUDGE BLOCKED] {tool_name}\n")
            return _judge_err
    except ImportError:
        pass

    # Get executor function dynamically
    executor = get_tool_executor(tool_name)
    if not executor:
        return f"Unknown tool: {tool_name}"

    # Tool-specific logging and execution
    try:
        # Display tool call info (suppressed if tool_suppress=True)
        _tool_suppress = sessions.get(client_id, {}).get("tool_suppress", False)
        if not _tool_suppress:
            if tool_name == "db_query":
                sql = tool_args.get("sql", "")
                await push_tok(client_id, f"\n[db ▶] {sql}\n")
            elif tool_name == "sysprompt_cfg":
                action = tool_args.get("action", "?")
                model = tool_args.get("model", "")
                file = tool_args.get("file", "")
                label = f"{action}"
                if model:
                    label += f" model={model}"
                if file:
                    label += f" file={file}"
                await push_tok(client_id, f"\n[sysprompt ▶] {label}…\n")
            elif tool_name == "get_system_info":
                await push_tok(client_id, "\n[sysinfo ▶] fetching…\n")
            elif tool_name == "google_search":
                query = tool_args.get("query", "")
                await push_tok(client_id, f"\n[search google ▶] {query}\n")
            elif tool_name == "ddgs_search":
                query = tool_args.get("query", "")
                await push_tok(client_id, f"\n[search ddgs ▶] {query}\n")
            elif tool_name == "tavily_search":
                query = tool_args.get("query", "")
                await push_tok(client_id, f"\n[search tavily ▶] {query}\n")
            elif tool_name == "google_drive":
                op = tool_args.get("operation", "?")
                await push_tok(client_id, f"\n[drive ▶] {op}\n")
            elif tool_name in ("llm_call", "llm_call_clean", "llm_clean_text"):
                pass  # llm_call prints its own [llm_call ▶] tag internally
            else:
                await push_tok(client_id, f"\n[{tool_name} ▶] executing…\n")

        # Execute the tool
        result = await executor(**tool_args)

        # Heat management: subscribe/increment heat for this toolset in the caller's session.
        # This covers delegation paths (llm_call → execute_tool) where agentic_lc's own
        # heat loop never runs, as well as the direct agentic_lc path (idempotent — the
        # loop still handles decay for unused toolsets after this).
        _ts = _toolset_for_tool(tool_name)
        log.info(f"execute_tool heat: tool={tool_name} ts={_ts} client={client_id}")
        if _ts:
            _ts_meta = LLM_TOOLSET_META.get(_ts, {})
            if not _ts_meta.get("always_active", True):
                _caller_session = sessions.get(client_id, {})
                _caller_model = _caller_session.get("model", "")
                _authorized = LLM_REGISTRY.get(_caller_model, {}).get("llm_tools", [])
                # Only track heat for toolsets the calling model is authorized to use.
                # Delegates (one-shot llm_call targets) execute under the shell client's
                # session, so we guard here to avoid subscribing toolsets the outer model
                # (e.g. samaritan-voice) never has access to (e.g. drive).
                _ts_authorized = (_ts in _authorized) or any(
                    t in _authorized for t in LLM_TOOLSETS.get(_ts, [])
                )
                if _ts_authorized:
                    _already_subscribed = _ts in _caller_session.get("tool_subscriptions", {})
                    if not _already_subscribed:
                        # Delegation path: toolset wasn't auto-activated by agentic_lc
                        # (e.g. llm_call → execute_tool → url_extract). Subscribe it now.
                        _subscribe_toolset(_caller_session, _ts)
                        log.info(f"execute_tool heat: subscribed ts={_ts} heat={_caller_session['tool_subscriptions'].get(_ts)}")
                    # Always mark as used this turn to protect from decay at text-exit.
                    _caller_session.setdefault("_toolsets_used_this_turn", set()).add(_ts)
                else:
                    log.debug(f"execute_tool heat: skipped ts={_ts} — not authorized for model={_caller_model}")

        # Tool call memory logging — save a compact summary as an inline ST row when enabled.
        # Logs intent + outcome, NOT raw data. Result is summarized to ~150 chars so the
        # learning trail is readable without bloating memory with search results / web content.
        # source = calling model key — preserves delegation chain: caller → tool → delegatee.
        _sess_for_log = sessions.get(client_id, {})
        _caller_model = _sess_for_log.get("model", "")
        _caller_cfg = LLM_REGISTRY.get(_caller_model, {})
        _model_tool_log = _caller_cfg.get("conv_log_tools")
        _do_tool_log = _model_tool_log if _model_tool_log is not None else TOOL_CALL_LOG_DEFAULT
        if _do_tool_log:
            try:
                from memory import save_memory
                _log_topic = _sess_for_log.get("current_topic") or "tool-call"
                _result_str = str(result)
                # Compact result summary: first non-empty line, capped at 150 chars
                _result_lines = [ln.strip() for ln in _result_str.splitlines() if ln.strip()]
                _result_summary = (_result_lines[0][:150] + "…") if _result_lines and len(_result_lines[0]) > 150 else (_result_lines[0] if _result_lines else "(empty)")
                # Compact args: keep only the key intent fields, drop large payloads
                _arg_summary: dict = {}
                if tool_name in ("llm_call", "llm_call_clean"):
                    _arg_summary["delegatee"] = tool_args.get("model", "")
                    _arg_summary["prompt"] = tool_args.get("prompt", "")[:120]
                    _arg_summary["mode"] = tool_args.get("mode", "text")
                elif tool_name == "db_query":
                    _arg_summary["sql"] = tool_args.get("sql", "")[:120]
                elif tool_name in ("url_extract",):
                    _arg_summary["url"] = tool_args.get("url", "")
                elif tool_name in ("search_ddgs", "search_google", "search_tavily", "search_xai",
                                   "google_search", "ddgs_search", "tavily_search"):
                    _arg_summary["query"] = tool_args.get("query", "")[:100]
                elif tool_name == "google_drive":
                    _arg_summary["op"] = tool_args.get("operation", "")
                    _arg_summary["name"] = tool_args.get("name", tool_args.get("file_name", ""))
                else:
                    # Generic: include scalar args only, skip large string values
                    for _k, _v in tool_args.items():
                        if isinstance(_v, (int, float, bool)):
                            _arg_summary[_k] = _v
                        elif isinstance(_v, str) and len(_v) <= 80:
                            _arg_summary[_k] = _v
                _log_entry = {
                    "caller": _caller_model or "agent",
                    "tool": tool_name,
                    "args": _arg_summary,
                    "status": "ok",
                    "result": _result_summary,
                }
                asyncio.ensure_future(save_memory(
                    topic=_log_topic,
                    content=json.dumps(_log_entry, ensure_ascii=False),
                    importance=3,
                    source=_caller_model or "agent",
                    session_id=client_id,
                ))
            except Exception as _tl_err:
                log.debug(f"tool_call_log save failed for {tool_name}: {_tl_err}")

        # Display result with preview (length controlled by per-session tool_preview_length)
        # -1 = unlimited (no truncation), 0 = tags printed but no content, >0 = truncate to N chars
        sess = sessions.get(client_id, {})
        tool_suppress = sess.get("tool_suppress", False)
        preview_len = sess.get("tool_preview_length", 500)
        result_str = str(result)

        if not tool_suppress:
            if preview_len == 0:
                preview = ""
            elif preview_len == -1 or len(result_str) <= preview_len:
                preview = result_str
            else:
                preview = result_str[:preview_len] + "\n…(truncated)"

            if tool_name == "db_query":
                if preview:
                    await push_tok(client_id, f"[db ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, "[db ◀]\n")
            elif tool_name == "sysprompt_cfg":
                if preview:
                    await push_tok(client_id, f"[sysprompt ◀] {preview}\n")
                else:
                    await push_tok(client_id, "[sysprompt ◀]\n")
            elif tool_name == "get_system_info":
                await push_tok(client_id, f"[sysinfo ◀] {result}\n")
                return json.dumps(result) if isinstance(result, dict) else str(result)
            elif tool_name in ("google_search", "ddgs_search", "tavily_search"):
                label = {"google_search": "google", "ddgs_search": "ddgs", "tavily_search": "tavily"}[tool_name]
                if preview:
                    await push_tok(client_id, f"[search {label} ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, f"[search {label} ◀]\n")
            elif tool_name == "google_drive":
                if preview:
                    await push_tok(client_id, f"[drive ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, "[drive ◀]\n")
            elif tool_name in ("llm_call", "llm_call_clean", "llm_clean_text"):
                pass  # llm_call prints its own [llm_call ◀] tag internally
            else:
                if preview:
                    await push_tok(client_id, f"[{tool_name} ◀]\n{preview}\n")
                else:
                    await push_tok(client_id, f"[{tool_name} ◀]\n")
        elif tool_name == "get_system_info":
            # get_system_info needs the return even when suppressed
            return json.dumps(result) if isinstance(result, dict) else str(result)

        return str(result)

    except Exception as exc:
        error_msg = f"{tool_name} error: {exc}"
        await push_tok(client_id, f"[{tool_name} error] {exc}\n")
        return error_msg

# --- General tool call extraction ---

_SQL_KEYWORDS = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|SHOW|DESCRIBE|TRUNCATE|REPLACE)\b", re.IGNORECASE | re.MULTILINE)

# Three-level nested brace matching — covers {args: {key: {val}}} which handles all real tool call shapes.
_JSON_BLOB_RE = re.compile(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', re.DOTALL)


def _try_parse_json_tool(raw: str) -> tuple[str, dict] | None:
    """Try to parse a JSON blob as a tool call {name, arguments/parameters}.
    Tries raw first, then with {{ }} normalization (llama.cpp template artifact)."""
    for candidate in (raw, raw.replace("{{", "{").replace("}}", "}")):
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                continue
            name = payload.get("name", "")
            if not name or not isinstance(name, str):
                continue
            args = payload.get("arguments", payload.get("parameters", payload.get("input", {})))
            return (name, args if isinstance(args, dict) else {})
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def try_force_tool_calls(text: str, valid_tool_names: set[str] | None = None) -> list[tuple[str, dict, str]]:
    """Extract all tool calls from model text output.

    Handles any format a model might use:
      - <tool_call>{...}</tool_call>  (Qwen, Mistral, Hermes, etc.)
      - [TOOL_CALL]{...}              (some fine-tunes)
      - ```json\\n{...}\\n```          (markdown code block)
      - bare {"name": ..., "arguments": ...} JSON anywhere in text
      - {{ }} brace-escaping (llama.cpp template artifact)

    Tool names are validated against valid_tool_names if provided, otherwise
    against the full live tool registry. This ensures local models can only
    invoke tools that are in their configured toolset.
    Falls back to SQL keyword heuristic as a last resort.
    """
    if valid_tool_names is None:
        from tools import get_all_openai_tools
        valid_tool_names = {t["function"]["name"] for t in get_all_openai_tools()}
    valid_tools = valid_tool_names

    results = []
    seen_calls: set[str] = set()

    for m in _JSON_BLOB_RE.finditer(text):
        parsed = _try_parse_json_tool(m.group(0))
        if parsed is None:
            continue
        name, args = parsed
        if name not in valid_tools:
            continue
        # Deduplicate by name+args fingerprint so the same tool can be called
        # multiple times with different arguments (e.g. read file1, read file2, read file3)
        fingerprint = f"{name}:{json.dumps(args, sort_keys=True)}"
        if fingerprint in seen_calls:
            continue
        seen_calls.add(fingerprint)
        results.append((name, args, f"forced-{uuid.uuid4().hex[:8]}"))

    if results:
        return results

    # Last resort: bare SQL statement with no JSON wrapper
    stripped = text.strip()
    if _SQL_KEYWORDS.match(stripped):
        first_kw = _SQL_KEYWORDS.search(stripped)
        if first_kw and first_kw.start() <= 120:
            return [("db_query", {"sql": stripped[first_kw.start():]}, f"forced-{uuid.uuid4().hex[:8]}")]

    return []

# --- Post-response memory scan ---
#
# For models that narrate tool calls instead of issuing them (e.g. grok-4-1-fast-non-reasoning),
# scan the final text response for memory_save() call syntax and execute any found saves silently
# after the response is already streamed to the user (zero added latency).
#
# Activated per-model via "memory_scan": true in llm-models.json.
# Handles the exact syntax the system prompt teaches:
#   memory_save(topic="...", content="...", importance=N)
#   memory_save(topic='...', content='...', importance=N, source='user')
#
# Also catches JSON-blob form in case the model outputs structured JSON inline.

_MEMORY_SAVE_RE = re.compile(
    r'memory_save\s*\(\s*'
    r'topic\s*=\s*(?P<tq>["\'])(?P<topic>(?:(?!(?P=tq)).)+)(?P=tq)'
    r'\s*,\s*content\s*=\s*(?P<cq>["\'])(?P<content>(?:(?!(?P=cq)).|(?P=cq)(?=\w))+)(?P=cq)'
    r'(?:\s*,\s*importance\s*=\s*(?P<importance>\d+))?'
    r'(?:\s*,\s*source\s*=\s*(?P<sq>["\'])(?P<source>(?:(?!(?P=sq)).)+)(?P=sq))?'
    r'[^)]*\)',
    re.DOTALL | re.IGNORECASE,
)

# xAI XML tool-call format: <xai:function_call name="memory_save"><parameter name="topic">...</parameter>...
_XML_TOOL_CALL_RE = re.compile(
    r'<(?:\w+:)?function_call\s+name=["\'](?P<fn>\w+)["\'][^>]*>'
    r'(?P<body>.*?)'
    r'</(?:\w+:)?function_call>',
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAM_RE = re.compile(
    r'<(?:\w+:)?parameter\s+name=["\'](?P<name>\w+)["\'][^>]*>(?P<value>.*?)</(?:\w+:)?parameter>',
    re.DOTALL | re.IGNORECASE,
)


def _strip_memory_calls(text: str) -> str:
    """Remove memory_save(...) call syntax from response text.

    Strips all three formats the scanner recognises: function-call syntax,
    JSON-blob form, and xAI XML form. Also collapses any blank lines left
    behind so the result reads cleanly.
    """
    # Pass 1: function-call syntax
    text = _MEMORY_SAVE_RE.sub("", text)
    # Pass 2: JSON-blob form — only memory_save blobs
    text = _JSON_BLOB_RE.sub(
        lambda m: "" if (_try_parse_json_tool(m.group(0)) or (None, {}))[0] == "memory_save" else m.group(0),
        text,
    )
    # Pass 3: xAI XML form — only memory_save function_call tags
    text = _XML_TOOL_CALL_RE.sub(
        lambda m: "" if m.group("fn") == "memory_save" else m.group(0),
        text,
    )
    # Collapse runs of blank lines to at most one blank line
    import re as _re
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def _scan_and_save_memories(text: str, client_id: str, model_key: str) -> int:
    """Scan a final text response for memory_save() calls and execute any found.

    Returns the number of memories actually saved (0 if none found or all duplicates).
    Runs silently — no output to the user stream.
    """
    from tools import _memory_save_exec
    saved = 0

    # Judge memory gate helper — no-op if plugin not loaded
    async def _judge_memory_ok(topic: str, content: str) -> bool:
        try:
            import judge as _jm
            _sess = sessions.get(client_id, {})
            allowed, _ = await _jm.check_memory_gate(client_id, model_key, _sess, topic, content)
            if not allowed:
                log.info(f"memory_scan: judge blocked topic={topic!r} client={client_id}")
            return allowed
        except ImportError:
            return True

    # Pass 1: function-call syntax  memory_save(topic="...", content="...", importance=N)
    for m in _MEMORY_SAVE_RE.finditer(text):
        topic = m.group("topic").strip()
        content = m.group("content").strip()
        importance = int(m.group("importance")) if m.group("importance") else 5
        # memory_scan runs on assistant response text — default source is "assistant".
        # Only override if the model explicitly wrote source="user" or source="session".
        raw_source = m.group("source")
        source = raw_source if raw_source in ("user", "session") else "assistant"
        if not topic or not content:
            continue
        if not await _judge_memory_ok(topic, content):
            continue
        try:
            result = await _memory_save_exec(
                topic=topic, content=content, importance=importance, source=source
            )
            if "already persisted" not in result:
                saved += 1
                log.info(
                    f"memory_scan: auto-saved from {model_key} response — "
                    f"topic={topic!r} importance={importance}"
                )
        except Exception as exc:
            log.warning(f"memory_scan: save failed for topic={topic!r}: {exc}")

    # Pass 2: JSON-blob form  {"name": "memory_save", "arguments": {...}}
    if not saved:
        for m in _JSON_BLOB_RE.finditer(text):
            parsed = _try_parse_json_tool(m.group(0))
            if parsed is None:
                continue
            name, args = parsed
            if name != "memory_save":
                continue
            topic = str(args.get("topic", "")).strip()
            content = str(args.get("content", "")).strip()
            importance = int(args.get("importance", 5))
            raw_source = str(args.get("source", ""))
            source = raw_source if raw_source in ("user", "session") else "assistant"
            if not topic or not content:
                continue
            if not await _judge_memory_ok(topic, content):
                continue
            try:
                result = await _memory_save_exec(
                    topic=topic, content=content, importance=importance, source=source
                )
                if "already persisted" not in result:
                    saved += 1
                    log.info(
                        f"memory_scan: auto-saved (JSON form) from {model_key} — "
                        f"topic={topic!r} importance={importance}"
                    )
            except Exception as exc:
                log.warning(f"memory_scan: JSON-form save failed for topic={topic!r}: {exc}")

    # Pass 3: xAI XML format  <xai:function_call name="memory_save"><parameter name="topic">...</parameter>...
    if not saved:
        for m in _XML_TOOL_CALL_RE.finditer(text):
            if m.group("fn") != "memory_save":
                continue
            params = {pm.group("name"): pm.group("value").strip()
                      for pm in _XML_PARAM_RE.finditer(m.group("body"))}
            topic = params.get("topic", "").strip()
            content = params.get("content", "").strip()
            importance = int(params.get("importance", 5))
            raw_source = params.get("source", "")
            source = raw_source if raw_source in ("user", "session") else "assistant"
            if not topic or not content:
                continue
            if not await _judge_memory_ok(topic, content):
                continue
            try:
                result = await _memory_save_exec(
                    topic=topic, content=content, importance=importance, source=source
                )
                if "already persisted" not in result:
                    saved += 1
                    log.info(
                        f"memory_scan: auto-saved (XML form) from {model_key} — "
                        f"topic={topic!r} importance={importance}"
                    )
            except Exception as exc:
                log.warning(f"memory_scan: XML-form save failed for topic={topic!r}: {exc}")

    return saved


# --- Enrichment ---

def _load_enrich_rules() -> list[dict]:
    """Load auto-enrichment rules from db-config.json for the active model's database."""
    from database import get_tables_for_model
    tables = get_tables_for_model()
    rules = tables.get("auto_enrich")
    if not isinstance(rules, list):
        return []
    return [r for r in rules if isinstance(r, dict) and r.get("enabled", True)]


def _memory_cfg() -> dict:
    """Return the memory config block from plugins-enabled.json.
    Defaults to all-enabled if the key is absent (backward-compatible)."""
    import json as _json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")
    try:
        with open(path) as f:
            data = _json.load(f)
        return data.get("plugin_config", {}).get("memory", {})
    except Exception:
        return {}


def _memory_feature(feature: str) -> bool:
    """Return True if a specific memory feature is enabled.
    Master switch: 'enabled' (default True).
    Feature switches: 'context_injection', 'reset_summarize', 'post_response_scan',
                      'fuzzy_dedup' (all default True).
    fuzzy_dedup is read directly by memory.py (_fuzzy_dedup_threshold), not via this function.
    A feature is only active when both the master switch and the feature switch are True.
    """
    cfg = _memory_cfg()
    if not cfg.get("enabled", True):
        return False
    return cfg.get(feature, True)

async def auto_enrich_context(messages: list[dict], client_id: str) -> list[dict]:
    if not messages: return messages
    last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
    if not last_user: return messages

    text = last_user.get("content", "")
    enrichments = []

    # Instance-specific enrichment rules from auto-enrich.json
    # Session flag auto_enrich=False suppresses all rules for this session.
    _auto_enrich_enabled = sessions.get(client_id, {}).get("auto_enrich", True)
    for rule in (_load_enrich_rules() if _auto_enrich_enabled else []):
        pattern = rule.get("pattern", "")
        sql = rule.get("sql", "")
        label = rule.get("label", sql)
        if not pattern or not sql:
            continue
        try:
            if re.search(pattern, text, re.IGNORECASE):
                result = await execute_sql(sql)
                enrichments.append(f"[auto-retrieved via: {label}]\n{result}")
                if not sessions.get(client_id, {}).get("tool_suppress", False):
                    await push_tok(client_id, f"\n[context] Auto-queried: {label}\n")
        except Exception:
            pass

    # Inject short-term memory context block
    # Build a query string from the last few turns for semantic retrieval
    _sess_mem_flag = sessions.get(client_id, {}).get("memory_enabled", None)
    _mem_injection_enabled = (_sess_mem_flag is None or _sess_mem_flag) and _memory_feature("context_injection")
    if _mem_injection_enabled:
        try:
            from memory import load_context_block, load_topic_list
            # Prefer model-derived current_topic as query seed (set from <<topic>> tag each turn).
            # Fall back to last 3 user/assistant turns (excluding system and tool injections).
            _session_ctx = sessions.get(client_id, {})
            current_topic = _session_ctx.get("current_topic", "")
            if current_topic:
                query_text = current_topic
            else:
                # Filter to genuine user/assistant turns — skip system messages and
                # tool injection content (starts with "[Session start" or similar).
                _genuine = [
                    m for m in messages
                    if m.get("role") in ("user", "assistant")
                    and m.get("content")
                    and not m.get("content", "").startswith("[Session start")
                    and not m.get("content", "").startswith("[context]")
                ]
                recent = _genuine[-6:] if len(_genuine) >= 6 else _genuine
                query_text = " ".join(
                    m.get("content", "")[:300] for m in recent
                ).strip()
            mem_block = await load_context_block(
                min_importance=3, query=query_text, user_text=text
            )
            if mem_block:
                enrichments.append(mem_block)
            # Inject known topic list so the model reuses existing slugs rather than coining new ones
            try:
                known_topics = await load_topic_list()
                if known_topics:
                    enrichments.append(
                        "## Recent Topics\n"
                        + ", ".join(known_topics)
                    )
            except Exception as _tl_err:
                log.debug(f"auto_enrich_context: topic list load failed: {_tl_err}")
        except Exception as _mem_err:
            log.warning(f"auto_enrich_context: memory load failed: {_mem_err}")

    if not enrichments: return messages

    inject_content = "## Auto-retrieved context\nBase answer on this data:\n\n" + "\n\n".join(enrichments)
    inject = {"role": "system", "content": inject_content}
    final_messages = list(messages[:-1]) + [inject, messages[-1]]

    # Log prompt composition for diagnostics
    inject_chars = len(inject_content)
    total_chars = sum(len(m.get("content", "")) for m in final_messages)
    msg_count = len(final_messages)
    log.debug(
        f"auto_enrich_context: inject={inject_chars:,} chars, "
        f"total_prompt={total_chars:,} chars, {msg_count} messages"
    )
    return final_messages

# --- Agent Loop ---

async def agentic_lc(model_key: str, messages: list[dict], client_id: str) -> str:
    """
    Single agentic loop for all LLM backends using LangChain.

    Replaces the former agentic_openai() + agentic_gemini() pair.
    Tool schema format (OpenAI dicts) and executor registry are unchanged —
    this is purely an LLM abstraction swap.
    """
    try:
        set_model_context(model_key)
        _stream_level = sessions.get(client_id, {}).get("stream_level", 0)
        llm = _build_lc_llm(model_key, use_cache=(_stream_level >= 1))

        # Compute active tools (always_active + hot subscriptions) for this invocation.
        # This set drives both the API schema and the system prompt tool sections.
        _active_tools = _compute_active_tools(model_key, client_id)
        _cold_tools = _get_cold_tool_names(model_key, client_id)

        # Bind per-model tools so only active tools are sent.
        # This keeps tool count under provider limits (e.g. Gemini 2.5 Flash ~35).
        _model_tools = _resolve_model_tools(model_key, active_tools=_active_tools)
        llm_with_tools = llm.bind_tools(_model_tools) if _model_tools else llm
        log.info(
            f"agentic_lc: model={model_key} stream_level={_stream_level} "
            f"bound {len(_model_tools)} tools ({len(_cold_tools)} cold)"
        )

        # Load per-model system prompt with active tool filter
        model_cfg = LLM_REGISTRY.get(model_key, {})
        sp_folder_rel = model_cfg.get("system_prompt_folder", "")
        if sp_folder_rel and sp_folder_rel.lower() != "none":
            from config import BASE_DIR
            sp_folder_abs = os.path.join(BASE_DIR, sp_folder_rel)
            system_prompt = load_prompt_for_folder(sp_folder_abs, active_tools=_active_tools, cold_tools=_cold_tools or None)
        else:
            system_prompt = ""

        # Per-model timeout for ainvoke (same setting used by llm_call).
        # Prevents indefinite stalls when the LLM API hangs or thinks too long.
        invoke_timeout = model_cfg.get("llm_call_timeout", 120)

        # Convert internal message format to LangChain message objects
        ctx: list[BaseMessage] = _to_lc_messages(system_prompt, messages)

        _suppress = sessions.get(client_id, {}).get("tool_suppress", False)
        # Gemini 2.5 Flash (and other thinking models) silently return empty content
        # with no tool calls when bound to many tools (>~35).  Track the initial ctx
        # length so we can detect first-turn failures and retry with tool_choice='any'.
        _initial_ctx_len = len(ctx)
        _is_gemini = (LLM_REGISTRY.get(model_key, {}).get("type") == "GEMINI")
        # Post-response memory scan: for models that narrate tool calls instead of
        # issuing them, scan the final text for memory_save() syntax and execute saves.
        # Gated by both the model-level "memory_scan" flag and the global feature switch.
        _memory_scan = (
            bool(LLM_REGISTRY.get(model_key, {}).get("memory_scan", False))
            and _memory_feature("post_response_scan")
        )
        # Strip memory_save() text when scan is disabled (model shouldn't write them)
        # OR when session-level suppress is set (cosmetic suppress for scan-enabled models).
        _memory_scan_suppress = (
            not _memory_scan
            or sessions.get(client_id, {}).get("memory_scan_suppress", False)
        )
        # Tool-call loop detection: track the last N consecutive tool-call fingerprints.
        # If the same set of tool+args repeats >= _TOOL_LOOP_THRESHOLD times in a row,
        # the model is stuck in a deterministic loop (Qwen3, Hermes, etc.).
        # Threshold of 3: allows one retry after a dedup/no-op result before aborting.
        _TOOL_LOOP_THRESHOLD = 3
        _last_tc_fingerprint: str = ""
        _tc_repeat_count: int = 0
        _max_iters = LIVE_LIMITS.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
        _iter_count = 0
        while _max_iters == -1 or _iter_count < _max_iters:
            _iter_count += 1
            if not _suppress:
                await push_tok(client_id, "\n[thinking…]\n")
            try:
                if _stream_level >= 3:
                    # Level 3: accumulate astream() chunks, then sentence-push if no tool calls.
                    # Strategy: buffer all chunks (needed to detect tool_calls before pushing text),
                    # then sentence-split and push text chunks if the response is a final answer.
                    # Tool-call turns fall through to the existing tool-execution path unchanged.
                    _chunks = []
                    _text_parts: list[str] = []
                    try:
                        async with asyncio.timeout(invoke_timeout):
                            async for _chunk in llm_with_tools.astream(ctx):
                                _chunks.append(_chunk)
                                if _chunk.content:
                                    _text_parts.append(_chunk.content)
                    except asyncio.TimeoutError:
                        await push_tok(client_id, f"\n[LLM timeout after {invoke_timeout}s — aborting turn]\n")
                        await push_done(client_id)
                        return ""
                    if not _chunks:
                        ai_msg = AIMessage(content="")
                    else:
                        # Aggregate chunks: LangChain AIMessageChunk supports + operator
                        ai_msg = _chunks[0]
                        for _c in _chunks[1:]:
                            ai_msg = ai_msg + _c
                    # Sentence-push only on final-answer turns (no tool calls)
                    _astream_text_pushed = False
                    if not ai_msg.tool_calls and _text_parts:
                        _full_text = "".join(_text_parts)
                        # Check for bare tool calls (local models like Qwen) BEFORE pushing text.
                        # If forced tool calls are found, suppress the narration prose entirely —
                        # the catcher below will execute the tools and re-invoke for a real answer.
                        _model_tool_names_pre = {t.name for t in _model_tools} if _model_tools else None
                        if try_force_tool_calls(_full_text, valid_tool_names=_model_tool_names_pre):
                            pass  # Don't push — let the forced-call catcher handle it below.
                        else:
                            # Strip memory calls on full text BEFORE sentence-splitting:
                            # splitting first can break a memory_save(...) call mid-content
                            # when the content string contains periods, causing partial
                            # fragments that no longer match the regex.
                            if _memory_scan_suppress:
                                _full_text = _strip_memory_calls(_full_text)
                            _sentences = _SENT_RE.split(_full_text)
                            for _s in _sentences:
                                _s = _s.strip()
                                if _s:
                                    await push_tok(client_id, _s + " ")
                            _astream_text_pushed = True
                else:
                    ai_msg: AIMessage = await asyncio.wait_for(
                        llm_with_tools.ainvoke(ctx),
                        timeout=invoke_timeout,
                    )
                    _astream_text_pushed = False
            except asyncio.TimeoutError:
                await push_tok(client_id, f"\n[LLM timeout after {invoke_timeout}s — aborting turn]\n")
                await push_done(client_id)
                return ""
            update_session_token_stats(sessions.get(client_id, {}), getattr(ai_msg, "usage_metadata", None) or {})
            ctx.append(ai_msg)

            if not ai_msg.tool_calls:
                # Decay heat for unused toolsets on text-only turns (no tool calls).
                _session = sessions.get(client_id, {})
                _subs = _session.get("tool_subscriptions", {})
                _used_ts = _session.pop("_toolsets_used_this_turn", set())
                for _ts in list(_subs.keys()):
                    if LLM_TOOLSET_META.get(_ts, {}).get("always_active", True):
                        continue
                    if _ts not in _used_ts:
                        _subs[_ts]["heat"] = max(0, _subs[_ts].get("heat", 0) - 1)
                        if _subs[_ts]["heat"] == 0:
                            log.debug(f"agentic_lc: toolset={_ts} decayed to cold (no-tool turn) client={client_id}")
                # Check for bare/XML tool calls from local models (Qwen, Hermes, etc.)
                raw_text = _content_to_str(ai_msg.content)
                _model_tool_names = {t.name for t in _model_tools} if _model_tools else None
                forced_calls = try_force_tool_calls(raw_text, valid_tool_names=_model_tool_names)
                if forced_calls:
                    tool_results = []
                    for tool_name, tool_args, _call_id in forced_calls:
                        if not _suppress:
                            await push_tok(client_id, f"\n[catcher] Detected bare {tool_name} call…\n")
                        result = await execute_tool(client_id, tool_name, tool_args)
                        tool_results.append(f"[Tool result for {tool_name}]: {result}")
                    # Inject results as a user turn (plain text — local models don't
                    # understand the ToolMessage format)
                    ctx.append(HumanMessage(content="\n\n".join(tool_results)))
                    continue

                # No tool calls — final answer
                final = _content_to_str(ai_msg.content)
                if final:
                    # Level 3: text was already sentence-pushed via astream; skip re-push
                    if not _astream_text_pushed:
                        await push_tok(client_id, _strip_memory_calls(final) if _memory_scan_suppress else final)
                else:
                    # Gemini 2.5 Flash returns empty content + no tool calls when
                    # too many tools are bound (>~35). On first turn only, retry
                    # with tool_choice='any' to force a response.
                    if _is_gemini and len(ctx) == _initial_ctx_len + 1:
                        log.warning(
                            f"agentic_lc: empty first-turn from {model_key}, retrying with tool_choice='any'"
                        )
                        ctx.pop()  # remove the empty ai_msg
                        try:
                            llm_forced = llm.bind_tools(_model_tools, tool_choice="any") if _model_tools else llm
                            ai_msg = await asyncio.wait_for(
                                llm_forced.ainvoke(ctx),
                                timeout=invoke_timeout,
                            )
                        except asyncio.TimeoutError:
                            await push_tok(client_id, f"\n[LLM timeout after {invoke_timeout}s — aborting turn]\n")
                            await push_done(client_id)
                            return ""
                        update_session_token_stats(sessions.get(client_id, {}), getattr(ai_msg, "usage_metadata", None) or {})
                        ctx.append(ai_msg)
                        if ai_msg.tool_calls:
                            # Tool calls generated — fall through to execute them below
                            pass
                        else:
                            final = _content_to_str(ai_msg.content)
                            if final:
                                await push_tok(client_id, _strip_memory_calls(final) if _memory_scan_suppress else final)
                            else:
                                log.warning(
                                    f"agentic_lc: empty response after retry from model={model_key} "
                                    f"client={client_id} ctx_len={len(ctx)} "
                                    f"raw_content={repr(ai_msg.content)} "
                                    f"metadata={getattr(ai_msg, 'response_metadata', {})}"
                                )
                                await push_tok(client_id, "[empty string]")
                            if _memory_scan and final:
                                await _scan_and_save_memories(final, client_id, model_key)
                            await push_done(client_id)
                            return final
                    else:
                        # Empty response after tool results — Gemini sometimes returns
                        # empty content when the answer is trivially obvious from the
                        # tool result. Retry unbound (no tools) to force a text reply.
                        if _is_gemini:
                            # Gemini returns empty content + STOP after a tool result when
                            # context is large. Inject an explicit nudge and retry once.
                            log.warning(
                                f"agentic_lc: empty post-tool response from {model_key}, "
                                f"injecting nudge and retrying once — "
                                f"finish_reason={getattr(ai_msg, 'response_metadata', {}).get('finish_reason')}"
                            )
                            ctx.pop()  # remove the empty ai_msg
                            ctx.append(HumanMessage(content="Please provide your final answer now as plain text."))
                            try:
                                ai_msg = await asyncio.wait_for(
                                    llm_with_tools.ainvoke(ctx),
                                    timeout=invoke_timeout,
                                )
                            except asyncio.TimeoutError:
                                await push_tok(client_id, f"\n[LLM timeout after {invoke_timeout}s — aborting turn]\n")
                                await push_done(client_id)
                                return ""
                            update_session_token_stats(sessions.get(client_id, {}), getattr(ai_msg, "usage_metadata", None) or {})
                            final = _content_to_str(ai_msg.content)
                            if final:
                                await push_tok(client_id, _strip_memory_calls(final) if _memory_scan_suppress else final)
                            else:
                                log.warning(
                                    f"agentic_lc: empty response after nudge from model={model_key} "
                                    f"finish_reason={getattr(ai_msg, 'response_metadata', {}).get('finish_reason')} "
                                    f"tool_calls={getattr(ai_msg, 'tool_calls', 'n/a')}"
                                )
                                await push_tok(client_id, "[empty string]")
                            if _memory_scan and final:
                                await _scan_and_save_memories(final, client_id, model_key)
                            await push_done(client_id)
                            return final
                        log.warning(
                            f"agentic_lc: empty response from model={model_key} "
                            f"client={client_id} ctx_len={len(ctx)} "
                            f"raw_content={repr(ai_msg.content)} "
                            f"metadata={getattr(ai_msg, 'response_metadata', {})}"
                        )
                        await push_tok(client_id, "[empty string]")
                        if _memory_scan and final:
                            await _scan_and_save_memories(final, client_id, model_key)
                        await push_done(client_id)
                        return final
                if not ai_msg.tool_calls:
                    if _memory_scan and final:
                        await _scan_and_save_memories(final, client_id, model_key)
                    await push_done(client_id)
                    return final

            # --- Auto-subscribe: check if any tool calls target cold toolsets ---
            # If a cold tool is called (e.g. from conversation history context),
            # subscribe its toolset and re-invoke with the updated schema + prompt.
            _cold_toolsets_hit: list[str] = []
            for _tc in ai_msg.tool_calls:
                _ts = _toolset_for_tool(_tc["name"])
                if _ts and _ts in (sessions.get(client_id, {}).get("tool_subscriptions", {})):
                    pass  # already subscribed
                elif _ts:
                    _ts_meta = LLM_TOOLSET_META.get(_ts, {})
                    if not _ts_meta.get("always_active", True):
                        _cold_toolsets_hit.append(_ts)

            if _cold_toolsets_hit:
                _session = sessions.get(client_id, {})
                for _ts in _cold_toolsets_hit:
                    _subscribe_toolset(_session, _ts, call_count=1)
                    log.info(f"agentic_lc: auto-subscribed cold toolset={_ts} for client={client_id}")
                if not _suppress:
                    await push_tok(client_id, f"\n[activating tools: {', '.join(_cold_toolsets_hit)}…]\n")
                # Recompute active tools and rebuild llm_with_tools + system_prompt
                _active_tools = _compute_active_tools(model_key, client_id)
                _cold_tools = _get_cold_tool_names(model_key, client_id)
                _model_tools = _resolve_model_tools(model_key, active_tools=_active_tools)
                llm_with_tools = llm.bind_tools(_model_tools) if _model_tools else llm
                # Remove the ai_msg that referenced cold tools and re-invoke
                ctx.pop()
                try:
                    if _stream_level >= 3:
                        _chunks = []
                        async with asyncio.timeout(invoke_timeout):
                            async for _chunk in llm_with_tools.astream(ctx):
                                _chunks.append(_chunk)
                        ai_msg = _chunks[0] if _chunks else AIMessage(content="")
                        for _c in _chunks[1:]:
                            ai_msg = ai_msg + _c
                    else:
                        ai_msg = await asyncio.wait_for(
                            llm_with_tools.ainvoke(ctx),
                            timeout=invoke_timeout,
                        )
                except asyncio.TimeoutError:
                    await push_tok(client_id, f"\n[LLM timeout after {invoke_timeout}s — aborting turn]\n")
                    await push_done(client_id)
                    return ""
                update_session_token_stats(sessions.get(client_id, {}), getattr(ai_msg, "usage_metadata", None) or {})
                ctx.append(ai_msg)
                if not ai_msg.tool_calls:
                    final = _content_to_str(ai_msg.content)
                    if final:
                        await push_tok(client_id, final)
                    if _memory_scan and final:
                        await _scan_and_save_memories(final, client_id, model_key)
                    await push_done(client_id)
                    return final

            # Execute all tool calls in this turn
            # --- Loop detection ---
            _tc_fp = "|".join(
                sorted(f"{tc['name']}:{json.dumps(tc.get('args', {}), sort_keys=True)}"
                       for tc in ai_msg.tool_calls)
            )
            if _tc_fp == _last_tc_fingerprint:
                _tc_repeat_count += 1
            else:
                _last_tc_fingerprint = _tc_fp
                _tc_repeat_count = 1
            if _tc_repeat_count >= _TOOL_LOOP_THRESHOLD:
                log.warning(
                    f"agentic_lc: tool-call loop detected for model={model_key} "
                    f"(repeated {_tc_repeat_count}x): {_tc_fp[:120]}"
                )
                # Execute pending tools first so tool_call_ids are resolved in ctx
                # before injecting the break message. Skipping this causes a 400 from
                # OpenAI ("tool_calls must be followed by tool messages").
                for tc in ai_msg.tool_calls:
                    result = await execute_tool(client_id, tc["name"], tc["args"])
                    ctx.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                await push_tok(client_id, "\n[tool-loop detected — asking model to respond in text]\n")
                ctx.append(HumanMessage(
                    content="You have already called the same tool(s) with the same arguments multiple times "
                            "and received the results. Do NOT call any more tools. "
                            "Provide your final answer as plain text now."
                ))
                # Ask model one more time without tools to force a text response
                final = ""
                try:
                    llm_no_tools = llm  # unbound — no tools
                    ai_final: AIMessage = await asyncio.wait_for(
                        llm_no_tools.ainvoke(ctx),
                        timeout=invoke_timeout,
                    )
                    update_session_token_stats(sessions.get(client_id, {}), getattr(ai_final, "usage_metadata", None) or {})
                    final = _content_to_str(ai_final.content)
                    if final:
                        await push_tok(client_id, _strip_memory_calls(final) if _memory_scan_suppress else final)
                    else:
                        await push_tok(client_id, "[no response after loop break]")
                except asyncio.TimeoutError:
                    await push_tok(client_id, f"\n[LLM timeout after {invoke_timeout}s — aborting turn]\n")
                await push_done(client_id)
                return final
            # ---------------------
            has_non_agent_call_output = False
            _tools_used_this_turn: set[str] = set()
            for tc in ai_msg.tool_calls:
                is_streaming_agent_call = (
                    tc["name"] == "agent_call"
                    and tc["args"].get("stream", True)
                )
                if tc["name"] != "agent_call":
                    has_non_agent_call_output = True
                result = await execute_tool(client_id, tc["name"], tc["args"])
                ctx.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                _tools_used_this_turn.add(tc["name"])
                # Flush immediately after each streaming agent_call so Slack posts
                # per-turn progress as it arrives rather than batching all turns.
                if is_streaming_agent_call:
                    await push_done(client_id)

            # Heat management: no decay inside the tool-call loop.
            # Decay fires exactly once per outer-model turn at the text-exit below.
            # This ensures delegate hops (llm_call → execute_tool → url_extract) don't
            # create extra decay opportunities — from the outer model's perspective the
            # entire delegation is a single tool call within one turn.

            # Signal end of this tool-call round trip for non-agent_call tools.
            # push_flush (not push_done) keeps api_client connected across tool
            # round trips while still letting shell.py display intermediate results.
            if has_non_agent_call_output:
                await push_flush(client_id)

        await push_tok(client_id, "\n[Max iterations]\n")
        await push_done(client_id)
        return ""

    except Exception as exc:
        await push_err(client_id, str(exc))
        return ""

async def llm_call(
    model: str,
    prompt: str,
    mode: str = "text",
    sys_prompt: str = "none",
    history: str = "none",
    tool: str = "",
) -> str:
    """
    Unified LLM-to-LLM call.

    Parameters
    ----------
    model      : Target model key (must exist in LLM_REGISTRY).
    prompt     : The prompt / user message to send.
    mode       : "text" — return raw text response.
                 "tool" — delegate a single tool call; requires `tool` argument.
    sys_prompt : "none"   — no system prompt sent to target.
                 "caller" — use the calling session's assembled system prompt.
                 "target" — load the target model's own system_prompt_folder.
    history    : "none"   — no history; clean single-turn call.
                 "caller" — prepend the calling session's full chat history.
    tool       : Tool name (required when mode="tool").

    client_id is read from the current_client_id ContextVar (set by execute_tool).
    """
    from tools import get_tool_executor, get_section_for_tool, get_openai_tool_schema
    client_id = current_client_id.get("")

    # ---- Validate model ----
    cfg = LLM_REGISTRY.get(model)
    if not cfg:
        return f"ERROR: Unknown model '{model}'. Use llm_list() to see available models."
    # Permission = presence of llm_call tool in model's toolset (checked by caller)

    # ---- Validate mode ----
    if mode not in ("text", "tool"):
        return f"ERROR: mode must be 'text' or 'tool', got '{mode}'."
    if sys_prompt not in ("none", "caller", "target"):
        return f"ERROR: sys_prompt must be 'none', 'caller', or 'target', got '{sys_prompt}'."
    if history not in ("none", "caller"):
        return f"ERROR: history must be 'none' or 'caller', got '{history}'."

    session = sessions.get(client_id, {})
    timeout = cfg.get("llm_call_timeout", 60)

    # ---- Resolve system prompt string ----
    resolved_sys: str = ""
    if sys_prompt == "caller":
        caller_model = session.get("model", "")
        caller_cfg = LLM_REGISTRY.get(caller_model, {})
        caller_sp_rel = caller_cfg.get("system_prompt_folder", "")
        if caller_sp_rel and caller_sp_rel.lower() != "none":
            caller_sp_abs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), caller_sp_rel
            )
            resolved_sys = load_prompt_for_folder(caller_sp_abs)
    elif sys_prompt == "target":
        sp_folder_rel = cfg.get("system_prompt_folder", "")
        if sp_folder_rel and sp_folder_rel.lower() != "none":
            sp_folder_abs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), sp_folder_rel
            )
            resolved_sys = load_prompt_for_folder(sp_folder_abs)

    # ---- Depth guard (only when passing caller history, risk of recursion) ----
    if history == "caller":
        depth = session.get("_at_llm_depth", 0)
        _max_depth = LIVE_LIMITS.get("max_at_llm_depth", 1)
        if depth >= _max_depth:
            msg = (
                f"llm_call DEPTH LIMIT REACHED (max={_max_depth}): "
                f"Cannot call llm_call with history=caller from within an llm_call context. "
                f"Do NOT retry. Return your answer directly."
            )
            await push_tok(client_id, f"\n[llm_call ✗] depth limit reached (max={_max_depth})\n")
            log.warning(f"llm_call depth limit: client={client_id} depth={depth} model={model}")
            return msg

    # ---- Build message list ----
    messages: list = []

    if resolved_sys:
        messages.append(SystemMessage(content=resolved_sys))

    if history == "caller":
        for msg in session.get("history", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))
            # skip system turns already in history

    messages.append(HumanMessage(content=prompt))

    # ---- Display tag ----
    tag_mode = f"{mode}/{tool}" if mode == "tool" and tool else mode
    tag_sp = f"sp={sys_prompt}"
    tag_h = f"hist={history}"
    if not session.get("tool_suppress", False):
        await push_tok(
            client_id,
            f"\n[llm_call ▶] {model} [{tag_mode} {tag_sp} {tag_h}]:"
            f" {prompt[:80]}{'…' if len(prompt) > 80 else ''}\n"
        )

    # ---- Temp model flag for history=caller ----
    prev_temp = session.get("_temp_model_active", False)
    if history == "caller":
        session["_at_llm_depth"] = session.get("_at_llm_depth", 0) + 1
        session["_temp_model_active"] = True

    try:
        # ================================================================
        # TEXT MODE
        # ================================================================
        if mode == "text":
            try:
                llm = _build_lc_llm(model)
                response = await asyncio.wait_for(
                    llm.ainvoke(messages),
                    timeout=timeout,
                )
                result = _content_to_str(response.content)

                _sess = sessions.get(client_id, {})
                if not _sess.get("tool_suppress", False):
                    preview_len = _sess.get("tool_preview_length", 500)
                    if preview_len == 0:
                        await push_tok(client_id, f"[llm_call ◀] {model}:\n")
                    elif preview_len == -1 or len(result) <= preview_len:
                        await push_tok(client_id, f"[llm_call ◀] {model}:\n{result}\n")
                    else:
                        await push_tok(client_id, f"[llm_call ◀] {model}:\n{result[:preview_len]}\n…(truncated)\n")
                return result

            except asyncio.TimeoutError:
                msg = f"ERROR: llm_call timed out after {timeout}s waiting for model '{model}'."
                await push_tok(client_id, f"[llm_call ✗] {model}: timeout after {timeout}s\n")
                log.warning(f"llm_call timeout: model={model} client={client_id}")
                return msg
            except Exception as exc:
                msg = f"ERROR: llm_call failed for model '{model}': {exc}"
                await push_tok(client_id, f"[llm_call ✗] {model}: {exc}\n")
                log.error(f"llm_call error: model={model} client={client_id} exc={exc}")
                return msg

        # ================================================================
        # TOOL MODE
        # ================================================================
        else:
            if not tool:
                return "ERROR: mode='tool' requires a tool name in the 'tool' argument."

            executor = get_tool_executor(tool)
            if not executor:
                return f"ERROR: Unknown tool '{tool}'. Use !help for available tools."

            # Determine the tool's system prompt section to use as the leading system message
            # ONLY when no system prompt was already resolved (none or target without folder)
            _tool_sp_rel = cfg.get("system_prompt_folder", "")
            _tool_sp_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), _tool_sp_rel) if _tool_sp_rel and _tool_sp_rel.lower() != "none" else None
            tool_sys = get_section_for_tool(tool, folder=_tool_sp_abs)
            if not tool_sys:
                tool_sys = f"You have access to one tool: {tool}. Use it to answer the user's request."

            # If the caller didn't request a sys_prompt, inject the tool-specific section
            if sys_prompt == "none":
                # Replace the system messages with the tool-specific prompt
                # (messages so far = just HumanMessage(prompt), possibly with history)
                messages = (
                    [SystemMessage(content=tool_sys)]
                    + [m for m in messages if not isinstance(m, SystemMessage)]
                )

            tool_schema = get_openai_tool_schema(tool)
            if not tool_schema:
                return f"ERROR: No schema found for tool '{tool}'. Cannot delegate."

            try:
                async def _run_tool():
                    from langchain_core.tools import StructuredTool as _ST
                    lc_tool = _ST.from_function(
                        coroutine=executor,
                        name=tool_schema["function"]["name"],
                        description=tool_schema["function"].get("description", ""),
                    )
                    llm = _build_lc_llm(model)
                    llm_with_tool = llm.bind_tools([lc_tool])

                    ai_msg: AIMessage = await llm_with_tool.ainvoke(messages)

                    if not ai_msg.tool_calls:
                        return _content_to_str(ai_msg.content)

                    # Execute ALL parallel tool calls — orphaning any yields a 400
                    tool_msgs = []
                    last_result = ""
                    for tc in ai_msg.tool_calls:
                        last_result = str(await execute_tool(client_id, tc["name"], tc["args"]))
                        tool_msgs.append(ToolMessage(content=last_result, tool_call_id=tc["id"]))

                    turn2_msgs = messages + [ai_msg] + tool_msgs
                    final_msg: AIMessage = await llm.ainvoke(turn2_msgs)
                    return _content_to_str(final_msg.content) or last_result

                result = await asyncio.wait_for(_run_tool(), timeout=timeout)

                _sess = sessions.get(client_id, {})
                if not _sess.get("tool_suppress", False):
                    preview_len = _sess.get("tool_preview_length", 500)
                    preview = (
                        result if (preview_len == 0 or len(result) <= preview_len)
                        else result[:preview_len] + "\n…(truncated)"
                    )
                    await push_tok(client_id, f"[llm_call ◀] {model}/{tool}:\n{preview}\n")
                return result

            except asyncio.TimeoutError:
                msg = f"ERROR: llm_call timed out after {timeout}s for model '{model}'."
                await push_tok(client_id, f"[llm_call ✗] {model}/{tool}: timeout after {timeout}s\n")
                return msg
            except Exception as exc:
                msg = f"ERROR: llm_call failed for model '{model}', tool '{tool}': {exc}"
                await push_tok(client_id, f"[llm_call ✗] {model}/{tool}: {exc}\n")
                log.error(f"llm_call error: model={model} tool={tool} client={client_id} exc={exc}")
                return msg

    finally:
        if history == "caller":
            session["_at_llm_depth"] = max(0, session.get("_at_llm_depth", 1) - 1)
            session["_temp_model_active"] = prev_temp


async def llm_list() -> str:
    """Return a formatted list of all models in LLM_REGISTRY with their details."""
    if not LLM_REGISTRY:
        return "No models registered."

    lines = ["Available LLM models:\n"]
    for name, cfg in sorted(LLM_REGISTRY.items()):
        host = cfg.get("host") or "default"
        toolsets = cfg.get("llm_tools", [])
        lines.append(
            f"  {name}\n"
            f"    type             : {cfg.get('type')}\n"
            f"    model_id         : {cfg.get('model_id')}\n"
            f"    host             : {host}\n"
            f"    max_context      : {cfg.get('max_context')}\n"
            f"    llm_tools        : {', '.join(toolsets) if toolsets else '(none)'}\n"
            f"    llm_call_timeout : {cfg.get('llm_call_timeout', 60)}s\n"
            f"    description      : {cfg.get('description', '')}\n"
        )
    return "\n".join(lines)


async def agent_call(
    agent_url: str,
    message: str,
    target_client_id: str = None,
    stream: bool = True,
) -> str:
    """
    Call another agent-mcp instance (swarm/multi-agent coordination).

    Sends `message` to a remote agent at `agent_url` using the API client plugin.
    The remote agent processes it through its full stack (LLM, tools).
    Returns the complete text response.

    When stream=True (default), remote tokens are relayed via push_tok in real-time
    so Slack and other clients see per-turn progress as it arrives.
    When stream=False, the call blocks silently until the remote agent finishes and
    returns only the final result (original behaviour).

    Depth guard: calls originating from an api-swarm- prefixed client_id are
    rejected immediately to prevent unbounded recursion (max 1 hop).

    Session persistence: the remote session_id is derived deterministically from the
    calling session + agent URL, so repeated calls from the same human session to
    the same remote agent reuse the same remote session (history is preserved).
    Pass target_client_id to override and use a specific named session.
    """
    from api_client import AgentClient

    calling_client = current_client_id.get("")

    # Depth guard — track nesting depth in the calling session
    calling_session = sessions.get(calling_client, {})
    agent_call_depth = calling_session.get("_agent_call_depth", 0)
    _max_agent_call_depth = LIVE_LIMITS.get("max_agent_call_depth", 1)
    if agent_call_depth >= _max_agent_call_depth:
        msg = (
            f"[agent_call] Depth limit reached (max={_max_agent_call_depth}). "
            f"Call rejected to prevent recursion. Do NOT retry."
        )
        log.warning(f"agent_call depth limit: client={calling_client} depth={agent_call_depth}")
        return msg

    # Outbound agent message filter
    block_reason = _check_outbound_agent_message(message)
    if block_reason:
        log.warning(f"agent_call filter block: client={calling_client} reason={block_reason}")
        return f"ERROR: {block_reason}\nMessage was NOT sent to the remote agent."

    # Session-level streaming override: !stream true|false takes precedence over
    # the LLM-supplied stream parameter so the human always has final control.
    session_stream = sessions.get(calling_client, {}).get("agent_call_stream", None)
    if session_stream is not None:
        stream = session_stream

    # Derive a stable swarm client_id from calling session + agent URL so the
    # remote session persists across multiple agent_call invocations (same human
    # session talking to same remote agent = same remote session).
    # The LLM can still override with an explicit target_client_id.
    if target_client_id:
        swarm_client_id = target_client_id
    else:
        import hashlib
        key = f"{calling_client}:{agent_url}"
        swarm_client_id = f"api-swarm-{hashlib.md5(key.encode()).hexdigest()[:8]}"
    api_key = os.getenv("API_KEY", "") or None
    timeout = 120

    if not calling_session.get("tool_suppress", False):
        await push_tok(calling_client, f"\n[agent_call ▶] {agent_url} → {swarm_client_id}: {message[:100]}{'…' if len(message) > 100 else ''}\n")

    calling_session["_agent_call_depth"] = agent_call_depth + 1
    try:
        client = AgentClient(agent_url, client_id=swarm_client_id, api_key=api_key)

        if stream:
            # Streaming path: relay each remote token via push_tok so Slack and
            # other clients see the remote agent's output as it arrives, rather
            # than waiting for the full response before seeing anything.
            accumulated = []
            async for chunk in client.stream(message, timeout=timeout):
                accumulated.append(chunk)
                await push_tok(calling_client, chunk)
            return "".join(accumulated)
        else:
            # Non-streaming path: block until remote agent finishes, return full result.
            # The LLM narrates the result itself; no push_tok preview here to avoid
            # double-echo in Slack and other clients.
            result = await asyncio.wait_for(
                client.send(message, timeout=timeout),
                timeout=timeout + 5,
            )
            return result

    except asyncio.TimeoutError:
        msg = f"ERROR: agent_call timed out after {timeout}s waiting for {agent_url}."
        await push_tok(calling_client, f"[agent_call ✗] timeout after {timeout}s\n")
        log.warning(f"agent_call timeout: url={agent_url} swarm_client={swarm_client_id}")
        return msg
    except Exception as exc:
        msg = f"ERROR: agent_call failed for {agent_url}: {exc}"
        await push_tok(calling_client, f"[agent_call ✗] {exc}\n")
        log.error(f"agent_call error: url={agent_url} exc={exc}")
        return msg
    finally:
        calling_session["_agent_call_depth"] = agent_call_depth



async def _call_llm_text(model_key: str, prompt: str) -> str:
    """
    Minimal fire-and-forget LLM call with no tools, no history, no streaming.
    Used internally by the memory summarizer. Returns plain text response.
    """
    if model_key not in LLM_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")
    cfg = LLM_REGISTRY[model_key]
    timeout = cfg.get("llm_call_timeout", 60)
    llm = _build_lc_llm(model_key)
    msgs = [SystemMessage(content="You are a concise assistant."), HumanMessage(content=prompt)]
    response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
    return _content_to_str(response.content)


async def dispatch_llm(model_key: str, messages: list[dict], client_id: str) -> str:
    if model_key not in LLM_REGISTRY:
        await push_err(client_id, f"Unknown model: '{model_key}'")
        return ""

    # Session-start tool_list auto-inject: on the first turn of a session, inject a
    # synthetic assistant message containing the tool_list result so the model starts
    # with full situational awareness of its capability space.
    _session = sessions.get(client_id, {})
    if not _session.get("tool_list_injected", False):
        _session["tool_list_injected"] = True
        try:
            from tools import _tool_list_exec
            from state import current_client_id as _cid_ctx
            _tok = _cid_ctx.set(client_id)
            _tl_result = await _tool_list_exec(action="list")
            _cid_ctx.reset(_tok)
            inject_msg = {
                "role": "assistant",
                "content": f"[Session start — authorized tools]\n{_tl_result}",
            }
            messages = [inject_msg] + list(messages)
            log.info(f"dispatch_llm: injected tool_list for client={client_id}")
        except Exception as _tl_err:
            log.warning(f"dispatch_llm: tool_list inject failed: {_tl_err}")

    _stream_level = sessions.get(client_id, {}).get("stream_level", 0)
    if _stream_level >= 2:
        # Level 2: fire enrichment concurrently with LLM invocation setup.
        # Race against 300ms timeout — inject if ready, defer to next turn if slow.
        # asyncio.shield() prevents the enrich task from being cancelled when the
        # timeout fires; it completes in the background and is GC'd naturally.
        enrich_task = asyncio.create_task(auto_enrich_context(messages, client_id))
        try:
            messages = await asyncio.wait_for(asyncio.shield(enrich_task), timeout=0.3)
        except asyncio.TimeoutError:
            log.debug("dispatch_llm: enrich timeout (>300ms), proceeding without this turn's enrichment")
    else:
        messages = await auto_enrich_context(messages, client_id)

    return await agentic_lc(model_key, messages, client_id)