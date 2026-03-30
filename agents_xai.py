"""agents_xai.py — Responses API stateful conversation handler (xAI + OpenAI).

Used when xai_responses_api=true or openai_responses_api=true in llm-models.json.

The Responses API (POST {host}/responses) stores conversation state server-side,
so only new messages need to be sent after the first turn instead of full history.

Key differences from agentic_lc():
  - Endpoint: {host}/responses  (not {host}/chat/completions)
  - Request:  `input` array  (not `messages`)
  - Response: `output` array  (not `choices`)
  - After turn 1: only new user message is sent; provider holds the rest
  - Tool results: `function_call_output` items in `input` (not ToolMessage objects)
  - session["responses_api_id"] tracks the chain tip; cleared on !reset

Supported providers:
  - xAI (api.x.ai):    xai_responses_api=true    — content type: output_text
  - OpenAI (api.openai.com): openai_responses_api=true — content type: output_text

Integration:
  - dispatch_llm() in agents.py calls agentic_responses_api() when is_responses_api() is True
  - config.py whitelist includes xai_responses_api and openai_responses_api
  - cmd_reset() and cmd_set_model() in routes.py clear session["responses_api_id"]
"""

import asyncio
import json
import os

import httpx

from config import LLM_REGISTRY, LIVE_LIMITS, MAX_TOOL_ITERATIONS, BASE_DIR, log


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def is_xai_responses_api(model_key: str) -> bool:
    """Return True if this model should use the xAI Responses API."""
    cfg = LLM_REGISTRY.get(model_key, {})
    return (
        bool(cfg.get("xai_responses_api", False))
        and "x.ai" in (cfg.get("host") or "")
    )


def is_openai_responses_api(model_key: str) -> bool:
    """Return True if this model should use the OpenAI Responses API."""
    cfg = LLM_REGISTRY.get(model_key, {})
    return (
        bool(cfg.get("openai_responses_api", False))
        and "openai.com" in (cfg.get("host") or "")
    )


def is_responses_api(model_key: str) -> bool:
    """Return True if this model uses any Responses API (xAI or OpenAI)."""
    return is_xai_responses_api(model_key) or is_openai_responses_api(model_key)


# Keep old name as alias for backward compatibility with agents.py import
is_xai_stateful = is_xai_responses_api


def _responses_api_url(cfg: dict) -> str:
    """Derive the Responses API endpoint from the model's configured host.

    Host already includes the /v1 prefix (e.g. https://api.x.ai/v1),
    so we just append /responses.
    """
    host = (cfg.get("host") or "").rstrip("/")
    return f"{host}/responses"


def _provider_label(cfg: dict) -> str:
    """Return 'xAI' or 'OpenAI' for log messages."""
    host = cfg.get("host") or ""
    if "x.ai" in host:
        return "xAI"
    if "openai.com" in host:
        return "OpenAI"
    return "Responses API"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _lc_tools_to_openai_schemas(lc_tools: list) -> list[dict]:
    """Convert LangChain StructuredTool objects to Responses API tool format.

    The Responses API uses a FLAT format:
        {"type": "function", "name": ..., "description": ..., "parameters": ...}

    This differs from chat completions which nests under "function":
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    We use convert_to_openai_tool() then flatten the result.
    """
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool
    except ImportError:
        log.warning("agents_xai: langchain_core.utils.function_calling not available — no tools sent")
        return []

    schemas: list[dict] = []
    for tool in lc_tools:
        try:
            openai_schema = convert_to_openai_tool(tool)
            # Flatten: lift function.{name,description,parameters} to top level
            func = openai_schema.get("function", {})
            flat = {
                "type": "function",
                "name": func.get("name", getattr(tool, "name", "")),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }
            schemas.append(flat)
        except Exception as exc:
            log.warning(f"agents_xai: skipping tool {getattr(tool, 'name', '?')}: {exc}")
    return schemas


def _extract_text(content) -> str:
    """Normalize an output content value (string or content-block list) to plain text.

    Both xAI and OpenAI Responses APIs return content as a list of content blocks:
        [{"type": "output_text", "text": "Hello!", ...}]
    We also accept "text" type for forward compatibility.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(block.get("text", ""))
        return " ".join(parts).strip()
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

async def agentic_responses_api(model_key: str, messages: list[dict], client_id: str) -> str:
    """
    Responses API agentic loop (xAI + OpenAI).

    Turn 1 (no previous_response_id):
        Sends system prompt + full message history in `input`.
        Stores returned `id` as session["responses_api_id"].

    Turn N (previous_response_id set):
        Sends only the new user message in `input` plus previous_response_id.
        Provider reconstructs context server-side — payload is tiny.

    Tool execution:
        Tool calls are returned as output items with type="function_call".
        Results are submitted as input items with type="function_call_output".
        The loop continues until a final text response with no tool calls.

    On !reset or !model switch:
        routes.py clears session["responses_api_id"] = None, breaking the chain.
        The next turn starts fresh (full context sent again).
    """
    # Local imports to avoid circular dependency (agents.py imports this module).
    from agents import (
        execute_tool,
        _compute_active_tools,
        _resolve_model_tools,
        _get_cold_tool_names,
        _scan_and_save_memories,
        _memory_feature,
    )
    from prompt import load_prompt_for_folder
    from state import sessions, push_tok, push_done

    cfg = LLM_REGISTRY.get(model_key, {})
    session = sessions.get(client_id, {})
    api_key = cfg.get("key") or ""
    invoke_timeout = cfg.get("llm_call_timeout", 120)
    _suppress = session.get("tool_suppress", False)
    _provider = _provider_label(cfg)
    _url = _responses_api_url(cfg)

    # ---- Active tools (mirrors agentic_lc behaviour) ----
    _active_tools = _compute_active_tools(model_key, client_id)
    _cold_tools = _get_cold_tool_names(model_key, client_id)
    _model_tools_lc = _resolve_model_tools(model_key, active_tools=_active_tools)
    tools_schema = _lc_tools_to_openai_schemas(_model_tools_lc) if _model_tools_lc else []

    log.info(
        f"responses_api: model={model_key} provider={_provider} client={client_id} "
        f"tools={len(tools_schema)}"
    )

    # ---- System prompt ----
    system_prompt = ""
    sp_folder_rel = cfg.get("system_prompt_folder", "")
    if sp_folder_rel and sp_folder_rel.lower() != "none":
        sp_folder_abs = os.path.join(BASE_DIR, sp_folder_rel)
        system_prompt = load_prompt_for_folder(
            sp_folder_abs,
            active_tools=_active_tools,
            cold_tools=_cold_tools or None,
        )

    # ---- Memory scan flag (mirrors agentic_lc) ----
    _memory_scan = (
        bool(cfg.get("memory_scan", False))
        and _memory_feature("post_response_scan")
    )

    # ---- Build first `input` payload ----
    previous_response_id: str | None = session.get("responses_api_id")

    if previous_response_id:
        # Stateful: server holds prior history — send enrichment + new user message only.
        # The Responses API only accepts a system message as the very first message
        # of a chain, so we fold auto_enrich_context's injection into the user message.
        new_user = [m for m in messages if m.get("role") == "user"]
        enrich_msg = next(
            (m for m in messages
             if m.get("role") == "system"
             and "Auto-retrieved context" in m.get("content", "")),
            None,
        )
        user_msg = new_user[-1] if new_user else {}
        if enrich_msg and user_msg:
            combined = f"{enrich_msg['content']}\n\n---\n\n{user_msg.get('content', '')}"
            input_msgs: list[dict] = [{"role": "user", "content": combined}]
        else:
            input_msgs = [user_msg] if user_msg else []
        log.info(
            f"responses_api: stateful turn — prev_id={previous_response_id} "
            f"sending {len(input_msgs)} message(s) "
            f"(enrichment={'folded' if enrich_msg else 'none'})"
        )
    else:
        # First turn: send full context including system prompt.
        input_msgs = []
        if system_prompt:
            input_msgs.append({"role": "system", "content": system_prompt})
        input_msgs.extend(messages)
        log.info(
            f"responses_api: first turn — sending {len(input_msgs)} message(s) "
            f"(full context)"
        )

    # ---- Tool loop ----
    max_iters = LIVE_LIMITS.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
    iter_count = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(invoke_timeout)) as http:
        while max_iters == -1 or iter_count < max_iters:
            iter_count += 1

            if not _suppress:
                await push_tok(client_id, "\n[thinking…]\n")

            # Build request payload
            payload: dict = {
                "model": cfg["model_id"],
                "input": input_msgs,
                "store": True,
            }
            if previous_response_id:
                payload["previous_response_id"] = previous_response_id
            if tools_schema:
                payload["tools"] = tools_schema

            # Sampling parameters
            if cfg.get("token_selection_setting") == "custom":
                payload["temperature"] = cfg.get("temperature", 1.0)
                # OpenAI accepts top_p; xAI may reject with reasoning models
                _top_p = cfg.get("top_p")
                if _top_p is not None and _provider == "OpenAI":
                    payload["top_p"] = _top_p

            max_tok = cfg.get("max_tokens")
            if max_tok:
                payload["max_output_tokens"] = int(max_tok)

            log.info(
                f"responses_api: iter={iter_count} model={model_key} "
                f"prev_id={previous_response_id} input_items={len(input_msgs)}"
            )

            # ---- HTTP call with timeout ----
            try:
                async with asyncio.timeout(invoke_timeout + 5):
                    resp = await http.post(
                        _url,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            except asyncio.TimeoutError:
                await push_tok(
                    client_id,
                    f"\n[{_provider} Responses API timeout after {invoke_timeout}s]\n",
                )
                await push_done(client_id)
                return ""

            if resp.status_code == 400 and previous_response_id and iter_count == 1:
                # Chain state corrupted — clear and retry as first turn (full context)
                snippet = resp.text[:400]
                log.warning(
                    f"responses_api: 400 with prev_id={previous_response_id}, "
                    f"resetting chain and retrying as first turn: {snippet}"
                )
                previous_response_id = None
                session["responses_api_id"] = None
                input_msgs = []
                if system_prompt:
                    input_msgs.append({"role": "system", "content": system_prompt})
                input_msgs.extend(messages)
                iter_count = 0  # reset so the retry counts as iter 1
                continue

            if resp.status_code != 200:
                snippet = resp.text[:400]
                log.error(
                    f"responses_api: HTTP {resp.status_code} from {_provider}: {snippet}"
                )
                await push_tok(
                    client_id,
                    f"\n[{_provider} API error {resp.status_code}: {snippet}]\n",
                )
                await push_done(client_id)
                return ""

            data = resp.json()

            # ---- Store chain tip ----
            response_id = data.get("id")
            if response_id:
                session["responses_api_id"] = response_id
                previous_response_id = response_id

            # ---- Parse output ----
            output = data.get("output", [])
            text_response = ""
            tool_calls: list[dict] = []

            for item in output:
                item_type = item.get("type")
                if item_type == "message":
                    text_response = _extract_text(item.get("content", ""))
                elif item_type == "function_call":
                    tool_calls.append(item)

            # ---- Final answer ----
            if not tool_calls:
                if _memory_scan and text_response:
                    await _scan_and_save_memories(text_response, client_id, model_key)
                if text_response:
                    await push_tok(client_id, text_response)
                await push_done(client_id)
                return text_response

            # ---- Execute tools and build next input ----
            tool_result_items: list[dict] = []
            for tc in tool_calls:
                t_name = tc.get("name", "")
                t_args_raw = tc.get("arguments", "{}")
                t_call_id = tc.get("call_id", "")

                try:
                    t_args = (
                        json.loads(t_args_raw)
                        if isinstance(t_args_raw, str)
                        else t_args_raw
                    )
                except json.JSONDecodeError:
                    t_args = {}

                if not _suppress:
                    await push_tok(client_id, f"\n[tool: {t_name}]\n")

                result = await execute_tool(client_id, t_name, t_args)
                tool_result_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": t_call_id,
                        "output": result,
                    }
                )

            # Next iteration: only tool results needed; context lives server-side.
            input_msgs = tool_result_items

    # Loop exhausted
    log.warning(
        f"responses_api: max iterations ({max_iters}) reached, model={model_key}"
    )
    await push_tok(client_id, "\n[max tool iterations reached]\n")
    await push_done(client_id)
    return ""


# Backward-compatible alias
agentic_xai = agentic_responses_api
