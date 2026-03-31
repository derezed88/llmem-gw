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

Retry / fallback (503 at-capacity):
  - retry_on_503: {"max_retries": N, "backoff": [s1, s2, ...]} — retry with countdown
  - backup_models: ["model-key", ...] — ordered fallback list on retry exhaustion
  - Background probe re-checks primary every 60s; auto-restores when healthy
  - While probe is running, dispatch skips primary and routes to backup directly

Integration:
  - dispatch_llm() in agents.py calls agentic_responses_api() when is_responses_api() is True
  - config.py whitelist includes xai_responses_api and openai_responses_api
  - cmd_reset() and cmd_set_model() in routes.py clear session["responses_api_id"]
"""

import asyncio
import json
import os
import re
import time

import httpx

from config import LLM_REGISTRY, LIVE_LIMITS, MAX_TOOL_ITERATIONS, BASE_DIR, log


# ---------------------------------------------------------------------------
# GED post-turn hook: detect quiz scores without a corresponding DB write
# ---------------------------------------------------------------------------

# Matches quiz/assessment result patterns:
#   "You got 5 out of 5", "got 7 out of 10", "scored 8 out of 10",
#   "score: 8/10", "perfect score", "5 out of 5!"
# Avoids false positives on math fractions by requiring "got/scored/score" context
# or the "N out of N" phrasing (fractions use "N/N" or LaTeX).
_GED_SCORE_RE = re.compile(
    r"(?:got|scored?)\s+\d+\s+out\s+of\s+\d+|perfect\s+score",
    re.IGNORECASE,
)

_GED_SCORE_NUDGE = (
    "[SYSTEM] Your response above reports quiz/assessment scores but you did NOT "
    "call llm_call to write them to the database. Lee's #ged dashboard will show "
    "\"No progress\" unless you persist the scores NOW.\n\n"
    "You MUST call llm_call(model=\"samaritan-execution\", ...) with the appropriate "
    "INSERT/UPDATE SQL for the topic scores table and quiz results table BEFORE "
    "continuing. Do this now — do not respond to Lee until the writes are done."
)


# ---------------------------------------------------------------------------
# 503 fallback state — module-level, shared across all sessions
# ---------------------------------------------------------------------------

# model_key → {"backup": "backup-model-key", "since": epoch, "probe_task": Task|None}
_fallback_state: dict[str, dict] = {}

# Probe interval: how often (seconds) the background task pings the primary
_PROBE_INTERVAL = 60


def is_model_in_fallback(model_key: str) -> bool:
    """Return True if this model is currently in fallback due to 503."""
    return model_key in _fallback_state


def get_fallback_model(model_key: str) -> str | None:
    """Return the active backup model key, or None if not in fallback."""
    state = _fallback_state.get(model_key)
    return state["backup"] if state else None


def clear_fallback(model_key: str):
    """Remove fallback state and cancel probe for a model."""
    state = _fallback_state.pop(model_key, None)
    if state and state.get("probe_task"):
        state["probe_task"].cancel()
    if state:
        log.info(f"fallback: cleared fallback state for {model_key}")


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
# Background probe — restores primary model when it comes back
# ---------------------------------------------------------------------------

async def _probe_primary(model_key: str):
    """Periodically ping the primary model's API until it responds 200.

    On success: clear fallback state and notify all sessions using this model.
    """
    cfg = LLM_REGISTRY.get(model_key, {})
    api_key = cfg.get("key") or ""
    url = _responses_api_url(cfg)
    provider = _provider_label(cfg)

    log.info(f"probe: started for {model_key} ({provider}), interval={_PROBE_INTERVAL}s")

    while model_key in _fallback_state:
        await asyncio.sleep(_PROBE_INTERVAL)

        if model_key not in _fallback_state:
            break  # cleared externally

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15)) as http:
                resp = await http.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": cfg.get("model_id", ""),
                        "input": [{"role": "user", "content": "ping"}],
                        "max_output_tokens": 1,
                        "store": False,
                    },
                )

            if resp.status_code == 200:
                log.info(f"probe: {model_key} ({provider}) is back — restoring as primary")
                _restore_primary(model_key)
                return
            else:
                log.info(
                    f"probe: {model_key} still unavailable "
                    f"(HTTP {resp.status_code}), next check in {_PROBE_INTERVAL}s"
                )
        except Exception as exc:
            log.info(f"probe: {model_key} ping failed ({exc}), next check in {_PROBE_INTERVAL}s")

    log.info(f"probe: stopped for {model_key} (fallback cleared externally)")


def _restore_primary(model_key: str):
    """Restore all sessions that were on fallback back to the primary model."""
    from state import sessions, push_tok

    state = _fallback_state.pop(model_key, None)
    if not state:
        return

    backup_key = state.get("backup", "?")
    elapsed = int(time.time() - state.get("since", 0))

    # Find sessions currently on the backup model and restore them
    restored = 0
    for cid, sess in sessions.items():
        if sess.get("_fallback_primary") == model_key:
            sess["model"] = model_key
            sess["responses_api_id"] = None  # fresh chain on restored primary
            del sess["_fallback_primary"]
            restored += 1
            # Notify async — fire and forget
            asyncio.ensure_future(push_tok(
                cid,
                f"\n[{_provider_label(LLM_REGISTRY.get(model_key, {}))} recovered — "
                f"restored to {model_key} (was on {backup_key} for {elapsed}s)]\n",
            ))

    log.info(
        f"fallback: restored {restored} session(s) from {backup_key} → {model_key} "
        f"(down for {elapsed}s)"
    )


def _activate_fallback(model_key: str, backup_key: str, client_id: str):
    """Switch a session to backup and start the background probe if not already running."""
    from state import sessions

    session = sessions.get(client_id, {})
    session["_fallback_primary"] = model_key
    session["model"] = backup_key
    session["responses_api_id"] = None  # fresh chain on backup

    if model_key not in _fallback_state:
        probe_task = asyncio.ensure_future(_probe_primary(model_key))
        _fallback_state[model_key] = {
            "backup": backup_key,
            "since": time.time(),
            "probe_task": probe_task,
        }
        log.info(f"fallback: {model_key} → {backup_key}, probe started")
    else:
        log.info(f"fallback: {model_key} → {backup_key} (probe already running)")


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
    from state import sessions, push_tok, push_done, set_progress_stage, start_progress_ticker, stop_progress_ticker

    cfg = LLM_REGISTRY.get(model_key, {})
    session = sessions.get(client_id, {})
    api_key = cfg.get("key") or ""
    invoke_timeout = cfg.get("llm_call_timeout", 120)
    _suppress = session.get("tool_suppress", False)
    _provider = _provider_label(cfg)
    _url = _responses_api_url(cfg)

    # Start progress ticker if configured
    _progress_freq = cfg.get("progress_response_freq")
    _progress_stages = cfg.get("progress_response_stages", False)
    if _progress_freq or _progress_stages:
        start_progress_ticker(client_id, _progress_freq, stages=_progress_stages)
        await set_progress_stage(client_id, "preparing")

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

    # ---- Guard: drop empty-content messages (e.g. silent voice turn) ----
    input_msgs = [
        m for m in input_msgs
        if m.get("type") == "function_call_output"  # tool results — always keep
        or (m.get("content") or "").strip()
    ]
    if not input_msgs:
        log.info("responses_api: empty input after filtering — silent turn, skipping API call")
        stop_progress_ticker(client_id)
        await push_done(client_id)
        return ""

    # ---- GED post-turn hook state ----
    _is_ged_model = model_key.startswith("ged-") and model_key not in ("ged-planner", "ged-quiz-gen")
    _saw_llm_call = False
    _ged_nudge_sent = False

    # ---- Tool loop ----
    max_iters = LIVE_LIMITS.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
    iter_count = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(invoke_timeout)) as http:
        while max_iters == -1 or iter_count < max_iters:
            iter_count += 1

            if _progress_freq or _progress_stages:
                await set_progress_stage(client_id, "waiting for LLM")
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

            # ---- HTTP call with timeout + 503 retry ----
            _retry_cfg = cfg.get("retry_on_503") or {}
            _max_retries = _retry_cfg.get("max_retries", 0) if iter_count == 1 else 0
            _backoff = _retry_cfg.get("backoff", [])
            _backup_models = cfg.get("backup_models") or []
            resp = None

            for _attempt in range(_max_retries + 1):
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
                    stop_progress_ticker(client_id)
                    await push_done(client_id)
                    return ""
                except httpx.TransportError as exc:
                    log.warning(
                        f"responses_api: transport error from {_provider} "
                        f"(attempt {_attempt + 1}/{_max_retries + 1}): {exc!r}"
                    )
                    if _attempt >= _max_retries:
                        await push_tok(
                            client_id,
                            f"\n[{_provider} connection error: {type(exc).__name__} — retrying may help]\n",
                        )
                        stop_progress_ticker(client_id)
                        await push_done(client_id)
                        return ""
                    # fall through to 503 retry logic below
                    continue

                if resp.status_code != 503 or _attempt >= _max_retries:
                    break  # not a 503, or retries exhausted

                # 503 — retry with countdown notification
                wait_secs = _backoff[_attempt] if _attempt < len(_backoff) else _backoff[-1] if _backoff else 10
                log.warning(
                    f"responses_api: 503 from {_provider} (attempt {_attempt + 1}/{_max_retries}), "
                    f"retrying in {wait_secs}s"
                )
                await push_tok(
                    client_id,
                    f"\n[{_provider} at capacity — retrying in {wait_secs}s...]\n",
                )
                await asyncio.sleep(wait_secs)

            if resp.status_code == 503 and _backup_models:
                # All retries exhausted — activate fallback
                backup_key = _backup_models[0]
                if backup_key in LLM_REGISTRY:
                    log.warning(
                        f"responses_api: 503 retries exhausted for {model_key}, "
                        f"falling back to {backup_key}"
                    )
                    await push_tok(
                        client_id,
                        f"\n[{_provider} unavailable — switching to {backup_key}]\n",
                    )
                    _activate_fallback(model_key, backup_key, client_id)
                    stop_progress_ticker(client_id)
                    # Re-dispatch through the backup model
                    from agents import dispatch_llm
                    result = await dispatch_llm(backup_key, messages, client_id)
                    return result

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
                # Re-apply empty content guard
                input_msgs = [
                    m for m in input_msgs
                    if m.get("type") == "function_call_output"
                    or (m.get("content") or "").strip()
                ]
                if not input_msgs:
                    log.info("responses_api: empty input after chain-reset filter — skipping")
                    stop_progress_ticker(client_id)
                    await push_done(client_id)
                    return ""
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
                stop_progress_ticker(client_id)
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
                # GED post-turn hook: if response has quiz scores but no llm_call
                # was used to persist them, nudge the model to write before finishing.
                if (
                    _is_ged_model
                    and not _ged_nudge_sent
                    and not _saw_llm_call
                    and text_response
                    and _GED_SCORE_RE.search(text_response)
                ):
                    _ged_nudge_sent = True
                    log.info(
                        f"responses_api: GED score-write nudge for model={model_key} "
                        f"client={client_id} — scores detected without llm_call"
                    )
                    # Stream the tutor's text to Lee, then inject the nudge
                    await push_tok(client_id, text_response)
                    input_msgs = [
                        {"role": "user", "content": _GED_SCORE_NUDGE},
                    ]
                    continue  # re-enter the loop so the model can make the write call

                if _memory_scan and text_response:
                    await _scan_and_save_memories(text_response, client_id, model_key)
                if text_response:
                    await push_tok(client_id, text_response)
                stop_progress_ticker(client_id)
                await push_done(client_id)
                return text_response

            # ---- Execute tools and build next input ----
            tool_result_items: list[dict] = []
            for tc in tool_calls:
                t_name = tc.get("name", "")
                if t_name == "llm_call":
                    _saw_llm_call = True
                if _progress_freq or _progress_stages:
                    await set_progress_stage(client_id, f"running {t_name}")
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
    stop_progress_ticker(client_id)
    await push_done(client_id)
    return ""


# Backward-compatible alias
agentic_xai = agentic_responses_api
