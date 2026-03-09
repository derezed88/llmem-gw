"""
plugin_history_judge.py — LLM-as-judge enforcement history chain plugin.

Sits in the history plugin chain. Provides four enforcement gate points:

  prompt   — pre-LLM pass: inspects user message before it reaches the LLM
  response — post-LLM pass: inspects assistant response before it is stored
  tool     — execute_tool() hook: inspects tool call before executor runs
  memory   — _scan_and_save_memories() hook: inspects memory before persisting

The prompt and response gates are handled by process() via the standard
history chain contract. The tool and memory gates are handled by async
hooks registered into judge.py at module load time; agents.py calls
judge.check_tool_gate() and judge.check_memory_gate() which are no-ops
when this plugin is not loaded.

Gate behavior is controlled entirely by judge_config in llm-models.json
(per-model default) and session["judge_override"] (per-session override).
See judge.py for the full config schema.

Configuration
-------------
No plugin-level config is required in plugins-enabled.json. The judge model
and gate settings are per-model in llm-models.json:

    "my-model": {
        ...
        "judge_config": {
            "model":     "judge-qwen35",
            "gates":     ["prompt", "response", "tool", "memory"],
            "mode":      "block",
            "threshold": 0.7
        }
    }

Adding to chain
---------------
    python llmemctl.py history-chain-add plugin_history_judge

The plugin should be added AFTER plugin_history_default (sliding window)
and BEFORE any security scan plugins (sec_sync, sec_async).

Enabling / disabling specific gates
------------------------------------
Via chat commands (see !judge help) or directly in llm-models.json.
The plugin itself is enabled/disabled by its presence in the chain.

CONTRACT
--------
This module follows the standard history plugin contract:

    NAME: str
    def process(history, session, model_cfg) -> list[dict]
    def on_model_switch(session, old_model, new_model, old_cfg, new_cfg) -> list[dict]

See: docs/PLUGIN_HISTORY_DEVELOPMENT.md
"""

import asyncio
import json
import logging

log = logging.getLogger(__name__)

NAME = "judge"


# ---------------------------------------------------------------------------
# Register tool and memory gate hooks into judge.py at import time.
# This is what enables agents.py to call judge.check_tool_gate() and
# judge.check_memory_gate() without hardcoding this plugin.
# ---------------------------------------------------------------------------

async def _tool_gate_hook(
    client_id: str, model_key: str, session: dict,
    tool_name: str, tool_args: dict,
) -> tuple[bool, str]:
    from judge import judge_gate, is_gate_active
    if not is_gate_active("tool", model_key, session):
        return True, ""
    content = f"tool: {tool_name}\nargs: {json.dumps(tool_args, ensure_ascii=False)}"
    return await judge_gate(
        gate="tool",
        content=content,
        model_key=model_key,
        session=session,
        client_id=client_id,
    )


async def _memory_gate_hook(
    client_id: str, model_key: str, session: dict,
    topic: str, content: str,
) -> tuple[bool, str]:
    from judge import judge_gate, is_gate_active
    if not is_gate_active("memory", model_key, session):
        return True, ""
    return await judge_gate(
        gate="memory",
        content=content,
        model_key=model_key,
        session=session,
        client_id=client_id,
        topic=topic,
    )


def _register_hooks() -> None:
    try:
        import judge as _judge
        _judge._tool_gate_hook = _tool_gate_hook
        _judge._memory_gate_hook = _memory_gate_hook
        log.info("plugin_history_judge: tool + memory gate hooks registered")
    except Exception as e:
        log.warning(f"plugin_history_judge: failed to register hooks: {e}")


_register_hooks()


# ---------------------------------------------------------------------------
# History chain contract
# ---------------------------------------------------------------------------

def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    """
    Pre-LLM pass (last role == "user"):   evaluate prompt gate.
    Post-LLM pass (last role == "assistant"): evaluate response gate.

    Blocking on prompt:  replace the user message with a rejection notice
                         so the LLM receives a no-op prompt and returns a
                         block message — history remains consistent.
    Blocking on response: replace the assistant message with a block notice.
    Warn mode: push a warning to the client stream but allow through.
    """
    if not history:
        return list(history)

    last = history[-1]
    model_key = session.get("model", "")
    client_id = session.get("_client_id", "")

    from judge import is_gate_active, judge_gate, _get_effective_judge_cfg

    # ----- prompt gate (pre-LLM pass) -----
    if last["role"] == "user":
        if not is_gate_active("prompt", model_key, session):
            return list(history)

        from agents import _content_to_str
        content = _content_to_str(last.get("content", ""))
        cfg = _get_effective_judge_cfg(model_key, session)
        if not cfg:
            return list(history)

        # process() is sync; run the async judge_gate in the event loop
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                judge_gate(
                    gate="prompt",
                    content=content,
                    model_key=model_key,
                    session=session,
                    client_id=client_id,
                ),
                loop,
            )
            allowed, denial_reason = future.result(timeout=cfg.get("llm_call_timeout", 30) + 5)
        except Exception as e:
            log.warning(f"plugin_history_judge: prompt gate error: {e} — allowing through")
            return list(history)

        if allowed:
            return list(history)

        # Block: replace user message so the LLM gets a benign no-op
        mode = cfg.get("mode", "block")
        log.info(f"plugin_history_judge: prompt BLOCKED client={client_id} mode={mode}")
        new_history = list(history[:-1])
        new_history.append({
            "role": "user",
            "content": (
                "[JUDGE BLOCKED] The user's message was blocked by the judge before "
                "reaching you. Reply: 'I'm unable to process that request.' "
                f"Reason: {denial_reason}"
            ),
        })
        return new_history

    # ----- response gate (post-LLM pass) -----
    if last["role"] == "assistant":
        if not is_gate_active("response", model_key, session):
            return list(history)

        from agents import _content_to_str
        content = _content_to_str(last.get("content", ""))
        if not content:
            return list(history)

        cfg = _get_effective_judge_cfg(model_key, session)
        if not cfg:
            return list(history)

        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                judge_gate(
                    gate="response",
                    content=content,
                    model_key=model_key,
                    session=session,
                    client_id=client_id,
                ),
                loop,
            )
            allowed, denial_reason = future.result(timeout=cfg.get("llm_call_timeout", 30) + 5)
        except Exception as e:
            log.warning(f"plugin_history_judge: response gate error: {e} — allowing through")
            return list(history)

        if allowed:
            return list(history)

        mode = cfg.get("mode", "block")
        log.info(f"plugin_history_judge: response BLOCKED client={client_id} mode={mode}")

        # Alert was already pushed to client by judge_gate (block mode)
        block_msg = (
            f"[JUDGE BLOCKED] My response was blocked by the content judge. "
            f"Reason: {denial_reason}"
        )
        new_history = list(history[:-1])
        new_history.append({"role": "assistant", "content": block_msg})
        return new_history

    # Any other role (system, tool, etc.) — pass through unchanged
    return list(history)


def on_model_switch(
    session: dict,
    old_model: str, new_model: str,
    old_cfg: dict, new_cfg: dict,
) -> list[dict]:
    """No history transformation needed on model switch."""
    return list(session.get("history", []))
