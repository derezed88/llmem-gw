"""
plugin_history_default.py — Default history management plugin.

Implements a simple sliding-window strategy: keeps the last N messages,
where N = min(agent_max_ctx, current_model.max_context).

This is the first plugin in the history chain and must always be present.
Additional plugin_history_*.py plugins may be appended to the chain via
llmemctl.py to further transform history before it is sent to the LLM.

CONTRACT:
---------
Every history plugin must expose:

    NAME: str
        Unique identifier, e.g. "sliding_window".

    def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
        Called once per request cycle, after the user message has been
        appended to history, before dispatch_llm() is called.

        Args:
            history   -- current session["history"] (do not mutate; return a new list)
            session   -- the full session dict (read-only access to model, flags, etc.)
            model_cfg -- LLM_REGISTRY entry for the current model (has "max_context")

        Returns:
            The history list to store in session["history"] AND send to the LLM.
            Return a plain list[dict] of {"role": ..., "content": ...} messages.
            LLM-framework conversion (LangChain, etc.) is done by core — not here.

    def on_model_switch(session: dict, old_model: str, new_model: str,
                        old_cfg: dict, new_cfg: dict) -> list[dict]:
        Called immediately when !model switches the active model.
        Allows the plugin to trim or adjust history proactively on switch,
        rather than waiting for the next user message.

        Args:
            session   -- the full session dict
            old_model -- model key before switch
            new_model -- model key after switch
            old_cfg   -- LLM_REGISTRY entry for old model
            new_cfg   -- LLM_REGISTRY entry for new model

        Returns:
            The history list to store in session["history"] after the switch.

TWO-VARIABLE WINDOW SYSTEM:
----------------------------
    agent_max_ctx   (int) — system-wide ceiling, never exceeded regardless of model.
                            Configured in plugins-enabled.json → plugin_config →
                            plugin_history_default → agent_max_ctx.
                            Manageable via: llmemctl.py history-maxctx <n>
                            Runtime command: !maxctx <n>

    model max_context (int) — per-model preferred window from llm-models.json.
                            Manageable via: llmemctl.py model-context <model> <n>

    effective window  = min(agent_max_ctx, model["max_context"])

    session["history_max_ctx"] is set to effective window:
      - at session creation
      - on every !model switch (immediately, not lazily on next message)

CHAIN PROTOCOL:
---------------
    Core calls each plugin in chain order:
        for plugin in history_chain:
            history = plugin.process(history, session, model_cfg)
        session["history"] = history

    Each plugin receives the output of the previous plugin.
    The chain order is configured in plugins-enabled.json → plugin_config →
    plugin_history_default → chain.

    plugin_history_default is always first in the chain.

See: docs/PLUGIN_HISTORY_DEVELOPMENT.md
"""

import json
import os

NAME = "sliding_window"

# Path to system config — same directory as this file
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
_DEFAULT_AGENT_MAX_CTX = 200

# Runtime override (set via !maxctx command without restart)
_runtime_agent_max_ctx: int | None = None


def _load_agent_max_ctx() -> int:
    """Read agent_max_ctx from plugins-enabled.json at call time."""
    try:
        with open(_CONFIG_PATH) as f:
            data = json.load(f)
        return int(
            data.get("plugin_config", {})
                .get("plugin_history_default", {})
                .get("agent_max_ctx", _DEFAULT_AGENT_MAX_CTX)
        )
    except Exception:
        return _DEFAULT_AGENT_MAX_CTX


def get_agent_max_ctx() -> int:
    """Return effective agent_max_ctx: runtime override takes priority over config."""
    if _runtime_agent_max_ctx is not None:
        return _runtime_agent_max_ctx
    return _load_agent_max_ctx()


def set_runtime_agent_max_ctx(value: int) -> None:
    """Set in-memory override for agent_max_ctx (survives until restart)."""
    global _runtime_agent_max_ctx
    _runtime_agent_max_ctx = value


def compute_effective_max_ctx(model_cfg: dict) -> int:
    """Return min(agent_max_ctx, model.max_context)."""
    model_max = model_cfg.get("max_context", _DEFAULT_AGENT_MAX_CTX)
    return min(get_agent_max_ctx(), model_max)


def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    """
    Sliding-window trim: keep the last N messages where N = session["history_max_ctx"].
    Returns a new list (does not mutate the input).
    """
    max_ctx = session.get("history_max_ctx", compute_effective_max_ctx(model_cfg))
    if max_ctx <= 0:
        return list(history)
    return list(history[-max_ctx:])


def on_model_switch(session: dict, old_model: str, new_model: str,
                    old_cfg: dict, new_cfg: dict) -> list[dict]:
    """
    Recompute effective window and trim history immediately on !model switch.
    Returns the trimmed history to store in session["history"].
    """
    new_effective = compute_effective_max_ctx(new_cfg)
    session["history_max_ctx"] = new_effective
    history = session.get("history", [])
    if new_effective <= 0:
        return list(history)
    return list(history[-new_effective:])
