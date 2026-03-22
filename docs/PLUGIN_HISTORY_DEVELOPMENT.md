# History Plugin Development Guide

This guide is for developers and architects who want to experiment with or replace the conversation history management strategy in llmem-gw.

---

## Terminology

**History** is the correct term for what this system manages. It is a rolling log of `{"role": "user"|"assistant", "content": "..."}` message dicts — the OpenAI Chat Completions format, the de facto industry standard accepted by all major LLM APIs.

**Memory** refers to persistent, structured knowledge (RAG, vector stores, semantic summaries). The memory system is documented separately in [MEMORY_PROJECT1.md](MEMORY_PROJECT1.md) and [COGNITION.md](COGNITION.md). History plugins do not interact with memory — they operate only on the conversation history list.

---

## Architecture Overview

History management is handled by a **plugin chain** — an ordered list of `plugin_history_*.py` modules. Each plugin in the chain receives the history list, may transform it, and passes the result to the next plugin. The final output is both stored in the session and sent to the LLM.

```
User message appended to session["history"]
        ↓
plugin_history_default.process()   ← always first
        ↓
plugin_history_foo.process()       ← optional, configured by operator
        ↓
plugin_history_bar.process()       ← optional
        ↓
session["history"] = result        ← stored AND sent to LLM
```

The chain is **LLM-agnostic**. Plugins work only in the neutral `list[dict]` format. LangChain / LLM-framework conversion is done by core code (`agents.py`) after the chain runs — never inside a history plugin.

---

## Plugin Contract

Every `plugin_history_*.py` file must expose:

### `NAME: str`
A unique short identifier for this plugin's strategy. Used in logs and display.

```python
NAME = "sliding_window"
```

---

### `process(history, session, model_cfg) -> list[dict]`

Called once per request cycle:
- **After** the user message has been appended to `session["history"]`
- **Before** `dispatch_llm()` is called

```python
def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    """
    Args:
        history   -- current history list (do NOT mutate; return a new list)
        session   -- full session dict (read-only; has "model", "history_max_ctx", etc.)
        model_cfg -- LLM_REGISTRY entry for current model (has "max_context", etc.)

    Returns:
        list[dict] — the history to store in session["history"] AND send to the LLM.
        Each item: {"role": "user"|"assistant", "content": "..."}
    """
    ...
```

**Important rules:**
- Do not mutate the `history` argument. Return a new list.
- Do not import or reference LangChain, OpenAI, or any LLM library.
- Do not modify `session["history"]` directly — core does that after the chain runs.
- Keep it fast — this runs on every user message.

---

### `on_model_switch(session, old_model, new_model, old_cfg, new_cfg) -> list[dict]`

Called immediately when the user runs `!model <name>` — **before** the next user message. This allows the plugin to trim or adjust history proactively rather than waiting.

```python
def on_model_switch(session: dict, old_model: str, new_model: str,
                    old_cfg: dict, new_cfg: dict) -> list[dict]:
    """
    Args:
        session   -- full session dict (may read and update session["history_max_ctx"])
        old_model -- model key before switch
        new_model -- model key after switch
        old_cfg   -- LLM_REGISTRY entry for old model
        new_cfg   -- LLM_REGISTRY entry for new model

    Returns:
        list[dict] — the history to store in session["history"] after the switch.
    """
    ...
```

If your plugin does not need to act on model switches, you can omit this function — core checks `hasattr(plugin, "on_model_switch")` before calling it.

---

## The Two-Variable Window System

History window size is controlled by **two variables** that combine to produce an effective limit:

| Variable | Meaning | Where configured |
|---|---|---|
| `agent_max_ctx` | System-wide ceiling — never exceeded regardless of model | `plugins-enabled.json` → `plugin_config.plugin_history_default.agent_max_ctx` |
| `model.max_context` | Per-model preferred window | `llm-models.json` per model |
| **`effective_ctx`** | `min(agent_max_ctx, model.max_context)` | Computed at session create and on every `!model` switch |

`session["history_max_ctx"]` holds the current effective value for the session. The default plugin reads this:

```python
max_ctx = session.get("history_max_ctx", compute_effective_max_ctx(model_cfg))
return list(history[-max_ctx:])
```

---

## Session Variables Used by History Plugins

| Key | Type | Set by | Meaning |
|---|---|---|---|
| `session["history"]` | `list[dict]` | core (routes.py) | The stored conversation |
| `session["history_max_ctx"]` | `int` | core on create + model switch | Effective window for this session |
| `session["model"]` | `str` | core | Current model key |
| `session["last_active"]` | `float` | core (time.time()) | Unix timestamp of last message |

---

## System-Wide Resource Configuration

These values are stored in `plugins-enabled.json` at the top level and managed via `llmemctl.py`:

| Setting | Default | Meaning |
|---|---|---|
| `max_users` | 50 | Max simultaneous sessions. New sessions rejected when limit reached. |
| `session_idle_timeout_minutes` | 60 | Sessions idle longer than this are reaped. 0 = disabled. |

And in `plugin_config.plugin_history_default`:

| Setting | Default | Meaning |
|---|---|---|
| `agent_max_ctx` | 200 | Agent-wide history ceiling |
| `chain` | `["plugin_history_default"]` | Ordered list of active history plugins |

---

## Chain Configuration

### Via llmemctl.py (persistent)

```bash
python llmemctl.py history-list                         # show chain + config
python llmemctl.py history-chain-add plugin_history_foo # append to chain
python llmemctl.py history-chain-remove plugin_history_foo
python llmemctl.py history-chain-move plugin_history_foo 1  # move to position 1
python llmemctl.py history-maxctx 500                   # set agent_max_ctx
python llmemctl.py max-users 100                        # set max sessions
python llmemctl.py session-timeout 120                  # set idle timeout (minutes)
```

### Via runtime commands (in-memory, lost on restart unless persisted)

```
!maxctx 500           — set agent_max_ctx (persisted by default)
!maxctx 500 temp      — set without persisting
!maxusers 100         — set max simultaneous sessions
!sessiontimeout 120   — set idle timeout in minutes (0 = disabled)
```

---

## Writing a New History Plugin

1. Create `plugin_history_yourname.py` in the `llmem-gw/` directory.

2. Implement the contract:

```python
NAME = "your_strategy"

def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    # Your logic here
    return list(history)  # return new list, don't mutate

def on_model_switch(session, old_model, new_model, old_cfg, new_cfg):
    # Optional: react to model changes
    return list(session.get("history", []))
```

3. Verify it's discovered:
```bash
python llmemctl.py history-list
```

4. Add it to the chain:
```bash
python llmemctl.py history-chain-add plugin_history_yourname
```

5. Restart the agent for the chain to reload.

---

## Plugin Discovery

`llmemctl.py` discovers history plugins by globbing `plugin_history_*.py` in the project directory. Any file matching that pattern is listed as "available". It is **not** in the chain unless explicitly added via `llmemctl.py history-chain-add` or by editing `plugins-enabled.json` directly.

`plugin_history_default` is always first in the chain and cannot be removed or moved from position 0.

---

## Example: Token-Budget Plugin

A plugin that estimates tokens and trims the oldest messages when over budget:

```python
NAME = "token_budget"

_CHARS_PER_TOKEN = 4  # rough estimate
_DEFAULT_BUDGET  = 6000  # tokens

def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    budget = model_cfg.get("token_budget", _DEFAULT_BUDGET)
    result = list(history)
    while result:
        total_chars = sum(len(m.get("content", "")) for m in result)
        if total_chars // _CHARS_PER_TOKEN <= budget:
            break
        result.pop(0)  # drop oldest
    return result

def on_model_switch(session, old_model, new_model, old_cfg, new_cfg):
    return list(session.get("history", []))
```

---

## Example: Role-Pair Enforcer

Some LLMs (e.g., Anthropic Claude) reject histories that don't strictly alternate user/assistant. This plugin ensures the history starts with a user turn and alternates correctly:

```python
NAME = "role_pair_enforcer"

def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    result = []
    expected = "user"
    for msg in history:
        if msg["role"] == expected:
            result.append(msg)
            expected = "assistant" if expected == "user" else "user"
        # mismatched role: skip silently
    # If last message is user (incomplete pair), include it
    return result

def on_model_switch(session, old_model, new_model, old_cfg, new_cfg):
    return list(session.get("history", []))
```

---

## Important Notes

- **History plugins are LLM-agnostic.** Any per-LLM formatting (role names, message wrapping, tool call formatting) stays in `agents.py:_to_lc_messages()`.
- **Tool turn messages** (role=`tool`) live only inside the LangChain `ctx` list during the agentic loop. They are not stored in `session["history"]` and will not appear in the list your plugin receives.
- **The chain reloads on agent restart.** If you add a plugin file, restart the agent. Runtime `!maxctx` / `!maxusers` changes do not require a restart.
- **No circular imports.** History plugins must not import from `routes.py`, `agents.py`, or any module that imports from them.
