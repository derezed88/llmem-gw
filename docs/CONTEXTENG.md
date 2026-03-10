# Context Engineering — llmem-gw

Last updated: 2026-03-09

This document describes every layer that contributes to the final prompt sent to an LLM, in injection order. Each layer has toggles at three scopes: **Global** (plugins-enabled.json), **Model** (llm-models.json), and **Session** (runtime state, set via `!config`/`!memory`).

---

## Mental Model

Think of the prompt as built up in layers, from static to dynamic:

```
[System Prompt Tree]           ← static, defined by model config
[Tool List]                    ← one-time per session
[Auto-Enrich: DB Rules]        ← dynamic, pattern-matched SQL
[Auto-Enrich: Memory Context]  ← semantic retrieval from MySQL+Qdrant
[Auto-Enrich: Known Topics]    ← topic list for slug reuse guidance
[History (sliding window)]     ← trimmed by plugin_history_default
[User Message]                 ← current turn input
```

Post-response layers (not in prompt, but affect memory for future turns):
```
[Judge Response Gate]          ← may replace assistant response
[Memory Scan]                  ← extracts inline memory_save() calls
[Conv Log]                     ← saves verbatim turn to ST memory
```

---

## Layer 0 — System Prompt Tree

**What:** The base system prompt, assembled from a folder of `.system_prompt` files.

**When:** Every request, built once per `agentic_lc()` call (`agents.py:~1192`).

**How assembled:** `load_prompt_for_folder()` in `prompt.py`:
- Reads root `.system_prompt` file in the folder
- Appends optional section files (`.system_prompt_<name>`) listed in `[SECTIONS]`
- Sections filtered by which tools are active (tool-conditional sections)

**System prompt trees:**
| Folder | Used by |
|---|---|
| `000_default` | General-purpose models (gpt4om, grok41fr, gemini25f, etc.) |
| `001_blank` | Bare minimum / testing |
| `002_mysqlonly` | MySQL-only restricted |
| `003_claudeVSCode` | Claude summarizer arm |
| `004_voice` | samaritan-voice |
| `004_reasoning` | samaritan-reasoning |
| `004_execution` | samaritan-execution |
| `005_fileops` | extractor-gemini, summarizer-gemini |
| `006_local` | Local Qwen models |
| `007_judge` | judge-gemini, judge-qwen35 |

**Toggles:**
| Scope | Key | Effect |
|---|---|---|
| Model | `system_prompt_folder` | Which tree to load; `""` or `"none"` → no system prompt |

**Minimum context:** A model with `system_prompt_folder: ""` gets NO system message — just history + user message.

---

## Layer 1 — Tool List Injection

**What:** A synthetic assistant message listing all tools the model is authorized to use for this session. Injected once on the first turn.

**When:** First turn only; `dispatch_llm()` in `agents.py:~1952`. Sets `session["tool_list_injected"] = True` so it never repeats.

**Format:** `[Session start — authorized tools]\n<tool list text>`

**Toggles:**
| Scope | Key | Effect |
|---|---|---|
| Session | `tool_list_injected` | Auto-set; no user control |
| Model | `tool_suppress` | Suppresses display of tool-related output, but does NOT remove this injection |

**Minimum context:** Cannot be disabled. It fires on every new session for any model with tools.

---

## Layer 2a — Auto-Enrich: DB Rules

**What:** SQL query results injected when the user message matches a regex pattern. Purpose: give the model live data from the DB without requiring it to call a tool.

**When:** Every request, inside `auto_enrich_context()` in `agents.py:~1076`.

**Rules source:** `db-config.json` → `auto_enrich` array. Each rule:
```json
{ "pattern": "<regex>", "sql": "<query>", "label": "<name>" }
```

**Injection format:** Prepended as a system message:
```
## Auto-retrieved context
Base answer on this data:

[auto-retrieved via: <label>]
<sql result>
```

**Toggles:**
| Scope | Key | Effect |
|---|---|---|
| Session | `auto_enrich` | `false` → skip all DB rules; default `true` |
| Global | `db-config.json auto_enrich` | Array of rules; empty array → no enrichment |

**Set via:** `!config write auto_enrich false`

---

## Layer 2b — Auto-Enrich: Memory Context

**What:** Semantically relevant memory retrieved from short-term (ST) and long-term (LT) MySQL tiers, indexed via Qdrant. This is the most dynamic part of the prompt — it changes every turn based on what the conversation is about.

**When:** Every request, inside `auto_enrich_context()` in `agents.py:~1104`.

**Query construction:**
- If `session["current_topic"]` is set (from prior turn's `<<topic>>` tag) → use that slug as query
- Otherwise → concatenate last 3 genuine user/assistant turns (300 chars each), strip vocative address if `identity_name` is configured

**Retrieval:** Two-pass Qdrant + MySQL (see MEMORY_PROJECT1.md for full detail):
- Pass 1: topic slug → embed → Qdrant (both tiers, top_k=20, min_score=0.45)
- Pass 2: user message text → embed → Qdrant (only if pass 1 yields < `two_pass_threshold` quality hits)
- ST rows with importance ≥ `min_importance_always` (default 8) always injected

**Injection format:**
```
## Active Memory (short-term recall)
...rows...

## Long-term Memory (relevant recalled)
...rows...
```

**Three-level veto (ALL must allow):**
| Scope | Key | `null` means | `false` means |
|---|---|---|---|
| Global | `plugins-enabled.json → memory.context_injection` | — | Skip context injection |
| Global | `plugins-enabled.json → memory.enabled` | — | Skip ALL memory features |
| Model | `llm-models.json → memory_enabled` | Defer (allow) | Always off for this model |
| Session | `session["memory_enabled"]` | Defer (allow) | Off for this session |

**Set via:** `!memory false` (session), `plugins-enabled.json` (global), model config (permanent per model)

**Minimum context without memory:** No `## Active Memory` block. Model only sees history and system prompt.

---

## Layer 2c — Auto-Enrich: Known Topics

**What:** A comma-separated list of recent topic slugs, injected alongside memory context. Helps the model reuse existing topic slugs rather than inventing new near-duplicates.

**When:** Same call as Layer 2b; injected only when memory context injection fires.

**Toggles:** Coupled to Layer 2b — no independent toggle.

---

## Layer 3a — History: Sliding Window Trim

**What:** Trims conversation history to stay within token budget before sending to LLM.

**When:** `_run_history_chain()` called in `routes.py:2088` (pre-LLM).

**Plugin:** `plugin_history_default` — truncates oldest turns until token count ≤ `agent_max_ctx`.

**Toggles:**
| Scope | Key | Effect |
|---|---|---|
| Global | `plugins-enabled.json → plugin_history_default.agent_max_ctx` | Token window size |

---

## Layer 3b — History: Judge Prompt Gate

**What:** Pre-LLM pass — evaluates the user's message against the judge model. If blocked, the user message is replaced with a rejection notice before the LLM ever sees it.

**When:** `_run_history_chain()` pre-LLM pass; `plugin_history_judge.py:~146`.

**Toggles:**
| Scope | Key | Effect |
|---|---|---|
| Model | `llm-models.json → judge_config` | Judge model, gates list, mode, threshold |
| Model | `judge_config.gates` includes `"prompt"` | Enables prompt gate for this model |
| Session | `judge_override` | Per-session override of judge config |

**Mode options:** `"warn"` (log only), `"block"` (replace message).

---

## Layer 4 — LangChain Message Conversion

Pure format conversion — no new content added. Internal dict history → LangChain `BaseMessage` objects. System prompt (Layer 0) becomes `SystemMessage` at position 0.

---

## Layer 5 — Tool Binding

**What:** Tools dynamically bound to the LLM before inference. The model can only call tools it's bound with.

**Toggles:**
| Scope | Key | Effect |
|---|---|---|
| Model | `llm-models.json → llm_tools` | Authorized toolset group names |
| Session | Tool heat subscriptions | Dynamic hot/cold lifecycle; some tools activate on demand |
| Model | `llm-models.json → llm_tools_gates` | Tools requiring human approval before execution |

---

## Layer 6 — Agent Loop

**What:** LLM invocation, tool execution, and iteration until a final text response is produced.

**Toggles:**
| Scope | Key | Effect |
|---|---|---|
| Model | `limits.max_tool_iterations` | Max iterations (`-1` = unlimited) |
| Session | `tool_suppress` | Suppresses `[thinking…]` stream output |
| Session | `stream_level` | 0=sync, 2=concurrent enrichment, 3=sentence-split streaming |

---

## Post-Response Layers

These don't add to the prompt but write to memory, shaping future turns.

### Layer 7 — Memory Scan (Inline Saves)

**What:** Scans the assistant's response for `memory_save()` calls and executes them to write to ST memory.

**Toggles:**
| Scope | Key | Effect |
|---|---|---|
| Model | `memory_scan` | `true` → model writes inline memory_save() syntax; scan fires |
| Global | `memory.post_response_scan` | Master switch for scan feature |
| Session | `memory_scan_suppress` | Suppress scan for this session |

**Note:** `memory_scan: false` + `conv_log: true` is the preferred pattern — model doesn't write inline saves; conv_log handles all persistence automatically.

### Layer 8 — Response Stripping (Cosmetic)

Auto-applied when `memory_scan: false`. Removes any `memory_save()` text from the response before it's stored in history or conv_log. Purely cosmetic — keeps history clean.

### Layer 9 — Judge Response Gate

**What:** Post-LLM pass — evaluates the assistant response. If blocked, replaces it with a rejection notice.

**Toggles:** Same as Layer 3b; gate name is `"response"`.

### Layer 10 — Conv Log (Verbatim Memory)

**What:** Saves the user prompt + assistant response as paired ST memory rows. This is how the conversation becomes retrievable in future turns via Layer 2b.

**Full gate logic — see MEMORY_PROJECT1.md "conv_log + memory_enabled".**

**Summary:**
- `conv_log: true` on model is the feature switch (absent = disabled)
- `memory_enabled` three-level veto (same as Layer 2b) controls whether the write happens
- Topic extracted from `<<topic>>` tag → stored in `session["current_topic"]` for next turn

---

## Configuration Reference by Scope

### Global — `plugins-enabled.json`

| Key | Layer | Default | Effect |
|---|---|---|---|
| `memory.enabled` | 2b, 10 | `true` | Master kill switch for all memory |
| `memory.context_injection` | 2b | `true` | Toggle memory retrieval injection |
| `memory.post_response_scan` | 7 | `true` | Toggle inline memory_save() scanning |
| `plugin_history_default.agent_max_ctx` | 3a | varies | History sliding window token budget |

### Model — `llm-models.json`

| Key | Layer | Notes |
|---|---|---|
| `system_prompt_folder` | 0 | Which prompt tree; `""` = no system prompt |
| `conv_log` | 10 | Feature switch for verbatim turn logging; absent = `false` |
| `memory_enabled` | 2b, 10 | `null`=defer, `false`=hard off for this model |
| `memory_scan` | 7 | `true` = model writes inline memory_save() syntax |
| `llm_tools` | 5 | Authorized toolset group names |
| `llm_tools_gates` | 5 | Tools requiring HITL gate approval |
| `judge_config` | 3b, 9 | Judge model + gates + mode + threshold |
| `limits.max_tool_iterations` | 6 | Agent loop cap |
| `tool_suppress` | 6 | Suppress tool output display |
| `stream_level` | 2, 6 | Streaming behavior level |

### Session — Runtime

| Key | Set via | Layer | Effect |
|---|---|---|---|
| `memory_enabled` | `!memory false` | 2b, 10 | Veto memory injection + conv_log |
| `auto_enrich` | `!config write auto_enrich false` | 2a | Disable DB auto-enrich rules |
| `tool_suppress` | `!config write tool_suppress true` | 6 | Suppress tool output |
| `stream_level` | `!config write stream_level N` | 2, 6 | Streaming mode |
| `memory_scan_suppress` | `!config write memory_scan_suppress true` | 7 | Suppress inline memory scan |
| `current_topic` | Auto-set from `<<topic>>` tag | 2b | Seed for semantic retrieval query |
| `tool_list_injected` | Auto-set on first turn | 1 | One-time tool list gate |

---

## Minimum to Maximum Context Examples

**Bare minimum** (a utility/role model — e.g. `extractor-gemini`):
- System prompt from `005_fileops` tree
- Tool list (one-time)
- History
- User message
- No memory injection (`memory_enabled: false`)
- No conv_log (`memory_enabled: false`)

**Standard interactive model** (e.g. `gemini25f`):
- System prompt from `000_default`
- Tool list (one-time)
- DB auto-enrich results (if patterns match)
- Memory context block (ST + LT semantic retrieval)
- Known topics list
- History (sliding window)
- User message
- Post: judge gate, conv_log to ST, Qdrant upsert

**Full Samaritan** (e.g. `samaritan-voice`):
- Everything above (different system prompt tree: `004_voice`)
- `memory_scan: false` (conv_log handles all persistence; no inline saves)
- `conv_log: true`
- `stream_level: 3` (sentence-split streaming)
- `agent_call_stream: true` (nested model call streaming)
- Topic tag extraction updates `session["current_topic"]` each turn
