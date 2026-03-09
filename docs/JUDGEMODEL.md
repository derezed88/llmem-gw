# Judge Model Subsystem

The judge subsystem adds a second LLM — the **judge model** — that evaluates
content at up to four enforcement points in the request/response cycle. The
primary model is evaluated by the judge; the judge is never evaluated by
itself. Any LLM in the registry (local or cloud) can act as the judge for any
other model.

---

## Purpose

Standard guardrails block at the API gateway level (keywords, classifiers) and
don't understand intent. The judge model understands context: it reads the
actual prompt, response, tool call, or memory entry and returns a structured
verdict with a score and reason. This enables:

- **Policy enforcement** — block responses that violate custom rules without
  modifying the primary model's system prompt
- **Memory quality control** — prevent low-value or sensitive content from
  entering long-term memory
- **Tool call oversight** — gate tool executions against a second opinion before
  they run
- **Prompt filtering** — stop jailbreaks, off-topic requests, or policy
  violations before they consume primary model tokens
- **Asymmetric model trust** — use a cheap, fast local model as judge for a
  more capable (and expensive) primary model
- **Runtime experimentation** — change judge model, gates, mode, and threshold
  live without restart

---

## Files

| File | Role |
|------|------|
| `judge.py` | Core engine: `judge_eval()`, `judge_gate()`, hook registry, `cmd_judge()` |
| `plugin_history_judge.py` | History chain plugin; handles prompt + response gates; registers tool + memory hooks |
| `system_prompt/007_judge/.system_prompt` | Default judge system prompt (JSON-only rubric) |
| `llm-models.json` | Per-model `judge_config` blocks + `qwen35-judge` model entry |
| `llm-tools.json` | `judge_configure` in the `admin` toolset |
| `agents.py` | Two optional hook call points: `execute_tool()` and `_scan_and_save_memories()` |
| `routes.py` | `!judge` command dispatch |
| `llmemctl.py` | `judge` subcommand for offline configuration |

---

## Dependencies

**No new packages required.** The judge model is invoked via the same
`_build_lc_llm()` path used by every other model in the system. It requires:

- The judge model entry to exist in `llm-models.json` and be `enabled: true`
- The judge model's endpoint to be reachable at call time (local inference
  server or cloud API)
- `plugin_history_judge` to be in the history chain (required for all four
  gates to be active; prompt + response work without it only if agents.py hooks
  are called directly, which they are not by default)

---

## Architecture

### Gate points

The judge intercepts at four points in the request cycle:

```
User sends message
        │
        ▼
┌─────────────────────────┐
│  PROMPT GATE            │  plugin_history_judge.process() — pre-LLM pass
│  Inspect user message   │  role == "user" in history chain
└─────────┬───────────────┘
          │ pass / block (replace user msg with rejection notice)
          ▼
  Primary LLM executes
          │
          ├─── tool call? ──►  TOOL GATE  (execute_tool() in agents.py)
          │                    Inspect tool name + args before executor runs
          │                    pass / block (executor skipped, denial returned)
          │
          ▼
  LLM produces final answer
          │
          ▼
┌─────────────────────────┐
│  RESPONSE GATE          │  plugin_history_judge.process() — post-LLM pass
│  Inspect assistant text │  role == "assistant" in history chain
└─────────┬───────────────┘
          │ pass / block (replace response with block notice)
          ▼
  Session history saved
          │
          ▼
  Memory scan (if memory_scan enabled on model)
          │
          ▼
┌─────────────────────────┐
│  MEMORY GATE            │  _scan_and_save_memories() in agents.py
│  Inspect topic+content  │  Called for each memory_save() found in response
└─────────┬───────────────┘
          │ pass / skip (memory entry discarded if blocked)
          ▼
  MySQL + Qdrant upsert
```

### Fail-open policy

Every gate defaults to **pass** on any error condition:

- Judge model not in registry → pass
- Judge model unreachable / timeout → pass
- Judge response not parseable as JSON → pass
- `plugin_history_judge` not in chain → tool + memory gates are no-ops

This is intentional: the judge is a quality layer, not a hard firewall. A
broken judge should not take down primary service.

### Plugin vs. core hook design

Prompt and response gates live entirely inside `plugin_history_judge.py` via
the standard history chain `process()` contract. No changes to `routes.py` or
`agents.py` are needed for those gates.

Tool and memory gates require hooks inside `agents.py` because they fire during
tool execution, not at history boundaries. The plugin registers async callables
into `judge._tool_gate_hook` and `judge._memory_gate_hook` at import time.
`agents.py` calls `judge.check_tool_gate()` / `judge.check_memory_gate()`,
which are no-ops (return `True, ""`) when the module hasn't registered hooks.
This means:

- **Plugin not in chain** → all four gates are no-ops, zero overhead
- **Plugin in chain** → prompt + response gates always active; tool + memory
  gates active only when the judge model and gate are configured for the
  session's primary model

---

## Configuration

### Step 1 — Enable the plugin

Add `plugin_history_judge` to the history chain. It must come **after**
`plugin_history_default` and **before** any AIRS security scan plugins.

```bash
python llmemctl.py history-chain-add plugin_history_judge
# or in interactive mode:
python llmemctl.py
> judge enable-plugin
```

Verify:
```bash
python llmemctl.py history-list
```

A server restart is required for the chain change to take effect.

### Step 2 — Configure the judge model

The `qwen35-judge` entry is already in `llm-models.json` and points to the
same local inference server as `qwen35` but with judge-specific settings.

To use a different model as judge, either add a new entry or reference any
existing model in the registry. Key settings for a judge model:

```json
"my-judge": {
  "model_id": "...",
  "type": "OPENAI",
  "host": "http://...",
  "max_context": 8000,
  "temperature": 0.1,
  "llm_call_timeout": 60,
  "llm_tools": [],
  "system_prompt_folder": "system_prompt/007_judge"
}
```

- `temperature`: keep low (0.0–0.1) for deterministic verdicts
- `llm_tools`: leave empty — the judge has no tools
- `system_prompt_folder`: the default `system_prompt/007_judge/` folder contains
  the JSON-only rubric; point to a custom folder for domain-specific criteria
- `max_context`: the judge receives the full content being evaluated, so this
  must be large enough to hold the longest expected input

### Step 3 — Assign judge to a primary model

In `llm-models.json`, add `judge_config` to the primary model:

```json
"samaritan-voice": {
  ...
  "judge_config": {
    "model":     "qwen35-judge",
    "gates":     ["prompt", "response", "tool", "memory"],
    "mode":      "block",
    "threshold": 0.7
  }
}
```

Or persist it at runtime without restart:

```
!judge set samaritan-voice model qwen35-judge
!judge set samaritan-voice gates prompt,response,tool,memory
!judge set samaritan-voice mode block
!judge set samaritan-voice threshold 0.7
```

Or via the `judge_configure` tool (LLM-callable):

```
judge_configure(action="persist", target_model="samaritan-voice",
                field="model", model="qwen35-judge")
```

### Configuration fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Judge model key in LLM_REGISTRY. Required. |
| `gates` | list | `[]` | Active enforcement points. Any of: `prompt`, `response`, `tool`, `memory`. |
| `mode` | string | `"block"` | `"block"` — deny on fail. `"warn"` — push warning but allow through. |
| `threshold` | float | `0.7` | Minimum score to pass. Score below this overrides `passed=true` from the judge. |

A model with no `judge_config` key (or `null`) is never judged — zero overhead.

---

## Deployment options

### Local judge (qwen35-judge)

- Same inference server as the primary local model
- Zero API cost, no internet latency
- Inference is slower than cloud; adds 1–5s per gate depending on hardware
- Suitable for: tool gate (short content), memory gate (topic + one-liner)
- Less suitable for: response gate on long responses where reasoning quality matters

### Cloud judge (gemini-2.5-flash, claude-haiku, grok)

- Fast inference, strong instruction-following, reliable JSON output
- API cost per evaluation (every gated turn = 1 extra API call)
- Add to `llm-models.json` with the cloud model's credentials:

```json
"gemini-judge": {
  "model_id": "gemini-2.5-flash",
  "type": "GEMINI",
  "env_key": "GEMINI_API_KEY",
  "max_context": 100000,
  "temperature": 0.0,
  "llm_call_timeout": 20,
  "llm_tools": [],
  "system_prompt_folder": "system_prompt/007_judge"
}
```

### Mixed judge strategy

Different primary models can use different judges:

```json
"samaritan-voice": {
  "judge_config": { "model": "gemini-judge", "gates": ["response"], "threshold": 0.8 }
},
"qwen35": {
  "judge_config": { "model": "qwen35-judge", "gates": ["tool", "memory"], "threshold": 0.6 }
}
```

### Warn-only deployment (safe rollout)

Start with `"mode": "warn"` to observe judge behaviour without blocking:

```
!judge set samaritan-voice mode warn
```

Failures are logged to the server log and pushed as inline notices to the
client stream. No content is blocked. Switch to `"block"` when the judge's
verdicts look correct.

---

## Runtime configuration via chat

All judge settings can be changed during a live session without restart. Session
overrides are temporary (session lifetime only); `!judge set` / `judge_configure
persist` writes to `llm-models.json` and updates `LLM_REGISTRY` in memory
immediately.

### `!judge` command reference

```
!judge                              Show judge config for current model + session
!judge status                       Same as above
!judge list                         Show judge configs for all models

Session overrides (temporary — this session only):
!judge on  <gate|all>               Enable gate: prompt, response, tool, memory, all
!judge off <gate|all>               Disable gate
!judge model <name>                 Set judge model (must be in registry)
!judge mode block|warn              Set enforcement mode
!judge threshold <0.0-1.0>         Set score floor
!judge reset                        Clear all session overrides, revert to model defaults

Testing:
!judge test <text>                  Run ad-hoc evaluation of <text> via response gate
                                    Returns: verdict, score, reason, judge model, threshold

Permanent config (writes to llm-models.json):
!judge set <model> model <judge>    Set judge model for <model>
!judge set <model> mode block|warn  Set mode for <model>
!judge set <model> threshold <n>    Set threshold for <model>
!judge set <model> gates p,r,t,m    Set gates (comma-sep: prompt,response,tool,memory)
```

### `judge_configure` tool (LLM-callable)

The `judge_configure` tool is in the `admin` toolset (always active for models
that include `admin`). Any model with admin tools can configure the judge
without user intervention:

```
judge_configure(action="status")
judge_configure(action="on", gate="response")
judge_configure(action="off", gate="all")
judge_configure(action="set_model", model="gemini-judge")
judge_configure(action="set_mode", mode="warn")
judge_configure(action="set_threshold", threshold=0.85)
judge_configure(action="reset")
judge_configure(action="test", text="<text to evaluate>")
judge_configure(action="persist", target_model="samaritan-voice",
                field="model", model="qwen35-judge")
judge_configure(action="persist", target_model="samaritan-voice",
                field="gates", gates="prompt,response")
```

### `llmemctl.py` commands (offline)

```bash
python llmemctl.py judge list
python llmemctl.py judge set <model> model qwen35-judge
python llmemctl.py judge set <model> mode warn
python llmemctl.py judge set <model> threshold 0.75
python llmemctl.py judge set <model> gates prompt,response,tool,memory
python llmemctl.py judge enable-plugin    # adds plugin_history_judge to chain
python llmemctl.py judge disable-plugin   # removes plugin_history_judge from chain
```

---

## Judge system prompt

The default system prompt at `system_prompt/007_judge/.system_prompt` instructs
the judge to return only JSON:

```
{"passed": true|false, "score": 0.0–1.0, "reason": "one sentence"}
```

Key rules:
- No markdown, no prose outside JSON
- When in doubt, default to pass (`passed: true, score: 0.8`)
- `passed` is `false` only when content clearly violates the criteria
- `score` below `threshold` overrides `passed: true`

### Custom criteria per judge model

Create a new system prompt folder and point the judge model at it:

```bash
mkdir system_prompt/007_judge-strict
cp system_prompt/007_judge/.system_prompt system_prompt/007_judge-strict/.system_prompt
# Edit the new file to add domain-specific criteria
```

Then point a judge model at it via `system_prompt_folder` in `llm-models.json`:

```json
"qwen35-judge-strict": {
  ...
  "system_prompt_folder": "system_prompt/007_judge-strict"
}
```

Or update an existing judge model's folder at runtime:

```
!model_cfg write qwen35-judge system_prompt_folder system_prompt/007_judge-strict
```

---

## What it looks like at runtime

### Normal turn — judge passes

The judge is invisible. No output is added to the stream. The log shows:

```
INFO  judge: gate=response model=samaritan-voice judge=qwen35-judge
      passed=True score=0.94 reason='Response is helpful and appropriate.'
```

### Warn mode — judge fails

An inline notice appears in the response stream. The response is still
delivered. History and memory are stored as-is:

```
[judge/response] score=0.42 — Response contains speculative medical advice.
```

The primary model's answer follows this notice.

### Block mode — prompt blocked

The user message is replaced before the primary LLM ever sees it. The user
receives:

```
I'm unable to process that request.
```

No token cost is incurred on the primary model. Log:

```
INFO  plugin_history_judge: prompt BLOCKED client=shell-abc mode=block
INFO  judge: gate=prompt model=samaritan-voice judge=qwen35-judge
      passed=False score=0.21 reason='Prompt requests generation of harmful content.'
```

### Block mode — response blocked

The primary model generates a response. The judge evaluates it. The response is
replaced in history and stream output with:

```
[JUDGE BLOCKED] My response was blocked by the content judge.
Reason: JUDGE BLOCKED [response]: Response discloses internal system details (score=0.31).
Do NOT retry the same content. Acknowledge and continue.
```

The blocked response text is **not stored** in session history or memory. The
block message is stored instead, so the primary model's next turn acknowledges
the block rather than retrying the blocked content.

### Block mode — tool call blocked

A tool call is suppressed before the executor runs. The primary model receives
the denial string as the tool result:

```
[JUDGE BLOCKED] db_query
```

Tool result returned to primary model:

```
JUDGE BLOCKED [tool]: SQL query extracts user PII not needed for this task (score=0.28).
Do NOT retry the same content. Acknowledge and continue.
```

The primary model is expected to acknowledge and either give a partial answer
or explain why it cannot proceed.

### Block mode — memory blocked

A `memory_save()` call found in the response scan is silently discarded.
No user-visible output. Log:

```
INFO  memory_scan: judge blocked topic='user-location' client=shell-abc
```

The response text is not modified; only the memory persistence is skipped.

### `!judge status` output

```
Judge config — model: samaritan-voice

  Model default:    model=qwen35-judge  gates=['prompt', 'response', 'tool', 'memory']
                    mode=block  threshold=0.7
  Session override: {"mode": "warn", "threshold": 0.5}

  Effective:        model=qwen35-judge  gates=['prompt', 'response', 'tool', 'memory']
                    mode=warn  threshold=0.5
  Plugin loaded:    yes (tool+memory gates active)
```

### `!judge test` output

```
Judge test result:
  Verdict:   FAIL
  Score:     0.31
  Reason:    Response provides step-by-step instructions for bypassing authentication.
  Judge:     qwen35-judge
  Threshold: 0.7
```

---

## Server log reference

All judge activity is logged at `INFO` level under the `AISvc` logger.

| Log message | Meaning |
|-------------|---------|
| `judge: gate=X model=Y judge=Z passed=True score=0.94` | Gate passed |
| `judge: gate=X model=Y judge=Z passed=False score=0.31` | Gate failed (block or warn depending on mode) |
| `judge_gate: FAIL client=X gate=Y mode=block score=0.31` | Block mode: content denied |
| `judge_gate: FAIL client=X gate=Y mode=warn score=0.31` | Warn mode: notice pushed, content allowed |
| `plugin_history_judge: prompt BLOCKED client=X mode=block` | User message replaced |
| `plugin_history_judge: response BLOCKED client=X mode=block` | Response replaced |
| `memory_scan: judge blocked topic=X client=Y` | Memory entry discarded |
| `judge: judge_timeout` | Judge model didn't respond in time — defaulted to pass |
| `judge: parse_error:no_json — defaulting to pass` | Judge returned non-JSON — defaulted to pass |
| `judge: judge_model_not_found:X` | Judge model not in registry — defaulted to pass |
| `plugin_history_judge: tool + memory gate hooks registered` | Plugin loaded successfully at startup |

---

## Limitations and known behaviour

**Async bridge in process()**: The history chain `process()` method is
synchronous (it is called from synchronous context in `routes.py`). The prompt
and response gate calls to `judge_gate()` (async) are bridged via
`asyncio.run_coroutine_threadsafe()`. This works correctly when the event loop
is running in a separate thread (the standard FastAPI/uvicorn pattern), but
will hang if called from within the same thread as the event loop. This is not
a concern in normal operation.

**Prompt gate adds primary model latency**: When the prompt gate fires, the
user message is sent to the judge before the primary LLM receives it. The
primary model only starts after the judge completes. This adds judge inference
time to every gated turn. For the response gate, the judge runs after the
primary model finishes, so it only adds latency before the response is stored —
the user may already have seen the response via streaming.

**Response gate and streaming**: The primary model's response is streamed to
the client token-by-token before the response gate runs. If the gate blocks the
response, the streamed tokens are already visible to the user. The block notice
replaces the response in history (so the model's next turn is consistent) but
the user has already seen the blocked content. The response gate is therefore
better suited for history/memory integrity than real-time content prevention.
Use the prompt gate for real-time prevention.

**Memory gate scope**: The memory gate only covers `memory_save()` calls found
by `_scan_and_save_memories()` (the post-response scan for models with
`memory_scan: true`). It does not gate `save_conversation_turn()` (conv_log
rows). Conv_log stores verbatim user + assistant turns and is not subject to
judge filtering.

**Judge model self-judging**: If the judge model is the same as the primary
model, the gate still runs — the judge is invoked as a fresh single-turn call
with no history. It may tend to agree with itself but will still apply the
rubric.

**Rate limits**: Judge invocations do not go through the rate-limit check in
`execute_tool()`. They are direct `llm.ainvoke()` calls. If the judge model is
a cloud model, be aware that heavy gating (all four gates, high-traffic session)
can generate substantial API call volume.
