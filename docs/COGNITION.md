# Cognitive Architecture — llmem-gw

Last updated: 2026-03-10

This document describes the cognitive architecture built on top of the tiered memory system. It was designed around the question: *what pieces does a reactive LLM agent need to become autonomous?* The answer shaped everything here.

---

## Overview

The architecture extends the two-tier memory system (ST/LT + Qdrant) into a broader cognitive stack. Rather than building separate infrastructure for each cognitive function, almost everything reuses existing write paths (`save_memory()`), existing query patterns (Qdrant semantic search), and existing background loop patterns (aging timers).

The design principle: **memory is the substrate**. Goals, beliefs, drives, self-knowledge, and procedural skills are all typed memory rows — they differ in how they are written, how they age, and how they are injected into context.

---

## Typed Memory System

**`memory.py` — `save_memory(type=...)`**

Memory rows carry a `type` field, validated against `_MEMORY_TYPES`:

| Type | Purpose | Aging |
|---|---|---|
| `context` | Default conversational context | Normal HWM/LWM |
| `goal` | Active objective | Protected (importance ≥ 9) |
| `plan` | Ordered steps toward a goal | Normal |
| `belief` | Asserted world-state fact | Protected, contradiction-scanned |
| `episodic` | Specific event or experience | Normal |
| `semantic` | General knowledge | Normal |
| `procedural` | Skill / reusable task pattern | Separate Qdrant collection |
| `autobiographical` | Identity and self-story | Protected |
| `prospective` | Future reminder with `due_at` | Expires on fire |
| `conditioned` | Learned stimulus-response pair | Strength-tracked |
| `self_model` | Synthesized self-summary | Protected (importance = 10) |

**Dedicated tables** (migrations `001`–`004`): `samaritan_goals`, `samaritan_plans`, `samaritan_beliefs`, `samaritan_episodic`, `samaritan_semantic`, `samaritan_procedural`, `samaritan_autobiographical`, `samaritan_prospective`, `samaritan_conditioned`, plus a drives table.

Context injection uses `load_typed_context_block()` which groups retrieved rows by type into labeled sections in the prompt.

---

## Goal & Plan Management

**`memory.py` lines 1414–1482 | `tools.py` lines 814–914**

Goals are stored in `samaritan_goals` with status: `active`, `done`, `blocked`, `abandoned`. Plans are ordered steps FK-linked to a goal.

LLM-callable tools: `set_goal`, `set_plan` (via `_set_goal_exec`, `_set_plan_exec` in tools.py).

Goal state feeds the drives system: completion rates nudge `task-completion` drive up; blocked goals nudge `discomfort` drive up and `task-completion` drive down.

**Gap**: Goals are stored but not yet auto-injected into context. A `load_goals()` section in `load_typed_context_block()` is the natural next step.

---

## Belief System & Contradiction Detection

**`memory.py` | `contradiction.py`**

Beliefs are stored in `samaritan_beliefs` with a `confidence` score (1–10) and status `active`/`retracted`.

`contradiction.py` runs as a background async task every 24 hours:
1. Fetches all active beliefs, groups by topic
2. Sends pairs to the judge/LLM for conflict scoring
3. Writes `contradiction-flag` rows (topic=`contradiction-flag`, confidence=9) when conflicts are detected
4. Optionally auto-retracts the lower-confidence belief
5. Measures resolution ratio (retracted flags / total flags) → feeds into the cognitive feedback loop

Beliefs are vector-indexed in Qdrant alongside other memory types, so they surface in semantic retrieval.

---

## Self-Model

**`memory.py` lines 1247–1325 | `reflection.py`**

The self-model is a reserved topic namespace written by the reflection loop:

| Topic prefix | Meaning |
|---|---|
| `self-capability-*` | Things the agent does well |
| `self-failure-*` | Things the agent struggles with |
| `self-preference-*` | Stylistic and behavioral tendencies |
| `self-summary` | Synthesized distillation of all self-* rows |

These topic prefixes are in `protected_topic_prefixes` — they are never chunked away during aging.

`refresh_self_summary()` runs every 5 reflection cycles: it pulls all `self-*` rows, distills them into 3–5 bullet points via LLM, and saves a `self-summary` row (type=`self_model`, importance=10). This row is injected as a `## Self-Model` section in context.

---

## Procedural Memory

**`memory.py` lines 1489–1600+ | `migrations/003_procedural_structured.sql`**

`save_procedure(topic, task_type, steps, outcome, notes, importance, source, session_id, id)` stores reusable task patterns:

- `steps` is a JSON list of `{action, detail}` dicts
- `outcome`: `success` / `partial` / `failure` / `unknown`
- `run_count` incremented on update (same `id`)
- Embedded and stored in a **separate Qdrant collection** (`samaritan_procedures`) so procedure search doesn't pollute memory search

Outcome updates are lightweight: `_update_procedure_outcome_vec()` does a payload-only Qdrant update without re-embedding.

---

## Drives / Affect

**`memory.py` lines 1329–1482 | `migrations/004_drives.sql` | `plugins-enabled.json` lines 129–166**

Five drives stored in `samaritan_drives`:

| Drive | Baseline | Role |
|---|---|---|
| `curiosity` | 0.6 | Follow novel threads |
| `task-completion` | 0.7 | Finish what was started |
| `user-trust` | 0.8 | Defer to user judgment |
| `discomfort` | 0.2 | Signal unease |
| `autonomy` | 0.4 | Initiate proactive suggestions |

Each drive has a `value` (0–1), a `baseline` it decays toward, and a `decay_rate` per reflection cycle. `discomfort` decays faster (0.1) than the others (0.05).

`decay_drives()` runs each reflection cycle: exponential decay toward baseline.
`update_drives_from_goals()` nudges values based on recent goal completions and blocks.

Drives are intended to modulate goal prioritization (sort active goals by `drive_weight × importance`) — the coupling to context injection is the natural next step.

---

## Proactive Cognition Loops

Three independent background async tasks, all using the same timer/init pattern as the existing aging loops:

### Reflection — `reflection.py`

**Interval**: 6 hours (configurable)

1. Pulls recent ST turns (oldest 40 rows, source=`user`/`assistant`)
2. LLM extracts insights, patterns, self-model rows
3. Saves up to 6 new ST rows per cycle (importance 5–10, typed)
4. Writes `self-capability-*`, `self-failure-*`, `self-preference-*` rows
5. Every 5 cycles: calls `refresh_self_summary()`
6. After each cycle: calls `update_drives_from_goals()`
7. Measures access ratio of rows saved → feeds cognitive feedback loop

### Contradiction Scan — `contradiction.py`

**Interval**: 24 hours (configurable)

See Belief System above. Feeds back into cognitive feedback loop via resolution ratio.

### Prospective Reminders

**Interval**: 5 minutes (configurable)

Queries `samaritan_prospective` for rows with `due_at <= now()`. Fires due reminders into the next session context. Measures usage ratio (reminders acted on vs. fired) → feeds cognitive feedback loop.

---

## Cognitive Feedback Loop

**`cogn_feedback.py` lines 1–506**

This is the meta-learning layer: loops self-assess their own usefulness and adjust their behavior.

Each loop has a **conditioned row** in `samaritan_conditioned` (topic=`cogn-feedback-<loop_name>`). The conditioned row has a `strength` (1–10) that tracks how useful the loop has been:

| Metric | Loop |
|---|---|
| Access ratio (rows saved vs. later retrieved) | Reflection |
| Usage ratio (reminders fired vs. acted on) | Prospective |
| Resolution ratio (contradiction flags retracted / total) | Contradiction |

Outcome ladder:

| Verdict | Action |
|---|---|
| `useful` | Decrement streak, recover if extinguished |
| `neutral` | No change |
| `low` | Increment strength |
| `throttle` (strength ≥ 7) | Double loop interval |
| `extinguish` (strength ≥ 10) | Disable loop |
| `recover` | Clear interval override, reactivate |

This means a loop that consistently produces memories nobody retrieves will throttle and eventually extinguish itself — without manual intervention.

---

## LLM-as-Judge

**`judge.py` | `plugin_history_judge.py` | `docs/JUDGEMODEL.md`**

Four gate points where a judge model evaluates content before it proceeds:

| Gate | Timing | Content |
|---|---|---|
| `prompt` | Pre-LLM | User message |
| `response` | Post-LLM | Assistant response |
| `tool` | Pre-execute | Tool call + arguments |
| `memory` | Pre-save | Memory content |

Each gate returns `{passed, score, reason}`. Mode is `block` (hard gate) or `warn` (log only). Fail-open policy: errors default to pass.

Per-model configuration in `llm-models.json`:
```json
"judge_config": {
  "model": "judge-qwen35",
  "gates": ["prompt", "response", "tool", "memory"],
  "mode": "block",
  "threshold": 0.7
}
```

The `!judge` command reports gate hit statistics per session.

---

## Architecture Map

```
External prompt
      │
      ▼
  [prompt gate]  ──judge.py──────────────────────────────┐
      │                                                   │
      ▼                                                   │
 auto_enrich_context()                                    │
  ├─ load_context_block()   ← ST + LT semantic retrieval  │
  ├─ load_typed_context_block()  ← goals, beliefs, self   │
  ├─ load_drives()          ← affect state                │
  └─ prospective due items                                │
      │                                                   │
      ▼                                                   │
   LLM call                                               │
      │                                                   │
      ▼                                                   │
  [response gate]  ────────────────────────────────────── ┤
      │                                                   │
      ▼                                                   │
  tool execution                                          │
      │                                                   │
  [tool gate]  ──────────────────────────────────────────┤
      │                                                   │
      ▼                                                   │
  memory scan  →  [memory gate]  →  save_memory()         │
      │                              │                    │
      │                    MySQL INSERT + Qdrant upsert   │
      │                                                   │
Background loops (async, independent):                    │
  reflection.py  (6h)   → save_memory() + refresh_self   │
  contradiction.py (24h) → write flags, retract beliefs  │
  prospective    (5m)   → fire due reminders              │
  cogn_feedback.py      → update conditioned strengths   │
  aging loops    (60s/360s) → ST→LT summarization        │
```

---

## What Was Planned vs. What Was Built

The original plan proposed seven missing layers. Here is the actual outcome:

| Planned | Outcome |
|---|---|
| Goal stack via type field + goal_manager.py | Built: typed goals table, tools, drive coupling. **Gap**: not yet auto-injected into context |
| World model / beliefs via loose triple store | Built as typed beliefs + full contradiction scanner — more complete than planned |
| Proactive loops via aging timer pattern | Built: reflection + contradiction + prospective, all with feedback learning |
| Surprise scoring at save time via Qdrant novelty check | **Not built as planned.** Replaced by retroactive access-ratio measurement — a stronger signal (actual usefulness vs. guessed novelty) |
| Self-model via protected topic namespace | Built exactly as planned, plus `refresh_self_summary()` synthesis loop |
| Drives via state.py session state | Built as persistent DB table with decay and goal-coupling, more durable than planned |
| Procedural memory as second Qdrant collection | Built exactly as planned |

The notable design divergence: **surprise scoring** was replaced by the cognitive feedback loop's access-ratio system. Instead of estimating novelty at write time (which requires a speculative Qdrant round-trip for every save), the system measures actual impact retroactively. A memory nobody ever retrieves is more actionable evidence than a save-time novelty estimate.

---

## Open Items

1. **Goal injection into context** — `load_typed_context_block()` needs a goals section so the model sees active goals during reasoning
2. **Drive-weighted goal prioritization** — sort injected goals by `drive.value × goal.importance`
3. **Prospective loop full implementation** — scaffolding exists, full fire/feedback cycle needs verification
4. **Goal completion detection** — model needs a signal path to mark goals done from conversation (currently tool-only)
