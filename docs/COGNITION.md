# Cognitive Architecture — llmem-gw

Last updated: 2026-04-01

This document describes the cognitive architecture built on top of the tiered memory system. It was designed around the question: *what pieces does a reactive LLM agent need to become autonomous?* The answer shaped everything here.

---

## Overview

The architecture extends the two-tier memory system (ST/LT + Qdrant) into a broader cognitive stack. Rather than building separate infrastructure for each cognitive function, almost everything reuses existing write paths (`save_memory()`), existing query patterns (Qdrant semantic search), and existing background loop patterns (aging timers).

The design principle: **memory is the substrate**. Goals, beliefs, drives, self-knowledge, and procedural skills are all typed memory rows — they differ in how they are written, how they age, and how they are injected into context.

As of 2026-04-01, three of the five original Python async cognitive loops (reflection, contradiction, goal health) have been replaced by an event-driven Claude Code session (`samaritan-cognition`). See the [Claude Code Cognition Session](#claude-code-cognition-session) section below.

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

**Dedicated tables** (migrations `001`–`010`): `samaritan_goals`, `samaritan_plans`, `samaritan_beliefs`, `samaritan_episodic`, `samaritan_semantic`, `samaritan_procedural`, `samaritan_autobiographical`, `samaritan_prospective`, `samaritan_conditioned`, `samaritan_drives`, `samaritan_temporal`, `samaritan_cognition`, `samaritan_tool_stats`.

Context injection uses `load_typed_context_block()` which groups retrieved rows by type into labeled sections in the prompt.

---

## Goal & Plan Management

**`memory.py` | `tools.py` | `plan_engine.py` | `goal_processor.py`**

### Goals

Goals are stored in `samaritan_goals` with status: `active`, `done`, `blocked`, `abandoned`. LLM-callable tools: `set_goal`, `set_plan` (via `_set_goal_exec`, `_set_plan_exec` in tools.py). Keyword blocker in `_set_goal_exec()` blocks goals with timer/swarm/DDL/cron keywords.

Goal state feeds the drives system: completion rates nudge `task-completion` drive up; blocked goals nudge `discomfort` drive up and `task-completion` drive down.

Active goals are injected into every context block via `load_typed_context_block()` as a `## Active Goals` section, sorted by drive-weighted priority (see Drives below). The reflection loop also detects goal completions from conversation evidence and marks them `done` automatically.

### Plan Engine — Two-Tier Decomposition

**`plan_engine.py` | migration `006_plan_decomposition.sql`**

Plans use a two-tier decomposition model:

1. **Concept steps** — Human-intent-level descriptions (e.g., "research unemployment benefits")
2. **Task steps** — Executable atoms with `tool_call` specs, FK-linked to a parent concept step via `parent_id`

Each step has:
- `step_type`: `concept` or `task`
- `target`: `model` (auto-execute), `human` (pause and notify), `investigate` (unresolved), `claude-code` (queued for a Claude Code session), `claude-cognition` (queued for samaritan-cognition session)
- `approval`: `proposed`, `approved`, `rejected`
- `tool_call`: JSON blob with `{tool, args}` for task steps
- `executor`: which model/session ran the step
- `result`: execution output

**Decomposition model**: `model_roles["plan_decomposer"]` (currently Sonnet 4.6, upgraded from Haiku 4.5 to eliminate model-name hallucinations in `tool_call` specs).

**Decomposer rules** (enforced in decomposer system prompt, `plan_engine.py`):
- Each task step = one atomic tool call or one human action
- Only tools from the available tools catalog may be used
- `samaritan-execution` **requires** `mode="tool"` with an explicit `tool=` argument (rejects `mode="text"`)
- For synthesis/summarization, use `summarizer-gemini` with `mode="text"`
- Sequential dependencies use placeholder descriptions, resolved at runtime

### Execution Chain (2 Tiers)

1. **Direct executor** — parse `tool_call` JSON, look up the Python executor function via `get_tool_executor()`, map arguments, and call directly. Zero LLM cost. This is the happy path for most steps.
2. **LLM executor** — LLM-based execution via `model_roles["plan_executor"]` (`samaritan-execution`). Only invoked if Tier 1 throws an exception. The LLM receives the tool name, args, step description, and the direct-execution error, then calls the tool via `llm_call(mode="tool")` with the ability to adapt arguments or recover. Provider failover is handled by `backup_models` on the model config (→ `gpt-4o-execution`) — no separate fallback role needed.

**When Tier 1 fails and escalates to Tier 2** — examples:

| Scenario | What happens | How Tier 2 recovers |
|---|---|---|
| Argument mismatch | Decomposer generates args that don't match the executor function signature (e.g., missing `mode` on `llm_call`) — direct call throws `TypeError` | LLM infers the missing parameter from context |
| Stale session reference | `tmux_exec` targets a session that doesn't exist (prior step failed or was skipped) — runtime error | LLM can create the session first or adapt the command |
| Template placeholders | Decomposer writes `{{qdrant_output}}` placeholder that isn't substituted — direct call sends the literal string | LLM interprets intent and fills in from prior step context |
| Dynamic argument construction | Step needs output from a previous step but `tool_call` args are static JSON | LLM pulls step history and constructs a meaningful prompt |
| Permission/state errors | `db_query` requires `set_model_context()` call first — direct call fails with DB context error | LLM sets up state before calling |

Tier 1 is a dumb function call — if the JSON spec is perfectly formed and the runtime state is exactly right, it works. Tier 2 adds reasoning to recover from gaps between the plan and reality.

### Completion & Cleanup Cascade

**Automatic completion** (`plan_engine.py`):

```
task done → _check_parent_completion() → concept done → _check_goal_completion() → goal done
```

- `_check_parent_completion`: when all task steps under a concept are done/skipped, concept is marked done
- `_check_goal_completion`: when all **approved** concept steps are done/skipped, goal is marked done
  - Only approved concepts block completion — proposed/rejected concepts do not
  - Must have at least one approved concept to complete (guard against empty plans)

**Orphan cleanup**: when a goal completes, all remaining pending/proposed plan steps are auto-skipped with result `"Skipped: parent goal completed"`. This prevents orphaned plans from accumulating under completed goals.

**Explicit closing**: user can close goals/plans directly:
- `!plan auto done <goal_id> <step_id>` — mark a specific step done and resume execution
- Model can delegate `set_goal(id=N, status='done')` or `set_plan(id=N, status='done')` to close items

### Goal Processor — Autonomous Goal Flow

**`goal_processor.py` | migration `010_goal_auto_process.sql`**

Background task that autonomously scans for goals, proposes plans, and executes approved steps. Modeled after the Claude Code approve-then-execute flow.

**`auto_process_status` state machine** (column on `samaritan_goals`):

```
NULL → proposed → approved → executing → completed
                ↘ deferred (with defer_until datetime)
                ↘ rejected
       approved → executing → paused_user (human step or failure) → approved (via !plan auto done)
```

### Full Goal Lifecycle — End-to-End

**Phase 1: Goal Creation**
- User asks for multi-step work → reasoning model creates goal via `set_goal` delegation + 1-2 concept steps via `set_plan`
- Model reports goal ID and step IDs, then **stops** (goal-first execution rule)
- Goal appears as `status=active`, `auto_process_status=NULL`

**Phase 2: Autonomous Planning**
- Goal processor scanner (30-min cycle) finds unplanned goals
- Decomposer creates concept + task steps via LLM
- Goal marked `auto_process_status=proposed`
- Notification sent to user: "Goal N: X steps (Y auto, Z user). `!plan auto approve N`"

**Phase 3: User Approval**
- `!plan auto approve <goal_id>`:
  - Validates the ID is a goal (not a step — shows "Did you mean?" if wrong)
  - Sets `auto_process_status=approved`, approves all proposed steps
  - Shows what happens next: "1 concept step(s) will be decomposed..." or "3 task(s) ready to execute"
  - Registers the session for progress notifications
  - Triggers goal processor immediately
- Alternatives: `!plan auto defer <goal_id>`, `!plan auto reject <goal_id>`

**Phase 4: Execution with Progress**
- Goal processor picks up approved goals, sets `auto_process_status=executing`
- **Auto-decomposition**: if approved concept steps have no task children, they are decomposed on the fly (with notification: "decomposing N concept step(s)...")
- Tasks execute serially via 2-tier chain (direct → LLM executor)
- **Per-step progress**: after each task completes, initiating session receives: "Goal N: step 3/5 done — Read competitor report"
- **Human steps**: execution pauses, user notified: "Goal N waiting on you: Step [M]: ... When done: `!plan auto done N M`"

**Phase 5: Error Handling**
- When a step fails:
  - Step reverted to `pending`, goal set to `active` + `paused_user` (NOT `blocked`)
  - `_check_parent_failure` notes error on parent concept (audit trail) but does **not** block the goal
  - User notified: "Goal N step [M] failed: ... Fix or skip: `!plan auto done N M`"
  - Double failures don't compound — goal stays `active` + `paused_user`
- Recovery via `!plan auto done <goal_id> <step_id>`:
  - Marks step done, clears FAILED text from parent concept
  - Sets goal to `active` + `approved` (not just `executing` — ensures processor picks it up)
  - Registers session for progress, triggers processor immediately

**Phase 6: Completion**
- All approved concept steps done → `_check_goal_completion` fires:
  - Goal marked `status=done`, `auto_process_status=completed`
  - Orphaned pending/proposed steps auto-skipped
- Completion notification to initiating session: "Goal N completed — X task(s) done. Results: ..."
- Initiator session tracking cleaned up

### Plan-Only Workflows (No Goal)

Plans can exist without goals (`goal_id=0`):
- `!plan adhoc <description>` — creates a concept step with `goal_id=0`
- `!plan decompose` — decomposes pending concepts (any goal, including 0)
- `!plan execute` — executes approved tasks (any goal, including 0)
- `!plan run` — full pipeline: decompose → auto-approve → execute
- These use `plan_engine.execute_pending_tasks()` directly, not the goal processor

### Manual Commands

| Command | Purpose |
|---|---|
| `!plan` | Show active plans (concept + task hierarchy) |
| `!plan all` | Show all plans including completed |
| `!plan <goal_id>` | Show plans for a specific goal |
| `!plan decompose [goal_id]` | Decompose pending concept steps into tasks |
| `!plan approve <concept_id>` | Approve proposed task steps under a concept |
| `!plan reject <concept_id>` | Reject proposed task steps |
| `!plan execute [goal_id]` | Execute approved task steps |
| `!plan run [goal_id]` | Full pipeline: decompose → auto-approve → execute |
| `!plan add <goal_id> <desc>` | Add a concept step to a goal |
| `!plan adhoc <desc>` | Create a concept step without a goal |
| `!plan auto` | Show goals with auto-process status |
| `!plan auto approve <goal_id>` | Approve goal for autonomous execution |
| `!plan auto reject <goal_id>` | Reject (never auto-process) |
| `!plan auto defer <goal_id>` | Defer until datetime (default +24h) |
| `!plan auto done <goal_id> <step_id>` | Mark user step done, resume execution |
| `!plan auto trigger` | Wake goal processor immediately |
| `!plan auto stats` | Show goal processor stats |

### Config

**Goal processor** (`plugins-enabled.json → plugin_config.goal_processor`):

| Key | Default | Purpose |
|---|---|---|
| `enabled` | false | Master switch |
| `interval_m` | 30 | Minutes between scans |
| `max_goals_per_cycle` | 3 | Max goals to propose per cycle |
| `max_exec_steps_per_cycle` | 10 | Max steps to execute per cycle |
| `defer_cooldown_hours` | 24 | Hours before re-proposing deferred goals |

---

## Belief System & Contradiction Detection

**`memory.py` | `contradiction.py`**

Beliefs are stored in `samaritan_beliefs` with a `confidence` score (1–10) and status `active`/`retracted`.

As of 2026-04-01, active contradiction scanning is handled by the `samaritan-cognition` Claude Code session (triggered on each `assert_belief` call) rather than the Python async task. The `contradiction.py` module remains available for direct invocation but is no longer registered as a background timer loop.

Contradiction check flow:
1. `assert_belief` is called → activity hook in `plugin_mcp_direct.py` queues a `claude-cognition` step
2. `samaritan-cognition` wakes, loads context, compares the new belief against existing beliefs
3. If conflict found: `assert_belief(topic="contradiction-<topic>", ...)` flags it
4. If no conflict: step marked done with a note

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

Drives modulate goal prioritization: `load_typed_context_block()` fetches current drive values and sorts active goals by `importance × drive_weight`, where `drive_weight = task-completion` for user goals and `max(task-completion, autonomy)` for assistant-initiated goals. Goals are annotated with their computed priority score (`pri=X.X`) before injection.

---

## Proactive Cognition Loops

As of 2026-04-01, the cognition architecture is a hybrid:

- **Event-driven (Claude Code)**: reflection, contradiction detection, goal health checks are handled by the `samaritan-cognition` Claude Code session. These are no longer Python async timer loops — they fire immediately when triggered by activity events. See the [Claude Code Cognition Session](#claude-code-cognition-session) section below.

- **Timer-driven (Python async)**: prospective reminders, temporal inference, the goal processor, and the cognitive feedback loop remain as Python background tasks. These do not involve LLM calls in their core loop (or their LLM use doesn't justify a full Claude Code session).

### Prospective Reminders — `prospective.py`

**Interval**: 5 minutes (configurable via `prospective_interval_m`)
**Model role**: `prospective`

Queries `samaritan_prospective` for rows with `due_at <= now()`. Fires due reminders into the next session context. Measures usage ratio (reminders acted on vs. fired) → feeds cognitive feedback loop.

### Goal Processor — `goal_processor.py`

**Interval**: 30 minutes (configurable via `goal_processor.interval_m`)
**Model role**: `plan_decomposer` (for decomposition)

Autonomous goal scanning, planning, and execution. See Goal Processor section above for full lifecycle.

### Temporal Inference — `temporal_inference.py`

**Interval**: 3 hours (configurable via `memory.temporal.inference_interval_h`)
**Model role**: `temporal_inference`

Analyzes recent ST topics and proposes temporal pattern queries. Stores results in `samaritan_temporal` with `source="inferred"`. Fills the gap that semantic retrieval (Qdrant) has no time dimension — "what do I usually do at 10 AM?" has zero semantic overlap with actual activities.

---

## Claude Code Cognition Session

**`plugin_mcp_direct.py` | `cognition-start.sh` | `.claude/rules/cognition-session.md`**

The `samaritan-cognition` session is a dedicated Claude Code process (Sonnet model) running in a tmux session. It sits idle at an interactive prompt and is woken by event hooks in `plugin_mcp_direct.py` when cognitive work is needed.

This replaces the Python async loops for reflection, contradiction detection, and goal health checks. The motivation: Claude performs substantially better reasoning on these tasks than the small models (Qwen, Gemini Flash) previously used in the Python loops, and the event-driven architecture eliminates the periodic LLM cost of timer-based loops that fire even when there is nothing new to process.

### Trigger Events

All triggers live in `plugin_mcp_direct.py` and fire as `asyncio.ensure_future()` calls immediately after the triggering MCP tool call completes.

| Trigger | Event | Step queued |
|---|---|---|
| `memory_save` called | Any save | `Check new memory for contradictions: [topic] content` |
| `assert_belief` called with `status="active"` | Belief created/updated | `Check belief for contradictions: [topic] content` |
| `goal_create` called | New goal created | `Review new goal for conflicts with active goals: [id] title` |
| `/conv_log` endpoint (every 5 turns) | Conversation activity | `Reflect on recent conversation (turn N): extract insights, update beliefs, check for new goals` |

The turn counter (`_cogn_turn_counter`) is a process-global incremented by every call to `endpoint_conv_log`. It persists across sessions but resets on llmem-gw restart.

### Step Routing

Steps are inserted into `samaritan_plans` with `target='claude-cognition'` under a singleton goal titled **"Ongoing Cognitive Processing"** (auto-created on first trigger, status=`active`). Steps have `approval='approved'` so they are immediately visible to `steps_for_cognition()`.

The `steps_for_cognition()` MCP tool queries:
```sql
SELECT ... FROM samaritan_plans
WHERE target = 'claude-cognition' AND status IN ('pending', 'in_progress')
ORDER BY goal_id, step_order
```

### Poke Mechanism

After every successful step insert, `_poke_cognition_session()` sends a wake signal:
```bash
tmux send-keys -t samaritan-cognition "Process pending cognition steps" Enter
```

Pokes are **debounced**: if one was sent within the last 60 seconds (`_COGN_POKE_COOLDOWN`), the new poke is suppressed. Steps still accumulate in the DB — the session picks up all of them on the next wake.

### Cognition Session Behavior

On receiving "Process pending cognition steps", the session (`cognition-session.md` rules):

1. Calls `steps_for_cognition()` — if empty, responds "No pending cognition steps." and stops
2. For each step: marks `in_progress`, executes, marks `done` with result
3. Processing all steps before finishing (no partial processing)

**Step type handlers**:

- **Contradiction check** (`Check new memory for contradictions` / `Check belief for contradictions`):
  1. `load_context(query="<topic>")` to pull existing beliefs and memories
  2. Compare new item against existing — look for direct conflicts, subtle tension, or outdated facts
  3. If conflict: `assert_belief(topic="contradiction-<topic>", content="<description>", confidence=8)`
  4. Mark step done with finding

- **Reflection** (`Reflect on recent conversation`):
  1. `memory_recall(topic="", tier="short", limit=10)` for recent context
  2. Identify patterns, repeated themes, insights worth preserving
  3. For significant findings: `assert_belief(topic="<topic>", content="<insight>", confidence=7)`
  4. Mark step done with summary

- **Goal health check** (`Review new goal for conflicts`):
  1. `goal_list(status="active")` to get all active goals
  2. Check for conflicts, duplicates, or blocking relationships
  3. If conflict: `memory_save(topic="goal-conflict", content="<description>", importance=7)`
  4. Mark step done with assessment

### Session Lifecycle

| Event | Behavior |
|---|---|
| `claude-start.sh` | Starts `samaritan-work`, then calls `cognition-start.sh` |
| `restart-llmem.sh` | `samaritan-cognition` in `MANAGED_SESSIONS` — killed + cold-started after server comes up |
| Claude process exits | Next `cognition-start.sh` run detects dead Claude, cold-starts |
| `.post-restart-instructions.samaritan-cognition` | Delivered by `_deliver_instructions()` after cold-start |

### Prerequisites

- llmem-gw running with `plugin_mcp_direct` enabled (port 8769)
- Claude Code CLI installed (`claude` in PATH)
- tmux available
- `samaritan_plans.target` ENUM includes `'claude-cognition'` (migration required — see Setup)
- MCP Direct server registered with Claude Code (`mcp-direct-enable.sh`)
- `cognition-session.md` rules file in `.claude/rules/`

### Setup Steps

1. **Apply the DB schema change** — add `claude-cognition` to the target ENUM:
   ```sql
   ALTER TABLE mymcp.samaritan_plans
     MODIFY COLUMN target
     ENUM('model','human','investigate','claude-code','claude-cognition')
     NOT NULL DEFAULT 'model';
   ```

2. **Copy `cognition-start.sh`** to your workspace root and make it executable:
   ```bash
   chmod +x cognition-start.sh
   ```

3. **Copy `.claude/rules/cognition-session.md`** to your Claude Code workspace rules directory.

4. **Update `claude-start.sh`** to call `cognition-start.sh` at the end:
   ```bash
   bash "$SCRIPT_DIR/cognition-start.sh"
   ```

5. **Add `samaritan-cognition` to `MANAGED_SESSIONS`** in `restart-llmem.sh`:
   ```bash
   "samaritan-cognition|${WORK_DIR}|--model claude-sonnet-4-6 -n 'samaritan-cognition'|"
   ```

6. **Start the session**:
   ```bash
   bash cognition-start.sh
   ```

7. **Smoke test**:
   - Call `steps_for_cognition()` — should return no rows
   - Call `memory_save` with any content
   - Call `steps_for_cognition()` again — should show a pending contradiction-check step
   - Check the tmux session: `tmux attach -t samaritan-cognition` — should show the session processing

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
  ├─ prospective due items                                │
  └─ temporal patterns      ← recall_temporal cache       │
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
      │                       │                           │
      │      ┌────────────────┴─────────────────┐        │
      │      │  activity hooks (plugin_mcp_direct)│        │
      │      │  memory_save   → contradiction    │        │
      │      │  assert_belief → contradiction    │        │
      │      │  goal_create   → goal health      │        │
      │      │  conv_log (/5) → reflection       │        │
      │      └────────────────┬─────────────────┘        │
      │                       │                           │
      │              samaritan_plans (target=claude-cog.) │
      │              + tmux poke (debounced 60s)          │
      │                       │                           │
      │              samaritan-cognition (Claude/Sonnet)  │
      │              steps_for_cognition()                │
      │              → reflect / contradict / goal-health │
      │                via MCP tools                      │
      │                                                   │
Background loops (Python async, independent):             │
  prospective.py   (5m)    → fire due reminders           │
  goal_processor   (30m)   → scan → propose → execute     │
  temporal_inf.    (3h)    → infer temporal patterns      │
  cogn_feedback.py         → update conditioned strengths │
  aging loops      (60s/6h) → ST→LT summarization         │
```

---

## Cognition Table

**`samaritan_cognition` | migration `009_cognition.sql`**

Cognitive loop outputs (contradiction flags, reflection insights, goal health decisions) are written to a dedicated table rather than polluting ST/LT. Each row has an `origin` enum identifying the source loop.

This separates cognitive meta-observations from conversational memory, preventing loops from retrieving their own prior outputs during semantic search and creating feedback spirals.

---

## Temporal Pattern Recall

**`tools.py` — `recall_temporal` | `temporal_inference.py` | migration `007_temporal.sql`**

Semantic retrieval (Qdrant) has no time dimension. The `recall_temporal` tool queries `created_at` timestamps across ST and LT to surface recurring time-based patterns.

Parameters: `query`, `group_by` (hour/day_of_week/date/week/month), `day_of_week`, `time_range` (morning/afternoon/evening/now/HH:MM-HH:MM), `lookback_days`, `limit`, `new` (bypass cache).

Results cached in `samaritan_temporal`. Aging: HWM 500 / LWM 300, timer every 6h, deletes lowest-hit oldest first.

---

## Model Roles

**`llm-models.json` → `model_roles`**

All cognitive model assignments are centralized in the `model_roles` section of `llm-models.json`. This separates model assignment (which model does what) from operational configuration (intervals, thresholds, limits) which stays in `plugins-enabled.json`.

| Role | Model key | Used by |
|---|---|---|
| `summarizer` | `summarizer-anthropic` | Memory aging, `!reset` summarization |
| `reviewer` | `reviewer-gemini` | Memory review (`!memreview`) |
| `extractor` | `extractor-gemini` | Content extraction |
| `judge` | `judge-gemini` | LLM-as-judge gates |
| `plan_decomposer` | `plan-decomposer` | Goal → concept → task decomposition |
| `plan_executor` | `samaritan-execution` | Plan step execution (backup via `backup_models` → `gpt-4o-execution`) |
| `contradiction` | *(samaritan-cognition session)* | Contradiction scanning — moved to Claude Code |
| `prospective` | `qwen25-cogn` | Prospective reminder evaluation |
| `reflection` | *(samaritan-cognition session)* | Reflection loop — moved to Claude Code |
| `goal_health` | *(samaritan-cognition session)* | Goal health pass — moved to Claude Code |
| `temporal_inference` | `summarizer-gemini` | Temporal pattern inference |

**Resolution order**: config override (plugins-enabled.json, if present) → `get_model_role()` (llm-models.json) → hardcoded fallback. Runtime overrides via `!cogn model <loop> <key>` are also supported.

---

## What Was Planned vs. What Was Built

The original plan proposed seven missing layers. Here is the actual outcome:

| Planned | Outcome |
|---|---|
| Goal stack via type field + goal_manager.py | Built: typed goals table, tools, drive coupling, auto-injection into context, drive-weighted sort, reflection-based completion detection |
| World model / beliefs via loose triple store | Built as typed beliefs + full contradiction scanner — more complete than planned |
| Proactive loops via aging timer pattern | Built: reflection + contradiction + prospective, all with feedback learning. Reflection and contradiction later migrated to Claude Code session (event-driven) |
| Surprise scoring at save time via Qdrant novelty check | **Not built as planned.** Replaced by retroactive access-ratio measurement — a stronger signal (actual usefulness vs. guessed novelty) |
| Self-model via protected topic namespace | Built exactly as planned, plus `refresh_self_summary()` synthesis loop |
| Drives via state.py session state | Built as persistent DB table with decay and goal-coupling, more durable than planned |
| Procedural memory as second Qdrant collection | Built exactly as planned |

The notable design divergence: **surprise scoring** was replaced by the cognitive feedback loop's access-ratio system. Instead of estimating novelty at write time (which requires a speculative Qdrant round-trip for every save), the system measures actual impact retroactively. A memory nobody ever retrieves is more actionable evidence than a save-time novelty estimate.

The second significant divergence: **Python async cognitive loops replaced by Claude Code session**. The Python loops used small models (Qwen, Gemini Flash) on timers. The Claude Code session uses Sonnet, fires immediately on relevant events, and eliminates timer-based LLM costs when there is no new activity.

---

## Completed Items

### 2026-03-11

1. **Goal injection into context** ✓ — `load_typed_context_block()` fetches active goals and injects as `## Active Goals`, sorted by drive-weighted priority.
2. **Drive-weighted goal prioritization** ✓ — `importance × drive_weight` per goal, annotated with `pri=X.X`.
3. **Prospective loop full implementation** ✓ — Fire/feedback cycle complete with usage ratio measurement.
4. **Goal completion detection** ✓ — Reflection LLM returns `goals_done` IDs, validated before marking done.

### 2026-03-12

5. **Plan engine two-tier decomposition** ✓ — Concept → task steps with approval, ownership, and tool_call specs. Two-tier execution chain (direct → LLM executor with backup_models failover).
6. **Cognition table** ✓ — `samaritan_cognition` separates cognitive outputs from ST/LT. Migration `009`.
7. **Temporal pattern recall** ✓ — `recall_temporal` tool + `temporal_inference.py` background inference + aging.

### 2026-03-13

8. **Goal processor** ✓ — Autonomous goal scanning, proposing, and serial execution. State machine: NULL → proposed → approved → executing → completed/paused_user/deferred/rejected. Notifier integration for proposals and human steps.
9. **Model roles centralization** ✓ — All cognitive model assignments moved from `plugins-enabled.json` to `model_roles` in `llm-models.json`. Resolution order: config override → `get_model_role()` → hardcoded fallback.
10. **Plan decomposer upgrade** ✓ — Upgraded from Haiku 4.5 to Sonnet 4.6 to eliminate model-name hallucinations in `tool_call` specs.

### 2026-04-01

11. **Claude Code cognition session** ✓ — `samaritan-cognition` tmux session replaces Python async loops for reflection, contradiction, and goal health. Event-driven: hooks in `plugin_mcp_direct.py` queue steps on `memory_save`, `assert_belief`, `goal_create`, and every 5 `conv_log` turns. Session woken via debounced tmux poke. Added `claude-cognition` target to `samaritan_plans` ENUM.

## Open Items

None at this time.
