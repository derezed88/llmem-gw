# Task Management: Goals, Plans, and Execution

Complete lifecycle documentation for the three-layer task management system:
**Goal Setters** (human / reasoning model / cognition) → **Planner** (Sonnet 4.6) → **Executor** (two-tier: direct → primary LLM → fallback LLM).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GOAL SETTERS                            │
│  Human (!goal via tool)  │  Reasoning model (set_goal tool) │
│  Cognition reflection    │  Goal health proposals           │
└──────────────┬──────────────────────────┬───────────────────┘
               │ creates                  │ creates
               ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    samaritan_goals                           │
│  id, title, description, status, importance, source,        │
│  session_id, childof, parentof, memory_link,                │
│  attempt_count, failure_count, abandon_reason               │
│  status: active | done | blocked | abandoned                │
└──────────────────────────┬──────────────────────────────────┘
                           │ has plan steps
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    samaritan_plans                           │
│  CONCEPT STEPS (step_type='concept', parent_id=NULL)        │
│  Natural-language human-readable intent                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  TASK STEPS (step_type='task', parent_id→concept)      │ │
│  │  Executable atoms with tool_call JSON specs            │ │
│  │  target: model | human | investigate                   │ │
│  │  approval: proposed | approved | rejected | auto       │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ approved + target=model
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    TWO-TIER EXECUTOR                          │
│  plan_engine.execute_task_step()                             │
│                                                              │
│  Tier 1: Direct tool call (zero LLM cost)                   │
│    get_tool_executor() → executor_fn(**mapped_args)          │
│         ↓ on failure                                         │
│  Tier 2: LLM executor (model_roles["plan_executor"])         │
│    llm_call(mode="tool") → samaritan-execution (DeepSeek-V3.2)│
│    Provider failover via backup_models → gpt-4o-execution    │
│                                                              │
│  executor column = "direct" | "<model_key>" (audit trail)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Goal Formation

Goals enter the system from four sources. All write to `samaritan_goals`.

### 1.1 Human-Created Goals (via model tool call)

A human user tells the interactive model what they want. The model calls the `set_goal` tool.

**Flow**: User message → samaritan-voice/reasoning interprets → `set_goal(title, description, importance)` tool call → `tools.py:_set_goal_exec()` → INSERT into `samaritan_goals`.

```
User: "I want to investigate whether our Qdrant retrieval is returning good results"

Model calls set_goal:
  title = "Investigate memory retrieval quality"
  description = "Determine if Qdrant retrieval is returning relevant results for topic queries"
  importance = 9
  → INSERT → id=501, status='active', source='assistant', session_id=<client_id>
```

**Source field**: `source='assistant'` (the model created the DB row on behalf of the user).

### 1.2 Model-Created Goals (reasoning model via tool)

The reasoning model (`samaritan-reasoning`, grok-4-1-fast-reasoning) can autonomously create goals when it identifies work to be done during conversation.

Same tool path as human goals, but `session_id` reflects the model's active session.

### 1.3 Cognition-Created Goals (reflection loop)

The reflection subsystem (`reflection.py`) runs periodically and can create goals in two ways:

**a) Goal health proposals** — During `_run_goal_health()`, the reasoning model evaluates active goals and proposes new ones based on failure patterns, curiosity threads, or remediation needs.

```python
# reflection.py:_run_goal_health()
# Proposal flow:
#   1. Reasoning model returns {"proposals": [{"title": ..., "description": ..., "importance": ...}]}
#   2. Abandon guard: keyword overlap check against abandoned goals (≥50% → blocked)
#   3. Active duplicate guard: title similarity check
#   4. Autonomy drive gate:
#        autonomy >= threshold (0.6) → auto-create, source='assistant', session_id='reflection'
#        autonomy < threshold       → create with session_id='reflection-proposed' for review
```

**b) Goal completion detection** — Reflection detects when active goals are achieved in conversation and marks them `status='done'`. Two-pass detection:
  1. Memory extraction model (summarizer-gemini) flags completions during insight extraction
  2. Dedicated goal-scan pass (samaritan-reasoning) with stricter "evidence of completion" criteria
  3. Union of both passes applied

### 1.4 Guards on Goal Creation

`_set_goal_exec()` in `tools.py` enforces two guards before INSERT:

| Guard | Check | Result |
|-------|-------|--------|
| **Keyword blocker** | Title/description contains timer, cron, swarm, DDL, schema migration, create/alter table | `set_goal BLOCKED` — prevents goals requiring unavailable infrastructure |
| **Abandon guard** | ≥50% keyword overlap with any `status='abandoned'` goal | `set_goal BLOCKED` — prevents retrying previously abandoned goals |

---

## 2. Plan Steps — The Two-Tier Model

Plans use a **self-referencing hierarchy** within `samaritan_plans`:

| Field | Concept Step | Task Step |
|-------|-------------|-----------|
| `step_type` | `'concept'` | `'task'` |
| `parent_id` | `NULL` | → concept step id |
| `goal_id` | → goal id (0 for ad-hoc) | → same goal id as parent |
| `target` | hint for decomposer | `'model'`, `'human'`, or `'investigate'` |
| `approval` | `'proposed'` | `'proposed'` (needs review) or `'auto'` |
| `tool_call` | `NULL` | JSON: `{"tool": "db_query", "args": {"sql": "SELECT..."}}` |
| `executor` | `NULL` | Written at execution: `'direct'` or `'<model_key>'` (e.g. `'samaritan-execution'`) |
| `result` | completion/failure notes | execution output text |

### 2.1 Concept Steps

Concept steps preserve **natural-language human intent**. They are written by humans or reasoning models and describe *what* needs to happen without specifying *how*.

**Creation paths**:
- `set_plan` tool with `step_type='concept'` (model-initiated)
- `!plan add <goal_id> <description>` command (human-initiated)
- `!plan adhoc <description>` command (no goal, goal_id=0)
- `plan_engine.create_concept_step()` (programmatic)

Example concept steps for a goal "Investigate memory retrieval quality":
```
step 1: Query Qdrant for a known topic and inspect the top-5 results for relevance
step 2: Compare Qdrant results with direct MySQL keyword search on same topic
step 3: Review the comparison and decide if embedding model needs retraining (→human)
step 4: If retraining needed, research nomic-embed-text alternatives (→investigate)
```

### 2.2 Task Steps (Decomposed from Concepts)

Task steps are **executable atoms** — each maps to exactly one tool call or one human action. They are created by the decomposition engine, never by hand.

Example task steps decomposed from concept "Compare Qdrant results with direct MySQL keyword search":
```
[266] tool:db_query  target=model       — Execute MySQL keyword search
[267] tool:llm_call  target=model       — Compare Qdrant and MySQL results
[268] tool:memory_save target=model     — Save comparison findings to memory
```

### 2.3 Ad-Hoc Plans (goal_id=0)

Plans can exist without a goal. `goal_id=0` means the plan is standalone — useful for one-off tasks that don't warrant a formal goal.

```
!plan adhoc Check if the Qdrant server is responding on port 6333
→ Creates concept step with goal_id=0
→ Can be decomposed and executed like any other plan
```

---

## 3. Decomposition: Concept → Task Steps

### 3.1 The Decomposer Model

| Setting | Value |
|---------|-------|
| Model key | `plan-decomposer` |
| Model ID | `claude-haiku-4-5-20251001` |
| Temperature | 0.2 |
| Cost | $1 / $5 per 1M tokens (input/output) |
| Tools | none (pure text/JSON generation) |
| Memory | disabled (`conv_log: false`, `memory_scan: false`) |

**Why Haiku 4.5**: Structured decomposition requires instruction-following and JSON generation, not deep reasoning. Haiku is 3x cheaper than Sonnet ($1/$5 vs $3/$15 per 1M tokens) with sufficient quality for this task.

### 3.2 Decomposition Flow

```
plan_engine.decompose_concept_step(concept_step_id)
  │
  ├─ 1. Load concept step + goal context from DB
  │
  ├─ 2. Load sibling concept steps (for sequencing awareness)
  │
  ├─ 3. Build tool catalog via _build_tool_catalog()
  │     Introspects get_all_openai_tools() → name(params): description
  │
  ├─ 4. Construct prompt:
  │     SYSTEM: _DECOMPOSER_SYSTEM (rules for JSON output, target types)
  │     USER: concept description + goal context + sibling steps + tool catalog
  │
  ├─ 5. Call plan-decomposer LLM → JSON array of task specs
  │
  ├─ 6. Parse response, validate targets, insert task rows:
  │     - target=model requires tool_call (downgraded to investigate if missing)
  │     - target=human/investigate may omit tool_call
  │     - reason field appended to description for human/investigate targets
  │     - executor set to 'plan-executor' for model targets
  │
  └─ 7. Mark concept step status='in_progress'
```

### 3.3 Decomposer System Prompt

The decomposer receives structured instructions including:

- **One atomic action per task step** — one tool call or one human action
- **Prefer existing tools** — do not invent tool names
- **Target assignment rules**:
  - `model`: tool_call is populated and can auto-execute
  - `human`: explicitly references a person or requires human judgment
  - `investigate`: cannot determine execution method with current tools
- **JSON-only response** — no markdown, no explanation

### 3.4 Tool Catalog

The decomposer sees a compact listing of all registered tools:

```
- db_query(sql): Execute a read-only SQL query against the active database
- memory_save(topic, content, importance): Save a short-term memory
- search_tavily(query): Search the web via Tavily
- llm_call(mode, model, prompt, tool): Delegate work to another model
- url_extract(url): Extract content from a URL
...
```

This is built dynamically from `get_all_openai_tools()` — new plugins that register tools are automatically included.

### 3.5 1→N Decomposition

A single concept step typically expands to 2-5 task steps. Complex concepts can generate more:

| Concept | Task Steps | Targets |
|---------|-----------|---------|
| "Count rows in shortterm table" | 1 | 1 model |
| "Check Qdrant server health" | 2 | 2 model |
| "Research embedding alternatives" | 3-4 | 2 model, 1 investigate, 1 human |
| Full 4-concept investigation plan | 14 total | 8 model, 4 investigate, 2 human |

---

## 4. Approval Workflow

Task steps are created with `approval='proposed'` by default. They must be approved before execution.

### 4.1 Manual Approval

```
!plan approve <concept_step_id>    → approves ALL proposed task steps under that concept
!plan reject <concept_step_id>     → rejects ALL proposed task steps under that concept
```

Approval/rejection operates on the concept level — all child task steps are updated together.

### 4.2 Auto-Approval

The `!plan run` command and `run_plan_pipeline()` pass `auto_approve=True` to decomposition, which sets `approval='auto'` on created task steps, bypassing the review gate.

### 4.3 Approval States

| State | Meaning |
|-------|---------|
| `proposed` | Created by decomposer, awaiting review |
| `approved` | Human/system approved for execution |
| `rejected` | Human rejected — will not execute |
| `auto` | Auto-approved (system-generated, no gate) |

---

## 5. Execution — Two-Tier Executor with Failover

Execution responsibility lives in the **plan engine system**, not in the reasoning model. This ensures deterministic routing, cost control, and automatic failover between executor models.

### 5.1 Executor Model Roles

Configured in `llm-models.json` under `model_roles`:

```json
"model_roles": {
    "plan_executor": "samaritan-execution"
}
```

| Role | Model Key | Model ID | Cost (in/out per 1M) | Purpose |
|------|-----------|----------|----------------------|---------|
| Primary | `samaritan-execution` | DeepSeek-V3.2 (FriendliAI) | $0.50 / $1.50 | Reliable tool caller |
| Backup | `gpt-4o-execution` | gpt-4o-mini (OpenAI) | $0.15 / $0.60 | Provider failover via `backup_models` |

Provider failover is handled by `backup_models` on the `samaritan-execution` model config in `llm-models.json` — not a separate role. `llm_call()` in `agents.py` automatically retries with `backup_models[0]` on timeout or exception.

Roles are read at execution time via `config.get_model_role()` — changing the model in the JSON takes effect immediately without code changes.

### 5.2 Two-Tier Execution Flow

```
plan_engine.execute_task_step(task_step_id)
  │
  ├─ 1. Validate: approval='approved', target='model', not already done
  │
  ├─ 2. Mark status='in_progress'
  │
  ├─ 3. Parse tool_call JSON → {"tool": "db_query", "args": {"sql": "..."}}
  │
  ├─ 4. TIER 1: Direct tool call (zero LLM cost)
  │     get_tool_executor(tool_name) → executor_fn
  │     _map_tool_args() → synonym mapping (query↔sql, text↔content, etc.)
  │     await executor_fn(**mapped_args)
  │     ├─ SUCCESS → executor='direct', status='done' → cascade check
  │     └─ FAILURE → log warning, fall through to Tier 2
  │
  ├─ 5. TIER 2: LLM executor (plan_executor role)
  │     _try_llm_executors() → _llm_execute_tool()
  │     llm_call(model=samaritan-execution, mode='tool', tool=tool_name)
  │     ├─ SUCCESS → executor='samaritan-execution', status='done' → cascade
  │     ├─ FAILURE → backup_models failover in llm_call → gpt-4o-execution
  │     └─ ALL FAILED → return None
  │
  └─ 6. ALL FAILED → _fail_task() → propagate up
         error = "All executors failed. Direct: <err>"
```

### 5.3 LLM Executor Delegation

When direct execution fails, `_llm_execute_tool()` delegates via `agents.llm_call(mode='tool')`:

```python
# Prompt sent to executor LLM
prompt = (
    f"Execute this tool call:\n"
    f"Tool: {tool_name}\n"
    f"Arguments: {args_json}\n"
    f"Context: {step_description}\n"
    f"Note: direct execution failed with: {direct_err}\n\n"
    f"Call the {tool_name} tool with the arguments above and return the result."
)

result = await llm_call(
    model=model_key,
    prompt=prompt,
    mode="tool",          # bind the tool, force tool call
    sys_prompt="target",  # use executor model's own system prompt
    history="none",       # clean single-turn call
    tool=tool_name,
)
```

The LLM executor sees the tool's OpenAI schema, understands the correct parameter names, and can fix mismatches that caused the direct call to fail (e.g., `search_query` → `query`).

### 5.4 Executor Audit Trail

The `executor` column records which tier actually executed the step:

| Value | Meaning |
|-------|---------|
| `'direct'` | Tier 1 — Python function called directly, zero LLM cost |
| `'samaritan-execution'` | Tier 2 — DeepSeek-V3.2 (FriendliAI) executed via llm_call |
| `'gpt-4o-execution'` | Tier 2 backup — gpt-4o-mini (OpenAI) via backup_models failover |

This enables cost attribution and reliability analysis — query `SELECT executor, COUNT(*) FROM samaritan_plans WHERE step_type='task' AND status='done' GROUP BY executor` to see execution distribution.

### 5.5 Argument Mapping (Tier 1)

The decomposer sometimes generates slightly different parameter names than the actual function signatures. The direct executor handles this with introspection + a synonym dictionary:

```python
_synonyms = {
    "query": "sql",    "sql": "query",
    "text": "content", "content": "text",
    "message": "prompt", "prompt": "message",
}
```

Example: decomposer generates `{"tool": "db_query", "args": {"query": "SELECT..."}}` but `db_query()` expects `sql` parameter. The mapper resolves `query` → `sql` automatically.

If synonym mapping fails and the direct call raises, the LLM executor tiers handle it — they see the full tool schema and can map arguments correctly.

### 5.6 Execution Targets

| Target | Behavior |
|--------|----------|
| `model` | Auto-executable via two-tier executor. |
| `human` | Requires human action. Cannot auto-execute. Shown in plan view for manual completion. |
| `investigate` | Needs further analysis. Decomposer couldn't resolve with current tools. |

Only `target='model'` steps are picked up by `execute_pending_tasks()`.

### 5.7 Batch Execution

```python
execute_pending_tasks(goal_id=None, max_steps=10)
```

Finds all pending + approved + model-targeted task steps, ordered by goal_id and step_order. Executes up to `max_steps` sequentially. Each step goes through the full two-tier execution path.

---

## 6. Completion Propagation (Bottom-Up)

When a task step completes, the system checks whether the parent concept and goal are also complete.

```
Task step → done
  │
  ├─ _check_parent_completion(parent_id)
  │   └─ All tasks under concept done/skipped?
  │       YES → concept step → done
  │             │
  │             └─ _check_goal_completion(goal_id)
  │                 └─ All concept steps for goal done/skipped?
  │                     YES → goal → done
  │                     NO  → (wait for remaining concepts)
  │
  │       NO → (wait for remaining tasks)
  │
  └─ (done)
```

### Cascade Example

```
Goal 499: "Exec Test"                    status=active
  └─ Concept 252: "Query the database"   status=in_progress
       └─ Task 253: "Run test query"     status=pending → done

After task 253 completes:
  1. _check_parent_completion(252): all tasks done → concept 252 → done
  2. _check_goal_completion(499): all concepts done → goal 499 → done
```

---

## 7. Failure Propagation (Bottom-Up)

When a task step fails, the failure propagates up to block the parent concept and goal.

```
Task step → execution fails
  │
  ├─ _fail_task(task_step_id, error)
  │   └─ task status='pending', result=error message
  │       │
  │       └─ _check_parent_failure(parent_id, error)
  │           └─ concept step: result += " | BLOCKED: <error>"
  │               │
  │               └─ goal: status='blocked' (if was 'active')
  │
  └─ Error returned to caller
```

### Failure Semantics

| Level | On Failure |
|-------|-----------|
| **Task step** | `status='pending'` (not 'failed' — allows retry), `result` stores the error |
| **Concept step** | `result` appended with `BLOCKED: <error>` context. Status stays pending/in_progress. |
| **Goal** | `status='blocked'` (only if was 'active' — won't downgrade already-blocked) |

### Why `pending` not `failed`?

Task steps revert to `pending` on failure rather than a terminal `failed` state. This allows:
- Re-execution after the root cause is fixed
- The decomposer or human to revise the approach
- `!plan execute` to pick them up again after approval

### Goal Health Escalation (reflection.py)

When the reflection loop runs, `_run_goal_health()` performs additional failure analysis:

| Condition | Action |
|-----------|--------|
| `failure_count >= 3` (replan threshold) | Reasoning model proposes new plan steps; `self-failure-replan-<id>` memory saved |
| `failure_count >= 5` (abandon threshold) | Goal auto-abandoned with reason; `self-failure-abandoned-goal-<id>` lesson learned saved |

Failure counting uses keyword overlap between `self-failure-*` memory rows and goal title/description.

---

## 8. Commands Reference

### Goal Management

Goals are managed via the `set_goal` tool (called by models) or by the cognition subsystem. There is no `!goal` command — goals are viewed through `!plan` (which shows goals + plans together) and `!cogn goals` (goal health dashboard).

| Tool Call | Effect |
|-----------|--------|
| `set_goal(title, description, importance)` | Create new goal |
| `set_goal(id=N, status='done')` | Mark goal complete |
| `set_goal(id=N, status='blocked')` | Block a goal |
| `set_goal(id=N, status='abandoned')` | Abandon a goal |

### Plan Management

| Command | Effect |
|---------|--------|
| `!plan` | Show active plan hierarchy (concepts + tasks) |
| `!plan all` | Include completed/skipped steps |
| `!plan <goal_id>` | Plans for a specific goal |
| `!plan decompose [goal_id]` | Decompose pending concepts → proposed task steps |
| `!plan approve <concept_id>` | Approve all proposed tasks under a concept |
| `!plan reject <concept_id>` | Reject all proposed tasks under a concept |
| `!plan execute [goal_id]` | Execute approved model-targeted tasks |
| `!plan run [goal_id]` | Full pipeline: decompose → auto-approve → execute |
| `!plan add <goal_id> <desc>` | Add concept step to existing goal |
| `!plan adhoc <desc>` | Create standalone concept step (goal_id=0) |

### Cognition Integration

| Command | Effect |
|---------|--------|
| `!cogn goals` | Goal health dashboard — shows active goals with failure counts |
| `!cogn goals approve <id>` | Approve a cognition-proposed goal |
| `!cogn goals reject <id>` | Reject a cognition-proposed goal |

---

## 9. Context Injection

Active goals and plans are injected into the model's prompt via `memory.py:load_typed_context_block()`:

```
## Active Goals

  [id=501 imp=9 pri=6.3] Investigate memory retrieval quality — Determine if...

## Active Plans

  **Investigate memory retrieval quality**
    ○ [id=259 step=1] Query Qdrant for a known topic... [proposed]
        · [263] Query Qdrant vector database... →investigate [proposed]
        · [264] Retrieve and inspect top-5 results... →investigate [proposed]
    ○ [id=260 step=2] Compare Qdrant results with MySQL...
        · [266] Execute MySQL keyword search tool:db_query [proposed]
```

This gives the active model awareness of current goals and plan progress without requiring it to query the database.

### Priority Scoring

Goals are sorted by a drive-weighted priority score:

```python
priority = importance × drive_weight
# drive_weight = task-completion drive (default 0.7)
# For assistant-sourced goals: max(task-completion, autonomy)
```

---

## 10. Database Schema

### samaritan_goals

```sql
CREATE TABLE samaritan_goals (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    title         VARCHAR(255) NOT NULL,
    description   TEXT NOT NULL,
    status        ENUM('active','done','blocked','abandoned') DEFAULT 'active',
    importance    TINYINT DEFAULT 9,
    source        ENUM('session','user','directive','assistant') DEFAULT 'user',
    session_id    VARCHAR(255),
    childof       TEXT,           -- JSON array of parent goal IDs
    parentof      TEXT,           -- JSON array of child goal IDs
    memory_link   TEXT,           -- JSON array of memory row IDs
    attempt_count INT DEFAULT 0,  -- incremented by reflection (005)
    failure_count INT DEFAULT 0,  -- self-failure-* pattern count (005)
    abandon_reason TEXT,          -- set on auto-abandon (005)
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### samaritan_plans (with 006 decomposition columns)

```sql
CREATE TABLE samaritan_plans (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    goal_id       INT NOT NULL,
    step_order    INT DEFAULT 1,
    description   TEXT NOT NULL,
    status        ENUM('pending','in_progress','done','skipped') DEFAULT 'pending',
    step_type     ENUM('concept','task') DEFAULT 'concept',     -- 006
    parent_id     INT DEFAULT NULL,                              -- 006: NULL=concept, non-NULL→concept
    target        ENUM('model','human','investigate') DEFAULT 'model', -- 006
    executor      VARCHAR(64),                                   -- 006: 'direct', model_key, or NULL
    tool_call     TEXT,                                          -- 006: JSON {"tool":"..","args":{}}
    result        TEXT,                                          -- 006: execution output
    approval      ENUM('proposed','approved','rejected','auto') DEFAULT 'proposed', -- 006
    source        ENUM('session','user','directive','assistant') DEFAULT 'assistant',
    session_id    VARCHAR(255),
    memory_link   TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_goal (goal_id),
    KEY idx_goal_order (goal_id, step_order),
    KEY idx_parent (parent_id),                                  -- 006
    KEY idx_exec_queue (step_type, status, approval, target)     -- 006
);
```

---

## 11. Full Pipeline Example

End-to-end walkthrough: human creates a goal, system decomposes and executes.

### Step 1: Goal Creation

```
User: "Count the rows in the shortterm memory table"
Model: calls set_goal(title="Count ST rows", description="...", importance=7)
→ Goal id=500 created, status=active
```

### Step 2: Add Concept Step

```
!plan add 500 Count the number of rows in samaritan_memory_shortterm
→ Concept step id=257 created, step_order=1, status=pending, approval=proposed
```

### Step 3: Full Pipeline Run

```
!plan run 500
```

This triggers `run_plan_pipeline(goal_id=500, auto_approve=True)`:

**a) Decompose**: Concept 257 sent to Haiku 4.5 with tool catalog context.

Haiku returns:
```json
[{
  "description": "Count rows in samaritan_memory_shortterm",
  "tool_call": {"tool": "db_query", "args": {"query": "SELECT COUNT(*) as row_count FROM samaritan_memory_shortterm"}},
  "target": "model"
}]
```

→ Task step 258 created, approval='auto' (auto-approve mode)

**b) Execute**: Task 258 enters two-tier execution:
- **Tier 1 (direct)**: `get_tool_executor('db_query')` resolves function, `_map_tool_args()` maps `query` → `sql`, direct call succeeds.
- Executor column written as `'direct'`.

Result: `row_count\n---------\n80`

**c) Auto-complete cascade**:
- Task 258 → done (executor=direct)
- `_check_parent_completion(257)` → all tasks done → concept 257 → done
- `_check_goal_completion(500)` → all concepts done → goal 500 → done

### Final State

```
Goal 500: "Count ST rows"           → done ✓
  Concept 257: "Count the rows..."  → done ✓
    Task 258: "Count rows in..."    → done ✓  executor=direct  result="row_count\n80"
```

---

## 12. Realistic Decomposition Example

A human creates a 4-step investigation plan. The decomposer expands it to 14 task steps.

### Goal: "Investigate memory retrieval quality" (importance=9)

**Concept steps (human-written)**:
1. Query Qdrant for a known topic and inspect the top-5 results for relevance
2. Compare Qdrant results with direct MySQL keyword search on same topic
3. Review the comparison and decide if embedding model needs retraining (→human)
4. If retraining needed, research nomic-embed-text alternatives (→investigate)

**Decomposition results**:

| Concept | Tasks | model | human | investigate |
|---------|-------|-------|-------|-------------|
| 1. Query Qdrant | 3 | 0 | 1 | 2 |
| 2. Compare results | 3 | 2 | 0 | 1 |
| 3. Review & decide | 3 | 2 | 0 | 1 |
| 4. Research alternatives | 5 | 3 | 1 | 1 |
| **Total** | **14** | **8** | **2** | **4** |

Notable decomposition behaviors:
- **Concept 1** (Qdrant query): decomposer recognized no direct Qdrant tool exists → assigned `investigate` target with reason "Need to determine if Qdrant can be accessed via tmux command-line"
- **Concept 3** (human review): decomposer correctly assigned `human` target for the judgment step
- **Concept 4** (research): mixed targets — `search_tavily` for web research (model), `url_extract` for deep reading (investigate — may need human to select URLs), final recommendation step (human)

---

## 13. Testing

### Test Suite: `test_plan_engine.py`

8 test groups covering the full lifecycle. All tests:
- Create isolated test data directly in MySQL
- Run operations via the API (`/submit` with `wait=true`)
- Verify DB state after each operation
- Clean up all test data on completion

| Group | Name | What It Tests |
|-------|------|---------------|
| G1 | Migration Check | All 7 migration-006 columns exist in samaritan_plans |
| G2 | Concept Step CRUD | Creating concept steps via `set_plan` tool with step_type, target, approval |
| G3 | Decomposition | Haiku 4.5 decomposes 3 concept steps → task steps with tool_call specs |
| G4 | Approval Workflow | `!plan approve` sets all child task steps to approved |
| G5 | Execution & Auto-Completion | Task executes → concept auto-completes → goal auto-completes |
| G6 | Ad-Hoc Plans | `!plan adhoc` creates goal_id=0 concept, decomposes successfully |
| G7 | Full Pipeline | `!plan run` does decompose → auto-approve → execute → cascade complete |
| G8 | Realistic Scenarios | 4 human-written concepts → 14 task steps with mixed targets |

### Running Tests

```bash
source venv/bin/activate
python test_plan_engine.py                    # all groups
python test_plan_engine.py -g G3,G5           # specific groups
python test_plan_engine.py --stop-on-fail     # halt on first failure
python test_plan_engine.py -v                 # verbose output
```

### Test Result Summary (G4-G8, from verified run)

```
[G4] Approval Workflow         ✓  3 task steps approved
[G5] Execution & Auto-Complete ✓  task done → concept done → goal done
[G6] Ad-Hoc Plans              ✓  2 task steps from ad-hoc concept
[G7] Full Pipeline             ✓  decompose → execute → cascade done
[G8] Realistic Scenarios       ✓  14 task steps, targets: 8 model / 2 human / 4 investigate
Results: 5 passed, 0 failed
```

### Two-Tier Executor Tests (verified 2026-03-12)

Manual E2E tests on running server validating the two-tier failover system:

| Test | Tool | Args | Tier Hit | Executor Column | Cascade |
|------|------|------|----------|-----------------|---------|
| Tier 1 direct | `db_query` | `query` (synonym-mapped to `sql`) | Direct | `direct` | goal auto-completed |
| Tier 2 failover | `search_tavily` | `search_query` (bad arg, no synonym) | Primary LLM | `samaritan-execution` | goal auto-completed |

**Tier 1 test**: `db_query` with `{"query": "SELECT COUNT(*) ..."}` — direct executor mapped `query→sql` via synonym dict, executed Python function directly. Zero LLM cost. Completion cascade: task → concept → goal all `done`.

**Tier 2 failover test**: `search_tavily` with `{"search_query": "..."}` — direct call failed (`got an unexpected keyword argument 'search_query'`). System fell through to primary executor (gpt-4o-mini), which understood the intent, called `search_tavily(query=...)` correctly, and returned web search results. Completion cascade fired.

**Server log trace** (confirming execution path):
```
WARNING execute_task_step: id=297 tool=search_tavily direct failed:
  SearchTavilyPlugin...() got an unexpected keyword argument 'search_query'
INFO _try_llm_executors: id=297 tool=search_tavily trying plan_executor=samaritan-execution
INFO execute_task_step: id=297 tool=search_tavily executor=samaritan-execution (plan_executor) → done
```

**Executor evaluation results** (model candidates tested):

| Model | Model ID | Result | Issue |
|-------|----------|--------|-------|
| Llama 3.1 8B | meta-llama-3.1-8b-instruct | FAILED | Hallucinates tool outputs in text instead of using function calling protocol |
| Qwen3-235B-A22B | Qwen/Qwen3-235B-A22B-Instruct-2507 | PASSED | Correctly calls all 5 tool types (db_query, search_tavily, search_xai, google_drive, url_extract) |

---

## 14. Key Files

| File | Role |
|------|------|
| `plan_engine.py` | Core: two-tier execution, decomposition, approval, completion/failure propagation |
| `tools.py` | `_set_goal_exec()`, `_set_plan_exec()`, `get_tool_executor()` |
| `routes.py:cmd_plan()` | `!plan` command handler with all subcommands |
| `memory.py:load_typed_context_block()` | Context injection of goals + plan hierarchy into prompts |
| `reflection.py` | Cognition loop: goal completion detection, goal health, proposals |
| `agents.py:llm_call()` | LLM-to-LLM delegation used by Tier 2/3 executors (`mode='tool'`) |
| `config.py:get_model_role()` | Reads `model_roles` from `llm-models.json` for executor routing |
| `llm-models.json` | `plan-decomposer` (Haiku 4.5), `model_roles.plan_executor` → `samaritan-execution` (+ `backup_models` failover) |
| `migrations/006_plan_decomposition.sql` | Schema: step_type, parent_id, target, executor, tool_call, result, approval |
| `migrations/005_goal_health.sql` | Schema: attempt_count, failure_count, abandon_reason on goals |
| `test_plan_engine.py` | 8-group integration test suite |

---

## 15. Known Constraints

1. **Anthropic API**: Cannot send `temperature` + `top_p` together. `agents.py:_build_lc_llm()` skips `top_p` when host is `anthropic.com`.

2. **Config whitelist**: `config.py:load_llm_registry()` silently drops unknown keys. `allow_text_mode`, `memory_scan`, `max_tokens` must be whitelisted explicitly.

3. **Argument mapping (Tier 1)**: The synonym dict covers common cases (`query↔sql`, `text↔content`, `message↔prompt`). Novel mismatches cause direct execution to fail, but the LLM executor tiers handle these — the LLM sees the full tool schema and maps arguments correctly.

4. **Execution is sequential**: `execute_pending_tasks()` runs tasks one at a time, in step_order. No parallel execution.

5. **Human and investigate targets**: These cannot be auto-executed. They appear in `!plan` output for manual handling. No notification system exists for alerting humans to pending tasks.

6. **LLM executor requires active session**: Tier 2/3 use `llm_call()` which reads `current_client_id` ContextVar. If no session context is set (e.g., background reflection calling plan execution), the LLM executor may not push SSE tokens to any client. Tool execution still works — only the display feedback is affected.

7. **Executor model tool binding**: Executor models (`samaritan-execution`, `gpt-4o-execution`) must use **literal tool names** (e.g., `"db_query"`, `"search_tavily"`) in `llm_tools`, not toolset group names. Toolset groups have `always_active: false` with heat curves — fresh sessions get 0 tools bound. Literal names bypass the hot/cold subscription system.
