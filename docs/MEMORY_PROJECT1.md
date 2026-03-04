# Tiered Memory System — Project 1

Automatic, topic-aware, tiered memory with cross-session recall for the Samaritan agent persona.

> **Optional feature.** The entire memory system — or individual sub-features — can be toggled
> without touching source code. See [Enabling / Disabling](#enabling--disabling).

---

## Overview

By default, LLM sessions are stateless — each `!reset` or reconnect starts from a blank slate. This feature adds a persistent memory layer beneath all sessions: facts distilled from conversations survive resets, accumulate across weeks of use, and are automatically injected back into future requests with zero manual overhead.

```
Every request
        │
        ▼
auto_enrich_context()
  ├── embeds last 3 conversation turns → Qdrant semantic search (top-K relevant rows)
  ├── always injects rows with importance ≥ 8 (regardless of semantic score)
  └── injects ## Active Memory block into system message
        │
        ▼
Grok reasons and responds
  ├── Path A: issues memory_save tool call directly → DB write + Qdrant upsert
  └── Path B: narrates memory_save() in text → post-response scanner → DB write + Qdrant upsert

Session ends (!reset)
        │
        ▼
[summarizer-anthropic] ──extracts──► JSON facts (topic, content, importance)
        │
        ▼
samaritan_memory_shortterm  ◄── MySQL source of truth; also indexed in Qdrant (tier="short")
        │
        │  after 48h (low-importance rows)
        ▼
samaritan_memory_longterm   ◄── on-demand recall via tool call; Qdrant tier updated to "long"
        │
        │  future
        ▼
Google Drive archive        ◄── bulk cold storage (not yet built)
```

---

## Dual Database Architecture

The memory system uses two databases with complementary roles. Neither can replace the other.

### Why two databases?

| Need | MySQL handles it | Qdrant handles it |
|---|---|---|
| Store memory content permanently | ✅ source of truth | stores payload copy (secondary) |
| Structured queries (importance ≥ N, age > X hours) | ✅ SQL WHERE clauses | not designed for this |
| Deduplication (topic+content exact match) | ✅ SELECT WHERE | not designed for this |
| Aging (move rows between tiers) | ✅ DELETE/INSERT | tier payload updated as side-effect |
| Session summarization (read conversation → extract facts) | ✅ reads from here | not involved |
| `!reset` summarize-before-clear | ✅ | not involved |
| **Semantic similarity search** | not possible without extension | ✅ vector ANN search |
| **Context-aware retrieval** (what's relevant *right now*?) | not possible | ✅ |
| **`last_accessed` selective update** (only matched rows) | ✅ writes on demand | provides the hit list |
| Recover from Qdrant outage | ✅ fallback: load-all | fails gracefully → fallback |

### How they divide the retrieval problem

**MySQL** answers: *"What rows satisfy these structural criteria?"*
- Give me all rows with importance ≥ 8
- Give me rows created more than 48 hours ago
- Count rows grouped by topic

**Qdrant** answers: *"What rows are semantically close to this conversation?"*
- The last 3 conversation turns are embedded to a 768-dim vector
- Qdrant returns the top-K rows whose stored vectors are nearest (cosine similarity)
- No SQL knowledge of topics, importance, or content wording needed

Every turn combines both: Qdrant finds what's relevant, MySQL's `min_importance_always` threshold guarantees critical rows are never filtered out.

### Consistency model

MySQL is always written first. Qdrant is updated as a fire-and-forget async side-effect. This means:

- **Qdrant can be behind** by one embed round-trip (~50ms) — acceptable for retrieval
- **Qdrant can be missing rows** if the embed server was down during a save — detected by the drift indicator in `!memstats` (MySQL ST+LT count vs Qdrant points_count); fix with `backfill()`
- **Qdrant orphans** are possible if a MySQL row is manually deleted without calling `vec.delete_memory(row_id)` — currently harmless (search returns the orphan's id but MySQL lookup finds nothing)
- **Qdrant outage** degrades gracefully: `search_memories()` returns `[]`, `load_context_block()` falls back to loading all rows above `min_importance`

### Before vs. after Qdrant

| Aspect | Before (MySQL only) | After (MySQL + Qdrant) |
|---|---|---|
| Retrieval strategy | Load top-15 by importance, every turn | Embed query → ANN search → top-K relevant rows |
| Rows injected | Fixed 15 rows, same set every turn | Variable — only semantically close rows + always-inject |
| `last_accessed` | Updated on all 15 rows every turn (meaningless) | Updated only on rows Qdrant returned (meaningful staleness signal) |
| Topic variance | `work schedule` ≠ `Lee's hours` ≠ `schedule` | All cluster near each other in vector space |
| Off-topic injection | Schedule rows injected during coding discussion | Schedule rows below min_score threshold — not injected |
| Token cost | Fixed ~300–500 tokens regardless | ~50–400 tokens proportional to matched set size |

---

## Dependencies

### Software

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.11+ | Runtime | System |
| agent-mcp | Agent server framework | `git clone` + `pip install -r requirements.txt` |
| MySQL / MariaDB | Short-term and long-term memory storage (source of truth) | System |
| Qdrant | Vector search index for semantic memory retrieval | Binary or Docker |
| `qdrant-client>=1.7` | Qdrant Python client | `pip install qdrant-client` |
| `httpx>=0.24` | Async HTTP for embedding endpoint calls | `pip install httpx` |
| nomic-embed-text (GGUF) | Embedding model — 768-dim vectors, 84MB | llama.cpp server |
| `aiomysql` | Async MySQL driver | In `requirements.txt` |
| `langchain-openai` | LLM dispatch (OpenAI-compatible) | In `requirements.txt` |
| `langchain-google-genai` | LLM dispatch (Gemini) | In `requirements.txt` |

### Infrastructure (nuc11 — 192.168.x.x)

| Service | Port | Purpose |
|---|---|---|
| Qdrant | 6333 | Vector search — collection `samaritan_memory` |
| llama.cpp (nomic-embed-text) | 8000 | Embedding endpoint — `/v1/embeddings` |

Firewall rules required: allow TCP 6333 and 8000 inbound from agent-mcp host.

### API Keys Required

| Key | Used For | Env Var |
|---|---|---|
| Anthropic | `summarizer-anthropic` (Claude Haiku) | `ANTHROPIC_API_KEY` |
| xAI | `samaritan-reasoning` (Grok) | `XAI_API_KEY` |
| OpenAI | `samaritan-execution` (gpt-4o-mini) | `OPENAI_API_KEY` |

Gemini (`GEMINI_API_KEY`) is optional — `summarizer-gemini` is a fallback summarizer.

### Files Added / Modified

| File | Role |
|---|---|
| `memory.py` | Core memory module — all read/write/age/summarize logic; `_parse_table()` fixed for pipe-separated output; `load_context_block()` updated for semantic retrieval via Qdrant |
| `database.py` | Added `execute_insert()` — returns `cursor.lastrowid` in same connection (fixes LAST_INSERT_ID race) |
| `agents.py` | `auto_enrich_context()` builds semantic query from last 3 turns + calls Qdrant; loop guard fixed; threshold 2→3 |
| `routes.py` | `cmd_reset()` triggers summarize-before-clear |
| `tools.py` | Memory tools registered in both `CORE_LC_TOOLS` and `core_executors` (previously only in CORE_LC_TOOLS); `memory` toolset added |
| `config.py` | `load_llm_registry()` whitelist expanded to include `memory_scan` and `max_tokens` (previously stripped silently) |
| `plugin_memory_vector_qdrant.py` | **New** — infrastructure plugin: Qdrant + nomic-embed-text; exposes `get_vector_api()` singleton; no LangChain tools |
| `plugin-manifest.json` | Registered `plugin_memory_vector_qdrant` (type=data_tool, priority=50) |
| `plugins-enabled.json` | Enabled `plugin_memory_vector_qdrant` with qdrant/embed config |
| `llm-models.json` | `samaritan-reasoning`: switched to `grok-4-1-fast-reasoning`, `memory_scan: true` added, `max_tokens: 4096`; `memory` toolset removed (uses intercept path only) |
| `llm-tools.json` | `"memory"` toolset added |
| `db-config.json` | Instance-specific (gitignored): database name + table names; loaded by `memory.py` and `database.py` at startup |
| `system_prompt/004_reasoning/.system_prompt_memory` | Rewritten: direct tool call instructions + CRITICAL hallucination warning |
| `system_prompt/004_execution/.system_prompt_memory` | Removed hardcoded table names |

---

## Database Setup

Run once against your `agent_mcp` database:

```sql
CREATE TABLE samaritan_memory_shortterm (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  topic         VARCHAR(255) NOT NULL,
  content       TEXT NOT NULL,
  importance    TINYINT DEFAULT 5 COMMENT '1=low 5=med 10=critical',
  source        ENUM('session','user','directive') DEFAULT 'session',
  session_id    VARCHAR(255),
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_topic (topic),
  INDEX idx_importance (importance DESC),
  INDEX idx_created (created_at)
);

CREATE TABLE samaritan_memory_longterm (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  topic         VARCHAR(255) NOT NULL,
  content       TEXT NOT NULL,
  importance    TINYINT DEFAULT 5,
  source        ENUM('session','user','directive') DEFAULT 'session',
  session_id    VARCHAR(255),
  shortterm_id  INT COMMENT 'original shortterm row id',
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  aged_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_topic (topic),
  INDEX idx_importance (importance DESC)
);

CREATE TABLE samaritan_chat_summaries (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  session_id    VARCHAR(255) NOT NULL,
  summary       TEXT NOT NULL,
  message_count INT DEFAULT 0,
  model_used    VARCHAR(100),
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_session (session_id),
  INDEX idx_created (created_at)
);
```

---

## Configuration

### `llm-models.json` — Model Roles and Parameters

The memory system depends on three models working together. Here is the relevant configuration and the significance of each parameter.

#### `samaritan-reasoning` — The Reasoning Brain (Grok)

```json
"samaritan-reasoning": {
  "model_id": "grok-4-1-fast-reasoning",
  "type": "OPENAI",
  "host": "https://api.x.ai/v1",
  "env_key": "XAI_API_KEY",
  "max_context": 5000,
  "llm_tools": ["get_system_info", "llm_call", "llm_list", "search_tavily", "search_xai", "extract"],
  "system_prompt_folder": "system_prompt/004_reasoning",
  "temperature": 0.6,
  "top_p": 0.9,
  "max_tokens": 4096,
  "token_selection_setting": "custom",
  "memory_scan": true
}
```

| Parameter | Value | Why |
|---|---|---|
| `model_id` | `grok-4-1-fast-reasoning` | Grok with internal chain-of-thought; better judgment for memory saves and delegation decisions |
| `llm_tools` | minimal set — no `memory` toolset | Memory saves handled via intercept path (post-response scan), not direct tool calls |
| `memory_scan` | `true` | Enables `_scan_and_save_memories()` post-response scanner; Grok writes `memory_save(...)` in text and it is intercepted by agents.py |
| `temperature` | `0.6` | Reasoning model ignores this at the API level; set for documentation purposes |
| `top_p` | `0.9` | Same — reasoning model ignores it |
| `max_tokens` | `4096` | Caps reasoning output length; prevents runaway chain-of-thought token spend |
| `token_selection_setting` | `"custom"` | Applies temp/top_p/max_tokens |
| `max_context` | `5000` | Message count ceiling; effective window is `min(5000, agent_max_ctx=200)` = 200 messages |
| `system_prompt_folder` | `004_reasoning` | Loads the memory-aware Samaritan prompt tree |

> **Memory intercept path:** `grok-4-1-fast-reasoning` writes `memory_save(topic=..., content=..., importance=...)` in its response text rather than issuing a tool call. `_scan_and_save_memories()` in `agents.py` captures this via regex and writes to DB silently. No `memory` toolset needed on the reasoning model.

> **`config.py` whitelist:** `load_llm_registry()` must include `memory_scan` and `max_tokens` in its explicit field whitelist or both are silently dropped from the loaded registry, disabling the intercept path entirely.

#### `samaritan-execution` — The Obedient Executor (gpt-4o-mini)

```json
"samaritan-execution": {
  "model_id": "gpt-4o-mini",
  "type": "OPENAI",
  "host": "https://api.openai.com/v1",
  "env_key": "OPENAI_API_KEY",
  "max_context": 100000,
  "llm_tools": ["core", "db", "drive", "search", "memory"],
  "system_prompt_folder": "system_prompt/004_execution",
  "temperature": 0.1,
  "top_p": 0.5,
  "token_selection_setting": "custom"
}
```

| Parameter | Value | Why |
|---|---|---|
| `model_id` | `gpt-4o-mini` | Most reliable tool caller tested; follows instructions precisely |
| `llm_tools` | `["core","db","drive","search","memory"]` | Full tool access including the memory toolset |
| `temperature` | `0.1` | Near-deterministic; no creative variation in tool execution |
| `top_p` | `0.5` | Aggressive nucleus cutoff — only the most likely tokens |
| `token_selection_setting` | `"custom"` | Enforces the low temperature/top_p |
| `max_context` | `100000` | Can handle large delegation prompts from Grok |

#### `summarizer-anthropic` — The Memory Distiller (Claude Haiku)

```json
"summarizer-anthropic": {
  "model_id": "claude-haiku-4-5-20251001",
  "type": "OPENAI",
  "host": "https://api.anthropic.com/v1",
  "env_key": "ANTHROPIC_API_KEY",
  "max_context": 100000,
  "llm_tools": ["get_system_info", "drive", "vscode"],
  "system_prompt_folder": "system_prompt/003_claudeVSCode",
  "token_selection_setting": "default"
}
```

| Parameter | Value | Why |
|---|---|---|
| `model_id` | `claude-haiku-4-5-20251001` | Fast, cheap, accurate at structured JSON extraction |
| `max_context` | `100000` | Can handle long conversation histories without truncation |
| `llm_tools` | minimal | Summarizer only reads and writes; no tool execution needed |
| `token_selection_setting` | `"default"` | Haiku's defaults are well-tuned for extraction tasks |

### `llm-tools.json` — The Memory Toolset

```json
"memory": [
  "memory_save",
  "memory_recall",
  "memory_age",
  "memory_update"
]
```

This toolset is assigned to `samaritan-execution`. `samaritan-reasoning` (Grok) does **not** have the `memory` toolset — it uses the intercept path exclusively (`memory_scan: true`). `samaritan-execution` at temp=0.1 reliably calls these tools; delegation via `llm_call` from Grok is the reliable path for explicit saves.

---

## System Prompt Tree (`system_prompt/004_reasoning/`)

The prompt is a tree of files. The root `.system_prompt` assembles the tree via `[SECTIONS]` markers. Each section entry **must** use the format `name: description` — bare names are silently skipped by the parser.

```
.system_prompt                          ← root: identity + prime directives
  ├── tools: Available tool definitions
  │     ├── tool_get_system_info
  │     ├── tool_db_query
  │     ├── tool_google_drive
  │     ├── tool_search_tavily
  │     ├── tool_search_xai
  │     ├── tool_llm_list
  │     └── tool_llm_delegation         ← describes llm_call delegation pattern
  ├── behavior: Behaviour and delegation rules
  └── continuity: Cross-session memory and continuity
        ├── continuity_tiered           ← Tier 1/2 startup SQL + decision logging
        └── memory                      ← memory save/recall/age procedures
```

### Memory-Relevant Sections

**`.system_prompt_behavior`** — tells Grok how to behave:
- Rule 6: Delegate DB writes and tool chains to `samaritan-execution` via `llm_call`
- Rule 7: Treat injected `## Active Memory` as ground truth; do not ask about known facts
- Rule 8: Mandatory memory capture — scan every user message for saveable facts; call `memory_save` before completing response if any of: future events, people, preferences, decisions, life facts, or explicit "remember this" are present

**`.system_prompt_continuity`** — describes the three-tier model and what triggers each tier

**`.system_prompt_memory`** — detailed procedures:
- When to save (every turn, behavior rule 8 categories)
- Topics are dynamic: use the **Known topics** list injected in `## Active Memory`; create new topics only when none fit
- Importance scale (6=useful context, 7-8=concrete plans, 9=high-stakes, 10=imminent/critical)
- How to trigger long-term recall via delegation
- How to trigger memory aging

**`.system_prompt_continuity_tiered`** — startup SQL for loading tasks/initiatives/assets from other `samaritan_*` tables (separate from the memory tables; used for operational continuity)

### What Triggers Memory Features

| Trigger | What Happens | Feature flag |
|---|---|---|
| Every request | `auto_enrich_context()` prepends `## Active Memory` + Known topics list to system message | `context_injection` |
| `!reset` with ≥4 messages in history | `cmd_reset()` calls `summarize_and_save()` before clearing | `reset_summarize` |
| Grok response contains `memory_save(...)` text | Post-response scanner regex extracts and writes to DB | `post_response_scan` |
| Grok issues actual `memory_save` tool call | `execute_tool()` → `_memory_save_exec()` → DB write | *(no separate flag — inherits master switch via tool availability)* |
| Any message: fact detected (rule 8 trigger) | Behavior rule 8 mandates Grok calls `memory_save` before completing response | *(system prompt — disable by editing behavior rule 8)* |
| Topic present but not in active memory | Grok delegates `memory_recall(topic, tier="long")` to `samaritan-execution` | *(on-demand — not auto)* |
| Session created or rehydrated from disk | `age_to_longterm()` runs as background task — rows >48h moved to long-term | *(always runs; no separate flag)* |

---

## Runtime: How Memory Flows

### Request Flow (Every Conversation Turn)

```
User message arrives
        │
        ▼
dispatch_llm() → agentic_lc()
        │
        ▼
auto_enrich_context()   [feature: context_injection]
  ├── build query = last 3 conversation turns (user+assistant, 300 chars each)
  ├── IF Qdrant plugin loaded AND query non-empty:
  │     ├── embed(query) → nomic-embed-text (nuc11:8000) → 768-dim vector
  │     ├── Qdrant query_points(tier="short", top_k=20, min_score=0.45) → relevant rows
  │     ├── load_short_term(min_importance=8) → always-injected high-importance rows
  │     ├── merge: always-rows first, then semantic hits not already included
  │     └── _update_last_accessed(semantic_hit_ids) — only matched rows updated
  ├── ELSE (fallback): load_short_term(limit=15, min_importance=3) — all rows
  └── injects "## Active Memory" system message (grouped by topic)
        │
        ▼
agentic_lc() — Grok sees enriched context
  ├── Behavior rule 8: scans user message for saveable facts
  │     └── if found → writes memory_save(...) in response text (Path B, not Path A)
  └── Grok responds
        │
        ▼
Post-response scanner   [feature: post_response_scan + model.memory_scan]
  └── regex scan of final text for memory_save(topic=...) syntax (Path B)
        └── if matched → _memory_save_exec() → save_memory() → DB write + Qdrant upsert
```

Token cost of injection: **~200–400 tokens** for the semantically matched rows, regardless of total rows in storage. High-importance rows always injected even if not semantically close to current topic.

### Save Path A: Direct Tool Call

samaritan-execution (gpt-4o-mini) issues a `memory_save` tool call in the normal agentic loop:

```
gpt-4o-mini turn
  └── memory_save(topic="X", content="Y", importance=8)   ← tool call
            │
            ▼
      execute_tool() → _memory_save_exec() in tools.py
        └── save_memory() in memory.py
              ├── dedup check (shortterm + longterm)
              ├── execute_insert() → cursor.lastrowid → DB row
              └── asyncio.create_task(vec.upsert_memory(row_id, ...))  ← fire-and-forget
                    └── embed(content) → Qdrant upsert(id=row_id, tier="short")
```

### Save Path B: Post-Response Scanner (Primary path for samaritan-reasoning)

Grok writes `memory_save(...)` in response text; the scanner intercepts and saves silently:

```
Grok final response text:
  "...memory_save(topic='family', content='Lee has Monday off', importance=7)..."
        │
        ▼
_scan_and_save_memories()   [agents.py, fires after response streams]
  ├── Pass 1: regex match on memory_save(topic=..., content=...) syntax
  │           (backreference pattern handles apostrophes in content)
  └── Pass 2: JSON-blob form fallback (if pass 1 finds nothing)
        │
        ▼
  _memory_save_exec() → save_memory() → DB + Qdrant (same path as Path A)
```

Activated per-model via `"memory_scan": true` in `llm-models.json`. Currently set on `samaritan-reasoning` only. **Both** `post_response_scan` (global) and `memory_scan` (per-model) must be `true` for this path to fire.

### Save Path C: Delegation Flow (Grok → gpt-4o-mini)

Grok can also delegate saves explicitly via `llm_call`:

```
Grok reasoning turn
  └── llm_call("samaritan-execution", "save memory: topic=X content=Y importance=8")
            │
            ▼
      samaritan-execution (gpt-4o-mini) receives prompt
        └── calls memory_save tool (Path A)
```

gpt-4o-mini at temp=0.1 reliably translates natural-language delegation prompts into precise tool calls.

### Reset Flow (Summarize → Save → Clear)

```
User sends !reset
        │
        ▼
cmd_reset() in routes.py   [feature: reset_summarize]
  ├── if history ≥ 4 messages:
  │     ├── push "[memory] Summarizing session to memory..."
  │     ├── summarize_and_save(session_id, history, "summarizer-anthropic")
  │     │     ├── builds condensed history text (last 60 turns, skip tool messages)
  │     │     ├── calls _call_llm_text("summarizer-anthropic", extraction_prompt)
  │     │     ├── parses JSON → [{topic, content, importance}, ...]
  │     │     ├── calls save_memory() for each fact → samaritan_memory_shortterm
  │     │     └── INSERT INTO samaritan_chat_summaries (full raw summary)
  │     └── push "[memory] Summarized N messages → M memories saved."
  ├── session["history"] = []
  └── push "Conversation history cleared."
```

### Topic Lifecycle

Topics are not configured anywhere — they emerge from use and persist in the DB:

```
First save with topic="travel-plans"
  └── INSERT INTO samaritan_memory_shortterm (topic='travel-plans', ...)
        │
        ▼
Next request: load_topic_list() queries DISTINCT topic from both tables
  └── "travel-plans" appears in Known topics list in ## Active Memory
        │
        ▼
Grok sees "travel-plans" in Known topics → reuses it for new saves
  (no config change, no restart needed)
```

Adding a topic: save any memory with a new topic name — it auto-appears in future injections.
Removing a topic: delete all rows with that topic from both memory tables.

### Aging Flow

Rows age from short-term to long-term via two independent continuous background tasks
started at server startup inside `agent-mcp.py`. No manual trigger or cron job is needed.

```
Server starts → asyncio.gather() launches two background loops:

_age_count_task()   (count-pressure, interval: memory_age_count_timer minutes)
  ├── reads _age_cfg() fresh each cycle
  ├── skips if auto_memory_age=false or timer=-1
  └── age_by_count(max_rows=200)
        ├── reads entry_limit from config (memory_age_entrycount)
        ├── if shortterm_count > entry_limit:
        │     move overflow rows (lowest importance, oldest last_accessed) to longterm
        └── asyncio.create_task(vec.update_tier(row_id, "long"))  ← Qdrant sync

_age_minutes_task()   (staleness, interval: memory_age_minutes_timer minutes)
  ├── reads _age_cfg() fresh each cycle
  ├── skips if auto_memory_age=false or timer=-1
  └── age_by_minutes(trigger_minutes=memory_age_trigger_minutes, max_rows=200)
        ├── SELECT rows WHERE last_accessed < NOW() - INTERVAL N MINUTE
        │   ORDER BY importance ASC, last_accessed ASC  LIMIT max_rows
        ├── for each row:
        │     ├── INSERT INTO longterm (copying all fields + shortterm_id)
        │     ├── DELETE FROM shortterm WHERE id = row.id
        │     └── asyncio.create_task(vec.update_tier(row_id, "long"))  ← Qdrant sync
        └── returns count of rows moved (logged only)
```

Manual override: ask Grok to delegate `memory_age(older_than_hours=24)` to age more aggressively.

---

## Enabling / Disabling

The memory system is an optional feature. Configuration lives in `plugins-enabled.json`
under `plugin_config.memory`. **All changes take effect immediately — no server restart required.**
The server re-reads this config on every request and every aging cycle.

### Via agentctl

```bash
# Show current state
python3 agentctl.py memory status

# Disable everything (master switch)
python3 agentctl.py memory disable

# Re-enable everything
python3 agentctl.py memory enable

# Disable a specific sub-feature only
python3 agentctl.py memory disable context_injection
python3 agentctl.py memory disable reset_summarize
python3 agentctl.py memory disable post_response_scan

# Re-enable a sub-feature
python3 agentctl.py memory enable reset_summarize
```

### Via JSON (manual)

`plugins-enabled.json` → `plugin_config.memory`:

```json
"memory": {
  "enabled": true,
  "context_injection": true,
  "reset_summarize": true,
  "post_response_scan": true,
  "fuzzy_dedup": true,
  "vector_search_qdrant": true
}
```

### Feature flags

| Key | Default | Controls |
|---|---|---|
| `enabled` | `true` | Master switch — disabling this overrides all sub-features |
| `context_injection` | `true` | `## Active Memory` + Known topics injected into every request |
| `reset_summarize` | `true` | Session summarized to memory on `!reset` |
| `post_response_scan` | `true` | Regex scan of final response text for `memory_save(...)` narration |
| `fuzzy_dedup` | `true` | Block near-duplicate saves via SequenceMatcher similarity |
| `vector_search_qdrant` | `true` | Semantic retrieval via Qdrant; disable to fall back to load-all by importance |

The `memory_scan` flag in `llm-models.json` is a **per-model** gate for post-response scanning.
Both `post_response_scan` (global) and `memory_scan` (per-model) must be `true` for scanning to fire.

### What is NOT gated

- The `memory_save` / `memory_recall` / `memory_age` **tools** remain registered regardless of these flags — they can still be called via direct tool use or delegation. The flags only suppress the *automatic* behaviors (injection, reset summarize, scan).
- Memory aging (`_age_count_task` / `_age_minutes_task`) runs as continuous background loops started at server startup; it is not gated by these flags.
- The behavior rule 8 mandatory-save instruction lives in the system prompt. To suppress it, edit `system_prompt/004_reasoning/.system_prompt_behavior` and remove rule 8, or set the model's `system_prompt_folder` to a folder without that rule.

---

## Use Cases

### 1. Preference Persistence

**Without memory:** Every new session needs re-establishing: "I use Python 3.11, always use f-strings, prefer concise responses."

**With memory:**
```
Session 3: "Always use type hints in code examples."
!reset

Session 4: [Active Memory already contains: "Prefers type hints in code examples (imp=7)"]
Grok uses type hints without being asked.
```

---

### 2. Project Continuity

```
Session 7: Long debug session on agent-mcp — found that Grok loops on tool calls,
           fixed with fingerprint detection.
!reset

→ [memory] 5 memories saved:
  technical-decisions: "Grok tool-call loop fixed via fingerprint detection in agents.py"
  project-status: "Memory system build complete, 18/18 tests passing"

Session 8: "Where did we leave off?"
Grok: "Last session: completed the tiered memory system. Key decision:
       Grok reasons, gpt-4o-mini executes. Both tested and working."
```

---

### 3. Explicit Important Fact

```
User: "Remember this: the production API rotates keys every 90 days, next rotation April 15."

Grok → delegates:
  llm_call("samaritan-execution",
    "save memory: topic=security, content=prod API key rotates every 90 days next April 15, importance=9")

Confirmed: memory saved (imp=9). Will appear in every future session.
```

---

### 4. Long-Term Recall

```
User: "What was the decision about the Hermes model?"
[Not in active short-term memory — too old, aged out]

Grok → delegates:
  llm_call("samaritan-execution", "recall memories about: Hermes tier=long")

← "Hermes-3-Llama-3.1-8B tested on nuc11, rejected — stalls after listing files,
   cannot self-chain tool calls. Decision: use Qwen2.5-7B instead."

Grok answers with full context from 3 weeks ago.
```

---

### 5. Keeping Short-Term Lean

After many sessions, short-term fills up. Trigger aging to keep injection fast:

```
!db_query SELECT COUNT(*) FROM samaritan_memory_shortterm
→ 87 rows

Ask Grok: "Age memories older than 24 hours."
Grok → delegates: memory_age(older_than_hours=24, max_rows=200)
→ "Aged 62 memories from short-term to long-term (threshold: 24h)."

!db_query SELECT COUNT(*) FROM samaritan_memory_shortterm
→ 25 rows  ← injection stays fast
```

---

### 6. Viewing What's Remembered

```
!db_query SELECT topic, content, importance FROM samaritan_memory_shortterm ORDER BY importance DESC LIMIT 10
!db_query SELECT session_id, message_count, created_at FROM samaritan_chat_summaries ORDER BY created_at DESC LIMIT 5
```

---

## `memory.py` Public API

```python
# Save a fact to short-term memory
await save_memory(topic, content, importance=5, source="session", session_id="")

# Load hot memories as list of dicts (importance DESC)
rows = await load_short_term(limit=20, min_importance=1)

# Move old rows from short-term to long-term; returns count moved
moved = await age_to_longterm(older_than_hours=48, max_rows=100)

# Get formatted string ready to inject into a system message
# query: recent conversation text used for semantic search (empty = fallback to load-all)
block = await load_context_block(min_importance=3, query="current conversation excerpt")

# Summarize a conversation history and save extracted facts to short-term
status = await summarize_and_save(session_id, history, model_key="summarizer-anthropic")
```

---

## Tool Reference

### `memory_save`
Save an explicit fact to short-term memory.

```
memory_save(
  topic      = "user-preferences",     # short label; groups display
  content    = "Prefers dark mode.",    # one concise sentence
  importance = 7,                       # 1=low, 5=medium, 10=critical
  source     = "user"                   # "user" | "session" | "directive"
)
```

### `memory_recall`
Query stored memories by topic and tier.

```
memory_recall(
  topic = "security",   # keyword filter on topic column (LIKE %topic%)
  tier  = "short",      # "short" (default) or "long"
  limit = 20
)
```

### `memory_age`
Move old short-term rows to long-term.

```
memory_age(
  older_than_hours = 48,    # move rows older than this
  max_rows         = 100    # cap per invocation
)
```

### `memory_update`
Update fields on an existing memory row by id.

```
memory_update(
  id         = 42,           # row id from memory_recall or !memory list
  tier       = "short",      # "short" (default) or "long"
  importance = 9,            # 1–10; omit or 0 to leave unchanged
  content    = "New text.",  # omit or "" to leave unchanged
  topic      = "new-topic"   # omit or "" to leave unchanged
)
```

---

## Vector Memory Plugin (`plugin_memory_vector_qdrant.py`)

An infrastructure plugin with no LangChain tools. Loaded by `plugin_loader.py`; the module-level singleton is accessed by `memory.py` and `agents.py` via `get_vector_api()`.

### Configuration (`plugins-enabled.json`)

```json
"plugin_memory_vector_qdrant": {
  "enabled": true,
  "qdrant_host": "192.168.x.x",
  "qdrant_port": 6333,
  "embed_url": "http://192.168.x.x:8000/v1/embeddings",
  "embed_model": "nomic-embed-text",
  "collection": "samaritan_memory",
  "vector_dims": 768,
  "top_k": 20,
  "min_score": 0.45,
  "min_importance_always": 8
}
```

| Parameter | Effect |
|---|---|
| `top_k` | Max rows returned per semantic search |
| `min_score` | Cosine similarity floor (0–1); lower = more permissive |
| `min_importance_always` | Rows at or above this importance are always injected regardless of semantic score |

### Public API

```python
from plugin_memory_vector_qdrant import get_vector_api

vec = get_vector_api()   # None if plugin not loaded/enabled

await vec.upsert_memory(row_id, topic, content, importance, tier="short")
await vec.search_memories(query_text, top_k, min_score, tier="short") -> list[dict]
await vec.delete_memory(row_id)
await vec.update_tier(row_id, new_tier)
await vec.backfill(rows, tier="short") -> int   # embed + upsert existing MySQL rows
```

### MySQL ↔ Qdrant consistency

| Operation | MySQL | Qdrant |
|---|---|---|
| `save_memory()` | INSERT → row_id | `upsert_memory(row_id)` — fire-and-forget |
| `age_to_longterm()` | move row | `update_tier(row_id, "long")` — fire-and-forget |
| Manual row delete | DELETE | `delete_memory(row_id)` — **not auto-called yet** |
| Qdrant outage | not affected | `search_memories` returns `[]`; fallback to load-all |

### Qdrant API note

qdrant-client 1.7+ renamed `.search()` to `.query_points()`. The result is accessed as `response.points` (not a direct list). Using the old `.search()` method raises `AttributeError`.

### Backfill

To embed and index existing MySQL rows that predate the plugin:

```python
rows = await load_short_term(limit=10000, min_importance=1)
vec = get_vector_api()
count = await vec.backfill(rows, tier="short")
```

---

## Bugs Discovered and Fixed (2026-03-01)

All bugs were independently silent — no exceptions were raised; the system appeared functional while saving nothing.

### Bug 1: Memory tools missing from `get_tool_executor()` — main culprit

**File:** `tools.py` — `get_tool_executor()`

`memory_save`, `memory_recall`, and `memory_age` were registered in `CORE_LC_TOOLS` (so the LLM could see them and issue calls) but were absent from the `core_executors` dict that `get_tool_executor()` uses to dispatch calls.

Every tool call returned `"Unknown tool: memory_save"` as a ToolMessage result. The model (gpt-4o-mini) retried with identical args → loop guard fired → zero rows saved, while the model responded "Memory saved successfully."

**Fix:** Added the three tools to `core_executors` in `get_tool_executor()`.

```python
# Before (missing):
core_executors = { 'get_system_info': ..., 'llm_call': ..., ... }

# After:
core_executors = {
    ...
    'memory_save':   _memory_save_exec,
    'memory_recall': _memory_recall_exec,
    'memory_age':    _memory_age_exec,
}
```

### Bug 2: `_parse_table()` split on wrong delimiter

**File:** `memory.py` — `_parse_table()`

`_parse_table()` split header and data lines on `\t` (tab character), but `execute_sql()` returns pipe-separated output:

```
id | topic         | content          | importance
---+---------------+------------------+-----------
16 | schedule      | Lee has Mon off  | 7
```

Every `load_short_term()` and `load_long_term()` call returned rows where all fields were `None`. Memory recall always said "No memories found" even when rows existed in the DB.

**Fix:** Changed delimiters from `\t` to `|` and added a separator-line filter:

```python
# Before:
headers = [h.strip() for h in lines[0].split("\t")]
vals = line.split("\t")

# After:
headers = [h.strip() for h in lines[0].split("|")]
if set(line.strip()) <= set("-+"):  # skip ---+--- separator lines
    continue
vals = line.split("|")
```

### Bug 3: `LAST_INSERT_ID()` race condition in `save_memory()`

**File:** `database.py` + `memory.py`

`save_memory()` ran the INSERT and `SELECT LAST_INSERT_ID()` as two separate `execute_sql()` calls. Each call opens and closes a fresh DB connection. `LAST_INSERT_ID()` is connection-scoped — calling it on a new connection always returns 0.

Effect: the second `memory_save` call in any session always returned `row_id=0`, which the dedup check treated as "already exists". The first call saved correctly (row was actually inserted) but reported id=0, then the retry on the second call hit the dedup check and silently skipped.

**Fix:** New `execute_insert()` in `database.py` that returns `cursor.lastrowid` before closing the connection:

```python
def _run_insert(sql: str) -> int:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    return cursor.lastrowid or 0   # same connection — correct value

async def execute_insert(sql: str) -> int:
    return await asyncio.to_thread(_run_insert, sql)
```

### Bug 4: `load_llm_registry()` whitelist silently drops `memory_scan` and `max_tokens`

**File:** `config.py` — `load_llm_registry()`

`load_llm_registry()` built the in-memory model registry using an explicit field whitelist (dict with hardcoded keys). Fields not in the whitelist were silently dropped — no warning, no error.

`memory_scan` was not in the whitelist → `_memory_scan` was always `False` in `dispatch_llm()` → `_scan_and_save_memories()` was never called → Path B saves never fired, for the entire lifetime of the feature.

`max_tokens` was also dropped, so Grok's chain-of-thought was uncapped.

**Fix:** Added both fields to the registry dict in `load_llm_registry()`:

```python
registry[name] = {
    ...
    "memory_scan": config.get("memory_scan", False),
    "max_tokens":  config.get("max_tokens"),
}
```

**Root cause pattern:** Explicit whitelist registries silently discard new model config fields. Any new field added to `llm-models.json` must also be added to the whitelist or it will never reach the runtime.

---

### Loop guard: OpenAI 400 on HumanMessage injection

**File:** `agents.py` — `agentic_lc()`

When the loop guard fired (same tool+args repeated `_TOOL_LOOP_THRESHOLD` times), it injected a `HumanMessage("stop, answer now")` into the context before resolving the current turn's `ai_msg.tool_calls`. OpenAI requires every `tool_calls` in an AIMessage to be followed by corresponding `ToolMessage` entries; the bare HumanMessage caused a 400 error.

**Fix:** Execute all pending tool calls (add ToolMessages to ctx) before injecting the HumanMessage break. Threshold also raised from 2→3 to allow one retry after a legitimate dedup/no-op result.

### Hallucination guard: system prompt CRITICAL warning → intercept path

**File:** `system_prompt/004_reasoning/.system_prompt_memory`

Grok was narrating saves without calling the tool. Initially added a CRITICAL instruction:

```
**CRITICAL**: You MUST actually invoke the `memory_save` tool. Never claim memory was saved
without having called the tool. If the tool call does not appear in your response, memory
was NOT saved.
```

**This instruction is no longer in the current prompt.** The system evolved: Grok's `memory` toolset was removed entirely and replaced with the intercept path (`memory_scan: true`). The current CRITICAL says the opposite — Grok does NOT have `memory_save` as a registered tool; instead it writes `memory_save(...)` literally in response text and `_scan_and_save_memories()` in `agents.py` intercepts and executes it silently.

The reliable path for guaranteed saves remains delegation to `samaritan-execution` via `llm_call`.

---

## What's Not Yet Built

Priority = functionality gain ÷ implementation complexity. P1 = high value, low effort. P4 = low value or high complexity.

| Priority | Feature | Status | Effort | Notes |
|---|---|---|---|---|
| P1 | Memory confirmation UX | **Built** | Low | `[memory] Summarized N messages → M memories saved, K duplicate(s) skipped.` pushed after every reset. |
| P1 | Deduplication | **Built** | Low | `save_memory()` checks both shortterm and longterm for identical topic+content before inserting. Duplicate returns 0 without touching the DB. |
| P2 | Importance decay | Not built | Low-Med | Scheduled SQL: `UPDATE shortterm SET importance = importance - 1 WHERE importance > 3 AND created_at < NOW() - INTERVAL 7 DAY`. Could run at session-start alongside aging. Prevents stale high-imp facts from crowding injection. |
| P2 | Per-topic retention policies | Not built | Med | JSON config mapping topic patterns → `age_after_hours` overrides. Requires modifying `age_to_longterm()` to apply per-row policy instead of a single threshold. High value for operational use but needs schema for the config. |
| P3 | Google Drive archival | Not built | Med | `memory_age` extended to optionally export aged rows to a Drive file before deletion. Google Drive plugin already exists — mainly plumbing. Low urgency since longterm table handles this well enough. |
| P3 | Drive → short-term reload | Not built | Med | Inverse of archival: parse Drive export back into shortterm. Depends on archival being built first; blocked on P3 above. |
| P3 | Hybrid search (BM25 + vector) | Not built | Low-Med | Add MySQL `FULLTEXT INDEX` on `content` + `topic` columns. In `load_context_block()` and `memory_recall`, run a `MATCH AGAINST` keyword query in parallel with the Qdrant ANN query, merge results, dedupe by `row_id`. Improves recall for exact-match queries (proper nouns, codes, dates, technical strings) where vector similarity may be weak. Scope: primarily benefits `memory_recall` tool calls; auto-injection already benefits from broad semantic matching. Implementation: ~3 lines schema, ~35 lines in `memory.py`. No new infrastructure — MySQL already holds the data. |
| P4 | Semantic/vector search | **Built (2026-03-01)** | High | `plugin_memory_vector_qdrant.py` — Qdrant + nomic-embed-text on nuc11. Replaces load-all with top-K semantic retrieval per turn. Also solves `last_accessed` staleness: only semantically matched rows get timestamp updated. |
| P4 | Stale Qdrant orphan cleanup | Not built | Low | If a MySQL row is deleted (e.g. manual dedup), the Qdrant point becomes a stale orphan. Currently harmless (hit returns a non-existent row_id which MySQL ignores). Fix: call `vec.delete_memory(row_id)` alongside any MySQL DELETE. |
