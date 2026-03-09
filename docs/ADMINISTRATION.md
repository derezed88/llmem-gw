# Administration Guide

## Installation

### Prerequisites

- Python 3.11+
- A Python virtual environment
- At minimum: no external services required for shell.py-only mode

### Setup

```bash
git clone <repo>
cd llmem-gw
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # or install per-plugin deps below
cp .env.example .env              # fill in API keys
```

### Environment Variables (`.env`)

| Variable | Required by | Description |
|---|---|---|
| `GEMINI_API_KEY` | gemini models, google_search | Google AI Studio key |
| `XAI_API_KEY` | grok models, xai_search | xAI key |
| `OPENAI_API_KEY` | openai model | OpenAI key |
| `MYSQL_USER` | plugin_database_mysql | Database username |
| `MYSQL_PASS` | plugin_database_mysql | Database password |
| `TAVILY_API_KEY` | tavily_search, url_extract | Tavily API key |
| `FOLDER_ID` | plugin_storage_googledrive | Google Drive folder ID |
| `SLACK_BOT_TOKEN` | plugin_client_slack | Slack bot token |
| `SLACK_APP_TOKEN` | plugin_client_slack | Slack app token (Socket Mode) |

---

## Starting the Server

```bash
source venv/bin/activate
python llmem-gw.py
```

The server starts on port **8765** (MCP/shell.py) by default. Optional flags:

```bash
python llmem-gw.py --help
```

---

## System Administration (`llmemctl.py`)

All plugin and model configuration is done through `llmemctl.py`. Run interactively or with CLI arguments.

```bash
python llmemctl.py           # interactive menu
python llmemctl.py <cmd>     # direct CLI
```

### Plugin Commands

```bash
python llmemctl.py list                    # list all plugins with status
python llmemctl.py info <plugin_name>      # detailed info + setup instructions
python llmemctl.py enable <plugin_name>    # enable a plugin
python llmemctl.py disable <plugin_name>   # disable a plugin
```

**Plugin status indicators:**
- `✓` Enabled — active, all dependencies met, all credentials present
- `–` Disabled — in `enabled_plugins` but turned off via `enabled: false` in `plugin_config`
- `○` Configured — available (file + deps present) but not in `enabled_plugins`
- `✗` Has Issues — enabled but missing dependencies, env vars, or config files
- `⊗` Unavailable — not enabled and has unresolved issues

**Plugin names** (use with enable/disable):

| Plugin | Type | What it enables |
|---|---|---|
| `plugin_client_shellpy` | client_interface | shell.py terminal client (always keep enabled) |
| `plugin_proxy_llama` | client_interface | OpenAI/Ollama API (port set via `llama_port` in `plugins-enabled.json`) |
| `plugin_client_slack` | client_interface | Slack bidirectional client (see tuning below) |
| `plugin_database_mysql` | data_tool | `db_query` tool |
| `plugin_storage_googledrive` | data_tool | `google_drive` tool |
| `plugin_search_ddgs` | data_tool | `ddgs_search` tool (no key required) |
| `plugin_search_tavily` | data_tool | `tavily_search` tool |
| `plugin_search_xai` | data_tool | `xai_search` tool |
| `plugin_search_google` | data_tool | `google_search` tool |
| `plugin_urlextract_tavily` | data_tool | `url_extract` tool |
| `plugin_tmux` | data_tool | PTY shell sessions (`tmux_new`, `tmux_exec`, etc.) |

### Slack Plugin Tuning

The Slack plugin posts each agent turn to Slack immediately as it completes, rather than
waiting for the full conversation to finish. After each turn it waits up to
`inter_turn_timeout` seconds for the next turn to begin before declaring the conversation done.

**Why you may need to tune this:**

- **Too low (< 10s):** The final summary from the orchestrating LLM (e.g. grok4) is cut off.
  After the last `agent_call` returns, the LLM still needs one more inference pass to generate
  its synthesis — frontier models typically need 3–10s for this. If the timeout expires first,
  the Slack conversation ends without the closing summary.
- **Too high (> 60s):** No functional harm, but the Slack thread stays "open" for longer after
  the last message appears, which may look like the agent is still working.

**Default:** 30 seconds. Configure in `plugins-enabled.json`:

```json
"plugin_client_slack": {
  "enabled": true,
  "slack_port": 8766,
  "slack_host": "0.0.0.0",
  "inter_turn_timeout": 30
}
```

Override without editing JSON using `.env`:
```
SLACK_INTER_TURN_TIMEOUT=45
```
The JSON value takes precedence over `.env` if both are set.

---

### Model Commands

```bash
python llmemctl.py models                              # list all models
python llmemctl.py model-info <model_name>            # detailed model info
python llmemctl.py model-add                          # interactive wizard
python llmemctl.py model <model_name>                 # set as default model
python llmemctl.py model-cfg list                     # list all models (compact)
python llmemctl.py model-cfg read <name>              # show full model config
python llmemctl.py model-cfg write <name> <field> <value>  # update a field
python llmemctl.py model-cfg copy <source> <new_name>      # clone a model
python llmemctl.py model-cfg enable <name>            # enable a model
python llmemctl.py model-cfg disable <name>           # disable a model
python llmemctl.py model-cfg delete <name>            # remove a model
```

**Safety rules:** The default model cannot be disabled or removed. Change the default first with `model <name>`.

### Rate and Depth Limit Commands

```bash
python llmemctl.py limits list                        # show all limits
python llmemctl.py limits read <key>                  # read a specific limit
python llmemctl.py limits write <key> <value>         # update a limit
```

Limit keys: `max_at_llm_depth`, `max_agent_call_depth`, `max_tool_iterations`,
`session_idle_timeout_minutes`, `max_users`, `rate_<type>_calls`, `rate_<type>_window`

Tool rate types: `llm_call`, `search`, `extract`, `drive`, `db`, `system`, `tmux`

### Memory System

The tiered memory system persists facts across sessions using MySQL. It is optional and
every feature is independently togglable.

#### Prerequisites

- `plugin_database_mysql` must be enabled and connected.
- Three tables must exist in the database. Create them once:

```sql
CREATE TABLE IF NOT EXISTS <prefix>memory_shortterm (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    topic         VARCHAR(255) NOT NULL,
    content       TEXT NOT NULL,
    importance    TINYINT DEFAULT 5,
    source        VARCHAR(50) DEFAULT 'session',
    session_id    VARCHAR(255) DEFAULT '',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS <prefix>memory_longterm (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    topic      VARCHAR(255) NOT NULL,
    content    TEXT NOT NULL,
    importance TINYINT DEFAULT 5,
    source     VARCHAR(50) DEFAULT 'session',
    session_id VARCHAR(255) DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS <prefix>chat_summaries (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255),
    summary    TEXT,
    msg_count  INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Replace `<prefix>` with your instance prefix (set in `db-config.json`).

#### `db-config.json` (gitignored, instance-specific)

Stores the table name prefix and fully-qualified table names for this deployment.
If absent, bare names `memory_shortterm`, `memory_longterm`, `chat_summaries` are used.

```json
{
  "managed_table_prefix": "myinstance_",
  "tables": {
    "memory_shortterm": "myinstance_memory_shortterm",
    "memory_longterm":  "myinstance_memory_longterm",
    "chat_summaries":   "myinstance_chat_summaries"
  }
}
```

#### `llmemctl.py` memory commands

```bash
python llmemctl.py memory status                                  # show all feature states + settings
python llmemctl.py memory enable                                  # enable master switch
python llmemctl.py memory disable                                 # disable everything
python llmemctl.py memory enable <feature>                        # enable one feature
python llmemctl.py memory disable <feature>                       # disable one feature
python llmemctl.py memory set fuzzy_dedup_threshold <0-1>        # adjust similarity threshold
python llmemctl.py memory set summarizer_model <model_key>       # change summarizer model
python llmemctl.py memory set auto_memory_age <true|false>       # enable/disable background aging
python llmemctl.py memory set memory_age_entrycount <n>          # max short-term rows
python llmemctl.py memory set memory_age_count_timer <min|-1>    # count-pressure interval (-1=off)
python llmemctl.py memory set memory_age_trigger_minutes <min>   # staleness threshold in minutes
python llmemctl.py memory set memory_age_minutes_timer <min|-1>  # staleness check interval (-1=off)
python llmemctl.py memory test                                    # live toggle test: disable, verify off, re-enable, verify on
```

All changes take effect immediately — no server restart required. The server re-reads `plugins-enabled.json` on every request.

#### Feature flags

All feature flags are live — changes to `plugins-enabled.json` take effect on the next request with no server restart. Configuration lives under `plugin_config.memory`:

```json
"memory": {
  "enabled": true,
  "context_injection": true,
  "reset_summarize": true,
  "post_response_scan": true,
  "fuzzy_dedup": true,
  "vector_search_qdrant": true,
  "fuzzy_dedup_threshold": 0.78,
  "summarizer_model": "summarizer-anthropic",
  "auto_memory_age": true,
  "memory_age_entrycount": 50,
  "memory_age_count_timer": 60,
  "memory_age_trigger_minutes": 2880,
  "memory_age_minutes_timer": 360
}
```

| Feature | What it controls |
|---|---|
| `enabled` | Master switch — when off, all memory features are suppressed regardless of individual flags |
| `context_injection` | Short-term memories injected into every request as `## Active Memory` block |
| `reset_summarize` | Conversation summarized to memory automatically on `!reset` |
| `post_response_scan` | Regex scan of final response text for `memory_save()` calls the LLM narrated instead of calling as a tool |
| `fuzzy_dedup` | Block near-duplicate saves using string similarity (SequenceMatcher) |
| `vector_search_qdrant` | Semantic retrieval via Qdrant + nomic-embed-text. Disable to fall back to keyword-only recall. Named `_qdrant` to leave namespace open for other vector backends. |

#### Settings

| Key | Default | Restart required? | Description |
|---|---|---|---|
| `fuzzy_dedup_threshold` | `0.78` | No | Similarity ratio (0.0–1.0) above which a new save is treated as a duplicate. Read fresh each call. |
| `summarizer_model` | `"summarizer-anthropic"` | No | Model key used by `summarize_and_save()` on `!reset`. Must be a valid key in `llm-models.json`. |
| `auto_memory_age` | `true` | No | Master switch for background aging tasks. When false, both timers are suppressed. |
| `memory_age_entrycount` | `50` | No | Max short-term rows before count-pressure aging removes the overflow. |
| `memory_age_count_timer` | `60` | No | How often (minutes) the count-pressure task runs. Set to `-1` to disable. |
| `memory_age_trigger_minutes` | `2880` | No | Staleness threshold in minutes (2880 = 48h). Rows not accessed in this time are candidates for staleness aging. |
| `memory_age_minutes_timer` | `360` | No | How often (minutes) the staleness task runs. Set to `-1` to disable. |

**Choosing a threshold:** SequenceMatcher ratio on real paraphrased duplicates typically
lands at 0.78–0.88. Genuinely distinct facts on the same topic land below 0.65. The
default of 0.78 catches paraphrase duplicates while allowing distinct new facts. Lower
the threshold (e.g. 0.72) for more aggressive dedup; raise it (e.g. 0.90) to only
block near-identical strings.

#### How deduplication works

`save_memory()` runs two passes before inserting:

1. **Exact match** — blocks if identical `topic + content` exists in either tier (zero DB cost after index scan).
2. **Fuzzy match** — if `fuzzy_dedup` is enabled, loads all existing `content` values for the same topic from both tiers (single `UNION ALL` query) and checks SequenceMatcher ratio against each. Skips insert if any match ≥ threshold.

On any DB or config error in pass 2, the save proceeds rather than blocking.

#### Memory tiers

| Tier | Table | Purpose | Loaded how |
|---|---|---|---|
| Short-term | `*_memory_shortterm` | Hot facts, injected every request | Automatically |
| Long-term | `*_memory_longterm` | Aged-out facts | On-demand via `memory_recall(tier="long")` |
| Archive | Google Drive | Bulk summaries | Future / manual |

Facts age from short-term to long-term via two independent background tasks (see below).

#### Background Aging

Two async tasks run continuously inside the server process. Both re-read config fresh on each cycle — no restart needed for config changes.

**Count-pressure task** (`memory_age_count_timer`, default: every 60 min)

Triggers when the short-term table exceeds `memory_age_entrycount` rows. Moves exactly the overflow rows — those with the lowest importance and oldest `last_accessed` time — to long-term. Has no age threshold; even recently saved rows can be moved if the table is over the cap.

**Staleness task** (`memory_age_minutes_timer`, default: every 360 min)

Moves all rows whose `last_accessed` is older than `memory_age_trigger_minutes` minutes (default: 2880 min = 48h). Safety ceiling: max 200 rows per pass. Also ordered by importance ASC, last_accessed ASC.

**`last_accessed` semantics:** Updated automatically whenever `load_context_block()` injects memories into a request. This means memories that are actively recalled stay in short-term longer. MySQL's `ON UPDATE CURRENT_TIMESTAMP` alone is insufficient — the code issues an explicit `UPDATE` after each injection batch.

**Disabling a timer:** Set the timer to `-1`. The task loop continues running (no restart needed) but skips the aging pass. The other timer remains active independently.

#### Runtime memory commands

```
!memstats                                 show memory system health: DB counts, vector index, retrieval stats, feature flags
!memory                                   list all short-term memories
!memory list [short|long]                 list by tier
!memory show <id> [short|long]            show one row in full
!memory update <id> [tier=short] [importance=N] [content=text] [topic=label]
!membackfill                              embed missing MySQL rows into Qdrant (gap-only, idempotent)
!memreconcile                             remove orphaned Qdrant points not in MySQL
!memreview [approve N,N|reject N,N|clear] AI-assisted topic review with HITL approval
!memage                                   manually trigger one topic-chunk aging pass
!memtrim [N]                              hard-delete N oldest ST rows (no summarization; default: trim to LWM)
```

**`!membackfill`** — compares all MySQL row IDs (both ST and LT) against Qdrant point IDs. Any MySQL rows missing from Qdrant are embedded and upserted. Reports full drift metrics: Qdrant point count, MySQL row count, in-sync count, missing count, and orphan count. Use after restoring a DB backup or if the embedding server was down during saves.

**`!memreconcile`** — the inverse of `!membackfill`. Finds Qdrant points whose IDs don't exist in either MySQL table (orphans created by manual DB deletes or failed aging) and batch-deletes them. Reports: total Qdrant points, MySQL rows, in-sync, orphans found, orphans deleted. Run this after manually deleting rows from `samaritan_memory_shortterm` or `samaritan_memory_longterm`.

**`!memreview`** — AI-assisted topic hygiene review using `reviewer-gemini` (gemini-2.5-flash, temp=0.2). Gathers all topics from both ST and LT with row counts and sample content, then asks the model to propose merge/rename operations. Proposals are displayed with numbered indices for human approval.

Usage workflow:
1. `!memreview` — generate proposals (or show pending ones if they exist)
2. Review the numbered list of merge/rename suggestions with reasons
3. `!memreview approve 1,3` — execute proposals #1 and #3 (updates topic column in both ST and LT tables)
4. `!memreview reject 2` — remove proposal #2 from the pending list
5. `!memreview clear` — discard all pending proposals

Proposals are stored in-memory per session and lost on server restart. The reviewer model is instructed to avoid false merges on shared-prefix topics (e.g. `memory-system` vs `memory-roadmap`).

**`!memstats`** includes a **Retrieval Stats** section (counters since last restart) showing how often single-pass topic-slug retrieval was sufficient vs when two-pass (user-text) retrieval was needed, plus average hit counts for each pass. Also includes a **Config snapshot** section showing the master switch (`enabled (master)`) and each feature flag. Any flag showing `OFF (inactive—master off)` means the master switch is suppressing it.

#### LLM memory tools

Models with the `memory` toolset have access to:

| Tool | Description |
|---|---|
| `memory_save` | Save a fact to short-term memory |
| `memory_recall` | Search short-term or long-term by topic keyword (also matches content) |
| `memory_update` | Update importance, content, or topic on an existing row by id |
| `memory_age` | Manually trigger aging of stale short-term rows to long-term |

Add the `memory` toolset to a model via:
```
!llm_tools add <model> memory
```
Or in `llm-models.json`:
```json
"llm_tools": ["core", "memory"]
```

> For the full memory system design, save paths, topic lifecycle, and change history, see [MEMORY_PROJECT1.md](MEMORY_PROJECT1.md).

---

## API Client Trust Model

**All clients connecting to the API port (`plugin_client_api`, default 8767) are treated as trusted administrators.**

There is no inbound command ACL — any client that can reach the port can send any message, including `!commands`. Tool access is controlled per-model via `llm_tools` in `llm-models.json`, but the human-facing command interface has no restrictions for API clients.

This is intentional. The API port is for trusted orchestrators: other llmem-gw instances, automation scripts, and inter-agent swarms. Inbound restriction is out of scope; network-level controls (firewall, SSH tunnel, VPN) are expected to restrict who can reach the port at all.

### Outbound Agent Message Filters (`OUTBOUND_AGENT_*`)

When an LLM makes an `agent_call` to a remote agent, the outbound message text can be filtered
before it is sent. This is a secondary safeguard for operators who want to restrict what
instructions their agent forwards to other agents.

Configure in `plugins-enabled.json` under `plugin_config.plugin_client_api`:

```json
"plugin_client_api": {
  "OUTBOUND_AGENT_ALLOWED_COMMANDS": [],
  "OUTBOUND_AGENT_BLOCKED_COMMANDS": [
    "rm -rf",
    "shutdown",
    "reboot"
  ]
}
```

**Semantics:**
- `OUTBOUND_AGENT_ALLOWED_COMMANDS`: empty `[]` = all outbound messages permitted (no check).
  Non-empty = message must start with one of the listed prefixes, otherwise blocked.
- `OUTBOUND_AGENT_BLOCKED_COMMANDS`: always checked when non-empty; empty `[]` = nothing blocked.
  Message must not start with any listed prefix.

**Default:** both lists are empty — all agent-to-agent messages are permitted. These filters exist
for operators who want an extra layer of control over what one agent forwards to another.

---

## Runtime Administration (shell.py commands)

Once connected via shell.py, all administration is done via `!commands`.

### Model Management

```
!model                     list available models (current marked)
!model <name>              switch active LLM for this session
```

### Tool Access Management

Tool access is controlled per-model via the `llm_tools` field in `llm-models.json`. Use the unified resource commands to view and manage toolsets:

```
!llm_tools list                         list all models with their tool access
!llm_tools read <model>                 show tools available to a specific model
!llm_tools write <model> all            give model access to all tools
!llm_tools write <model> tool1,tool2    set specific tool list for a model
!llm_tools write <model> none           remove all tool access (text-only)
```

### Hot/Cold Tool Lifecycle (`llm-tools.json`)

Tool registration and associated system prompt sections can be dynamically aged in and out of the active schema on a per-toolset basis. This is configured in `llm-tools.json` — **not** `llm-models.json`.

#### How it works

Each toolset entry in `llm-tools.json` has three lifecycle fields:

| Field | Type | Description |
|---|---|---|
| `always_active` | bool | If `true`, tools are always registered in the LLM schema (never aged out). If `false`, tools participate in the hot/cold lifecycle. |
| `heat_curve` | list of ints | Heat values indexed by call count: `[heat_after_1st_call, heat_after_2nd_call, ..., cap]`. The last value is the cap — all further calls use it. `null` defaults to heat=3. |
| `sp_section` | string or null | Name of the `.system_prompt_<name>` section file that documents this toolset's tools. If set, that system prompt section is **only included** when the toolset is active (hot). |

#### Heat and decay

- A toolset starts **cold** (heat = 0) — its tools are absent from the JSON schema sent to the LLM, and its `sp_section` is excluded from the system prompt.
- When any tool in the toolset is called, the toolset becomes **hot**: heat is set from `heat_curve[call_count - 1]`.
- Each subsequent LLM turn that does **not** use the toolset, heat decays by 1. When heat reaches 0, the toolset goes cold again.
- Cold tools are still *authorized* for the model — the LLM is told about them via a hint line in the system prompt: `Cold (available on demand): tool_a, tool_b`.

Example `heat_curve: [3, 5, 8]`:
- 1st call → heat=3 (stays hot for 3 idle turns)
- 2nd call → heat=5
- 3rd+ call → heat=8 (cap)

#### Toolset object format (v1.1+)

```json
"db": {
  "tools": ["db_query"],
  "always_active": false,
  "heat_curve": [3, 5, 8],
  "sp_section": "tool_db_query"
}
```

Legacy v1.0 format (plain list) is still supported; it is treated as `always_active: true, heat_curve: null, sp_section: null`.

#### Inspecting tool heat at runtime

```
!tools                                  show hot/cold status for all tools in current session
```

#### Adding or editing toolsets

Edit `llm-tools.json` directly, or use the `llm_tools` tool at the shell:

```
!llm_tools list                         list toolsets with always_active/heat status
```

#### System prompt coupling (`sp_section`)

When `sp_section` is set, `prompt.py:load_prompt_for_folder()` automatically includes or excludes the matching `.system_prompt_<name>` section file based on whether the toolset is hot. This means:

- Cold toolset → its documentation is stripped from the system prompt → fewer tokens sent to the LLM.
- Hot toolset → full tool documentation is injected → LLM knows exactly how to use the tool.

Sections whose short name starts with `tool_` but whose toolset is cold are skipped. All other sections are always included.

---

### Tool Call Gates (`llm_tools_gates`)

Tool call gates require a human to approve a specific tool call before the LLM can execute it.
Gates are configured per-model using the `llm_tools_gates` field in `llm-models.json`.

#### Gate entry syntax

| Entry | Effect |
|---|---|
| `db_query` | Gate **all** `db_query` calls |
| `model_cfg write` | Gate only `model_cfg` calls where `action == "write"` |
| `google_drive` | Gate all Google Drive operations |

Multiple entries are comma-separated.

#### Configuring gates at runtime

```
!model_cfg write <model> llm_tools_gates <entry1,entry2,...>
```

Examples:
```
!model_cfg write gemini25f llm_tools_gates db_query
!model_cfg write gemini25f llm_tools_gates db_query,model_cfg write,google_drive
!model_cfg write gemini25f llm_tools_gates        (empty value = clear all gates)
```

Via llmemctl:
```bash
python llmemctl.py model-cfg write gemini25f llm_tools_gates db_query,model_cfg write
```

Or directly in `llm-models.json`:
```json
"gemini25f": {
  "llm_tools_gates": ["db_query", "model_cfg write"]
}
```

#### How gates work by client

| Client | Gate behavior |
|---|---|
| **shell.py** | Shows gate prompt; user types `y`/`yes` to allow, anything else to deny |
| **llama proxy** | Auto-denied immediately; LLM receives denial message |
| **Slack** | Auto-denied immediately; LLM receives denial message |
| **Timeout** | Auto-denied after 120 seconds if no response |

When a gate is pending in shell.py, the status bar changes to `GATE: type y/yes to allow, anything else to deny`. The next input you type is consumed as the gate answer and not sent to the LLM.

#### What the LLM sees on denial

```
GATE DENIED: tool call '<name>' was denied by the user (or timed out).
Do NOT retry the same call. Acknowledge the denial and continue without it.
```

---

### Configuration Management

Five unified resource commands replace the old individual `!commands`:

```
!model_cfg read <model>                 show model configuration
!model_cfg write <model> <field> <val>  update a model config field
!sysprompt_cfg read                     show system prompt
!sysprompt_cfg read <section>           show a specific section
!config_cfg read                        show server configuration
!limits_cfg read                        show depth and rate limits
!limits_cfg write <key> <value>         update a limit value
```

### @model — Per-Turn Model Switch

Prefix any prompt with `@ModelName` to temporarily use a different model for that one turn:

```
@localmodel extract https://www.example.com and summarize it
@gemini25 what is the weather like today?
```

- Result lands in shared session history
- Original model restored after the turn
- Uses the target model's `llm_tools` set for that turn

### Session Management

```
!session                        list all active sessions with shorthand IDs
!session <ID> delete            delete a session (ID = shorthand integer or full ID)
!reset                          clear conversation history for current session
```

#### Session Idle-Timeout Reaper

Sessions that have been inactive longer than the configured timeout are automatically evicted.
The reaper runs every 60 seconds and checks each session's `last_active` timestamp.

```
!sessiontimeout                 show current setting
!sessiontimeout <n>             set timeout to n minutes (runtime only, lost on restart)
!sessiontimeout 0               disable reaping entirely (runtime only)
```

**Default:** 60 minutes. To persist the change across restarts, use `llmemctl`:

```bash
python llmemctl.py session-timeout <minutes>   # 0 = disabled
```

This writes `session_idle_timeout_minutes` to `plugins-enabled.json`.

### Tool Preview Control

Tool results are always sent in full to the LLM. This controls what is displayed in the chat:

```
!tool_preview_length            show current setting (default: 500 chars)
!tool_preview_length <n>        set to n characters
!tool_preview_length 0          unlimited (no truncation)
```

### Agent Streaming Control

```
!stream                 show current agent_call streaming setting (default: enabled)
!stream <true|false>    enable/disable real-time token relay from remote agent
```

> **Note:** The primary node is always the orchestrator. The remote agent responds
> to single messages — it does not itself call agent_call back. Multi-turn conversations
> are conducted by the primary node making repeated agent_call invocations, one per turn.

When enabled (default), remote agent tokens are relayed via push_tok in real-time —
Slack sees each remote turn as it completes rather than as a batch at the end.
Set to `false` to suppress streaming and return only the final result.

---

### PTY Shell Sessions (`plugin_tmux`)

The tmux plugin provides persistent PTY (pseudo-terminal) shell sessions. LLMs interact via
tool calls; humans manage sessions via `!tmux` commands.

> **Advanced users only.** PTY sessions give an LLM direct shell access. Output is captured
> after a silence timeout — long-running commands should be backgrounded with `&` and polled.
> See the "Advanced: PTY Session Semantics" section below.

#### Tool Access

Tmux tool access is controlled per-model via `llm_tools` in `llm-models.json`. Add the
tmux tool names (`tmux_new`, `tmux_exec`, `tmux_ls`, `tmux_history`, `tmux_kill_session`,
`tmux_kill_server`) to a model's `llm_tools` list to grant access, or use `"all"` to
include all tools.

```
!llm_tools read <model>                    show which tools a model can use
!llm_tools write <model> tmux_exec,tmux_ls grant specific tmux tools
```

#### Session Commands

```
!tmux new <name>              — create a new PTY session
!tmux ls                      — list active sessions
!tmux kill-session <name>     — terminate one session
!tmux kill-server             — terminate all sessions
!tmux a <name>                — show session history (attach view)
!tmux history-limit [n]       — show or set rolling history line limit
!tmux filters                 — show current command filter configuration
```

#### Rate Limiting

```
!tmux_call_limit                        — show current rate limit
!tmux_call_limit <calls> <window_secs>  — set rate limit
```

Default: 30 calls per 60 seconds. `auto_disable=true` — on breach, all tmux tools are
disabled until agent restart. Configure base values in `plugins-enabled.json`:

```json
"rate_limits": {
  "tmux": { "calls": 30, "window_seconds": 60, "auto_disable": true }
}
```

#### Command Filtering

Two filter lists in `plugin_config.plugin_tmux` control which commands can be sent to PTY sessions:

```json
"plugin_tmux": {
  "TMUX_ALLOWED_COMMANDS": [],
  "TMUX_BLOCKED_COMMANDS": ["rm -rf", "dd if=", "mkfs", "shutdown", "reboot"]
}
```

**Semantics:**
- `TMUX_ALLOWED_COMMANDS`: empty `[]` = all commands permitted. Non-empty = command must
  match a listed prefix, otherwise blocked.
- `TMUX_BLOCKED_COMMANDS`: always checked; empty `[]` = nothing blocked.

Additionally, `OUTBOUND_AGENT_BLOCKED_COMMANDS` (from `plugin_client_api`) is also applied
inside `tmux_exec` — so the same patterns that block outbound agent messages also block PTY
commands when configured.

#### Security: Shell Access Considerations

> **Critical security consideration.** If a model has tmux tools in its `llm_tools` list, it can execute arbitrary shell commands — including editing `plugins-enabled.json` and `llm-models.json` directly. An LLM that can run `sed` or `python` can rewrite any config file on the host.
>
> **Recommendation:** Only grant tmux tools to models you trust. Use specific tool lists in `llm_tools` rather than `"all"` when tmux access is not needed. Use the command filtering (see below) as an additional safeguard.

#### Advanced: PTY Session Semantics

PTY sessions are true pseudo-terminals with persistent state (cwd, environment, background
jobs). Key behaviors to understand:

- **Output capture:** output is drained after a configurable silence timeout (default 10s).
  If a command produces no output for 10 seconds, the call returns with what was captured so far.
- **Long-running commands:** background with `&` and tee to a log file. Poll with
  `tail logfile` + `jobs` in subsequent `tmux_exec` calls.
- **Credential exposure:** any secrets printed to the terminal (API keys, passwords, tokens)
  will appear in the captured output and be visible to the LLM. Use `.env` files and avoid
  echoing secrets.
- **Prompt injection risk:** malicious content in command output (e.g. from a web response
  stored in a file and cat'd) could attempt to manipulate the LLM. Review outputs before
  passing them back to untrusted LLMs.
- **No interactive prompts:** PTY commands that pause for interactive input (sudo password,
  confirmation prompts) will hang until the timeout expires. Use `-y` flags, `yes |`, or
  pre-configure passwordless sudo for commands the LLM needs to run.

---

### LLM Delegation

The agent supports four modes of LLM delegation — ways for either the user or the LLM itself to
invoke another model. Each mode provides a different level of context isolation and is subject to
different gates and rate limits.

> **Security note:** Tool access is controlled per-model via `llm_tools` in `llm-models.json`.
> A model with access to configuration tools can switch models, reset history, edit system prompts,
> and change limit settings on its own. Only grant the tools you intend to. When an
> `at_llm` or `agent_call` is delegating to another model, that remote model uses its own
> `llm_tools` set — read the depth limit section below before granting broad access.

---

#### Delegation Method Comparison

| Method | User command | Tool call | System prompt | Chat history | Result in history | Controls |
|---|---|---|---|---|---|---|
| `@model` prefix | `@gpt5m <prompt>` | — | ✓ current session | ✓ full | ✓ yes | target model's `llm_tools` |
| `llm_call` (history=caller) | `!llm_call_invoke <model> <prompt> history=caller` | `llm_call(model, prompt, history="caller")` | caller's or target's (sys_prompt=) | ✓ full | ✗ no | target model's `llm_tools`, rate limited, depth guarded |
| `llm_call` (history=none) | `!llm_call_invoke <model> <prompt>` | `llm_call(model, prompt)` | none by default | ✗ none | ✗ no | rate limited |
| `llm_call` (mode=tool) | `!llm_call_invoke <model> <prompt> mode=tool tool=<name>` | `llm_call(model, prompt, mode="tool", tool="name")` | ✗ none (tool schema only) | ✗ none | ✗ no | rate limited |
| `agent_call` | — | `agent_call(agent_url, message)` | ✓ remote instance's own | ✓ remote session's own | ✗ no | rate limited, depth guarded |

---

#### `@model` — Per-Turn Model Switch (user only)

Prefix any message with `@ModelName` to use a different model for that one turn. The target model
uses its own `llm_tools` set. The result is added to shared session history and the original model
is restored afterward.

```
@gpt5m summarize what we discussed so far
@gemini25 what is 2+2
```

---

#### `llm_call` — Unified LLM Delegation (tool + user command)

The single unified delegation tool. Behaviour is controlled by three parameters:

**`mode`** — `"text"` (default) returns a text response; `"tool"` forces a single tool call
(requires `tool=<name>`).

**`sys_prompt`** — `"none"` (default): no system prompt; `"caller"`: calling model's assembled
prompt; `"target"`: target model's own folder prompt.

**`history`** — `"none"` (default): no chat history; `"caller"`: full current session history
(depth-guarded via `max_at_llm_depth`).

```
# Stateless text call (replaces llm_clean_text):
!llm_call_invoke nuc11Local "Summarize: the quick brown fox..."
llm_call(model="nuc11Local", prompt="Summarize the following text: ...")

# Full-context delegation (replaces at_llm):
!llm_call_invoke gpt5m "Review the last tool result." history=caller sys_prompt=caller
llm_call(model="gpt5m", prompt="Review the last tool result.", history="caller", sys_prompt="caller")

# Tool delegation (replaces llm_clean_tool):
!llm_call_invoke nuc11Local "https://example.com summarize" mode=tool tool=url_extract
llm_call(model="nuc11Local", prompt="https://example.com summarize", mode="tool", tool="url_extract")
```

**Rate limited:** default 3 calls per 20 seconds. `history=caller` calls are additionally
depth-guarded via `max_at_llm_depth`.

---

#### `agent_call` — Remote Agent Delegation (tool only)

Sends a single message to another running llmem-gw instance and returns its response. The remote
agent runs with its **own session context** (its own system prompt, its own history) — it receives
only the message string. By default, remote tokens are streamed in real-time via `push_tok`.

Use for: multi-agent swarms, parallel task execution, cross-instance verification.

```
# LLM tool call:
agent_call(agent_url="http://192.168.1.50:8765", message="Search for recent Python 3.13 release notes.")
```

**Rate limited:** default 5 calls per 60 seconds. Subject to `max_agent_call_depth` (see below).

---

### Delegation Depth Limits

Unconstrained delegation chains can grow multiplicatively. Each `at_llm` call gets a **fresh**
tool loop counter (up to `MAX_TOOL_ITERATIONS=10`), so an unbounded chain of depth N could
execute up to 10^N tool calls. `agent_call` chains have the same issue across instances.

Depth limits are enforced via session-stored counters. When the limit is reached, the delegation
is rejected with an instructive message telling the LLM not to retry.

#### `max_at_llm_depth`

Controls how many nested `at_llm` calls are allowed within a single session turn.

- **1 (default):** The LLM can call `at_llm` once, but the called model cannot itself call
  `at_llm` again. No recursion.
- **2+:** Allows chaining — model A calls model B which calls model C. Each hop multiplies the
  maximum tool iterations.
- **0:** Disables `at_llm` entirely (every call is rejected immediately).

> **Warning:** Setting this above 1 allows recursive model chains. With `max_at_llm_depth=3`
> and `MAX_TOOL_ITERATIONS=10`, a worst-case chain could issue 10³ = 1,000 tool calls in a
> single session turn. Keep this at 1 unless you have a specific controlled use case.

#### `max_agent_call_depth`

Controls how many nested `agent_call` hops are allowed from a given session.

- **1 (default):** The orchestrator can call a remote agent, but that remote agent cannot itself
  call `agent_call` back. The remote agent responds directly.
- **2+:** Allows multi-hop swarms (A → B → C). Each hop is a separate instance with its own
  tool loop.
- **0:** Disables `agent_call` entirely.

> **Warning:** Multi-hop swarms are difficult to observe and kill. Remote instances do not share
> your gate state, so a delegated model at hop 2 may operate with fewer constraints than expected.
> Keep this at 1 for controlled swarms where you are the explicit orchestrator.

#### Kill Switch

If a runaway delegation chain occurs, switch models in shell.py:

```
!model <any_model>
```

This calls `cancel_active_task()` which propagates `CancelledError` through all nested `at_llm`
and `agent_call` awaits in the current coroutine chain, terminating the entire tree.

#### Managing Limits via `llmemctl.py`

View and update limits from the command line (requires agent restart to take effect):

```bash
python llmemctl.py limit-list                         # show current values
python llmemctl.py limit-set max_at_llm_depth 1       # set at_llm depth
python llmemctl.py limit-set max_agent_call_depth 1   # set agent_call depth
```

Limits are stored in the `"limits"` section of `llm-models.json`:

```json
"limits": {
  "max_at_llm_depth": 1,
  "max_agent_call_depth": 1
}
```

#### Managing Limits at Runtime (shell.py / tool calls)

View and update limits without restarting the agent. Changes persist to `llm-models.json` but
only affect new sessions after restart.

```
!limits_cfg read                         show all limits with current values
!limits_cfg write max_at_llm_depth 2     set at_llm depth limit
!limits_cfg write max_agent_call_depth 1 set agent_call depth limit
```

These commands are also available as LLM tool calls via the `limits_cfg` tool. Tool access is controlled per-model via `llm_tools`.

### System Prompt

```
!sysprompt_cfg read                     show current assembled system prompt
!sysprompt_cfg read <section>           show section by name (e.g. "tool-url-extract")
!sysprompt reload                       reload all .system_prompt_* files from disk
```

### Direct SQL

```
!db <SQL>                       run SQL directly (no LLM, no gate)
```

---

## System Prompt Administration

The system prompt is split into section files under the project directory. Edit them directly or via the `sysprompt_cfg` tool.

### File structure

```
.system_prompt                      root file (main paragraph + [SECTIONS] list)
.system_prompt_<section-name>       each section file
```

A section file that contains `[SECTIONS]` is a container — it declares child sections. A section file with body text is a leaf.

### Adding a new section

1. Add the entry to the `[SECTIONS]` list in the appropriate parent file:
   ```
   my-new-rule: Description of the rule
   ```
2. Create `.system_prompt_my-new-rule` with the content.
3. Run `!sysprompt reload` in shell.py.

**Loop detection:** If a section name appears in its own ancestor chain, the loader substitutes an error placeholder and logs the error. Duplicate section names across branches are also caught.

### LLM-editable sections

The `sysprompt_cfg` tool can perform surgical edits:
- `append` — add to end of a section
- `prepend` — add to beginning
- `replace` — find exact string and replace
- `delete` — remove lines containing a target string
- `overwrite` — replace entire section (requires `confirm_overwrite=true`)

Container sections cannot be directly edited — edit their child sections instead.

---

## Deployment

### Development (foreground)

```bash
source venv/bin/activate
python llmem-gw.py
```

### Production (systemd)

Create `/etc/systemd/system/llmem-gw.service`:

```ini
[Unit]
Description=Agent MCP
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/projects/llmem-gw
ExecStart=/home/YOUR_USER/projects/llmem-gw/venv/bin/python llmem-gw.py
Restart=on-failure
RestartSec=5
EnvironmentFile=/home/YOUR_USER/projects/llmem-gw/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable llmem-gw
sudo systemctl start llmem-gw
sudo journalctl -u llmem-gw -f
```

### Development (tmux)

```bash
tmux new-session -d -s mcp 'cd /home/YOUR_USER/projects/llmem-gw && source venv/bin/activate && python llmem-gw.py'
tmux attach -t mcp
```

### Remote access via SSH tunnel

The agent can be exposed remotely via an SSH tunnel service (e.g. Pinggy, ngrok, Cloudflare Tunnel).
Configure the tunnel to forward your local port to the remote endpoint, then add the remote host
to `llm-models.json` for any models served via that tunnel.

---

## Configuration Files

| File | Purpose | Edited by |
|---|---|---|
| `.env` | API keys and credentials | Admin manually |
| `plugins-enabled.json` | Which plugins are active + per-plugin config + rate limits | `llmemctl.py` or direct edit |
| `plugin-manifest.json` | Plugin registry — metadata, deps, env vars (read-only) | Plugin authors only |
| `llm-models.json` | Model registry (enabled, model_id, type, etc.) | `llmemctl.py` |
| `.system_prompt` | Root system prompt file | Admin manually or LLM via tool |
| `.system_prompt_*` | Individual section files | Admin manually or LLM via tool |
| `.aiops_session_id` | shell.py session persistence | shell.py automatically |
| `auto-enrich.json` | Instance-specific context auto-enrichment rules (gitignored) | Admin manually — see below |
| `db-config.json` | Instance-specific DB name and table name overrides for memory system (gitignored) | Admin manually |

### `plugin-manifest.json` vs `plugins-enabled.json`

These two files have distinct, non-overlapping roles:

**`plugin-manifest.json` — the plugin catalog (read-only)**

Declares that a plugin *exists* and what it needs to run: its Python file, type,
pip dependencies, required `.env` variables, and load priority.  This file is
maintained by plugin authors and committed to the repo.  The agent and
`llmemctl.py` read it purely for validation — to check whether a plugin's
dependencies and credentials are present before attempting to load it.  You never
edit this file to enable or disable plugins.

**`plugins-enabled.json` — the operator control panel (read/write)**

Determines what actually runs.  It has three jobs:

1. **`enabled_plugins` list** — the ordered list of plugin names the agent will
   attempt to load at startup.  Add a plugin here to activate it; remove it to
   deactivate it entirely.  Managed by `llmemctl.py enable/disable` or by
   direct edit.

2. **`plugin_config` blocks** — per-plugin runtime settings such as port, host,
   and the `enabled` flag.  The `enabled: false` pattern lets you keep a plugin
   in `enabled_plugins` (preserving its config) without starting it.  This is how
   `plugin_proxy_llama` and `plugin_client_slack` ship: configured but off until
   you flip `"enabled": true` or run `llmemctl.py enable <plugin>`.

3. **`rate_limits`** and **`default_model`** — server-wide settings also stored here.

**Practical rule:** to enable or disable a plugin, always use `llmemctl.py`
or edit `plugins-enabled.json`.  Never add enable/disable logic to `plugin-manifest.json`.

**Fresh installs:** `setup-llmem-gw.sh` clones the repo and copies credentials
(`.env`, `credentials.json`, `llm-models.json`) from a reference installation.
It intentionally does *not* copy `plugins-enabled.json` — the repo's version is
the authoritative default for new installs, and port assignments are adjusted
per-instance afterward with `llmemctl.py port-set`.

---

## Context Auto-Enrichment (`auto-enrich.json`)

When a user message arrives, `auto_enrich_context()` in `agents.py` can automatically
query the database and inject the results into the system message — before the LLM ever
sees the request. This gives the LLM instant, grounded context without requiring a tool call.

This is instance-specific configuration: the tables queried and the trigger phrases that
activate them are personal to each deployment and must **not** be hardcoded into the server.
They are stored in `auto-enrich.json` in the project root, which is gitignored.

### Why it exists

Some data is so fundamental to every interaction that waiting for the LLM to decide to
query it wastes a round-trip and tokens. A `person` table with admin identity, a `config`
table with deployment settings, a `contacts` table — these are best injected automatically
when the user's message suggests they are relevant.

### Format

`auto-enrich.json` is a JSON array. Each rule has four fields:

```json
[
  {
    "pattern": "\\b(?:my\\s+(?:details?|info|profile)|who\\s+am\\s+i)\\b",
    "sql": "SELECT * FROM person",
    "label": "person table",
    "enabled": true
  }
]
```

| Field | Required | Description |
|---|---|---|
| `pattern` | Yes | Python regex (case-insensitive) tested against the last user message. If it matches, the SQL is executed. |
| `sql` | Yes | SQL query to run against your MySQL database when the pattern matches. |
| `label` | No | Human-readable label shown in the `[context] Auto-queried: <label>` status message. Defaults to the SQL string if omitted. |
| `enabled` | No | Set to `false` to disable this rule without removing it. Defaults to `true` if omitted. |

Multiple rules are evaluated independently — all matching rules fire for a given message.
Rules with `"enabled": false` are skipped at load time.

### Runtime controls

Auto-enrichment can be suppressed at the session level without editing the JSON file:

```
!config write auto_enrich false    # disable all auto-enrich rules for this session
!config write auto_enrich true     # re-enable (default)
!config list                       # shows current auto_enrich state
```

This is a per-session flag — it does not affect other connected clients and resets when
the session ends. Use `"enabled": false` in the JSON to disable a rule permanently.

### Setup

1. Copy the template:
   ```bash
   cp auto-enrich.json.example auto-enrich.json
   ```
2. Edit `auto-enrich.json` with your patterns and queries.
3. No restart required — rules are loaded fresh on each request.

### Security note

`auto-enrich.json` is gitignored. Do not commit it. It may contain table names, column
names, and trigger patterns that are specific to your deployment and should not be public.

If `auto-enrich.json` is absent, `auto_enrich_context()` skips the enrichment step silently —
the server starts and runs normally without it.

---

## Diagnostics

### Test scripts

All `test_*.sh` scripts verify client protocol compatibility. Run against a live server:

```bash
./test_ollama_client.sh          # Ollama API endpoints
./test_openai_models.sh          # OpenAI /v1/models endpoint
./test_enchanted_app.sh          # Enchanted iOS hybrid endpoints
./test_ios_compatibility.sh      # OpenAI API for iOS apps
./test_openwebui.sh              # open-webui bare paths
./test_llama_proxy.sh            # llama proxy !commands
./test_streaming_role_fix.sh     # first-chunk role field (OpenAI spec)
./test_history_ignore.sh         # server ignores client-sent history
./test_model_ignore.sh           # server ignores client-sent model
./test_model_disable.sh          # disabled models excluded from registry
./test_unknown_commands.sh       # unknown !commands caught
./test_immediate_disconnect.sh   # disconnect handling
./test_win11_response.sh         # local model streaming
```

### Health endpoint

```bash
curl http://localhost:8765/health
```

Returns: server status and loaded models.
