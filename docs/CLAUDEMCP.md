# Claude Code + MCP Direct Integration

Last updated: 2026-04-01

This document describes how Claude Code sessions connect to llmem-gw via the MCP Direct plugin, covering architecture, prerequisites, setup steps, session types, and the hooks that close the loop between Claude's responses and the memory system.

---

## Overview

The MCP Direct plugin (`plugin_mcp_direct.py`) exposes llmem-gw's full data layer as an MCP (Model Context Protocol) server on port 8769. Claude Code connects to it over SSE and gets ~33 tools covering memory, goals, plans, beliefs, database access, Google services, and more.

The result: Claude Code sessions can read and write the same shared memory/goal state as the Python LLM pipeline, Slack dispatch, voice relay, and any other llmem-gw frontend — with no routing layer. Claude IS the reasoning engine; the MCP tools provide persistence and services.

```
Claude Code session (tmux)
        │
        │  SSE  (port 8769)
        ▼
plugin_mcp_direct.py  ←→  MySQL (mymcp database)
                      ←→  Qdrant (semantic search)
                      ←→  Google APIs (Drive, Calendar, Tasks)
                      ←→  External services (weather, places, SMS)
```

---

## Architecture

### Sessions

Three Claude Code sessions run as tmux processes, each with a distinct role:

| Session | tmux name | Model | Purpose |
|---|---|---|---|
| samaritan-work | `samaritan-work` | Default (Opus) | Primary interactive session. Receives Slack/voice dispatch, handles user requests, code work, planning. |
| samaritan-rc | `samaritan-rc` | Default | Remote control channel. Receives tmux-based control commands without voice relay dependency. |
| samaritan-cognition | `samaritan-cognition` | Sonnet 4.6 | Background cognitive processing only. Handles reflection, contradiction detection, and goal health checks triggered by activity events. Never receives user messages. |

All three sessions share the same MCP server, the same MySQL database, and the same Qdrant collections. A goal created in one session is immediately visible in the others.

### How MCP tools reach Claude Code

Claude Code's MCP client connects to the SSE endpoint at startup:
```
http://localhost:8769/sse
```

Tools are registered with `mcp-direct-enable.sh` (runs `claude mcp add --transport sse llmem-gw-direct http://localhost:8769/sse`). After registration, every new Claude Code session in the project directory automatically has the tools available.

Tool calls flow:
```
Claude Code (tool call) → SSE → plugin_mcp_direct.py → MySQL/Qdrant/API → response → Claude Code
```

### Rules system

Each session's behavior is shaped by Markdown rule files in `.claude/rules/`. Claude Code loads all files in this directory at session start. The rules provide:

- **Identity and directives** (`identity.md`) — who Claude is and what governs behavior
- **Available MCP tools** (`mcp-tools.md`) — tool reference and when to use each
- **Workflow patterns** (`workflow-patterns.md`) — how to start a session, complete steps, create goals
- **Session-specific behavior** (`cognition-session.md`, `remote-control.md`, `voice-relay.md`) — instructions specific to each session type
- **Memory boundaries** (`memory-reconciliation.md`) — what goes in Claude Code local memory vs. llmem-gw
- **Restart continuity** (`restart-continuity.md`, `server-restart.md`) — how to write handoff files before restarts

### Stop hook — conversation logging

Every Claude Code session has a `Stop` hook that fires after each assistant response:

```json
"Stop": [{
  "hooks": [{
    "type": "command",
    "command": "python3 /path/to/.claude/hooks/conv_log.py"
  }]
}]
```

`conv_log.py` reads the session transcript, extracts the last user+assistant exchange, strips emotion tags (see below), and POSTs to `http://localhost:8769/conv_log`. The server saves the turn to short-term memory, increments the cognition turn counter, and optionally triggers a reflection step.

This means every conversation turn is automatically persisted to llmem-gw — no explicit `memory_save` calls needed for routine conversation.

### Live emotion tagging

The `live-emotion.md` rule instructs Claude to append an invisible emotion tag to every response:
```
<<emotion:user:hopeful:0.6:happy>>
```

The `conv_log.py` hook strips this tag before saving the assistant text and forwards it to the server as metadata. The server writes it to `samaritan_emotions` linked to the memory entry.

---

## Prerequisites

### Hard requirements

- **llmem-gw running** with `plugin_mcp_direct` enabled. Verify:
  ```bash
  curl -s http://localhost:8769/health
  ```

- **Claude Code CLI** installed and in PATH:
  ```bash
  claude --version
  ```

- **tmux** installed:
  ```bash
  tmux -V
  ```

- **Python 3** (for the conv_log hook — stdlib only, no pip installs needed)

- **MySQL** — the `mymcp` database with all migrations applied. Check:
  ```bash
  mysql mymcp -e "SHOW TABLES LIKE 'samaritan_%';"
  ```

- **MCP Direct registered** with Claude Code:
  ```bash
  claude mcp list  # should show llmem-gw-direct
  ```

### For the cognition session specifically

- `samaritan_plans.target` ENUM must include `'claude-cognition'` (see setup step 3 below)
- `cognition-start.sh` present and executable
- `.claude/rules/cognition-session.md` present in the project rules directory

---

## Setup — New Installation

### 1. Register the MCP server with Claude Code

```bash
cd /path/to/llmem-gw
bash mcp-direct-enable.sh
```

This runs:
```bash
claude mcp add --transport sse llmem-gw-direct http://localhost:8769/sse
```

Registration is stored in `~/.claude/claude_desktop_config.json` (user scope). All Claude Code sessions on the machine will have access after this.

Verify:
```bash
claude mcp list
# should show: llmem-gw-direct  http://localhost:8769/sse
```

### 2. Configure settings.json

Create `.claude/settings.json` in your workspace:

```json
{
  "permissions": {
    "allow": [
      "mcp__llmem-gw-direct",
      "WebFetch",
      "WebSearch",
      "Read",
      "Edit",
      "Write",
      "Bash",
      "Glob",
      "Grep"
    ]
  },
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/.claude/hooks/conv_log.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

Adjust the `allow` list for your security requirements. The `mcp__llmem-gw-direct` entry auto-approves all tool calls from that server without prompting.

### 3. Apply the cognition session DB migration

The `samaritan_plans.target` column needs `claude-cognition` as a valid value:

```sql
ALTER TABLE mymcp.samaritan_plans
  MODIFY COLUMN target
  ENUM('model','human','investigate','claude-code','claude-cognition')
  NOT NULL DEFAULT 'model';
```

If you skip this step, cognition step inserts will silently fail with a data truncation warning.

### 4. Create the rules directory and copy rule files

```bash
mkdir -p /path/to/workspace/.claude/rules
# Copy all .md files from the rules directory in this repo
```

At minimum for a functioning setup:
- `identity.md` — prime directives and persona
- `mcp-tools.md` — tool reference
- `workflow-patterns.md` — session start protocol
- `memory-reconciliation.md` — memory system boundaries
- `live-emotion.md` — emotion inference tagging

For the cognition session:
- `cognition-session.md` — step processing rules

### 5. Install the conv_log hook

Copy `.claude/hooks/conv_log.py` to your project hooks directory and ensure the path in `settings.json` matches.

Set the MCP endpoint if your server runs on a non-default port:
```bash
export MCP_DIRECT_URL=http://localhost:8769
```

### 6. Start the sessions

**First start** (llmem-gw not yet running):
```bash
cd /path/to/llmem-gw && bash llmemrestart.sh
```

**With sessions** (llmem-gw already running):
```bash
bash claude-start.sh   # starts samaritan-work + samaritan-cognition
bash rc-start.sh       # starts samaritan-rc
```

**Restart everything** (mid-session server restart):
```bash
nohup bash restart-llmem.sh &
```
This restarts llmem-gw and cold-starts all managed Claude Code sessions with fresh MCP connections.

### 7. Verify

```bash
# Check all sessions are running
tmux list-sessions

# Send a test message to samaritan-work
tmux send-keys -t samaritan-work "What MCP tools do you have available?" Enter

# Check cognition session is receiving pokes
tmux attach -t samaritan-cognition
# (after a memory_save, should see "Process pending cognition steps")
```

---

## Session Startup Protocol

Every Claude Code session should call these two tools at the start of each conversation, before doing anything else (enforced by `workflow-patterns.md`):

```python
load_context()          # pull active goals, beliefs, drives, recent memories, procedures
steps_for_claude_code() # check if autonomous system queued work for this session
```

Without `load_context()`, the session has no knowledge of what other sessions or the autonomous system have been doing. Without `steps_for_claude_code()`, queued work goes unexecuted.

---

## Context Loading

`load_context(query, include_typed, include_procedures, include_temporal)` mirrors the `auto_enrich_context()` function that runs before every LLM call in the Python pipeline. It returns:

- **Recent short-term memory** (keyword + semantic search on `query`)
- **Active goals** (sorted by drive-weighted priority)
- **Active beliefs** (topic matches)
- **Drive values** (current affect state)
- **Active conditioned behaviors**
- **Relevant procedures** (if `include_procedures=True`)
- **Temporal patterns** (if `include_temporal=True`)

Call `load_context(query="specific topic")` when switching topics mid-session to refresh relevant memories for that domain.

---

## Tool Reference (Summary)

Full tool documentation is in `docs/mcp-tools.md` or available via the tools themselves. Quick reference by category:

| Category | Tools |
|---|---|
| Context | `load_context`, `cogn_status`, `cogn_control` |
| Memory | `memory_save`, `memory_recall`, `memory_search_semantic`, `memory_update` |
| Goals | `goal_create`, `goal_update`, `goal_list` |
| Plans | `step_create`, `step_update`, `step_list`, `plan_decompose`, `plan_check_completion`, `steps_for_claude_code`, `steps_for_cognition` |
| Typed memory | `assert_belief`, `save_memory_typed`, `set_conditioned`, `recall_temporal`, `procedure_save`, `procedure_recall` |
| Data | `db_query`, `google_drive`, `google_calendar`, `weather`, `places` |
| File | `file_extract` |
| Search | `perplexity_search`, `xai_search` |
| Cross-system | `llm_call`, `llm_list`, `sms_send` |
| Cognition | `steps_for_cognition` |

---

## Session Restart and Continuity

When llmem-gw restarts, all MCP connections become stale. Tool calls return `-32602` errors until the session is cold-started with a fresh connection.

**Never call `llmemrestart.sh` directly** from a running session. Always use the orchestrated wrapper:
```bash
nohup bash restart-llmem.sh &
```

`restart-llmem.sh`:
1. Checks which sessions were active
2. Restarts llmem-gw
3. Waits for health
4. Cold-starts each managed session (kills tmux session, relaunches Claude Code)
5. Recovers voice relay if it was active
6. Reloads VS Code windows

### Post-restart continuity

If a session has in-progress work when a restart is triggered, write a `.post-restart-instructions.<session-name>` file before issuing the restart command. `restart-llmem.sh` / `rc-start.sh` / `cognition-start.sh` deliver this file to the new session via `_deliver_instructions()` after cold-start. The new session reads it and picks up where the previous one left off.

File naming:
- `samaritan-work` session → `.post-restart-instructions.samaritan-work`
- `samaritan-rc` session → `.post-restart-instructions.samaritan-rc`
- `samaritan-cognition` session → `.post-restart-instructions.samaritan-cognition`

---

## Memory System Boundaries

Two memory systems coexist. Use the right one:

| Use case | System |
|---|---|
| User preferences, feedback, how to work with this user | Claude Code local memory (`~/.claude/projects/.../memory/`) |
| Project conventions specific to Claude Code work | Claude Code local memory |
| Shared facts, world state, events | llmem-gw (`memory_save`) |
| Goals and plans | llmem-gw (`goal_create`, `step_create`) |
| Beliefs with confidence | llmem-gw (`assert_belief`) |
| Procedures reusable by any executor | llmem-gw (`procedure_save`) |
| Anything samaritan-reasoning or Slack should also see | llmem-gw |

**Do not duplicate.** The `memory-reconciliation.md` rule enforces this. Automatic conversation logging (`conv_log.py` Stop hook) already captures all turns — calling `memory_save` for routine conversation creates duplicates.

---

## Troubleshooting

**Tool calls return `-32602`**
MCP session is stale. Restart the session: `bash claude-start.sh` (or the appropriate session script).

**Cognition steps not being queued after `memory_save`**
Check `samaritan_plans.target` ENUM — `'claude-cognition'` must be a valid value. See setup step 3.

**samaritan-cognition not waking on pokes**
Check the tmux session exists: `tmux list-sessions`. Check llmem-gw log for `cogn_poke` entries. Pokes are debounced 60s — wait and try again.

**`load_context` returns empty results**
llmem-gw may not have `plugin_mcp_direct` enabled. Check `plugins-enabled.json` and restart llmem-gw.

**conv_log hook not firing**
Verify `settings.json` has the Stop hook configured and the path to `conv_log.py` is correct. Check `~/.claude/logs/` for hook errors.
