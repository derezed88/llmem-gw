# Swarm Design Guide

This document describes the llmem-gw swarm foundation and the design space for extending it into a full multi-agent coordination system. The foundation is intentionally minimal — the transport and session plumbing are in place, but no discovery or naming scheme is built in. This leaves every topology option open for operators and contributors to implement as their use case demands.

---

## What Exists Today

The following is already implemented and working:

**`plugin_client_api`** — each llmem-gw instance can expose a JSON/SSE HTTP API (port 8767 by default). Any process that can make HTTP requests can submit messages and receive streaming responses. No special protocol beyond HTTP/SSE.

**`agent_call(agent_url, message)`** — a core tool available to every LLM on every instance. The LLM calls another agent by URL, the remote agent processes the message through its full stack (LLM, tools, gates), and returns the complete response as a tool result. The calling LLM sees the result in its tool context and continues reasoning.

**Persistent swarm sessions** — the remote session ID is derived deterministically from `md5(calling_session_id + ":" + agent_url)`. Repeated calls from the same human session to the same remote agent always reuse the same remote session. The remote agent accumulates conversation history across calls, exactly like a human user's session.

**Depth guard** — sessions with `client_id` prefix `api-swarm-` cannot initiate further `agent_call` calls. This caps recursion at one hop. A calling agent can reach a target agent; the target agent cannot turn around and call further agents. This is a safety default, not a permanent architectural limit.

**`api_client.py`** — a standalone async Python client library for the API plugin. Usable directly by scripts, automation, or other frameworks. `agent_call` uses it internally.

**Gate handling** — API and swarm sessions see gate events on the SSE stream. Direct API clients with `auto_approve_gates` configured can respond within milliseconds. Swarm sessions auto-reject gates (same as llama proxy clients) — the LLM on the remote side receives rejection feedback and can work around it.

**Rate limiting** — `agent_call` calls are rate-limited per the `agent_call` entry in `plugins-enabled.json` (default: 5 calls / 60 s).

What is deliberately **not** implemented: agent discovery, naming, registration, or any topology convention. Those are open design choices described below.

---

## Network Topology Considerations

Before choosing a discovery scheme, it is worth thinking about what network topologies are realistic.

**Same machine, multiple instances** — the simplest case. All instances share localhost, each on its own port. URLs are `http://localhost:<port>`. No NAT, no firewall, no latency. Good for development and testing.

**Same LAN, multiple machines** — instances reach each other by IP. URLs are `http://192.168.x.x:<port>`. Works as long as the firewall on each machine allows the API port. No NAT problem because machines are on the same network segment.

**Internet-reachable node (EC2, VPS, etc.)** — an instance with a public IP and an open port is reachable by any node anywhere. This creates an immediately useful topology: NAT-constrained nodes (home networks, laptops) can reach the public node, even if they cannot reach each other directly.

**NAT-constrained nodes** — a node behind NAT can make outbound connections but cannot accept inbound connections from the internet. It can `agent_call` an internet-reachable node, but cannot be called back directly. This asymmetry matters for hub designs.

**SSH tunnels and reverse proxies** — a NAT-constrained node can open an SSH tunnel or use a service like Pinggy, ngrok, or Cloudflare Tunnel to expose its API port to the internet. The tunnel gives the node a public URL it can register with peers. This is already used in this project for exposing local llama.cpp models.

The topology choice drives the discovery design. A flat mesh needs every node to be mutually reachable. A hub model only requires every node to reach the hub.

---

## Option 1: System Prompt Registry (Static, No Code)

Store known peer names and URLs in a system prompt section. The LLM reads them at context time and resolves names to URLs when constructing `agent_call` arguments.

**Setup:**

Add a section to the system prompt:

```
!sysprompt
```

Then create `.system_prompt_swarm-peers` with content like:

```
Known swarm peers:
  NUC11:    http://192.168.x.x:8767
  EC2-Hub:  http://54.210.x.x:8767
  Agent-B:  http://localhost:8777
```

Register it in the parent section's `[SECTIONS]` list and reload:

```
!sysprompt reload
```

**Usage:**

```
Ask the NUC11 agent to run !model
```

The LLM sees the peer list in its context, resolves "NUC11" to the URL, and calls `agent_call("http://192.168.x.x:8767", "!model")`.

**Properties:**
- Zero new code
- Persists across restarts (files on disk)
- Editable live via `update_system_prompt` tool (if write gate is open)
- LLM must reason from the context — works well with capable models, may need explicit phrasing for weaker ones
- Static — you update the file when peers change

**Best for:** Small, stable topologies where the operator knows all peers at setup time.

---

## Option 2: Database Registry (Dynamic, Shared)

Store peer registrations in a MySQL table. All agents that share the same database share the same registry. Agents can register themselves on startup and heartbeat to signal liveness.

**Schema:**

```sql
CREATE TABLE swarm_agents (
    name         VARCHAR(64) PRIMARY KEY,
    url          VARCHAR(255) NOT NULL,
    model        VARCHAR(64),
    description  TEXT,
    last_seen    DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

**Startup registration** (could be a startup hook or a shell script):

```bash
mysql -u $MYSQL_USER -p$MYSQL_PASS llmem_gw \
  -e "INSERT INTO swarm_agents (name, url, model, description)
      VALUES ('NUC11', 'http://192.168.x.x:8767', 'qwen25', 'Local 7B model')
      ON DUPLICATE KEY UPDATE url=VALUES(url), last_seen=NOW();"
```

**LLM usage** — the LLM discovers peers by querying the table before calling `agent_call`:

```
Look up the available agents in the swarm_agents table, then ask the one
named NUC11 to summarise the latest project notes.
```

**Liveness check** — agents can be considered stale if `last_seen` is older than a threshold:

```sql
SELECT name, url FROM swarm_agents WHERE last_seen > NOW() - INTERVAL 5 MINUTE;
```

**Properties:**
- Dynamic — agents register themselves without operator intervention
- Shared state across all nodes on the same DB
- Liveness signal via `last_seen`
- Adds a two-step LLM reasoning path (query DB, then call agent)
- Couples all nodes to the database — DB must be reachable by all participants
- Survives restarts (data in DB, not memory)

**Best for:** Topologies where multiple agents share a MySQL instance and dynamic registration matters more than simplicity.

---

## Option 3: API Registration Endpoint (Dynamic, Decentralised)

Add `POST /api/v1/register` and `GET /api/v1/agents` endpoints to `plugin_client_api`. Agents call `/register` on a hub or peer at startup to announce themselves. The recipient stores the registration in memory (or DB for persistence). `/agents` returns the current registry.

**Registration payload:**

```json
{
  "name": "NUC11",
  "url": "http://192.168.x.x:8767",
  "model": "qwen25",
  "description": "Local 7B model node"
}
```

**Hub topology:** One designated instance (e.g. the EC2 node) acts as the registry. All other nodes register with it on startup. Any node can query the hub's `/agents` to discover peers.

```
Agent A                    Hub (EC2)               Agent B
   │                          │                        │
   │  POST /api/v1/register   │                        │
   │─────────────────────────►│                        │
   │                          │  POST /api/v1/register │
   │                          │◄───────────────────────│
   │  GET /api/v1/agents      │                        │
   │─────────────────────────►│                        │
   │◄── [{name,url}, ...]  ───│                        │
   │                          │                        │
   │  agent_call("Agent B")   │                        │
   │─────────────────────────────────────────────────►│
```

**Mesh topology:** Every node registers with every other known node directly. No hub needed but requires all nodes to be mutually reachable (no NAT problem).

**Properties:**
- No DB dependency for in-memory registries
- Hub topology works well for NAT-constrained nodes (all make outbound connections to the hub)
- In-memory registry resets on hub restart unless persisted to DB or Drive
- Registration could include a heartbeat TTL — entries expire if not refreshed
- `agent_call` could accept a name instead of a URL if it queries `/agents` internally

**Best for:** Topologies with a stable, internet-reachable hub node and NAT-constrained leaf nodes.

---

## Option 4: Shell Command Registry (`!agent`)

A new `!agent` command that manages a local name→URL mapping file (`swarm-agents.json`). Simple, operator-controlled, no DB or network dependency.

**Commands:**

```
!agent list                          list known agents
!agent add <name> <url>             add or update an agent
!agent remove <name>                remove an agent
!agent call <name> <message>        send message to named agent
```

The registry is persisted to `swarm-agents.json` in the instance directory, so it survives restarts. Each instance maintains its own independent list.

`!agent call NUC11 !model` expands to `agent_call("http://192.168.x.x:8767", "!model")` internally, removing the need to know or type the URL.

Optionally, the `!agent list` output could be injected into the system prompt at context-build time so the LLM can also resolve names in natural language without explicit `!agent call` syntax.

**Properties:**
- Simplest operator experience — one command to add a peer, one to use it
- No infrastructure dependencies
- Per-instance — doesn't propagate to other agents automatically
- Could be combined with Option 1 (inject the list into the system prompt)

**Best for:** Operators who want a clean command-line interface and don't need dynamic registration.

---

## Hybrid and Extended Topologies

These options are not mutually exclusive. A realistic deployment might combine them:

- **Static system prompt** for well-known stable peers + **DB registration** for ephemeral or dynamically provisioned agents
- **Option 4 shell command** as the operator interface, with Option 3 registration happening in the background at startup
- **Hub node on EC2** that all NAT-constrained nodes register with; NAT nodes use the hub's `/agents` endpoint to discover each other, then communicate directly if both have tunnels, or relay through the hub otherwise

**Database-brokered relay** — for nodes that cannot reach each other directly (both behind NAT, no tunnels), the shared database can act as a message broker. Agent A writes a message to a `swarm_messages` table, Agent B polls and reads it. This is not the fastest path — it adds DB round-trip latency and requires polling — but it works through any NAT without any tunnel configuration. A `swarm_messages` table with `sender`, `recipient`, `message`, `response`, `status` columns is the entire implementation.

**EC2 hub as relay** — rather than or in addition to DB brokering, an EC2 hub with a public IP can forward messages between NAT-constrained nodes. Node A sends to hub, hub calls Node B (if B registered a tunnel URL with the hub), hub returns response to A. This requires Node B to have exposed its API port via a tunnel, but only Node B needs to do that — Node A only needs outbound access to the hub.

---

## Current Limits and Known Gaps

| Limit | Current behaviour | Design note |
|---|---|---|
| Recursion depth | 1 hop (hard-coded prefix check) | Could be a configurable depth counter passed as a header or in the swarm session metadata |
| Discovery | None — URL must be explicit | Any of the four options above |
| Gate approval in swarm | Auto-reject | Swarm clients cannot interact with the human; an approval-by-message-passing scheme via DB or hub relay would be needed for gated tools |
| Liveness | No health checking | Callers receive a timeout error; routing around dead nodes requires caller-side retry logic |
| Authentication | Optional single shared `API_KEY` | Per-node keys, JWT, or mTLS for production multi-tenant deployments |
| Message size | No limit enforced | Long responses may be slow over tunnels; chunked streaming is already the default |

---

## Getting Started Today

Without implementing any discovery scheme, the swarm works right now by specifying URLs directly:

```
Use agent_call to ask the agent at http://192.168.x.x:8767 to run !model
```

The quickest ergonomic improvement with no code is Option 1 — add a `.system_prompt_swarm-peers` file with your known agents and reload. The LLM will resolve names from context for the rest of the session.

See [docs/ADMINISTRATION.md](ADMINISTRATION.md#swarm--multi-agent-setup) for setup steps and [docs/plugin-client-api.md](plugin-client-api.md) for the API reference.
