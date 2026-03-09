import asyncio
import json
import importlib
import os
from typing import AsyncGenerator
from starlette.requests import Request
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from config import log, LLM_REGISTRY, DEFAULT_MODEL, LLM_MODELS_FILE
from state import sessions, get_queue, push_tok, push_done, push_model, active_tasks, cancel_active_task
from database import execute_sql, set_model_context
from agents import dispatch_llm
from tools import get_tool_executor, get_plugin_command, get_plugin_help_sections

# ---------------------------------------------------------------------------
# History plugin chain
# ---------------------------------------------------------------------------
_PLUGINS_ENABLED_PATH = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")

def _load_history_chain() -> list:
    """Load and return the ordered list of history plugin module objects."""
    try:
        with open(_PLUGINS_ENABLED_PATH) as f:
            cfg = json.load(f)
        chain_names = (
            cfg.get("plugin_config", {})
               .get("plugin_history_default", {})
               .get("chain", ["plugin_history_default"])
        )
    except Exception:
        chain_names = ["plugin_history_default"]

    chain = []
    for name in chain_names:
        try:
            mod = importlib.import_module(name)
            chain.append(mod)
        except ImportError as e:
            log.warning(f"History chain: cannot load '{name}': {e}")
    if not chain:
        # Always fall back to default to prevent unguarded raw appends
        import plugin_history_default as _phd
        chain = [_phd]
    return chain

_history_chain: list = _load_history_chain()

def _run_history_chain(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    """Pass history through all plugins in chain order. Returns final list."""
    for plugin in _history_chain:
        history = plugin.process(history, session, model_cfg)
    return history

def _notify_chain_model_switch(session: dict, old_model: str, new_model: str,
                                old_cfg: dict, new_cfg: dict) -> list[dict]:
    """Notify each chain plugin of a model switch; return final history."""
    history = list(session.get("history", []))
    for plugin in _history_chain:
        if hasattr(plugin, "on_model_switch"):
            history = plugin.on_model_switch(session, old_model, new_model, old_cfg, new_cfg)
    return history

def _load_system_int(key: str, default: int) -> int:
    """Read a top-level integer from plugins-enabled.json."""
    try:
        with open(_PLUGINS_ENABLED_PATH) as f:
            return int(json.load(f).get(key, default))
    except Exception:
        return default

# Runtime overrides (survive until restart)
_runtime_max_users: int | None = None
_runtime_session_idle_timeout: int | None = None
_runtime_tool_preview_length: int | None = None
_runtime_tool_suppress: bool | None = None

def get_max_users() -> int:
    if _runtime_max_users is not None:
        return _runtime_max_users
    return _load_system_int("max_users", 50)

def get_session_idle_timeout() -> int:
    if _runtime_session_idle_timeout is not None:
        return _runtime_session_idle_timeout
    return _load_system_int("session_idle_timeout_minutes", 60)

def get_default_tool_preview_length() -> int:
    if _runtime_tool_preview_length is not None:
        return _runtime_tool_preview_length
    return _load_system_int("tool_preview_length", 500)

def get_default_tool_suppress() -> bool:
    if _runtime_tool_suppress is not None:
        return _runtime_tool_suppress
    try:
        with open(_PLUGINS_ENABLED_PATH) as f:
            return bool(json.load(f).get("tool_suppress", False))
    except Exception:
        return False

# Context flag for batch command processing
_batch_mode = {}  # client_id -> bool

async def conditional_push_done(client_id: str):
    """Only call push_done if not in batch mode"""
    if not _batch_mode.get(client_id, False):
        await push_done(client_id)

async def cmd_help(client_id: str):
    help_text = (
        "Available commands:\n"
        "  !model                                    - list available models (current marked)\n"
        "  !model <key>                              - switch active LLM\n"
        "  @<model> <prompt>                         - one-turn model switch (e.g. @gpt5m explain this)\n"
        "  !stop                                     - interrupt the running LLM job\n"
        "  !reset                                    - clear conversation history\n"
        "  !help                                     - this help\n"
        "  !input_lines <n>                          - resize input area (client-side only)\n"
        "  !tools                                    - show tool heat status (hot/cold) for current model\n"
        "  !tools reset                              - decay all tool subscriptions to cold\n"
        "  !tools hot <toolset,...>                  - force-subscribe toolsets immediately\n"
        "\n"
        "Database:\n"
        "  !db_query <sql>                           - run SQL directly (no LLM)\n"
        "\n"
        "Memory:\n"
        "  !memory [true|false]                      - enable/disable memory logging (MySQL+Qdrant)\n"
        "  !memory list [short|long]                 - list by tier\n"
        "  !memory show <id> [short|long]            - show one row in full\n"
        "  !memory update <id> [tier=short] [importance=N] [content=text] [topic=label]\n"
        "  !memstats                                 - memory system health dashboard\n"
        "  !membackfill                              - embed missing rows into Qdrant vector index\n"
        "  !memreconcile                             - remove orphaned Qdrant points not in MySQL\n"
        "  !memreview [approve N,N|reject N,N|clear] - AI topic review with HITL approval\n"
        "  !memage                                   - manually trigger one topic-chunk aging pass\n"
        "  !memtrim [N]                              - hard-delete N oldest ST rows (escape valve; default: trim to LWM)\n"
        "\n"
        "Search & Extract:\n"
        "  !search_ddgs <query>                      - search via DuckDuckGo\n"
        "  !search_google <query>                    - search via Google (Gemini grounding)\n"
        "  !search_tavily <query>                    - search via Tavily AI\n"
        "  !search_xai <query>                       - search via xAI Grok\n"
        "  !url_extract <url> [query]                - extract web page content\n"
        "\n"
        "Google Drive:\n"
        "  !google_drive <operation> [args...]       - Drive CRUD operation\n"
        "\n"
        "VSCode Sessions:\n"
        "  !vscode list [date=YYYY-MM-DD] [project=name]   - list local Claude Code sessions\n"
        "  !vscode read <id>[,<id>...] [mode=full|summary] [model=<key>]  - pull session into context\n"
        "\n"
        + "".join(get_plugin_help_sections())
        + "\n"
        "Session & History:\n"
        "  !session                                  - list all active sessions\n"
        "  !session <ID> delete                      - delete a session from server\n"
        "\n"
        "Utilities:\n"
        "  !get_system_info                          - show date/time/status\n"
        "  !llm_list                                 - list LLM models\n"
        "  !llm_call_invoke <model> <prompt> [opts]  - call a model directly (see options below)\n"
        "    opts: mode=text|tool  sys_prompt=none|caller|target  history=none|caller  tool=<name>\n"
        "  !sleep <seconds>                          - sleep 1-300 seconds\n"
        "\n"
        "Unified Resource Commands:\n"
        "  !llm_tools [list|read|write|delete|add] [name] [tools]\n"
        "    list                                     - show all toolsets and model assignments\n"
        "    read <name>                              - show one toolset\n"
        "    write <name> <tool1,tool2,...>            - overwrite toolset\n"
        "    delete <name>                            - remove toolset\n"
        "    add <name> <tool1,tool2,...>              - add tools to toolset\n"
        "  !model_cfg [list|read|write|copy|delete|enable|disable] [name] [field] [value]\n"
        "    list                                     - show all models with toolsets\n"
        "    read <model>                             - show full model config\n"
        "    write <model> <field> <value>            - set a model field\n"
        "    copy <source> <new_name>                 - copy model\n"
        "    delete <model>                           - delete model\n"
        "    enable/disable <model>                   - enable/disable model\n"
        "  !model_cfg write <model> llm_tools_gates <entry1,entry2,...>\n"
        "    Configure tool call gates requiring human approval before execution.\n"
        "    Each entry is either a tool name or 'tool_name subcommand'.\n"
        "    Examples:\n"
        "      !model_cfg write gemini25f llm_tools_gates db_query\n"
        "      !model_cfg write gemini25f llm_tools_gates db_query,model_cfg write\n"
        "      !model_cfg write gemini25f llm_tools_gates   (empty = clear all gates)\n"
        "    Gate behavior by client:\n"
        "      shell.py : prompts user; next input is Y/N\n"
        "      llama/slack : auto-denied (cannot prompt interactively)\n"
        "      timeout 120s: auto-denied if no response\n"
        "  !sysprompt_cfg [list_dir|list|read|write|delete|copy_dir|set_dir] [model] [file|dir] [content]\n"
        "  !config [list|read|write] [key] [value]\n"
        "    list                                     - show all config settings\n"
        "    read <key>                               - show one setting\n"
        "    write <key> <value>                      - set a setting\n"
        "  !limits [list|read|write] [key] [value]\n"
        "    list                                     - show all depth/rate limits\n"
        "    read <key>                               - show one limit\n"
        "    write <key> <value>                      - set a limit\n"
        "\n"
        "Judge (LLM-as-judge enforcement):\n"
        "  !judge                                    - show judge config for current model/session\n"
        "  !judge list                               - list judge configs for all models\n"
        "  !judge on <gate|all>                      - enable gate (prompt/response/tool/memory/all)\n"
        "  !judge off <gate|all>                     - disable gate for this session\n"
        "  !judge model <name>                       - set judge model for this session\n"
        "  !judge mode block|warn                    - block (deny) or warn (log+allow)\n"
        "  !judge threshold <0.0-1.0>                - set score threshold\n"
        "  !judge reset                              - clear all session overrides\n"
        "  !judge test <text>                        - run ad-hoc evaluation\n"
        "  !judge set <model> <field> <value>        - persist judge_config to llm-models.json\n"
        "    fields: model, mode, threshold, gates\n"
        "  Note: enable plugin_history_judge in chain for tool+memory gates to be active.\n"
        "\n"
        "AI tools:\n"
        "  get_system_info()                         - show date/time/status\n"
        "  llm_call(model, prompt, ...)              - call a model (mode/sys_prompt/history/tool)\n"
        "  llm_list()                                - list LLM models\n"
        "  agent_call(agent_url, message)            - send message to remote agent\n"
        "  sleep(seconds)                            - pause 1-300s\n"
        "  session/model/reset                       - session management\n"
        "  llm_tools/model_cfg/sysprompt_cfg/config_cfg/limits_cfg - resource management\n"
    )
    await push_tok(client_id, help_text)
    await conditional_push_done(client_id)

async def cmd_db_query(client_id: str, sql: str, model_key: str = ""):
    """Execute SQL directly without LLM."""
    sql = sql.strip()
    if not sql:
        await push_tok(client_id,
            "Usage: !db_query <SQL>\n"
            "Examples:\n"
            "  !db_query SELECT * FROM users LIMIT 5\n"
            "  !db_query SHOW TABLES")
        await conditional_push_done(client_id)
        return
    set_model_context(model_key)
    try:
        result = await execute_sql(sql)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: Database query failed\n{exc}")
    await conditional_push_done(client_id)


async def cmd_memory(client_id: str, arg: str):
    """
    !memory                         - list all short-term memories
    !memory list [short|long]       - list short or long-term memories
    !memory show <id> [short|long]  - show one row in full
    !memory update <id> [tier=short] [importance=N] [content=...] [topic=...]
    """
    from memory import load_short_term, load_long_term, update_memory, _mem_plugin_cfg
    parts = arg.split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if subcmd in ("list", "short", "long", ""):
        tier = "long" if subcmd == "long" else "short"
        if subcmd == "list" and rest in ("long", "short"):
            tier = rest
        _mem_flag = sessions[client_id].get("memory_enabled", None)
        _global_mem = _mem_plugin_cfg().get("enabled", True) if _mem_flag is None else None
        _mem_active = _mem_flag if _mem_flag is not None else _global_mem
        _mem_src = "session" if _mem_flag is not None else "global"
        await push_tok(client_id, f"Memory logging: {'ON' if _mem_active else 'OFF'} ({_mem_src})")
        rows = await (load_long_term(limit=100) if tier == "long" else load_short_term(limit=100, min_importance=1))
        if not rows:
            await push_tok(client_id, f"No {tier}-term memories.")
        else:
            lines = [f"{tier}-term memory ({len(rows)} rows):\n"]
            for r in rows:
                imp = r.get("importance", "?")
                lines.append(f"  [{r['id']:>3}] imp={imp} [{r.get('topic','')}] {r.get('content','')}")
            await push_tok(client_id, "\n".join(lines))

    elif subcmd == "show":
        tparts = rest.split()
        if not tparts:
            await push_tok(client_id, "Usage: !memory show <id> [short|long]")
            await conditional_push_done(client_id)
            return
        try:
            row_id = int(tparts[0])
        except ValueError:
            await push_tok(client_id, f"Invalid id: {tparts[0]}")
            await conditional_push_done(client_id)
            return
        tier = tparts[1] if len(tparts) > 1 and tparts[1] in ("short", "long") else "short"
        rows = await (load_long_term(limit=1000) if tier == "long" else load_short_term(limit=1000, min_importance=1))
        row = next((r for r in rows if str(r.get("id")) == str(row_id)), None)
        if not row:
            await push_tok(client_id, f"No {tier}-term row with id={row_id}.")
        else:
            await push_tok(client_id, "\n".join(f"  {k}: {v}" for k, v in row.items()))

    elif subcmd in ("true", "false", "on", "off"):
        # Per-session on/off switch — stored in session dict, not global config
        val = subcmd in ("true", "on")
        sessions[client_id]["memory_enabled"] = val
        state_str = "ON" if val else "OFF"
        await push_tok(client_id,
            f"Memory logging {state_str} for this session.\n"
            + ("  conv_log writes to MySQL+Qdrant re-enabled." if val else
               "  conv_log writes to MySQL+Qdrant suppressed. !reset will wipe history without summarizing.")
        )

    elif subcmd == "update":
        tparts = rest.split(maxsplit=1)
        if not tparts:
            await push_tok(client_id, "Usage: !memory update <id> [tier=short] [importance=N] [content=...] [topic=...]")
            await conditional_push_done(client_id)
            return
        try:
            row_id = int(tparts[0])
        except ValueError:
            await push_tok(client_id, f"Invalid id: {tparts[0]}")
            await conditional_push_done(client_id)
            return
        kwargs_str = tparts[1] if len(tparts) > 1 else ""
        # Parse key=value pairs; content/topic may contain spaces so grab them last
        import re as _re
        tier = "short"
        importance = None
        content = None
        topic = None
        m = _re.search(r'\btier=(short|long)\b', kwargs_str)
        if m:
            tier = m.group(1)
            kwargs_str = kwargs_str[:m.start()] + kwargs_str[m.end():]
        m = _re.search(r'\bimportance=(\d+)\b', kwargs_str)
        if m:
            importance = int(m.group(1))
            kwargs_str = kwargs_str[:m.start()] + kwargs_str[m.end():]
        m = _re.search(r'\btopic=(\S+)', kwargs_str)
        if m:
            topic = m.group(1)
            kwargs_str = kwargs_str[:m.start()] + kwargs_str[m.end():]
        m = _re.search(r'\bcontent=(.+)', kwargs_str, _re.DOTALL)
        if m:
            content = m.group(1).strip()
        result = await update_memory(row_id=row_id, tier=tier,
                                     importance=importance, content=content, topic=topic)
        await push_tok(client_id, result)

    else:
        await push_tok(client_id,
            "Usage:\n"
            "  !memory [true|false]                 - enable/disable memory logging (MySQL+Qdrant)\n"
            "  !memory list [short|long]            - list by tier\n"
            "  !memory show <id> [short|long]       - show one row\n"
            "  !memory update <id> [tier=short] [importance=N] [content=text] [topic=label]")

    await conditional_push_done(client_id)


async def cmd_memstats(client_id: str, model_key: str = ""):
    """
    !memstats — memory system health dashboard.
    Shows row counts, topic distribution, importance spread, aging history,
    summarizer runs, dedup config, and last activity timestamps.
    """
    set_model_context(model_key)
    from database import execute_sql
    from memory import _ST, _LT, _SUM, _COLLECTION, _age_cfg, _mem_plugin_cfg, get_retrieval_stats

    lines = ["## Memory System Stats\n"]

    async def q(sql: str) -> str:
        try:
            return await execute_sql(sql)
        except Exception as e:
            return f"(error: {e})"

    def _int_from(raw: str) -> int:
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
            parts = line.split()
            if parts and parts[-1].isdigit():
                return int(parts[-1])
        return 0

    def _rows_from(raw: str) -> list[list[str]]:
        """Parse pipe-separated output into rows of stripped strings."""
        result = []
        for line in raw.strip().splitlines()[1:]:  # skip header
            if not line.strip() or set(line.strip()) <= set("-+|"):
                continue
            result.append([c.strip() for c in line.split("|")])
        return result

    # ── Tier counts ──────────────────────────────────────────────────────────
    st_count = _int_from(await q(f"SELECT COUNT(*) FROM {_ST()}"))
    lt_count = _int_from(await q(f"SELECT COUNT(*) FROM {_LT()}"))
    sum_count = _int_from(await q(f"SELECT COUNT(*) FROM {_SUM()}"))
    lines.append(f"**Tier counts**")
    lines.append(f"  short-term : {st_count} rows")
    lines.append(f"  long-term  : {lt_count} rows")
    lines.append(f"  summaries  : {sum_count} rows\n")

    # ── Short-term: topic breakdown ───────────────────────────────────────────
    st_topics_raw = await q(
        f"SELECT topic, COUNT(*) as n, ROUND(AVG(importance),1) as avg_imp "
        f"FROM {_ST()} GROUP BY topic ORDER BY n DESC LIMIT 20"
    )
    st_topic_rows = _rows_from(st_topics_raw)
    if st_topic_rows:
        lines.append(f"**Short-term by topic** (top {len(st_topic_rows)})")
        for row in st_topic_rows:
            if len(row) >= 3:
                lines.append(f"  {row[0]:<30} {row[1]:>4} rows  avg_imp={row[2]}")
        lines.append("")

    # ── Long-term: topic breakdown ────────────────────────────────────────────
    lt_topics_raw = await q(
        f"SELECT topic, COUNT(*) as n, ROUND(AVG(importance),1) as avg_imp "
        f"FROM {_LT()} GROUP BY topic ORDER BY n DESC LIMIT 20"
    )
    lt_topic_rows = _rows_from(lt_topics_raw)
    if lt_topic_rows:
        lines.append(f"**Long-term by topic** (top {len(lt_topic_rows)})")
        for row in lt_topic_rows:
            if len(row) >= 3:
                lines.append(f"  {row[0]:<30} {row[1]:>4} rows  avg_imp={row[2]}")
        lines.append("")

    # ── Importance distribution (short-term) ─────────────────────────────────
    imp_raw = await q(
        f"SELECT importance, COUNT(*) as n FROM {_ST()} GROUP BY importance ORDER BY importance"
    )
    imp_rows = _rows_from(imp_raw)
    if imp_rows:
        lines.append("**Short-term importance distribution**")
        dist = "  " + "  ".join(f"[{r[0]}]={r[1]}" for r in imp_rows if len(r) >= 2)
        lines.append(dist)
        lines.append("")

    # ── Source breakdown (short-term) ─────────────────────────────────────────
    src_raw = await q(
        f"SELECT source, COUNT(*) as n FROM {_ST()} GROUP BY source ORDER BY n DESC"
    )
    src_rows = _rows_from(src_raw)
    if src_rows:
        lines.append("**Short-term by source**")
        for row in src_rows:
            if len(row) >= 2:
                lines.append(f"  {row[0]:<20} {row[1]:>4} rows")
        lines.append("")

    # ── Aging: long-term source of truth ─────────────────────────────────────
    lt_from_st_raw = await q(
        f"SELECT COUNT(*) FROM {_LT()} WHERE shortterm_id IS NOT NULL AND shortterm_id > 0"
    )
    lt_aged = _int_from(lt_from_st_raw)
    lt_direct_raw = await q(
        f"SELECT COUNT(*) FROM {_LT()} WHERE shortterm_id IS NULL OR shortterm_id = 0"
    )
    lt_direct = _int_from(lt_direct_raw)
    lines.append("**Long-term origin**")
    lines.append(f"  aged from short-term : {lt_aged} rows")
    lines.append(f"  direct saves         : {lt_direct} rows\n")

    # ── Last activity timestamps ──────────────────────────────────────────────
    st_newest_raw = await q(f"SELECT MAX(created_at) FROM {_ST()}")
    st_oldest_raw = await q(f"SELECT MIN(created_at) FROM {_ST()}")
    st_lru_raw    = await q(f"SELECT MIN(last_accessed) FROM {_ST()}")
    lt_newest_raw = await q(f"SELECT MAX(created_at) FROM {_LT()}")
    sum_newest_raw = await q(f"SELECT MAX(created_at) FROM {_SUM()}")

    def _scalar(raw: str) -> str:
        for line in raw.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("MAX") and not line.startswith("MIN") and not set(line).issubset(set("-+| ")):
                return line
        return "(none)"

    lines.append("**Timestamps**")
    lines.append(f"  ST newest created    : {_scalar(st_newest_raw)}")
    lines.append(f"  ST oldest created    : {_scalar(st_oldest_raw)}")
    lines.append(f"  ST least recently    : {_scalar(st_lru_raw)}")
    lines.append(f"  LT newest created    : {_scalar(lt_newest_raw)}")
    lines.append(f"  Summaries newest     : {_scalar(sum_newest_raw)}\n")

    # ── Summarizer runs ───────────────────────────────────────────────────────
    if sum_count > 0:
        sum_model_raw = await q(
            f"SELECT model_used, COUNT(*) as n FROM {_SUM()} GROUP BY model_used ORDER BY n DESC"
        )
        sum_model_rows = _rows_from(sum_model_raw)
        if sum_model_rows:
            lines.append("**Summarizer runs by model**")
            for row in sum_model_rows:
                if len(row) >= 2:
                    lines.append(f"  {row[0]:<30} {row[1]:>3} runs")
            lines.append("")

    # ── Vector index stats ────────────────────────────────────────────────────
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if vec:
            from memory import _COLLECTION
            vstats = await vec.get_stats(collection=_COLLECTION())
            vcfg   = vec.cfg()
            qd = vstats.get("qdrant", {})
            em = vstats.get("embed", {})

            lines.append("**Vector Index (Qdrant)**")
            if "error" in qd:
                lines.append(f"  status               : ERROR — {qd['error']}")
            else:
                mysql_total = st_count + lt_count
                qdrant_pts  = qd.get("points_count", "?")
                drift = ""
                if isinstance(qdrant_pts, int) and mysql_total > 0:
                    diff = mysql_total - qdrant_pts
                    if diff != 0:
                        drift = f"  !! drift: MySQL={mysql_total} Qdrant={qdrant_pts} (diff={diff:+d})"
                lines.append(f"  status               : {qd.get('status', '?')}")
                lines.append(f"  points_count         : {qdrant_pts}  (MySQL ST+LT={mysql_total})")
                if drift:
                    lines.append(drift)
                lines.append(f"  indexed_vectors      : {qd.get('indexed_vectors_count', '?')}")
                lines.append(f"  segments             : {qd.get('segments_count', '?')}")
                lines.append(f"  optimizer            : {qd.get('optimizer_status', '?')}")
                lines.append(f"  collection           : {vcfg.get('collection')}  @{vcfg.get('qdrant_host')}:{vcfg.get('qdrant_port')}")
                lines.append(f"  top_k / min_score    : {vcfg.get('top_k')} / {vcfg.get('min_score')}")
                lines.append(f"  always_inject_imp>=  : {vcfg.get('min_importance_always')}")
            lines.append("")

            lines.append("**Embedding Server (nomic-embed-text)**")
            lines.append(f"  health               : {em.get('health', '?')}")
            if "kv_cache_pct" in em:
                lines.append(f"  kv_cache_usage       : {em['kv_cache_pct']}")
            if "kv_cache_tokens" in em:
                lines.append(f"  kv_cache_tokens      : {em['kv_cache_tokens']}")
            if "embed_tps" in em:
                lines.append(f"  embed_throughput     : {em['embed_tps']}")
            if "requests_processing" in em:
                lines.append(f"  requests_processing  : {em['requests_processing']}")
            if "requests_deferred" in em:
                lines.append(f"  requests_deferred    : {em['requests_deferred']}")
            if "metrics" in em:
                lines.append(f"  metrics endpoint     : {em['metrics']}")
            lines.append(f"  url                  : {vcfg.get('embed_url')}")
            lines.append("")
        else:
            lines.append("**Vector Index**: not loaded\n")
    except Exception as _vec_err:
        lines.append(f"**Vector Index**: error — {_vec_err}\n")

    # ── Retrieval stats (two-pass) ───────────────────────────────────────────
    rstats = get_retrieval_stats()
    if rstats["total"] > 0:
        single = rstats["single_pass_sufficient"]
        two = rstats["two_pass_needed"]
        fallback = rstats["fallback_no_vec"]
        total = rstats["total"]
        single_pct = int(single / total * 100) if total else 0
        two_pct = int(two / total * 100) if total else 0
        lines.append("**Retrieval Stats (since restart)**")
        lines.append(f"  total retrievals     : {total}")
        lines.append(f"  single-pass sufficient: {single} ({single_pct}%)")
        lines.append(f"  two-pass needed      : {two} ({two_pct}%)")
        lines.append(f"  fallback (no vector) : {fallback}")
        qfloor = float(_mem_plugin_cfg().get("two_pass_quality_floor", 0.75))
        lines.append(f"  pass-1 avg quality hits: {rstats['pass1_avg_hits']:.1f} (score ≥ {qfloor})")
        if two > 0:
            lines.append(f"  pass-2 avg extra hits: {rstats['pass2_avg_extra']:.1f}")
        lines.append("")

    # ── Config snapshot ───────────────────────────────────────────────────────
    mem_cfg = _mem_plugin_cfg()
    age_cfg = _age_cfg()
    lines.append("**Config (plugins-enabled.json)**")
    master_on = mem_cfg.get("enabled", True)
    lines.append(f"  {'enabled (master)':<28}: {'on' if master_on else 'OFF'}")
    _sess_mem = sessions[client_id].get("memory_enabled", None) if client_id in sessions else None
    if _sess_mem is not None:
        lines.append(f"  {'enabled (this session)':<28}: {'on' if _sess_mem else 'OFF'} (overrides master)")
    features = ("context_injection", "reset_summarize", "post_response_scan", "fuzzy_dedup", "vector_search_qdrant")
    for f in features:
        val = mem_cfg.get(f, True)
        lines.append(f"  {f:<28}: {'on' if val else 'OFF'}{' (inactive—master off)' if not master_on else ''}")
    lines.append(f"  {'fuzzy_dedup_threshold':<28}: {mem_cfg.get('fuzzy_dedup_threshold', 0.78):.2f}")
    lines.append(f"  {'summarizer_model':<28}: {mem_cfg.get('summarizer_model', 'summarizer-anthropic')}")
    hwm = age_cfg["short_hwm"]
    lwm = age_cfg["short_lwm"]
    lines.append(f"  {'auto_memory_age':<28}: {'on' if age_cfg['auto_memory_age'] else 'OFF'}")
    lines.append(f"  {'short_hwm (aging trigger)':<28}: {hwm} rows")
    lines.append(f"  {'short_lwm (aging target)':<28}: {lwm} rows")
    lines.append(f"  {'recent_turns_protect':<28}: {age_cfg['recent_turns_protect']} turns")
    lines.append(f"  {'staleness_override_minutes':<28}: {age_cfg['staleness_override_minutes']} min ({age_cfg['staleness_override_minutes']//60}h)")
    lines.append(f"  {'chunk_importance_threshold':<28}: {age_cfg['chunk_importance_threshold']}")

    def _timer(v: int) -> str:
        return "disabled" if v == -1 else f"{v} min"

    lines.append(f"  {'memory_age_count_timer':<28}: {_timer(age_cfg['memory_age_count_timer'])}")
    lines.append(f"  {'memory_age_trigger_minutes':<28}: {age_cfg['memory_age_trigger_minutes']} min ({age_cfg['memory_age_trigger_minutes']//60}h)")
    lines.append(f"  {'memory_age_minutes_timer':<28}: {_timer(age_cfg['memory_age_minutes_timer'])}")

    # Pressure bar vs HWM/LWM
    pct_hwm = int(st_count / hwm * 100) if hwm > 0 else 0
    filled  = min(20, pct_hwm // 5)
    lwm_pos = min(20, int(lwm / hwm * 20)) if hwm > 0 else 0
    bar_chars = list("." * 20)
    for i in range(filled):
        bar_chars[i] = "#"
    if 0 <= lwm_pos < 20:
        bar_chars[lwm_pos] = "|"  # LWM marker
    bar = "".join(bar_chars)
    status = "AGING NOW" if st_count >= hwm else ("near HWM" if pct_hwm >= 80 else "ok")
    lines.append(f"\n  ST pressure  [{bar}] {st_count}/{hwm} rows ({pct_hwm}%)  LWM={lwm}  [{status}]")

    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


async def cmd_membackfill(client_id: str, model_key: str = ""):
    """
    !membackfill — embed and upsert any MySQL memory rows missing from Qdrant.
    Compares all MySQL row IDs against Qdrant point IDs; only processes the gap.
    """
    set_model_context(model_key)
    from database import fetch_dicts
    from memory import _ST, _LT, _COLLECTION

    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if not vec:
            await push_tok(client_id, "Vector plugin not available.")
            await conditional_push_done(client_id)
            return
    except Exception as e:
        await push_tok(client_id, f"Vector plugin error: {e}")
        await conditional_push_done(client_id)
        return

    await push_tok(client_id, "Scanning for missing Qdrant points...\n")

    coll = _COLLECTION()
    try:
        qdrant_ids = vec.get_all_point_ids(collection=coll)
    except Exception as e:
        await push_tok(client_id, f"Failed to fetch Qdrant point IDs: {e}")
        await conditional_push_done(client_id)
        return

    st_rows = await fetch_dicts(f"SELECT id, topic, content, importance FROM {_ST()}")
    lt_rows = await fetch_dicts(f"SELECT id, topic, content, importance FROM {_LT()}")

    mysql_ids = {int(r["id"]) for r in st_rows} | {int(r["id"]) for r in lt_rows}
    missing_st = [r for r in st_rows if int(r["id"]) not in qdrant_ids]
    missing_lt = [r for r in lt_rows if int(r["id"]) not in qdrant_ids]
    total_missing = len(missing_st) + len(missing_lt)
    orphan_count = len(qdrant_ids - mysql_ids)

    # Report metrics
    report = (
        f"Collection:     {coll}\n"
        f"Qdrant points:  {len(qdrant_ids)}\n"
        f"MySQL rows:     {len(mysql_ids)} ({len(st_rows)} ST, {len(lt_rows)} LT)\n"
        f"In sync:        {len(qdrant_ids & mysql_ids)}\n"
        f"Missing from Q: {total_missing} ({len(missing_st)} ST, {len(missing_lt)} LT)\n"
        f"Orphans in Q:   {orphan_count} (use !memreconcile to clean)\n"
    )
    await push_tok(client_id, report)

    if total_missing == 0:
        await push_tok(client_id, "No missing points — Qdrant has all MySQL rows.")
        await conditional_push_done(client_id)
        return

    await push_tok(client_id, f"Backfilling {total_missing} missing rows...")

    saved_st = await vec.backfill(missing_st, tier="short", collection=coll) if missing_st else 0
    saved_lt = await vec.backfill(missing_lt, tier="long",  collection=coll) if missing_lt else 0

    await push_tok(client_id, f"Done. Upserted {saved_st} short-term + {saved_lt} long-term rows into Qdrant.")
    await conditional_push_done(client_id)


async def cmd_memreconcile(client_id: str, model_key: str = ""):
    """
    !memreconcile — remove orphaned Qdrant points whose MySQL rows no longer exist.
    Inverse of !membackfill: Qdrant has points that MySQL doesn't → delete from Qdrant.
    Reports metrics: total Qdrant points, MySQL rows, orphans found, orphans deleted.
    """
    set_model_context(model_key)
    from database import fetch_dicts
    from memory import _ST, _LT, _COLLECTION

    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if not vec:
            await push_tok(client_id, "Vector plugin not available.")
            await conditional_push_done(client_id)
            return
    except Exception as e:
        await push_tok(client_id, f"Vector plugin error: {e}")
        await conditional_push_done(client_id)
        return

    await push_tok(client_id, "Scanning for orphaned Qdrant points...\n")

    coll = _COLLECTION()
    try:
        qdrant_ids = vec.get_all_point_ids(collection=coll)
    except Exception as e:
        await push_tok(client_id, f"Failed to fetch Qdrant point IDs: {e}")
        await conditional_push_done(client_id)
        return

    # Collect all valid MySQL IDs from both tiers
    st_ids = await fetch_dicts(f"SELECT id FROM {_ST()}")
    lt_ids = await fetch_dicts(f"SELECT id FROM {_LT()}")
    mysql_ids = {int(r["id"]) for r in st_ids} | {int(r["id"]) for r in lt_ids}

    orphan_ids = qdrant_ids - mysql_ids
    in_sync = qdrant_ids & mysql_ids

    # Report metrics
    report = (
        f"Collection:     {coll}\n"
        f"Qdrant points:  {len(qdrant_ids)}\n"
        f"MySQL rows:     {len(mysql_ids)}\n"
        f"In sync:        {len(in_sync)}\n"
        f"Orphans found:  {len(orphan_ids)}\n"
    )
    await push_tok(client_id, report)

    if not orphan_ids:
        await push_tok(client_id, "No orphans — Qdrant is fully reconciled with MySQL.")
        await conditional_push_done(client_id)
        return

    # Delete orphaned points in batches
    deleted = 0
    batch_size = 500
    orphan_list = sorted(orphan_ids)
    for i in range(0, len(orphan_list), batch_size):
        batch = orphan_list[i : i + batch_size]
        try:
            vec._qc.delete(
                collection_name=coll,
                points_selector=batch,
            )
            deleted += len(batch)
        except Exception as e:
            await push_tok(client_id, f"Delete batch failed at offset {i}: {e}\n")

    await push_tok(client_id, f"Deleted {deleted}/{len(orphan_ids)} orphaned Qdrant points.")
    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# !memreview — AI-assisted topic review with HITL approval
# ---------------------------------------------------------------------------
_pending_reviews: dict[str, list[dict]] = {}  # client_id → list of proposals


async def cmd_memreview(client_id: str, arg: str = "", model_key: str = ""):
    """
    !memreview [approve N,N,...] [reject N,N,...] [clear]

    No args: Calls reviewer model (config: reviewer_model) to analyse all topics and propose merges/renames.
    approve N,N,...: Execute approved proposals by number.
    reject N,N,...: Remove proposals from the pending list.
    clear: Discard all pending proposals.
    """
    set_model_context(model_key)
    from database import fetch_dicts, execute_sql
    from memory import _ST, _LT, _COLLECTION

    parts = arg.strip().split(maxsplit=1) if arg.strip() else []
    subcmd = parts[0].lower() if parts else ""

    # ── approve / reject / clear ─────────────────────────────────────────
    if subcmd == "clear":
        _pending_reviews.pop(client_id, None)
        await push_tok(client_id, "Pending review proposals cleared.")
        await conditional_push_done(client_id)
        return

    if subcmd in ("approve", "reject"):
        proposals = _pending_reviews.get(client_id, [])
        if not proposals:
            await push_tok(client_id, "No pending proposals. Run !memreview first.")
            await conditional_push_done(client_id)
            return

        nums_str = parts[1] if len(parts) > 1 else ""
        try:
            nums = [int(n.strip()) for n in nums_str.split(",") if n.strip()]
        except ValueError:
            await push_tok(client_id, "Usage: !memreview approve 1,2,3 or !memreview reject 1,2")
            await conditional_push_done(client_id)
            return

        if not nums:
            await push_tok(client_id, "Specify proposal numbers: !memreview approve 1,3")
            await conditional_push_done(client_id)
            return

        if subcmd == "reject":
            rejected = []
            for n in sorted(nums, reverse=True):
                if 1 <= n <= len(proposals):
                    p = proposals.pop(n - 1)
                    rejected.append(f"#{n} ({p.get('action', '?')} {p.get('from', '?')})")
            if not proposals:
                _pending_reviews.pop(client_id, None)
            await push_tok(client_id, f"Rejected: {', '.join(rejected) if rejected else 'none matched'}")
            await conditional_push_done(client_id)
            return

        # ── approve: execute proposals ───────────────────────────────
        applied = []
        errors = []
        for n in sorted(nums):
            if n < 1 or n > len(proposals):
                errors.append(f"#{n}: out of range")
                continue
            p = proposals[n - 1]
            action = p.get("action", "")
            old_topic = p.get("from", "")
            new_topic = p.get("to", "")
            if not old_topic or not new_topic:
                errors.append(f"#{n}: invalid proposal")
                continue

            try:
                # Collect affected row IDs before updating
                affected_ids = []
                for table in (_ST(), _LT()):
                    rows = await fetch_dicts(
                        f"SELECT id FROM {table} WHERE topic = '{old_topic}'"
                    )
                    affected_ids.extend(int(r["id"]) for r in rows)

                # Update MySQL (both ST and LT tables)
                for table in (_ST(), _LT()):
                    await execute_sql(
                        f"UPDATE {table} SET topic = '{new_topic}' "
                        f"WHERE topic = '{old_topic}'"
                    )

                # Update Qdrant payload to match
                if affected_ids:
                    try:
                        from plugin_memory_vector_qdrant import get_vector_api
                        vec = get_vector_api()
                        if vec:
                            vec._qc.set_payload(
                                collection_name=_COLLECTION(),
                                payload={"topic": new_topic},
                                points=affected_ids,
                            )
                    except Exception as qe:
                        errors.append(f"#{n}: MySQL OK but Qdrant update failed: {qe}")

                applied.append(f"#{n}: {action} '{old_topic}' → '{new_topic}' ({len(affected_ids)} rows)")
            except Exception as e:
                errors.append(f"#{n}: {e}")

        # Remove applied proposals (reverse order to preserve indices)
        for n in sorted(nums, reverse=True):
            if 1 <= n <= len(proposals):
                proposals.pop(n - 1)
        if not proposals:
            _pending_reviews.pop(client_id, None)

        result_lines = []
        if applied:
            result_lines.append("Applied:\n" + "\n".join(f"  {a}" for a in applied))
        if errors:
            result_lines.append("Errors:\n" + "\n".join(f"  {e}" for e in errors))
        remaining = len(_pending_reviews.get(client_id, []))
        if remaining:
            result_lines.append(f"\n{remaining} proposal(s) still pending.")
        await push_tok(client_id, "\n".join(result_lines) if result_lines else "No valid proposals to apply.")
        await conditional_push_done(client_id)
        return

    # ── show pending proposals if any exist ───────────────────────────────
    if subcmd == "" and client_id in _pending_reviews and _pending_reviews[client_id]:
        proposals = _pending_reviews[client_id]
        lines = [f"**Pending Topic Review ({len(proposals)} proposals)**\n"]
        for i, p in enumerate(proposals, 1):
            reason = p.get("reason", "")
            lines.append(
                f"  {i}. [{p.get('action', '?')}] "
                f"'{p.get('from', '?')}' → '{p.get('to', '?')}'"
                f"  ({p.get('from_count', '?')} rows → {p.get('to_count', '?')} rows)"
            )
            if reason:
                lines.append(f"     Reason: {reason}")
        lines.append("\nUsage: !memreview approve 1,3  |  !memreview reject 2  |  !memreview clear")
        await push_tok(client_id, "\n".join(lines))
        await conditional_push_done(client_id)
        return

    # ── generate new review ──────────────────────────────────────────────
    from memory import _mem_plugin_cfg
    _review_model = _mem_plugin_cfg().get("reviewer_model", "reviewer-gemini")
    await push_tok(client_id, f"Analysing topics with {_review_model}...\n")

    # Gather topic stats + sample content from both tiers
    st_rows = await fetch_dicts(
        f"SELECT topic, content, importance FROM {_ST()} ORDER BY topic, id DESC"
    )
    lt_rows = await fetch_dicts(
        f"SELECT topic, content, importance FROM {_LT()} ORDER BY topic, id DESC"
    )

    # Build topic summary with sample content (max 3 samples per topic per tier)
    topic_data: dict[str, dict] = {}
    for tier_name, rows in [("ST", st_rows), ("LT", lt_rows)]:
        for r in rows:
            t = r.get("topic", "unknown")
            if t not in topic_data:
                topic_data[t] = {"st_count": 0, "lt_count": 0, "samples": []}
            topic_data[t][f"{tier_name.lower()}_count"] = topic_data[t].get(f"{tier_name.lower()}_count", 0) + 1
            # Keep max 3 samples per topic (truncated)
            samples = topic_data[t]["samples"]
            if len([s for s in samples if s.startswith(f"[{tier_name}]")]) < 3:
                content = str(r.get("content", ""))[:200]
                samples.append(f"[{tier_name}] {content}")

    if not topic_data:
        await push_tok(client_id, "No topics found in memory.")
        await conditional_push_done(client_id)
        return

    # Build the review prompt
    topic_lines = []
    for slug, data in sorted(topic_data.items()):
        total = data.get("st_count", 0) + data.get("lt_count", 0)
        topic_lines.append(
            f"\n### {slug} (ST={data.get('st_count', 0)}, LT={data.get('lt_count', 0)}, total={total})"
        )
        for s in data["samples"]:
            topic_lines.append(f"  {s}")

    prompt = (
        "You are a topic hygiene reviewer for a personal AI memory system.\n"
        "Below is a list of all topic slugs with row counts and sample content.\n\n"
        "Analyse the topics and propose:\n"
        "1. **merge**: Two topics that represent the same real-world subject but have different slugs. "
        "The 'to' field should be the more established topic (higher row count or better name).\n"
        "2. **rename**: A single topic with a poor or inconsistent slug name.\n\n"
        "Rules:\n"
        "- Only propose merges when content clearly overlaps (same real-world subject).\n"
        "- Do NOT merge topics that merely share a word prefix (e.g. 'memory-system' and 'memory-roadmap' are distinct).\n"
        "- Do NOT propose changes for topics that are already well-named and distinct.\n"
        "- Prefer shorter, clearer slugs.\n"
        "- If no changes are needed, return an empty proposals list.\n\n"
        "Return ONLY valid JSON (no markdown fences, no commentary):\n"
        '{"proposals": [{"action": "merge"|"rename", "from": "old-slug", "to": "new-slug", "reason": "brief explanation"}]}\n\n'
        "## Topics\n"
        + "\n".join(topic_lines)
    )

    # Call reviewer model
    try:
        from agents import llm_call as _llm_call
        from state import current_client_id

        token = current_client_id.set(client_id)
        try:
            result = await _llm_call(
                model=_review_model,
                prompt=prompt,
                mode="text",
                sys_prompt="none",
                history="none",
            )
        finally:
            current_client_id.reset(token)
    except Exception as e:
        await push_tok(client_id, f"Review failed: {e}")
        await conditional_push_done(client_id)
        return

    # Parse JSON response
    import json as _json
    try:
        # Strip markdown fences if model adds them despite instructions
        raw = result.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        parsed = _json.loads(raw.strip())
        proposals = parsed.get("proposals", [])
    except (_json.JSONDecodeError, AttributeError) as e:
        await push_tok(client_id, f"Failed to parse reviewer response:\n{result[:500]}\n\nError: {e}")
        await conditional_push_done(client_id)
        return

    if not proposals:
        await push_tok(client_id, "Reviewer found no changes needed. Topics look clean.")
        await conditional_push_done(client_id)
        return

    # Enrich proposals with row counts
    for p in proposals:
        from_slug = p.get("from", "")
        to_slug = p.get("to", "")
        from_data = topic_data.get(from_slug, {})
        to_data = topic_data.get(to_slug, {})
        p["from_count"] = from_data.get("st_count", 0) + from_data.get("lt_count", 0)
        p["to_count"] = to_data.get("st_count", 0) + to_data.get("lt_count", 0)

    # Store and display
    _pending_reviews[client_id] = proposals

    lines = [f"**Topic Review ({len(proposals)} proposals)**\n"]
    for i, p in enumerate(proposals, 1):
        lines.append(
            f"  {i}. [{p.get('action', '?')}] "
            f"'{p.get('from', '?')}' → '{p.get('to', '?')}'"
            f"  ({p.get('from_count', '?')} rows → {p.get('to_count', '?')} rows)"
        )
        reason = p.get("reason", "")
        if reason:
            lines.append(f"     Reason: {reason}")
    lines.append("\nUsage: !memreview approve 1,3  |  !memreview reject 2  |  !memreview clear")
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


async def cmd_memage(client_id: str, model_key: str = ""):
    """
    !memage — manually trigger one full aging pass (count-pressure + staleness).
    Uses the same logic as the background tasks but runs immediately.
    """
    from memory import age_by_count, age_by_minutes, _age_cfg, _st_count
    set_model_context(model_key)

    cfg     = _age_cfg()
    before  = await _st_count()
    await push_tok(client_id, f"ST rows before: {before} (HWM={cfg['short_hwm']}, LWM={cfg['short_lwm']})\n")
    await push_tok(client_id, "Running count-pressure aging pass...")

    deleted_count = await age_by_count()
    mid = await _st_count()
    await push_tok(client_id, f" done ({deleted_count} rows deleted, ST now={mid})\n")

    await push_tok(client_id, "Running staleness aging pass...")
    deleted_min = await age_by_minutes(trigger_minutes=cfg["memory_age_trigger_minutes"])
    after = await _st_count()
    await push_tok(
        client_id,
        f" done ({deleted_min} rows deleted, ST now={after})\n"
        f"Total deleted: {deleted_count + deleted_min} rows.\n"
    )
    await conditional_push_done(client_id)


async def cmd_memtrim(client_id: str, arg: str, model_key: str = ""):
    """
    !memtrim [N] — hard-delete N oldest/least-important ST rows without summarizing.
    If N is omitted, trims to short_lwm.
    Escape valve for when topic-chunk aging can't make progress.
    """
    set_model_context(model_key)
    from memory import trim_st_to_lwm, _st_count, _age_cfg, _COLLECTION
    from database import execute_sql, fetch_dicts
    from memory import _ST

    cfg    = _age_cfg()
    before = await _st_count()

    if arg.strip().isdigit():
        n = int(arg.strip())
        await push_tok(client_id, f"Hard-trimming {n} oldest ST rows (ST before={before})...\n")
        # Fetch target rows and delete them
        try:
            from plugin_memory_vector_qdrant import get_vector_api
            vec = get_vector_api()
        except Exception:
            vec = None
        try:
            import asyncio
            rows = await fetch_dicts(
                f"SELECT id FROM {_ST()} ORDER BY importance ASC, last_accessed ASC LIMIT {n}"
            )
            deleted = 0
            for r in rows:
                rid = r.get("id")
                if rid:
                    await execute_sql(f"DELETE FROM {_ST()} WHERE id = {int(rid)}")
                    deleted += 1
                    if vec:
                        asyncio.create_task(vec.delete_memory(int(rid), collection=_COLLECTION()))
            after = await _st_count()
            await push_tok(client_id, f"Deleted {deleted} rows. ST now={after}.\n")
        except Exception as e:
            await push_tok(client_id, f"Error during trim: {e}\n")
    else:
        await push_tok(client_id, f"Trimming ST to LWM={cfg['short_lwm']} (ST before={before})...\n")
        deleted = await trim_st_to_lwm()
        after   = await _st_count()
        await push_tok(client_id, f"Deleted {deleted} rows. ST now={after}.\n")

    await conditional_push_done(client_id)


async def cmd_search(client_id: str, engine: str, query: str):
    """Execute a search using the named engine's executor."""
    from tools import get_tool_executor
    executor = get_tool_executor(f"search_{engine}")
    if executor is None:
        await push_tok(client_id, f"ERROR: Search engine 'search_{engine}' not loaded.")
        await conditional_push_done(client_id)
        return
    if not query.strip():
        await push_tok(client_id, f"Usage: !search_{engine} <query>")
        await conditional_push_done(client_id)
        return
    try:
        result = await executor(query=query.strip())
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: search_{engine} failed: {exc}")
    await conditional_push_done(client_id)


async def cmd_url_extract(client_id: str, args: str):
    """Extract content from a URL using Tavily."""
    from tools import get_tool_executor
    parts = args.split(maxsplit=1)
    if not parts:
        await push_tok(client_id, "Usage: !url_extract <url> [query]")
        await conditional_push_done(client_id)
        return
    url = parts[0].strip()
    # Slack wraps URLs in angle brackets: <https://example.com> or <https://example.com|display>
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1].split("|")[0]
    query = parts[1].strip() if len(parts) > 1 else ""
    executor = get_tool_executor("url_extract")
    if executor is None:
        await push_tok(client_id, "ERROR: url_extract plugin not loaded.")
        await conditional_push_done(client_id)
        return
    try:
        result = await executor(method="tavily", url=url, query=query)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: url_extract failed: {exc}")
    await conditional_push_done(client_id)


async def cmd_google_drive(client_id: str, args: str):
    """Execute a Google Drive operation."""
    parts = args.split(maxsplit=1)
    if not parts:
        await push_tok(client_id,
            "Usage: !google_drive <operation> [file_id] [file_name] [content] [folder_id]\n"
            "Operations: list, read, create, append, delete")
        await conditional_push_done(client_id)
        return
    operation = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    from drive import run_drive_op
    try:
        result = await run_drive_op(operation, None, rest or None, None, None)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: google_drive failed: {exc}")
    await conditional_push_done(client_id)


async def cmd_plugin_command(client_id: str, cmd: str, args: str):
    """
    Generic dispatcher for plugin-registered !commands.
    Looks up cmd in the _PLUGIN_COMMANDS registry and calls the handler.
    Handler signature: async (args: str) -> str
    Sets current_client_id ContextVar so handlers can access session/state.
    """
    handler = get_plugin_command(cmd)
    if handler is None:
        await push_tok(client_id, f"Unknown command: !{cmd}\nUse !help to see available commands.")
        await conditional_push_done(client_id)
        return
    from state import current_client_id as _ccid
    token = _ccid.set(client_id)
    try:
        result = await handler(args)
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: !{cmd} failed: {exc}")
    finally:
        _ccid.reset(token)
    await conditional_push_done(client_id)


async def cmd_get_system_info(client_id: str):
    """Show current date/time and system status."""
    from tools import get_system_info
    from state import current_client_id
    token = current_client_id.set(client_id)
    try:
        result = await get_system_info()
    finally:
        current_client_id.reset(token)
    await push_tok(client_id, str(result))
    await conditional_push_done(client_id)


async def cmd_llm_list(client_id: str):
    """List LLM models with their configuration."""
    from agents import llm_list
    result = await llm_list()
    await push_tok(client_id, result)
    await conditional_push_done(client_id)


async def cmd_llm_call_invoke(client_id: str, args: str):
    """
    Invoke a target LLM directly.

    Usage: !llm_call_invoke <model> <prompt> [key=value ...]

    Keyword options (all optional, defaults shown):
      mode=text          text | tool
      sys_prompt=none    none | caller | target
      history=none       none | caller
      tool=<name>        required when mode=tool

    Examples:
      !llm_call_invoke nuc11Local Summarize this text for me.
      !llm_call_invoke grok4 What's the weather? mode=text sys_prompt=caller
      !llm_call_invoke nuc11Local Find the top result tool=ddgs_search mode=tool
    """
    # Parse: first token = model, remaining tokens parsed for key=value overrides,
    # everything else concatenated as the prompt.
    parts = args.split()
    if len(parts) < 2:
        await push_tok(client_id,
            "Usage: !llm_call_invoke <model> <prompt> [mode=text|tool] [sys_prompt=none|caller|target] "
            "[history=none|caller] [tool=<name>]")
        await conditional_push_done(client_id)
        return

    model = parts[0].strip()
    kwargs = {"mode": "text", "sys_prompt": "none", "history": "none", "tool": ""}
    kw_keys = set(kwargs.keys())
    prompt_parts = []

    for token in parts[1:]:
        if "=" in token:
            k, _, v = token.partition("=")
            if k in kw_keys:
                kwargs[k] = v
                continue
        prompt_parts.append(token)

    prompt = " ".join(prompt_parts).strip()
    if not prompt:
        await push_tok(client_id, "ERROR: prompt is required.\nUsage: !llm_call_invoke <model> <prompt> [options]")
        await conditional_push_done(client_id)
        return

    from agents import llm_call as _llm_call
    from state import current_client_id
    token = current_client_id.set(client_id)
    try:
        result = await _llm_call(
            model=model,
            prompt=prompt,
            mode=kwargs["mode"],
            sys_prompt=kwargs["sys_prompt"],
            history=kwargs["history"],
            tool=kwargs["tool"],
        )
        await push_tok(client_id, result)
    except Exception as exc:
        await push_tok(client_id, f"ERROR: llm_call_invoke failed: {exc}")
    finally:
        current_client_id.reset(token)
    await conditional_push_done(client_id)






async def cmd_list_models(client_id: str, current: str):
    lines = ["Available models:"]
    # Enabled models from in-memory registry
    for key, meta in LLM_REGISTRY.items():
        model_id = meta.get("model_id", key)
        marker = " (current)" if key == current else ""
        lines.append(f"  {key:<12} {model_id}{marker}")
    # Disabled models from llm-models.json (not in LLM_REGISTRY)
    try:
        import json as _json
        with open(LLM_MODELS_FILE, "r") as _f:
            _data = _json.load(_f)
        for key, cfg in _data.get("models", {}).items():
            if not cfg.get("enabled", True) and key not in LLM_REGISTRY:
                model_id = cfg.get("model_id", key)
                lines.append(f"  {key:<12} {model_id}  [disabled]")
    except Exception:
        pass
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)

async def cmd_stop(client_id: str):
    """Cancel the currently running LLM job for this client, if any."""
    cancelled = await cancel_active_task(client_id)
    if cancelled:
        await push_tok(client_id, "Job stopped.")
    else:
        await push_tok(client_id, "No job running.")
    await push_done(client_id)


async def cmd_set_model(client_id: str, key: str, session: dict):
    """Set the active LLM model for this session."""
    if not key or not key.strip():
        await push_tok(client_id,
            "ERROR: Model name required\n"
            "Usage: !model <model_name>\n"
            "Use !model to list available models")
        await conditional_push_done(client_id)
        return

    key = key.strip()
    if key in LLM_REGISTRY:
        await cancel_active_task(client_id)
        old_model = session["model"]
        old_cfg = LLM_REGISTRY.get(old_model, {})
        new_cfg = LLM_REGISTRY.get(key, {})
        session["model"] = key
        # Recompute effective window and trim history immediately
        trimmed = _notify_chain_model_switch(session, old_model, key, old_cfg, new_cfg)
        prev_len = len(session.get("history", []))
        session["history"] = trimmed
        dropped = prev_len - len(trimmed)
        await push_model(client_id, key)
        msg = f"Model set to '{key}'."
        if dropped > 0:
            msg += f" History trimmed: {dropped} message(s) removed ({len(trimmed)} kept)."
        await push_tok(client_id, msg)
    else:
        available = ", ".join(LLM_REGISTRY.keys())
        await push_tok(client_id,
            f"ERROR: Unknown model '{key}'\n"
            f"Available models: {available}\n"
            f"Use !model to list all models")
    await conditional_push_done(client_id)


async def cmd_reset(client_id: str, session: dict):
    """Clear conversation history for current session, summarizing to memory first."""
    from state import delete_history
    history = list(session.get("history", []))
    history_len = len(history)

    # Summarize departing history into short-term memory before clearing.
    # Skip entirely when memory.enabled is false — just wipe history.
    from agents import _memory_feature, _memory_cfg
    if history_len >= 4 and _memory_feature("enabled") and _memory_feature("reset_summarize"):
        try:
            from memory import summarize_and_save
            set_model_context(session.get("model", ""))
            summarizer_model = _memory_cfg().get("summarizer_model", "summarizer-anthropic")
            _reset_suppress = session.get("tool_suppress", False)
            if not _reset_suppress:
                await push_tok(client_id, "[memory] Summarizing session to memory...\n")
            status = await summarize_and_save(
                session_id=client_id,
                history=history,
                model_key=summarizer_model,
            )
            if not _reset_suppress:
                await push_tok(client_id, f"[memory] {status}\n")
        except Exception as _mem_err:
            log.warning(f"cmd_reset: memory summarize failed: {_mem_err}")

    session["history"] = []
    session["tool_subscriptions"] = {}
    session["tool_list_injected"] = False
    model_cfg = LLM_REGISTRY.get(session.get("model", ""), {})
    import plugin_history_default as _phd
    session["history_max_ctx"] = _phd.compute_effective_max_ctx(model_cfg)
    delete_history(client_id)
    await push_tok(client_id, f"Conversation history cleared ({history_len} messages removed).")
    await conditional_push_done(client_id)

async def cmd_tools(client_id: str, arg: str, session: dict):
    """
    !tools                        - show heat status of all tools for current model
    !tools reset                  - decay all subscriptions to cold, re-baseline
    !tools hot <toolsets>         - force-subscribe toolsets (comma or space separated)
    """
    from config import LLM_REGISTRY, LLM_TOOLSETS, LLM_TOOLSET_META
    from agents import _compute_active_tools, _subscribe_toolset

    model_key = session.get("model", "")
    cfg = LLM_REGISTRY.get(model_key, {})
    authorized_toolsets = cfg.get("llm_tools", [])
    subs = session.setdefault("tool_subscriptions", {})

    subcmd = arg.strip().lower() if arg else ""

    if subcmd == "reset":
        session["tool_subscriptions"] = {}
        session["tool_list_injected"] = False
        await push_tok(client_id, "Tool subscriptions cleared. Tool list will re-inject on next turn.")
        await conditional_push_done(client_id)
        return

    if subcmd.startswith("hot"):
        rest = subcmd[3:].strip().replace(",", " ").split()
        if not rest:
            await push_tok(client_id, "Usage: !tools hot <toolset1> [toolset2 ...]")
            await conditional_push_done(client_id)
            return
        activated = []
        unknown = []
        for ts in rest:
            if ts in authorized_toolsets and ts in LLM_TOOLSETS:
                _subscribe_toolset(session, ts, call_count=1)
                activated.append(ts)
            else:
                unknown.append(ts)
        msg_parts = []
        if activated:
            msg_parts.append(f"Force-subscribed: {', '.join(activated)}")
        if unknown:
            msg_parts.append(f"Unknown/unauthorized: {', '.join(unknown)}")
        await push_tok(client_id, "\n".join(msg_parts))
        await conditional_push_done(client_id)
        return

    # Default: show heat status table.
    # Resolve each llm_tools entry: toolset group names are used directly; literal
    # tool names are mapped to their containing toolset via _toolset_for_tool().
    # Multiple literals from the same toolset collapse into one row.
    from agents import _toolset_for_tool
    seen_toolsets: set[str] = set()
    resolved: list[tuple[str, list[str], bool]] = []  # (ts_name, tools_list, is_group)
    for entry in authorized_toolsets:
        if entry in LLM_TOOLSETS:
            # Proper toolset group name
            if entry not in seen_toolsets:
                seen_toolsets.add(entry)
                resolved.append((entry, LLM_TOOLSETS[entry], True))
        else:
            # Literal tool name — find its toolset
            parent_ts = _toolset_for_tool(entry)
            if parent_ts and parent_ts not in seen_toolsets:
                seen_toolsets.add(parent_ts)
                resolved.append((parent_ts, LLM_TOOLSETS[parent_ts], False))
            elif not parent_ts:
                # Orphan tool with no toolset — show as always-active literal
                resolved.append((entry, [entry], False))

    lines = [f"Tool heat status — model: {model_key}"]
    lines.append(f"{'Toolset':<16} {'Status':<18} {'Tools'}")
    lines.append("-" * 70)
    for ts_name, tools_in_set, is_group in resolved:
        meta = LLM_TOOLSET_META.get(ts_name, {})
        if meta.get("always_active", True):
            status = "always-active"
        else:
            sub = subs.get(ts_name, {})
            heat = sub.get("heat", 0)
            call_count = sub.get("call_count", 0)
            heat_curve = meta.get("heat_curve", [])
            cap = heat_curve[-1] if heat_curve else "?"
            status = f"hot heat={heat}/{cap} calls={call_count}" if heat > 0 else "cold"
        lines.append(f"  {ts_name:<14} {status:<18} {', '.join(tools_in_set)}")
    await push_tok(client_id, "\n".join(lines))
    await conditional_push_done(client_id)


async def cmd_session(client_id: str, arg: str):
    """
    Manage sessions: list, attach, or delete.
    Usage:
      !session                - list all sessions (current marked)
      !session <ID> attach    - switch to different session
      !session <ID> delete    - delete a session
    """
    from state import sessions, get_or_create_shorthand_id, get_session_by_shorthand, remove_shorthand_mapping, estimate_history_size, format_session_token_line
    from memory import _mem_plugin_cfg as _mpcfg

    parts = arg.split()

    if not arg:
        # List all sessions with shorthand IDs
        if not sessions:
            await push_tok(client_id, "No active sessions.")
        else:
            lines = ["Active sessions:"]
            for sid, data in sessions.items():
                marker = " (current)" if sid == client_id else ""
                model = data.get("model", "unknown")
                history = data.get("history", [])
                history_len = len(history)
                shorthand_id = get_or_create_shorthand_id(sid)
                peer_ip = data.get("peer_ip")
                ip_str = f", ip={peer_ip}" if peer_ip else ""
                size = estimate_history_size(history)
                char_k = size["char_count"]
                tok_est = size["token_est"]
                size_str = f" (~{char_k:,} chars, ~{tok_est:,} tok est)"
                token_line = format_session_token_line(data)
                _mem_flag = data.get("memory_enabled", None)
                if _mem_flag is not None:
                    mem_str = f", mem={'ON' if _mem_flag else 'OFF'}(session)"
                else:
                    mem_str = f", mem={'ON' if _mpcfg().get('enabled', True) else 'OFF'}(global)"
                lines.append(
                    f"  ID [{shorthand_id}] {sid}: model={model}, "
                    f"history={history_len} msgs{size_str}{ip_str}{mem_str}{marker}"
                )
                lines.append(token_line)
            await push_tok(client_id, "\n".join(lines))
    elif len(parts) == 2:
        target_arg, action = parts[0], parts[1].lower()

        # Try to parse as shorthand ID (integer)
        target_sid = None
        try:
            shorthand_id = int(target_arg)
            target_sid = get_session_by_shorthand(shorthand_id)
            if not target_sid:
                await push_tok(client_id, f"Session ID [{shorthand_id}] not found.")
                await conditional_push_done(client_id)
                return
        except ValueError:
            # Not an integer, treat as full session ID
            target_sid = target_arg

        if action == "attach":
            await push_tok(client_id, f"ERROR: Session switching not supported via llama proxy.\nSession switching requires shell.py client with .aiops_session_id file.")
        elif action == "delete":
            if target_sid in sessions:
                # Get shorthand ID before deleting
                shorthand_id = get_or_create_shorthand_id(target_sid)
                del sessions[target_sid]
                remove_shorthand_mapping(target_sid)
                await push_tok(client_id, f"Deleted session ID [{shorthand_id}]: {target_sid}")
            else:
                await push_tok(client_id, f"Session not found: {target_sid}")
        else:
            await push_tok(client_id, f"Unknown action: {action}\nUse: !session <ID> attach|delete")
    else:
        await push_tok(client_id, "Usage: !session | !session <ID> attach | !session <ID> delete")

    await conditional_push_done(client_id)


async def cmd_sleep(client_id: str, arg: str):
    """
    Sleep for a specified number of seconds.

    !sleep <seconds>  - pause for 1–300 seconds
    """
    import asyncio as _asyncio
    arg = arg.strip()
    if not arg:
        await push_tok(client_id, "Usage: !sleep <seconds>  (1–300)")
        await conditional_push_done(client_id)
        return
    try:
        seconds = int(arg.split()[0])
    except ValueError:
        await push_tok(client_id, f"ERROR: '{arg}' is not a valid integer.\nUsage: !sleep <seconds>")
        await conditional_push_done(client_id)
        return
    if seconds < 1 or seconds > 300:
        await push_tok(client_id, "ERROR: seconds must be between 1 and 300.")
        await conditional_push_done(client_id)
        return
    await push_tok(client_id, f"Sleeping for {seconds} second(s)...")
    await _asyncio.sleep(seconds)
    await push_tok(client_id, f"Done. Slept for {seconds} second(s).")
    await conditional_push_done(client_id)


# ---------------------------------------------------------------------------
# Phase 3 — Unified resource !commands
# Pattern: !<resource> <action> [args...]
# ---------------------------------------------------------------------------

async def cmd_llm_tools(client_id: str, arg: str):
    """!llm_tools <action> [name] [tools]"""
    from tools import _llm_tools_exec
    parts = arg.strip().split(None, 2)
    action = parts[0] if parts else "list"
    name = parts[1] if len(parts) > 1 else ""
    tools = parts[2] if len(parts) > 2 else ""
    result = await _llm_tools_exec(action, name=name, tools=tools)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_model_cfg(client_id: str, arg: str):
    """!model_cfg <action> [name] [field] [value]"""
    from tools import _model_cfg_exec
    parts = arg.strip().split(None, 3)
    action = parts[0] if parts else "list"
    name = parts[1] if len(parts) > 1 else ""
    field = parts[2] if len(parts) > 2 else ""
    value = parts[3] if len(parts) > 3 else ""
    result = await _model_cfg_exec(action, name=name, field=field, value=value)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_sysprompt_cfg(client_id: str, arg: str):
    """!sysprompt_cfg <action> [model] [file|newdir] [content]"""
    from tools import _sysprompt_cfg_exec
    parts = arg.strip().split(None, 3)
    action = parts[0] if parts else "list_dir"
    model = parts[1] if len(parts) > 1 else ""
    file_or_newdir = parts[2] if len(parts) > 2 else ""
    content = parts[3] if len(parts) > 3 else ""
    # Map file/newdir based on action
    if action in ("copy_dir", "set_dir"):
        result = await _sysprompt_cfg_exec(action, model=model, newdir=file_or_newdir)
    elif action == "write":
        result = await _sysprompt_cfg_exec(action, model=model, file=file_or_newdir, content=content)
    else:
        result = await _sysprompt_cfg_exec(action, model=model, file=file_or_newdir)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_config_cfg(client_id: str, arg: str):
    """!config <action> [key] [value]"""
    from tools import _config_cfg_exec
    from state import current_client_id
    parts = arg.strip().split(None, 2)
    action = parts[0] if parts else "list"
    key = parts[1] if len(parts) > 1 else ""
    value = parts[2] if len(parts) > 2 else ""
    token = current_client_id.set(client_id)
    try:
        result = await _config_cfg_exec(action, key=key, value=value)
    finally:
        current_client_id.reset(token)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_limits_cfg(client_id: str, arg: str):
    """!limits <action> [key] [value]"""
    from tools import _limits_cfg_exec
    from state import current_client_id
    parts = arg.strip().split(None, 2)
    action = parts[0] if parts else "list"
    key = parts[1] if len(parts) > 1 else ""
    value = parts[2] if len(parts) > 2 else ""
    token = current_client_id.set(client_id)
    try:
        result = await _limits_cfg_exec(action, key=key, value=value)
    finally:
        current_client_id.reset(token)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_vscode(client_id: str, arg: str):
    """!vscode list|read — browse and pull local Claude Code sessions into chat context."""
    from plugin_claude_vscode_sessions import _cmd_vscode
    from state import current_client_id
    token = current_client_id.set(client_id)
    try:
        result = await _cmd_vscode(arg)
    finally:
        current_client_id.reset(token)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_judge(client_id: str, arg: str, session: dict):
    """!judge — LLM-as-judge configuration and control."""
    from judge import cmd_judge as _judge_cmd
    result = await _judge_cmd(client_id, arg, session)
    await push_tok(client_id, result + "\n")
    await conditional_push_done(client_id)


async def cmd_stream(client_id: str, arg: str, session: dict):
    """!stream [0-3] — show or set response streaming/latency optimization level.

    0: off — ainvoke(), fresh client per request, serial enrich (default, conservative)
    1: LLM client cache per model-key (~100–200ms saved per turn)
    2: + parallel auto-enrich with 300ms timeout (inject if ready, defer if slow)
    3: + astream() sentence-chunking to TTS (first sentence arrives in ~1s)
    """
    LEVELS = {
        "0": "off — ainvoke(), fresh client, serial enrich",
        "1": "LLM client cache per model-key (~100–200ms saved)",
        "2": "+ parallel auto-enrich, 300ms timeout, inject-or-defer",
        "3": "+ astream() sentence-chunking to TTS",
    }
    arg = arg.strip()
    if not arg:
        cur = session.get("stream_level", 0)
        lines = [f"stream_level: {cur}\n"]
        for k, v in LEVELS.items():
            marker = "▶" if str(cur) == k else " "
            lines.append(f" {marker} {k}: {v}")
        await push_tok(client_id, "\n".join(lines) + "\n")
    elif arg in LEVELS:
        session["stream_level"] = int(arg)
        await push_tok(client_id, f"Stream level set to {arg}: {LEVELS[arg]}\n")
    else:
        await push_tok(client_id, "Usage: !stream [0-3]\n" + "\n".join(f"  {k}: {v}" for k, v in LEVELS.items()) + "\n")
    await conditional_push_done(client_id)


async def process_request(client_id: str, text: str, raw_payload: dict, peer_ip: str = None):
    from state import get_or_create_shorthand_id, load_history, load_session_config

    if client_id not in sessions:
        # Enforce max_users limit before creating new session
        max_u = get_max_users()
        if max_u > 0 and len(sessions) >= max_u:
            await push_tok(client_id,
                f"ERROR: Session limit reached ({max_u} active sessions). "
                f"Try again later or ask an administrator to increase !maxusers.")
            await push_done(client_id)
            return
        model_key = raw_payload.get("default_model", DEFAULT_MODEL)
        if model_key not in LLM_REGISTRY:
            model_key = DEFAULT_MODEL
        model_cfg = LLM_REGISTRY.get(model_key, {})
        import plugin_history_default as _phd
        effective_ctx = _phd.compute_effective_max_ctx(model_cfg)
        import time as _time_init
        prior_history = load_history(client_id)
        prior_cfg = load_session_config(client_id)
        _model_tool_suppress = model_cfg.get("tool_suppress", get_default_tool_suppress())
        # Explicit user override (prior_cfg has the key) wins; otherwise model setting wins.
        _effective_tool_suppress = prior_cfg["tool_suppress"] if "tool_suppress" in prior_cfg else _model_tool_suppress
        _effective_mss = model_cfg.get("memory_scan_suppress", False) or prior_cfg.get("memory_scan_suppress", False)
        # agent_call_stream: model False wins (restrictive); model None defers to prior_cfg/default
        _model_stream = model_cfg.get("agent_call_stream", None)
        _effective_stream = (False if _model_stream is False else prior_cfg.get("agent_call_stream", True))
        # stream_level: model sets default if specified; prior_cfg (user !stream command) overrides
        _model_stream_level = model_cfg.get("stream_level", None)
        _effective_stream_level = prior_cfg.get("stream_level", _model_stream_level if _model_stream_level is not None else 0)
        sessions[client_id] = {
            "model": model_key,
            "history": prior_history,
            "history_max_ctx": effective_ctx,
            "tool_preview_length": prior_cfg.get("tool_preview_length", get_default_tool_preview_length()),
            "tool_suppress": _effective_tool_suppress,
            "memory_scan_suppress": _effective_mss,
            "agent_call_stream": _effective_stream,
            "stream_level": _effective_stream_level,
            "_client_id": client_id,
            "created_at": _time_init.time(),
            "tool_subscriptions": {},
            "tool_list_injected": False,
        }
        # Assign shorthand ID when session is created
        get_or_create_shorthand_id(client_id)
        # Age stale short-term memories to long-term on every session start (new or rehydrated)
        import asyncio as _asyncio
        try:
            from memory import age_to_longterm
            _asyncio.create_task(age_to_longterm())
        except Exception:
            pass
    # Store/update peer IP whenever we have it
    if peer_ip:
        sessions[client_id]["peer_ip"] = peer_ip
    session = sessions[client_id]

    # Reject requests from sessions flagged by AIRS async violation
    if session.get("airs_blocked"):
        import time as _time
        session["last_active"] = _time.time()  # keep session alive for admin review
        await push_tok(client_id,
            "[SECURITY] This session has been blocked due to an AI Runtime Security policy "
            "violation detected in a previous response. Contact an administrator to review "
            f"the violation report (report_id: {session.get('airs_block_report_id', 'unknown')}).")
        await push_done(client_id)
        return

    # If session's model was deleted from the registry, fall back to DEFAULT_MODEL
    if session.get("model") not in LLM_REGISTRY:
        old_model = session.get("model", "")
        session["model"] = DEFAULT_MODEL
        await push_tok(client_id,
            f"[WARNING] Model '{old_model}' is no longer available. "
            f"Switched to default model '{DEFAULT_MODEL}'.")

    stripped = text.strip()

    # Check if this is a multi-line message with multiple commands
    lines = stripped.split('\n')
    command_lines = [line.strip() for line in lines if line.strip().startswith('!')]

    # If we have multiple command lines, process them sequentially
    if len(command_lines) > 1:
        # Enable batch mode to suppress push_done in individual commands
        _batch_mode[client_id] = True
        try:
            for cmd_line in command_lines:
                parts = cmd_line[1:].split(maxsplit=1)
                cmd, arg = parts[0].lower(), parts[1].strip() if len(parts) > 1 else ""

                # Route each command
                if cmd == "help":
                    await cmd_help(client_id)
                elif cmd == "reset":
                    await cmd_reset(client_id, session)
                elif cmd == "tools":
                    await cmd_tools(client_id, arg, session)
                elif cmd == "db_query":
                    await cmd_db_query(client_id, arg, session.get("model", ""))
                elif cmd in ("search_ddgs", "search_google", "search_tavily", "search_xai"):
                    engine = cmd[len("search_"):]
                    await cmd_search(client_id, engine, arg)
                elif cmd == "url_extract":
                    await cmd_url_extract(client_id, arg)
                elif cmd == "google_drive":
                    await cmd_google_drive(client_id, arg)
                elif cmd == "get_system_info":
                    await cmd_get_system_info(client_id)
                elif cmd == "llm_list":
                    await cmd_llm_list(client_id)
                elif cmd == "llm_call_invoke":
                    await cmd_llm_call_invoke(client_id, arg)
                elif cmd == "model":
                    if arg:
                        await cmd_set_model(client_id, arg, session)
                    else:
                        await cmd_list_models(client_id, session["model"])
                elif cmd == "session":
                    await cmd_session(client_id, arg)
                elif cmd == "stop":
                    await cmd_stop(client_id)
                elif cmd == "sleep":
                    await cmd_sleep(client_id, arg)
                elif cmd == "llm_tools":
                    await cmd_llm_tools(client_id, arg)
                elif cmd == "model_cfg":
                    await cmd_model_cfg(client_id, arg)
                elif cmd == "sysprompt_cfg":
                    await cmd_sysprompt_cfg(client_id, arg)
                elif cmd == "config":
                    await cmd_config_cfg(client_id, arg)
                elif cmd == "limits":
                    await cmd_limits_cfg(client_id, arg)
                elif cmd == "vscode":
                    await cmd_vscode(client_id, arg)
                elif cmd == "memory":
                    await cmd_memory(client_id, arg)
                elif cmd == "memstats":
                    await cmd_memstats(client_id, session.get("model", ""))
                elif cmd == "membackfill":
                    await cmd_membackfill(client_id, session.get("model", ""))
                elif cmd == "memreconcile":
                    await cmd_memreconcile(client_id, session.get("model", ""))
                elif cmd == "memreview":
                    await cmd_memreview(client_id, arg, session.get("model", ""))
                elif cmd == "memage":
                    await cmd_memage(client_id)
                elif cmd == "memtrim":
                    await cmd_memtrim(client_id, arg, session.get("model", ""))
                elif cmd == "stream":
                    await cmd_stream(client_id, arg, session)
                elif get_plugin_command(cmd) is not None:
                    await cmd_plugin_command(client_id, cmd, arg)
                else:
                    await push_tok(client_id, f"Unknown command: !{cmd}\nUse !help to see available commands.\n")

                # Add newline between command outputs for readability
                await push_tok(client_id, "\n")
        finally:
            # Disable batch mode
            _batch_mode[client_id] = False

        # After processing all commands, check if there's non-command text to send to LLM
        non_command_text = '\n'.join([line for line in lines if not line.strip().startswith('!')]).strip()
        if non_command_text:
            # Process the remaining text as a normal message to the LLM
            stripped = non_command_text
        else:
            # Only commands, no LLM interaction needed
            await conditional_push_done(client_id)
            return

    # Single command handling (original logic)
    elif stripped.startswith("!"):
        parts = stripped[1:].split(maxsplit=1)
        cmd, arg = parts[0].lower(), parts[1].strip() if len(parts) > 1 else ""

        # Command routing with validation
        if cmd == "help":
            await cmd_help(client_id)
            return
        if cmd == "reset":
            await cmd_reset(client_id, session)
            return
        if cmd == "tools":
            await cmd_tools(client_id, arg, session)
            return
        if cmd == "db_query":
            await cmd_db_query(client_id, arg, session.get("model", ""))
            return
        if cmd in ("search_ddgs", "search_google", "search_tavily", "search_xai"):
            engine = cmd[len("search_"):]
            await cmd_search(client_id, engine, arg)
            return
        if cmd == "url_extract":
            await cmd_url_extract(client_id, arg)
            return
        if cmd == "google_drive":
            await cmd_google_drive(client_id, arg)
            return
        if cmd == "get_system_info":
            await cmd_get_system_info(client_id)
            return
        if cmd == "llm_list":
            await cmd_llm_list(client_id)
            return
        if cmd == "llm_call_invoke":
            await cmd_llm_call_invoke(client_id, arg)
            return
        if cmd == "model":
            if arg:
                await cmd_set_model(client_id, arg, session)
            else:
                await cmd_list_models(client_id, session["model"])
            return
        if cmd == "session":
            await cmd_session(client_id, arg)
            return
        if cmd == "sleep":
            await cmd_sleep(client_id, arg)
            return
        if cmd == "llm_tools":
            await cmd_llm_tools(client_id, arg)
            return
        if cmd == "model_cfg":
            await cmd_model_cfg(client_id, arg)
            return
        if cmd == "sysprompt_cfg":
            await cmd_sysprompt_cfg(client_id, arg)
            return
        if cmd == "config":
            await cmd_config_cfg(client_id, arg)
            return
        if cmd == "limits":
            await cmd_limits_cfg(client_id, arg)
            return
        if cmd == "vscode":
            await cmd_vscode(client_id, arg)
            return
        if cmd == "memory":
            await cmd_memory(client_id, arg)
            return
        if cmd == "memstats":
            await cmd_memstats(client_id, session.get("model", ""))
            return
        if cmd == "membackfill":
            await cmd_membackfill(client_id, session.get("model", ""))
            return
        if cmd == "memreconcile":
            await cmd_memreconcile(client_id, session.get("model", ""))
            return
        if cmd == "memreview":
            await cmd_memreview(client_id, arg, session.get("model", ""))
            return
        if cmd == "memage":
            await cmd_memage(client_id)
            return
        if cmd == "memtrim":
            await cmd_memtrim(client_id, arg, session.get("model", ""))
            return
        if cmd == "stream":
            await cmd_stream(client_id, arg, session)
            return
        if cmd == "judge":
            await cmd_judge(client_id, arg, session)
            return
        if cmd == "stop":
            await cmd_stop(client_id)
            return

        # Plugin-registered commands (e.g. !tmux, !tmux_call_limit from plugin_tmux)
        if get_plugin_command(cmd) is not None:
            await cmd_plugin_command(client_id, cmd, arg)
            return

        # Catch-all for unknown commands - don't pass to LLM
        await push_tok(client_id, f"Unknown command: !{cmd}\nUse !help to see available commands.")
        await conditional_push_done(client_id)
        return

    # @<model> per-turn model switch
    # Syntax: @ModelName <prompt text>
    # - If @model is the current model: strip prefix, continue as normal
    # - If @model is unknown: return error, don't dispatch
    # - Otherwise: temporarily switch to named model for this turn, restore after
    temp_model = None
    if stripped.startswith("@"):
        first_space = stripped.find(" ")
        if first_space > 1:
            model_token = stripped[1:first_space]  # strip leading @
            rest = stripped[first_space:].strip()
        else:
            model_token = stripped[1:]
            rest = ""
        if model_token in LLM_REGISTRY:
            if model_token == session["model"]:
                # Same model — strip prefix, keep temp_model flag for this turn
                session["_temp_model_active"] = True
                temp_model = session["model"]  # set so finally block clears the flag
                stripped = rest
            else:
                # Different model — temp switch for this turn
                temp_model = session["model"]
                session["model"] = model_token
                session["_temp_model_active"] = True
                stripped = rest
        else:
            available = ", ".join(LLM_REGISTRY.keys())
            await push_tok(client_id, f"ERROR: Unknown model '@{model_token}'\nAvailable models: {available}")
            await conditional_push_done(client_id)
            return

    import time as _time
    session["last_active"] = _time.time()
    model_cfg = LLM_REGISTRY.get(session["model"], {})
    session["history"].append({"role": "user", "content": stripped})
    session["history"] = _run_history_chain(session["history"], session, model_cfg)

    try:
        final = await dispatch_llm(session["model"], session["history"], client_id)
        if final:
            # Strip memory_save() calls from final when memory_scan is disabled —
            # keeps history and conv_log clean of inline save noise.
            if not model_cfg.get("memory_scan", False):
                from agents import _strip_memory_calls
                final = _strip_memory_calls(final)
            session["history"].append({"role": "assistant", "content": final})
            # Post-response chain pass: plugins that inspect history[-1]["role"] == "assistant"
            # can filter or replace the response (e.g. security scanning).
            session["history"] = _run_history_chain(session["history"], session, model_cfg)
            # Verbatim conversation logging — save user prompt + assistant response as paired
            # memory rows when the model has conv_log enabled.
            # session["memory_enabled"] overrides the global plugin config (per-session toggle).
            _sess_mem = session.get("memory_enabled", None)
            log.debug(f"conv_log gate: conv_log={model_cfg.get('conv_log')} sess_mem={_sess_mem} model={session['model']}")
            if model_cfg.get("conv_log") and (_sess_mem is None or _sess_mem):
                try:
                    from memory import save_conversation_turn
                    set_model_context(session.get("model", ""))
                    _, _, _topic = await save_conversation_turn(
                        user_text=stripped,
                        assistant_text=final,
                        session_id=client_id,
                    )
                    if _topic:
                        session["current_topic"] = _topic
                except Exception as _cl_err:
                    import logging as _log
                    _log.getLogger("routes").warning(f"conv_log save failed: {_cl_err}")
        else:
            # Remove dangling user message if LLM returned empty — prevents consecutive
            # user turns in history, which causes Gemini to return empty on next request
            if session["history"] and session["history"][-1]["role"] == "user":
                session["history"].pop()
    except asyncio.CancelledError:
        # Remove the dangling user message so history stays consistent
        if session["history"] and session["history"][-1]["role"] == "user":
            session["history"].pop()
        raise
    finally:
        if temp_model is not None:
            session["model"] = temp_model
            session.pop("_temp_model_active", None)
        active_tasks.pop(client_id, None)

async def endpoint_submit(request: Request) -> JSONResponse:
    try: payload = await request.json()
    except: return JSONResponse({"status": "error"}, 400)

    client_id, text = payload.get("client_id"), payload.get("text", "")
    if not client_id or not text: return JSONResponse({"error": "Missing fields"}, 400)

    peer_ip = request.client.host if request.client else None
    await cancel_active_task(client_id)
    task = asyncio.create_task(process_request(client_id, text, payload, peer_ip=peer_ip))
    active_tasks[client_id] = task
    return JSONResponse({"status": "OK"})

async def endpoint_stream(request: Request):
    client_id = request.query_params.get("client_id")
    if not client_id: return JSONResponse({"error": "Missing client_id"}, 400)

    # Register session on connect so it shows up in !session before first message
    from state import get_or_create_shorthand_id, load_history, load_session_config
    if client_id not in sessions:
        import plugin_history_default as _phd
        _mcfg = LLM_REGISTRY.get(DEFAULT_MODEL, {})
        prior_history = load_history(client_id)
        prior_cfg = load_session_config(client_id)
        _model_tool_suppress = _mcfg.get("tool_suppress", get_default_tool_suppress())
        # Explicit user override (prior_cfg has the key) wins; otherwise model setting wins.
        _effective_tool_suppress = prior_cfg["tool_suppress"] if "tool_suppress" in prior_cfg else _model_tool_suppress
        _effective_mss = _mcfg.get("memory_scan_suppress", False) or prior_cfg.get("memory_scan_suppress", False)
        _model_stream = _mcfg.get("agent_call_stream", None)
        _effective_stream = (False if _model_stream is False else prior_cfg.get("agent_call_stream", True))
        _model_stream_level = _mcfg.get("stream_level", None)
        _effective_stream_level = prior_cfg.get("stream_level", _model_stream_level if _model_stream_level is not None else 0)
        sessions[client_id] = {
            "model": DEFAULT_MODEL,
            "history": prior_history,
            "history_max_ctx": _phd.compute_effective_max_ctx(_mcfg),
            "tool_preview_length": prior_cfg.get("tool_preview_length", get_default_tool_preview_length()),
            "tool_suppress": _effective_tool_suppress,
            "memory_scan_suppress": _effective_mss,
            "agent_call_stream": _effective_stream,
            "stream_level": _effective_stream_level,
            "_client_id": client_id,
            "tool_subscriptions": {},
            "tool_list_injected": False,
        }
        get_or_create_shorthand_id(client_id)
        # Age stale short-term memories to long-term on every session start (new or rehydrated)
        import asyncio as _asyncio
        try:
            from memory import age_to_longterm
            _asyncio.create_task(age_to_longterm())
        except Exception:
            pass
    peer_ip = request.client.host if request.client else None
    if peer_ip:
        sessions[client_id]["peer_ip"] = peer_ip

    # Fix stale model in session (e.g. model was deleted while server was running)
    if sessions[client_id].get("model") not in LLM_REGISTRY:
        sessions[client_id]["model"] = DEFAULT_MODEL

    q = await get_queue(client_id)
    # Push current model so client knows what model is active immediately
    q.put_nowait({"t": "model", "d": sessions[client_id]["model"]})

    async def generator() -> AsyncGenerator[dict, None]:
        while True:
            if await request.is_disconnected(): break
            try:
                item = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield {"comment": "keepalive"}
                continue
            
            t = item.get("t")
            if t == "tok": yield {"data": item["d"]}
            elif t == "done": yield {"event": "done", "data": ""}
            elif t == "err": yield {"event": "error", "data": json.dumps({"error": item["d"]})}
            elif t == "model": yield {"event": "model", "data": item["d"]}
            elif t == "gate": yield {"event": "gate", "data": item["d"]}

    return EventSourceResponse(generator())

async def endpoint_stop(request: Request) -> JSONResponse:
    """Cancel the active LLM job for a client without starting a new one."""
    try: payload = await request.json()
    except: return JSONResponse({"error": "json"}, 400)
    client_id = payload.get("client_id")
    if not client_id: return JSONResponse({"error": "Missing client_id"}, 400)
    cancelled = await cancel_active_task(client_id)
    if cancelled:
        await push_done(client_id)
    return JSONResponse({"status": "OK", "cancelled": cancelled})


async def endpoint_gate_respond(request: Request) -> JSONResponse:
    """
    Resolve a pending gate request for a client.

    POST /gate_respond
    Body: {"client_id": "...", "approved": true|false}

    Returns {"status": "ok", "resolved": true} if a gate was pending,
    or {"status": "ok", "resolved": false} if no gate was waiting.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    client_id = payload.get("client_id")
    approved = payload.get("approved")
    if not client_id:
        return JSONResponse({"error": "Missing client_id"}, status_code=400)
    if not isinstance(approved, bool):
        return JSONResponse({"error": "approved must be boolean"}, status_code=400)

    from state import resolve_gate
    resolved = resolve_gate(client_id, approved)
    return JSONResponse({"status": "ok", "resolved": resolved})


async def endpoint_health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "models": list(LLM_REGISTRY.keys()),
    })

async def endpoint_list_sessions(request: Request) -> JSONResponse:
    """List all active sessions with metadata."""
    from state import sessions, sse_queues, get_or_create_shorthand_id, estimate_history_size

    client_id_filter = request.query_params.get("client_id")

    session_list = []
    for cid, data in sessions.items():
        if client_id_filter and cid != client_id_filter:
            continue
        history = data.get("history", [])
        size = estimate_history_size(history)
        session_list.append({
            "client_id": cid,
            "shorthand_id": get_or_create_shorthand_id(cid),
            "model": data.get("model", "unknown"),
            "history_length": len(history),
            "history_chars": size["char_count"],
            "history_token_est": size["token_est"],
            "tokens_in_total": data.get("tokens_in_total", 0),
            "tokens_out_total": data.get("tokens_out_total", 0),
            "tokens_in_last": data.get("tokens_in_last"),
            "tokens_out_last": data.get("tokens_out_last"),
            "peer_ip": data.get("peer_ip"),
            "memory_enabled": data.get("memory_enabled", None),
        })

    return JSONResponse({"sessions": session_list})

async def endpoint_delete_session(request: Request) -> JSONResponse:
    """Delete a specific session by ID."""
    from state import sessions, sse_queues

    sid = request.path_params.get("sid")

    if sid in sessions:
        del sessions[sid]
        # Also clean up associated queue
        if sid in sse_queues:
            del sse_queues[sid]
        return JSONResponse({"status": "OK", "deleted": sid})

    return JSONResponse({"error": "session not found"}, 404)