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
from database import execute_sql
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
        "\n"
        "Database:\n"
        "  !db_query <sql>                           - run SQL directly (no LLM)\n"
        "\n"
        "Memory:\n"
        "  !memory                                   - list all short-term memories\n"
        "  !memory list [short|long]                 - list by tier\n"
        "  !memory show <id> [short|long]            - show one row in full\n"
        "  !memory update <id> [tier=short] [importance=N] [content=text] [topic=label]\n"
        "  !memstats                                 - memory system health dashboard\n"
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

async def cmd_db_query(client_id: str, sql: str):
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
    from memory import load_short_term, load_long_term, update_memory
    parts = arg.split(maxsplit=1)
    subcmd = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if subcmd in ("list", "short", "long", ""):
        tier = "long" if subcmd == "long" else "short"
        if subcmd == "list" and rest in ("long", "short"):
            tier = rest
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
            "  !memory                              - list short-term memories\n"
            "  !memory list [short|long]            - list by tier\n"
            "  !memory show <id> [short|long]       - show one row\n"
            "  !memory update <id> [tier=short] [importance=N] [content=text] [topic=label]")

    await conditional_push_done(client_id)


async def cmd_memstats(client_id: str):
    """
    !memstats — memory system health dashboard.
    Shows row counts, topic distribution, importance spread, aging history,
    summarizer runs, dedup config, and last activity timestamps.
    """
    from database import execute_sql
    from memory import _ST, _LT, _SUM, _age_cfg, _mem_plugin_cfg

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
    st_count = _int_from(await q(f"SELECT COUNT(*) FROM {_ST}"))
    lt_count = _int_from(await q(f"SELECT COUNT(*) FROM {_LT}"))
    sum_count = _int_from(await q(f"SELECT COUNT(*) FROM {_SUM}"))
    lines.append(f"**Tier counts**")
    lines.append(f"  short-term : {st_count} rows")
    lines.append(f"  long-term  : {lt_count} rows")
    lines.append(f"  summaries  : {sum_count} rows\n")

    # ── Short-term: topic breakdown ───────────────────────────────────────────
    st_topics_raw = await q(
        f"SELECT topic, COUNT(*) as n, ROUND(AVG(importance),1) as avg_imp "
        f"FROM {_ST} GROUP BY topic ORDER BY n DESC LIMIT 20"
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
        f"FROM {_LT} GROUP BY topic ORDER BY n DESC LIMIT 20"
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
        f"SELECT importance, COUNT(*) as n FROM {_ST} GROUP BY importance ORDER BY importance"
    )
    imp_rows = _rows_from(imp_raw)
    if imp_rows:
        lines.append("**Short-term importance distribution**")
        dist = "  " + "  ".join(f"[{r[0]}]={r[1]}" for r in imp_rows if len(r) >= 2)
        lines.append(dist)
        lines.append("")

    # ── Source breakdown (short-term) ─────────────────────────────────────────
    src_raw = await q(
        f"SELECT source, COUNT(*) as n FROM {_ST} GROUP BY source ORDER BY n DESC"
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
        f"SELECT COUNT(*) FROM {_LT} WHERE shortterm_id IS NOT NULL AND shortterm_id > 0"
    )
    lt_aged = _int_from(lt_from_st_raw)
    lt_direct_raw = await q(
        f"SELECT COUNT(*) FROM {_LT} WHERE shortterm_id IS NULL OR shortterm_id = 0"
    )
    lt_direct = _int_from(lt_direct_raw)
    lines.append("**Long-term origin**")
    lines.append(f"  aged from short-term : {lt_aged} rows")
    lines.append(f"  direct saves         : {lt_direct} rows\n")

    # ── Last activity timestamps ──────────────────────────────────────────────
    st_newest_raw = await q(f"SELECT MAX(created_at) FROM {_ST}")
    st_oldest_raw = await q(f"SELECT MIN(created_at) FROM {_ST}")
    st_lru_raw    = await q(f"SELECT MIN(last_accessed) FROM {_ST}")
    lt_newest_raw = await q(f"SELECT MAX(created_at) FROM {_LT}")
    sum_newest_raw = await q(f"SELECT MAX(created_at) FROM {_SUM}")

    def _scalar(raw: str) -> str:
        for line in raw.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("MAX") and not line.startswith("MIN") and set(line) > set("-+|"):
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
            f"SELECT model_used, COUNT(*) as n FROM {_SUM} GROUP BY model_used ORDER BY n DESC"
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
            vstats = await vec.get_stats()
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

    # ── Config snapshot ───────────────────────────────────────────────────────
    mem_cfg = _mem_plugin_cfg()
    age_cfg = _age_cfg()
    lines.append("**Config (plugins-enabled.json)**")
    master_on = mem_cfg.get("enabled", True)
    lines.append(f"  {'enabled (master)':<28}: {'on' if master_on else 'OFF'}")
    features = ("context_injection", "reset_summarize", "post_response_scan", "fuzzy_dedup", "vector_search_qdrant")
    for f in features:
        val = mem_cfg.get(f, True)
        lines.append(f"  {f:<28}: {'on' if val else 'OFF'}{' (inactive—master off)' if not master_on else ''}")
    lines.append(f"  {'fuzzy_dedup_threshold':<28}: {mem_cfg.get('fuzzy_dedup_threshold', 0.78):.2f}")
    lines.append(f"  {'summarizer_model':<28}: {mem_cfg.get('summarizer_model', 'summarizer-anthropic')}")
    lines.append(f"  {'auto_memory_age':<28}: {'on' if age_cfg['auto_memory_age'] else 'OFF'}")
    lines.append(f"  {'memory_age_entrycount':<28}: {age_cfg['memory_age_entrycount']}")

    def _timer(v: int) -> str:
        return "disabled" if v == -1 else f"{v} min"

    lines.append(f"  {'memory_age_count_timer':<28}: {_timer(age_cfg['memory_age_count_timer'])}")
    lines.append(f"  {'memory_age_trigger_minutes':<28}: {age_cfg['memory_age_trigger_minutes']} min ({age_cfg['memory_age_trigger_minutes']//60}h)")
    lines.append(f"  {'memory_age_minutes_timer':<28}: {_timer(age_cfg['memory_age_minutes_timer'])}")

    # Pressure indicator
    if st_count > 0:
        limit = age_cfg["memory_age_entrycount"]
        pct = int(st_count / limit * 100) if limit > 0 else 0
        bar = "#" * min(20, pct // 5) + "." * max(0, 20 - pct // 5)
        lines.append(f"\n  ST pressure: [{bar}] {pct}% of {limit} row limit")

    await push_tok(client_id, "\n".join(lines))
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

    # Summarize departing history into short-term memory before clearing
    from agents import _memory_feature, _memory_cfg
    if history_len >= 4 and _memory_feature("reset_summarize"):
        try:
            from memory import summarize_and_save
            summarizer_model = _memory_cfg().get("summarizer_model", "summarizer-anthropic")
            await push_tok(client_id, "[memory] Summarizing session to memory...\n")
            status = await summarize_and_save(
                session_id=client_id,
                history=history,
                model_key=summarizer_model,
            )
            await push_tok(client_id, f"[memory] {status}\n")
        except Exception as _mem_err:
            log.warning(f"cmd_reset: memory summarize failed: {_mem_err}")

    session["history"] = []
    model_cfg = LLM_REGISTRY.get(session.get("model", ""), {})
    import plugin_history_default as _phd
    session["history_max_ctx"] = _phd.compute_effective_max_ctx(model_cfg)
    delete_history(client_id)
    await push_tok(client_id, f"Conversation history cleared ({history_len} messages removed).")
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
                lines.append(
                    f"  ID [{shorthand_id}] {sid}: model={model}, "
                    f"history={history_len} msgs{size_str}{ip_str}{marker}"
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
        sessions[client_id] = {
            "model": model_key,
            "history": prior_history,
            "history_max_ctx": effective_ctx,
            "tool_preview_length": prior_cfg.get("tool_preview_length", get_default_tool_preview_length()),
            "tool_suppress": prior_cfg.get("tool_suppress", _model_tool_suppress),
            "_client_id": client_id,
            "created_at": _time_init.time(),
        }
        if "agent_call_stream" in prior_cfg:
            sessions[client_id]["agent_call_stream"] = prior_cfg["agent_call_stream"]
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
                elif cmd == "db_query":
                    await cmd_db_query(client_id, arg)
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
                    await cmd_memstats(client_id)
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
        if cmd == "db_query":
            await cmd_db_query(client_id, arg)
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
            await cmd_memstats(client_id)
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
            session["history"].append({"role": "assistant", "content": final})
            # Post-response chain pass: plugins that inspect history[-1]["role"] == "assistant"
            # can filter or replace the response (e.g. security scanning).
            session["history"] = _run_history_chain(session["history"], session, model_cfg)
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
        sessions[client_id] = {
            "model": DEFAULT_MODEL,
            "history": prior_history,
            "history_max_ctx": _phd.compute_effective_max_ctx(_mcfg),
            "tool_preview_length": prior_cfg.get("tool_preview_length", get_default_tool_preview_length()),
            "tool_suppress": prior_cfg.get("tool_suppress", _model_tool_suppress),
            "_client_id": client_id,
        }
        if "agent_call_stream" in prior_cfg:
            sessions[client_id]["agent_call_stream"] = prior_cfg["agent_call_stream"]
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