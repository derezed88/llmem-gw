"""
Temporal Inference Engine — periodic background task.

Analyzes short-term memory topics to identify "interest" queries worth
running against recall_temporal. Checks samaritan_temporal for existing
results before issuing new queries.

Flow each cycle:
  1. Load recent ST memory rows (distinct topics + content snippets).
  2. Send to a lightweight LLM (summarizer-gemini) with a prompt:
     "Given these recent memory topics, what temporal pattern queries
      would be insightful? Return JSON array of query specs."
  3. For each proposed query, check samaritan_temporal for existing match.
  4. Run recall_temporal(new=True, source='inferred') for missing queries.
  5. Results auto-cached by recall_temporal's cache_store logic.

Config (plugins-enabled.json → memory.temporal):
  inference_enabled       bool   — master switch
  inference_interval_h    float  — hours between cycles (default 3)
  inference_model         str    — LLM for topic analysis (default summarizer-gemini)
  inference_max_queries   int    — max new queries per cycle (default 5)
  inference_min_st_rows   int    — skip cycle if fewer ST rows than this (default 10)
  inference_cache_ttl_hours int  — skip queries with cache younger than this (default 24)
"""

import asyncio
import json
import logging

log = logging.getLogger("temporal_inference")

# ---------------------------------------------------------------------------
# Runtime stats — shared between background task and !timers command
# ---------------------------------------------------------------------------

_stats: dict = {
    "runs": 0,
    "queries_proposed": 0,
    "queries_executed": 0,
    "queries_cached": 0,
    "errors": 0,
    "last_run_at": None,      # ISO string
    "last_run_duration_s": None,
    "last_run_proposed": 0,
    "last_run_executed": 0,
    "last_error": None,
}


_wake_event: asyncio.Event | None = None


def trigger_now() -> None:
    """Wake the sleeping temporal inference task immediately."""
    if _wake_event:
        _wake_event.set()


def get_temporal_inference_stats() -> dict:
    return dict(_stats)


def _ti_cfg() -> dict:
    """Load temporal inference config from plugins-enabled.json."""
    try:
        with open("plugins-enabled.json") as f:
            cfg = json.load(f)
        mem = cfg.get("plugin_config", {}).get("memory", {})
        tcfg = mem.get("temporal", {}) if isinstance(mem.get("temporal"), dict) else {}
    except Exception:
        tcfg = {}
    return {
        "enabled": tcfg.get("inference_enabled", False),
        "interval_h": float(tcfg.get("inference_interval_h", 3)),
        "model": tcfg.get("inference_model", ""),
        "max_queries": int(tcfg.get("inference_max_queries", 5)),
        "min_st_rows": int(tcfg.get("inference_min_st_rows", 10)),
        "cache_ttl_hours": int(tcfg.get("inference_cache_ttl_hours", 24)),
    }


async def _load_recent_topics() -> list[dict]:
    """Load distinct topics from ST with occurrence counts and sample content."""
    from database import execute_sql, set_model_context
    from config import DEFAULT_MODEL
    from memory import _ST
    set_model_context(DEFAULT_MODEL)
    sql = (
        f"SELECT topic, COUNT(*) AS cnt, "
        f"GROUP_CONCAT(LEFT(content, 100) ORDER BY created_at DESC SEPARATOR ' | ') AS samples "
        f"FROM {_ST()} "
        f"WHERE source IN ('user', 'assistant') "
        f"GROUP BY topic "
        f"ORDER BY cnt DESC "
        f"LIMIT 30"
    )
    raw = await execute_sql(sql)
    # Parse the formatted table output into dicts
    lines = raw.strip().split("\n")
    if len(lines) < 3:
        return []
    headers = [h.strip() for h in lines[0].split("|")]
    results = []
    for line in lines[2:]:  # skip header + separator
        vals = [v.strip() for v in line.split("|")]
        if len(vals) >= len(headers):
            results.append(dict(zip(headers, vals)))
    return results


async def _existing_temporal_keys(ttl_hours: int) -> set[str]:
    """Return set of query_keys that already have recent cached results."""
    from database import execute_sql
    from tools import _temporal_table
    tbl = _temporal_table()
    sql = (
        f"SELECT query_key FROM {tbl} "
        f"WHERE created_at >= DATE_SUB(NOW(), INTERVAL {ttl_hours} HOUR)"
    )
    raw = await execute_sql(sql)
    lines = raw.strip().split("\n")
    if len(lines) < 3 or "(no rows)" in raw:
        return set()
    return {line.strip() for line in lines[2:]}


async def _propose_queries(topics: list[dict], model: str, max_queries: int) -> list[dict]:
    """Use an LLM to propose temporal queries based on recent topic patterns."""
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import HumanMessage

    topic_text = "\n".join(
        f"- {t.get('topic', '?')} ({t.get('cnt', '?')} entries): {t.get('samples', '')[:200]}"
        for t in topics[:20]
    )

    prompt = (
        "You analyze memory topics to propose temporal pattern queries.\n"
        "Given these recent memory topics and sample content:\n\n"
        f"{topic_text}\n\n"
        "Propose up to {max_q} temporal pattern queries that would reveal interesting "
        "time-based patterns (routines, habits, recurring events). "
        "Focus on topics with enough entries to show patterns (3+ occurrences).\n\n"
        "Return ONLY a JSON array. Each element:\n"
        '{{"query": "<keyword>", "group_by": "<hour|day_of_week|date|week|month>", '
        '"day_of_week": "<optional>", "time_range": "<optional>", "lookback_days": <int>}}\n\n'
        "Rules:\n"
        "- query should be a simple keyword from the topic/content (e.g. 'Lee', 'walk', 'gym')\n"
        "- group_by should match what would reveal patterns (day_of_week for routines, hour for schedules)\n"
        "- Only propose queries where the data suggests a temporal pattern exists\n"
        "- Do not propose queries for one-off events\n"
        f"- Maximum {max_queries} queries\n"
        "Return ONLY the JSON array, no markdown, no explanation."
    ).format(max_q=max_queries)

    try:
        llm = _build_lc_llm(model)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = _content_to_str(response.content).strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        proposals = json.loads(text)
        if isinstance(proposals, list):
            return proposals[:max_queries]
    except Exception as e:
        log.warning(f"_propose_queries: LLM call or parse failed: {e}")
    return []


async def run_temporal_inference() -> dict:
    """
    Single inference cycle. Returns stats dict:
    {proposed: N, skipped_cached: N, executed: N, errors: N}
    """
    from database import set_db_override, list_managed_databases
    from tools import _recall_temporal_exec, _temporal_query_key

    cfg = _ti_cfg()
    stats = {"proposed": 0, "skipped_cached": 0, "executed": 0, "errors": 0}

    model_key = cfg["model"]
    if not model_key:
        from config import get_model_role
        try:
            model_key = get_model_role("temporal_inference")
        except KeyError:
            model_key = "summarizer-gemini"

    for db_name in list_managed_databases():
        set_db_override(db_name)
        try:
            # 1. Load recent topics
            topics = await _load_recent_topics()
            if len(topics) < cfg["min_st_rows"]:
                log.debug(f"temporal_inference[{db_name}]: only {len(topics)} topics, below min {cfg['min_st_rows']}")
                continue

            # 2. Get existing cached keys
            existing = await _existing_temporal_keys(cfg["cache_ttl_hours"])

            # 3. Propose queries via LLM
            proposals = await _propose_queries(topics, model_key, cfg["max_queries"])
            stats["proposed"] += len(proposals)
            log.info(f"temporal_inference[{db_name}]: {len(proposals)} queries proposed from {len(topics)} topics")

            # 4. Execute missing queries
            for p in proposals:
                q = p.get("query", "")
                gb = p.get("group_by", "day_of_week")
                dow = p.get("day_of_week", "")
                tr = p.get("time_range", "")
                lb = int(p.get("lookback_days", 30))

                qkey = _temporal_query_key(q, gb, dow, tr)
                if qkey in existing:
                    stats["skipped_cached"] += 1
                    log.debug(f"temporal_inference[{db_name}]: cached, skipping: {qkey}")
                    continue

                try:
                    await _recall_temporal_exec(
                        query=q, group_by=gb, day_of_week=dow,
                        time_range=tr, lookback_days=lb,
                        new=True, source="inferred",
                    )
                    stats["executed"] += 1
                    existing.add(qkey)
                    log.info(f"temporal_inference[{db_name}]: executed: {qkey}")
                    import asyncio as _asyncio
                    try:
                        import notifier as _notifier
                        _asyncio.ensure_future(_notifier.fire_event(
                            "temporal_pattern_inferred",
                            f"query={q!r} group_by={gb}",
                            f"lookback={lb}d",
                        ))
                    except Exception:
                        pass
                except Exception as e:
                    stats["errors"] += 1
                    log.warning(f"temporal_inference[{db_name}]: query failed: {qkey}: {e}")
        except Exception as e:
            stats["errors"] += 1
            log.warning(f"temporal_inference[{db_name}]: error: {e}")
    set_db_override("")

    return stats


async def temporal_inference_task() -> None:
    """Long-running asyncio task. Loops every inference_interval_h hours."""
    import time as _time
    from datetime import datetime, timezone
    from timer_registry import register_timer, timer_start, timer_end, timer_sleep, timer_disabled

    global _wake_event
    _wake_event = asyncio.Event()

    register_timer("temporal_inference", "3h")
    # Initial delay — let the system warm up
    timer_sleep("temporal_inference", 120)
    await asyncio.sleep(120)

    while True:
        t0 = None
        try:
            cfg = _ti_cfg()
            if not cfg["enabled"]:
                timer_disabled("temporal_inference")
                await asyncio.sleep(300)
                continue

            interval_h = cfg["interval_h"]
            if interval_h <= 0:
                timer_disabled("temporal_inference")
                await asyncio.sleep(3600)
                continue

            register_timer("temporal_inference", f"{interval_h}h")
            t0 = timer_start("temporal_inference")
            cycle_stats = await run_temporal_inference()

            # Update module-level stats
            _stats["runs"] += 1
            _stats["queries_proposed"] += cycle_stats.get("proposed", 0)
            _stats["queries_executed"] += cycle_stats.get("executed", 0)
            _stats["queries_cached"] += cycle_stats.get("skipped_cached", 0)
            _stats["errors"] += cycle_stats.get("errors", 0)
            _stats["last_run_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _stats["last_run_proposed"] = cycle_stats.get("proposed", 0)
            _stats["last_run_executed"] = cycle_stats.get("executed", 0)

            timer_end("temporal_inference", t0)
            log.info(
                f"temporal_inference cycle complete: "
                f"proposed={cycle_stats['proposed']}, cached={cycle_stats['skipped_cached']}, "
                f"executed={cycle_stats['executed']}, errors={cycle_stats['errors']}"
            )
        except Exception as e:
            log.error(f"temporal_inference_task error: {e}")
            _stats["errors"] += 1
            _stats["last_error"] = str(e)[:200]
            if t0 is not None:
                timer_end("temporal_inference", t0, error=str(e))

        try:
            cfg = _ti_cfg()
            sleep_sec = max(300, cfg["interval_h"] * 3600)
        except Exception:
            sleep_sec = 10800  # 3h fallback

        # Backoff: jump to 6h when both contradiction and prospective at cap
        effective_desc = None
        try:
            from state import backoff_interval, idle_seconds, fmt_interval
            import json as _json
            with open(os.path.join(os.path.dirname(__file__), "plugins-enabled.json")) as _f:
                _pcog = _json.load(_f).get("plugin_config", {}).get("proactive_cognition", {})
            contra_eff = backoff_interval(int(_pcog.get("contradiction_interval_m", 2)), 60)
            prosp_eff = backoff_interval(int(_pcog.get("prospective_interval_m", 1)), 120)
            if contra_eff >= 60 and prosp_eff >= 120:
                sleep_sec = 21600  # 6 hours
                log.info(f"temporal_inference: backoff → 6h (siblings at cap, idle {idle_seconds()/60:.0f}m)")
            effective_desc = fmt_interval(sleep_sec / 60)
        except Exception:
            pass  # keep original sleep_sec on any error

        timer_sleep("temporal_inference", sleep_sec, interval_desc=effective_desc)
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            log.info("temporal_inference_task: woken early by trigger")
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass
