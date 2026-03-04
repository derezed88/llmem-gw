"""
memory.py — Tiered memory system

Tiers:
  Short-term  : MySQL (hot, injected every request)
  Long-term   : MySQL (aged-out, on-demand recall)
  Archive     : Google Drive (bulk export, future)

Public API:
  save_memory(topic, content, importance, source, session_id)
  load_short_term(limit, min_importance) -> list[dict]
  age_by_count(max_rows) -> int
  age_by_minutes(trigger_minutes, max_rows) -> int
  load_context_block(limit, min_importance) -> str   # ready to prepend to prompt
  summarize_and_save(session_id, history, model_key) -> str
"""

import asyncio
import difflib
import json
import logging
import os
from datetime import datetime, timezone

from database import execute_sql, execute_insert

log = logging.getLogger("memory")

# ---------------------------------------------------------------------------
# Load table names from db-config.json (instance-specific, gitignored)
# ---------------------------------------------------------------------------

def _load_db_config() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db-config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("db-config.json not found — using default table names")
        return {}
    except Exception as e:
        log.warning(f"db-config.json load failed: {e} — using default table names")
        return {}

_db_cfg = _load_db_config()
_tables = _db_cfg.get("tables", {})
_ST  = _tables.get("memory_shortterm", "memory_shortterm")
_LT  = _tables.get("memory_longterm",  "memory_longterm")
_SUM = _tables.get("chat_summaries",   "chat_summaries")

# ---------------------------------------------------------------------------
# Config helpers (read from plugins-enabled.json at call time)
# ---------------------------------------------------------------------------

def _mem_plugin_cfg() -> dict:
    """Return the plugin_config.memory dict from plugins-enabled.json, or {}."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")
    try:
        with open(path) as f:
            return json.load(f).get("plugin_config", {}).get("memory", {})
    except Exception:
        return {}


def _age_cfg() -> dict:
    """
    Return aging config with defaults. Keys:
      auto_memory_age         bool  — master switch for background aging
      memory_age_entrycount   int   — max shortterm rows before count-pressure aging
      memory_age_count_timer  int   — minutes between count-pressure checks (-1 = disabled)
      memory_age_trigger_minutes int — age threshold in minutes for staleness aging
      memory_age_minutes_timer   int — minutes between staleness checks (-1 = disabled)
    """
    cfg = _mem_plugin_cfg()
    return {
        "auto_memory_age":          cfg.get("auto_memory_age",          True),
        "memory_age_entrycount":    int(cfg.get("memory_age_entrycount",    50)),
        "memory_age_count_timer":   int(cfg.get("memory_age_count_timer",   60)),
        "memory_age_trigger_minutes": int(cfg.get("memory_age_trigger_minutes", 2880)),
        "memory_age_minutes_timer": int(cfg.get("memory_age_minutes_timer", 360)),
    }


def _fuzzy_dedup_threshold() -> float | None:
    """
    Return fuzzy dedup threshold (0.0–1.0), or None if feature is disabled.
    Reads plugins-enabled.json each call so live config changes take effect
    without restart.
    Default threshold: 0.78
    """
    mem_cfg = _mem_plugin_cfg()
    if not mem_cfg.get("enabled", True):
        return None
    if not mem_cfg.get("fuzzy_dedup", True):
        return None
    return float(mem_cfg.get("fuzzy_dedup_threshold", 0.78))


def _fuzzy_similar(a: str, b: str, threshold: float) -> bool:
    """
    Return True if strings a and b are similar enough to be considered duplicates.
    Uses SequenceMatcher ratio (word-level tokens for speed on longer strings).
    Both inputs are lowercased and stripped before comparison.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


# ---------------------------------------------------------------------------
# Short-term: save
# ---------------------------------------------------------------------------

async def save_memory(
    topic: str,
    content: str,
    importance: int = 5,
    source: str = "session",
    session_id: str = "",
) -> int:
    """Insert a new short-term memory row. Returns new row id, or 0 if duplicate/error."""
    topic = topic.replace("'", "''")[:255]
    content = content.replace("'", "''")
    session_id = (session_id or "").replace("'", "''")[:255]
    source = source if source in ("session", "user", "directive") else "session"
    importance = max(1, min(10, int(importance)))

    # Dedup pass 1: exact match on topic+content in both tiers
    try:
        dup_check = (
            f"SELECT 1 FROM {_ST} "
            f"WHERE topic = '{topic}' AND content = '{content}' LIMIT 1"
        )
        if "1" in (await execute_sql(dup_check)).strip():
            return 0
        dup_check_lt = (
            f"SELECT 1 FROM {_LT} "
            f"WHERE topic = '{topic}' AND content = '{content}' LIMIT 1"
        )
        if "1" in (await execute_sql(dup_check_lt)).strip():
            return 0
    except Exception as e:
        log.warning(f"save_memory exact-dedup check failed: {e}")

    # Dedup pass 2: fuzzy similarity against existing rows for the same topic
    threshold = _fuzzy_dedup_threshold()
    if threshold is not None:
        try:
            # Load content of existing rows with the same topic (both tiers)
            rows_sql = (
                f"SELECT content FROM {_ST} WHERE topic = '{topic}' "
                f"UNION ALL "
                f"SELECT content FROM {_LT} WHERE topic = '{topic}'"
            )
            raw = await execute_sql(rows_sql)
            # Parse: first line is header "content", rest are values
            existing = [
                line.strip() for line in raw.strip().splitlines()[1:]
                if line.strip() and not set(line.strip()) <= set("-+|")
            ]
            for existing_content in existing:
                if _fuzzy_similar(content, existing_content, threshold):
                    log.debug(
                        f"save_memory fuzzy-dedup: skipped topic={topic!r} "
                        f"(ratio>={threshold:.2f} vs existing row)"
                    )
                    return 0
        except Exception as e:
            log.warning(f"save_memory fuzzy-dedup check failed: {e}")

    sql = (
        f"INSERT INTO {_ST} "
        f"(topic, content, importance, source, session_id) "
        f"VALUES ('{topic}', '{content}', {importance}, '{source}', '{session_id}')"
    )
    try:
        row_id = await execute_insert(sql)
    except Exception as e:
        log.error(f"save_memory failed: {e}")
        return 0

    # Upsert into Qdrant vector index (fire-and-forget, non-blocking)
    if row_id:
        try:
            from plugin_memory_vector_qdrant import get_vector_api
            vec = get_vector_api()
            if vec:
                asyncio.create_task(vec.upsert_memory(
                    row_id=row_id,
                    topic=topic,
                    content=content,
                    importance=importance,
                    tier="short",
                ))
        except Exception as e:
            log.warning(f"save_memory vector upsert skipped: {e}")

    return row_id


# ---------------------------------------------------------------------------
# Short-term: load (returns list of dicts, most important first)
# ---------------------------------------------------------------------------

async def load_short_term(limit: int = 20, min_importance: int = 1) -> list[dict]:
    """Load recent short-term memories, highest importance first."""
    sql = (
        f"SELECT id, topic, content, importance, source, session_id, "
        f"created_at, last_accessed "
        f"FROM {_ST} "
        f"WHERE importance >= {min_importance} "
        f"ORDER BY importance DESC, created_at DESC "
        f"LIMIT {limit}"
    )
    try:
        raw = await execute_sql(sql)
        return _parse_table(raw)
    except Exception as e:
        log.error(f"load_short_term failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Long-term: load (returns list of dicts, most important first)
# ---------------------------------------------------------------------------

async def load_long_term(limit: int = 20, topic: str = "") -> list[dict]:
    """Load long-term memories, optionally filtered by topic or content substring."""
    where = (
        f"WHERE topic LIKE '%{topic}%' OR content LIKE '%{topic}%'"
        if topic else ""
    )
    sql = (
        f"SELECT id, topic, content, importance, created_at "
        f"FROM {_LT} {where} "
        f"ORDER BY importance DESC, created_at DESC "
        f"LIMIT {limit}"
    )
    try:
        raw = await execute_sql(sql)
        return _parse_table(raw)
    except Exception as e:
        log.error(f"load_long_term failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Update: change fields on an existing short-term or long-term row
# ---------------------------------------------------------------------------

async def update_memory(
    row_id: int,
    tier: str = "short",
    importance: int | None = None,
    content: str | None = None,
    topic: str | None = None,
) -> str:
    """
    Update one or more fields on an existing memory row.
    tier: 'short' (default) or 'long'.
    Returns a status string.
    """
    table = _ST if tier == "short" else _LT
    sets = []
    if importance is not None:
        importance = max(1, min(10, int(importance)))
        sets.append(f"importance = {importance}")
    if content is not None:
        sets.append(f"content = '{content.replace(chr(39), chr(39)*2)}'")
    if topic is not None:
        sets.append(f"topic = '{topic.replace(chr(39), chr(39)*2)[:255]}'")
    if not sets:
        return "Nothing to update — provide at least one of: importance, content, topic."
    sql = f"UPDATE {table} SET {', '.join(sets)} WHERE id = {int(row_id)}"
    try:
        result = await execute_sql(sql)
        # Verify the row actually exists (rows affected: 0 can mean no row OR same value)
        check = await execute_sql(f"SELECT id FROM {table} WHERE id = {int(row_id)} LIMIT 1")
        if not check.strip() or str(row_id) not in check:
            return f"No row found with id={row_id} in {tier}-term memory."
        return f"Memory id={row_id} updated ({tier}): {', '.join(sets)}"
    except Exception as e:
        return f"update_memory failed: {e}"


# ---------------------------------------------------------------------------
# Aging: move rows from short-term to long-term
# Two independent modes:
#   age_by_count   — count-pressure: moves overflow rows (ignores age threshold)
#   age_by_minutes — staleness: moves rows not accessed in N minutes
# ---------------------------------------------------------------------------

async def _move_rows_to_longterm(rows: list[dict]) -> int:
    """Move a list of parsed short-term rows to long-term. Returns count moved."""
    moved = 0
    for row in rows:
        rid = row.get("id", "")
        if not rid:
            continue
        topic   = row.get("topic",      "").replace("'", "''")
        content = row.get("content",    "").replace("'", "''")
        imp     = row.get("importance", 5)
        src     = row.get("source",     "session")
        sid     = row.get("session_id", "").replace("'", "''")
        try:
            await execute_sql(
                f"INSERT INTO {_LT} "
                f"(topic, content, importance, source, session_id, shortterm_id) "
                f"VALUES ('{topic}', '{content}', {imp}, '{src}', '{sid}', {rid})"
            )
            await execute_sql(f"DELETE FROM {_ST} WHERE id = {rid}")
            moved += 1
            # Update Qdrant tier payload short→long (fire-and-forget)
            try:
                from plugin_memory_vector_qdrant import get_vector_api
                vec = get_vector_api()
                if vec:
                    asyncio.create_task(vec.update_tier(int(rid), "long"))
            except Exception:
                pass
        except Exception as e:
            log.warning(f"_move_rows_to_longterm: failed for id={rid}: {e}")
    return moved


async def age_by_count(max_rows: int = 200) -> int:
    """
    Count-pressure aging: if short-term row count exceeds memory_age_entrycount,
    move exactly the overflow rows to long-term.
    Rows are selected by importance ASC, last_accessed ASC (least important,
    least recently used first).  Ignores age threshold.
    Returns number of rows moved.
    """
    cfg = _age_cfg()
    entry_limit = cfg["memory_age_entrycount"]
    try:
        count_raw = await execute_sql(f"SELECT COUNT(*) FROM {_ST}")
        lines = [l.strip() for l in count_raw.strip().splitlines() if l.strip()]
        current_count = 0
        for line in lines:
            if line.isdigit():
                current_count = int(line)
                break
            # handle "COUNT(*)\n123" format
            parts = line.split()
            if parts and parts[-1].isdigit():
                current_count = int(parts[-1])
                break

        overflow = current_count - entry_limit
        if overflow <= 0:
            return 0

        # Move exactly overflow rows, capped by max_rows safety ceiling
        n_to_move = min(overflow, max_rows)
        select_sql = (
            f"SELECT * FROM {_ST} "
            f"ORDER BY importance ASC, last_accessed ASC "
            f"LIMIT {n_to_move}"
        )
        raw = await execute_sql(select_sql)
        rows = _parse_table(raw)
        moved = await _move_rows_to_longterm(rows)
        if moved:
            log.info(f"age_by_count: moved {moved} rows (overflow={overflow}, limit={entry_limit})")
        return moved
    except Exception as e:
        log.error(f"age_by_count failed: {e}")
        return 0


async def age_by_minutes(trigger_minutes: int, max_rows: int = 200) -> int:
    """
    Staleness aging: move rows whose last_accessed is older than trigger_minutes.
    Ordered by importance ASC, last_accessed ASC.
    max_rows is a safety ceiling per pass.
    Returns number of rows moved.
    """
    if trigger_minutes <= 0:
        return 0
    try:
        select_sql = (
            f"SELECT * FROM {_ST} "
            f"WHERE last_accessed < NOW() - INTERVAL {int(trigger_minutes)} MINUTE "
            f"ORDER BY importance ASC, last_accessed ASC "
            f"LIMIT {int(max_rows)}"
        )
        raw = await execute_sql(select_sql)
        rows = _parse_table(raw)
        if not rows:
            return 0
        moved = await _move_rows_to_longterm(rows)
        if moved:
            log.info(f"age_by_minutes: moved {moved} rows (threshold={trigger_minutes}min)")
        return moved
    except Exception as e:
        log.error(f"age_by_minutes failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# Context block: formatted string ready to inject into system prompt
# ---------------------------------------------------------------------------

async def load_topic_list() -> list[str]:
    """Return distinct topic names from both short-term and long-term memory, sorted."""
    try:
        raw_st = await execute_sql(f"SELECT DISTINCT topic FROM {_ST} ORDER BY topic")
        raw_lt = await execute_sql(f"SELECT DISTINCT topic FROM {_LT} ORDER BY topic")
        topics: set[str] = set()
        for raw in (raw_st, raw_lt):
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("topic") or line.startswith("(") or line.startswith("--"):
                    continue
                if set(line) <= set("-+|"):
                    continue
                # Only accept lines that look like valid topic names (no spaces, no parens)
                if " " in line or "(" in line or ")" in line:
                    continue
                topics.add(line)
        return sorted(topics)
    except Exception as e:
        log.debug(f"load_topic_list failed: {e}")
        return []


async def _update_last_accessed(row_ids: list) -> None:
    """Fire-and-forget: update last_accessed for a list of shortterm row IDs."""
    if not row_ids:
        return
    ids_str = ", ".join(str(rid) for rid in row_ids if str(rid).isdigit())
    if not ids_str:
        return
    try:
        await execute_sql(
            f"UPDATE {_ST} SET last_accessed = NOW() WHERE id IN ({ids_str})"
        )
    except Exception as e:
        log.debug(f"_update_last_accessed failed: {e}")


async def load_context_block(min_importance: int = 3, query: str = "") -> str:
    """
    Return a formatted string of short-term memories for prompt injection.

    When a query string is provided and the vector plugin is available:
      - Semantic search returns the top-K most relevant rows
      - Rows with importance >= min_importance_always are always included
      - Only semantically retrieved rows have last_accessed updated

    When no query or vector plugin unavailable:
      - Falls back to loading all rows meeting min_importance (original behaviour)
      - last_accessed updated for all rows

    Includes the full list of known topics so the model can reuse existing
    categories rather than inventing new ones each turn.
    Returns empty string if no memories and no topics.
    """
    from plugin_memory_vector_qdrant import get_vector_api
    vec = get_vector_api()

    topics_task = asyncio.create_task(load_topic_list())

    if vec and query:
        # --- Semantic retrieval path ---
        always_importance = vec.cfg().get("min_importance_always", 8)

        # Run semantic search and high-importance fetch in parallel
        semantic_task = asyncio.create_task(vec.search_memories(query, tier="short"))
        always_task   = asyncio.create_task(
            load_short_term(limit=10000, min_importance=always_importance)
        )
        semantic_hits, always_rows, topics = await asyncio.gather(
            semantic_task, always_task, topics_task
        )

        # Merge: always_rows first (authoritative), then semantic hits not already included
        seen_ids = {str(r.get("id", "")) for r in always_rows}
        merged = list(always_rows)
        for hit in semantic_hits:
            if str(hit.get("id", "")) not in seen_ids:
                merged.append(hit)
                seen_ids.add(str(hit.get("id", "")))

        rows = merged
        # Only update last_accessed for semantically retrieved rows (not always-rows)
        semantic_ids = [h.get("id") for h in semantic_hits if h.get("id")]
        if semantic_ids:
            asyncio.create_task(_update_last_accessed(semantic_ids))
        log.debug(
            f"load_context_block: semantic={len(semantic_hits)} "
            f"always={len(always_rows)} merged={len(merged)}"
        )
    else:
        # --- Fallback: load all rows meeting min_importance ---
        rows, topics = await asyncio.gather(
            load_short_term(limit=10000, min_importance=min_importance),
            topics_task,
        )
        row_ids = [row.get("id", "") for row in rows if row.get("id", "")]
        if row_ids:
            asyncio.create_task(_update_last_accessed(row_ids))

    if not rows and not topics:
        return ""

    lines = ["## Active Memory (short-term recall)\n"]

    if rows:
        # Group by topic
        by_topic: dict[str, list[dict]] = {}
        for row in rows:
            t = row.get("topic", "general")
            by_topic.setdefault(t, []).append(row)

        for topic, items in by_topic.items():
            lines.append(f"**{topic}**")
            for item in items:
                imp = item.get("importance", 5)
                lines.append(f"  [imp={imp}] {item.get('content', '')}")
            lines.append("")

    if topics:
        lines.append(f"**Known topics** (reuse these for new saves; add new ones only when needed):")
        lines.append(f"  {', '.join(topics)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summarize and save: call summarizer LLM on a history, store as memory
# ---------------------------------------------------------------------------

async def summarize_and_save(
    session_id: str,
    history: list[dict],
    model_key: str,
) -> str:
    """
    Call a summarizer LLM on conversation history.
    Extracts topic-tagged memories and saves them to short-term.
    Also saves a chat summary row.
    Returns a status string.
    """
    if not history:
        return "No history to summarize."

    # Build condensed history text (skip tool calls to save tokens)
    lines = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or role == "tool":
            continue
        if isinstance(content, list):
            # Extract text parts from structured content
            parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(parts)
        if content:
            lines.append(f"{role.upper()}: {content[:500]}")

    history_text = "\n".join(lines[-60:])  # last ~60 turns max
    if not history_text.strip():
        return "History had no text content to summarize."

    # Load known topics so the summarizer reuses existing labels
    known_topics = await load_topic_list()
    if known_topics:
        topics_hint = (
            "EXISTING TOPICS (reuse these exactly; only add a new topic if nothing fits):\n"
            f"  {', '.join(known_topics)}\n\n"
        )
    else:
        topics_hint = "Topic examples: user-preferences, project-status, technical-decisions, security, tasks.\n\n"

    prompt = (
        "You are a memory distillation engine. Given this conversation, extract the most important facts, "
        "decisions, preferences, and context. Output ONLY valid JSON — a list of objects with keys: "
        "topic (short kebab-case string), content (one concise sentence), importance (1-10 int). "
        f"{topics_hint}"
        "Output 3-8 items maximum. No markdown, no explanation, just the JSON array.\n\n"
        f"CONVERSATION:\n{history_text}"
    )

    try:
        # Use llm_call infrastructure directly
        from agents import _call_llm_text
        result_text = await _call_llm_text(model_key, prompt)
    except Exception as e:
        log.error(f"summarize_and_save LLM call failed: {e}")
        result_text = None

    memories_saved = 0
    memories_skipped = 0
    if result_text:
        # Strip any markdown fences
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])

        try:
            items = json.loads(cleaned)
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    topic = str(item.get("topic", "general"))[:255]
                    content = str(item.get("content", ""))[:2000]
                    importance = int(item.get("importance", 5))
                    if topic and content:
                        new_id = await save_memory(
                            topic=topic,
                            content=content,
                            importance=importance,
                            source="session",
                            session_id=session_id,
                        )
                        if new_id:
                            memories_saved += 1
                        else:
                            memories_skipped += 1
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"summarize_and_save JSON parse failed: {e}. Raw: {result_text[:200]}")

    # Save chat summary row regardless
    summary_text = result_text or "(summarization failed)"
    summary_text = summary_text.replace("'", "''")
    msg_count = len(history)
    model_key_safe = model_key.replace("'", "''")
    sid_safe = session_id.replace("'", "''")
    await execute_sql(
        f"INSERT INTO {_SUM} "
        f"(session_id, summary, message_count, model_used) "
        f"VALUES ('{sid_safe}', '{summary_text[:4000]}', {msg_count}, '{model_key_safe}')"
    )

    skip_note = f", {memories_skipped} duplicate(s) skipped" if memories_skipped else ""
    return f"Summarized {msg_count} messages → {memories_saved} memories saved{skip_note}."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_table(raw: str) -> list[dict]:
    """Parse pipe-separated execute_sql output into list of dicts."""
    lines = raw.strip().splitlines()
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split("|")]
    rows = []
    for line in lines[1:]:
        if not line.strip() or line.startswith("---") or set(line.strip()) <= set("-+"):
            continue
        vals = line.split("|")
        row = {}
        for i, h in enumerate(headers):
            v = vals[i].strip() if i < len(vals) else ""
            row[h] = v
        rows.append(row)
    return rows
