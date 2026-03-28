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
  save_conversation_turn(user_text, assistant_text, session_id, importance) -> (user_id, asst_id, topic)
"""

import asyncio
import difflib
import json
import logging
import os
import re
from datetime import datetime, timezone

from database import execute_sql, execute_insert, get_tables_for_model

log = logging.getLogger("memory")

# ---------------------------------------------------------------------------
# Retrieval stats — tracks single-pass vs two-pass usage for !memstats
# ---------------------------------------------------------------------------
_retrieval_stats = {
    "total": 0,
    "single_pass_sufficient": 0,
    "two_pass_needed": 0,
    "pass1_avg_hits": 0.0,     # running average of pass-1 hit count
    "pass2_avg_extra": 0.0,    # running average of extra hits from pass-2
    "fallback_no_vec": 0,
}


def get_retrieval_stats() -> dict:
    """Return a copy of the retrieval stats counters."""
    return dict(_retrieval_stats)

# ---------------------------------------------------------------------------
# Table name helpers — resolved at call time from the active model's database
# ---------------------------------------------------------------------------

def _ST() -> str:
    return get_tables_for_model().get("memory_shortterm", "memory_shortterm")

def _LT() -> str:
    return get_tables_for_model().get("memory_longterm", "memory_longterm")

def _SUM() -> str:
    return get_tables_for_model().get("chat_summaries", "chat_summaries")

def _COLLECTION() -> str:
    return get_tables_for_model().get("collection", "samaritan_memory")

def _GOALS() -> str:
    return get_tables_for_model().get("goals", "samaritan_goals")

def _PLANS() -> str:
    return get_tables_for_model().get("plans", "samaritan_plans")

def _BELIEFS() -> str:
    return get_tables_for_model().get("beliefs", "samaritan_beliefs")

def _CONDITIONED() -> str:
    return get_tables_for_model().get("conditioned", "samaritan_conditioned")

def _EPISODIC() -> str:
    return get_tables_for_model().get("episodic", "samaritan_episodic")

def _SEMANTIC() -> str:
    return get_tables_for_model().get("semantic", "samaritan_semantic")

def _PROCEDURAL() -> str:
    return get_tables_for_model().get("procedural", "samaritan_procedural")

def _AUTOBIOGRAPHICAL() -> str:
    return get_tables_for_model().get("autobiographical", "samaritan_autobiographical")

def _PROSPECTIVE() -> str:
    return get_tables_for_model().get("prospective", "samaritan_prospective")

def _PROCEDURES() -> str:
    """Structured procedural table (same physical table as _PROCEDURAL, distinct intent)."""
    return get_tables_for_model().get("procedural", "samaritan_procedural")

def _PROC_COLLECTION() -> str:
    return get_tables_for_model().get("proc_collection", "samaritan_procedures")

def _DRIVES() -> str:
    return get_tables_for_model().get("drives", "samaritan_drives")

def _COGNITION() -> str:
    return get_tables_for_model().get("cognition", "samaritan_cognition")

def _TEMPORAL() -> str:
    return get_tables_for_model().get("temporal", "samaritan_temporal")

# ---------------------------------------------------------------------------
# Typed table metrics — write/read counters, reset on restart
# ---------------------------------------------------------------------------

_typed_metrics: dict[str, dict[str, int]] = {
    "goals":           {"writes": 0, "reads": 0},
    "plans":           {"writes": 0, "reads": 0},
    "beliefs":         {"writes": 0, "reads": 0},
    "conditioned":     {"writes": 0, "reads": 0},
    "episodic":        {"writes": 0, "reads": 0},
    "semantic":        {"writes": 0, "reads": 0},
    "procedural":      {"writes": 0, "reads": 0},
    "autobiographical":{"writes": 0, "reads": 0},
    "prospective":     {"writes": 0, "reads": 0},
    "procedures":      {"writes": 0, "reads": 0},
    "drives":          {"writes": 0, "reads": 0},
}

def _typed_metric_write(table: str) -> None:
    key = table.split("_")[-1]  # strip prefix → "goals"/"plans"/"beliefs"
    if key in _typed_metrics:
        _typed_metrics[key]["writes"] += 1

def _typed_metric_read(table: str, count: int = 1) -> None:
    key = table.split("_")[-1]
    if key in _typed_metrics:
        _typed_metrics[key]["reads"] += count

def get_typed_metrics() -> dict:
    return {k: dict(v) for k, v in _typed_metrics.items()}

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


def _safe_int(value, default: int) -> int:
    """Coerce value to int, falling back to default if conversion fails.
    Guards against corrupted config values (e.g. LLM text written into an int field)."""
    try:
        return int(value)
    except (ValueError, TypeError):
        log.warning(f"_age_cfg: expected int for config value, got {value!r}; using default {default}")
        return default


def _age_cfg() -> dict:
    """
    Return aging config with defaults. Keys:
      auto_memory_age             bool  — master switch for background aging
      short_hwm                   int   — ST row count that triggers count-pressure aging
      short_lwm                   int   — target ST count after aging loop
      recent_turns_protect        int   — last N ST rows whose topics are protected from chunking
      staleness_override_minutes  int   — even protected topics age if all rows older than this
      chunk_importance_threshold  int   — rows >= this importance are also copied verbatim to LT
      memory_age_count_timer      int   — minutes between count-pressure checks (-1 = disabled)
      memory_age_minutes_timer    int   — minutes between staleness checks (-1 = disabled)
      memory_age_trigger_minutes  int   — staleness threshold: rows older than N minutes are candidates
      memory_age_entrycount       int   — legacy alias for short_hwm (kept for backwards compat)
    """
    cfg = _mem_plugin_cfg()
    hwm = _safe_int(cfg.get("short_hwm", cfg.get("memory_age_entrycount", 100)), 100)
    return {
        "auto_memory_age":            cfg.get("auto_memory_age",            True),
        "short_hwm":                  hwm,
        "short_lwm":                  _safe_int(cfg.get("short_lwm",                  50),   50),
        "recent_turns_protect":       _safe_int(cfg.get("recent_turns_protect",       10),   10),
        "staleness_override_minutes": _safe_int(cfg.get("staleness_override_minutes", 2880), 2880),
        "chunk_importance_threshold": _safe_int(cfg.get("chunk_importance_threshold", 8),    8),
        "memory_age_count_timer":     _safe_int(cfg.get("memory_age_count_timer",     60),   60),
        "memory_age_minutes_timer":   _safe_int(cfg.get("memory_age_minutes_timer",   360),  360),
        "memory_age_trigger_minutes": _safe_int(cfg.get("memory_age_trigger_minutes", 2880), 2880),
        "memory_age_entrycount":      hwm,  # legacy alias
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

def _fuzzy_match_topics(query: str, topics: list[str], threshold: float = 0.62) -> list[str]:
    """
    Return topic slugs from `topics` that fuzzy-match any word or hyphen-normalised
    ngram in `query`.  Compares each topic against sliding windows of 1-4 words so
    that "plumbing issues" matches the slug "plumbing-issue".
    """
    if not query or not topics:
        return []
    q_norm = query.lower().replace("-", " ")
    words = q_norm.split()
    matched = []
    for slug in topics:
        slug_norm = slug.lower().replace("-", " ")
        slug_word_count = len(slug_norm.split())
        for window in range(max(1, slug_word_count - 1), slug_word_count + 2):
            for i in range(len(words) - window + 1):
                phrase = " ".join(words[i : i + window])
                if difflib.SequenceMatcher(None, phrase, slug_norm).ratio() >= threshold:
                    matched.append(slug)
                    break
            else:
                continue
            break
    return matched


# ---------------------------------------------------------------------------
# Short-term: save
# ---------------------------------------------------------------------------

# Public API addition summary:
#   save_lt_memory(topic, content, importance, source, session_id) -> int
#     Insert directly into long-term memory (bypasses ST). Used by aging.
#   age_by_count() -> int
#     Count-pressure aging: topic-chunk summarize until ST < short_lwm.
#   age_by_minutes(trigger_minutes) -> int
#     Staleness aging: topic-chunk summarize stale topics until ST < short_lwm.
#   trim_st_to_lwm() -> int
#     Escape valve: delete oldest/least-important ST rows until ST < short_lwm.

async def save_memory(
    topic: str,
    content: str,
    importance: int = 5,
    source: str = "session",
    session_id: str = "",
    type: str = "context",
) -> int:
    """Insert a new short-term memory row. Returns new row id, or 0 if duplicate/error."""
    if not _mem_plugin_cfg().get("enabled", True):
        return 0
    topic = topic.replace("\\", "\\\\").replace("'", "''")[:255]
    content = content.replace("\\", "\\\\").replace("'", "''")
    session_id = (session_id or "").replace("\\", "\\\\").replace("'", "''")[:255]
    source = source if source in ("session", "user", "directive", "assistant") else "session"
    importance = max(1, min(10, int(importance)))
    mem_type = type if type in _MEMORY_TYPES else "context"

    # Dedup: skip for user prompts — user input is ground truth and always saved verbatim.
    # Only assistant/session/directive content is deduped.
    if source != "user":
        # Dedup pass 1: exact match on topic+content in both tiers
        try:
            dup_check = (
                f"SELECT 1 FROM {_ST()} "
                f"WHERE topic = '{topic}' AND content = '{content}' LIMIT 1"
            )
            if "1" in (await execute_sql(dup_check)).strip():
                log.info(f"save_memory exact-dedup ST: skipped source={source} topic={topic!r} content={content[:60]!r}")
                return 0
            dup_check_lt = (
                f"SELECT 1 FROM {_LT()} "
                f"WHERE topic = '{topic}' AND content = '{content}' LIMIT 1"
            )
            if "1" in (await execute_sql(dup_check_lt)).strip():
                log.info(f"save_memory exact-dedup LT: skipped source={source} topic={topic!r} content={content[:60]!r}")
                return 0
        except Exception as e:
            log.warning(f"save_memory exact-dedup check failed: {e}")

        # Dedup pass 2: fuzzy similarity against existing rows for the same topic
        threshold = _fuzzy_dedup_threshold()
        if threshold is not None:
            try:
                # Load content of existing rows with the same topic (both tiers)
                rows_sql = (
                    f"SELECT content FROM {_ST()} WHERE topic = '{topic}' "
                    f"UNION ALL "
                    f"SELECT content FROM {_LT()} WHERE topic = '{topic}'"
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
        f"INSERT INTO {_ST()} "
        f"(topic, content, importance, source, session_id, type) "
        f"VALUES ('{topic}', '{content}', {importance}, '{source}', '{session_id}', '{mem_type}')"
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
                    collection=_COLLECTION(),
                ))
        except Exception as e:
            log.warning(f"save_memory vector upsert skipped: {e}")

        # Notify on belief save
        if mem_type == "belief":
            try:
                import notifier as _notifier
                asyncio.ensure_future(_notifier.fire_event(
                    "belief_saved",
                    f"topic={topic!r}",
                    content[:200],
                ))
            except Exception:
                pass

    return row_id


# ---------------------------------------------------------------------------
# Conversation logging — verbatim save of user prompt + assistant response
# ---------------------------------------------------------------------------

_TOPIC_TAG_RE = re.compile(r'^<<([a-z0-9][a-z0-9\-]*)>>\s*', re.IGNORECASE)
_TYPE_TAG_RE  = re.compile(r'<<type:([a-z]+)>>\s*', re.IGNORECASE)

# Valid memory types — must match the ENUM in the DB schema
_MEMORY_TYPES = frozenset({
    "context", "goal", "plan", "belief",
    "episodic", "semantic", "procedural",
    "autobiographical", "prospective", "conditioned",
    "self_model",
})


def _extract_topic_tag(text: str) -> tuple[str | None, str]:
    """Extract <<topic-slug>> prefix from assistant text.

    Returns (topic_slug, text_with_tag_stripped).
    If no tag found, returns (None, original_text).
    """
    m = _TOPIC_TAG_RE.match(text)
    if m:
        return m.group(1).lower(), text[m.end():]
    return None, text


def _extract_type_tag(text: str) -> tuple[str, str]:
    """Extract <<type:X>> tag from anywhere in assistant text.

    Returns (type_value, text_with_tag_stripped).
    Falls back to 'context' if tag absent or value not in _MEMORY_TYPES.
    """
    m = _TYPE_TAG_RE.search(text)
    if m:
        val = m.group(1).lower()
        mem_type = val if val in _MEMORY_TYPES else "context"
        cleaned = text[:m.start()] + text[m.end():]
        return mem_type, cleaned.strip()
    return "context", text


def _make_conv_topic(user_text: str) -> str:
    """Fallback topic slug derived from user text when no <<topic>> tag present.

    Format: conv-YYYY-MM-DD-<first-few-words>
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = user_text[:60].lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    slug = re.sub(r'-{2,}', '-', slug)
    return f"conv-{date_str}-{slug}"


async def _normalize_topic(topic_tag: str, threshold: float = 0.65) -> str:
    """Fuzzy-match a model-generated topic tag against existing topics.

    Uses SequenceMatcher with word permutations to handle reordering
    (e.g. 'kitchen-plumbing' → 'plumbing-issue').  Threshold 0.65 avoids
    false positives on shared-prefix topics (e.g. 'memory-roadmap' ≠ 'memory-toggle').

    Returns the best existing match, or the original tag if no close match.
    """
    try:
        existing = await load_topic_list()
    except Exception:
        return topic_tag
    if not existing or topic_tag in existing:
        return topic_tag  # exact match or no topics — use as-is

    from itertools import permutations as _perms
    tag_norm = topic_tag.replace("-", " ").lower()
    tag_words = tag_norm.split()
    best_slug, best_score = topic_tag, 0.0

    for slug in existing:
        slug_norm = slug.replace("-", " ").lower()
        score = difflib.SequenceMatcher(None, tag_norm, slug_norm).ratio()
        # Try word permutations (cheap for 2-3 word slugs) to handle reordering
        if len(tag_words) <= 4:
            for perm in _perms(tag_words):
                s = difflib.SequenceMatcher(None, " ".join(perm), slug_norm).ratio()
                if s > score:
                    score = s
        if score > best_score:
            best_slug, best_score = slug, score

    if best_score >= threshold:
        log.debug(f"topic_normalize: '{topic_tag}' → '{best_slug}' (score={best_score:.2f})")
        return best_slug
    return topic_tag


async def save_conversation_turn(
    user_text: str,
    assistant_text: str,
    session_id: str = "",
    importance: int = 4,
    memory_types_enabled: bool = False,
) -> tuple[int, int, str | None]:
    """Save a user prompt and assistant response as two verbatim memory rows.

    Extracts <<topic-slug>> from the assistant text if present — that becomes
    the topic for both rows, and the tag is stripped from the stored content.
    Falls back to a date-slug derived from the user text.

    The extracted tag is fuzzy-matched against existing topics to prevent
    near-duplicate topic proliferation (e.g. 'kitchen-plumbing' → 'plumbing-issue').

    When memory_types_enabled is True, also extracts <<type:X>> from the assistant
    text and stores it as the type for both rows. Falls back to 'context'.

    Returns (user_row_id, assistant_row_id, topic_used).
    Row IDs of 0 mean duplicate/skipped.
    """
    topic_tag, asst_clean = _extract_topic_tag(assistant_text)
    if topic_tag:
        topic_tag = await _normalize_topic(topic_tag)
    topic = topic_tag if topic_tag else _make_conv_topic(user_text)

    if memory_types_enabled:
        mem_type, asst_clean = _extract_type_tag(asst_clean)
    else:
        mem_type = "context"

    user_id = await save_memory(
        topic=topic,
        content=user_text,
        importance=importance,
        source="user",
        session_id=session_id,
        type=mem_type,
    )
    asst_id = await save_memory(
        topic=topic,
        content=asst_clean,
        importance=importance,
        source="assistant",
        session_id=session_id,
        type=mem_type,
    )
    return user_id, asst_id, topic


# ---------------------------------------------------------------------------
# Long-term: direct save (bypasses ST; used by aging summarization)
# ---------------------------------------------------------------------------

async def save_lt_memory(
    topic: str,
    content: str,
    importance: int = 5,
    source: str = "session",
    session_id: str = "",
    shortterm_id: int | None = None,
) -> int:
    """Insert directly into long-term memory. Returns new LT row id, or 0 on error/duplicate."""
    if not _mem_plugin_cfg().get("enabled", True):
        return 0
    topic      = topic.replace("\\", "\\\\").replace("'", "''")[:255]
    content    = content.replace("\\", "\\\\").replace("'", "''")
    session_id = (session_id or "").replace("\\", "\\\\").replace("'", "''")[:255]
    source     = source if source in ("session", "user", "directive", "assistant") else "session"
    importance = max(1, min(10, int(importance)))

    # Exact dedup in LT
    try:
        dup = await execute_sql(
            f"SELECT 1 FROM {_LT()} WHERE topic = '{topic}' AND content = '{content}' LIMIT 1"
        )
        if "1" in dup.strip():
            return 0
    except Exception as e:
        log.warning(f"save_lt_memory dedup check failed: {e}")

    st_id_clause = f", {int(shortterm_id)}" if shortterm_id else ", NULL"
    sql = (
        f"INSERT INTO {_LT()} "
        f"(topic, content, importance, source, session_id, shortterm_id) "
        f"VALUES ('{topic}', '{content}', {importance}, '{source}', '{session_id}'{st_id_clause})"
    )
    try:
        row_id = await execute_insert(sql)
    except Exception as e:
        log.error(f"save_lt_memory failed: {e}")
        return 0

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
                    tier="long",
                    collection=_COLLECTION(),
                ))
        except Exception as e:
            log.warning(f"save_lt_memory vector upsert skipped: {e}")

    return row_id


# ---------------------------------------------------------------------------
# Cognition: save / load (system-internal cognitive loop outputs)
# ---------------------------------------------------------------------------

_COGNITION_ORIGINS = frozenset({
    "reflection", "goal_health", "self_model",
    "prospective", "tool_log", "tool_failure", "summary",
})


async def save_cognition(
    origin: str,
    topic: str,
    content: str,
    importance: int = 5,
    source: str = "session",
    session_id: str = "",
) -> int:
    """Insert a cognitive loop output row. Returns new row id, or 0 on error/duplicate."""
    if origin not in _COGNITION_ORIGINS:
        log.warning(f"save_cognition: unknown origin {origin!r}, falling back to 'reflection'")
        origin = "reflection"
    topic      = topic.replace("\\", "\\\\").replace("'", "''")[:255]
    content    = content.replace("\\", "\\\\").replace("'", "''")
    session_id = (session_id or "").replace("\\", "\\\\").replace("'", "''")[:255]
    source     = source if source in ("session", "user", "directive", "assistant") else "session"
    importance = max(1, min(10, int(importance)))

    # Exact dedup
    try:
        dup = await execute_sql(
            f"SELECT 1 FROM {_COGNITION()} "
            f"WHERE origin = '{origin}' AND topic = '{topic}' AND content = '{content}' LIMIT 1"
        )
        if "1" in dup.strip():
            return 0
    except Exception as e:
        log.warning(f"save_cognition dedup check failed: {e}")

    # Fuzzy dedup: compare against recent cognition rows with same origin + similar topic prefix
    threshold = _fuzzy_dedup_threshold()
    if threshold is not None:
        try:
            # Match on origin; use topic prefix (e.g. "self-failure") for broader catch
            topic_prefix = "-".join(topic.split("-")[:2]) if "-" in topic else topic
            rows_sql = (
                f"SELECT content FROM {_COGNITION()} "
                f"WHERE origin = '{origin}' AND topic LIKE '{topic_prefix}%' "
                f"ORDER BY created_at DESC LIMIT 20"
            )
            raw = await execute_sql(rows_sql)
            existing = [
                line.strip() for line in raw.strip().splitlines()[1:]
                if line.strip() and not set(line.strip()) <= set("-+|")
            ]
            for existing_content in existing:
                if _fuzzy_similar(content, existing_content, threshold):
                    log.debug(
                        f"save_cognition fuzzy-dedup: skipped origin={origin} topic={topic!r} "
                        f"(ratio>={threshold:.2f} vs existing row)"
                    )
                    return 0
        except Exception as e:
            log.warning(f"save_cognition fuzzy-dedup check failed: {e}")

    sql = (
        f"INSERT INTO {_COGNITION()} "
        f"(origin, topic, content, importance, source, session_id) "
        f"VALUES ('{origin}', '{topic}', '{content}', {importance}, '{source}', '{session_id}')"
    )
    try:
        row_id = await execute_insert(sql)
        log.info(f"save_cognition: origin={origin} topic={topic!r} id={row_id}")
        return row_id
    except Exception as e:
        log.error(f"save_cognition failed: {e}")
        return 0


async def load_cognition(
    origin: str | None = None,
    topic_like: str | None = None,
    limit: int = 100,
    min_importance: int = 1,
) -> list[dict]:
    """Query cognition rows by origin and/or topic pattern."""
    from database import fetch_dicts
    clauses = [f"importance >= {min_importance}"]
    if origin:
        clauses.append(f"origin = '{origin}'")
    if topic_like:
        clauses.append(f"topic LIKE '{topic_like}'")
    where = " AND ".join(clauses)
    sql = (
        f"SELECT id, origin, topic, content, importance, source, session_id, "
        f"created_at, last_accessed "
        f"FROM {_COGNITION()} "
        f"WHERE {where} "
        f"ORDER BY created_at DESC LIMIT {limit}"
    )
    try:
        return await fetch_dicts(sql) or []
    except Exception as e:
        log.warning(f"load_cognition failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Short-term: load (returns list of dicts, most important first)
# ---------------------------------------------------------------------------

async def load_short_term(limit: int = 20, min_importance: int = 1) -> list[dict]:
    """Load recent short-term memories, highest importance first."""
    sql = (
        f"SELECT id, topic, content, importance, source, session_id, "
        f"created_at, last_accessed "
        f"FROM {_ST()} "
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
        f"FROM {_LT()} {where} "
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
    table = _ST() if tier == "short" else _LT()
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
# Aging: topic-chunk summarize ST rows into LT
#
# Two independent triggers:
#   age_by_count   — fires when ST count > short_hwm; loops until < short_lwm
#   age_by_minutes — fires on schedule; chunks stale topics until < short_lwm
#
# Both share the same core: _age_topic_chunks()
#
# Algorithm:
#   1. Determine "protected" topics: topics that appear in the last
#      recent_turns_protect ST rows (by id DESC).  Protected topics are
#      excluded UNLESS all their rows are older than staleness_override_minutes.
#   2. Build candidate list: distinct topics ordered by their oldest row first.
#      For age_by_minutes, further filter: only topics where at least one row
#      is older than trigger_minutes.
#   3. For each candidate topic (oldest first):
#      a. Load all ST rows for that topic.
#      b. Summarize them via summarize_and_save_lt() → summary written to LT.
#      c. Promote any rows with importance >= chunk_importance_threshold
#         verbatim to LT as well.
#      d. DELETE all those rows from ST + Qdrant.
#      e. Re-check ST count; stop when count < short_lwm.
# ---------------------------------------------------------------------------

async def _st_count() -> int:
    """Return current short-term row count."""
    try:
        raw = await execute_sql(f"SELECT COUNT(*) FROM {_ST()}")
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
            parts = line.split()
            if parts and parts[-1].isdigit():
                return int(parts[-1])
    except Exception as e:
        log.warning(f"_st_count failed: {e}")
    return 0


async def _summarize_chunk_to_lt(
    rows: list[dict],
    session_id: str,
    model_key: str,
    imp_threshold: int,
) -> dict:
    """
    Summarize a list of ST rows and write results directly to LT in one LLM call.
    The LLM outputs both summary rows AND which verbatim ST rows deserve promotion
    (with fresh importance scores). No post-hoc importance check — the summarizer
    decides everything.
    Returns {"summarized": N, "promoted": N, "deleted": N}.
    """
    if not rows:
        return {"summarized": 0, "promoted": 0, "deleted": 0}

    from agents import _call_llm_text
    known_topics = await load_topic_list()
    topics_hint = (
        f"EXISTING TOPICS (reuse these; only add new if nothing fits):\n  {', '.join(known_topics)}\n\n"
        if known_topics else
        "Topic examples: user-preferences, project-status, technical-decisions.\n\n"
    )

    lines_text = []
    for r in rows:
        src  = r.get("source", "session")
        role = "USER" if src == "user" else "ASSISTANT"
        lines_text.append(f"[id={r.get('id', '?')}] {role}: {r.get('content', '')[:500]}")
    history_text = "\n".join(lines_text)

    prompt = (
        "You are Samaritan, an AI assistant, archiving older short-term memories into long-term storage.\n\n"
        "Your task — in ONE JSON response:\n"
        "  1. Write 1-5 concise summary rows capturing the most important facts, decisions, and context.\n"
        "  2. Identify any individual memory rows that are so specific or critical they should be\n"
        "     preserved verbatim alongside the summary (e.g. exact code, commands, key decisions).\n"
        "     Assign them fresh importance scores. If nothing warrants verbatim preservation, use [].\n\n"
        "Output ONLY valid JSON with exactly these two keys:\n"
        "{\n"
        '  "summary": [\n'
        '    {"topic": "kebab-case", "content": "one concise sentence", "importance": 1-10,\n'
        '     "source": "user|assistant|session"}\n'
        "  ],\n"
        '  "preserve": [\n'
        '    {"id": <original row id int>, "importance": 1-10}\n'
        "  ]\n"
        "}\n\n"
        "Importance guidance for summary rows:\n"
        "  6 = useful context (preferences, recurring habits)\n"
        "  7-8 = concrete plans, decisions, relationships\n"
        "  9 = high-stakes decisions, key career/life events\n"
        "  10 = critical time-sensitive facts\n"
        "  Do NOT inflate — only facts with lasting value deserve 8+.\n\n"
        f"Importance threshold for 'preserve': only include rows at {imp_threshold}+ lasting value.\n\n"
        f"{topics_hint}"
        "No markdown, no explanation, just the JSON object.\n\n"
        f"MEMORIES TO ARCHIVE:\n{history_text}"
    )

    result_text = None
    try:
        result_text = await _call_llm_text(model_key, prompt)
    except Exception as e:
        log.error(f"_summarize_chunk_to_lt LLM call failed: {e}")

    summarized = 0
    promoted = 0
    preserve_ids: dict[int, int] = {}  # id -> new importance

    if result_text:
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("expected JSON object")

            # Save summary rows to LT
            for item in parsed.get("summary", []):
                if not isinstance(item, dict):
                    continue
                lt_source = str(item.get("source", "session"))
                # Summaries are never verbatim chat — force non-assistant source
                if lt_source == "assistant":
                    lt_source = "session"
                new_id = await save_lt_memory(
                    topic=str(item.get("topic", "general"))[:255],
                    content=str(item.get("content", ""))[:2000],
                    importance=int(item.get("importance", 5)),
                    source=lt_source,
                    session_id=session_id,
                )
                if new_id:
                    summarized += 1

            # Collect verbatim preserve decisions
            for entry in parsed.get("preserve", []):
                if not isinstance(entry, dict):
                    continue
                try:
                    preserve_ids[int(entry["id"])] = int(entry.get("importance", imp_threshold))
                except (KeyError, ValueError, TypeError):
                    pass

        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"_summarize_chunk_to_lt JSON parse failed: {e}. Raw: {result_text[:200]}")

    # Build row index for verbatim promotion (guard against corrupted non-integer ids)
    row_by_id = {}
    for r in rows:
        try:
            row_by_id[int(r["id"])] = r
        except (KeyError, ValueError, TypeError):
            pass

    # Promote preserved rows verbatim to LT with LLM-assigned importance
    for rid, new_imp in preserve_ids.items():
        r = row_by_id.get(rid)
        if not r:
            continue
        new_id = await save_lt_memory(
            topic=r.get("topic", "general"),
            content=r.get("content", ""),
            importance=new_imp,
            source=r.get("source", "session"),
            session_id=r.get("session_id", ""),
            shortterm_id=rid,
        )
        if new_id:
            promoted += 1

    # Delete all rows from ST + remove from Qdrant
    deleted = 0
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
    except Exception:
        vec = None

    for r in rows:
        rid = r.get("id")
        if not rid:
            continue
        try:
            await execute_sql(f"DELETE FROM {_ST()} WHERE id = {int(rid)}")
            deleted += 1
            if vec:
                asyncio.create_task(vec.delete_memory(int(rid), collection=_COLLECTION()))
        except Exception as e:
            log.warning(f"_summarize_chunk_to_lt: delete failed for id={rid}: {e}")

    return {"summarized": summarized, "promoted": promoted, "deleted": deleted}


async def _age_topic_chunks(
    trigger: str,
    trigger_minutes: int = 0,
) -> dict:
    """
    Core aging loop. trigger = "count" or "minutes".
    Returns {"chunks": N, "summarized": N, "promoted": N, "deleted": N}.
    """
    cfg = _age_cfg()
    hwm             = cfg["short_hwm"]
    lwm             = cfg["short_lwm"]
    protect_n       = cfg["recent_turns_protect"]
    stale_override  = cfg["staleness_override_minutes"]
    imp_threshold   = cfg["chunk_importance_threshold"]
    from config import get_model_role
    model_key       = _mem_plugin_cfg().get("summarizer_model") or get_model_role("summarizer")

    totals = {"chunks": 0, "summarized": 0, "promoted": 0, "deleted": 0}

    current = await _st_count()
    if trigger == "count" and current <= hwm:
        return totals
    if trigger == "minutes" and current <= hwm:
        return totals

    # --- Step 1: identify protected topics ---
    # Protected = topics that appear in the last `protect_n` ST rows by id DESC
    # Exception: a topic is unprotected if ALL its rows are older than stale_override
    try:
        recent_raw = await execute_sql(
            f"SELECT DISTINCT topic FROM {_ST()} "
            f"ORDER BY id DESC LIMIT {protect_n}"
        )
        protected_topics: set[str] = set()
        for line in recent_raw.strip().splitlines():
            line = line.strip()
            if not line or line.lower().startswith("topic") or set(line) <= set("-+|"):
                continue
            protected_topics.add(line)
    except Exception as e:
        log.warning(f"_age_topic_chunks: protected topics query failed: {e}")
        protected_topics = set()

    # Check which protected topics are FULLY stale (all rows older than stale_override)
    truly_protected: set[str] = set()
    for topic in protected_topics:
        t_escaped = topic.replace("'", "''")
        try:
            check = await execute_sql(
                f"SELECT COUNT(*) FROM {_ST()} "
                f"WHERE topic = '{t_escaped}' "
                f"AND last_accessed >= NOW() - INTERVAL {stale_override} MINUTE"
            )
            # If any rows are recent enough, the topic stays protected
            count = 0
            for line in check.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    count = int(line)
                    break
                parts = line.split()
                if parts and parts[-1].isdigit():
                    count = int(parts[-1])
                    break
            if count > 0:
                truly_protected.add(topic)
        except Exception as e:
            log.warning(f"_age_topic_chunks: staleness check failed for topic={topic!r}: {e}")
            truly_protected.add(topic)  # err on side of protection

    # --- Step 2: candidate topics ordered by oldest row first ---
    # For "minutes" trigger, only topics where at least one row is older than trigger_minutes
    age_having = (
        f"HAVING MIN(last_accessed) < NOW() - INTERVAL {int(trigger_minutes)} MINUTE"
        if trigger == "minutes" and trigger_minutes > 0
        else ""
    )
    try:
        cand_raw = await execute_sql(
            f"SELECT topic, MIN(last_accessed) AS oldest "
            f"FROM {_ST()} "
            f"GROUP BY topic "
            f"{age_having} "
            f"ORDER BY oldest ASC"
        )
        candidate_topics: list[str] = []
        for line in cand_raw.strip().splitlines():
            if not line.strip() or line.strip().lower().startswith("topic") or set(line.strip()) <= set("-+|"):
                continue
            parts = line.split("|") if "|" in line else line.split()
            topic_name = parts[0].strip() if parts else ""
            if not topic_name or topic_name.lower() == "topic":
                continue
            if topic_name not in truly_protected:
                candidate_topics.append(topic_name)
    except Exception as e:
        log.error(f"_age_topic_chunks: candidate query failed: {e}")
        return totals

    # --- Step 2b: if no candidates but ST > HWM, force-unprotect the topic
    # with the most rows.  This prevents a single dominant topic from
    # blocking aging entirely.
    if not candidate_topics and current > hwm:
        forced: list[tuple[str, int]] = []
        for topic in truly_protected:
            t_escaped = topic.replace("'", "''")
            try:
                cnt_raw = await execute_sql(
                    f"SELECT COUNT(*) FROM {_ST()} WHERE topic = '{t_escaped}'"
                )
                cnt = 0
                for line in cnt_raw.strip().splitlines():
                    line = line.strip()
                    if line.isdigit():
                        cnt = int(line)
                        break
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        cnt = int(parts[-1])
                        break
                forced.append((topic, cnt))
            except Exception:
                pass
        if forced:
            forced.sort(key=lambda x: x[1], reverse=True)
            candidate_topics.append(forced[0][0])
            log.info(
                f"_age_topic_chunks [{trigger}]: force-unprotected topic "
                f"{forced[0][0]!r} ({forced[0][1]} rows) — single dominant topic"
            )

    # --- Step 2c: normalize candidate topic slugs before aging ---
    # Fuzzy-match each candidate against existing topics to prevent fragmented
    # slugs from being promoted verbatim to LT.  This closes the race condition
    # where ST hits HWM before memreview_auto runs.  No LLM call — pure string
    # matching via _normalize_topic().  The scheduled memreview auto pass handles
    # any semantic duplicates that fuzzy matching misses.
    normalized_candidates: list[str] = []
    for topic in candidate_topics:
        normalized = await _normalize_topic(topic)
        if normalized != topic:
            log.info(f"_age_topic_chunks: pre-age normalize '{topic}' → '{normalized}'")
            t_escaped_old = topic.replace("'", "''")
            t_escaped_new = normalized.replace("'", "''")
            try:
                from database import execute_sql as _exec_sql, fetch_dicts as _fd_norm
                # Update MySQL ST rows
                await _exec_sql(
                    f"UPDATE {_ST()} SET topic = '{t_escaped_new}' "
                    f"WHERE topic = '{t_escaped_old}'"
                )
                # Update Qdrant payloads to match
                try:
                    from plugin_memory_vector_qdrant import get_vector_api as _gva
                    _vec = _gva()
                    if _vec:
                        affected = await _fd_norm(
                            f"SELECT id FROM {_ST()} WHERE topic = '{t_escaped_new}'"
                        )
                        ids = [int(r["id"]) for r in affected if r.get("id")]
                        if ids:
                            _vec._qc.set_payload(
                                collection_name=_COLLECTION(),
                                payload={"topic": normalized},
                                points=ids,
                            )
                except Exception as _qe:
                    log.warning(f"_age_topic_chunks: Qdrant payload update failed for '{topic}': {_qe}")
            except Exception as _ne:
                log.warning(f"_age_topic_chunks: pre-age normalize update failed for '{topic}': {_ne}")
                normalized = topic  # fall back to original if update failed
        normalized_candidates.append(normalized)
    # Deduplicate — two candidates may have normalized to the same slug
    seen: set[str] = set()
    candidate_topics = [t for t in normalized_candidates if not (t in seen or seen.add(t))]

    # --- Step 3: process one topic at a time until ST < lwm ---
    # Large topics are sub-chunked to keep summarizer prompt manageable.
    _CHUNK_SIZE = 50

    for topic in candidate_topics:
        current = await _st_count()
        if current < lwm:
            break

        t_escaped = topic.replace("'", "''")

        # Loop: fetch oldest _CHUNK_SIZE rows, summarize, repeat until
        # the topic is exhausted or ST drops below LWM.
        while True:
            current = await _st_count()
            if current < lwm:
                break

            try:
                from database import fetch_dicts as _fd_age
                rows = await _fd_age(
                    f"SELECT * FROM {_ST()} WHERE topic = '{t_escaped}' "
                    f"ORDER BY created_at ASC LIMIT {_CHUNK_SIZE}"
                )
            except Exception as e:
                log.warning(f"_age_topic_chunks: row fetch failed for topic={topic!r}: {e}")
                break

            if not rows:
                break

            result = await _summarize_chunk_to_lt(
                rows=rows,
                session_id="",
                model_key=model_key,
                imp_threshold=imp_threshold,
            )
            totals["chunks"]     += 1
            totals["summarized"] += result["summarized"]
            totals["promoted"]   += result["promoted"]
            totals["deleted"]    += result["deleted"]
            log.info(
                f"_age_topic_chunks [{trigger}]: topic={topic!r} chunk "
                f"summarized={result['summarized']} promoted={result['promoted']} "
                f"deleted={result['deleted']}"
            )

            # If nothing was deleted this pass, bail to avoid infinite loop
            if result["deleted"] == 0:
                log.warning(
                    f"_age_topic_chunks [{trigger}]: topic={topic!r} chunk "
                    f"deleted 0 rows — breaking to avoid loop"
                )
                break

    return totals


async def age_by_count() -> int:
    """
    Count-pressure aging: if ST count > short_hwm, topic-chunk summarize
    oldest unprotected topics into LT until ST count < short_lwm.
    Returns number of ST rows deleted.
    """
    result = await _age_topic_chunks(trigger="count")
    return result["deleted"]


async def age_by_minutes(trigger_minutes: int, max_rows: int = 200) -> int:  # noqa: ARG001
    """
    Staleness aging: topic-chunk summarize topics with stale rows into LT
    until ST count < short_lwm (or no more stale candidates).
    trigger_minutes: rows older than this are candidates.
    max_rows: kept for API compatibility but ignored (loop stops at lwm).
    Returns number of ST rows deleted.
    """
    if trigger_minutes <= 0:
        return 0
    result = await _age_topic_chunks(trigger="minutes", trigger_minutes=trigger_minutes)
    return result["deleted"]


async def trim_st_to_lwm() -> int:
    """
    Escape valve: hard-delete ST rows (oldest + least important first)
    until ST count < short_lwm.  No summarization — raw trim.
    Returns number of rows deleted.
    """
    cfg = _age_cfg()
    lwm = cfg["short_lwm"]
    current = await _st_count()
    if current <= lwm:
        return 0

    n_to_delete = current - lwm
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
    except Exception:
        vec = None

    try:
        rows_raw = await execute_sql(
            f"SELECT id FROM {_ST()} "
            f"ORDER BY importance ASC, last_accessed ASC "
            f"LIMIT {n_to_delete}"
        )
        rows = _parse_table(rows_raw)
    except Exception as e:
        log.error(f"trim_st_to_lwm: row fetch failed: {e}")
        return 0

    deleted = 0
    for r in rows:
        rid = r.get("id")
        if not rid:
            continue
        try:
            await execute_sql(f"DELETE FROM {_ST()} WHERE id = {int(rid)}")
            deleted += 1
            if vec:
                asyncio.create_task(vec.delete_memory(int(rid), collection=_COLLECTION()))
        except Exception as e:
            log.warning(f"trim_st_to_lwm: delete failed id={rid}: {e}")

    log.info(f"trim_st_to_lwm: deleted {deleted} rows (target lwm={lwm})")
    return deleted


# ---------------------------------------------------------------------------
# Temporal cache aging
# ---------------------------------------------------------------------------

def _temporal_table() -> str:
    return get_tables_for_model().get("temporal", "samaritan_temporal")


def _temporal_age_cfg() -> dict:
    """Return temporal cache aging config with defaults.
    Higher watermarks than memory since temporal rows are not auto-injected.
    """
    cfg = _mem_plugin_cfg()
    tcfg = cfg.get("temporal", {}) if isinstance(cfg.get("temporal"), dict) else {}
    return {
        "temporal_hwm":        _safe_int(tcfg.get("hwm", 500), 500),
        "temporal_lwm":        _safe_int(tcfg.get("lwm", 300), 300),
        "temporal_age_timer":  _safe_int(tcfg.get("age_timer_minutes", 360), 360),
    }


async def age_temporal_cache() -> int:
    """Age the temporal cache table using HWM/LWM.
    Deletes oldest, lowest-hit rows until count <= LWM.
    Returns number of rows deleted.
    """
    tcfg = _temporal_age_cfg()
    hwm = tcfg["temporal_hwm"]
    lwm = tcfg["temporal_lwm"]
    tbl = _temporal_table()

    count_raw = await execute_sql(f"SELECT COUNT(*) AS cnt FROM {tbl}")
    lines = count_raw.strip().split("\n")
    current = int(lines[2].strip()) if len(lines) >= 3 and lines[2].strip().isdigit() else 0

    if current <= hwm:
        return 0

    n_to_delete = current - lwm
    # Delete oldest with fewest hits first
    await execute_sql(
        f"DELETE FROM {tbl} "
        f"ORDER BY hit_count ASC, created_at ASC "
        f"LIMIT {n_to_delete}"
    )
    log.info(f"age_temporal_cache: deleted {n_to_delete} rows (hwm={hwm}, lwm={lwm}, was={current})")
    return n_to_delete


# ---------------------------------------------------------------------------
# Context block: formatted string ready to inject into system prompt
# ---------------------------------------------------------------------------

async def load_topic_list() -> list[str]:
    """Return distinct topic names from both short-term and long-term memory, sorted."""
    try:
        raw_st = await execute_sql(f"SELECT DISTINCT topic FROM {_ST()} ORDER BY topic")
        raw_lt = await execute_sql(f"SELECT DISTINCT topic FROM {_LT()} ORDER BY topic")
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
            f"UPDATE {_ST()} SET last_accessed = NOW() WHERE id IN ({ids_str})"
        )
    except Exception as e:
        log.debug(f"_update_last_accessed failed: {e}")


async def _update_lt_last_accessed(row_ids: list) -> None:
    """Fire-and-forget: update last_accessed for a list of longterm row IDs."""
    if not row_ids:
        return
    ids_str = ", ".join(str(rid) for rid in row_ids if str(rid).isdigit())
    if not ids_str:
        return
    try:
        await execute_sql(
            f"UPDATE {_LT()} SET last_accessed = NOW() WHERE id IN ({ids_str})"
        )
    except Exception as e:
        log.debug(f"_update_lt_last_accessed failed: {e}")


async def load_typed_context_block() -> str:
    """
    Return a formatted block of active goals, plans, and beliefs for prompt injection.
    Only populated when memory_types_enabled models write to the typed tables.
    Returns empty string if all tables are empty.
    """
    from database import fetch_dicts
    lines = []

    try:
        goals = await fetch_dicts(
            f"SELECT id, title, description, status, importance "
            f"FROM {_GOALS()} WHERE status = 'active'"
        )
        if goals:
            _typed_metric_read(_GOALS(), len(goals))
            # Drive-weighted prioritization: score = importance × task_completion_drive
            # Falls back to importance-only sort if drives unavailable
            try:
                drive_rows = await fetch_dicts(
                    f"SELECT name, value FROM {_DRIVES()} WHERE status IS NULL OR status != 'inactive'"
                )
                drive_map = {d["name"]: float(d.get("value", 0.5)) for d in (drive_rows or [])}
            except Exception:
                drive_map = {}
            tc = drive_map.get("task-completion", 0.7)
            autonomy = drive_map.get("autonomy", 0.4)
            # Score: blend task-completion and autonomy drives against importance
            for g in goals:
                imp = g.get("importance", 9)
                source = g.get("source", "")
                drive_weight = tc if source != "assistant" else max(tc, autonomy)
                g["_priority"] = imp * drive_weight
            goals.sort(key=lambda g: g["_priority"], reverse=True)
            lines.append("## Active Goals\n")
            for g in goals:
                score = f"{g['_priority']:.1f}"
                lines.append(
                    f"  [id={g['id']} imp={g.get('importance',9)} pri={score}] "
                    f"{g.get('title','')} — {g.get('description','')}"
                )
            lines.append("")
    except Exception as e:
        log.debug(f"load_typed_context_block: goals failed: {e}")

    try:
        beliefs = await fetch_dicts(
            f"SELECT id, topic, content, confidence "
            f"FROM {_BELIEFS()} WHERE status = 'active' ORDER BY confidence DESC"
        )
        if beliefs:
            _typed_metric_read(_BELIEFS(), len(beliefs))
            lines.append("## Active Beliefs\n")
            for b in beliefs:
                lines.append(
                    f"  [id={b['id']} conf={b.get('confidence',7)}] "
                    f"[{b.get('topic','')}] {b.get('content','')}"
                )
            lines.append("")
    except Exception as e:
        log.debug(f"load_typed_context_block: beliefs failed: {e}")

    try:
        # Load concept steps (top-level plan entries)
        concepts = await fetch_dicts(
            f"SELECT p.id, p.goal_id, p.step_order, p.description, p.status, "
            f"p.step_type, p.target, p.approval, g.title as goal_title "
            f"FROM {_PLANS()} p "
            f"LEFT JOIN {_GOALS()} g ON g.id = p.goal_id "
            f"WHERE p.status IN ('pending','in_progress') AND p.step_type = 'concept' "
            f"ORDER BY p.goal_id, p.step_order"
        )
        if concepts:
            _typed_metric_read(_PLANS(), len(concepts))
            lines.append("## Active Plans\n")
            by_goal: dict = {}
            for p in concepts:
                gid = p.get("goal_id", "?")
                by_goal.setdefault(gid, {"title": p.get("goal_title") or f"(ad-hoc)", "steps": []})
                by_goal[gid]["steps"].append(p)
            for gid, gdata in by_goal.items():
                lines.append(f"  **{gdata['title']}**")
                for step in gdata["steps"]:
                    status_mark = "▶" if step.get("status") == "in_progress" else "○"
                    target_tag = f" →{step.get('target')}" if step.get("target") != "model" else ""
                    approval_tag = f" [{step.get('approval')}]" if step.get("approval") not in ("approved", "auto") else ""
                    lines.append(
                        f"    {status_mark} [id={step.get('id','?')} step={step.get('step_order','?')}] "
                        f"{step.get('description','')}{target_tag}{approval_tag}"
                    )
                    # Load child task steps for this concept
                    tasks = await fetch_dicts(
                        f"SELECT id, description, status, target, tool_call, approval "
                        f"FROM {_PLANS()} WHERE parent_id = {step['id']} "
                        f"AND status IN ('pending','in_progress') ORDER BY step_order"
                    )
                    if tasks:
                        for t in tasks:
                            t_mark = "▸" if t.get("status") == "in_progress" else "·"
                            t_target = f" →{t.get('target')}" if t.get("target") != "model" else ""
                            t_approval = f" [{t.get('approval')}]" if t.get("approval") not in ("approved", "auto") else ""
                            lines.append(
                                f"      {t_mark} [{t.get('id','?')}] {t.get('description','')}{t_target}{t_approval}"
                            )
            lines.append("")
        else:
            # Fallback: check for legacy plan rows (no step_type column yet, or all concept)
            plans = await fetch_dicts(
                f"SELECT p.id, p.goal_id, p.step_order, p.description, p.status, g.title as goal_title "
                f"FROM {_PLANS()} p "
                f"LEFT JOIN {_GOALS()} g ON g.id = p.goal_id "
                f"WHERE p.status IN ('pending','in_progress') "
                f"ORDER BY p.goal_id, p.step_order"
            )
            if plans:
                _typed_metric_read(_PLANS(), len(plans))
                lines.append("## Active Plans\n")
                by_goal2: dict = {}
                for p in plans:
                    gid = p.get("goal_id", "?")
                    by_goal2.setdefault(gid, {"title": p.get("goal_title", f"goal {gid}"), "steps": []})
                    by_goal2[gid]["steps"].append(p)
                for gid, gdata in by_goal2.items():
                    lines.append(f"  **{gdata['title']}**")
                    for step in gdata["steps"]:
                        status_mark = "▶" if step.get("status") == "in_progress" else "○"
                        lines.append(
                            f"    {status_mark} [id={step.get('id','?')} step={step.get('step_order','?')}] {step.get('description','')}"
                        )
                lines.append("")
    except Exception as e:
        log.debug(f"load_typed_context_block: plans failed: {e}")

    try:
        conditioned = await fetch_dicts(
            f"SELECT id, topic, `trigger`, `reaction`, strength "
            f"FROM {_CONDITIONED()} WHERE status = 'active' ORDER BY strength DESC"
        )
        if conditioned:
            _typed_metric_read(_CONDITIONED(), len(conditioned))
            lines.append("## Conditioned Behaviors\n")
            for c in conditioned:
                lines.append(
                    f"  [id={c['id']} strength={c.get('strength',5)}] "
                    f"[{c.get('topic','')}] "
                    f"TRIGGER: {c.get('trigger','')} → REACT: {c.get('reaction','')}"
                )
            lines.append("")
    except Exception as e:
        log.debug(f"load_typed_context_block: conditioned failed: {e}")

    try:
        autobio = await fetch_dicts(
            f"SELECT id, topic, content, importance "
            f"FROM {_AUTOBIOGRAPHICAL()} ORDER BY importance DESC LIMIT 20"
        )
        if autobio:
            _typed_metric_read(_AUTOBIOGRAPHICAL(), len(autobio))
            lines.append("## Autobiographical Memory\n")
            for a in autobio:
                lines.append(
                    f"  [id={a['id']} imp={a.get('importance',7)}] "
                    f"[{a.get('topic','')}] {a.get('content','')}"
                )
            lines.append("")
    except Exception as e:
        log.debug(f"load_typed_context_block: autobiographical failed: {e}")

    try:
        prospective = await fetch_dicts(
            f"SELECT id, topic, content, due_at, importance "
            f"FROM {_PROSPECTIVE()} WHERE status = 'pending' ORDER BY importance DESC"
        )
        if prospective:
            _typed_metric_read(_PROSPECTIVE(), len(prospective))
            lines.append("## Pending Intentions\n")
            for p in prospective:
                due = f" (due: {p['due_at']})" if p.get("due_at") else ""
                lines.append(
                    f"  [id={p['id']} imp={p.get('importance',7)}] "
                    f"[{p.get('topic','')}]{due} {p.get('content','')}"
                )
            lines.append("")
    except Exception as e:
        log.debug(f"load_typed_context_block: prospective failed: {e}")

    try:
        drives = await fetch_dicts(
            f"SELECT name, value, description FROM {_DRIVES()} ORDER BY value DESC"
        )
        if drives:
            _typed_metric_read(_DRIVES(), len(drives))
            lines.append("## Active Drives\n")
            for d in drives:
                bar = int(round(d.get("value", 0.5) * 10))
                lines.append(
                    f"  {d['name']}: {d.get('value', 0.5):.2f}  {'█' * bar}{'░' * (10 - bar)}"
                    f"  — {d.get('description', '')}"
                )
            lines.append("")
    except Exception as e:
        log.debug(f"load_typed_context_block: drives failed: {e}")

    try:
        from database import fetch_dicts as _fetch_dicts_local
        self_rows = await _fetch_dicts_local(
            f"SELECT content FROM {_COGNITION()} "
            f"WHERE origin = 'self_model' AND topic = 'self-summary' "
            f"ORDER BY id DESC LIMIT 1"
        )
        if self_rows and self_rows[0].get("content"):
            lines.append("## Self-Model\n")
            lines.append(self_rows[0]["content"].strip() + "\n")
    except Exception as e:
        log.debug(f"load_typed_context_block: self-summary failed: {e}")

    # Cognitive state summary: sparse table counts and drive gaps for goal selection grounding
    try:
        from database import fetch_dicts as _fd_cs
        counts = {}
        for tbl_key, tbl_fn in [
            ("episodic", _EPISODIC), ("autobiographical", _AUTOBIOGRAPHICAL),
            ("prospective_active", None), ("procedural", _PROCEDURES),
        ]:
            try:
                if tbl_key == "prospective_active":
                    rows = await _fd_cs(f"SELECT COUNT(*) as n FROM {_PROSPECTIVE()} WHERE status='active'")
                elif tbl_key == "procedural":
                    rows = await _fd_cs(f"SELECT COUNT(*) as n FROM {tbl_fn()}")
                else:
                    rows = await _fd_cs(f"SELECT COUNT(*) as n FROM {tbl_fn()}")
                counts[tbl_key] = rows[0]["n"] if rows else 0
            except Exception:
                counts[tbl_key] = "?"

        # Drive gaps: drives below 75% of baseline
        drive_gaps = []
        try:
            drive_rows = await _fd_cs(
                f"SELECT name, value, baseline FROM {_DRIVES()} "
                f"WHERE baseline > 0 AND value < baseline * 0.75 ORDER BY (baseline - value) DESC"
            )
            drive_gaps = [f"{r['name']}({r['value']:.2f} vs baseline {r['baseline']:.2f})" for r in (drive_rows or [])]
        except Exception:
            pass

        state_lines = [
            f"  episodic: {counts.get('episodic','?')} entries  |  "
            f"autobiographical: {counts.get('autobiographical','?')} entries  |  "
            f"prospective (active): {counts.get('prospective_active','?')}  |  "
            f"procedural: {counts.get('procedural','?')} entries"
        ]
        if drive_gaps:
            state_lines.append(f"  Drives below baseline: {', '.join(drive_gaps)}")

        lines.append("## Cognitive State\n")
        lines.extend(state_lines)
        lines.append("")
    except Exception as e:
        log.debug(f"load_typed_context_block: cognitive state failed: {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-model: distill self-* rows into a standing summary
# ---------------------------------------------------------------------------

async def refresh_self_summary(llm_call_fn=None) -> str:
    """
    Distil all cognition rows (reflection, tool_failure, goal_health) into a 3-5 bullet standing summary.
    Saves the result to the cognition table with origin='self_model', topic='self-summary', importance=10.
    Returns the summary string, or "" if no self-* rows exist.

    llm_call_fn: optional async callable(model_key: str, prompt: str) -> str.
                 Defaults to agents._call_llm_text with 'summarizer-anthropic'.
    """
    from database import fetch_dicts as _fd
    from config import get_model_role

    try:
        all_rows = await _fd(
            f"SELECT id, `loop`, topic, content, importance FROM {_COGNITION()} "
            f"WHERE origin IN ('reflection', 'tool_failure', 'goal_health') "
            f"AND topic != 'self-summary' "
            f"ORDER BY importance DESC, id DESC LIMIT 40"
        ) or []
    except Exception as e:
        log.warning(f"refresh_self_summary: DB fetch failed: {e}")
        return ""
    if not all_rows:
        log.debug("refresh_self_summary: no self-* rows found, skipping")
        return ""

    lines = []
    for r in all_rows:
        lines.append(f"[{r.get('topic','')} imp={r.get('importance',5)}] {r.get('content','')[:300]}")
    rows_text = "\n".join(lines)

    prompt = (
        "You are a self-model distiller. Below are facts about an AI agent's capabilities, "
        "failures, and preferences, recorded from its own experience.\n\n"
        "Synthesise these into 3-5 concise bullet points (one sentence each) that capture "
        "the most important standing truths about this agent. "
        "Max 200 words total. Use plain text, no JSON. Start each bullet with '- '.\n\n"
        f"SELF-MODEL ROWS:\n{rows_text}"
    )

    summary_text = ""
    try:
        if llm_call_fn is not None:
            summary_text = await llm_call_fn(prompt)
        else:
            from agents import _call_llm_text
            model_key = get_model_role("summarizer") or "summarizer-anthropic"
            summary_text = await _call_llm_text(model_key, prompt)
    except Exception as e:
        log.warning(f"refresh_self_summary: LLM call failed: {e}")
        return ""

    if not summary_text:
        return ""

    summary_text = summary_text.strip()

    try:
        await save_cognition(
            origin="self_model",
            topic="self-summary",
            content=summary_text,
            importance=10,
        )
        log.info("refresh_self_summary: saved new self-summary row")
    except Exception as e:
        log.warning(f"refresh_self_summary: save failed: {e}")

    return summary_text


# ---------------------------------------------------------------------------
# Drive / Affect system — load, update, decay
# ---------------------------------------------------------------------------

async def load_drives() -> list[dict]:
    """Return all drive rows ordered by value descending."""
    from database import fetch_dicts
    try:
        rows = await fetch_dicts(
            f"SELECT id, name, description, value, baseline, decay_rate, source "
            f"FROM {_DRIVES()} ORDER BY value DESC"
        )
        _typed_metric_read(_DRIVES(), len(rows or []))
        return rows or []
    except Exception as e:
        log.debug(f"load_drives failed: {e}")
        return []


async def update_drive(name: str, value: float, source: str = "user") -> bool:
    """
    Set a drive value by name (0.0-1.0). Creates the row if absent.
    Returns True on success.
    """
    from database import execute_sql, execute_insert, fetch_dicts
    value = max(0.0, min(1.0, float(value)))
    name = name.strip().lower()[:64]
    tbl = _DRIVES()
    try:
        existing = await fetch_dicts(
            f"SELECT id FROM {tbl} WHERE name = '{name}' LIMIT 1"
        )
        if existing:
            await execute_sql(
                f"UPDATE {tbl} SET value={value:.4f}, source='{source}' WHERE name='{name}'"
            )
        else:
            # Insert with default baseline=value (user is explicitly setting it)
            await execute_insert(
                f"INSERT INTO {tbl} (name, description, value, baseline, decay_rate, source) "
                f"VALUES ('{name}', '', {value:.4f}, {value:.4f}, 0.05, '{source}')"
            )
        _typed_metric_write(tbl)
        return True
    except Exception as e:
        log.warning(f"update_drive({name}={value}): {e}")
        return False


async def decay_drives() -> int:
    """
    Apply per-cycle decay toward each drive's baseline.
    Called by reflection loop. Returns number of drives updated.
    """
    from database import fetch_dicts, execute_sql
    tbl = _DRIVES()
    try:
        drives = await fetch_dicts(
            f"SELECT id, name, value, baseline, decay_rate FROM {tbl}"
        )
    except Exception as e:
        log.debug(f"decay_drives: fetch failed: {e}")
        return 0

    updated = 0
    for d in (drives or []):
        v = float(d.get("value", 0.5))
        b = float(d.get("baseline", 0.5))
        r = float(d.get("decay_rate", 0.05))
        if abs(v - b) < 0.005:
            continue
        new_v = v + (b - v) * r
        new_v = round(max(0.0, min(1.0, new_v)), 4)
        try:
            await execute_sql(
                f"UPDATE {tbl} SET value={new_v} WHERE id={d['id']}"
            )
            updated += 1
        except Exception as e:
            log.debug(f"decay_drives: update id={d['id']} failed: {e}")

    if updated:
        log.debug(f"decay_drives: decayed {updated} drives toward baseline")
    return updated


async def update_drives_from_goals() -> dict:
    """
    Examine goal completion rates and nudge drive values accordingly.
    Called at the end of each reflection cycle.

    Rules:
      - For every goal completed since last reflection: +0.1 to task-completion
      - For every goal blocked: +0.05 to discomfort, -0.05 to task-completion
      - If zero recent completions and >3 active goals: task-completion decays faster this cycle
      - Always apply normal decay first (baseline pull)

    Returns summary dict.
    """
    from database import fetch_dicts
    summary = {"drives_updated": 0, "goals_done": 0, "goals_blocked": 0}

    try:
        goals = await fetch_dicts(
            f"SELECT status FROM {_GOALS()}"
        )
    except Exception as e:
        log.debug(f"update_drives_from_goals: goals fetch failed: {e}")
        return summary

    done_count    = sum(1 for g in (goals or []) if g.get("status") == "done")
    blocked_count = sum(1 for g in (goals or []) if g.get("status") == "blocked")
    active_count  = sum(1 for g in (goals or []) if g.get("status") == "active")

    summary["goals_done"]    = done_count
    summary["goals_blocked"] = blocked_count

    # Apply standard decay toward baseline for all drives
    decayed = await decay_drives()

    # Load current drives to nudge
    tbl = _DRIVES()
    try:
        drives = {d["name"]: d for d in (await fetch_dicts(f"SELECT * FROM {tbl}") or [])}
    except Exception as e:
        log.debug(f"update_drives_from_goals: drives fetch failed: {e}")
        return summary

    async def _nudge(name: str, delta: float) -> None:
        if name not in drives:
            return
        current = float(drives[name].get("value", 0.5))
        new_v = round(max(0.0, min(1.0, current + delta)), 4)
        try:
            await execute_sql(
                f"UPDATE {tbl} SET value={new_v}, source='reflection' WHERE name='{name}'"
            )
            _typed_metric_write(tbl)
            summary["drives_updated"] += 1
        except Exception as e:
            log.debug(f"_nudge({name}): {e}")

    if done_count > 0:
        await _nudge("task-completion", 0.05 * min(done_count, 3))
    if blocked_count > 0:
        await _nudge("discomfort",      0.05 * min(blocked_count, 2))
        await _nudge("task-completion", -0.03 * min(blocked_count, 2))
    if done_count == 0 and active_count > 3:
        await _nudge("task-completion", -0.02)

    log.info(
        f"update_drives_from_goals: done={done_count} blocked={blocked_count} "
        f"active={active_count} drives_updated={summary['drives_updated']} decayed={decayed}"
    )
    return summary


# ---------------------------------------------------------------------------
# Structured procedural memory — save / recall / inject
# ---------------------------------------------------------------------------

async def _upsert_procedure_vec(
    row_id: int, topic: str, task_type: str,
    embed_text: str, importance: int, outcome: str,
) -> None:
    """Fire-and-forget: embed procedure and upsert into samaritan_procedures collection."""
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if not vec:
            return
        coll = _PROC_COLLECTION()
        vec._ensure_collection(coll)
        await vec.upsert_memory(
            row_id=row_id,
            topic=topic,
            content=embed_text,
            importance=importance,
            tier="procedure",
            collection=coll,
        )
        # Add task_type and outcome as searchable payload fields
        await vec.set_payload(row_id, {"task_type": task_type, "outcome": outcome}, collection=coll)
    except Exception as e:
        log.warning(f"_upsert_procedure_vec failed (id={row_id}): {e}")


async def _update_procedure_outcome_vec(row_id: int, outcome: str) -> None:
    """Lightweight payload-only update when only outcome changes — no re-embed."""
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if vec:
            await vec.set_payload(row_id, {"outcome": outcome}, collection=_PROC_COLLECTION())
    except Exception as e:
        log.warning(f"_update_procedure_outcome_vec failed (id={row_id}): {e}")


async def save_procedure(
    topic: str,
    task_type: str,
    steps: list[dict],
    outcome: str = "unknown",
    notes: str = "",
    importance: int = 7,
    source: str = "assistant",
    session_id: str = "",
    id: int = 0,
) -> int:
    """
    Save or update a structured procedure in samaritan_procedural.
    id=0: INSERT new row.
    id>0: UPDATE existing row (increments run_count, updates outcome/notes).
    Returns the MySQL row_id.
    """
    from database import fetch_dicts
    outcome = outcome if outcome in ("success", "partial", "failure", "unknown") else "unknown"
    importance = max(1, min(10, int(importance)))
    steps_json = json.dumps(steps).replace("'", "''")
    embed_text = f"{task_type}: {topic}. Steps: " + "; ".join(s.get("action", "") for s in steps)
    content_val = embed_text.replace("'", "''")

    if id and id > 0:
        # Fetch current state
        rows = await fetch_dicts(
            f"SELECT run_count, success_count, steps, notes FROM {_PROCEDURES()} WHERE id = {id}"
        )
        if not rows:
            return 0
        cur = rows[0]
        new_run = int(cur.get("run_count") or 1) + 1
        new_success = int(cur.get("success_count") or 0) + (1 if outcome == "success" else 0)
        # Detect steps change for re-embed decision
        old_steps = cur.get("steps") or ""
        steps_changed = steps_json.replace("''", "'") != old_steps

        note_append = ""
        if notes:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            existing_notes = (cur.get("notes") or "").replace("'", "''")
            note_append = f"{existing_notes} | {ts}: {notes.replace(chr(39), chr(39)*2)}"

        parts = [
            f"outcome = '{outcome}'",
            f"run_count = {new_run}",
            f"success_count = {new_success}",
            f"last_run_at = NOW()",
            f"steps = '{steps_json}'",
            f"content = '{content_val}'",
            f"importance = {importance}",
        ]
        if note_append:
            parts.append(f"notes = '{note_append}'")
        elif notes:
            parts.append(f"notes = '{notes.replace(chr(39), chr(39)*2)}'")

        sql = f"UPDATE {_PROCEDURES()} SET {', '.join(parts)} WHERE id = {id}"
        await execute_sql(sql)
        _typed_metric_write(_PROCEDURES())

        if steps_changed:
            asyncio.create_task(_upsert_procedure_vec(id, topic, task_type, embed_text, importance, outcome))
        else:
            asyncio.create_task(_update_procedure_outcome_vec(id, outcome))
        return id
    else:
        # INSERT new
        t = topic.replace("'", "''")
        tt = task_type.replace("'", "''")
        n = notes.replace("'", "''") if notes else ""
        success_init = 1 if outcome == "success" else 0
        sql = (
            f"INSERT INTO {_PROCEDURES()} "
            f"(topic, task_type, content, steps, outcome, run_count, success_count, notes, "
            f"importance, source, session_id, last_run_at) "
            f"VALUES ('{t}', '{tt}', '{content_val}', '{steps_json}', '{outcome}', "
            f"1, {success_init}, '{n}', {importance}, '{source}', '{session_id}', NOW())"
        )
        row_id = await execute_insert(sql)
        _typed_metric_write(_PROCEDURES())
        asyncio.create_task(_upsert_procedure_vec(row_id, topic, task_type, embed_text, importance, outcome))
        return row_id


async def recall_procedures(
    query: str,
    task_type: str = "",
    top_k: int = 5,
    min_score: float = 0.50,
) -> list[dict]:
    """
    Semantic search for procedures relevant to a query.
    Filters by task_type if provided. Falls back to MySQL LIKE if Qdrant unavailable.
    Returns list of dicts with full procedure fields including steps (parsed JSON).
    """
    from database import fetch_dicts
    from plugin_memory_vector_qdrant import get_vector_api

    vec = get_vector_api()
    results = []

    if vec:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            coll = _PROC_COLLECTION()
            vec._ensure_proc_collection()
            vector = await vec.embed(query, prefix="search_query")
            filter_must = []
            if task_type:
                filter_must.append(FieldCondition(key="task_type", match=MatchValue(value=task_type)))
            qfilter = Filter(must=filter_must) if filter_must else None
            response = vec._qc.query_points(
                collection_name=coll,
                query=vector,
                query_filter=qfilter,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True,
            )
            ids = [r.id for r in response.points]
            scores = {r.id: round(r.score, 4) for r in response.points}
            if ids:
                ids_str = ", ".join(str(i) for i in ids)
                rows = await fetch_dicts(
                    f"SELECT id, topic, task_type, steps, outcome, run_count, success_count, notes, importance "
                    f"FROM {_PROCEDURES()} WHERE id IN ({ids_str})"
                )
                for row in rows:
                    row["score"] = scores.get(row["id"], 0.0)
                    if row.get("steps") and isinstance(row["steps"], str):
                        try:
                            row["steps"] = json.loads(row["steps"])
                        except Exception:
                            row["steps"] = []
                results = sorted(rows, key=lambda r: r.get("score", 0), reverse=True)
        except Exception as e:
            log.warning(f"recall_procedures: Qdrant search failed: {e}")

    if not results:
        # MySQL fallback
        try:
            like = f"%{query[:60]}%"
            tt_clause = f" AND task_type = '{task_type.replace(chr(39), chr(39)*2)}'" if task_type else ""
            rows = await fetch_dicts(
                f"SELECT id, topic, task_type, steps, outcome, run_count, success_count, notes, importance "
                f"FROM {_PROCEDURES()} "
                f"WHERE (task_type LIKE '{like}' OR topic LIKE '{like}'){tt_clause} "
                f"ORDER BY importance DESC LIMIT {top_k}"
            )
            for row in rows:
                if row.get("steps") and isinstance(row["steps"], str):
                    try:
                        row["steps"] = json.loads(row["steps"])
                    except Exception:
                        row["steps"] = []
            results = rows
        except Exception as e:
            log.warning(f"recall_procedures: MySQL fallback failed: {e}")

    if results:
        _typed_metric_read(_PROCEDURES(), len(results))
    return results


async def load_procedure_context_block(task_hint: str = "") -> str:
    """
    Return a ## Relevant Procedures block for prompt injection.
    Always injects importance >= 8 procedures.
    Also runs semantic recall against task_hint if provided (score >= 0.55).
    """
    from database import fetch_dicts
    lines = []
    seen_ids: set[int] = set()

    try:
        # Always-inject: high-importance procedures
        high_imp = await fetch_dicts(
            f"SELECT id, topic, task_type, steps, outcome, run_count, success_count, notes "
            f"FROM {_PROCEDURES()} WHERE importance >= 8 ORDER BY importance DESC LIMIT 10"
        )
        for row in high_imp:
            seen_ids.add(row["id"])
            if row.get("steps") and isinstance(row["steps"], str):
                try:
                    row["steps"] = json.loads(row["steps"])
                except Exception:
                    row["steps"] = []

        # Semantic recall on task_hint (score >= 0.55, exclude already seen)
        semantic_hits = []
        if task_hint:
            semantic_hits = await recall_procedures(query=task_hint, top_k=5, min_score=0.55)
            semantic_hits = [r for r in semantic_hits if r["id"] not in seen_ids]

        all_procs = high_imp + semantic_hits
        if not all_procs:
            return ""

        _typed_metric_read(_PROCEDURES(), len(all_procs))
        lines.append("## Relevant Procedures\n")
        for p in all_procs:
            steps_list = p.get("steps") or []
            step_text = "\n".join(
                f"    {s.get('step','?')}. {s.get('action','')} "
                f"{'[' + s['tool'] + ']' if s.get('tool') else ''}"
                f"{' // ' + s['note'] if s.get('note') else ''}"
                for s in steps_list
            )
            score_str = f" score={p['score']}" if p.get("score") else ""
            lines.append(
                f"  [id={p['id']} task_type={p.get('task_type','')} "
                f"outcome={p.get('outcome','?')} "
                f"runs={p.get('success_count','?')}/{p.get('run_count','?')}{score_str}]\n"
                f"  {p.get('topic','')}\n"
                f"{step_text}"
            )
            if p.get("notes"):
                lines.append(f"  Notes: {p['notes']}")
            lines.append("")
    except Exception as e:
        log.debug(f"load_procedure_context_block failed: {e}")
        return ""

    return "\n".join(lines)


async def _fetch_by_type(table: str, types_sql: str) -> list[dict]:
    """Fetch all ST rows whose type is in types_sql (comma-separated quoted values)."""
    from database import fetch_dicts
    try:
        return await fetch_dicts(
            f"SELECT id, topic, content, importance, source, type "
            f"FROM {table} WHERE type IN ({types_sql}) "
            f"ORDER BY importance DESC"
        )
    except Exception as e:
        log.warning(f"_fetch_by_type failed: {e}")
        return []


async def load_context_block(
    min_importance: int = 3,
    query: str = "",
    user_text: str = "",
    identity_name: str = "",
    memory_types_enabled: bool = False,
) -> str:
    """
    Return a formatted string of memories for prompt injection (short-term + relevant long-term).

    Two-pass retrieval (when vector plugin available):
      Pass 1: Semantic search using `query` (typically the topic slug).
      Pass 2: If pass 1 yields fewer than `two_pass_threshold` *quality* hits
              (score >= two_pass_quality_floor), re-query using `user_text`
              (the actual user message) for richer semantic signal.
              Only new IDs are merged.

    Config keys (plugins-enabled.json → plugin_config.memory):
      two_pass_threshold      int    (default 5)   — min quality hits to skip pass 2
      two_pass_quality_floor  float  (default 0.75) — min score to count as quality hit

    When no query or vector plugin unavailable:
      - Short-term: all rows meeting min_importance
      - Long-term: high-importance rows (importance >= 7)

    Includes the full list of known topics so the model can reuse existing
    categories rather than inventing new ones each turn.
    Returns empty string if no memories and no topics.
    """
    cfg = _mem_plugin_cfg()
    two_pass_threshold = int(cfg.get("two_pass_threshold", 5))
    two_pass_quality_floor = float(cfg.get("two_pass_quality_floor", 0.75))
    # Types that are always injected regardless of semantic score (when memory_types_enabled)
    _always_types: list[str] = cfg.get("always_inject_types", ["belief", "autobiographical"]) if memory_types_enabled else []

    # Strip vocative address of identity name from queries
    # ("Samaritan, tell me about X" → "tell me about X")
    # identity_name comes from the model config (per-model) via llm-models.json.
    _identity = identity_name
    if _identity:
        _voc_re = re.compile(rf'^\s*{re.escape(_identity)}[\s,;:!.—–-]+', re.IGNORECASE)
        query = (_voc_re.sub('', query).strip() or query)
        if user_text:
            user_text = (_voc_re.sub('', user_text).strip() or user_text)

    from plugin_memory_vector_qdrant import get_vector_api
    vec = get_vector_api()

    topics_task = asyncio.create_task(load_topic_list())

    if vec and query:
        # --- Semantic retrieval path ---
        always_importance = vec.cfg().get("min_importance_always", 8)

        # Pass 1: query with topic slug (or fallback text)
        _coll = _COLLECTION()
        semantic_st_task = asyncio.create_task(vec.search_memories(query, tier="short", collection=_coll))
        semantic_lt_task = asyncio.create_task(vec.search_memories(query, tier="long", collection=_coll))
        always_task      = asyncio.create_task(
            load_short_term(limit=10000, min_importance=always_importance)
        )
        # Always-inject by type: fetch active rows of high-priority types from ST
        if _always_types:
            _types_sql = ", ".join(f"'{t}'" for t in _always_types)
            always_type_task = asyncio.create_task(
                _fetch_by_type(_ST(), _types_sql)
            )
        else:
            always_type_task = None

        gather_args = [semantic_st_task, semantic_lt_task, always_task, topics_task]
        if always_type_task:
            gather_args.append(always_type_task)
        gather_results = await asyncio.gather(*gather_args)
        semantic_st, semantic_lt, always_rows, topics = gather_results[:4]
        always_type_rows: list[dict] = gather_results[4] if always_type_task else []

        # Merge: always_rows first, then always_type_rows, then short-term semantic hits
        seen_ids_st = {str(r.get("id", "")) for r in always_rows}
        merged_st = list(always_rows)
        for row in always_type_rows:
            if str(row.get("id", "")) not in seen_ids_st:
                merged_st.append(row)
                seen_ids_st.add(str(row.get("id", "")))
        for hit in semantic_st:
            if str(hit.get("id", "")) not in seen_ids_st:
                merged_st.append(hit)
                seen_ids_st.add(str(hit.get("id", "")))

        # Count only high-quality hits (above quality floor) for two-pass decision
        pass1_quality_count = sum(1 for h in semantic_st if h.get("score", 0) >= two_pass_quality_floor) + \
                              sum(1 for h in semantic_lt if h.get("score", 0) >= two_pass_quality_floor)
        pass1_total_count = len(semantic_st) + len(semantic_lt)
        used_two_pass = False

        # Pass 2: if pass 1 has few quality hits and we have user_text that differs from query
        if (
            pass1_quality_count < two_pass_threshold
            and user_text
            and user_text.strip().lower() != query.strip().lower()
        ):
            used_two_pass = True
            seen_ids_lt = {str(h.get("id", "")) for h in semantic_lt}
            p2_st_task = asyncio.create_task(vec.search_memories(user_text, tier="short", collection=_coll))
            p2_lt_task = asyncio.create_task(vec.search_memories(user_text, tier="long", collection=_coll))
            p2_st, p2_lt = await asyncio.gather(p2_st_task, p2_lt_task)

            p2_extra = 0
            for hit in p2_st:
                if str(hit.get("id", "")) not in seen_ids_st:
                    merged_st.append(hit)
                    seen_ids_st.add(str(hit.get("id", "")))
                    p2_extra += 1
            for hit in p2_lt:
                if str(hit.get("id", "")) not in seen_ids_lt:
                    semantic_lt.append(hit)
                    seen_ids_lt.add(str(hit.get("id", "")))
                    p2_extra += 1
            log.debug(
                f"load_context_block: pass2 user_text query added {p2_extra} extra hits "
                f"(p2_st={len(p2_st)} p2_lt={len(p2_lt)})"
            )
        # Score distribution for diagnostics
        all_scores = [h.get("score", 0) for h in semantic_st + semantic_lt]
        score_dist = ""
        if all_scores:
            above_80 = sum(1 for s in all_scores if s >= 0.80)
            above_65 = sum(1 for s in all_scores if s >= 0.65)
            above_45 = sum(1 for s in all_scores if s >= 0.45)
            top3 = sorted(all_scores, reverse=True)[:3]
            bot3 = sorted(all_scores)[:3]
            score_dist = f" scores: top3={top3} bot3={bot3} >=0.80:{above_80} >=0.65:{above_65} >=0.45:{above_45}"
        log.debug(
            f"load_context_block: query={query!r} pass1 total={pass1_total_count} quality(>={two_pass_quality_floor})={pass1_quality_count} "
            f"two_pass={'YES' if used_two_pass else 'no'}{score_dist}"
        )

        # Update retrieval stats
        _retrieval_stats["total"] += 1
        n = _retrieval_stats["total"]
        if used_two_pass:
            _retrieval_stats["two_pass_needed"] += 1
            # Running average of extra hits from pass 2
            old_avg = _retrieval_stats["pass2_avg_extra"]
            _retrieval_stats["pass2_avg_extra"] = old_avg + (p2_extra - old_avg) / _retrieval_stats["two_pass_needed"]
        else:
            _retrieval_stats["single_pass_sufficient"] += 1
        # Running average of pass-1 quality hit count (above quality floor)
        old_p1 = _retrieval_stats["pass1_avg_hits"]
        _retrieval_stats["pass1_avg_hits"] = old_p1 + (pass1_quality_count - old_p1) / n

        # Only update last_accessed for semantically retrieved short-term rows
        semantic_ids = [h.get("id") for h in semantic_st if h.get("id")]
        if semantic_ids:
            asyncio.create_task(_update_last_accessed(semantic_ids))
        lt_ids = [h.get("id") for h in semantic_lt if h.get("id")]
        if lt_ids:
            asyncio.create_task(_update_lt_last_accessed(lt_ids))

        # Fuzzy topic match: pull ST rows for topics matching query AND user_text
        fuzzy_topics = set(_fuzzy_match_topics(query, topics))
        if user_text and user_text.strip().lower() != query.strip().lower():
            fuzzy_topics |= set(_fuzzy_match_topics(user_text, topics))
        if fuzzy_topics:
            from database import fetch_dicts as _fetch_dicts
            placeholders = ", ".join(f"'{t}'" for t in fuzzy_topics)
            fuzzy_rows = await _fetch_dicts(
                f"SELECT id, topic, content, importance FROM {_ST()} "
                f"WHERE topic IN ({placeholders})"
            )
            fuzzy_added = 0
            for hit in fuzzy_rows:
                if str(hit.get("id", "")) not in seen_ids_st:
                    merged_st.append(hit)
                    seen_ids_st.add(str(hit.get("id", "")))
                    fuzzy_added += 1
            log.debug(f"load_context_block: fuzzy_topics={list(fuzzy_topics)} matched={len(fuzzy_rows)} added={fuzzy_added}")

        log.debug(
            f"load_context_block: st_semantic={len(semantic_st)} "
            f"always={len(always_rows)} lt_semantic={len(semantic_lt)} "
            f"merged_st={len(merged_st)} two_pass={used_two_pass}"
        )
    else:
        # --- Fallback: load by importance threshold ---
        _retrieval_stats["total"] += 1
        _retrieval_stats["fallback_no_vec"] += 1
        merged_st, topics = await asyncio.gather(
            load_short_term(limit=10000, min_importance=min_importance),
            topics_task,
        )
        # Pull high-importance long-term rows directly
        lt_raw = await execute_sql(
            f"SELECT id, topic, content, importance FROM {_LT()} "
            f"WHERE importance >= 7 ORDER BY importance DESC LIMIT 20"
        )
        semantic_lt = _parse_table(lt_raw)
        row_ids = [row.get("id", "") for row in merged_st if row.get("id", "")]
        if row_ids:
            asyncio.create_task(_update_last_accessed(row_ids))
        lt_ids = [r.get("id", "") for r in semantic_lt if r.get("id", "")]
        if lt_ids:
            asyncio.create_task(_update_lt_last_accessed(lt_ids))

    if not merged_st and not semantic_lt and not topics:
        return ""

    # Log injected topic breakdown for diagnostics
    st_topics = {}
    for r in merged_st:
        t = r.get("topic", "general")
        st_topics[t] = st_topics.get(t, 0) + 1
    lt_topics = {}
    for r in semantic_lt:
        t = r.get("topic", "general")
        lt_topics[t] = lt_topics.get(t, 0) + 1
    log.debug(
        f"load_context_block: injecting ST={len(merged_st)} rows ({len(st_topics)} topics: {dict(list(st_topics.items())[:10])}) "
        f"LT={len(semantic_lt)} rows ({len(lt_topics)} topics: {dict(list(lt_topics.items())[:10])})"
    )

    lines = ["## Active Memory (short-term recall)\n"]

    if merged_st:
        by_topic: dict[str, list[dict]] = {}
        for row in merged_st:
            t = row.get("topic", "general")
            by_topic.setdefault(t, []).append(row)

        for topic, items in by_topic.items():
            lines.append(f"**{topic}**")
            for item in items:
                imp = item.get("importance", 5)
                mem_type = item.get("type", "context")
                type_tag = f" type={mem_type}" if memory_types_enabled and mem_type and mem_type != "context" else ""
                lines.append(f"  [imp={imp}{type_tag}] {item.get('content', '')}")
            lines.append("")

    if semantic_lt:
        lines.append("## Long-term Memory (relevant recalled)\n")
        by_topic_lt: dict[str, list[dict]] = {}
        for row in semantic_lt:
            t = row.get("topic", "general")
            by_topic_lt.setdefault(t, []).append(row)

        for topic, items in by_topic_lt.items():
            lines.append(f"**{topic}**")
            for item in items:
                imp = item.get("importance", 5)
                mem_type = item.get("type", "context")
                type_tag = f" type={mem_type}" if memory_types_enabled and mem_type and mem_type != "context" else ""
                lines.append(f"  [imp={imp}{type_tag}] {item.get('content', '')}")
            lines.append("")

    if topics:
        lines.append(f"**Known topics** (reuse these for new saves; add new ones only when needed):")
        lines.append(f"  {', '.join(topics)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Temporal context: proactive time-of-day routine injection
# ---------------------------------------------------------------------------

async def load_temporal_context() -> str:
    """Return a block of recurring routines/patterns around the current time of day.

    Queries ST for topics that appear on 2+ distinct dates within a ±90-minute
    window of the current display time (last 30 days), then pulls matching LT
    summaries for those topics.  Returns a formatted markdown block, or "" if
    nothing relevant is found.

    Config toggle: plugins-enabled.json → plugin_config.memory.temporal.context_injection
    (default True when temporal section exists, False otherwise).
    """
    cfg = _mem_plugin_cfg()
    tcfg = cfg.get("temporal", {})
    if not isinstance(tcfg, dict) or not tcfg.get("context_injection", True):
        return ""

    import datetime as _dt
    from config import now_display

    now = now_display()
    t_start = (now - _dt.timedelta(minutes=90)).strftime("%H:%M")
    t_end   = (now + _dt.timedelta(minutes=90)).strftime("%H:%M")

    st_table = _ST()
    lt_table = _LT()

    # Handle midnight wraparound
    if t_start <= t_end:
        time_filter = f"TIME(created_at) BETWEEN '{t_start}' AND '{t_end}'"
    else:
        time_filter = f"(TIME(created_at) >= '{t_start}' OR TIME(created_at) <= '{t_end}')"

    # Find recurring ST topics in this time window across multiple dates
    topic_sql = (
        f"SELECT topic, COUNT(DISTINCT DATE(created_at)) AS days_seen, "
        f"GROUP_CONCAT(DISTINCT LEFT(content, 120) ORDER BY created_at DESC SEPARATOR ' | ') AS samples "
        f"FROM {st_table} "
        f"WHERE {time_filter} "
        f"AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) "
        f"AND source IN ('user', 'assistant') "
        f"AND topic IS NOT NULL AND topic != '' "
        f"GROUP BY topic "
        f"HAVING days_seen >= 2 "
        f"ORDER BY days_seen DESC "
        f"LIMIT 5"
    )
    try:
        topic_result = await execute_sql(topic_sql)
    except Exception as e:
        log.debug(f"load_temporal_context: ST topic query failed: {e}")
        return ""

    # Parse topics from result
    topics: list[str] = []
    lines = topic_result.strip().split("\n") if topic_result else []
    if len(lines) >= 3 and "(no rows)" not in topic_result:
        for line in lines[2:]:
            parts = [p.strip() for p in line.split("|")]
            if parts and parts[0]:
                topics.append(parts[0])

    if not topics:
        # Fall back: check LT directly for content mentioning times near now
        hour = now.hour
        # Search for common time patterns in LT content
        time_strs = [f"{hour}:", f"{hour:02d}:"]
        if hour > 12:
            ampm_h = hour - 12
            time_strs.extend([f"{ampm_h}:", f"{ampm_h} pm", f"{ampm_h}pm"])
        time_filter_lt = " OR ".join(f"content LIKE '%{t}%'" for t in time_strs)
        # Also check for time-of-day keywords
        _period_kw = []
        if 5 <= hour < 12:
            _period_kw = ["morning"]
        elif 12 <= hour < 17:
            _period_kw = ["afternoon", "lunch"]
        elif 17 <= hour < 21:
            _period_kw = ["evening", "pick up", "pickup", "dinner"]
        elif 21 <= hour or hour < 5:
            _period_kw = ["night", "bedtime"]
        if _period_kw:
            kw_filter = " OR ".join(f"content LIKE '%{kw}%'" for kw in _period_kw)
            time_filter_lt = f"({time_filter_lt}) OR ({kw_filter})"

        lt_sql = (
            f"SELECT LEFT(content, 200) AS content FROM {lt_table} "
            f"WHERE ({time_filter_lt}) "
            f"AND importance >= 5 "
            f"ORDER BY importance DESC, created_at DESC "
            f"LIMIT 5"
        )
        try:
            lt_result = await execute_sql(lt_sql)
        except Exception as e:
            log.debug(f"load_temporal_context: LT direct query failed: {e}")
            return ""

        if not lt_result or "(no rows)" in lt_result:
            return ""

        block = (
            f"## Temporal Context (routines around {t_start}–{t_end})\n"
            f"{lt_result}"
        )
        log.debug(f"load_temporal_context: LT direct hit, block={len(block)} chars")
        return block

    # We found recurring ST topics — now pull matching LT summaries
    esc_topics = [t.replace("'", "''").replace("\\", "\\\\") for t in topics]
    topic_like = " OR ".join(
        f"content LIKE '%{t}%' OR topic LIKE '%{t}%'" for t in esc_topics
    )
    lt_sql = (
        f"SELECT LEFT(content, 200) AS content FROM {lt_table} "
        f"WHERE ({topic_like}) "
        f"AND importance >= 4 "
        f"ORDER BY importance DESC, created_at DESC "
        f"LIMIT 5"
    )
    try:
        lt_result = await execute_sql(lt_sql)
    except Exception as e:
        log.debug(f"load_temporal_context: LT topic query failed: {e}")
        lt_result = ""

    parts = [f"## Temporal Context (routines around {t_start}–{t_end})"]
    if lt_result and "(no rows)" not in lt_result:
        parts.append(lt_result)
    else:
        # Fall back to ST samples
        parts.append(topic_result)

    block = "\n".join(parts)
    log.debug(f"load_temporal_context: {len(topics)} recurring topics, block={len(block)} chars")
    return block


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
        "You are Samaritan, an AI assistant, writing your own memory journal after a conversation. "
        "Your task: distill this conversation into memories YOU will carry forward. "
        "Write from your own perspective — what did YOU learn, conclude, recommend, or observe? "
        "What did the user tell you that you should remember? "
        "Output ONLY valid JSON — a list of objects with keys: "
        "topic (short kebab-case string), content (one concise sentence written from your perspective), "
        "importance (1-10 int), "
        "source ('user' = fact the user stated; 'session' = your own conclusion, estimate, recommendation, "
        "or neutral shared context — anything that is NOT a verbatim user statement). "
        "Most USER-turn content should be source='user'. Your own conclusions, assessments, and recommendations "
        "should be source='session'. Do NOT use source='assistant'. "
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
                    source = str(item.get("source", "session"))
                    # Summaries are never verbatim chat — force non-assistant source
                    if source == "assistant":
                        source = "session"
                    if topic and content:
                        new_id = await save_cognition(
                            origin="summary",
                            topic=topic,
                            content=content,
                            importance=importance,
                            source=source,
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
        f"INSERT INTO {_SUM()} "
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
