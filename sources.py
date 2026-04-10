"""
Universal knowledge-sources index — canonicalization, authority lookup,
content hashing, and source_record/source_query/source_reference exec functions.

Tables (MySQL, mymcp database):
  - samaritan_sources: main index
  - samaritan_source_references: outcome log
  - samaritan_belief_sources: belief↔source join
  - samaritan_source_authority: domain-pattern → initial scoring heuristic

This module provides:
  - canonicalize_url(url) — normalize URLs before storage (dedupe key)
  - lookup_authority(url) — domain-pattern match → (label, truth_score, half_life)
  - compute_content_hash(content) — sha256 of normalized content
  - _source_record_exec, _source_query_exec, _source_reference_exec — MCP execs
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional
from urllib.parse import (
    urlparse, urlunparse, parse_qsl, urlencode, ParseResult,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------

# Tracking parameters stripped on canonicalization. Case-insensitive match.
_TRACKING_PARAM_PATTERNS = [
    re.compile(r"^utm_", re.IGNORECASE),
    re.compile(r"^fbclid$", re.IGNORECASE),
    re.compile(r"^gclid$", re.IGNORECASE),
    re.compile(r"^mc_cid$", re.IGNORECASE),
    re.compile(r"^mc_eid$", re.IGNORECASE),
    re.compile(r"^_hsenc$", re.IGNORECASE),
    re.compile(r"^_hsmi$", re.IGNORECASE),
    re.compile(r"^ref$", re.IGNORECASE),
    re.compile(r"^ref_$", re.IGNORECASE),
    re.compile(r"^share$", re.IGNORECASE),
    re.compile(r"^source$", re.IGNORECASE),
    re.compile(r"^igshid$", re.IGNORECASE),
    re.compile(r"^si$", re.IGNORECASE),
]

_SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "buff.ly", "is.gd", "rebrand.ly", "short.link",
}

# Default schemes. Ports matching the scheme default are stripped.
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _is_tracking_param(key: str) -> bool:
    return any(p.match(key) for p in _TRACKING_PARAM_PATTERNS)


def _resolve_shortener(url: str, timeout: float = 5.0) -> str:
    """Resolve a URL shortener by following HTTP redirects.

    Returns the final URL, or the original URL if resolution fails.
    Uses HEAD to avoid downloading full content.
    """
    try:
        import requests
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        return resp.url or url
    except Exception as e:
        log.debug(f"shortener resolve failed for {url}: {e}")
        return url


def canonicalize_url(url: str, resolve_shorteners: bool = True) -> str:
    """Normalize a URL for use as a dedupe key.

    Steps:
      1. Strip whitespace + normalize unicode
      2. If shortener domain, resolve via HEAD redirect chain
      3. Lowercase scheme + host
      4. Strip default port (:80 for http, :443 for https)
      5. Strip tracking params (utm_*, fbclid, gclid, ref, etc.)
      6. Sort remaining query params alphabetically
      7. Normalize trailing slash (remove, unless path is "/")
      8. Strip fragment (#anchor)

    Args:
        url: The raw URL
        resolve_shorteners: If True, resolve shortener domains via HEAD

    Returns:
        Canonicalized URL string, or empty string if input is malformed.
    """
    if not url or not isinstance(url, str):
        return ""

    url = url.strip()
    if not url:
        return ""

    # Add scheme if missing (common in user-entered URLs)
    if "://" not in url:
        url = "http://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return ""

    if not parsed.netloc:
        return ""

    # Step 2: Resolve shorteners
    host = parsed.netloc.lower().split(":")[0]  # strip port for comparison
    if resolve_shorteners and host in _SHORTENER_DOMAINS:
        resolved = _resolve_shortener(url)
        if resolved and resolved != url:
            return canonicalize_url(resolved, resolve_shorteners=False)

    # Step 3: Lowercase scheme + host
    scheme = parsed.scheme.lower() or "http"
    netloc = parsed.netloc.lower()

    # Step 4: Strip default port
    if ":" in netloc:
        host_only, port_str = netloc.rsplit(":", 1)
        try:
            port = int(port_str)
            if _DEFAULT_PORTS.get(scheme) == port:
                netloc = host_only
        except ValueError:
            pass  # non-numeric port, leave as-is

    # Step 5-6: Strip tracking params, sort remaining
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(k, v) for k, v in query_pairs if not _is_tracking_param(k)]
    filtered.sort(key=lambda kv: kv[0].lower())
    query = urlencode(filtered, doseq=True)

    # Step 7: Normalize trailing slash
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Step 8: Drop fragment
    canonical = urlunparse(ParseResult(
        scheme=scheme, netloc=netloc, path=path,
        params=parsed.params, query=query, fragment="",
    ))

    return canonical


# ---------------------------------------------------------------------------
# Authority lookup
# ---------------------------------------------------------------------------

async def lookup_authority(url: str) -> tuple[str, int, int]:
    """Match URL's domain against samaritan_source_authority, return scoring defaults.

    Domain matching is suffix-based: "docs.aws.amazon.com" matches pattern
    "docs.aws.amazon.com" (exact) or "aws.amazon.com" (parent). Lowest
    match_priority wins.

    Returns:
        (authority_label, initial_truth_score, default_half_life_days)

    Falls back to the '*' row if no domain match. If even that row is
    missing (shouldn't happen post-seed), returns ('unknown', 4, 90).
    """
    from database import fetch_dicts

    if not url:
        return ("unknown", 4, 90)

    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.netloc.lower().split(":")[0]
    except Exception:
        return ("unknown", 4, 90)

    if not host:
        return ("unknown", 4, 90)

    # Build candidate patterns: full host + progressively shorter suffixes
    # e.g. "docs.aws.amazon.com" → ["docs.aws.amazon.com", "aws.amazon.com", "amazon.com"]
    parts = host.split(".")
    candidates = []
    for i in range(len(parts) - 1):  # stop before TLD-only
        candidates.append(".".join(parts[i:]))
    if host not in candidates:
        candidates.insert(0, host)

    if not candidates:
        candidates = [host]

    # Query for any matching pattern, ordered by priority ASC (lowest wins)
    in_clause = ",".join(
        f"'{c.replace(chr(39), chr(39)*2)}'" for c in candidates
    )
    sql = (
        f"SELECT authority_label, initial_truth_score, default_half_life_days "
        f"FROM mymcp.samaritan_source_authority "
        f"WHERE domain_pattern IN ({in_clause}) "
        f"ORDER BY match_priority ASC LIMIT 1"
    )

    try:
        rows = await fetch_dicts(sql)
        if rows:
            r = rows[0]
            return (
                r["authority_label"],
                int(r["initial_truth_score"]),
                int(r["default_half_life_days"]),
            )
    except Exception as e:
        log.warning(f"lookup_authority query failed: {e}")

    # Fallback: fetch the '*' wildcard row
    try:
        rows = await fetch_dicts(
            "SELECT authority_label, initial_truth_score, default_half_life_days "
            "FROM mymcp.samaritan_source_authority WHERE domain_pattern = '*' LIMIT 1"
        )
        if rows:
            r = rows[0]
            return (
                r["authority_label"],
                int(r["initial_truth_score"]),
                int(r["default_half_life_days"]),
            )
    except Exception as e:
        log.warning(f"lookup_authority fallback query failed: {e}")

    return ("unknown", 4, 90)


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def compute_content_hash(content: str) -> str:
    """Return sha256 hex of content after whitespace normalization.

    Normalization: collapse runs of whitespace to single space, strip outer.
    Two pages with identical content but different formatting hash the same.
    """
    if not content:
        return ""
    normalized = re.sub(r"\s+", " ", content).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# MCP exec functions: source_record, source_query, source_reference
# ---------------------------------------------------------------------------

import asyncio
import json
from datetime import datetime

# Qdrant collection for semantic source search. 768-dim (nomic-embed-text), COSINE.
SOURCE_COLLECTION = "samaritan_sources"
EMBED_MODEL_NAME = "nomic-embed-text"
EMBED_VERSION = 1


def _build_embed_text(title: str, summary: str) -> str:
    """Combine title + summary for embedding. Title alone is too short for good retrieval."""
    t = (title or "").strip()
    s = (summary or "").strip()
    if t and s:
        return f"{t}. {s}"
    return t or s


async def _upsert_source_vec(
    src_id: int,
    canonical_url: str,
    title: str,
    summary: str,
    domain_tags_json: Optional[str],
    authority: str,
    source_type: str,
    collection: str = "",
) -> None:
    """Fire-and-forget: embed source and upsert into samaritan_sources qdrant collection.

    On success, UPDATE samaritan_sources SET embedding_model=EMBED_MODEL_NAME WHERE id=src_id.
    On failure, embedding_model stays NULL so a backfill cognition step can retry.
    """
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        from qdrant_client.models import PointStruct
        from database import execute_sql

        vec = get_vector_api()
        if not vec:
            log.debug(f"_upsert_source_vec: qdrant unavailable, leaving id={src_id} pending")
            return

        embed_text = _build_embed_text(title, summary)
        if not embed_text:
            log.debug(f"_upsert_source_vec: empty embed text for id={src_id}, skipping")
            return

        vec._ensure_collection(SOURCE_COLLECTION)
        vector = await vec.embed(embed_text, prefix="search_document")

        # Parse domain_tags JSON back into a list for Qdrant payload
        tags_payload: list = []
        if domain_tags_json:
            try:
                parsed = json.loads(domain_tags_json)
                if isinstance(parsed, list):
                    tags_payload = parsed
            except Exception:
                pass

        vec._qc.upsert(
            collection_name=SOURCE_COLLECTION,
            points=[PointStruct(
                id=src_id,
                vector=vector,
                payload={
                    "source_id":         src_id,
                    "canonical_url":     canonical_url,
                    "title":             title or "",
                    "summary":           summary or "",
                    "domain_tags":       tags_payload,
                    "authority":         authority or "unknown",
                    "collection":        collection or "",
                    "source_type":       source_type or "internet",
                    "embedding_model":   EMBED_MODEL_NAME,
                    "embedding_version": EMBED_VERSION,
                },
            )],
        )

        # Mark as embedded in MySQL so backfill scans skip it
        await execute_sql(
            f"UPDATE mymcp.samaritan_sources "
            f"SET embedding_model = {_sql_str(EMBED_MODEL_NAME)} "
            f"WHERE id = {int(src_id)}"
        )
        log.debug(f"_upsert_source_vec: embedded id={src_id} title={title[:40]!r}")

        # Log embed cost event (local ollama = free, but track token estimate)
        try:
            from cost_events import log_cost_event
            est_tokens = max(1, len(embed_text) // 4)
            await log_cost_event(
                provider="ollama",
                service=EMBED_MODEL_NAME,
                tool_name="source_embed",
                cost_usd=0.0,
                tokens_in=est_tokens,
                tokens_out=0,
                unit="tokens",
                notes="local embed, token count estimated",
            )
        except Exception:
            pass  # cost logging must never break embed
    except Exception as e:
        log.warning(f"_upsert_source_vec failed (id={src_id}): {e}")


def _sql_str(value: Optional[str]) -> str:
    """Quote a value for inline SQL or return NULL."""
    if value is None or value == "":
        return "NULL"
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _sql_json(value) -> str:
    """JSON-encode for inline SQL."""
    if value is None:
        return "NULL"
    if isinstance(value, str):
        # Already a JSON string?
        try:
            json.loads(value)
            return _sql_str(value)
        except (json.JSONDecodeError, ValueError):
            # Treat as single-tag list
            return _sql_str(json.dumps([value]))
    return _sql_str(json.dumps(value))


async def _source_record_exec(
    source_type: str = "internet",
    source_ref: str = "",
    title: str = "",
    summary: str = "",
    content: str = "",
    domain_tags: str = "",
    collection: str = "",
    authority_override: str = "",
    hash_source: str = "fetched",
) -> str:
    """Record a knowledge source in the index.

    Args:
        source_type: 'internet'|'drive'|'mysql'|'memory'|'procedure'|'codebase'|'llm-synthesis'
        source_ref: URL for internet, file_id for drive, table.query for mysql, etc.
        title: Human-readable title
        summary: 2-3 sentence summary of what this covers
        content: Full content for hashing (optional — falls back to summary)
        domain_tags: JSON array of tags (string) OR comma-separated list OR single tag
        collection: Optional coarse grouping
        authority_override: Optional authority label to override heuristic
        hash_source: 'fetched' when content/summary came from fetching the URL,
                     'synthesized' when it came from an LLM answer / query-filtered
                     extract. First source_recheck on a 'synthesized' row adopts
                     the fetched page as the baseline without applying drift penalty.

    Returns:
        Status string with id of new/existing source row.
    """
    from database import execute_sql, execute_insert, fetch_dicts

    source_type = source_type if source_type in (
        "internet", "drive", "mysql", "memory", "procedure", "codebase", "llm-synthesis"
    ) else "internet"

    hash_source = hash_source if hash_source in ("fetched", "synthesized") else "fetched"

    if not source_ref:
        return "source_record: source_ref is required"

    # Step (a): canonicalize if internet
    if source_type == "internet":
        canonical = canonicalize_url(source_ref)
        if not canonical:
            return f"source_record: malformed URL: {source_ref!r}"
    else:
        canonical = source_ref  # non-internet refs used as-is

    # Step (b): compute content_hash
    hash_input = content or summary or title or source_ref
    content_hash = compute_content_hash(hash_input)

    # Step (c): dedupe check by canonical_url alone.
    # URL is the source identity; content_hash tracks snapshot version, not identity.
    # If hash differs on re-fetch, update the snapshot fields on the existing row.
    try:
        existing = await fetch_dicts(
            f"SELECT id, content_hash FROM mymcp.samaritan_sources "
            f"WHERE canonical_url = {_sql_str(canonical)} LIMIT 1"
        )
        if existing:
            row = existing[0]
            src_id = row["id"]
            old_hash = row.get("content_hash") or ""
            snapshot_changed = bool(content_hash) and content_hash != old_hash
            if snapshot_changed:
                await execute_sql(
                    f"UPDATE mymcp.samaritan_sources "
                    f"SET usage_count = usage_count + 1, last_used_at = NOW(), "
                    f"last_checked_at = NOW(), content_hash = {_sql_str(content_hash)} "
                    f"WHERE id = {src_id}"
                )
                # Fetch fields needed for re-embed + fire async upsert
                try:
                    rows = await fetch_dicts(
                        f"SELECT title, summary, domain_tags, authority, source_type, collection "
                        f"FROM mymcp.samaritan_sources WHERE id = {src_id} LIMIT 1"
                    )
                    if rows:
                        r = rows[0]
                        tags_val = r.get("domain_tags")
                        tags_str = tags_val if isinstance(tags_val, str) else (
                            json.dumps(tags_val) if tags_val else None
                        )
                        asyncio.create_task(_upsert_source_vec(
                            src_id=src_id,
                            canonical_url=canonical,
                            title=r.get("title") or "",
                            summary=r.get("summary") or "",
                            domain_tags_json=tags_str,
                            authority=r.get("authority") or "unknown",
                            source_type=r.get("source_type") or "internet",
                            collection=r.get("collection") or "",
                        ))
                except Exception as e:
                    log.warning(f"source_record snapshot re-embed failed: {e}")
                return f"source_record: existing source id={src_id} reused, snapshot updated (usage_count++)"
            await execute_sql(
                f"UPDATE mymcp.samaritan_sources "
                f"SET usage_count = usage_count + 1, last_used_at = NOW() "
                f"WHERE id = {src_id}"
            )
            return f"source_record: existing source id={src_id} reused (usage_count++)"
    except Exception as e:
        log.warning(f"source_record dedupe check failed: {e}")

    # Step (d): authority lookup
    if source_type == "internet":
        authority_label, truth_score, half_life = await lookup_authority(canonical)
    else:
        # Non-internet defaults: high truth for local sources
        authority_label, truth_score, half_life = "internal", 7, 365

    if authority_override:
        authority_label = authority_override

    # Normalize tags to JSON
    if isinstance(domain_tags, str) and domain_tags:
        if domain_tags.strip().startswith("["):
            tags_json = domain_tags  # assume valid JSON
        else:
            tags_list = [t.strip() for t in domain_tags.split(",") if t.strip()]
            tags_json = json.dumps(tags_list) if tags_list else None
    else:
        tags_json = None

    # Step (e): insert
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sql = (
            f"INSERT INTO mymcp.samaritan_sources "
            f"(source_type, source_ref, canonical_url, title, summary, domain_tags, "
            f"collection, content_hash, hash_source, fetched_at, half_life_days, last_checked_at, "
            f"truth_score, applicability_score, authority, status, embedding_model) "
            f"VALUES ("
            f"{_sql_str(source_type)}, "
            f"{_sql_str(source_ref)}, "
            f"{_sql_str(canonical)}, "
            f"{_sql_str(title)}, "
            f"{_sql_str(summary)}, "
            f"{_sql_json(tags_json) if tags_json else 'NULL'}, "
            f"{_sql_str(collection) if collection else 'NULL'}, "
            f"{_sql_str(content_hash)}, "
            f"{_sql_str(hash_source)}, "
            f"'{now}', "
            f"{half_life}, "
            f"'{now}', "
            f"{truth_score}, "
            f"5, "  # applicability starts at 5, promoted by outcomes
            f"{_sql_str(authority_label)}, "
            f"'active', "
            f"NULL"  # embedding_model populated in Phase 4
            f")"
        )
        src_id = await execute_insert(sql)
        # Fire async embed + qdrant upsert. On success, updates embedding_model=nomic-embed-text.
        # On failure, embedding_model stays NULL for backfill to pick up later.
        asyncio.create_task(_upsert_source_vec(
            src_id=src_id,
            canonical_url=canonical,
            title=title,
            summary=summary,
            domain_tags_json=tags_json,
            authority=authority_label,
            source_type=source_type,
            collection=collection,
        ))
        return (
            f"source_record: created id={src_id} "
            f"[{authority_label}] truth={truth_score} half_life={half_life}d "
            f"(canonical={canonical[:80]})"
        )
    except Exception as e:
        return f"source_record insert failed: {e}"


async def _source_query_exec(
    query_text: str = "",
    tags: str = "",
    min_truth: int = 1,
    min_applicability: int = 1,
    max_age_days: int = 0,
    collection: str = "",
    limit: int = 10,
) -> str:
    """Query the knowledge-sources index.

    When query_text is provided and qdrant is available, uses semantic search
    with combined scoring:
        combined = semantic_sim * 0.5 + truth_score/10 * 0.25 + applicability/10 * 0.25
    Filters by min_truth/min_applicability/max_age_days/tags/collection at MySQL level
    (scores drift as source_reference outcomes accumulate; MySQL is source of truth).

    When query_text is empty OR qdrant unavailable, falls back to structured
    LIKE+tag match ranked by (truth desc, applicability desc, last_used_at desc).

    Args:
        query_text: Keywords/concepts to match against title/summary
        tags: Comma-separated tag list or single tag to match
        min_truth: Minimum truth_score (default 1)
        min_applicability: Minimum applicability_score (default 1)
        max_age_days: If > 0, filter to fetched_at within last N days
        collection: Filter to specific collection
        limit: Max results (default 10)

    Returns:
        Formatted result list or status string.
    """
    from database import fetch_dicts

    min_truth = max(1, min(10, int(min_truth)))
    min_applicability = max(1, min(10, int(min_applicability)))
    limit = max(1, min(50, int(limit)))

    # Helper to format a result row
    def _fmt(r: dict, label: str = "") -> str:
        sem = f" sem={r['_semantic']:.3f} score={r['_score']:.3f}" if "_semantic" in r else ""
        title_or_url = r.get("title") or (r.get("canonical_url") or "")[:80]
        return (
            f"  id={r['id']} [{r['authority']}] truth={r['truth_score']} "
            f"app={r['applicability_score']} use={r['usage_count']}{sem} "
            f"| {title_or_url}"
        )

    # ---- Semantic path: query_text + qdrant available ----
    if query_text:
        try:
            from plugin_memory_vector_qdrant import get_vector_api
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            vec = get_vector_api()
            if vec:
                vec._ensure_collection(SOURCE_COLLECTION)
                vector = await vec.embed(query_text, prefix="search_query")

                filter_must = []
                if collection:
                    filter_must.append(FieldCondition(
                        key="collection", match=MatchValue(value=collection)
                    ))
                qfilter = Filter(must=filter_must) if filter_must else None

                # Overfetch 3x to give headroom for MySQL-level filtering
                qresp = vec._qc.query_points(
                    collection_name=SOURCE_COLLECTION,
                    query=vector,
                    query_filter=qfilter,
                    limit=limit * 3,
                    score_threshold=0.30,
                    with_payload=False,
                )
                semantic_hits = {int(r.id): round(r.score, 4) for r in qresp.points}

                if semantic_hits:
                    ids_str = ",".join(str(i) for i in semantic_hits.keys())
                    where = [
                        "status = 'active'",
                        f"id IN ({ids_str})",
                        f"truth_score >= {min_truth}",
                        f"applicability_score >= {min_applicability}",
                    ]
                    if max_age_days and max_age_days > 0:
                        where.append(
                            f"fetched_at > DATE_SUB(NOW(), INTERVAL {int(max_age_days)} DAY)"
                        )
                    if tags:
                        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                        tag_ors = " OR ".join(
                            f"JSON_SEARCH(domain_tags, 'one', {_sql_str(t)}) IS NOT NULL"
                            for t in tag_list
                        )
                        if tag_ors:
                            where.append(f"({tag_ors})")

                    rows = await fetch_dicts(
                        f"SELECT id, source_type, canonical_url, title, summary, collection, "
                        f"domain_tags, authority, truth_score, applicability_score, usage_count, "
                        f"fetched_at, last_used_at "
                        f"FROM mymcp.samaritan_sources "
                        f"WHERE {' AND '.join(where)}"
                    )

                    # Rerank: semantic * 0.5 + truth/10 * 0.25 + app/10 * 0.25
                    for r in rows:
                        sem = semantic_hits.get(int(r["id"]), 0.0)
                        t = int(r["truth_score"]) / 10.0
                        a = int(r["applicability_score"]) / 10.0
                        r["_semantic"] = sem
                        r["_score"] = sem * 0.5 + t * 0.25 + a * 0.25

                    rows.sort(key=lambda r: r["_score"], reverse=True)
                    rows = rows[:limit]

                    if rows:
                        lines = [f"source_query: {len(rows)} semantic result(s)"]
                        for r in rows:
                            lines.append(_fmt(r))
                        return "\n".join(lines)
                # else: qdrant returned nothing → fall through to structured
        except Exception as e:
            log.warning(f"source_query semantic path failed, falling back: {e}")

    # ---- Structured path: no query_text OR semantic unavailable/empty ----
    where_clauses = ["status = 'active'"]
    where_clauses.append(f"truth_score >= {min_truth}")
    where_clauses.append(f"applicability_score >= {min_applicability}")

    if max_age_days and max_age_days > 0:
        where_clauses.append(f"fetched_at > DATE_SUB(NOW(), INTERVAL {int(max_age_days)} DAY)")

    if collection:
        where_clauses.append(f"collection = {_sql_str(collection)}")

    if query_text:
        q = query_text.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
        where_clauses.append(
            f"(title LIKE '%{q}%' OR summary LIKE '%{q}%' OR canonical_url LIKE '%{q}%')"
        )

    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        tag_ors = " OR ".join(
            f"JSON_SEARCH(domain_tags, 'one', {_sql_str(t)}) IS NOT NULL"
            for t in tag_list
        )
        if tag_ors:
            where_clauses.append(f"({tag_ors})")

    where_sql = " AND ".join(where_clauses)

    sql = (
        f"SELECT id, source_type, canonical_url, title, summary, collection, "
        f"domain_tags, authority, truth_score, applicability_score, usage_count, "
        f"fetched_at, last_used_at "
        f"FROM mymcp.samaritan_sources "
        f"WHERE {where_sql} "
        f"ORDER BY truth_score DESC, applicability_score DESC, last_used_at DESC "
        f"LIMIT {limit}"
    )

    try:
        rows = await fetch_dicts(sql)
    except Exception as e:
        return f"source_query failed: {e}"

    if not rows:
        return "source_query: no matching sources"

    lines = [f"source_query: {len(rows)} result(s)"]
    for r in rows:
        lines.append(_fmt(r))
    return "\n".join(lines)


async def _source_reference_exec(
    source_id: int,
    context_topic: str = "",
    outcome: str = "unknown",
    notes: str = "",
) -> str:
    """Log a reference to a source + update its applicability score based on outcome.

    Args:
        source_id: Source row ID
        context_topic: What the source was used for
        outcome: 'useful' | 'irrelevant' | 'misleading' | 'unknown'
        notes: Additional context

    Returns:
        Status string.
    """
    from database import execute_sql, execute_insert, fetch_dicts
    from state import current_client_id

    outcome = outcome if outcome in ("useful", "irrelevant", "misleading", "unknown") else "unknown"
    session_id = current_client_id.get("") or ""

    # Confirm source exists
    try:
        existing = await fetch_dicts(
            f"SELECT id, truth_score, applicability_score FROM mymcp.samaritan_sources "
            f"WHERE id = {int(source_id)} LIMIT 1"
        )
        if not existing:
            return f"source_reference: source id={source_id} not found"
    except Exception as e:
        return f"source_reference lookup failed: {e}"

    src = existing[0]
    truth = int(src["truth_score"])
    app = int(src["applicability_score"])

    # Insert reference log entry
    try:
        ref_id = await execute_insert(
            f"INSERT INTO mymcp.samaritan_source_references "
            f"(source_id, context_topic, context_session, outcome, notes) VALUES ("
            f"{int(source_id)}, "
            f"{_sql_str(context_topic)}, "
            f"{_sql_str(session_id)}, "
            f"{_sql_str(outcome)}, "
            f"{_sql_str(notes)})"
        )
    except Exception as e:
        return f"source_reference insert failed: {e}"

    # Score updates based on outcome
    if outcome == "useful":
        new_app = min(10, app + 1)
        new_truth = truth
    elif outcome == "misleading":
        new_app = max(1, app - 1)
        new_truth = max(1, truth - 1)
    elif outcome == "irrelevant":
        new_app = max(1, app - 1)
        new_truth = truth
    else:  # unknown
        new_app = app
        new_truth = truth

    try:
        await execute_sql(
            f"UPDATE mymcp.samaritan_sources "
            f"SET usage_count = usage_count + 1, last_used_at = NOW(), "
            f"truth_score = {new_truth}, applicability_score = {new_app} "
            f"WHERE id = {int(source_id)}"
        )
    except Exception as e:
        return f"source_reference update failed: {e}"

    return (
        f"source_reference: logged ref_id={ref_id} outcome={outcome} "
        f"(truth {truth}→{new_truth}, app {app}→{new_app})"
    )


# ---------------------------------------------------------------------------
# source_recheck — re-fetch a source, detect drift, update scores
# ---------------------------------------------------------------------------


def _strip_html_to_text(html: str, max_chars: int = 4000) -> tuple[str, str]:
    """Strip HTML, return (title, body_text). Title from <title>, body from visible text."""
    if not html:
        return "", ""
    # Extract title
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = (m.group(1).strip() if m else "")[:400]
    title = re.sub(r"\s+", " ", title)
    # Strip script/style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return title, text[:max_chars]


def _compute_drift(old_summary: str, new_summary: str) -> tuple[str, float]:
    """Return (drift_label, word_overlap_ratio).

    drift_label: 'unchanged' | 'minor' | 'major'
    Word-overlap heuristic — cheap, no embeddings needed for a decision gate.
    """
    def _norm_words(s: str) -> set:
        if not s:
            return set()
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", s.lower())
        return set(tokens)

    old_words = _norm_words(old_summary)
    new_words = _norm_words(new_summary)
    if not old_words and not new_words:
        return "unchanged", 1.0
    if not old_words or not new_words:
        return "major", 0.0
    intersection = old_words & new_words
    union = old_words | new_words
    ratio = len(intersection) / len(union) if union else 1.0
    if ratio >= 0.95:
        return "unchanged", ratio
    if ratio >= 0.7:
        return "minor", ratio
    if ratio >= 0.4:
        return "moderate", ratio
    return "major", ratio


async def _source_recheck_exec(source_id: int = 0) -> str:
    """Re-fetch a source URL, detect drift, update scores.

    Flow:
      (a) re-fetch canonical_url via httpx GET
      (b) strip HTML → text, compute content_hash
      (c) if unchanged → UPDATE last_checked_at=NOW(), done
      (d) if changed → extract new title+summary, re-embed, update content_hash
      (e) drift minor → no score change. drift major → truth_score -= 2
      (f) on fetch failure (timeout, 4xx, 5xx) → status='broken', truth_score -= 3

    Args:
        source_id: Row ID in samaritan_sources

    Returns:
        Status string describing what changed.
    """
    from database import fetch_dicts, execute_sql

    try:
        src_id = int(source_id)
    except (TypeError, ValueError):
        return "source_recheck: source_id required (integer)"

    rows = await fetch_dicts(
        f"SELECT id, canonical_url, title, summary, content_hash, hash_source, truth_score, "
        f"domain_tags, authority, source_type, collection, status "
        f"FROM mymcp.samaritan_sources WHERE id = {src_id} LIMIT 1"
    )
    if not rows:
        return f"source_recheck: source_id={src_id} not found"

    row = rows[0]
    url = row.get("canonical_url") or ""
    if not url or not url.startswith(("http://", "https://")):
        return f"source_recheck: id={src_id} has no http(s) canonical_url ({url!r})"

    old_hash = row.get("content_hash") or ""
    old_summary = row.get("summary") or ""
    old_truth = int(row.get("truth_score") or 5)
    old_hash_source = row.get("hash_source") or "fetched"

    # Step (a): fetch
    try:
        import httpx
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            })
        status_code = resp.status_code
        if status_code >= 400:
            new_truth = max(1, old_truth - 3)
            await execute_sql(
                f"UPDATE mymcp.samaritan_sources "
                f"SET status='broken', truth_score={new_truth}, last_checked_at=NOW() "
                f"WHERE id={src_id}"
            )
            return f"source_recheck: id={src_id} fetch returned {status_code}, marked broken (truth {old_truth}→{new_truth})"
        content_type = (resp.headers.get("content-type") or "").lower()
        body_bytes = resp.content
    except Exception as e:
        new_truth = max(1, old_truth - 3)
        await execute_sql(
            f"UPDATE mymcp.samaritan_sources "
            f"SET status='broken', truth_score={new_truth}, last_checked_at=NOW() "
            f"WHERE id={src_id}"
        )
        return f"source_recheck: id={src_id} fetch failed ({e}), marked broken (truth {old_truth}→{new_truth})"

    # Step (b): extract + hash — detect PDF vs HTML
    is_pdf = (
        "application/pdf" in content_type
        or url.lower().split("?", 1)[0].endswith(".pdf")
        or body_bytes[:5] == b"%PDF-"
    )
    if is_pdf:
        # PDFs: use sentinel summary (no PDF extraction library available)
        # but still hash the raw bytes for drift detection on subsequent fetches.
        size_kb = len(body_bytes) // 1024
        from urllib.parse import urlparse
        basename = (urlparse(url).path.rsplit("/", 1)[-1] or "document") or "document"
        new_title = basename
        new_summary = f"[PDF] {basename} ({size_kb} KB) — {url}"
        new_hash = compute_content_hash(body_bytes.hex())
    else:
        html = body_bytes.decode(resp.encoding or "utf-8", errors="replace")
        new_title, new_text = _strip_html_to_text(html)
        new_summary = new_text[:400].strip() or new_title
        new_hash = compute_content_hash(new_text or new_title or url)

    # Step (c): unchanged?
    if old_hash and new_hash == old_hash:
        await execute_sql(
            f"UPDATE mymcp.samaritan_sources SET last_checked_at=NOW() WHERE id={src_id}"
        )
        return f"source_recheck: id={src_id} unchanged (hash match), last_checked_at updated"

    # Step (d): changed — compute drift and update
    drift_label, ratio = _compute_drift(old_summary, new_summary)

    # Build UPDATE
    updates = [
        f"content_hash={_sql_str(new_hash)}",
        f"last_checked_at=NOW()",
    ]
    if new_title:
        updates.append(f"title={_sql_str(new_title)}")
    if new_summary:
        updates.append(f"summary={_sql_str(new_summary)}")
    # Clear embedding_model so backfill/upsert will re-embed
    updates.append("embedding_model=NULL")

    # Baseline path: a 'synthesized' row's hash was computed from an LLM answer
    # (sonar/xai) or query-filtered extract (tavily), not from the actual page.
    # First recheck on such a row adopts the fetched page as the baseline —
    # no drift penalty, flip hash_source to 'fetched' for future rechecks.
    new_truth = old_truth
    baseline_established = False
    if old_hash_source == "synthesized":
        baseline_established = True
        updates.append("hash_source='fetched'")
    elif drift_label == "major":
        new_truth = max(1, old_truth - 2)
        updates.append(f"truth_score={new_truth}")

    update_sql = (
        f"UPDATE mymcp.samaritan_sources SET {', '.join(updates)} WHERE id={src_id}"
    )
    await execute_sql(update_sql)

    # Fire async re-embed with new summary
    tags_val = row.get("domain_tags")
    if tags_val is None:
        tags_str = None
    elif isinstance(tags_val, (bytes, bytearray)):
        tags_str = tags_val.decode("utf-8", errors="ignore")
    elif isinstance(tags_val, str):
        tags_str = tags_val
    else:
        tags_str = json.dumps(tags_val)

    asyncio.create_task(_upsert_source_vec(
        src_id=src_id,
        canonical_url=url,
        title=new_title or row.get("title") or "",
        summary=new_summary,
        domain_tags_json=tags_str,
        authority=row.get("authority") or "unknown",
        source_type=row.get("source_type") or "internet",
        collection=row.get("collection") or "",
    ))

    if baseline_established:
        return (
            f"source_recheck: id={src_id} baseline established from fetched page "
            f"(was synthesized, word_overlap={ratio:.2f}), truth unchanged at {old_truth}, "
            f"re-embed queued"
        )
    return (
        f"source_recheck: id={src_id} drift={drift_label} (word_overlap={ratio:.2f}), "
        f"truth {old_truth}→{new_truth}, re-embed queued"
    )


# ---------------------------------------------------------------------------
# source_verify — LLM-based semantic verification of source summaries
# ---------------------------------------------------------------------------

_VERIFY_METHODS_DRIVE    = "drive+xai+sonar"
_VERIFY_METHODS_INTERNET = "url+xai+sonar+google"

_DRIVE_VERIFY_PROMPT = """You are verifying whether a stored summary accurately describes an internally-authored Google Drive document.

DOCUMENT CONTENT (read from Google Drive — "{title}"):
{doc_content}

XAI/GROK SEARCH RESULTS (cross-checking external facts cited in the doc):
{xai_content}

SONAR/PERPLEXITY (cross-checking external facts cited in the doc):
{sonar_content}

STORED SUMMARY TO VERIFY:
{summary}

This is an internal document. The document content above IS the source of truth.
Task:
1. Check whether the stored summary accurately describes the document contents.
2. Optionally flag factual claims in the doc contradicted by the web search results.

IMPORTANT — use SEMANTIC matching, not literal string matching:
- A summary claim is supported if the CONCEPT appears ANYWHERE in the doc, even briefly.
- Do NOT require a dedicated section or heading. A single paragraph or passing reference counts.
- Do NOT flag a claim as unsupported because the doc "lacks specific details" or "has no dedicated section."
- Only flag as unsupported if the concept is completely absent or directly contradicted.

"Unverifiable" is NOT the same as "unsupported". Only return "unsupported" if you found CONTRADICTING evidence.

Respond with ONLY valid JSON (no markdown):
{{"verdict": "supported"|"partial"|"unsupported", "confidence": 0.0-1.0, "unsupported_claims": ["claim 1"], "notes": "one sentence"}}"""

_INTERNET_VERIFY_PROMPT = """You are verifying whether a stored source summary is supported by the actual source content.

SOURCE URL CONTENT (extracted from {url}):
{doc_content}

XAI/GROK SEARCH RESULTS for "{title}":
{xai_content}

SONAR/PERPLEXITY RESULTS for "{title}":
{sonar_content}

GOOGLE SEARCH RESULTS for "{title}":
{google_content}

STORED SUMMARY TO VERIFY:
{summary}

IMPORTANT: "Unverifiable" is NOT the same as "unsupported".
- If none of the above sources contain relevant content, return "partial" with confidence <= 0.3.
- Only return "unsupported" if you found CONTRADICTING evidence — not merely absent evidence.
- Only return "supported" if you found CONFIRMING evidence.

Respond with ONLY valid JSON (no markdown):
{{"verdict": "supported"|"partial"|"unsupported", "confidence": 0.0-1.0, "unsupported_claims": ["claim 1"], "notes": "one sentence"}}"""


async def _source_verify_exec(source_id: int) -> dict:
    """Run LLM-based semantic verification on a single source.

    Returns a dict with keys: verdict, confidence, unsupported_claims, notes,
    new_truth_score, methods, doc_modified_at (Drive only), error (on failure).
    Updates verified_at, verification_methods, truth_score, and doc_modified_at in DB.
    Writes assert_belief for each unsupported claim found.
    """
    import asyncio as _asyncio
    import httpx as _httpx
    import json as _json
    from database import fetch_dicts, execute_sql
    from config import MCP_DIRECT_URL

    rows = await fetch_dicts(
        f"SELECT id, title, summary, source_type, source_ref, drive_file_id, "
        f"canonical_url, truth_score, verified_at, doc_modified_at "
        f"FROM mymcp.samaritan_sources WHERE id={int(source_id)} AND status='active' LIMIT 1"
    )
    if not rows:
        return {"error": f"source_id={source_id} not found or inactive"}

    row        = rows[0]
    title      = row.get("title") or ""
    summary    = row.get("summary") or ""
    src_type   = row.get("source_type") or "internet"
    source_ref = row.get("source_ref") or ""
    canon_url  = row.get("canonical_url") or ""
    old_truth  = int(row.get("truth_score") or 5)

    if not summary:
        return {"error": f"source_id={source_id} has no summary to verify"}

    # ── Extract Drive file_id ────────────────────────────────────────────────
    drive_file_id = row.get("drive_file_id") or ""
    if not drive_file_id and src_type == "drive":
        import re as _re
        m = _re.search(r'gdrive:([A-Za-z0-9_\-]+)', source_ref)
        if m:
            drive_file_id = m.group(1)

    is_drive = src_type == "drive" and bool(drive_file_id)

    doc_modified_at = None

    async with _httpx.AsyncClient(timeout=_httpx.Timeout(connect=10, read=60, write=10, pool=10)) as http:
        if is_drive:
            # Fetch Drive content + metadata + xAI + Sonar in parallel
            meta_task  = http.post(f"{MCP_DIRECT_URL}/google_drive",
                                   json={"operation": "metadata", "file_id": drive_file_id})
            doc_task   = http.post(f"{MCP_DIRECT_URL}/google_drive",
                                   json={"operation": "read", "file_id": drive_file_id})
            xai_task   = http.post(f"{MCP_DIRECT_URL}/xai_search",
                                   json={"query": f"{title} {' '.join(summary.split()[:6])}"})
            sonar_task = http.post(f"{MCP_DIRECT_URL}/sonar_answer",
                                   json={"query": f"What does '{title}' cover? Verify: {summary[:300]}"})
            meta_resp, doc_resp, xai_resp, sonar_resp = await _asyncio.gather(
                meta_task, doc_task, xai_task, sonar_task, return_exceptions=True)

            # Parse modifiedTime from metadata
            if not isinstance(meta_resp, Exception) and meta_resp.is_success:
                import re as _re2
                mt = _re2.search(r'modifiedTime:\s*(\S+)', meta_resp.json().get("result", ""))
                if mt:
                    from datetime import datetime
                    try:
                        doc_modified_at = datetime.fromisoformat(mt.group(1).rstrip("Z")).strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass

            doc_content  = str(doc_resp.json().get("result", ""))[:4000] if not isinstance(doc_resp, Exception) and doc_resp.is_success else ""
            xai_content  = str(xai_resp.json().get("result", ""))[:2000] if not isinstance(xai_resp, Exception) and xai_resp.is_success else ""
            sonar_content = str(sonar_resp.json().get("result", ""))[:2000] if not isinstance(sonar_resp, Exception) and sonar_resp.is_success else ""
            google_content = ""
            methods = _VERIFY_METHODS_DRIVE

            prompt = _DRIVE_VERIFY_PROMPT.format(
                title=title,
                doc_content=doc_content if len(doc_content) > 100 else "(document read failed)",
                xai_content=xai_content or "(no results)",
                sonar_content=sonar_content or "(no results)",
                summary=summary,
            )
        else:
            # Internet path: url_extract + xAI + Sonar + Google in parallel
            url_task    = http.post(f"{MCP_DIRECT_URL}/url_extract_tavily",
                                    json={"url": canon_url or source_ref, "query": title})
            xai_task    = http.post(f"{MCP_DIRECT_URL}/xai_search",
                                    json={"query": f"{title} {' '.join(summary.split()[:6])}"})
            sonar_task  = http.post(f"{MCP_DIRECT_URL}/sonar_answer",
                                    json={"query": f"Verify: {summary[:300]}"})
            google_task = http.post(f"{MCP_DIRECT_URL}/google_search",
                                    json={"query": f"{title} site facts"})
            url_resp, xai_resp, sonar_resp, google_resp = await _asyncio.gather(
                url_task, xai_task, sonar_task, google_task, return_exceptions=True)

            doc_content    = str(url_resp.json().get("result", ""))[:3000] if not isinstance(url_resp, Exception) and url_resp.is_success else ""
            xai_content    = str(xai_resp.json().get("result", ""))[:2000] if not isinstance(xai_resp, Exception) and xai_resp.is_success else ""
            sonar_content  = str(sonar_resp.json().get("result", ""))[:2000] if not isinstance(sonar_resp, Exception) and sonar_resp.is_success else ""
            google_content = str(google_resp.json().get("result", ""))[:2000] if not isinstance(google_resp, Exception) and google_resp.is_success else ""
            methods = _VERIFY_METHODS_INTERNET

            prompt = _INTERNET_VERIFY_PROMPT.format(
                title=title, url=canon_url or source_ref,
                doc_content=doc_content or "(extraction failed)",
                xai_content=xai_content or "(no results)",
                sonar_content=sonar_content or "(no results)",
                google_content=google_content or "(no results)",
                summary=summary,
            )

        # LLM assessment
        llm_resp = await http.post(f"{MCP_DIRECT_URL}/llm_call",
                                   json={"model": "reason-gemini", "prompt": prompt, "mode": "text"})

    assessment = {"verdict": "partial", "confidence": 0.3, "unsupported_claims": [], "notes": "LLM unavailable"}
    if not isinstance(llm_resp, Exception) and llm_resp.is_success:
        try:
            raw = llm_resp.json().get("result", "").strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            assessment = _json.loads(raw)
        except Exception as e:
            log.warning("source_verify: LLM parse failed for id=%s: %s", source_id, e)

    verdict    = assessment.get("verdict", "partial")
    confidence = float(assessment.get("confidence", 0.5))
    unsupported = assessment.get("unsupported_claims", [])

    score_map     = {"supported": 8, "partial": 4, "unsupported": 1}
    new_raw_score = score_map.get(verdict, 4)
    has_evidence  = any([doc_content, xai_content, sonar_content, google_content])
    can_downgrade = has_evidence and confidence >= 0.6
    can_upgrade   = verdict == "supported" and confidence >= 0.6
    should_update = can_upgrade or (can_downgrade and verdict != "supported")

    # ── Write contradiction beliefs for unsupported claims ────────────────────
    if unsupported:
        async with _httpx.AsyncClient(timeout=_httpx.Timeout(connect=5, read=15, write=5, pool=5)) as belief_http:
            for claim in unsupported:
                slug = re.sub(r'[^a-z0-9]+', '-', claim.lower())[:40].strip('-')
                try:
                    await belief_http.post(f"{MCP_DIRECT_URL}/assert_belief",
                                json={"topic": f"source-contradiction-{slug}",
                                      "content": f"Source id={source_id} ('{title}') has an unverified claim: '{claim}'. "
                                                 f"Verdict={verdict} confidence={confidence:.2f}. "
                                                 f"Methods: {methods}. Notes: {assessment.get('notes', '')}",
                                      "confidence": 8})
                except Exception as e:
                    log.warning("source_verify: assert_belief failed for claim %r: %s", claim, e)

    # ── Update DB ────────────────────────────────────────────────────────────
    set_clauses = [
        "verified_at = NOW()",
        f"verification_methods = {_sql_str(methods)}",
    ]
    if should_update:
        set_clauses.append(f"truth_score = {new_raw_score}")
    if doc_modified_at:
        set_clauses.append(f"doc_modified_at = {_sql_str(doc_modified_at)}")
    await execute_sql(
        f"UPDATE mymcp.samaritan_sources SET {', '.join(set_clauses)} WHERE id = {int(source_id)}"
    )

    return {
        "verdict":           verdict,
        "confidence":        confidence,
        "unsupported_claims": unsupported,
        "notes":             assessment.get("notes", ""),
        "new_truth_score":   new_raw_score if should_update else old_truth,
        "score_updated":     should_update,
        "methods":           methods,
        "doc_modified_at":   doc_modified_at,
    }


async def _source_verify_batch_exec(max_verifications: int = 30) -> str:
    """Off-peak batch verification of sources with stale or missing verification.

    Candidate selection:
    - Drive sources: verified_at IS NULL OR doc_modified_at > verified_at
      (re-verify only when doc has changed since last verification)
    - Internet sources: verified_at IS NULL OR verified_at < NOW() - INTERVAL 30 DAY

    For Drive sources, fetches file metadata first (lightweight) to check
    modifiedTime before doing full content verification.

    Designed to run nightly (3am) via prospective memory trigger.
    Contradiction beliefs are written automatically — no automatic doc edits.
    """
    import asyncio as _asyncio
    import httpx as _httpx
    from database import fetch_dicts, execute_sql

    try:
        limit = max(1, min(int(max_verifications), 100))
    except (TypeError, ValueError):
        limit = 30

    from config import MCP_DIRECT_URL

    # ── Step 1: Collect Drive candidates (check modifiedTime before verifying) ─
    drive_candidates_sql = (
        "SELECT id, title, source_ref, drive_file_id, verified_at, doc_modified_at "
        "FROM mymcp.samaritan_sources "
        "WHERE status='active' AND source_type='drive' "
        "AND (verified_at IS NULL OR doc_modified_at IS NULL OR doc_modified_at > verified_at) "
        f"ORDER BY verified_at ASC LIMIT {limit}"
    )
    drive_rows = await fetch_dicts(drive_candidates_sql)

    # For Drive rows without doc_modified_at, fetch metadata to populate it first
    import re as _re2
    from datetime import datetime as _dt
    drive_to_verify = []
    async with _httpx.AsyncClient(timeout=_httpx.Timeout(connect=10, read=30, write=10, pool=10)) as http:
        meta_tasks = []
        for row in drive_rows:
            fid = row.get("drive_file_id") or ""
            if not fid:
                m = _re2.search(r'gdrive:([A-Za-z0-9_\-]+)', row.get("source_ref") or "")
                if m:
                    fid = m.group(1)
            if fid:
                meta_tasks.append((row["id"], fid,
                    http.post(f"{MCP_DIRECT_URL}/google_drive",
                              json={"operation": "metadata", "file_id": fid})))

        for src_id, fid, coro in meta_tasks:
            try:
                resp = await coro
                if resp.is_success:
                    mt = _re2.search(r'modifiedTime:\s*(\S+)', resp.json().get("result", ""))
                    if mt:
                        try:
                            mod_str = _dt.fromisoformat(mt.group(1).rstrip("Z")).strftime("%Y-%m-%d %H:%M:%S")
                            await execute_sql(
                                f"UPDATE mymcp.samaritan_sources SET doc_modified_at={_sql_str(mod_str)} "
                                f"WHERE id={src_id}"
                            )
                            # Check if modified after verified_at
                            rows_check = await fetch_dicts(
                                f"SELECT verified_at FROM mymcp.samaritan_sources WHERE id={src_id} LIMIT 1"
                            )
                            if rows_check:
                                v_at = rows_check[0].get("verified_at")
                                if v_at is None or mod_str > str(v_at):
                                    drive_to_verify.append(src_id)
                        except ValueError:
                            drive_to_verify.append(src_id)  # can't parse date, verify anyway
            except Exception as e:
                log.warning("source_verify_batch: metadata fetch failed for id=%s: %s", src_id, e)

    # ── Step 2: Collect internet candidates ──────────────────────────────────
    internet_candidates_sql = (
        "SELECT id FROM mymcp.samaritan_sources "
        "WHERE status='active' AND source_type='internet' "
        "AND (verified_at IS NULL OR verified_at < NOW() - INTERVAL 30 DAY) "
        f"ORDER BY verified_at ASC LIMIT {limit}"
    )
    internet_rows = await fetch_dicts(internet_candidates_sql)
    internet_ids  = [r["id"] for r in internet_rows]

    all_ids = drive_to_verify + internet_ids
    if not all_ids:
        return "source_verify_batch: no candidates needing verification"

    # Cap total at limit
    all_ids = all_ids[:limit]

    # ── Step 3: Verify each candidate sequentially (avoid rate-limiting) ──────
    results = {"supported": 0, "partial": 0, "unsupported": 0, "error": 0, "contradictions": 0}
    for src_id in all_ids:
        try:
            r = await _source_verify_exec(source_id=src_id)
            if "error" in r:
                results["error"] += 1
            else:
                results[r.get("verdict", "partial")] += 1
                results["contradictions"] += len(r.get("unsupported_claims", []))
        except Exception as e:
            log.warning("source_verify_batch: verify failed for id=%s: %s", src_id, e)
            results["error"] += 1

    total = len(all_ids)
    return (
        f"source_verify_batch: {total} sources verified "
        f"(supported={results['supported']}, partial={results['partial']}, "
        f"unsupported={results['unsupported']}, errors={results['error']}, "
        f"contradiction beliefs written={results['contradictions']})"
    )


# ---------------------------------------------------------------------------
# source_curate_scan — daily scheduled maintenance of the sources index
# ---------------------------------------------------------------------------


async def _source_curate_scan_exec(max_rechecks: int = 20) -> str:
    """Scheduled curation pass over samaritan_sources.

    (a) Find sources where usage_count > 0 AND
        last_checked_at + half_life_days < NOW() → recheck top-K.
    (b) Find sources where applicability_score <= 2 AND usage_count >= 3
        → archive (status='archived').

    Args:
        max_rechecks: Max number of source_recheck calls this pass (rate limit).

    Returns:
        Summary string with counts of rechecks, archives, and drift outcomes.
    """
    from database import fetch_dicts, execute_sql

    try:
        limit = max(1, min(int(max_rechecks), 100))
    except (TypeError, ValueError):
        limit = 20

    # Step (a): find stale sources needing recheck
    stale_sql = (
        "SELECT id, canonical_url, last_checked_at, half_life_days, usage_count "
        "FROM mymcp.samaritan_sources "
        "WHERE status = 'active' AND usage_count > 0 "
        "AND last_checked_at IS NOT NULL "
        "AND DATE_ADD(last_checked_at, INTERVAL half_life_days DAY) < NOW() "
        "ORDER BY usage_count DESC, last_checked_at ASC "
        f"LIMIT {limit}"
    )
    stale_rows = await fetch_dicts(stale_sql)

    recheck_results = {"unchanged": 0, "minor": 0, "major": 0, "broken": 0, "error": 0}
    for r in stale_rows:
        src_id = int(r["id"])
        try:
            result = await _source_recheck_exec(source_id=src_id)
            if "unchanged" in result:
                recheck_results["unchanged"] += 1
            elif "drift=major" in result:
                recheck_results["major"] += 1
            elif "drift=minor" in result or "drift=moderate" in result:
                recheck_results["minor"] += 1
            elif "broken" in result:
                recheck_results["broken"] += 1
            else:
                recheck_results["error"] += 1
        except Exception:
            recheck_results["error"] += 1

    # Step (b): archive persistently-irrelevant sources
    archive_sql = (
        "UPDATE mymcp.samaritan_sources "
        "SET status = 'archived' "
        "WHERE status = 'active' AND applicability_score <= 2 AND usage_count >= 3"
    )
    archive_rows_before = await fetch_dicts(
        "SELECT COUNT(*) AS n FROM mymcp.samaritan_sources "
        "WHERE status = 'active' AND applicability_score <= 2 AND usage_count >= 3"
    )
    n_to_archive = int(archive_rows_before[0]["n"]) if archive_rows_before else 0
    if n_to_archive > 0:
        await execute_sql(archive_sql)

    return (
        f"source_curate_scan: {len(stale_rows)} rechecks "
        f"(unchanged={recheck_results['unchanged']}, "
        f"minor={recheck_results['minor']}, major={recheck_results['major']}, "
        f"broken={recheck_results['broken']}, error={recheck_results['error']}), "
        f"{n_to_archive} sources archived"
    )


# ---------------------------------------------------------------------------
# Self-tests — run as: python3 sources.py
# ---------------------------------------------------------------------------

def _test_canonicalize_url():
    cases = [
        # (input, expected, description)
        ("https://example.com/page?utm_source=newsletter&id=42",
         "https://example.com/page?id=42",
         "strip utm_ tracking param"),
        ("HTTPS://Example.COM:443/page/",
         "https://example.com/page",
         "lowercase scheme+host, strip :443, strip trailing slash"),
        ("http://example.com/",
         "http://example.com/",
         "root path keeps slash"),
        ("https://example.com/a?b=2&a=1&utm_medium=email",
         "https://example.com/a?a=1&b=2",
         "sort params, strip tracking"),
        ("https://example.com/page#section-2",
         "https://example.com/page",
         "strip fragment"),
        ("  https://example.com/p  ",
         "https://example.com/p",
         "strip whitespace"),
        ("example.com/path",
         "http://example.com/path",
         "add scheme if missing"),
        ("",
         "",
         "empty input"),
        ("https://example.com:8080/api",
         "https://example.com:8080/api",
         "non-default port preserved"),
        ("https://example.com/?fbclid=ABC&gclid=DEF&q=search",
         "https://example.com/?q=search",
         "strip fbclid and gclid"),
    ]
    passed = failed = 0
    for url, expected, desc in cases:
        # Disable shortener resolution for deterministic tests
        got = canonicalize_url(url, resolve_shorteners=False)
        if got == expected:
            passed += 1
        else:
            failed += 1
            print(f"FAIL ({desc}): {url!r} -> {got!r}, expected {expected!r}")
    print(f"canonicalize_url: {passed}/{passed+failed} passed")
    return failed == 0


def _test_compute_content_hash():
    cases = [
        ("hello world", "hello world"),
        ("hello  world", "hello world"),  # collapsed whitespace
        ("  hello\nworld  ", "hello world"),  # normalized
    ]
    hashes = [compute_content_hash(text) for text, _ in cases]
    # All three should produce the same hash due to normalization
    if len(set(hashes)) == 1 and hashes[0]:
        print("compute_content_hash: 1/1 passed (whitespace normalization works)")
        return True
    else:
        print(f"FAIL compute_content_hash: got {hashes}")
        return False


# ---------------------------------------------------------------------------
# Auto-archive helper — called by search tool plugins after successful calls
# ---------------------------------------------------------------------------


async def archive_from_search(
    url: str,
    title: str,
    summary: str,
    origin_tool: str,
    domain_tags: str = "",
    collection: str = "auto-archived",
) -> None:
    """Fire-and-forget archive of a URL cited by a search tool.

    Wraps _source_record_exec so search tools can call
    `asyncio.create_task(archive_from_search(...))` without caring about
    return values or errors. Never raises.

    Args:
        url: Canonical URL of the cited source
        title: Page title if known, else synthesized from query
        summary: First ~250 chars of the answer/content (drives embedding)
        origin_tool: Name of the search tool ('sonar_answer', 'xai_search', etc.)
        domain_tags: Additional comma-separated tags (origin_tool auto-added)
        collection: Collection grouping (default: 'auto-archived')
    """
    try:
        if not url:
            return
        tags = f"{domain_tags},{origin_tool}" if domain_tags else origin_tool
        await _source_record_exec(
            source_type="internet",
            source_ref=url,
            title=title or "",
            summary=summary or "",
            content="",
            domain_tags=tags,
            collection=collection,
            authority_override="",
            hash_source="synthesized",
        )
    except Exception as e:
        log.warning(f"archive_from_search failed for {url!r}: {e}")


if __name__ == "__main__":
    import sys
    ok1 = _test_canonicalize_url()
    ok2 = _test_compute_content_hash()
    sys.exit(0 if (ok1 and ok2) else 1)
