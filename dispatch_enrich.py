"""
Dispatch-side enrichment for Claude Code prompt submissions.

Prepends a [context] block to every Claude Code dispatch containing:
  1. Time + GPS-resolved location name (always)
  2. Active routine beliefs — topic LIKE 'routine-%', confidence >= 7 (always)
  3. Pattern-triggered SQL rules from mymcp.samaritan_dispatch_rules table

All work runs within a 200ms deadline; partial results degrade gracefully.
Enrichment failure never blocks the dispatch.
"""

import asyncio
import logging
import re
import time as _time
from datetime import datetime

log = logging.getLogger("dispatch_enrich")

_DEADLINE_MS = 200          # total budget for enrichment
_GPS_THRESHOLD_M = 300      # max distance (meters) to claim a named location
_BELIEF_TRUNC = 200         # max chars per belief in context block
_RULE_RESULT_LINES = 3      # max result lines per matched rule
_ROUTINE_MAX = 3            # max routines to inject even when triggered

# Routines are only injected when the user text touches a schedule-relevant topic.
# Keeps technical/coding conversations clean of irrelevant lifestyle context.
_ROUTINE_TRIGGER_RE = re.compile(
    r'\b(lee|gym|pickup|pick\s*up|drop\s*off|dropoff|chinatown|nevada|'
    r'trader\s*joe|target|tonight|this\s*evening|this\s*morning|'
    r'appointment|grocery|groceries|power\s*wagon|challenger|'
    r'soho|soma|pick\s*her\s*up|pick\s*him\s*up|walk\s*to|'
    r'what\s*time.*(?:leave|go|pick|walk)|when.*(?:leave|pick|walk))\b',
    re.IGNORECASE,
)


async def _sql(query: str) -> str:
    from database import execute_sql
    return await execute_sql(query)


async def _dicts(query: str) -> list[dict]:
    from database import fetch_dicts
    return await fetch_dicts(query)


def _parse_rows(raw: str, ncols: int = 2) -> list[list[str]]:
    """Parse execute_sql pipe-formatted table into list of row value lists."""
    lines = raw.strip().split("\n")
    if len(lines) < 3 or "(no rows)" in raw:
        return []
    rows = []
    for line in lines[2:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= ncols:
            rows.append(parts[:ncols])
    return rows


# ── GPS resolution ────────────────────────────────────────────────────────────

async def resolve_gps_location(lat: float, lon: float) -> str | None:
    """Return nearest place_name from mymcp.coordinates within threshold, or None."""
    try:
        rows = await _dicts(
            f"SELECT place_name, "
            f"SQRT(POW(({lat} - latitude) * 111320, 2) + "
            f"POW(({lon} - longitude) * 111320 * COS(RADIANS({lat})), 2)) AS dist_m "
            f"FROM mymcp.coordinates ORDER BY dist_m LIMIT 1"
        )
        if rows:
            name = rows[0].get("place_name")
            dist = rows[0].get("dist_m")
            try:
                if name and float(dist) <= _GPS_THRESHOLD_M:
                    return name
            except (ValueError, TypeError):
                pass
    except Exception as e:
        log.debug(f"resolve_gps_location: {e}")
    return None


def log_gps_async(lat: float, lon: float, accuracy_m: float | None, session_id: str) -> None:
    """Fire-and-forget: log GPS fix to mymcp.samaritan_location."""
    async def _insert() -> None:
        try:
            acc = str(accuracy_m) if accuracy_m is not None else "NULL"
            sid = (session_id or "dispatch").replace("'", "''")[:100]
            await _sql(
                f"INSERT INTO mymcp.samaritan_location "
                f"(latitude, longitude, accuracy_m, session_id) "
                f"VALUES ({lat}, {lon}, {acc}, '{sid}')"
            )
        except Exception as e:
            log.debug(f"log_gps_async insert failed: {e}")
    asyncio.ensure_future(_insert())


# ── Always-on context ─────────────────────────────────────────────────────────

def _routine_relevance(text: str, topic: str, content: str) -> int:
    """Keyword overlap score — higher = more relevant to current text."""
    text_words = set(re.findall(r'\w+', text.lower()))
    routine_words = set(re.findall(r'\w+', (topic + " " + content).lower()))
    return len(text_words & routine_words)


async def _load_routine_beliefs() -> list[tuple[str, str]]:
    """Load all active routine beliefs (topic LIKE 'routine-%')."""
    try:
        rows = await _dicts(
            "SELECT topic, content FROM mymcp.samaritan_beliefs "
            "WHERE status='active' AND topic LIKE 'routine-%' AND confidence >= 7 "
            "ORDER BY confidence DESC, topic"
        )
        return [(r["topic"], r["content"]) for r in rows]
    except Exception as e:
        log.debug(f"_load_routine_beliefs: {e}")
        return []


# ── Pattern-triggered rules ───────────────────────────────────────────────────

async def _run_dispatch_rules(text: str) -> list[tuple[str, str]]:
    """Match text against samaritan_dispatch_rules; return (label, compact_result) pairs."""
    results: list[tuple[str, str]] = []
    try:
        rule_rows = await _dicts(
            "SELECT pattern, query_sql, label FROM mymcp.samaritan_dispatch_rules "
            "WHERE enabled=1 ORDER BY priority DESC"
        )
        if not rule_rows:
            return []

        triggered: list[tuple[str, str]] = []  # (label, sql)
        for row in rule_rows:
            pattern = row.get("pattern", "")
            query_sql = row.get("query_sql", "")
            label = row.get("label", "")
            if not pattern or not query_sql:
                continue
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    triggered.append((label, query_sql))
                    log.info(f"dispatch_enrich: rule '{label}' triggered")
            except re.error:
                log.warning(f"dispatch_enrich: bad pattern in rule '{label}': {pattern!r}")

        if not triggered:
            log.info(f"dispatch_enrich: no rules triggered for text={text[:60]!r}")
            return []

        tasks = [asyncio.create_task(_dicts(q)) for _, q in triggered]
        done, pending = await asyncio.wait(tasks, timeout=0.12)
        for t in pending:
            t.cancel()

        for i, (label, _) in enumerate(triggered):
            task = tasks[i]
            if task not in done or task.cancelled():
                log.info(f"dispatch_enrich: rule '{label}' SQL hit deadline or cancelled")
                continue
            try:
                rows = task.result()
                if not rows:
                    continue
                # Compact: join first N rows as "col1: col2" strings
                parts = []
                for r in rows[:_RULE_RESULT_LINES]:
                    vals = [str(v) for v in r.values() if v is not None]
                    parts.append(": ".join(vals))
                compact = " | ".join(parts)
                if compact:
                    results.append((label, compact))
            except Exception as e:
                log.debug(f"dispatch_enrich: rule '{label}' result error: {e}")
    except Exception as e:
        log.warning(f"dispatch_enrich: _run_dispatch_rules error: {e}")
    return results


# ── Main entry point ──────────────────────────────────────────────────────────

async def build_context_prefix(
    text: str,
    location: dict | None = None,
) -> tuple[str | None, str | None]:
    """
    Build a context prefix string and resolve GPS to a named location.

    Args:
        text:     The raw user message text (before prefix formatting).
        location: Dict with latitude, longitude, accuracy_m (from voice payload).

    Returns:
        (context_prefix, location_name)
        context_prefix — string to prepend to the dispatch prompt, or None
        location_name  — human-readable place name for GPS coords, or None
    """
    t0 = _time.monotonic()

    lat = lon = acc = None
    if location:
        lat = location.get("latitude")
        lon = location.get("longitude")
        acc = location.get("accuracy_m")

    # Fire all tasks in parallel
    gps_task: asyncio.Task | None = None
    if lat is not None and lon is not None:
        gps_task = asyncio.create_task(resolve_gps_location(lat, lon))
        log_gps_async(lat, lon, acc, "dispatch")  # fire-and-forget, not awaited

    # Only load routines when text is schedule/person/location relevant
    routines_task = (
        asyncio.create_task(_load_routine_beliefs())
        if _ROUTINE_TRIGGER_RE.search(text) else None
    )
    rules_task = asyncio.create_task(_run_dispatch_rules(text))

    all_tasks = [t for t in [gps_task, routines_task, rules_task] if t is not None]

    remaining = max(0.05, (_DEADLINE_MS / 1000) - (_time.monotonic() - t0))
    done, pending = await asyncio.wait(all_tasks, timeout=remaining)
    for t in pending:
        t.cancel()

    if pending:
        log.debug(f"dispatch_enrich: {len(pending)} tasks hit deadline "
                  f"({(_time.monotonic()-t0)*1000:.0f}ms elapsed)")

    # Collect results
    location_name: str | None = None
    if gps_task and gps_task in done and not gps_task.cancelled():
        try:
            location_name = gps_task.result()
        except Exception:
            pass

    routines: list[tuple[str, str]] = []
    if routines_task and routines_task in done and not routines_task.cancelled():
        try:
            all_routines = routines_task.result()
            # Sort by keyword overlap with user text, cap at max
            all_routines.sort(
                key=lambda r: _routine_relevance(text, r[0], r[1]), reverse=True
            )
            routines = all_routines[:_ROUTINE_MAX]
        except Exception:
            pass

    rule_results: list[tuple[str, str]] = []
    if rules_task in done and not rules_task.cancelled():
        try:
            rule_results = rules_task.result()
        except Exception:
            pass

    # Assemble context prefix
    now = datetime.now()
    time_str = f"{now.strftime('%A')} {now.strftime('%H:%M')}"
    near_str = f" | near: {location_name}" if location_name else ""
    header = f"[context: {time_str}{near_str}]"

    parts = [header]

    if routines:
        belief_parts = []
        for topic, content in routines:
            trunc = (content[:_BELIEF_TRUNC].rsplit(" ", 1)[0] + "…"
                     if len(content) > _BELIEF_TRUNC else content)
            belief_parts.append(f"{topic}: {trunc}")
        parts.append("Routines — " + " | ".join(belief_parts))

    for label, compact in rule_results:
        parts.append(f"{label} — {compact}")

    # If only the header and no data, still return it (time+location always useful)
    prefix = " ".join(parts)
    log.debug(
        f"dispatch_enrich: {len(prefix)}ch prefix, "
        f"routines={len(routines)}, rules={len(rule_results)}, "
        f"{(_time.monotonic()-t0)*1000:.0f}ms"
    )
    return prefix, location_name
