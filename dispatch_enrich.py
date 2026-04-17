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
_MOVEMENT_THRESHOLD_M = 200 # min distance (meters) between fixes to count as movement
_MOVEMENT_WINDOW_MIN = 15   # look back N minutes for movement detection
_MOVEMENT_MIN_FIXES = 2     # need at least N fixes in window to assess movement
_TRIGGER_SCAN_LINES = 5     # how many head lines of belief content to scan for TRIGGER|
_ROUTINE_CACHE_TTL_S = 60   # compiled routine-trigger cache TTL

# Routine triggers are NOT hardcoded here. Each routine-* belief carries its own
# activation pattern as a structured line `TRIGGER|regex=<pattern>` within the
# first few lines of its content. dispatch_enrich loads routine beliefs, parses
# their TRIGGER lines, compiles the regexes, and matches user text per-belief.
# Adding a new routine = asserting a new belief with a TRIGGER line. No code edit.
# Beliefs with no usable TRIGGER declaration (missing, malformed, or non-compiling)
# are skipped entirely — they never load on dispatch.

# Per-process cache: list of (topic, content, compiled_trigger_or_None) tuples +
# unix epoch of last refresh. Small cache avoids recompiling on every dispatch
# while still picking up belief edits within _ROUTINE_CACHE_TTL_S seconds.
_routine_cache: list[tuple[str, str, "re.Pattern | None"]] = []
_routine_cache_ts: float = 0.0


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


# ── Movement detection ───────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Equirectangular distance in meters — accurate enough for < 50 km."""
    import math
    dx = (lat2 - lat1) * 111320
    dy = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dx * dx + dy * dy)


async def detect_movement(lat: float, lon: float) -> dict:
    """Classify location state from recent GPS history.

    Returns dict with:
        known_place: str | None  — name of nearest known place (within threshold)
        moving: bool             — significant position change in recent window
        speed_kmh: float | None  — estimated speed if moving
        location_state: str      — "stationary-known" | "stationary-unknown" | "moving"
    """
    result = {
        "known_place": None,
        "moving": False,
        "speed_kmh": None,
        "location_state": "stationary-unknown",
    }

    try:
        # Check if current position is near a known place
        result["known_place"] = await resolve_gps_location(lat, lon)

        # Get recent GPS fixes for movement detection
        rows = await _dicts(
            f"SELECT latitude, longitude, created_at "
            f"FROM mymcp.samaritan_location "
            f"WHERE created_at >= DATE_SUB(NOW(), INTERVAL {_MOVEMENT_WINDOW_MIN} MINUTE) "
            f"ORDER BY created_at DESC LIMIT 20"
        )

        if len(rows) < _MOVEMENT_MIN_FIXES:
            # Not enough data — classify by known/unknown only
            result["location_state"] = (
                "stationary-known" if result["known_place"] else "stationary-unknown"
            )
            return result

        # Compute max displacement from current position across recent fixes
        max_dist = 0.0
        oldest_row = rows[-1]
        for row in rows:
            d = _haversine_m(lat, lon, float(row["latitude"]), float(row["longitude"]))
            if d > max_dist:
                max_dist = d

        if max_dist >= _MOVEMENT_THRESHOLD_M:
            result["moving"] = True
            result["location_state"] = "moving"
            # Estimate speed from oldest fix in window to current position
            oldest_lat = float(oldest_row["latitude"])
            oldest_lon = float(oldest_row["longitude"])
            total_dist = _haversine_m(lat, lon, oldest_lat, oldest_lon)
            oldest_time = oldest_row["created_at"]
            if isinstance(oldest_time, str):
                oldest_time = datetime.fromisoformat(oldest_time)
            elapsed_h = (datetime.now() - oldest_time).total_seconds() / 3600
            if elapsed_h > 0:
                result["speed_kmh"] = round(total_dist / 1000 / elapsed_h, 1)
        else:
            result["location_state"] = (
                "stationary-known" if result["known_place"] else "stationary-unknown"
            )

    except Exception as e:
        log.debug(f"detect_movement: {e}")

    return result


# ── Always-on context ─────────────────────────────────────────────────────────

def _routine_relevance(text: str, topic: str, content: str) -> int:
    """Keyword overlap score — higher = more relevant to current text."""
    text_words = set(re.findall(r'\w+', text.lower()))
    routine_words = set(re.findall(r'\w+', (topic + " " + content).lower()))
    return len(text_words & routine_words)


def _extract_trigger_regex(content: str) -> str | None:
    """Scan first _TRIGGER_SCAN_LINES lines for `TRIGGER|regex=<pattern>`.

    Returns the raw regex string (without the `regex=` prefix) or None if no
    TRIGGER line is present. The TRIGGER line does not have to be the first
    line — a belief may have preamble like `verified:` or `direct:` markers
    before it, which we skip over.
    """
    for line in content.splitlines()[:_TRIGGER_SCAN_LINES]:
        line = line.strip()
        if not line.startswith("TRIGGER|"):
            continue
        idx = line.find("regex=")
        if idx == -1:
            continue
        pattern = line[idx + len("regex="):]
        if pattern:
            return pattern
    return None


async def _refresh_routine_cache() -> list[tuple[str, str, "re.Pattern | None"]]:
    """Query all active routine-* beliefs, parse+compile their TRIGGER lines,
    and cache the result for _ROUTINE_CACHE_TTL_S seconds. Returns the cached list."""
    global _routine_cache, _routine_cache_ts
    now = _time.monotonic()
    if _routine_cache and (now - _routine_cache_ts) < _ROUTINE_CACHE_TTL_S:
        return _routine_cache
    try:
        rows = await _dicts(
            "SELECT topic, content FROM mymcp.samaritan_beliefs "
            "WHERE status='active' AND topic LIKE 'routine-%' AND confidence >= 7 "
            "ORDER BY confidence DESC, topic"
        )
        built: list[tuple[str, str, "re.Pattern | None"]] = []
        for r in rows:
            topic = r["topic"]
            content = r["content"] or ""
            pattern_str = _extract_trigger_regex(content)
            compiled: "re.Pattern | None" = None
            if pattern_str:
                try:
                    compiled = re.compile(pattern_str, re.IGNORECASE)
                except re.error as e:
                    log.warning(f"routine trigger compile failed for {topic!r}: {e}")
                    compiled = None
            built.append((topic, content, compiled))
        _routine_cache = built
        _routine_cache_ts = now
        log.info(f"dispatch_enrich: routine cache refreshed ({len(built)} beliefs, "
                 f"{sum(1 for _, _, c in built if c)} with trigger)")
        return built
    except Exception as e:
        log.debug(f"_refresh_routine_cache: {e}")
        return _routine_cache  # stale is better than empty


async def _load_routine_beliefs(text: str) -> list[tuple[str, str]]:
    """Return routine beliefs whose declared TRIGGER regex matches `text`.

    Only beliefs with a successfully parsed and compiled TRIGGER|regex=... declaration
    are eligible. Beliefs with no usable trigger declaration are skipped.
    """
    cache = await _refresh_routine_cache()
    if not cache:
        return []
    out: list[tuple[str, str]] = []
    for topic, content, compiled in cache:
        if compiled is None:
            # No TRIGGER line — skip; routine beliefs must declare explicit triggers
            continue
        if compiled.search(text):
            out.append((topic, content))
    return out


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
    movement_task: asyncio.Task | None = None
    if lat is not None and lon is not None:
        gps_task = asyncio.create_task(resolve_gps_location(lat, lon))
        movement_task = asyncio.create_task(detect_movement(lat, lon))
        log_gps_async(lat, lon, acc, "dispatch")  # fire-and-forget, not awaited

    # Routine beliefs self-select via per-belief TRIGGER regex (parsed from content).
    # No hardcoded gate — the loader returns only matching routines, or [] when
    # nothing matches, which is cheap.
    routines_task = asyncio.create_task(_load_routine_beliefs(text))
    rules_task = asyncio.create_task(_run_dispatch_rules(text))

    all_tasks = [t for t in [gps_task, movement_task, routines_task, rules_task] if t is not None]

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

    movement: dict | None = None
    if movement_task and movement_task in done and not movement_task.cancelled():
        try:
            movement = movement_task.result()
            # movement's known_place is authoritative (same query), sync location_name
            if movement and movement.get("known_place"):
                location_name = movement["known_place"]
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
    # Location state: stationary-known, stationary-unknown, or moving (+ speed)
    state_str = ""
    if movement:
        loc_state = movement.get("location_state", "")
        if loc_state == "moving":
            speed = movement.get("speed_kmh")
            # Convert to mph for US customary
            mph = round(speed * 0.621371, 0) if speed else None
            speed_part = f" ~{int(mph)}mph" if mph else ""
            state_str = f" | moving{speed_part}"
        elif loc_state == "stationary-unknown":
            state_str = " | new location"
        # stationary-known: no extra tag needed — "near: <place>" already signals it
    header = f"[context: {time_str}{near_str}{state_str}]"

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
