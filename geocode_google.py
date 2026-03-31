"""
Google Maps Geocoding with MySQL coordinate cache.

Converts place names to GPS coordinates. Results are cached permanently
in the `coordinates` table (mymcp database) to avoid repeated API calls.

Fuzzy matching: normalizes place names (strip punctuation, collapse whitespace)
and falls back to LIKE-based search to avoid duplicate entries for the same place.
"""

import os
import re
import asyncio
import googlemaps
import mysql.connector

from config import log

_gmaps_client = None

# DB credentials (same as rest of system)
_DB_HOST = os.environ.get("DB_HOST", "localhost")
_DB_USER = os.environ.get("MYSQL_USER", "")
_DB_PASS = os.environ.get("MYSQL_PASS", "")
_DB_NAME = "mymcp"


def _normalize(name: str) -> str:
    """Normalize a place name for consistent cache keys.

    Strips punctuation, collapses whitespace, lowercases.
    'The Metreon, San Francisco' -> 'the metreon san francisco'
    'The Metreon San Francisco'  -> 'the metreon san francisco'
    """
    s = name.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)  # replace punctuation with space
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
    return s


def _get_gmaps():
    """Return cached Google Maps client."""
    global _gmaps_client
    if _gmaps_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set (used for Google Maps Geocoding)")
        _gmaps_client = googlemaps.Client(key=api_key)
    return _gmaps_client


def _get_db():
    """Return a fresh MySQL connection to mymcp."""
    return mysql.connector.connect(
        host=_DB_HOST, user=_DB_USER, password=_DB_PASS, database=_DB_NAME
    )


def _cache_lookup(place_name: str) -> dict | None:
    """Check if coordinates are already cached.

    1. Exact match on normalized name
    2. Fuzzy fallback: LIKE search using key words from the query
    """
    norm = _normalize(place_name)
    conn = _get_db()
    try:
        cur = conn.cursor(dictionary=True)

        # 1. Exact match on normalized name
        cur.execute(
            "SELECT latitude, longitude, formatted_address, place_id "
            "FROM coordinates WHERE place_name = %s",
            (norm,),
        )
        row = cur.fetchone()
        if row:
            cur.close()
            return row

        # 2. Fuzzy: build LIKE conditions from significant words (3+ chars)
        words = [w for w in norm.split() if len(w) >= 3]
        if words:
            like_clauses = " AND ".join(["place_name LIKE %s"] * len(words))
            like_params = [f"%{w}%" for w in words]
            cur.execute(
                f"SELECT latitude, longitude, formatted_address, place_id "
                f"FROM coordinates WHERE {like_clauses} LIMIT 1",
                like_params,
            )
            row = cur.fetchone()
            if row:
                log.info("Fuzzy cache hit for '%s' (normalized: '%s')", place_name, norm)
                cur.close()
                return row

        cur.close()
        return None
    finally:
        conn.close()


def _cache_save(place_name: str, lat: float, lng: float,
                formatted_address: str, place_id: str) -> None:
    """Save coordinates to cache using normalized name."""
    norm = _normalize(place_name)
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO coordinates (place_name, latitude, longitude, formatted_address, place_id) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE latitude=%s, longitude=%s, formatted_address=%s, place_id=%s",
            (norm, lat, lng, formatted_address, place_id,
             lat, lng, formatted_address, place_id),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def _geocode(place_name: str) -> str:
    """Geocode a place name. Returns cached result or calls Google Maps API."""
    cached = _cache_lookup(place_name)
    if cached:
        return (
            f"Coordinates for '{place_name}' (cached):\n"
            f"  Latitude:  {cached['latitude']}\n"
            f"  Longitude: {cached['longitude']}\n"
            f"  Address:   {cached['formatted_address'] or 'N/A'}"
        )

    gmaps = _get_gmaps()
    results = gmaps.geocode(place_name)

    if not results:
        return f"No geocoding results found for '{place_name}'."

    result = results[0]
    loc = result["geometry"]["location"]
    lat = loc["lat"]
    lng = loc["lng"]
    formatted = result.get("formatted_address", "")
    place_id = result.get("place_id", "")

    _cache_save(place_name, lat, lng, formatted, place_id)
    log.info("Geocoded '%s' -> (%s, %s), saved to cache", place_name, lat, lng)

    return (
        f"Coordinates for '{place_name}':\n"
        f"  Latitude:  {lat}\n"
        f"  Longitude: {lng}\n"
        f"  Address:   {formatted}"
    )


def _list_cached() -> str:
    """List all cached coordinates."""
    conn = _get_db()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT place_name, latitude, longitude, formatted_address "
            "FROM coordinates ORDER BY place_name"
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    if not rows:
        return "No cached coordinates."

    lines = [f"Cached coordinates ({len(rows)}):"]
    for r in rows:
        lines.append(
            f"  - {r['place_name']}: ({r['latitude']}, {r['longitude']}) "
            f"— {r['formatted_address'] or 'N/A'}"
        )
    return "\n".join(lines)


# --- async dispatcher ---

async def run_geocode_op(operation: str, place_name: str = "") -> str:
    """Dispatch geocode operations."""
    op = (operation or "").strip().lower()

    try:
        if op == "geocode":
            if not place_name:
                return "Error: place_name required for 'geocode'."
            return await asyncio.to_thread(_geocode, place_name)

        elif op == "list_cached":
            return await asyncio.to_thread(_list_cached)

        else:
            return "Error: unknown operation. Valid: geocode | list_cached"

    except Exception as exc:
        log.exception("Geocode op '%s' failed", op)
        return f"Geocode error ({op}): {exc}"
