"""
Google Places API (New v1) operations.

Provides nearby search, text search, and place details.
Coordinates for location-based searches come from the coordinates DB cache.
No caching of results — always fresh data.
"""

import os
import asyncio
import httpx

from config import log

_API_KEY = None
_BASE = "https://places.googleapis.com/v1"

# Field masks by operation
_NEARBY_FIELDS = (
    "places.id,places.displayName,places.formattedAddress,"
    "places.rating,places.userRatingCount,places.primaryType,"
    "places.priceLevel,places.location"
)
_TEXT_FIELDS = (
    "places.id,places.displayName,places.formattedAddress,"
    "places.rating,places.userRatingCount,places.primaryType,"
    "places.priceLevel,places.location,nextPageToken"
)
_DETAILS_FIELDS = (
    "id,displayName,formattedAddress,location,types,primaryType,"
    "rating,userRatingCount,priceLevel,websiteUri,"
    "internationalPhoneNumber,currentOpeningHours,businessStatus,"
    "editorialSummary,reviews"
)


def _get_key() -> str:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.environ.get("GEMINI_API_KEY", "")
        if not _API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set (used for Places API)")
    return _API_KEY


def _fmt_price(level: str) -> str:
    return {
        "PRICE_LEVEL_FREE": "Free",
        "PRICE_LEVEL_INEXPENSIVE": "$",
        "PRICE_LEVEL_MODERATE": "$$",
        "PRICE_LEVEL_EXPENSIVE": "$$$",
        "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
    }.get(level, "")


def _fmt_place(p: dict) -> str:
    """Format a single place from search results."""
    name = p.get("displayName", {}).get("text", "(unknown)")
    addr = p.get("formattedAddress", "")
    ptype = (p.get("primaryType") or "").replace("_", " ")
    rating = p.get("rating")
    reviews = p.get("userRatingCount")
    price = _fmt_price(p.get("priceLevel", ""))
    place_id = p.get("id", "")

    line = f"  - {name}"
    if ptype:
        line += f" ({ptype})"
    if rating:
        line += f" [{rating}"
        if reviews:
            line += f", {reviews} reviews"
        line += "]"
    if price:
        line += f" {price}"
    if addr:
        line += f"\n    {addr}"
    if place_id:
        line += f"\n    place_id: {place_id}"
    return line


def _nearby(lat: float, lng: float, radius: float = 500.0,
            place_type: str = "", max_results: int = 10,
            rank_by: str = "POPULARITY") -> str:
    """Search for places near a location."""
    body = {
        "maxResultCount": min(max_results, 20),
        "rankPreference": rank_by.upper(),
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": min(radius, 50000.0),
            }
        },
    }
    if place_type:
        body["includedTypes"] = [place_type]

    resp = httpx.post(
        f"{_BASE}/places:searchNearby",
        json=body,
        headers={
            "X-Goog-Api-Key": _get_key(),
            "X-Goog-FieldMask": _NEARBY_FIELDS,
        },
        timeout=15,
    )
    resp.raise_for_status()
    places = resp.json().get("places", [])

    if not places:
        desc = f"type={place_type}" if place_type else "any type"
        return f"No places found near ({lat}, {lng}) within {radius}m ({desc})."

    type_desc = f" ({place_type.replace('_', ' ')})" if place_type else ""
    lines = [f"Places near ({lat}, {lng}){type_desc} — {len(places)} results:"]
    for p in places:
        lines.append(_fmt_place(p))
    return "\n".join(lines)


def _text_search(query: str, lat: float = 0.0, lng: float = 0.0,
                 radius: float = 0.0, max_results: int = 10,
                 open_now: bool = False, min_rating: float = 0.0,
                 price_levels: str = "") -> str:
    """Search for places by text query with optional location bias."""
    body = {
        "textQuery": query,
        "pageSize": min(max_results, 20),
    }

    if lat and lng:
        bias = {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius if radius else 5000.0,
            }
        }
        body["locationBias"] = bias

    if open_now:
        body["openNow"] = True
    if min_rating > 0:
        body["minRating"] = min_rating
    if price_levels:
        body["priceLevels"] = [p.strip() for p in price_levels.split(",") if p.strip()]

    resp = httpx.post(
        f"{_BASE}/places:searchText",
        json=body,
        headers={
            "X-Goog-Api-Key": _get_key(),
            "X-Goog-FieldMask": _TEXT_FIELDS,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    places = data.get("places", [])

    if not places:
        return f"No places found for '{query}'."

    lines = [f"Search results for '{query}' — {len(places)} results:"]
    for p in places:
        lines.append(_fmt_place(p))
    return "\n".join(lines)


def _details(place_id: str) -> str:
    """Get detailed info about a specific place."""
    resp = httpx.get(
        f"{_BASE}/places/{place_id}",
        headers={
            "X-Goog-Api-Key": _get_key(),
            "X-Goog-FieldMask": _DETAILS_FIELDS,
        },
        timeout=15,
    )
    resp.raise_for_status()
    p = resp.json()

    name = p.get("displayName", {}).get("text", "(unknown)")
    addr = p.get("formattedAddress", "")
    ptype = (p.get("primaryType") or "").replace("_", " ")
    rating = p.get("rating")
    reviews_count = p.get("userRatingCount")
    price = _fmt_price(p.get("priceLevel", ""))
    phone = p.get("internationalPhoneNumber", "")
    website = p.get("websiteUri", "")
    status = (p.get("businessStatus") or "").replace("_", " ").title()
    editorial = p.get("editorialSummary", {}).get("text", "")

    lines = [f"Place Details: {name}"]
    if ptype:
        lines.append(f"  Type:    {ptype}")
    if addr:
        lines.append(f"  Address: {addr}")
    if status:
        lines.append(f"  Status:  {status}")
    if rating:
        r_str = f"{rating}"
        if reviews_count:
            r_str += f" ({reviews_count} reviews)"
        lines.append(f"  Rating:  {r_str}")
    if price:
        lines.append(f"  Price:   {price}")
    if phone:
        lines.append(f"  Phone:   {phone}")
    if website:
        lines.append(f"  Website: {website}")

    # Opening hours
    hours = p.get("currentOpeningHours", {})
    weekday_text = hours.get("weekdayDescriptions", [])
    if weekday_text:
        lines.append("  Hours:")
        for day_text in weekday_text:
            lines.append(f"    {day_text}")

    if editorial:
        lines.append(f"  Summary: {editorial}")

    # Top reviews (up to 3)
    reviews = p.get("reviews", [])
    if reviews:
        lines.append(f"  Reviews ({len(reviews)} shown):")
        for rev in reviews[:3]:
            author = rev.get("authorAttribution", {}).get("displayName", "Anonymous")
            rev_rating = rev.get("rating", "?")
            text = rev.get("text", {}).get("text", "")
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"    [{rev_rating}/5] {author}: {text}")

    loc = p.get("location", {})
    if loc:
        lines.append(f"  Location: ({loc.get('latitude', '?')}, {loc.get('longitude', '?')})")

    return "\n".join(lines)


# --- async dispatcher ---

async def run_places_op(
    operation: str,
    latitude: float = 0.0,
    longitude: float = 0.0,
    radius: float = 500.0,
    place_type: str = "",
    max_results: int = 10,
    rank_by: str = "POPULARITY",
    query: str = "",
    open_now: bool = False,
    min_rating: float = 0.0,
    price_levels: str = "",
    place_id: str = "",
) -> str:
    """Dispatch places operations."""
    op = (operation or "").strip().lower()

    try:
        if op == "nearby":
            if not latitude and not longitude:
                return "Error: latitude and longitude required. Get them from coordinates DB or geocode tool."
            return await asyncio.to_thread(
                _nearby, latitude, longitude, radius, place_type, max_results, rank_by
            )

        elif op == "search":
            if not query:
                return "Error: query required for 'search'."
            return await asyncio.to_thread(
                _text_search, query, latitude, longitude, radius,
                max_results, open_now, min_rating, price_levels
            )

        elif op == "details":
            if not place_id:
                return "Error: place_id required for 'details'. Get it from a nearby or search result."
            return await asyncio.to_thread(_details, place_id)

        else:
            return "Error: unknown operation. Valid: nearby | search | details"

    except httpx.HTTPStatusError as exc:
        log.exception("Places API HTTP error for op '%s'", op)
        return f"Places API error ({op}): {exc.response.status_code} — {exc.response.text[:200]}"
    except Exception as exc:
        log.exception("Places op '%s' failed", op)
        return f"Places error ({op}): {exc}"
