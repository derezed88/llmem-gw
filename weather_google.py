"""
Google Maps Weather API operations.

Provides current conditions, daily/hourly forecasts, and weather alerts.
Always returns fresh data — no caching (weather changes too fast).
Coordinates should be looked up from the coordinates DB cache first.
"""

import os
import asyncio
import httpx

from config import log

_API_KEY = None
_BASE = "https://weather.googleapis.com/v1"


def _get_key() -> str:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.environ.get("GEMINI_API_KEY", "")
        if not _API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set (used for Weather API)")
    return _API_KEY


def _current_conditions(lat: float, lng: float) -> str:
    """Get current weather conditions."""
    resp = httpx.get(
        f"{_BASE}/currentConditions:lookup",
        params={
            "key": _get_key(),
            "location.latitude": lat,
            "location.longitude": lng,
            "unitsSystem": "IMPERIAL",
        },
        timeout=15,
    )
    resp.raise_for_status()
    d = resp.json()

    cond = d.get("weatherCondition", {})
    desc = cond.get("description", {}).get("text", "Unknown")
    temp = d.get("temperature", {})
    feels = d.get("feelsLikeTemperature", {})
    wind = d.get("wind", {})
    wind_dir = wind.get("direction", {}).get("cardinal", "?")
    wind_speed = wind.get("speed", {}).get("value", "?")
    wind_unit = wind.get("speed", {}).get("unit", "")
    gust = wind.get("gust", {})
    precip = d.get("precipitation", {})
    rain_prob = precip.get("probability", {}).get("percent", 0)

    lines = [
        f"Current Conditions: {desc}",
        f"  Temperature: {temp.get('degrees', '?')}°{_unit_short(temp.get('unit', ''))}",
        f"  Feels Like:  {feels.get('degrees', '?')}°{_unit_short(feels.get('unit', ''))}",
        f"  Humidity:    {d.get('relativeHumidity', '?')}%",
        f"  Wind:        {wind_speed} {_unit_short(wind_unit)} {wind_dir}",
    ]
    if gust.get("value"):
        lines.append(f"  Gusts:       {gust['value']} {_unit_short(gust.get('unit', ''))}")
    lines.append(f"  Rain Chance: {rain_prob}%")
    lines.append(f"  UV Index:    {d.get('uvIndex', '?')}")
    lines.append(f"  Visibility:  {_fmt_distance(d.get('visibility', {}))}")
    lines.append(f"  Cloud Cover: {d.get('cloudCover', '?')}%")
    lines.append(f"  Daytime:     {'Yes' if d.get('isDaytime') else 'No'}")

    return "\n".join(lines)


def _daily_forecast(lat: float, lng: float, days: int = 5) -> str:
    """Get daily forecast."""
    resp = httpx.get(
        f"{_BASE}/forecast/days:lookup",
        params={
            "key": _get_key(),
            "location.latitude": lat,
            "location.longitude": lng,
            "days": min(days, 10),
            "unitsSystem": "IMPERIAL",
        },
        timeout=15,
    )
    resp.raise_for_status()
    d = resp.json()

    forecast_days = d.get("forecastDays", [])
    if not forecast_days:
        return "No forecast data available."

    lines = [f"Daily Forecast ({len(forecast_days)} days):"]
    for day in forecast_days:
        dd = day.get("displayDate", {})
        date_str = f"{dd.get('year', '?')}-{dd.get('month', '?'):02d}-{dd.get('day', '?'):02d}"
        hi = day.get("maxTemperature", {})
        lo = day.get("minTemperature", {})
        daytime = day.get("daytimeForecast", {})
        day_cond = daytime.get("weatherCondition", {}).get("description", {}).get("text", "")
        nighttime = day.get("nighttimeForecast", {})
        night_cond = nighttime.get("weatherCondition", {}).get("description", {}).get("text", "")
        # Rain from daytime forecast
        day_precip = daytime.get("precipitation", {})
        rain_prob = day_precip.get("probability", {}).get("percent", 0)
        # Wind from daytime forecast
        day_wind = daytime.get("wind", {})
        wind_speed = day_wind.get("speed", {}).get("value", "?")
        wind_unit = day_wind.get("speed", {}).get("unit", "")
        wind_dir = day_wind.get("direction", {}).get("cardinal", "")

        sun = day.get("sunEvents", {})
        sunrise = _fmt_time(sun.get("sunriseTime", ""))
        sunset = _fmt_time(sun.get("sunsetTime", ""))

        lines.append(f"  {date_str}: {day_cond}")
        lines.append(f"    High: {hi.get('degrees', '?')}° / Low: {lo.get('degrees', '?')}°")
        lines.append(f"    Night: {night_cond}")
        wind_str = f"{wind_speed} {_unit_short(wind_unit)}"
        if wind_dir:
            wind_str += f" {wind_dir}"
        lines.append(f"    Rain: {rain_prob}% | Wind: {wind_str}")
        if sunrise != "?" or sunset != "?":
            lines.append(f"    Sunrise: {sunrise} / Sunset: {sunset}")

    return "\n".join(lines)


def _hourly_forecast(lat: float, lng: float, hours: int = 24) -> str:
    """Get hourly forecast."""
    resp = httpx.get(
        f"{_BASE}/forecast/hours:lookup",
        params={
            "key": _get_key(),
            "location.latitude": lat,
            "location.longitude": lng,
            "hours": min(hours, 240),
            "unitsSystem": "IMPERIAL",
        },
        timeout=15,
    )
    resp.raise_for_status()
    d = resp.json()

    forecast_hours = d.get("forecastHours", [])
    if not forecast_hours:
        return "No hourly forecast data available."

    lines = [f"Hourly Forecast ({len(forecast_hours)} hours):"]
    for hr in forecast_hours:
        interval = hr.get("interval", {})
        start = _fmt_time(interval.get("startTime", ""))
        cond = hr.get("weatherCondition", {}).get("description", {}).get("text", "")
        temp = hr.get("temperature", {})
        rain_prob = hr.get("precipitation", {}).get("probability", {}).get("percent", 0)
        wind = hr.get("wind", {})
        wind_speed = wind.get("speed", {}).get("value", "?")
        wind_unit = wind.get("speed", {}).get("unit", "")

        lines.append(
            f"  {start}: {temp.get('degrees', '?')}° {cond} "
            f"| Rain: {rain_prob}% | Wind: {wind_speed} {_unit_short(wind_unit)}"
        )

    return "\n".join(lines)


def _weather_alerts(lat: float, lng: float) -> str:
    """Get active weather alerts."""
    resp = httpx.get(
        f"{_BASE}/publicAlerts:lookup",
        params={
            "key": _get_key(),
            "location.latitude": lat,
            "location.longitude": lng,
        },
        timeout=15,
    )
    resp.raise_for_status()
    d = resp.json()

    alerts = d.get("alerts", [])
    if not alerts:
        return "No active weather alerts for this location."

    lines = [f"Weather Alerts ({len(alerts)}):"]
    for alert in alerts:
        event = alert.get("event", "Unknown")
        severity = alert.get("severity", "?")
        headline = alert.get("headline", "")
        desc = alert.get("description", "")
        if len(desc) > 300:
            desc = desc[:300] + "..."
        lines.append(f"  [{severity}] {event}")
        if headline:
            lines.append(f"    {headline}")
        if desc:
            lines.append(f"    {desc}")

    return "\n".join(lines)


# --- helpers ---

def _unit_short(unit: str) -> str:
    """Convert unit enum to short display."""
    return {
        "FAHRENHEIT": "F",
        "CELSIUS": "C",
        "MILES_PER_HOUR": "mph",
        "KILOMETERS_PER_HOUR": "km/h",
        "INCHES": "in",
        "MILLIMETERS": "mm",
        "MILES": "mi",
        "KILOMETERS": "km",
    }.get(unit, unit)


def _fmt_distance(vis: dict) -> str:
    """Format visibility distance."""
    val = vis.get("distance", vis.get("value", "?"))
    unit = _unit_short(vis.get("unit", ""))
    return f"{val} {unit}" if unit else str(val)


def _fmt_time(ts: str, tz_id: str = "America/Los_Angeles") -> str:
    """Convert RFC3339/UTC timestamp to local time string."""
    if not ts:
        return "?"
    try:
        from datetime import datetime, timezone as tz
        from zoneinfo import ZoneInfo
        # Parse UTC timestamp like "2026-03-19T14:14:20.557225780Z"
        clean = ts.split(".")[0].replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz.utc)
        local = dt.astimezone(ZoneInfo(tz_id))
        return local.strftime("%I:%M %p")
    except Exception:
        # Fallback: just extract time portion
        if "T" in ts:
            return ts.split("T")[1][:5]
        return ts


# --- async dispatcher ---

async def run_weather_op(
    operation: str,
    latitude: float = 0.0,
    longitude: float = 0.0,
    days: int = 5,
    hours: int = 24,
) -> str:
    """Dispatch weather operations."""
    op = (operation or "").strip().lower()

    if not latitude and not longitude:
        return "Error: latitude and longitude required. Use the geocode tool or coordinates DB to get them first."

    try:
        if op == "current":
            return await asyncio.to_thread(_current_conditions, latitude, longitude)

        elif op == "daily":
            return await asyncio.to_thread(_daily_forecast, latitude, longitude, days)

        elif op == "hourly":
            return await asyncio.to_thread(_hourly_forecast, latitude, longitude, hours)

        elif op == "alerts":
            return await asyncio.to_thread(_weather_alerts, latitude, longitude)

        else:
            return "Error: unknown operation. Valid: current | daily | hourly | alerts"

    except httpx.HTTPStatusError as exc:
        log.exception("Weather API HTTP error for op '%s'", op)
        return f"Weather API error ({op}): {exc.response.status_code} — {exc.response.text[:200]}"
    except Exception as exc:
        log.exception("Weather op '%s' failed", op)
        return f"Weather error ({op}): {exc}"
