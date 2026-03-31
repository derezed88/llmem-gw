"""
Google Calendar API operations.

Provides calendar listing, event querying, freebusy checks, and event creation.
Uses a separate token file (token_calendar.json) from Drive to keep scopes isolated.
"""

import os
import asyncio
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import (
    log,
    CALENDAR_TOKEN_FILE,
    CALENDAR_CREDS_FILE,
    CALENDAR_SCOPES,
)

_calendar_service = None


def _get_calendar_service():
    """Build or return cached Calendar API v3 service."""
    global _calendar_service
    creds = None

    if os.path.exists(CALENDAR_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            CALENDAR_TOKEN_FILE, CALENDAR_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GAuthRequest())
        else:
            if not os.path.exists(CALENDAR_CREDS_FILE):
                raise FileNotFoundError(
                    "Missing 'credentials.json'. Download from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                CALENDAR_CREDS_FILE, CALENDAR_SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(CALENDAR_TOKEN_FILE, "w") as fh:
            fh.write(creds.to_json())

    _calendar_service = build("calendar", "v3", credentials=creds)
    return _calendar_service


def _list_calendars() -> str:
    """List all calendars on the user's calendar list."""
    svc = _get_calendar_service()
    result = svc.calendarList().list().execute()
    items = result.get("items", [])
    if not items:
        return "No calendars found."
    lines = ["Calendars:"]
    for cal in items:
        primary = " [PRIMARY]" if cal.get("primary") else ""
        lines.append(
            f"  - {cal.get('summary', '(no name)')}{primary}  (id: {cal['id']})"
        )
    return "\n".join(lines)


def _list_events(
    calendar_id: str = "primary",
    time_min: str = "",
    time_max: str = "",
    max_results: int = 10,
    query: str = "",
) -> str:
    """List events from a calendar within an optional time range."""
    svc = _get_calendar_service()

    now = datetime.now(timezone.utc)

    # Default time_min: now
    if not time_min:
        time_min = now.isoformat()
    # Default time_max: 7 days from time_min
    if not time_max:
        time_max = (now + timedelta(days=7)).isoformat()

    # Ensure RFC3339 with Z suffix if no tz offset present
    for ts in [time_min, time_max]:
        if ts and "T" not in ts:
            pass  # API accepts date-only too

    kwargs = {
        "calendarId": calendar_id,
        "timeMin": time_min,
        "timeMax": time_max,
        "maxResults": max_results,
        "singleEvents": True,
        "orderBy": "startTime",
    }
    if query:
        kwargs["q"] = query

    result = svc.events().list(**kwargs).execute()
    items = result.get("items", [])

    if not items:
        return f"No events found in '{calendar_id}' for the specified range."

    lines = [f"Events ({len(items)}):"]
    for ev in items:
        start = ev.get("start", {})
        start_str = start.get("dateTime", start.get("date", "?"))
        end = ev.get("end", {})
        end_str = end.get("dateTime", end.get("date", "?"))
        summary = ev.get("summary", "(no title)")
        location = ev.get("location", "")
        status = ev.get("status", "")
        event_id = ev.get("id", "")

        line = f"  - {start_str} -> {end_str}: {summary}"
        if location:
            line += f"  @ {location}"
        if status and status != "confirmed":
            line += f"  [{status}]"
        line += f"  (id: {event_id})"
        lines.append(line)

    return "\n".join(lines)


def _get_event(calendar_id: str, event_id: str) -> str:
    """Get detailed info about a single event."""
    svc = _get_calendar_service()
    ev = svc.events().get(calendarId=calendar_id, eventId=event_id).execute()

    start = ev.get("start", {})
    end = ev.get("end", {})
    lines = [
        f"Event: {ev.get('summary', '(no title)')}",
        f"  Status: {ev.get('status', '?')}",
        f"  Start: {start.get('dateTime', start.get('date', '?'))}",
        f"  End:   {end.get('dateTime', end.get('date', '?'))}",
    ]
    if ev.get("location"):
        lines.append(f"  Location: {ev['location']}")
    if ev.get("description"):
        desc = ev["description"]
        if len(desc) > 500:
            desc = desc[:500] + "..."
        lines.append(f"  Description: {desc}")
    if ev.get("attendees"):
        att_list = ", ".join(
            f"{a.get('email', '?')} ({a.get('responseStatus', '?')})"
            for a in ev["attendees"][:10]
        )
        lines.append(f"  Attendees: {att_list}")
    if ev.get("recurrence"):
        lines.append(f"  Recurrence: {ev['recurrence']}")
    if ev.get("htmlLink"):
        lines.append(f"  Link: {ev['htmlLink']}")

    return "\n".join(lines)


def _freebusy(
    calendar_ids: list[str],
    time_min: str = "",
    time_max: str = "",
) -> str:
    """Check free/busy status for one or more calendars."""
    svc = _get_calendar_service()
    now = datetime.now(timezone.utc)

    if not time_min:
        time_min = now.isoformat()
    if not time_max:
        time_max = (now + timedelta(days=1)).isoformat()

    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "items": [{"id": cid} for cid in calendar_ids],
    }
    result = svc.freebusy().query(body=body).execute()

    calendars = result.get("calendars", {})
    lines = ["Free/Busy:"]
    for cid, info in calendars.items():
        busy = info.get("busy", [])
        if not busy:
            lines.append(f"  {cid}: FREE for entire range")
        else:
            lines.append(f"  {cid}: {len(busy)} busy block(s)")
            for block in busy[:20]:
                lines.append(f"    {block['start']} -> {block['end']}")
        errors = info.get("errors", [])
        for err in errors:
            lines.append(f"    ERROR: {err.get('reason', '?')}")

    return "\n".join(lines)


def _create_event(
    calendar_id: str,
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    all_day: bool = False,
) -> str:
    """Create a new calendar event.

    Args:
        calendar_id: Calendar to create the event in.
        summary: Event title.
        start_time: RFC3339 datetime (e.g. '2026-03-25T14:00:00-07:00') or
                     date string for all-day events (e.g. '2026-03-25').
        end_time: RFC3339 datetime or date string.
        description: Optional event description.
        location: Optional event location.
        all_day: If True, use date-only start/end for an all-day event.
    """
    svc = _get_calendar_service()

    if all_day:
        # Strip any time component for all-day events
        start_date = start_time[:10] if "T" in start_time else start_time
        end_date = end_time[:10] if "T" in end_time else end_time
        event_body = {
            "summary": summary,
            "start": {"date": start_date},
            "end": {"date": end_date},
        }
    else:
        event_body = {
            "summary": summary,
            "start": {"dateTime": start_time},
            "end": {"dateTime": end_time},
        }

    if description:
        event_body["description"] = description
    if location:
        event_body["location"] = location

    created = svc.events().insert(calendarId=calendar_id, body=event_body).execute()

    start = created.get("start", {})
    start_str = start.get("dateTime", start.get("date", "?"))
    return (
        f"Created event: {created.get('summary', '(no title)')} "
        f"at {start_str} (id: {created.get('id', '?')})"
    )


def _delete_event(calendar_id: str, event_id: str) -> str:
    """Delete a calendar event."""
    svc = _get_calendar_service()
    svc.events().delete(calendarId=calendar_id, eventId=event_id).execute()
    return f"Deleted event: {event_id}"


# --- async wrappers for tool executors ---

async def run_calendar_op(
    operation: str,
    calendar_id: str = "primary",
    event_id: str = "",
    time_min: str = "",
    time_max: str = "",
    max_results: int = 10,
    query: str = "",
    calendar_ids: str = "",
    summary: str = "",
    start_time: str = "",
    end_time: str = "",
    description: str = "",
    location: str = "",
    all_day: bool = False,
) -> str:
    """Dispatch calendar operations."""
    op = (operation or "").strip().lower()

    try:
        if op == "list_calendars":
            return await asyncio.to_thread(_list_calendars)

        elif op == "list_events":
            return await asyncio.to_thread(
                _list_events, calendar_id, time_min, time_max, max_results, query
            )

        elif op == "get_event":
            if not event_id:
                return "Error: event_id required for 'get_event'."
            return await asyncio.to_thread(_get_event, calendar_id, event_id)

        elif op == "freebusy":
            ids = [c.strip() for c in (calendar_ids or calendar_id).split(",") if c.strip()]
            if not ids:
                ids = ["primary"]
            return await asyncio.to_thread(_freebusy, ids, time_min, time_max)

        elif op == "create_event":
            if not summary:
                return "Error: summary (event title) required for 'create_event'."
            if not start_time:
                return "Error: start_time required for 'create_event'."
            if not end_time:
                return "Error: end_time required for 'create_event'."
            return await asyncio.to_thread(
                _create_event, calendar_id, summary, start_time, end_time,
                description, location, all_day,
            )

        elif op == "delete_event":
            if not event_id:
                return "Error: event_id required for 'delete_event'."
            return await asyncio.to_thread(_delete_event, calendar_id, event_id)

        else:
            return (
                "Error: unknown operation. "
                "Valid: list_calendars | list_events | get_event | freebusy | create_event | delete_event"
            )

    except Exception as exc:
        log.exception("Calendar op '%s' failed", op)
        return f"Calendar error ({op}): {exc}"
