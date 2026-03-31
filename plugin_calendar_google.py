"""
Google Calendar Plugin for MCP Agent

Provides google_calendar tool for operations on Google Calendar.
Operations: list_calendars, list_events, get_event, freebusy, create_event
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from calendar_google import run_calendar_op


class _GoogleCalendarArgs(BaseModel):
    operation: Literal["list_calendars", "list_events", "get_event", "freebusy", "create_event", "delete_event"] = Field(
        description="Operation to perform: list_calendars, list_events, get_event, freebusy, create_event, delete_event"
    )
    calendar_id: Optional[str] = Field(
        default="primary",
        description="Calendar ID (default 'primary'). Use list_calendars to discover IDs."
    )
    event_id: Optional[str] = Field(
        default="",
        description="Event ID for get_event operation"
    )
    time_min: Optional[str] = Field(
        default="",
        description="Start of time range (RFC3339, e.g. '2026-03-19T00:00:00Z'). Defaults to now."
    )
    time_max: Optional[str] = Field(
        default="",
        description="End of time range (RFC3339). Defaults to 7 days from now for list_events, 1 day for freebusy."
    )
    max_results: Optional[int] = Field(
        default=10,
        description="Max events to return (list_events only, default 10)"
    )
    query: Optional[str] = Field(
        default="",
        description="Free-text search filter for list_events (matches summary, description, location, attendees)"
    )
    calendar_ids: Optional[str] = Field(
        default="",
        description="Comma-separated calendar IDs for freebusy (defaults to calendar_id)"
    )
    summary: Optional[str] = Field(
        default="",
        description="Event title (create_event only)"
    )
    start_time: Optional[str] = Field(
        default="",
        description="Event start (RFC3339 datetime e.g. '2026-03-25T14:00:00-07:00', or date '2026-03-25' for all-day). Required for create_event."
    )
    end_time: Optional[str] = Field(
        default="",
        description="Event end (RFC3339 datetime or date). Required for create_event."
    )
    description: Optional[str] = Field(
        default="",
        description="Event description (create_event only)"
    )
    location: Optional[str] = Field(
        default="",
        description="Event location (create_event only)"
    )
    all_day: Optional[bool] = Field(
        default=False,
        description="If true, create an all-day event using date-only start/end (create_event only)"
    )


async def google_calendar_executor(
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
    """Execute Google Calendar operation."""
    return await run_calendar_op(
        operation,
        calendar_id or "primary",
        event_id or "",
        time_min or "",
        time_max or "",
        max_results or 10,
        query or "",
        calendar_ids or "",
        summary or "",
        start_time or "",
        end_time or "",
        description or "",
        location or "",
        all_day or False,
    )


class GoogleCalendarPlugin(BasePlugin):
    """Google Calendar operations plugin."""

    PLUGIN_NAME = "plugin_calendar_google"
    PLUGIN_VERSION = "1.1.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Google Calendar operations (list calendars, events, freebusy, create events)"
    DEPENDENCIES = ["google-auth", "google-auth-oauthlib", "google-api-python-client"]
    ENV_VARS = []

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize Google Calendar plugin."""
        try:
            import os
            from config import CALENDAR_CREDS_FILE, CALENDAR_TOKEN_FILE

            if not os.path.exists(CALENDAR_CREDS_FILE):
                print("Google Calendar plugin: credentials.json not found")
                return False

            if not os.path.exists(CALENDAR_TOKEN_FILE):
                print(
                    f"Google Calendar plugin: {CALENDAR_TOKEN_FILE} not found. "
                    "Run calendar_google_auth.py to authorize."
                )
                return False

            self.enabled = True
            return True
        except Exception as e:
            print(f"Google Calendar plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=google_calendar_executor,
                    name="google_calendar",
                    description=(
                        "Operations on Google Calendar. "
                        "Operations: list_calendars (show all calendars), "
                        "list_events (query events with optional time range and search), "
                        "get_event (detailed info for one event), "
                        "freebusy (check availability), "
                        "create_event (create a new event with summary, start_time, end_time, optional description/location/all_day), "
                        "delete_event (delete an event by event_id). "
                        "Use list_calendars first to discover calendar IDs."
                    ),
                    args_schema=_GoogleCalendarArgs,
                )
            ]
        }
