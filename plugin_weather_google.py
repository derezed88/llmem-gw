"""
Google Maps Weather Plugin for MCP Agent

Provides weather tool for current conditions, forecasts, and alerts.
Requires lat/lng coordinates — use the geocode tool or coordinates DB first.
Operations: current, daily, hourly, alerts
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from weather_google import run_weather_op


class _WeatherArgs(BaseModel):
    operation: Literal["current", "daily", "hourly", "alerts"] = Field(
        description="Operation: 'current' for now, 'daily' for up to 10-day forecast, 'hourly' for up to 240h forecast, 'alerts' for active weather alerts"
    )
    latitude: float = Field(
        description="Latitude coordinate (from coordinates DB or geocode tool)"
    )
    longitude: float = Field(
        description="Longitude coordinate (from coordinates DB or geocode tool)"
    )
    days: Optional[int] = Field(
        default=5,
        description="Number of forecast days (daily only, max 10, default 5)"
    )
    hours: Optional[int] = Field(
        default=24,
        description="Number of forecast hours (hourly only, max 240, default 24)"
    )


async def weather_executor(
    operation: str,
    latitude: float = 0.0,
    longitude: float = 0.0,
    days: int = 5,
    hours: int = 24,
) -> str:
    """Execute weather operation."""
    return await run_weather_op(operation, latitude, longitude, days or 5, hours or 24)


class WeatherGooglePlugin(BasePlugin):
    """Google Maps Weather API plugin."""

    PLUGIN_NAME = "plugin_weather_google"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Google Maps Weather API (current conditions, daily/hourly forecasts, alerts)"
    DEPENDENCIES = ["httpx"]
    ENV_VARS = ["GEMINI_API_KEY"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        try:
            import os
            if not os.environ.get("GEMINI_API_KEY"):
                print("Weather plugin: GEMINI_API_KEY not set")
                return False
            self.enabled = True
            return True
        except Exception as e:
            print(f"Weather plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=weather_executor,
                    name="weather",
                    description=(
                        "Get weather data for a location using GPS coordinates. "
                        "IMPORTANT: Get coordinates from the coordinates DB first (or geocode tool if not cached). "
                        "Operations: current (right now), daily (up to 10-day forecast), "
                        "hourly (up to 240h forecast), alerts (active weather alerts). "
                        "All data is in Imperial units (°F, mph, inches)."
                    ),
                    args_schema=_WeatherArgs,
                )
            ]
        }
