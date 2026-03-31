"""
Google Maps Geocoding Plugin for MCP Agent

Provides geocode tool for converting place names to GPS coordinates.
Results are permanently cached in MySQL to minimize API costs.
Operations: geocode, list_cached
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from geocode_google import run_geocode_op


class _GeocodeArgs(BaseModel):
    operation: Literal["geocode", "list_cached"] = Field(
        description="Operation: 'geocode' to convert a place name to coordinates, 'list_cached' to show all saved coordinates"
    )
    place_name: Optional[str] = Field(
        default="",
        description="Place name or address to geocode (e.g. 'The Metreon, San Francisco')"
    )


async def geocode_executor(
    operation: str,
    place_name: str = "",
) -> str:
    """Execute geocode operation."""
    return await run_geocode_op(operation, place_name or "")


class GeocodeGooglePlugin(BasePlugin):
    """Google Maps Geocoding plugin with coordinate caching."""

    PLUGIN_NAME = "plugin_geocode_google"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Google Maps Geocoding with permanent coordinate cache"
    DEPENDENCIES = ["googlemaps"]
    ENV_VARS = ["GEMINI_API_KEY"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        try:
            import os
            if not os.environ.get("GEMINI_API_KEY"):
                print("Geocode plugin: GEMINI_API_KEY not set")
                return False
            self.enabled = True
            return True
        except Exception as e:
            print(f"Geocode plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=geocode_executor,
                    name="geocode",
                    description=(
                        "Convert a place name or address to GPS coordinates (latitude/longitude). "
                        "Results are permanently cached to avoid repeated API calls. "
                        "Operations: geocode (look up coordinates), list_cached (show all saved coordinates). "
                        "Use this before calling weather APIs that need lat/lng."
                    ),
                    args_schema=_GeocodeArgs,
                )
            ]
        }
