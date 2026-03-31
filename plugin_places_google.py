"""
Google Places Plugin for MCP Agent

Provides places tool for nearby search, text search, and place details.
Coordinates from the coordinates DB should be used for location-based searches.
Operations: nearby, search, details
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from places_google import run_places_op


class _PlacesArgs(BaseModel):
    operation: Literal["nearby", "search", "details"] = Field(
        description=(
            "Operation: 'nearby' to find places around coordinates, "
            "'search' for free-text place search, "
            "'details' for full info on a specific place by place_id"
        )
    )
    latitude: Optional[float] = Field(
        default=0.0,
        description="Latitude (from coordinates DB). Required for 'nearby', optional bias for 'search'."
    )
    longitude: Optional[float] = Field(
        default=0.0,
        description="Longitude (from coordinates DB). Required for 'nearby', optional bias for 'search'."
    )
    radius: Optional[float] = Field(
        default=500.0,
        description="Search radius in meters (max 50000, default 500 for nearby, 5000 for search)"
    )
    place_type: Optional[str] = Field(
        default="",
        description="Filter by place type for 'nearby' (e.g. restaurant, cafe, gas_station, museum, hotel, park)"
    )
    max_results: Optional[int] = Field(
        default=10,
        description="Max results to return (max 20, default 10)"
    )
    rank_by: Optional[str] = Field(
        default="POPULARITY",
        description="Ranking for 'nearby': POPULARITY (default) or DISTANCE"
    )
    query: Optional[str] = Field(
        default="",
        description="Free-text search query for 'search' operation (e.g. 'best sushi in San Francisco')"
    )
    open_now: Optional[bool] = Field(
        default=False,
        description="Filter to only currently open places (search only)"
    )
    min_rating: Optional[float] = Field(
        default=0.0,
        description="Minimum rating filter 0.0-5.0 (search only)"
    )
    price_levels: Optional[str] = Field(
        default="",
        description="Comma-separated price levels: PRICE_LEVEL_INEXPENSIVE, PRICE_LEVEL_MODERATE, PRICE_LEVEL_EXPENSIVE, PRICE_LEVEL_VERY_EXPENSIVE (search only)"
    )
    place_id: Optional[str] = Field(
        default="",
        description="Place ID for 'details' operation (from a previous nearby or search result)"
    )


async def places_executor(
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
    """Execute places operation."""
    return await run_places_op(
        operation,
        latitude or 0.0,
        longitude or 0.0,
        radius or 500.0,
        place_type or "",
        max_results or 10,
        rank_by or "POPULARITY",
        query or "",
        open_now or False,
        min_rating or 0.0,
        price_levels or "",
        place_id or "",
    )


class PlacesGooglePlugin(BasePlugin):
    """Google Places API plugin."""

    PLUGIN_NAME = "plugin_places_google"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Google Places API (nearby search, text search, place details)"
    DEPENDENCIES = ["httpx"]
    ENV_VARS = ["GEMINI_API_KEY"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        try:
            import os
            if not os.environ.get("GEMINI_API_KEY"):
                print("Places plugin: GEMINI_API_KEY not set")
                return False
            self.enabled = True
            return True
        except Exception as e:
            print(f"Places plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=places_executor,
                    name="places",
                    description=(
                        "Find businesses, POIs, and places using Google Places API. "
                        "IMPORTANT: Get coordinates from the coordinates DB first for nearby searches. "
                        "Operations: nearby (places around a location with optional type filter), "
                        "search (free-text query like 'best pizza in SF' with optional location bias), "
                        "details (full info on a place by place_id from a prior search). "
                        "Returns names, addresses, ratings, price levels, and place_ids for follow-up."
                    ),
                    args_schema=_PlacesArgs,
                )
            ]
        }
