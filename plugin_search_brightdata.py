"""
Bright Data Search Plugin for MCP Agent

Provides search_brightdata tool for web search via Bright Data Web Unlocker
API (same endpoint used by the official @brightdata/mcp npm package).

Requires BRIGHTDATA_API_KEY in .env.
"""

import json
from typing import Dict, Any, Optional, Literal
from urllib.parse import quote_plus
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin

_API_URL = "https://api.brightdata.com/request"
_UNLOCKER_ZONE = "mcp_unlocker"


class _BrightdataSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    engine: Optional[Literal["google", "bing", "yandex"]] = Field(
        default="google",
        description="Search engine to use: 'google' (default), 'bing', or 'yandex'"
    )


class SearchBrightdataPlugin(BasePlugin):
    """Bright Data web search plugin (Web Unlocker SERP)."""

    PLUGIN_NAME = "plugin_search_brightdata"
    PLUGIN_VERSION = "1.1.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Web search via Bright Data SERP API (Google/Bing/Yandex with proxy network)"
    DEPENDENCIES = ["httpx"]
    ENV_VARS = ["BRIGHTDATA_API_KEY"]

    def __init__(self):
        self.enabled = False
        self._api_key = None

    def init(self, config: dict) -> bool:
        try:
            import os
            api_key = os.getenv("BRIGHTDATA_API_KEY")
            if not api_key:
                print("Bright Data search plugin: BRIGHTDATA_API_KEY not set in .env")
                return False
            self._api_key = api_key
            self.enabled = True
            return True
        except Exception as e:
            print(f"Bright Data search plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self._api_key = None
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        api_key = self._api_key

        async def search_brightdata_executor(query: str, engine: str = "google") -> str:
            return await _run_brightdata_search(api_key, query, engine)

        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=search_brightdata_executor,
                    name="search_brightdata",
                    description=(
                        "Search the web using Bright Data SERP API. "
                        "Supports Google, Bing, and Yandex search engines. "
                        "Results include titles, URLs, and descriptions. "
                        "Uses Bright Data's proxy network for reliable, "
                        "geo-targeted scraping."
                    ),
                    args_schema=_BrightdataSearchArgs,
                )
            ]
        }


def _build_search_url(engine: str, query: str) -> str:
    """Build the actual search URL for the given engine."""
    q = quote_plus(query)
    if engine == "bing":
        return f"https://www.bing.com/search?q={q}"
    elif engine == "yandex":
        return f"https://yandex.com/search/?text={q}"
    # Google: append brd_json=1 so Bright Data returns parsed JSON
    return f"https://www.google.com/search?q={q}&brd_json=1"


def _clean_google_results(data: dict) -> list:
    """Extract organic results from Bright Data's parsed Google SERP."""
    organic = data.get("organic", [])
    results = []
    for item in organic:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "description": item.get("description", ""),
        })
    return results


async def _run_brightdata_search(api_key: str, query: str, engine: str = "google") -> str:
    """Run a Bright Data SERP search via the Web Unlocker API."""
    import httpx
    from cost_events import log_cost_event, BRIGHTDATA_PRICING

    if not api_key:
        return "Bright Data search: BRIGHTDATA_API_KEY not configured."

    engine = (engine or "google").lower().strip()
    is_google = (engine == "google")
    search_url = _build_search_url(engine, query)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": search_url,
        "zone": _UNLOCKER_ZONE,
        "format": "raw",
        "data_format": "parsed_light" if is_google else "markdown",
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(_API_URL, headers=headers, json=payload)

        if resp.status_code != 200:
            return f"Bright Data search error (HTTP {resp.status_code}): {resp.text[:400]}"

        await log_cost_event(
            provider="brightdata",
            service="serp-api",
            tool_name="search_brightdata",
            cost_usd=BRIGHTDATA_PRICING["serp"],
            unit="api_call",
            unit_count=1,
            notes=f"engine={engine}",
        )

        if is_google:
            try:
                data = json.loads(resp.text)
                organic = _clean_google_results(data)
                return _format_results(query, engine, organic)
            except json.JSONDecodeError:
                return f"Bright Data search: failed to parse Google SERP response"
        else:
            # Bing/Yandex returns markdown — parse it into rough results
            return f"Bright Data {engine.title()} search results for: {query}\n\n{resp.text[:3000]}"

    except Exception as e:
        return f"Bright Data search error: {e}"


def _format_results(query: str, engine: str, results: list) -> str:
    """Format search results into a readable string."""
    lines = [f"Bright Data {engine.title()} search results for: {query}\n"]

    for i, r in enumerate(results[:10], 1):
        title = r.get("title", "No title")
        link = r.get("link", r.get("url", ""))
        desc = r.get("description", r.get("snippet", ""))
        lines.append(f"{i}. {title}")
        if link:
            lines.append(f"   URL: {link}")
        if desc:
            lines.append(f"   {desc[:250]}")
        lines.append("")

    if not results:
        lines.append("No results found.")

    return "\n".join(lines)
