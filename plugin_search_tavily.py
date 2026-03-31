"""
Tavily Search Plugin for MCP Agent

Provides tavily_search tool for web search via Tavily AI search API.
Requires TAVILY_API_KEY in .env.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin


class _TavilySearchArgs(BaseModel):
    query: str = Field(description="Search query")
    search_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="Search depth: 'basic' (faster) or 'advanced' (deeper, more thorough). Default: basic"
    )


class SearchTavilyPlugin(BasePlugin):
    """Tavily AI web search plugin."""

    PLUGIN_NAME = "plugin_search_tavily"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Web search via Tavily AI search API (AI-curated results with answers)"
    DEPENDENCIES = ["tavily-python"]
    ENV_VARS = ["TAVILY_API_KEY"]

    def __init__(self):
        self.enabled = False
        self._client = None

    def init(self, config: dict) -> bool:
        """Initialize Tavily search plugin."""
        try:
            import os
            from tavily import TavilyClient

            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                print("Tavily search plugin: TAVILY_API_KEY not set in .env")
                return False

            self._client = TavilyClient(api_key)
            self.enabled = True
            return True
        except Exception as e:
            print(f"Tavily search plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup Tavily search resources."""
        self._client = None
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        """Return Tavily search tool definitions in LangChain StructuredTool format."""
        client = self._client

        async def search_tavily_executor(query: str, search_depth: str = "basic") -> str:
            return await _run_tavily_search(client, query, search_depth)

        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=search_tavily_executor,
                    name="search_tavily",
                    description=(
                        "Search the web using Tavily AI search API. "
                        "Returns AI-curated results with an answer summary and source URLs. "
                        "Higher quality than raw search engines for research questions."
                    ),
                    args_schema=_TavilySearchArgs,
                )
            ]
        }


async def _run_tavily_search(client, query: str, search_depth: str = "basic") -> str:
    """Run a Tavily search and return formatted results."""
    import asyncio
    from cost_events import log_cost_event, TAVILY_PRICING

    if client is None:
        return "Tavily search client not initialized. Check TAVILY_API_KEY in .env."

    def _sync_search():
        return client.search(
            query=query,
            include_answer="basic",
            search_depth=search_depth
        )

    try:
        response = await asyncio.get_event_loop().run_in_executor(None, _sync_search)

        cost = TAVILY_PRICING.get(search_depth, TAVILY_PRICING["basic"])
        await log_cost_event(
            provider="tavily",
            service="search-api",
            tool_name="search_tavily",
            cost_usd=cost,
            unit="api_call",
            unit_count=1,
            notes=f"search_depth={search_depth}",
        )

        lines = [f"Tavily search results for: {query}\n"]

        answer = response.get("answer", "")
        if answer:
            lines.append(f"Answer: {answer}\n")

        results = response.get("results", [])
        if results:
            lines.append("Sources:")
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.get('title', 'No title')}")
                lines.append(f"   URL: {r.get('url', '')}")
                content = r.get('content', '')
                if content:
                    lines.append(f"   {content[:200]}")
                lines.append("")
        else:
            lines.append("No results found.")

        return "\n".join(lines)

    except Exception as e:
        return f"Tavily search error: {e}"
