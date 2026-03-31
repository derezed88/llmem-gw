"""
xAI Search Plugin for MCP Agent

Provides xai_search tool for web search via xAI's x_search tool (Grok).
Requires XAI_API_KEY in .env.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin

logger = logging.getLogger("AISvc")


class _XaiSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    model: Optional[str] = Field(
        default="grok-4-1-fast-reasoning",
        description=(
            "xAI model to use. Options: 'grok-4-1-fast-reasoning' (default, "
            "fast reasoning with search), 'grok-4' (full reasoning). "
            "Only grok-4 family supports x_search."
        )
    )


class SearchXaiPlugin(BasePlugin):
    """xAI (Grok) web search plugin using x_search."""

    PLUGIN_NAME = "plugin_search_xai"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Web search via xAI Grok x_search (real-time web and X/Twitter search)"
    DEPENDENCIES = ["xai-sdk"]
    ENV_VARS = ["XAI_API_KEY"]

    def __init__(self):
        self.enabled = False
        self._api_key = None

    def init(self, config: dict) -> bool:
        """Initialize xAI search plugin."""
        try:
            import os
            from xai_sdk import Client

            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                print("xAI search plugin: XAI_API_KEY not set in .env")
                return False

            self._api_key = api_key
            self.enabled = True
            return True
        except Exception as e:
            print(f"xAI search plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup xAI search resources."""
        self._api_key = None
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        """Return xAI search tool definitions in LangChain StructuredTool format."""
        api_key = self._api_key

        async def search_xai_executor(query: str, model: str = "grok-4-1-fast-reasoning") -> str:
            return await _run_xai_search(api_key, query, model)

        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=search_xai_executor,
                    name="search_xai",
                    description=(
                        "Search the web and X/Twitter using xAI Grok's x_search tool. "
                        "Returns AI-synthesized answer with citations from real-time web results. "
                        "Best for current events, social media trends, and real-time information."
                    ),
                    args_schema=_XaiSearchArgs,
                )
            ]
        }


async def _run_xai_search(api_key: str, query: str, model: str = "grok-4-1-fast-reasoning") -> str:
    """Run an xAI search using x_search tool and return formatted results."""
    import asyncio

    if not api_key:
        return "xAI search client not initialized. Check XAI_API_KEY in .env."

    search_result = {"text": "", "tokens_in": 0, "tokens_out": 0}

    def _sync_search():
        from xai_sdk import Client
        from xai_sdk.chat import user
        from xai_sdk.tools import x_search

        client = Client(api_key=api_key)
        chat = client.chat.create(
            model=model,
            tools=[x_search()],
        )
        chat.append(user(query))

        response = None
        for response, chunk in chat.stream():
            pass  # consume stream to completion

        if response is None:
            search_result["text"] = f"No response from xAI for: {query}"
            return search_result["text"]

        # Capture token usage if available
        if hasattr(response, 'usage') and response.usage:
            search_result["tokens_in"] = getattr(response.usage, 'prompt_tokens', 0) or 0
            search_result["tokens_out"] = getattr(response.usage, 'completion_tokens', 0) or 0

        lines = [f"xAI search results for: {query}\n"]

        content = response.content if hasattr(response, 'content') else ""
        if content:
            lines.append(f"Answer: {content}\n")

        citations = response.citations if hasattr(response, 'citations') else []
        if citations:
            lines.append("Citations:")
            for i, cite in enumerate(citations, 1):
                if isinstance(cite, dict):
                    url = cite.get('url', cite.get('link', str(cite)))
                    title = cite.get('title', url)
                    lines.append(f"{i}. {title}")
                    lines.append(f"   URL: {url}")
                else:
                    lines.append(f"{i}. {cite}")
            lines.append("")
        elif not content:
            lines.append("No results found.")

        search_result["text"] = "\n".join(lines)
        return search_result["text"]

    try:
        logger.info(f"xai_search: querying {model} for: {query}")
        result = await asyncio.get_event_loop().run_in_executor(None, _sync_search)
        logger.info(f"xai_search: completed for: {query}")

        # Log cost event
        from cost_events import log_cost_event, estimate_xai_cost
        tokens_in = search_result["tokens_in"]
        tokens_out = search_result["tokens_out"]
        cost = estimate_xai_cost(model, tokens_in, tokens_out)
        await log_cost_event(
            provider="xai",
            service=model,
            tool_name="search_xai",
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            unit="tokens",
            notes="estimated" if (tokens_in == 0 and tokens_out == 0) else None,
        )

        return result
    except Exception as e:
        logger.error(f"xai_search: error for '{query}': {e}")
        return f"xAI search error: {e}"
