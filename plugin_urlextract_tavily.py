"""
Tavily URL Extraction Plugin for MCP Agent

Provides url_extract tool for extracting web page content via Tavily.
Supports both plain extraction and query-focused extraction.

Tool call syntax:
  url_extract tavily <URL>
  url_extract tavily <URL> <query>

Requires TAVILY_API_KEY in .env.
"""

import re
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin


class _UrlExtractArgs(BaseModel):
    method: Literal["tavily"] = Field(description="Extraction method name (e.g. 'tavily')")
    url: str = Field(description="The URL of the web page to extract content from")
    query: Optional[str] = Field(
        default="",
        description=(
            "Optional query to focus the extraction. "
            "When provided, only the most relevant content chunks "
            "matching the query are returned. "
            "Omit for full-page extraction."
        )
    )


class UrlextractTavilyPlugin(BasePlugin):
    """Tavily URL extraction plugin."""

    PLUGIN_NAME = "plugin_urlextract_tavily"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Web page content extraction via Tavily (plain or query-focused)"
    DEPENDENCIES = ["tavily-python"]
    ENV_VARS = ["TAVILY_API_KEY"]

    def __init__(self):
        self.enabled = False
        self._client = None

    def init(self, config: dict) -> bool:
        """Initialize Tavily extraction plugin."""
        try:
            import os
            from tavily import TavilyClient

            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                print("Tavily extract plugin: TAVILY_API_KEY not set in .env")
                return False

            self._client = TavilyClient(api_key)
            self.enabled = True
            return True
        except Exception as e:
            print(f"Tavily extract plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup Tavily extraction resources."""
        self._client = None
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        """Return url_extract tool definitions in LangChain StructuredTool format."""
        client = self._client

        async def url_extract_executor(method: str, url: str, query: str = "") -> str:
            method = method.lower().strip()
            if method == "tavily":
                return await _run_tavily_extract(client, url, query)
            return f"Unknown extraction method '{method}'. Supported methods: tavily"

        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=url_extract_executor,
                    name="url_extract",
                    description=(
                        "Extract web page content from a URL using a named extraction method. "
                        "Returns the full page content in markdown format, with links converted "
                        "to numbered references. "
                        "Optionally focus the extraction on a specific query. "
                        "Method 'tavily' uses the Tavily extraction API (requires TAVILY_API_KEY). "
                        "Use this when you need to read the actual content of a specific URL, "
                        "not just search results."
                    ),
                    args_schema=_UrlExtractArgs,
                )
            ]
        }


def _compress_markdown(text: str) -> str:
    """Convert inline markdown links to numbered reference-style links.

    Keeps all link text and URLs, but deduplicates URLs and moves them
    to a reference section at the end — like Lynx browser output.
    """
    url_to_ref = {}
    ref_list = []

    def replace_link(m):
        anchor, url = m.group(1), m.group(2)
        url = re.sub(r'\s+"[^"]*"$', '', url).strip()
        if url not in url_to_ref:
            ref_num = len(ref_list) + 1
            url_to_ref[url] = ref_num
            ref_list.append((ref_num, url))
        return f"{anchor}[{url_to_ref[url]}]"

    compressed = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, text)

    if ref_list:
        compressed += "\n\n---\n"
        for num, url in ref_list:
            compressed += f"[{num}]: {url}\n"

    return compressed


async def _run_tavily_extract(client, url: str, query: str = "") -> str:
    """Run Tavily extraction and return formatted content."""
    import asyncio

    if client is None:
        return "Tavily extract client not initialized. Check TAVILY_API_KEY in .env."

    # Tavily requires a fully-qualified URL with scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if query:
        def _sync_extract():
            return client.extract(
                urls=url,
                query=query,
                chunks_per_source=5,
                format="markdown"
            )
    else:
        def _sync_extract():
            return client.extract(
                urls=url,
                format="markdown"
            )

    try:
        response = await asyncio.get_event_loop().run_in_executor(None, _sync_extract)

        results = response.get("results", [])
        if not results:
            failed = response.get("failed_results", [])
            if failed:
                return f"Extraction failed for {url}.\nDetails: {failed}"
            return f"No content extracted from {url}."

        lines = []
        archive_entries: list = []  # (url, title, summary_snippet)
        for result in results:
            lines.append(f"Source: {result['url']}")
            title = result.get("title", "")
            if title:
                lines.append(f"Title: {title}")
            if query:
                lines.append(f"Query: {query}")
            lines.append("")
            raw = result.get("raw_content", "")
            compressed = _compress_markdown(raw)
            lines.append(compressed)
            # Collect for archiving — strip leading whitespace/markdown noise from snippet
            r_url = (result.get("url") or "").strip()
            if r_url:
                clean_raw = re.sub(r"\s+", " ", (raw or "")).strip()
                summary_snippet = clean_raw[:250] if clean_raw else (query or "Tavily extraction")
                archive_entries.append((r_url, title or "", summary_snippet))

        # Auto-archive extracted URL(s) — fire-and-forget
        if archive_entries:
            try:
                import asyncio as _asyncio
                from sources import archive_from_search
                for r_url, r_title, r_summary in archive_entries:
                    _asyncio.create_task(archive_from_search(
                        url=r_url,
                        title=r_title or f"[tavily-extract] {r_url[:120]}",
                        summary=r_summary,
                        origin_tool="url_extract_tavily",
                    ))
            except Exception:
                pass

        return "\n".join(lines)

    except Exception as e:
        return f"Tavily extract error: {e}"
