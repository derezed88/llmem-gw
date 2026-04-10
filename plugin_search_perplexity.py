"""
Perplexity Search Plugin for MCP Agent

Provides three tools:
  - perplexity_search: raw web search (ranked results, no LLM)
  - sonar_answer: web-grounded AI answers with inline citations (Perplexity Sonar)
  - perplexity_research: multi-step research workflows (Perplexity Agent API)

Requires PERPLEXITY_API_KEY in .env.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin


# ── Arg schemas ──────────────────────────────────────────────

class _PerplexitySearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: Optional[int] = Field(
        default=5,
        description="Number of results to return (1-20). Default: 5"
    )
    country: Optional[str] = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code filter (e.g. 'US', 'GB')"
    )


class _SonarAnswerArgs(BaseModel):
    query: str = Field(description="Question or topic for web-grounded AI answer")
    model: Optional[Literal["sonar", "sonar-pro"]] = Field(
        default="sonar",
        description="Sonar model: 'sonar' (cheaper, faster) or 'sonar-pro' (deeper). Default: sonar"
    )


class _PerplexityResearchArgs(BaseModel):
    query: str = Field(description="Research question or task requiring multi-step investigation")
    instructions: Optional[str] = Field(
        default=None,
        description="Optional system instructions to guide the research agent"
    )


# ── Plugin class ─────────────────────────────────────────────

class SearchPerplexityPlugin(BasePlugin):
    """Perplexity AI search plugin (search, sonar, agent)."""

    PLUGIN_NAME = "plugin_search_perplexity"
    PLUGIN_VERSION = "1.1.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Web search via Perplexity AI (raw search, sonar AI answers, agent research)"
    DEPENDENCIES = ["httpx"]
    ENV_VARS = ["PERPLEXITY_API_KEY"]

    def __init__(self):
        self.enabled = False
        self._api_key = None

    def init(self, config: dict) -> bool:
        import os
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            print("Perplexity plugin: PERPLEXITY_API_KEY not set in .env")
            return False
        self._api_key = api_key
        self.enabled = True
        return True

    def shutdown(self) -> None:
        self._api_key = None
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        api_key = self._api_key

        async def search_exec(query: str, max_results: int = 5, country: str = None) -> str:
            return await _run_perplexity_search(api_key, query, max_results, country)

        async def sonar_exec(query: str, model: str = "sonar") -> str:
            return await _run_perplexity_sonar(api_key, query, model)

        async def research_exec(query: str, instructions: str = None) -> str:
            return await _run_perplexity_agent(api_key, query, instructions=instructions)

        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=search_exec,
                    name="perplexity_search",
                    description=(
                        "Raw web search via Perplexity. Returns ranked results with "
                        "titles, URLs, and snippets. No AI summarization. "
                        "Trigger: prompt says 'perplexity search'."
                    ),
                    args_schema=_PerplexitySearchArgs,
                ),
                StructuredTool.from_function(
                    coroutine=sonar_exec,
                    name="sonar_answer",
                    description=(
                        "Web-grounded AI answer via Perplexity Sonar. Returns a "
                        "synthesized answer with inline citations from web sources. "
                        "Trigger: prompt says 'sonar', or escalating from a failed search."
                    ),
                    args_schema=_SonarAnswerArgs,
                ),
                StructuredTool.from_function(
                    coroutine=research_exec,
                    name="perplexity_research",
                    description=(
                        "Deep multi-step research via Perplexity Agent. Performs "
                        "multiple web searches and synthesizes a comprehensive report. "
                        "Trigger: prompt says 'research' for deep multi-source investigation."
                    ),
                    args_schema=_PerplexityResearchArgs,
                ),
            ]
        }


# ── Executors ────────────────────────────────────────────────

_BASE_URL = "https://api.perplexity.ai"


async def _run_perplexity_search(api_key: str, query: str,
                                  max_results: int = 5,
                                  country: str = None) -> str:
    """Raw web search via Perplexity Search API."""
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"query": query, "max_results": max_results}
    if country:
        payload["country"] = country

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{_BASE_URL}/search", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return f"Perplexity search: no results for '{query}'"

        lines = [f"Perplexity search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   URL: {r.get('url', '')}")
            snippet = r.get("snippet", "")
            if snippet:
                lines.append(f"   {snippet[:300]}")
            updated = r.get("last_updated", "")
            if updated:
                lines.append(f"   Updated: {updated}")
            lines.append("")
        return "\n".join(lines)

    except Exception as e:
        return f"Perplexity search error: {e}"


async def _run_perplexity_sonar(api_key: str, query: str,
                                 model: str = "sonar") -> str:
    """Web-grounded AI answer via Perplexity Sonar API."""
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "user": "llmem-gw",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{_BASE_URL}/v1/sonar", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        lines = [f"Perplexity Sonar ({model}) answer for: {query}\n"]

        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content:
                lines.append(content)
                lines.append("")

        citations = data.get("citations", [])
        if citations:
            lines.append("Citations:")
            for i, url in enumerate(citations, 1):
                lines.append(f"  {i}. {url}")
            lines.append("")

        usage = data.get("usage", {})
        cost = usage.get("cost", {})
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        if cost:
            total = float(cost.get("total_cost", 0) or 0)
            lines.append(f"Cost: ${total:.4f}")
            # Persist per-call cost event
            try:
                from cost_events import log_cost_event
                await log_cost_event(
                    provider="perplexity",
                    service=model,
                    tool_name="sonar_answer",
                    cost_usd=total,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    unit="tokens" if (tokens_in or tokens_out) else "api_call",
                    unit_count=None if (tokens_in or tokens_out) else 1,
                )
            except Exception:
                pass  # cost logging must never fail the call

        # Auto-archive: one llm-synthesis record per sonar query (not one per citation URL).
        # Storing N identical-summary rows per query bloats the knowledge graph with
        # duplicate nodes that all look the same.
        if citations or (choices and choices[0].get("message", {}).get("content")):
            try:
                import asyncio as _asyncio
                import hashlib
                from sources import _source_record_exec
                answer_content = ""
                if choices:
                    answer_content = choices[0].get("message", {}).get("content", "") or ""
                citations_clean = [u.strip() for u in citations if isinstance(u, str) and u.strip()]
                citations_block = ("\n\nCited sources:\n" + "\n".join(f"- {u}" for u in citations_clean)) if citations_clean else ""
                query_hash = hashlib.md5(f"{model}:{query}".encode()).hexdigest()[:16]
                _asyncio.create_task(_source_record_exec(
                    source_type="llm-synthesis",
                    source_ref=f"sonar-{model}://{query_hash}",
                    title=f"[sonar-{model}] {query[:120]}",
                    summary=answer_content[:250] if answer_content else f"Sonar {model} answer for: {query}",
                    content=answer_content + citations_block,
                    domain_tags="sonar_answer",
                    collection="sonar-synthesis",
                    hash_source="synthesized",
                ))
            except Exception:
                pass  # archiving must never fail the search

        return "\n".join(lines)

    except Exception as e:
        return f"Perplexity Sonar error: {e}"


async def _run_perplexity_agent(api_key: str, query: str,
                                 model: str = "openai/gpt-5.4",
                                 instructions: str = None) -> str:
    """Multi-step research via Perplexity Agent API."""
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": query,
        "tools": [{"type": "web_search"}],
        "user": "llmem-gw",
    }
    if instructions:
        payload["instructions"] = instructions

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{_BASE_URL}/v1/agent", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        lines = [f"Perplexity Agent research for: {query}\n"]

        # Capture first message text as summary snippet for archiving
        archive_summary = ""
        archive_urls: list = []

        for item in data.get("output", []):
            item_type = item.get("type", "")

            if item_type == "message":
                for content_block in item.get("content", []):
                    text = content_block.get("text", "")
                    if text:
                        lines.append(text)
                        lines.append("")
                        if not archive_summary:
                            archive_summary = text[:250]

            elif item_type == "search_results":
                queries = item.get("queries", [])
                if queries:
                    lines.append(f"Searches performed: {', '.join(queries)}")
                results = item.get("results", [])
                if results:
                    lines.append(f"Sources found: {len(results)}")
                    for r in results[:5]:
                        lines.append(f"  - {r.get('title', '')}: {r.get('url', '')}")
                    lines.append("")
                # Collect ALL search-result URLs for archiving
                for r in results:
                    u = (r.get("url") or "").strip()
                    t = (r.get("title") or "").strip()
                    if u:
                        archive_urls.append((u, t))

        usage = data.get("usage", {})
        cost = usage.get("cost", {})
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        if cost:
            total = float(cost.get("total_cost", 0) or 0)
            lines.append(f"Cost: ${total:.4f}")
            try:
                from cost_events import log_cost_event
                await log_cost_event(
                    provider="perplexity",
                    service=model,
                    tool_name="perplexity_research",
                    cost_usd=total,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    unit="tokens" if (tokens_in or tokens_out) else "api_call",
                    unit_count=None if (tokens_in or tokens_out) else 1,
                )
            except Exception:
                pass

        # Auto-archive search-result URLs (fire-and-forget)
        if archive_urls:
            try:
                import asyncio as _asyncio
                from sources import archive_from_search
                snippet = archive_summary or f"Cited in Perplexity Agent research for query: {query}"
                for url, title in archive_urls:
                    _asyncio.create_task(archive_from_search(
                        url=url,
                        title=title or f"[perplexity-research] {query[:120]}",
                        summary=snippet,
                        origin_tool="perplexity_research",
                    ))
            except Exception:
                pass

        return "\n".join(lines)

    except Exception as e:
        return f"Perplexity Agent error: {e}"
