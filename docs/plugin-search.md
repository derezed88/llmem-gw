# Plugins: Search Tools

Five search plugins are available. All are read-only.

## Tool access

Search tool access is controlled per-model via `llm_tools` in `llm-models.json`. Add the
search tool names to a model's tool list to grant access, or use `"all"` to include all tools.

```
!llm_tools read <model>                    show which tools a model can use
!llm_tools write <model> search_ddgs       grant DuckDuckGo only
!llm_tools write <model> search_ddgs,search_tavily,search_xai,search_google   grant all search
```

---

## plugin_search_ddgs — DuckDuckGo Search

**Tool:** `search_ddgs(query: str, max_results: int = 10) → str`

No API key required. Returns titles, URLs, and snippets. First choice in the PDDS search chain.

```bash
pip install ddgs
python llmemctl.py enable plugin_search_ddgs
```

---

## plugin_search_tavily — Tavily AI Search

**Tool:** `search_tavily(query: str, search_depth: str = "basic") → str`

AI-curated results with an answer summary. Use when DuckDuckGo results are insufficient.

`search_depth`: `"basic"` (faster) or `"advanced"` (more thorough)

```bash
pip install tavily-python
# .env: TAVILY_API_KEY=...
python llmemctl.py enable plugin_search_tavily
```

---

## plugin_search_xai — xAI Grok Search

**Tool:** `search_xai(query: str, model: str = "grok-4-1-fast-reasoning") → str`

Real-time web and X/Twitter search via xAI Grok. Use for current events and social media.

`model`: `"grok-4-1-fast-reasoning"` (default) or `"grok-4"` (full reasoning)

```bash
pip install xai-sdk
# .env: XAI_API_KEY=...
python llmemctl.py enable plugin_search_xai
```

---

## plugin_search_google — Google Search (Gemini Grounding)

**Tool:** `search_google(query: str) → str`

Google Search via Gemini grounding. Use when Gemini-grounded results are specifically needed.

```bash
pip install httpx
# .env: GEMINI_API_KEY=...
python llmemctl.py enable plugin_search_google
```

---

## plugin_urlextract_tavily — URL Content Extraction

**Tool:** `url_extract(method: str, url: str, query: str = "") → str`

Extracts full page content from a URL in markdown format. Optionally filter to content matching a query.

`method`: currently `"tavily"` only

```bash
pip install tavily-python
# .env: TAVILY_API_KEY=...
python llmemctl.py enable plugin_urlextract_tavily
```

Access controlled per-model via `llm_tools` — add `"url_extract"` to a model's tool list.

---

## plugin_search_perplexity — Perplexity AI Search

**Tools:**
- `perplexity_search(query: str) → str` — citation-backed web search via Perplexity
- `sonar_answer(query: str) → str` — direct Sonar model answer with inline citations

```bash
pip install openai
# .env: PERPLEXITY_API_KEY=...
python llmemctl.py enable plugin_search_perplexity
```
