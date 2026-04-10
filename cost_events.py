"""
Per-call cost event logger for samaritan_cost_events table.

Usage (async context):
    from cost_events import log_cost_event
    await log_cost_event(provider="tavily", service="search-api", tool_name="search_tavily", cost_usd=0.005)

Usage (sync context):
    from cost_events import log_cost_event_sync
    log_cost_event_sync(provider="tavily", service="search-api", cost_usd=0.005)

The daily aggregator (cost_aggregator.py) sums these events via fetch_cost_events().
"""

import asyncio
import logging
import datetime
from typing import Optional

log = logging.getLogger("AISvc")

# Pricing constants — shared with cost_aggregator.py
# Per 1M tokens (input, output)
GEMINI_PRICING = {
    "gemini-2.5-flash":       {"input": 0.50,  "output": 3.00},
    "gemini-2.5-flash-lite":  {"input": 0.10,  "output": 0.40},
    "gemini-2.5-pro":         {"input": 1.25,  "output": 10.00},
    "gemini-2.0-flash":       {"input": 0.10,  "output": 0.40},
    "gemini-3.1-pro-preview":  {"input": 2.00,  "output": 15.00},
}

# xAI Grok pricing per 1M tokens (from xai.com/api pricing, updated March 2026)
XAI_PRICING = {
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4":                  {"input": 2.00, "output": 10.00},
    "grok-4-5":                {"input": 2.00, "output": 10.00},
}

# FriendliAI pricing per 1M tokens
FRIENDLI_PRICING = {
    "qwen3-235b": {"input": 0.20, "output": 0.80},
}

# Tavily pricing: $0.005 per basic search, $0.01 per advanced search (Researcher plan)
TAVILY_PRICING = {
    "basic":    0.005,
    "advanced": 0.010,
}

# Bright Data SERP: ~$0.001 per request (estimate — actual billed by GB)
BRIGHTDATA_PRICING = {
    "serp": 0.001,
}


def _get_db_conn():
    """Get a MySQL connection using the same credentials as database.py."""
    import os
    import mysql.connector
    from pathlib import Path

    # Load .env if not already loaded
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

    return mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
        database="mymcp",
    )


def log_cost_event_sync(
    provider: str,
    service: str,
    cost_usd: float,
    tool_name: Optional[str] = None,
    model_key: Optional[str] = None,
    client_id: Optional[str] = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    unit: Optional[str] = None,
    unit_count: Optional[float] = None,
    notes: Optional[str] = None,
) -> None:
    """Synchronous version — use in sync contexts or when already in executor."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO samaritan_cost_events
                (provider, service, tool_name, model_key, client_id,
                 tokens_in, tokens_out, cost_usd, unit, unit_count, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (provider, service, tool_name, model_key, client_id,
             tokens_in, tokens_out, cost_usd, unit, unit_count, notes),
        )
        conn.commit()
    except Exception as e:
        log.debug(f"cost_events: log_cost_event_sync failed: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


async def log_cost_event(
    provider: str,
    service: str,
    cost_usd: float,
    tool_name: Optional[str] = None,
    model_key: Optional[str] = None,
    client_id: Optional[str] = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    unit: Optional[str] = None,
    unit_count: Optional[float] = None,
    notes: Optional[str] = None,
) -> None:
    """Async fire-and-forget cost event logger. Never raises."""
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: log_cost_event_sync(
                provider=provider,
                service=service,
                cost_usd=cost_usd,
                tool_name=tool_name,
                model_key=model_key,
                client_id=client_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                unit=unit,
                unit_count=unit_count,
                notes=notes,
            )
        )
    except Exception as e:
        log.debug(f"cost_events: log_cost_event async wrapper failed: {e}")


def estimate_gemini_cost(model_name: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate Gemini API cost given model and token counts."""
    # Normalize model name to a pricing key
    for key in GEMINI_PRICING:
        if key in model_name.lower():
            p = GEMINI_PRICING[key]
            return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000
    # Fallback: flash pricing
    p = GEMINI_PRICING["gemini-2.5-flash"]
    return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000


def estimate_xai_cost(model_name: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate xAI/Grok API cost given model and token counts."""
    for key in XAI_PRICING:
        if key in model_name.lower():
            p = XAI_PRICING[key]
            return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000
    # Fallback: fast-reasoning pricing
    p = XAI_PRICING["grok-4-1-fast-reasoning"]
    return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000


def estimate_friendli_cost(model_name: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate FriendliAI cost given model and token counts."""
    for key in FRIENDLI_PRICING:
        if key in model_name.lower():
            p = FRIENDLI_PRICING[key]
            return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000
    # Fallback: qwen3-235b pricing
    p = FRIENDLI_PRICING["qwen3-235b"]
    return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000


def _estimate_cost_for_model(cfg: dict, tokens_in: int, tokens_out: int) -> Optional[float]:
    """Estimate cost for any model given its config dict. Returns None if unknown provider."""
    model_id = cfg.get("model_id", "")
    model_type = cfg.get("type", "")
    host = cfg.get("host", "")
    if model_type == "GEMINI":
        return estimate_gemini_cost(model_id, tokens_in, tokens_out)
    if model_type == "OPENAI" and ("xai" in host.lower() or "grok" in model_id.lower()):
        return estimate_xai_cost(model_id, tokens_in, tokens_out)
    if model_type == "OPENAI" and "friendli" in host.lower():
        return estimate_friendli_cost(model_id, tokens_in, tokens_out)
    return None


def fetch_events_for_date(date) -> list:
    """
    Pull all cost events for a given date. Used by cost_aggregator.py.
    Returns list of dicts: {provider, service, calls, tokens_in, tokens_out, cost_usd}.
    """
    try:
        conn = _get_db_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT
                provider,
                service,
                COUNT(*) AS calls,
                COALESCE(SUM(tokens_in), 0) AS tokens_in,
                COALESCE(SUM(tokens_out), 0) AS tokens_out,
                COALESCE(SUM(cost_usd), 0) AS cost_usd
            FROM samaritan_cost_events
            WHERE DATE(ts) = %s
            GROUP BY provider, service
            """,
            (str(date),),
        )
        return cur.fetchall()
    except Exception as e:
        log.warning(f"cost_events: fetch_events_for_date failed: {e}")
        return []
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass
