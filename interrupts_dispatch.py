"""
interrupts_dispatch.py — Background loop that delivers high-priority interrupts
from samaritan_interrupts to the notifier (which routes to Slack via hook).

Polls every POLL_INTERVAL_S seconds for priority='high' AND delivered_at IS NULL
AND target_frontend IN ('any','slack'). For each row:
  1. Calls notifier.fire_event("interrupt", summary, detail) — the registered
     Slack delivery hook posts to the configured channel.
  2. UPDATE samaritan_interrupts SET delivered_at = NOW() WHERE id = N.

Medium/low interrupts are NOT delivered by this loop — they surface at
session start via the interrupts_pending MCP tool (read path in workflow-patterns.md).
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

log = logging.getLogger("interrupts_dispatch")

POLL_INTERVAL_S = 30

_stats: dict = {
    "polls_run": 0,
    "delivered": 0,
    "failed": 0,
    "last_poll_at": None,
    "last_error": None,
}


def get_stats() -> dict:
    return dict(_stats)


async def _fetch_pending_high() -> list[dict]:
    """Return list of undelivered high-priority interrupts as dicts."""
    from database import fetch_dicts, set_db_override
    try:
        set_db_override("mymcp")  # background loop has no request context
        rows = await fetch_dicts(
            "SELECT id, topic, content, priority, source, target_frontend "
            "FROM samaritan_interrupts "
            "WHERE priority = 'high' AND delivered_at IS NULL "
            "AND target_frontend IN ('any', 'slack') "
            "ORDER BY created_at ASC LIMIT 10"
        )
        return rows or []
    except Exception as e:
        log.warning(f"interrupts_dispatch: fetch failed: {e}")
        _stats["last_error"] = str(e)
        return []


async def _mark_delivered(interrupt_id: int) -> None:
    from database import execute_sql, set_db_override
    try:
        set_db_override("mymcp")
        await execute_sql(
            f"UPDATE samaritan_interrupts SET delivered_at = CURRENT_TIMESTAMP "
            f"WHERE id = {interrupt_id} AND delivered_at IS NULL"
        )
    except Exception as e:
        log.warning(f"interrupts_dispatch: mark_delivered failed for id={interrupt_id}: {e}")


async def _deliver_one(row: dict) -> bool:
    """Fire notifier event for one interrupt. Return True on success."""
    try:
        import notifier
        topic = row.get("topic", "")
        content = row.get("content", "")
        source = row.get("source", "")
        summary = f"[{source}] {topic}"
        # Truncate detail to fit Slack's usual limits
        detail = content[:2500] if content else ""
        await notifier.fire_event("interrupt", summary, detail)
        return True
    except Exception as e:
        log.warning(f"interrupts_dispatch: fire_event failed for id={row.get('id')}: {e}")
        _stats["last_error"] = str(e)
        return False


async def run_poll() -> dict:
    """Run one poll cycle. Safe to call manually for testing."""
    summary = {"fetched": 0, "delivered": 0, "failed": 0}
    rows = await _fetch_pending_high()
    summary["fetched"] = len(rows)
    for row in rows:
        ok = await _deliver_one(row)
        if ok:
            await _mark_delivered(int(row["id"]))
            summary["delivered"] += 1
            _stats["delivered"] += 1
            log.info(f"interrupts_dispatch: delivered id={row['id']} topic={row.get('topic','')}")
        else:
            summary["failed"] += 1
            _stats["failed"] += 1
    _stats["polls_run"] += 1
    _stats["last_poll_at"] = datetime.now(timezone.utc).isoformat()
    return summary


async def dispatch_task() -> None:
    """
    Long-running asyncio task. Polls POLL_INTERVAL_S seconds for pending
    high-priority interrupts and dispatches them via notifier.
    """
    log.info(f"interrupts_dispatch: starting, poll interval {POLL_INTERVAL_S}s")
    # Startup delay — let llmem-gw finish initializing
    await asyncio.sleep(15)
    while True:
        try:
            await run_poll()
        except Exception as e:
            log.warning(f"interrupts_dispatch: unhandled error: {e}")
            _stats["last_error"] = str(e)
        await asyncio.sleep(POLL_INTERVAL_S)
