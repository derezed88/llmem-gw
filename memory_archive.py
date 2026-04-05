"""
memory_archive.py — LT → Google Drive archival for the tiered memory system.

When a domain's long-term memory table exceeds its configured row threshold,
the lowest-priority rows are moved to Google Drive. Qdrant is updated in-place:
the point keeps its ID but gets tier="archive" and source="drive" in its payload,
plus the Drive file ID for re-injection. Normal ST/LT searches filter by tier
so archived points are invisible unless explicitly queried.

Architecture:
  MySQL LT (bounded)  ←→  active recall, aging from ST
  Google Drive        ←→  subconscious: vast, dormant until triggered
  Qdrant              ←→  search bridge across all tiers; tier field routes retrieval

Archive flow:
  MySQL LT row → JSON file in Drive → Qdrant payload: {tier=archive, source=drive, drive_file_id}
  → MySQL row deleted

Re-injection flow (Phase 2):
  Qdrant search hit with tier=archive → fetch JSON from Drive → INSERT into LT → upsert new Qdrant
  point → delete old archive point

Configuration is read from db-config.json under tables.<db_key>.archival:
  enabled              bool   — skip if false
  drive_root_folder_id str    — root Drive folder (from .env FOLDER_ID as default)
  drive_subfolder      str    — subfolder name created under root (e.g. "mymcp")
  lt_row_threshold     int    — archive only when LT row count exceeds this
  archive_batch_size   int    — how many rows to archive per run
  archive_score_expr   str    — SQL expression for ranking (lower = archive first)
  min_importance_floor int    — rows at or above this importance are never archived
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("memory_archive")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DB_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db-config.json")

_ARCHIVAL_DEFAULTS = {
    "enabled": False,
    "drive_root_folder_id": "",
    "drive_subfolder": "",
    "lt_row_threshold": 5000,
    "archive_batch_size": 500,
    "archive_score_expr": "importance * (access_count + 1)",
    "min_importance_floor": 9,
}


def _load_archival_cfg(db_key: str) -> dict | None:
    """Return the archival config for db_key, or None if not enabled."""
    try:
        with open(_DB_CONFIG_PATH) as fh:
            db_cfg = json.load(fh)
        domain = db_cfg.get("tables", {}).get(db_key, {})
        archival = domain.get("archival")
        if not archival:
            return None
        cfg = {**_ARCHIVAL_DEFAULTS, **archival}
        if not cfg.get("enabled"):
            return None
        return cfg
    except Exception as e:
        log.warning("_load_archival_cfg(%r) failed: %s", db_key, e)
        return None


# ---------------------------------------------------------------------------
# Drive folder management
# ---------------------------------------------------------------------------

async def _ensure_archive_folder(root_folder_id: str, subfolder_name: str) -> str:
    """Return the Drive folder ID for archival, creating the subfolder if needed."""
    from drive import _drive_get_or_create_folder
    return await asyncio.to_thread(_drive_get_or_create_folder, subfolder_name, root_folder_id)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

async def _select_archive_candidates(
    lt_table: str,
    score_expr: str,
    min_importance_floor: int,
    batch_size: int,
) -> list[dict]:
    """Return lowest-priority LT rows ranked by score_expr ASC."""
    from database import fetch_dicts
    sql = (
        f"SELECT id, topic, content, importance, source, type, "
        f"created_at, aged_at, access_count "
        f"FROM {lt_table} "
        f"WHERE importance < {int(min_importance_floor)} "
        f"ORDER BY {score_expr} ASC "
        f"LIMIT {int(batch_size)}"
    )
    try:
        return await fetch_dicts(sql)
    except Exception as e:
        log.error("_select_archive_candidates failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Core archive operation
# ---------------------------------------------------------------------------

async def _archive_row(
    row: dict,
    lt_table: str,
    drive_folder_id: str,
    vec_api,
    collection: str,
) -> bool:
    """
    Archive a single LT row:
      1. Upload JSON to Drive
      2. Update Qdrant payload: tier=archive, source=drive, drive_file_id=<id>
      3. Delete MySQL row

    Returns True on success.
    """
    from drive import _drive_create_file
    from database import execute_sql as _exec

    row_id = row.get("id")
    topic = row.get("topic", "unknown")

    # 1. Build Drive file content
    payload = {
        "id": row_id,
        "topic": topic,
        "content": row.get("content", ""),
        "importance": row.get("importance", 5),
        "source": row.get("source", "session"),
        "type": row.get("type", "context"),
        "created_at": str(row.get("created_at", "")),
        "aged_at": str(row.get("aged_at", "")),
        "access_count": row.get("access_count", 0),
        "archived_at": datetime.now(timezone.utc).isoformat(),
    }
    file_name = f"mem_{row_id}_{topic[:40].replace('/', '_')}.json"
    file_content = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        result = await asyncio.to_thread(
            _drive_create_file, file_name, file_content, drive_folder_id
        )
        # result = "Created '<name>' — id: <file_id>"
        drive_file_id = result.split("id: ")[-1].strip()
    except Exception as e:
        log.warning("archive row %s: Drive upload failed: %s", row_id, e)
        return False

    # 2. Update Qdrant payload (keep point ID, update tier+source+drive_file_id)
    if vec_api:
        try:
            await vec_api.set_payload(
                row_id=int(row_id),
                payload={
                    "tier": "archive",
                    "source": "drive",
                    "drive_file_id": drive_file_id,
                },
                collection=collection,
            )
        except Exception as e:
            log.warning("archive row %s: Qdrant payload update failed: %s", row_id, e)
            # Non-fatal: Drive file is uploaded; Qdrant can be backfilled later

    # 3. Delete from MySQL LT
    try:
        await _exec(f"DELETE FROM {lt_table} WHERE id = {int(row_id)}")
    except Exception as e:
        log.error("archive row %s: MySQL delete failed: %s", row_id, e)
        return False

    log.debug("archived row %s (topic=%r) → Drive file %s", row_id, topic, drive_file_id)
    return True


# ---------------------------------------------------------------------------
# Batch archival
# ---------------------------------------------------------------------------

async def _archive_batch(
    rows: list[dict],
    lt_table: str,
    drive_folder_id: str,
    vec_api,
    collection: str,
) -> dict:
    """Archive a list of candidate rows. Returns {archived, failed}."""
    archived = 0
    failed = 0
    for row in rows:
        ok = await _archive_row(row, lt_table, drive_folder_id, vec_api, collection)
        if ok:
            archived += 1
        else:
            failed += 1
    return {"archived": archived, "failed": failed}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def check_and_archive(db_key: str = "mymcp") -> dict:
    """
    Check if the LT table for db_key needs archival and run a batch if so.

    Returns a result dict:
      {status, lt_count, threshold, archived, failed, skipped_reason}
    """
    cfg = _load_archival_cfg(db_key)
    if cfg is None:
        return {"status": "skipped", "skipped_reason": "archival not enabled for this domain"}

    lt_table = _get_lt_table(db_key)
    if not lt_table:
        return {"status": "error", "skipped_reason": f"no lt table found for db_key={db_key!r}"}

    collection = _get_collection(db_key)

    # Check current LT count
    from database import execute_sql as _exec
    try:
        count_raw = await _exec(f"SELECT COUNT(*) as cnt FROM {lt_table}")
        lt_count = _parse_count(count_raw)
    except Exception as e:
        return {"status": "error", "skipped_reason": f"COUNT failed: {e}"}

    threshold = cfg["lt_row_threshold"]
    if lt_count <= threshold:
        return {
            "status": "ok",
            "lt_count": lt_count,
            "threshold": threshold,
            "archived": 0,
            "skipped_reason": "below threshold",
        }

    log.info("check_and_archive(%r): lt_count=%d > threshold=%d — archiving batch", db_key, lt_count, threshold)

    # Ensure Drive folder exists
    root_folder_id = cfg.get("drive_root_folder_id", "")
    subfolder_name = cfg.get("drive_subfolder", db_key)
    try:
        drive_folder_id = await _ensure_archive_folder(root_folder_id, subfolder_name)
    except Exception as e:
        return {"status": "error", "skipped_reason": f"Drive folder setup failed: {e}"}

    # Select candidates
    candidates = await _select_archive_candidates(
        lt_table=lt_table,
        score_expr=cfg["archive_score_expr"],
        min_importance_floor=cfg["min_importance_floor"],
        batch_size=cfg["archive_batch_size"],
    )
    if not candidates:
        return {
            "status": "ok",
            "lt_count": lt_count,
            "threshold": threshold,
            "archived": 0,
            "skipped_reason": "no archivable candidates (all at or above importance floor)",
        }

    # Get Qdrant API if available
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec_api = get_vector_api()
    except Exception:
        vec_api = None

    result = await _archive_batch(candidates, lt_table, drive_folder_id, vec_api, collection)
    log.info(
        "check_and_archive(%r): archived=%d failed=%d (lt_before=%d)",
        db_key, result["archived"], result["failed"], lt_count,
    )
    return {
        "status": "done",
        "lt_count": lt_count,
        "threshold": threshold,
        **result,
    }


# ---------------------------------------------------------------------------
# Re-injection stub (Phase 2)
# ---------------------------------------------------------------------------

async def recall_from_archive(drive_file_id: str, db_key: str = "mymcp") -> dict:
    """
    Re-inject an archived memory from Drive back into LT.

    Fetches the JSON file from Drive, inserts a new LT row, upserts a new
    Qdrant point, and deletes the old archive point.

    Returns {status, new_lt_id} or {status, error}.
    """
    # Phase 2 — not yet implemented
    return {"status": "not_implemented", "error": "recall_from_archive is Phase 2"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_lt_table(db_key: str) -> str | None:
    try:
        with open(_DB_CONFIG_PATH) as fh:
            db_cfg = json.load(fh)
        return db_cfg.get("tables", {}).get(db_key, {}).get("memory_longterm")
    except Exception:
        return None


def _get_collection(db_key: str) -> str:
    try:
        with open(_DB_CONFIG_PATH) as fh:
            db_cfg = json.load(fh)
        return db_cfg.get("tables", {}).get(db_key, {}).get("collection", "samaritan_memory")
    except Exception:
        return "samaritan_memory"


def _parse_count(raw: str) -> int:
    for line in raw.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return 0
