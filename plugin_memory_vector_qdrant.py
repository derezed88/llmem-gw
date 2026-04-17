"""
plugin_memory_vector_qdrant.py — Vector search plugin for the tiered memory system.

Wraps Qdrant + a local embedding endpoint (nomic-embed-text via llama.cpp).
MySQL remains the source of truth; Qdrant is the retrieval index.

This plugin registers no LangChain tools — it is an infrastructure plugin
that exposes its API directly to memory.py via get_vector_api().

Public API (accessed via get_vector_api()):
    embed(text, prefix)                                      -> list[float]
    upsert_memory(row_id, topic, content, importance, tier)
    search_memories(query_text, top_k, min_score, tier)      -> list[dict]
    delete_memory(row_id)
    update_tier(row_id, new_tier)
    backfill(rows, tier)                                     -> int

Configuration (plugins-enabled.json → plugin_config.plugin_memory_vector_qdrant):
    enabled:                  true
    qdrant_host:              "192.168.10.101"
    qdrant_port:              6333
    embed_url:                "http://192.168.10.101:8000/v1/embeddings"
    embed_model:              "nomic-embed-text"
    collection:               "samaritan_memory"
    vector_dims:              768
    top_k:                    20
    min_score:                0.45
    min_importance_always:    8        # rows at or above this importance always injected
"""

import logging
import time
from typing import Dict, Any, List

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, VectorParams,
    Filter, FieldCondition, MatchValue,
)
from plugin_loader import BasePlugin

log = logging.getLogger("plugin_memory_vector_qdrant")

# ---------------------------------------------------------------------------
# Module-level singleton — set by plugin.init(), read by get_vector_api()
# ---------------------------------------------------------------------------
_INSTANCE: "QdrantVectorPlugin | None" = None


def get_vector_api() -> "QdrantVectorPlugin | None":
    """Return the initialised plugin instance, or None if not loaded/enabled.

    Also checks plugin_config.memory.vector_search_qdrant (default True) so
    Qdrant semantic search can be toggled live without restarting the server,
    alongside context_injection / reset_summarize / post_response_scan.
    """
    if not (_INSTANCE and _INSTANCE.enabled):
        return None
    try:
        import json as _json, os as _os
        path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "plugins-enabled.json")
        with open(path) as fh:
            mem_cfg = _json.load(fh).get("plugin_config", {}).get("memory", {})
        if not mem_cfg.get("vector_search_qdrant", True):
            return None
    except Exception:
        pass  # if config unreadable, default to enabled
    return _INSTANCE


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class QdrantVectorPlugin(BasePlugin):
    """Infrastructure plugin: Qdrant vector store + nomic-embed-text embeddings."""

    PLUGIN_NAME    = "plugin_memory_vector_qdrant"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE    = "data_tool"
    DESCRIPTION    = "Qdrant vector search + nomic-embed-text for semantic memory retrieval"
    DEPENDENCIES   = ["qdrant-client>=1.7", "httpx>=0.24"]
    ENV_VARS: list = []

    # Retry backoff: wait at least this many seconds between reconnect attempts
    _RECONNECT_BACKOFF = 30
    _RECONNECT_MAX_BACKOFF = 300  # cap at 5 minutes

    def __init__(self):
        self.enabled      = False
        self._cfg: dict   = {}
        self._qc: QdrantClient | None = None
        self._last_connect_attempt: float = 0.0
        self._backoff: float = self._RECONNECT_BACKOFF

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, config: dict) -> bool:
        global _INSTANCE
        self._cfg = {
            "qdrant_host":           config.get("qdrant_host",           "192.168.10.101"),
            "qdrant_port":           config.get("qdrant_port",           6333),
            "embed_url":             config.get("embed_url",             "http://192.168.10.101:8000/v1/embeddings"),
            "embed_model":           config.get("embed_model",           "nomic-embed-text"),
            "collection":            config.get("collection",            "samaritan_memory"),
            "proc_collection":       config.get("proc_collection",       "samaritan_procedures"),
            "vector_dims":           config.get("vector_dims",           768),
            "top_k":                 config.get("top_k",                 20),
            "min_score":             config.get("min_score",             0.45),
            "min_importance_always": config.get("min_importance_always", 8),
        }
        # Always mark enabled and set instance — Qdrant connection is lazy
        self.enabled = True
        _INSTANCE = self
        # Attempt initial connection (non-fatal if NUC11 is down)
        self._try_connect()
        return True

    def _try_connect(self) -> bool:
        """Attempt to connect to Qdrant. Returns True if connected."""
        now = time.monotonic()
        if self._qc is not None:
            return True
        if now - self._last_connect_attempt < self._backoff:
            return False
        self._last_connect_attempt = now
        try:
            qc = QdrantClient(
                host=self._cfg["qdrant_host"],
                port=self._cfg["qdrant_port"],
                timeout=10,
            )
            # Verify connectivity by listing collections
            existing = [c.name for c in qc.get_collections().collections]
            if self._cfg["collection"] not in existing:
                qc.create_collection(
                    collection_name=self._cfg["collection"],
                    vectors_config=VectorParams(
                        size=self._cfg["vector_dims"],
                        distance=Distance.COSINE,
                    ),
                )
                log.info(f"Created Qdrant collection '{self._cfg['collection']}'")
            else:
                log.info(f"Qdrant collection '{self._cfg['collection']}' ready")
            self._qc = qc
            self._backoff = self._RECONNECT_BACKOFF  # reset backoff on success
            log.info("QdrantVectorPlugin connected to %s:%s",
                     self._cfg["qdrant_host"], self._cfg["qdrant_port"])
            return True
        except Exception as e:
            # Increase backoff (capped) for next attempt
            self._backoff = min(self._backoff * 2, self._RECONNECT_MAX_BACKOFF)
            log.warning("QdrantVectorPlugin connect failed (retry in %.0fs): %s",
                        self._backoff, e)
            return False

    def _connected(self) -> bool:
        """Check if connected, attempt reconnect if not. Use before any Qdrant op."""
        if self._qc is not None:
            return True
        return self._try_connect()

    def _handle_connection_error(self, e: Exception) -> None:
        """If the error looks like a connection failure, drop _qc so next call reconnects."""
        err_str = str(e).lower()
        if any(kw in err_str for kw in ("connect", "refused", "timeout", "unreachable", "reset")):
            log.warning("Connection to Qdrant lost, will reconnect on next call: %s", e)
            self._qc = None

    def shutdown(self) -> None:
        global _INSTANCE
        self.enabled = False
        self._qc = None
        _INSTANCE = None

    def get_tools(self) -> Dict[str, Any]:
        return {"lc": []}

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def embed(self, text: str, prefix: str = "search_document") -> list[float]:
        """
        Embed text via the local nomic-embed-text llama.cpp server.
        prefix: "search_document" for storage, "search_query" for retrieval.
        """
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                self._cfg["embed_url"],
                json={"input": f"{prefix}: {text}", "model": self._cfg["embed_model"]},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    # ------------------------------------------------------------------
    # Upsert / delete / update
    # ------------------------------------------------------------------

    def _resolve_collection(self, collection: str | None) -> str:
        return collection if collection else self._cfg["collection"]

    def _ensure_collection(self, collection: str) -> None:
        """Create the Qdrant collection if it does not already exist."""
        if not self._connected():
            return
        try:
            existing = [c.name for c in self._qc.get_collections().collections]
            if collection not in existing:
                self._qc.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self._cfg["vector_dims"],
                        distance=Distance.COSINE,
                    ),
                )
                log.info(f"Created Qdrant collection '{collection}'")
        except Exception as e:
            self._handle_connection_error(e)
            log.warning(f"_ensure_collection({collection!r}) failed: {e}")

    def proc_collection(self) -> str:
        """Return the configured procedures collection name."""
        return self._cfg.get("proc_collection", "samaritan_procedures")

    def _ensure_proc_collection(self) -> None:
        """Create the procedures Qdrant collection with task_type payload index if needed."""
        if not self._connected():
            return
        coll = self.proc_collection()
        try:
            existing = [c.name for c in self._qc.get_collections().collections]
            if coll not in existing:
                self._qc.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(
                        size=self._cfg["vector_dims"],
                        distance=Distance.COSINE,
                    ),
                )
                log.info(f"Created Qdrant procedures collection '{coll}'")
                # Add payload index on task_type for filtered queries
                try:
                    from qdrant_client.models import PayloadSchemaType
                    self._qc.create_payload_index(
                        collection_name=coll,
                        field_name="task_type",
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                    log.info(f"Created task_type payload index on '{coll}'")
                except Exception as idx_e:
                    log.warning(f"_ensure_proc_collection: payload index failed: {idx_e}")
        except Exception as e:
            log.warning(f"_ensure_proc_collection failed: {e}")

    def ensure_collections(self, collection: str, proc_collection: str) -> list[str]:
        """Ensure both memory and procedures collections exist. Returns list of created names."""
        if not self._connected():
            return []
        created = []
        for coll, with_proc_index in [(collection, False), (proc_collection, True)]:
            try:
                existing = [c.name for c in self._qc.get_collections().collections]
                if coll not in existing:
                    self._qc.create_collection(
                        collection_name=coll,
                        vectors_config=VectorParams(
                            size=self._cfg["vector_dims"],
                            distance=Distance.COSINE,
                        ),
                    )
                    created.append(coll)
                    log.info(f"Created Qdrant collection '{coll}'")
                    if with_proc_index:
                        try:
                            from qdrant_client.models import PayloadSchemaType
                            self._qc.create_payload_index(
                                collection_name=coll,
                                field_name="task_type",
                                field_schema=PayloadSchemaType.KEYWORD,
                            )
                        except Exception as idx_e:
                            log.warning(f"ensure_collections: payload index on '{coll}' failed: {idx_e}")
            except Exception as e:
                log.warning(f"ensure_collections({coll!r}) failed: {e}")
        return created

    def delete_collections(self, collection: str, proc_collection: str) -> list[str]:
        """Delete both memory and procedures Qdrant collections. Returns list of deleted names."""
        if not self._connected():
            return []
        deleted = []
        for coll in (collection, proc_collection):
            try:
                existing = [c.name for c in self._qc.get_collections().collections]
                if coll in existing:
                    self._qc.delete_collection(collection_name=coll)
                    deleted.append(coll)
                    log.info(f"Deleted Qdrant collection '{coll}'")
            except Exception as e:
                log.warning(f"delete_collections({coll!r}) failed: {e}")
        return deleted

    async def set_payload(self, row_id: int, payload: dict, collection: str | None = None) -> None:
        """Update payload fields on an existing Qdrant point without re-embedding."""
        if not self._connected():
            return
        coll = self._resolve_collection(collection)
        try:
            self._qc.set_payload(
                collection_name=coll,
                payload=payload,
                points=[row_id],
            )
        except Exception as e:
            self._handle_connection_error(e)
            log.warning(f"set_payload failed (id={row_id}): {e}")

    async def upsert_memory(
        self,
        row_id: int,
        topic: str,
        content: str,
        importance: int,
        tier: str = "short",
        collection: str | None = None,
    ) -> None:
        """Embed content and upsert a point into Qdrant using MySQL row_id as point ID."""
        if not self._connected():
            log.debug(f"upsert_memory skipped (id={row_id}): Qdrant not connected")
            return
        coll = self._resolve_collection(collection)
        try:
            vector = await self.embed(content, prefix="search_document")
            self._ensure_collection(coll)
            self._qc.upsert(
                collection_name=coll,
                points=[PointStruct(
                    id=row_id,
                    vector=vector,
                    payload={
                        "topic":      topic,
                        "content":    content,
                        "importance": importance,
                        "tier":       tier,
                    },
                )],
            )
            log.debug(f"upsert_memory: id={row_id} topic={topic!r} tier={tier}")
        except Exception as e:
            self._handle_connection_error(e)
            log.warning(f"upsert_memory failed (id={row_id}): {e}")

    async def delete_memory(self, row_id: int, collection: str | None = None) -> None:
        """Remove a point from Qdrant by MySQL row_id."""
        if not self._connected():
            return
        coll = self._resolve_collection(collection)
        try:
            self._qc.delete(
                collection_name=coll,
                points_selector=[row_id],
            )
        except Exception as e:
            self._handle_connection_error(e)
            log.warning(f"delete_memory failed (id={row_id}): {e}")

    async def update_tier(self, row_id: int, new_tier: str, collection: str | None = None) -> None:
        """Update the tier payload field when a row ages short→long."""
        if not self._connected():
            return
        coll = self._resolve_collection(collection)
        try:
            self._qc.set_payload(
                collection_name=coll,
                payload={"tier": new_tier},
                points=[row_id],
            )
        except Exception as e:
            self._handle_connection_error(e)
            log.warning(f"update_tier failed (id={row_id}): {e}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_memories(
        self,
        query_text: str,
        top_k: int | None = None,
        min_score: float | None = None,
        tier: str = "short",
        collection: str | None = None,
    ) -> list[dict]:
        """
        Semantic search over Qdrant filtered by tier.
        Returns list of dicts: id, topic, content, importance, score.
        """
        coll = self._resolve_collection(collection)
        if top_k is None:
            top_k = self._cfg["top_k"]
        if min_score is None:
            min_score = self._cfg["min_score"]
        if not self._connected():
            return []
        try:
            vector = await self.embed(query_text, prefix="search_query")
            response = self._qc.query_points(
                collection_name=coll,
                query=vector,
                query_filter=Filter(
                    must=[FieldCondition(key="tier", match=MatchValue(value=tier))]
                ),
                limit=top_k,
                score_threshold=min_score,
                with_payload=True,
            )
            return [
                {
                    "id":         r.id,
                    "topic":      r.payload.get("topic", ""),
                    "content":    r.payload.get("content", ""),
                    "importance": r.payload.get("importance", 5),
                    "score":      round(r.score, 4),
                }
                for r in response.points
            ]
        except Exception as e:
            self._handle_connection_error(e)
            log.warning(f"search_memories failed: {e}")
            return []

    async def search_presence_triggers(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.55,
        collection: str = "samaritan_presence_triggers",
    ) -> list[dict]:
        """
        Semantic search against the presence trigger collection.

        Unlike search_memories, this does NOT filter by a tier payload — points
        in samaritan_presence_triggers have no tier field. Returns full trigger
        payload so the audit handler has enough context to decide whether the
        match warrants a pre-commit check.
        """
        if not self._connected():
            return []
        try:
            vector = await self.embed(query_text, prefix="search_query")
            response = self._qc.query_points(
                collection_name=collection,
                query=vector,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True,
            )
            return [
                {
                    "id":              r.id,
                    "trigger_type":    r.payload.get("trigger_type", ""),
                    "trigger_pattern": r.payload.get("trigger_pattern", ""),
                    "check_action":    r.payload.get("check_action", ""),
                    "rationale":       r.payload.get("rationale", ""),
                    "matcher":         r.payload.get("matcher", ""),
                    "importance":      r.payload.get("importance", 5),
                    "score":           round(r.score, 4),
                }
                for r in response.points
            ]
        except Exception as e:
            self._handle_connection_error(e)
            log.warning(f"search_presence_triggers failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    async def backfill(self, rows: list[dict], tier: str = "short", collection: str | None = None) -> int:
        """
        Embed and upsert a list of MySQL rows into Qdrant.
        rows: list of dicts with keys id, topic, content, importance.
        Returns count of successfully upserted rows.
        """
        coll = self._resolve_collection(collection)
        saved = 0
        for row in rows:
            row_id = row.get("id")
            if not row_id:
                continue
            try:
                await self.upsert_memory(
                    row_id=int(row_id),
                    topic=row.get("topic", ""),
                    content=row.get("content", ""),
                    importance=int(row.get("importance", 5)),
                    tier=tier,
                    collection=coll,
                )
                saved += 1
            except Exception as e:
                log.warning(f"backfill failed row id={row_id}: {e}")
        log.info(f"backfill: upserted {saved}/{len(rows)} rows (tier={tier}, collection={coll!r})")
        return saved

    def get_all_point_ids(self, collection: str | None = None) -> set[int]:
        """Return all point IDs currently in Qdrant (scrolls all pages)."""
        if not self._connected():
            return set()
        coll = self._resolve_collection(collection)
        ids: set[int] = set()
        offset = None
        while True:
            result, next_offset = self._qc.scroll(
                collection_name=coll,
                offset=offset,
                limit=1000,
                with_payload=False,
                with_vectors=False,
            )
            for p in result:
                ids.add(p.id)
            if next_offset is None:
                break
            offset = next_offset
        return ids

    # ------------------------------------------------------------------
    # Config snapshot (for !memstats etc.)
    # ------------------------------------------------------------------

    def cfg(self) -> dict:
        return dict(self._cfg)

    async def get_stats(self, collection: str | None = None) -> dict:
        """
        Collect health/count stats from Qdrant and the embedding server.
        Returns a dict with keys: qdrant, embed.
        All fields are strings or ints; errors are reported as strings.
        """
        coll = self._resolve_collection(collection)
        stats: dict = {"qdrant": {}, "embed": {}}

        # ── Qdrant collection info ────────────────────────────────────────
        if not self._connected():
            stats["qdrant"]["error"] = "not connected (will retry)"
            # Still check embed health below
        else:
            try:
                info = self._qc.get_collection(coll)
                stats["qdrant"] = {
                    "status":                str(info.status.value if hasattr(info.status, "value") else info.status),
                    "points_count":          info.points_count or 0,
                    "indexed_vectors_count": info.indexed_vectors_count or 0,
                    "segments_count":        info.segments_count or 0,
                    "optimizer_status":      str(info.optimizer_status.ok if hasattr(info.optimizer_status, "ok") else info.optimizer_status),
                }
            except Exception as e:
                self._handle_connection_error(e)
                stats["qdrant"]["error"] = str(e)

        # ── Embedding server health + metrics ─────────────────────────────
        base = self._cfg["embed_url"].rsplit("/v1/", 1)[0]
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                health_resp = await client.get(f"{base}/health")
                stats["embed"]["health"] = "ok" if health_resp.status_code == 200 else f"HTTP {health_resp.status_code}"
        except Exception as e:
            stats["embed"]["health"] = f"unreachable: {e}"

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                metrics_resp = await client.get(f"{base}/metrics")
            if metrics_resp.status_code == 200:
                raw = metrics_resp.text
                def _prom_val(name: str) -> str | None:
                    for line in raw.splitlines():
                        if line.startswith(name + " ") or line.startswith(name + "{"):
                            parts = line.rsplit(" ", 1)
                            return parts[-1].strip() if len(parts) >= 2 else None
                    return None

                kv_ratio = _prom_val("llamacpp_kv_cache_usage_ratio")
                kv_tokens = _prom_val("llamacpp_kv_cache_tokens")
                proc      = _prom_val("llamacpp_requests_processing")
                deferred  = _prom_val("llamacpp_requests_deferred")
                tps       = _prom_val("llamacpp_predicted_tokens_seconds")
                embed_tps = _prom_val("llamacpp_prompt_tokens_seconds")

                if kv_ratio is not None:
                    try:
                        stats["embed"]["kv_cache_pct"] = f"{float(kv_ratio)*100:.1f}%"
                    except ValueError:
                        pass
                if kv_tokens is not None:
                    stats["embed"]["kv_cache_tokens"] = kv_tokens
                if proc is not None:
                    stats["embed"]["requests_processing"] = proc
                if deferred is not None:
                    stats["embed"]["requests_deferred"] = deferred
                if embed_tps is not None:
                    stats["embed"]["embed_tps"] = f"{float(embed_tps):.0f} tok/s"
                elif tps is not None:
                    stats["embed"]["embed_tps"] = f"{float(tps):.0f} tok/s"
            elif metrics_resp.status_code == 501:
                stats["embed"]["metrics"] = "disabled (start llama.cpp with --metrics)"
            else:
                stats["embed"]["metrics"] = f"HTTP {metrics_resp.status_code}"
        except Exception as e:
            stats["embed"]["metrics"] = f"unavailable: {e}"

        return stats
