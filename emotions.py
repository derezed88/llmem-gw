"""
emotions.py — Emotion inference engine for the Feelings Wheel taxonomy.

Runs as a background asyncio task (see llmem-gw.py). On each cycle it:
  1. Fetches short-term and long-term memory entries that lack emotion annotations.
  2. Sends a batch to a cheap LLM with the Feelings Wheel taxonomy.
  3. Writes inferred emotion records to samaritan_emotions.
  4. Optionally surfaces reflection check-in prompts at natural breakpoints.

Config (plugins-enabled.json → plugin_config.emotions):
    enabled:                bool   — master switch (default false)
    inference_enabled:      bool   — emotion inference loop (default true when master on)
    inference_interval_m:   int    — minutes between inference cycles (default 10)
    inference_model:        str    — model key for inference (default "summarizer-gemini")
    inference_batch_size:   int    — max memories per batch (default 10)
    confidence_threshold:   float  — min confidence to store (default 0.3)
    reflection_prompts:     bool   — enable reflection check-ins (default true)
    reflection_interval_turns: int — turns between check-in opportunities (default 20)

Runtime control (no restart needed):
    get_emotion_stats()        → dict of counters + last-run info
    get_runtime_overrides()    → dict of session overrides
    set_runtime_override(k, v) → set a runtime override
    clear_runtime_overrides()  → reset to config-file values
    trigger_now()              → wake the sleeping task immediately
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

log = logging.getLogger("emotions")


# ---------------------------------------------------------------------------
# Feelings Wheel taxonomy — 3-layer hierarchy
# ---------------------------------------------------------------------------

FEELINGS_WHEEL = {
    "angry": {
        "mad": ["furious", "jealous"],
        "aggressive": ["provoked", "hostile"],
        "frustrated": ["infuriated", "annoyed"],
        "distant": ["withdrawn", "numb"],
        "critical": ["sceptical", "dismissive"],
        "disapproving": ["judgmental", "embarrassed"],
    },
    "sad": {
        "hurt": ["embarrassed", "disappointed"],
        "depressed": ["inferior", "empty"],
        "guilty": ["remorseful", "ashamed"],
        "vulnerable": ["fragile", "victimised"],
        "lonely": ["isolated", "abandoned"],
        "despair": ["grief", "powerless"],
    },
    "disgusted": {
        "awful": ["nauseated", "detestable"],
        "repelled": ["horrified", "hesitant"],
        "revolted": ["appalled", "revolted"],
    },
    "happy": {
        "peaceful": ["content", "intimate"],
        "trusting": ["sensitive", "loving"],
        "optimistic": ["hopeful", "inspired"],
        "loving": ["respected", "valued"],
        "thankful": ["grateful", "appreciative"],
        "creative": ["courageous", "energetic"],
    },
    "surprised": {
        "excited": ["eager", "energetic"],
        "playful": ["aroused", "cheeky"],
        "confused": ["disillusioned", "perplexed"],
        "amazed": ["astonished", "awe"],
        "startled": ["shocked", "dismayed"],
    },
    "bad": {
        "stressed": ["overwhelmed", "out of control"],
        "tired": ["sleepy", "unfocused"],
        "bored": ["indifferent", "apathetic"],
        "anxious": ["worried", "frightened"],
        "scared": ["helpless", "threatened"],
        "weak": ["insignificant", "worthless"],
    },
    "fearful": {
        "worried": ["nervous", "anxious"],
        "overwhelmed": ["pressured", "rushed"],
        "frightened": ["helpless", "threatened"],
        "rejected": ["excluded", "persecuted"],
        "inadequate": ["inferior", "worthless"],
        "exposed": ["humiliated", "ridiculed"],
    },
}

# Core emotions as a flat list
CORE_EMOTIONS = list(FEELINGS_WHEEL.keys())

# All emotion labels (flattened) for validation
ALL_LABELS = set()
for core, middle_map in FEELINGS_WHEEL.items():
    ALL_LABELS.add(core)
    for mid, outers in middle_map.items():
        ALL_LABELS.add(mid)
        ALL_LABELS.update(outers)


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_stats: dict = {
    "scans_run":              0,
    "memories_analyzed":      0,
    "emotions_inferred":      0,
    "emotions_stored":        0,
    "below_threshold":        0,
    "corrections_received":   0,
    "avg_confidence":         0.0,
    "last_scan_at":           None,
    "last_scan_duration_s":   None,
    "last_scan_batch":        0,
    "last_scan_stored":       0,
    "last_error":             None,
}

_overrides: dict = {}
_wake_event: asyncio.Event | None = None
# Running confidence average state
_conf_sum: float = 0.0
_conf_count: int = 0


def get_emotion_stats() -> dict:
    return dict(_stats)


def get_runtime_overrides() -> dict:
    return dict(_overrides)


def set_runtime_override(key: str, value) -> None:
    _overrides[key] = value


def clear_runtime_overrides() -> None:
    _overrides.clear()


def trigger_now() -> None:
    if _wake_event:
        _wake_event.set()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PLUGINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")


def _emotion_cfg() -> dict:
    """Return emotions config with defaults. Runtime overrides shadow file values."""
    try:
        with open(_PLUGINS_PATH) as f:
            raw = json.load(f).get("plugin_config", {}).get("emotions", {})
    except Exception:
        raw = {}

    base = {
        "enabled":                    raw.get("enabled",                    False),
        "inference_enabled":          raw.get("inference_enabled",          True),
        "inference_interval_m":       int(raw.get("inference_interval_m",   10)),
        "inference_model":            raw.get("inference_model",            "summarizer-gemini"),
        "inference_batch_size":       int(raw.get("inference_batch_size",   10)),
        "confidence_threshold":       float(raw.get("confidence_threshold", 0.3)),
        "reflection_prompts":         raw.get("reflection_prompts",         True),
        "reflection_interval_turns":  int(raw.get("reflection_interval_turns", 20)),
    }

    base.update(_overrides)
    return base


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an emotion analysis engine using the Feelings Wheel taxonomy. "
    "The Feelings Wheel has 7 core emotions: angry, sad, disgusted, happy, surprised, bad, fearful. "
    "Each core emotion branches into middle-ring and outer-ring granular emotions.\n\n"
    "Analyze the user's text and infer their emotional state. Focus on the USER's emotions "
    "(not the AI assistant's). If a memory entry is purely factual/informational with no "
    "emotional content, return null for that entry.\n\n"
    "Each entry includes surrounding conversation context (marked BEFORE/AFTER) to help you "
    "understand tone and intent. Use this context to disambiguate — e.g., 'whatever you think' "
    "after a frustrating exchange reads differently than after a smooth one. Only analyze the "
    "TARGET line, but let context inform your interpretation.\n\n"
    "Be conservative — only assign emotions when there are genuine signals in the text."
)

_USER_PROMPT_TMPL = (
    "Analyze each memory entry below and infer the user's emotional state.\n"
    "Return a JSON array with one object per entry:\n"
    '{{"memory_id": N, "core_emotion": "angry|sad|...|null", '
    '"emotion_label": "specific label from wheel or null", '
    '"intensity": 0.0-1.0, "confidence": 0.0-1.0, '
    '"context": "brief reason for this inference"}}\n\n'
    "If no emotion is detectable, set core_emotion to null.\n\n"
    "ENTRIES:\n{entries_text}"
)


async def _fetch_unanalyzed_memories(batch_size: int) -> list[dict]:
    """Fetch memories from both short-term and long-term that lack emotion annotations."""
    from database import fetch_dicts
    from memory import _ST, _LT

    try:
        rows = await fetch_dicts(
            f"(SELECT st.id, st.topic, st.content, st.importance, st.source, st.created_at, "
            f"  'shortterm' AS memory_tier "
            f" FROM {_ST()} st "
            f" LEFT JOIN samaritan_emotions em ON em.memory_table = 'shortterm' AND em.memory_id = st.id "
            f" WHERE em.id IS NULL "
            f" AND st.source IN ('user', 'session'))"
            f" UNION ALL "
            f"(SELECT lt.id, lt.topic, lt.content, lt.importance, lt.source, lt.created_at, "
            f"  'longterm' AS memory_tier "
            f" FROM {_LT()} lt "
            f" LEFT JOIN samaritan_emotions em ON em.memory_table = 'longterm' AND em.memory_id = lt.id "
            f" WHERE em.id IS NULL "
            f" AND lt.source IN ('user', 'session'))"
            f" ORDER BY created_at DESC "
            f" LIMIT {int(batch_size)}"
        )
        return rows or []
    except Exception as e:
        log.warning(f"emotions: fetch_unanalyzed failed: {e}")
        return []


async def _fetch_context_window(memories: list[dict], window: int = 2,
                                 max_gap_minutes: int = 30) -> dict[str, list[dict]]:
    """For each memory, fetch surrounding messages within a time window.

    Only includes neighbors within max_gap_minutes of the target — entries from
    different conversations (large time gaps) are excluded even if IDs are adjacent.

    Returns {"{tier}:{id}": [ordered neighbor rows]} keyed by tier:id.
    Uses a single query per tier for efficiency.
    """
    from database import fetch_dicts
    from memory import _ST, _LT
    from datetime import timedelta

    by_tier: dict[str, list[dict]] = {}
    for m in memories:
        tier = m.get("memory_tier", "shortterm")
        by_tier.setdefault(tier, []).append(m)

    tier_tables = {"shortterm": _ST(), "longterm": _LT()}
    context_map: dict[str, list[dict]] = {}
    gap = timedelta(minutes=max_gap_minutes)

    for tier, mems in by_tier.items():
        table = tier_tables.get(tier)
        if not table:
            continue

        # Fetch a range of rows covering all targets ± window IDs
        ids = [m["id"] for m in mems]
        min_id = min(ids) - window
        max_id = max(ids) + window
        try:
            neighbors = await fetch_dicts(
                f"SELECT id, content, source, created_at FROM {table} "
                f"WHERE id BETWEEN {min_id} AND {max_id} "
                f"ORDER BY id ASC"
            )
        except Exception as e:
            log.debug(f"emotions: context fetch failed for {tier}: {e}")
            continue

        neighbor_by_id = {r["id"]: r for r in neighbors}

        for m in mems:
            mid = m["id"]
            target_ts = m.get("created_at")
            ctx = []
            for offset in range(-window, window + 1):
                nid = mid + offset
                if nid == mid or nid not in neighbor_by_id:
                    continue
                n = neighbor_by_id[nid]
                # Only include if within time window
                if target_ts and n.get("created_at"):
                    delta = abs(target_ts - n["created_at"])
                    if isinstance(delta, timedelta) and delta > gap:
                        continue
                ctx.append(n)
            context_map[f"{tier}:{mid}"] = ctx

    return context_map


async def _call_llm(model_key: str, entries: list[dict],
                     context_map: dict[str, list[dict]] | None = None) -> list[dict]:
    """Call the emotion inference model; return list of emotion dicts."""
    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage
    import asyncio as _asyncio

    context_map = context_map or {}
    parts = []
    for e in entries:
        tier = e.get("memory_tier", "shortterm")
        mid = e["id"]
        key = f"{tier}:{mid}"
        ctx = context_map.get(key, [])

        before = [c for c in ctx if c["id"] < mid]
        after = [c for c in ctx if c["id"] > mid]

        lines = []
        if before:
            lines.append("  BEFORE:")
            for c in before:
                lines.append(f"    [{c.get('source','?')}] {str(c.get('content',''))[:200]}")
        lines.append(f"  TARGET [{mid}] (source={e.get('source','?')}) {str(e.get('content',''))[:500]}")
        if after:
            lines.append("  AFTER:")
            for c in after:
                lines.append(f"    [{c.get('source','?')}] {str(c.get('content',''))[:200]}")
        parts.append("\n".join(lines))

    entries_text = "\n---\n".join(parts)

    prompt = _USER_PROMPT_TMPL.format(entries_text=entries_text)
    try:
        if model_key not in LLM_REGISTRY:
            log.warning(f"emotions: unknown model {model_key!r}, falling back")
            model_key = next(iter(LLM_REGISTRY))
        cfg = LLM_REGISTRY[model_key]
        timeout = cfg.get("llm_call_timeout", 60)
        llm = _build_lc_llm(model_key)
        msgs = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        response = await _asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)
    except Exception as e:
        log.warning(f"emotions: LLM call failed: {e}")
        return []

    if not raw:
        return []

    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        items = json.loads(cleaned)
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"emotions: JSON parse failed: {e}. raw={raw[:200]}")
    return []


async def _write_emotion(item: dict, confidence_threshold: float) -> bool:
    """Write an emotion record to samaritan_emotions. Returns True if stored."""
    from database import execute_sql
    global _conf_sum, _conf_count

    core = item.get("core_emotion")
    if not core or core == "null":
        return False

    confidence = float(item.get("confidence", 0.5))
    if confidence < confidence_threshold:
        _stats["below_threshold"] += 1
        return False

    mem_id = int(item.get("memory_id", 0))
    mem_table = str(item.get("_memory_table", "shortterm"))[:20]
    label = str(item.get("emotion_label") or core)[:50]
    intensity = max(0.0, min(1.0, float(item.get("intensity", 0.5))))
    context = str(item.get("context", ""))[:500].replace("'", "''")
    core_clean = str(core)[:20]

    # Build the 7-dimension vector from the core emotion
    dims = {e: 0.0 for e in CORE_EMOTIONS}
    if core_clean in dims:
        dims[core_clean] = intensity

    cols = ", ".join([
        "memory_table", "memory_id",
        "angry", "sad", "disgusted", "happy", "surprised", "bad", "fearful",
        "emotion_label", "intensity", "confidence", "source", "context"
    ])
    vals = ", ".join([
        f"'{mem_table}'", str(mem_id),
        str(dims["angry"]), str(dims["sad"]), str(dims["disgusted"]),
        str(dims["happy"]), str(dims["surprised"]), str(dims["bad"]),
        str(dims["fearful"]),
        f"'{label}'", str(intensity), str(confidence),
        "'inferred'", f"'{context}'"
    ])

    try:
        await execute_sql(f"INSERT INTO samaritan_emotions ({cols}) VALUES ({vals})")
        # Update running average
        _conf_sum += confidence
        _conf_count += 1
        _stats["avg_confidence"] = round(_conf_sum / _conf_count, 3)
        return True
    except Exception as e:
        log.warning(f"emotions: write failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main scan cycle
# ---------------------------------------------------------------------------

async def run_scan() -> dict:
    """Single emotion inference cycle. Safe to call manually."""
    from database import set_db_override
    set_db_override("mymcp")

    cfg = _emotion_cfg()
    model = cfg["inference_model"]
    batch_size = cfg["inference_batch_size"]
    threshold = cfg["confidence_threshold"]

    summary = {"memories": 0, "inferred": 0, "stored": 0, "error": None}
    t_start = time.monotonic()

    try:
        memories = await _fetch_unanalyzed_memories(batch_size)
        summary["memories"] = len(memories)

        if not memories:
            summary["skipped_reason"] = "no unanalyzed memories"
            return summary

        # Build tier lookup so _write_emotion knows shortterm vs longterm
        tier_map = {m["id"]: m.get("memory_tier", "shortterm") for m in memories}

        # Fetch conversational context (±2 surrounding messages per entry)
        context_map = await _fetch_context_window(memories, window=2)

        results = await _call_llm(model, memories, context_map)
        summary["inferred"] = len(results)
        _stats["memories_analyzed"] += len(memories)
        _stats["emotions_inferred"] += len(results)

        stored = 0
        for item in results:
            mid = int(item.get("memory_id", 0))
            item["_memory_table"] = tier_map.get(mid, "shortterm")
            if await _write_emotion(item, threshold):
                stored += 1
        summary["stored"] = stored
        _stats["emotions_stored"] += stored

    except Exception as e:
        log.error(f"emotions: scan error: {e}")
        summary["error"] = str(e)
        _stats["last_error"] = str(e)

    duration = time.monotonic() - t_start
    _stats["scans_run"] += 1
    _stats["last_scan_at"] = datetime.now(timezone.utc).isoformat()
    _stats["last_scan_duration_s"] = round(duration, 2)
    _stats["last_scan_batch"] = summary["memories"]
    _stats["last_scan_stored"] = summary["stored"]

    set_db_override("")

    log.info(
        f"emotions: scan done — memories={summary['memories']} "
        f"inferred={summary['inferred']} stored={summary['stored']} "
        f"dur={duration:.1f}s"
    )
    return summary


# ---------------------------------------------------------------------------
# Background task entry point — called from llmem-gw.py
# ---------------------------------------------------------------------------

async def emotions_task() -> None:
    """Long-running asyncio task for emotion inference."""
    from timer_registry import register_timer, timer_sleep
    register_timer("emotions", "cogn")

    global _wake_event
    _wake_event = asyncio.Event()

    # Skip immediate run on startup — wait before first run
    log.info("emotions_task: startup delay 5m before first run")
    try:
        await asyncio.wait_for(_wake_event.wait(), timeout=300)
        _wake_event.clear()
    except asyncio.TimeoutError:
        pass

    while True:
        cfg = _emotion_cfg()

        if not cfg["enabled"] or not cfg["inference_enabled"]:
            _wake_event.clear()
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=300)
                _wake_event.clear()
            except asyncio.TimeoutError:
                pass
            continue

        interval_m = cfg["inference_interval_m"]
        if interval_m <= 0:
            await asyncio.sleep(3600)
            continue

        scan_result = {}
        try:
            scan_result = await run_scan()
        except Exception as e:
            log.warning(f"emotions_task: unhandled error: {e}")
            _stats["last_error"] = str(e)

        # Backoff during idle periods — skip backoff while backfilling unanalyzed memories
        batch_size = cfg["inference_batch_size"]
        from state import backoff_interval, idle_seconds, fmt_interval
        has_backlog = scan_result.get("memories", 0) >= batch_size
        if has_backlog:
            effective_m = interval_m
        else:
            effective_m = backoff_interval(interval_m, 60)
        sleep_sec = effective_m * 60
        if effective_m != interval_m:
            log.info(f"emotions: backoff {interval_m}m → {effective_m:.0f}m (idle {idle_seconds()/60:.0f}m)")
        timer_sleep("emotions", sleep_sec, interval_desc=fmt_interval(effective_m))
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            log.info("emotions_task: woken early by trigger")
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass
