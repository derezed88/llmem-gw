"""
MCP Direct Access Plugin for Claude Code

Exposes llmem-gw's data layer (memory, goals, plans, beliefs, DB, etc.)
as MCP tools via SSE transport. Claude Code connects as an MCP client and
calls tools directly — no LLM routing, no submit→stream pipeline.

Claude Code IS the reasoning engine; this plugin provides the persistence
and service layer.

Transport: SSE on a dedicated port (default 8769)
Auth: Optional API_KEY from environment

Usage in Claude Code .mcp.json:
  {
    "mcpServers": {
      "llmem-gw": {
        "type": "sse",
        "url": "http://localhost:8769/sse"
      }
    }
  }
"""

import json
import os
import logging
from typing import List, Any, Dict

from mcp.server.fastmcp import FastMCP, Context
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse

from plugin_loader import BasePlugin
from config import log

# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "llmem-gw-direct",
    instructions=(
        "Direct access to llmem-gw's tiered memory system, goal/plan engine, "
        "typed memories, temporal patterns, and data services. "
        "Use these tools to persist information across Claude Code sessions, "
        "track goals and plans, and access the MySQL database and Google services."
    ),
)

# ---------------------------------------------------------------------------
# Context helpers — workspace registration + DB routing
# ---------------------------------------------------------------------------

_DEFAULT_DB = os.getenv("MCP_DIRECT_DB", "mymcp")
_CLIENT_ID_PREFIX = "claude-code"

# Workspace registry: MCP session_id → {name, database, channel}
# Each Claude Code workspace gets its own MCP SSE session.
_session_workspaces: dict = {}  # mcp_session_id → workspace config
_workspaces: dict = {}          # workspace_name → workspace config (for relay lookups)


def _get_workspace_for_session() -> dict:
    """Get workspace config for the current MCP session, or default."""
    try:
        ctx = mcp.get_context()
        if ctx and ctx.session:
            sid = id(ctx.session)
            if sid in _session_workspaces:
                return _session_workspaces[sid]
    except Exception:
        pass
    return {"name": "default", "database": _DEFAULT_DB, "channel": "default"}


def _set_context(database: str = ""):
    """Set database and client context for tool execution."""
    from database import set_db_override
    from state import current_client_id
    ws = _get_workspace_for_session()
    db = database or ws.get("database", _DEFAULT_DB)
    set_db_override(db)
    cid = current_client_id.get("")
    ws_name = ws.get("name", "default")
    if not cid or not cid.startswith(_CLIENT_ID_PREFIX):
        current_client_id.set(f"{_CLIENT_ID_PREFIX}-{ws_name}")


@mcp.tool()
async def workspace_register(
    name: str,
    database: str = "mymcp",
) -> str:
    """Register this Claude Code workspace with a name and database target.

    Call this ONCE at session start. All subsequent tool calls will use this
    workspace's database for memory, goals, plans, etc.

    Args:
        name: Workspace identifier (e.g. 'default', 'ged-math', 'ged-reading').
              Also used as the voice relay channel name.
        database: MySQL database to use (e.g. 'mymcp', 'gedmath', 'gedreading')
    """
    ws_config = {
        "name": name,
        "database": database,
        "channel": name,
        "registered_at": __import__("time").time(),
    }
    _workspaces[name] = ws_config
    # Map this MCP session to the workspace
    try:
        ctx = mcp.get_context()
        if ctx and ctx.session:
            _session_workspaces[id(ctx.session)] = ws_config
    except Exception:
        pass
    _set_context()
    return f"Workspace '{name}' registered → database={database}, relay channel={name}"


# Stop words excluded from topic slug generation
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "about between through during before after above below up down out off over "
    "under again further then once here there when where why how all each every "
    "both few more most other some such no nor not only own same so than too very "
    "i me my we our you your he him his she her it its they them their what which "
    "who whom this that these those am just don also but and or if because until "
    "while let get got like think know want need make go going went come take "
    "tell said say really actually please thanks thank help much well yeah yes".split()
)


import re as _re


async def _generate_topic_slug(user_text: str, assistant_text: str) -> str | None:
    """Generate a 2-word topic slug and fuzzy-match against existing topics.

    Target format: 'job-seeking', 'health-advice', 'plumbing-issue' (2 hyphenated words).
    Strongly prefers reusing existing topics over creating new ones.
    """
    # Combine user text (primary) with first line of assistant (secondary signal)
    asst_first = assistant_text.split('\n')[0][:100] if assistant_text else ""
    combined = user_text[:200] + " " + asst_first

    # Strip tags, code blocks, URLs
    combined = _re.sub(r'<[^>]+>[^<]*</[^>]+>', '', combined)
    combined = _re.sub(r'https?://\S+', '', combined)

    # Tokenize and filter
    words = _re.findall(r'[a-zA-Z][a-zA-Z0-9]*', combined.lower())
    significant = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    if not significant:
        return None

    # Try to match against existing topics FIRST (aggressive fuzzy match)
    try:
        from memory import load_topic_list, _fuzzy_match_topics
        existing = await load_topic_list()
        if existing:
            # Try matching the whole user text against existing topic slugs
            matches = _fuzzy_match_topics(user_text[:200], existing, threshold=0.65)
            if matches:
                return matches[0]  # Best match — reuse it
    except Exception:
        pass

    # No existing match — generate a new 2-word slug
    from collections import Counter
    freq = Counter(significant)

    # Boost words appearing in both user and assistant
    user_words = set(_re.findall(r'[a-z]+', user_text[:200].lower()))
    asst_words = set(_re.findall(r'[a-z]+', asst_first.lower()))
    for w in freq:
        if w in user_words and w in asst_words:
            freq[w] += 3

    # Take top 2 words by frequency, preserving first-appearance order
    top_words = {w for w, _ in freq.most_common(6)}
    seen = set()
    slug_words = []
    for w in significant:
        if w in top_words and w not in seen:
            seen.add(w)
            slug_words.append(w)
            if len(slug_words) >= 2:
                break

    if not slug_words:
        return None

    raw_slug = "-".join(slug_words)

    # Final normalize pass — might still match an existing topic
    try:
        from memory import _normalize_topic
        return await _normalize_topic(raw_slug)
    except Exception:
        return raw_slug


# ═══════════════════════════════════════════════════════════════════════════
# TIER 0: Context retrieval + Cognitive system access
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def cogn_status() -> str:
    """Get the full cognitive system status dashboard.

    Returns the state of all cognitive loops (reflection, contradiction,
    prospective, temporal), their run stats, feedback verdicts, timer states,
    drive values, pending goal proposals, and active contradiction flags.

    Call this to understand what the autonomous cognitive system has been doing.
    """
    _set_context()
    lines = ["## Cognitive System Status\n"]

    # --- Timer/loop config ---
    try:
        from contradiction import get_contradiction_stats, get_runtime_overrides, _cogn_cfg
        cfg = _cogn_cfg()
        ovr = get_runtime_overrides()
        master = cfg["enabled"]

        def _ovr_tag(k):
            return " [runtime]" if k in ovr else ""

        lines.append(f"**Master**: {'ON' if master else 'OFF'}{_ovr_tag('enabled')}\n")
    except Exception as e:
        lines.append(f"(config unavailable: {e})\n")
        cfg = {}
        master = False

    # --- Timer registry (shows backoff state) ---
    try:
        from timer_registry import get_all_timers
        timers = get_all_timers()
        if timers:
            lines.append("**Timers**")
            for name, t in timers.items():
                interval = t.get("interval_desc", "?")
                last = t.get("last_run_desc", "never")
                lines.append(f"  {name:<25}: every {interval}  last={last}")
            lines.append("")
    except Exception:
        pass

    # --- Contradiction stats ---
    try:
        cs = get_contradiction_stats()
        lines.append("**Contradiction Scanner**")
        lines.append(
            f"  scans={cs['scans_run']}  pairs={cs['pairs_evaluated']}  "
            f"found={cs['contradictions_found']}  flags={cs['flags_written']}"
        )
        fb = cs.get("last_feedback")
        if fb:
            lines.append(
                f"  feedback: verdict={fb.get('verdict','?')}  "
                f"ratio={fb.get('ratio', 'n/a')}  "
                f"strength={fb.get('strength',0)}/10  streak={fb.get('streak',0)}"
            )
        if cs.get("last_error"):
            lines.append(f"  last_error: {cs['last_error']}")
        lines.append("")
    except Exception:
        pass

    # --- Prospective stats ---
    try:
        from prospective import get_prospective_stats
        ps = get_prospective_stats()
        lines.append("**Prospective Memory**")
        lines.append(
            f"  checks={ps['checks_run']}  fired={ps['rows_fired']}  "
            f"reminders={ps['reminders_written']}"
        )
        fb = ps.get("last_feedback")
        if fb:
            lines.append(
                f"  feedback: verdict={fb.get('verdict','?')}  "
                f"strength={fb.get('strength',0)}/10"
            )
        lines.append("")
    except Exception:
        pass

    # --- Reflection stats ---
    try:
        from reflection import get_reflection_stats
        rs = get_reflection_stats()
        lines.append("**Reflection Loop**")
        lines.append(
            f"  runs={rs['runs']}  turns_processed={rs['turns_processed']}  "
            f"memories_saved={rs['memories_saved']}  skipped={rs['memories_skipped']}"
        )
        lines.append(
            f"  last_dur={rs['last_run_duration_s']}s  last_saved={rs['last_run_saved']}"
        )
        fb = rs.get("last_feedback")
        if fb:
            lines.append(
                f"  feedback: verdict={fb.get('verdict','?')}  "
                f"strength={fb.get('strength',0)}/10"
            )
        lines.append("")
    except Exception:
        pass

    # --- Emotion inference stats ---
    try:
        from emotions import get_emotion_stats, _emotion_cfg
        ecfg = _emotion_cfg()
        es = get_emotion_stats()
        lines.append("**Emotion Inference**")
        lines.append(
            f"  enabled={'ON' if ecfg['enabled'] else 'OFF'}  "
            f"scans={es['scans_run']}  analyzed={es['memories_analyzed']}  "
            f"stored={es['emotions_stored']}  below_threshold={es['below_threshold']}"
        )
        lines.append(
            f"  avg_confidence={es['avg_confidence']}  "
            f"last={es.get('last_scan_at', 'never')}"
        )
        if es.get("last_error"):
            lines.append(f"  last_error: {es['last_error']}")
        lines.append("")
    except Exception:
        pass

    # --- Drives ---
    try:
        from memory import load_drives
        drives = await load_drives()
        if drives:
            lines.append("**Drives**")
            for d in drives:
                lines.append(
                    f"  {d['name']:<20}: {float(d.get('value',0)):.2f}  "
                    f"(baseline={float(d.get('baseline',0)):.2f}  "
                    f"decay={float(d.get('decay_rate',0)):.2f})"
                )
            lines.append("")
    except Exception:
        pass

    # --- Pending goal proposals ---
    try:
        from database import fetch_dicts
        from memory import _GOALS
        proposed = await fetch_dicts(
            f"SELECT id, title, description, importance, created_at "
            f"FROM {_GOALS()} WHERE session_id='reflection-proposed' "
            f"AND status='active' ORDER BY created_at DESC LIMIT 10"
        ) or []
        auto = await fetch_dicts(
            f"SELECT id, title, importance, created_at "
            f"FROM {_GOALS()} WHERE session_id='reflection' "
            f"AND status='active' ORDER BY created_at DESC LIMIT 5"
        ) or []
        if proposed:
            lines.append(f"**Pending Goal Proposals** ({len(proposed)})")
            for p in proposed:
                lines.append(
                    f"  [{p['id']}] {p.get('title','')} (imp={p.get('importance',5)}) "
                    f"— {p.get('description','')[:100]}"
                )
            lines.append("")
        if auto:
            lines.append(f"**Auto-Created Goals** ({len(auto)})")
            for a in auto:
                lines.append(f"  [{a['id']}] {a.get('title','')}")
            lines.append("")
    except Exception:
        pass

    # --- Active contradiction flags ---
    try:
        from database import fetch_dicts
        from memory import _BELIEFS
        flags = await fetch_dicts(
            f"SELECT id, content, confidence, created_at "
            f"FROM {_BELIEFS()} WHERE topic='contradiction-flag' AND status='active' "
            f"ORDER BY created_at DESC LIMIT 10"
        ) or []
        if flags:
            lines.append(f"**Active Contradiction Flags** ({len(flags)})")
            for f_ in flags:
                lines.append(f"  [{f_['id']}] {f_.get('content','')[:160]}")
            lines.append("")
    except Exception:
        pass

    # --- Cognition table recent entries ---
    try:
        from database import fetch_dicts
        from memory import _COGNITION
        recent = await fetch_dicts(
            f"SELECT id, origin, topic, LEFT(content, 150) as content, importance, created_at "
            f"FROM {_COGNITION()} ORDER BY created_at DESC LIMIT 10"
        ) or []
        if recent:
            lines.append("**Recent Cognition Outputs** (last 10)")
            for r in recent:
                lines.append(
                    f"  [{r['id']}] {r.get('origin','?')}/{r.get('topic','')} "
                    f"imp={r.get('importance','')} — {r.get('content','')[:120]}"
                )
            lines.append("")
    except Exception:
        pass

    return "\n".join(lines) if len(lines) > 2 else "(cognitive system data unavailable)"


@mcp.tool()
async def cogn_control(
    action: str,
    target: str = "",
    value: str = "",
) -> str:
    """Control cognitive loops — enable/disable, trigger runs, manage proposals.

    Args:
        action: One of:
            'enable' / 'disable' — master switch for all cognitive loops
            'loop_on' / 'loop_off' — enable/disable a specific loop (target=loop name)
            'loop_run' — trigger an immediate run of a loop (target=loop name)
            'approve_goal' — approve a reflection-proposed goal (target=goal ID)
            'reject_goal' — reject a reflection-proposed goal (target=goal ID)
            'clear_flags' — retract all open contradiction flags
            'reset_feedback' — reset feedback streak for a loop (target=loop name)
            'set_interval' — set loop interval in minutes (target=loop name, value=minutes)
        target: Loop name ('contradiction', 'prospective', 'reflection', 'temporal', 'emotions')
                or goal ID (for approve/reject)
        value: Interval value for set_interval (minutes)
    """
    _set_context()

    # --- Master enable/disable ---
    if action == "enable":
        from contradiction import set_runtime_override
        set_runtime_override("enabled", True)
        return "Cognitive system master switch: ON"

    if action == "disable":
        from contradiction import set_runtime_override
        set_runtime_override("enabled", False)
        return "Cognitive system master switch: OFF"

    # --- Loop on/off/run ---
    _loop_map = {
        "contradiction": {
            "flag": "contradiction_enabled",
            "trig_fn": lambda: __import__("contradiction").trigger_now,
            "run_fn": lambda: __import__("contradiction").run_scan,
            "label": "Contradiction scanner",
        },
        "prospective": {
            "flag": "prospective_enabled",
            "trig_fn": lambda: __import__("prospective").trigger_now,
            "run_fn": lambda: __import__("prospective").run_check,
            "label": "Prospective memory loop",
        },
        "reflection": {
            "flag": "reflection_enabled",
            "trig_fn": lambda: __import__("reflection").trigger_now,
            "run_fn": lambda: __import__("reflection").run_reflection,
            "label": "Reflection loop",
        },
        "temporal": {
            "flag": "inference_enabled",
            "trig_fn": lambda: __import__("temporal_inference").trigger_now,
            "run_fn": lambda: __import__("temporal_inference").run_temporal_inference,
            "label": "Temporal inference",
        },
        "emotions": {
            "flag": "inference_enabled",
            "trig_fn": lambda: __import__("emotions").trigger_now,
            "run_fn": lambda: __import__("emotions").run_scan,
            "label": "Emotion inference",
        },
    }

    if action == "loop_on" and target in _loop_map:
        if target == "emotions":
            from emotions import set_runtime_override as _emo_ovr
            _emo_ovr(_loop_map[target]["flag"], True)
        else:
            from contradiction import set_runtime_override
            set_runtime_override(_loop_map[target]["flag"], True)
        return f"{_loop_map[target]['label']}: ON"

    if action == "loop_off" and target in _loop_map:
        if target == "emotions":
            from emotions import set_runtime_override as _emo_ovr
            _emo_ovr(_loop_map[target]["flag"], False)
        else:
            from contradiction import set_runtime_override
            set_runtime_override(_loop_map[target]["flag"], False)
        return f"{_loop_map[target]['label']}: OFF"

    if action == "loop_run" and target in _loop_map:
        lm = _loop_map[target]
        try:
            lm["trig_fn"]()()
            run_fn = lm["run_fn"]()
            summary = await run_fn()
            skip = summary.get("skipped_reason") or summary.get("skipped")
            if skip:
                return f"{lm['label']} skipped: {skip}"
            err = summary.get("error")
            if err:
                return f"{lm['label']} error: {err}"
            parts = [f"{k}={v}" for k, v in summary.items()
                     if k not in ("error", "skipped", "skipped_reason")]
            return f"{lm['label']} completed — {', '.join(parts)}"
        except Exception as e:
            return f"{lm['label']} failed: {e}"

    # --- Set interval ---
    if action == "set_interval" and target in _loop_map:
        try:
            minutes = int(value)
            if minutes < 0:
                return "Interval must be >= 0"
            from contradiction import set_runtime_override
            _interval_keys = {
                "contradiction": "contradiction_interval_m",
                "prospective": "prospective_interval_m",
                "reflection": "reflection_interval_m",
                "emotions": "inference_interval_m",
            }
            key = _interval_keys.get(target)
            if not key:
                return f"Interval setting not supported for {target}"
            if target == "emotions":
                from emotions import set_runtime_override as _emo_ovr
                _emo_ovr(key, minutes)
            else:
                set_runtime_override(key, minutes)
            return f"{target} interval set to {minutes}m"
        except ValueError:
            return f"Invalid interval value: {value}"

    # --- Approve/reject goal proposals ---
    if action == "approve_goal":
        try:
            gid = int(target)
            from database import execute_sql
            from memory import _GOALS, load_drives, update_drive
            await execute_sql(
                f"UPDATE {_GOALS()} SET session_id='reflection-approved' "
                f"WHERE id={gid} AND session_id='reflection-proposed'"
            )
            # Nudge autonomy drive up (user endorsed the proposal)
            drives = await load_drives()
            cur = next((float(d.get("value", 0.5)) for d in drives if d["name"] == "autonomy"), 0.5)
            new_val = round(max(0.0, min(1.0, cur + 0.03)), 3)
            await update_drive("autonomy", new_val, source="user")
            return f"Goal id={gid} approved. autonomy: {cur:.2f} → {new_val:.2f} (+0.03)"
        except Exception as e:
            return f"Approve failed: {e}"

    if action == "reject_goal":
        try:
            gid = int(target)
            from database import execute_sql
            from memory import _GOALS, load_drives, update_drive
            await execute_sql(
                f"UPDATE {_GOALS()} SET status='abandoned', "
                f"abandon_reason='rejected via claude-code cogn_control' "
                f"WHERE id={gid} AND session_id='reflection-proposed'"
            )
            drives = await load_drives()
            cur = next((float(d.get("value", 0.5)) for d in drives if d["name"] == "autonomy"), 0.5)
            new_val = round(max(0.0, min(1.0, cur - 0.05)), 3)
            await update_drive("autonomy", new_val, source="user")
            return f"Goal id={gid} rejected. autonomy: {cur:.2f} → {new_val:.2f} (-0.05)"
        except Exception as e:
            return f"Reject failed: {e}"

    # --- Clear contradiction flags ---
    if action == "clear_flags":
        try:
            from database import execute_sql
            from memory import _BELIEFS
            await execute_sql(
                f"UPDATE {_BELIEFS()} SET status='retracted' "
                f"WHERE topic='contradiction-flag' AND status='active'"
            )
            return "All open contradiction flags retracted."
        except Exception as e:
            return f"Clear flags failed: {e}"

    # --- Reset feedback ---
    if action == "reset_feedback" and target:
        valid = ("contradiction", "prospective", "reflection")
        if target not in valid:
            return f"Invalid loop: {target}. Valid: {', '.join(valid)}"
        try:
            from cogn_feedback import reset_feedback_state
            reset_feedback_state(target)
            return f"Feedback state reset for {target}. Streak and strength cleared."
        except Exception as e:
            return f"Reset failed: {e}"

    return (
        f"Unknown action '{action}'. Valid: enable, disable, loop_on, loop_off, "
        f"loop_run, approve_goal, reject_goal, clear_flags, reset_feedback, set_interval"
    )

async def _load_emotion_context() -> str:
    """Build a concise emotional tone summary from recent samaritan_emotions data."""
    from database import fetch_dicts

    lines = []
    try:
        # Recent emotional tone (last 24h)
        recent = await fetch_dicts(
            "SELECT emotion_label, COUNT(*) as cnt, "
            "ROUND(AVG(intensity),2) as avg_int, ROUND(AVG(confidence),2) as avg_conf "
            "FROM samaritan_emotions "
            "WHERE created_at >= NOW() - INTERVAL 24 HOUR "
            "GROUP BY emotion_label ORDER BY cnt DESC LIMIT 8"
        )
        if recent:
            top = ", ".join(
                f"{r['emotion_label']} ({r['cnt']}×, int={r['avg_int']})"
                for r in recent
            )
            lines.append(f"**Last 24h**: {top}")

        # Emotional trend — compare last 24h dominant emotion vs prior 7 days
        trend = await fetch_dicts(
            "SELECT "
            "  (SELECT emotion_label FROM samaritan_emotions "
            "   WHERE created_at >= NOW() - INTERVAL 24 HOUR "
            "   GROUP BY emotion_label ORDER BY COUNT(*) DESC LIMIT 1) AS recent_dominant, "
            "  (SELECT emotion_label FROM samaritan_emotions "
            "   WHERE created_at BETWEEN NOW() - INTERVAL 7 DAY AND NOW() - INTERVAL 24 HOUR "
            "   GROUP BY emotion_label ORDER BY COUNT(*) DESC LIMIT 1) AS prior_dominant, "
            "  (SELECT ROUND(AVG(intensity),2) FROM samaritan_emotions "
            "   WHERE created_at >= NOW() - INTERVAL 24 HOUR) AS recent_avg_int, "
            "  (SELECT ROUND(AVG(intensity),2) FROM samaritan_emotions "
            "   WHERE created_at BETWEEN NOW() - INTERVAL 7 DAY AND NOW() - INTERVAL 24 HOUR) AS prior_avg_int"
        )
        if trend and trend[0].get("recent_dominant"):
            t = trend[0]
            recent_d = t.get("recent_dominant", "?")
            prior_d = t.get("prior_dominant")
            recent_i = t.get("recent_avg_int", 0)
            prior_i = t.get("prior_avg_int")
            if prior_d and prior_d != recent_d:
                lines.append(f"**Shift**: dominant tone moved from {prior_d} → {recent_d}")
            if prior_i and recent_i:
                delta = float(recent_i) - float(prior_i)
                if abs(delta) >= 0.1:
                    direction = "↑" if delta > 0 else "↓"
                    lines.append(f"**Intensity**: {direction} {abs(delta):.2f} vs prior week")

        # Overall stats
        total = await fetch_dicts(
            "SELECT COUNT(*) as total FROM samaritan_emotions"
        )
        if total:
            lines.append(f"**Total emotions tracked**: {total[0]['total']}")

    except Exception as e:
        return ""

    if not lines:
        return ""

    return "## Emotional Context\n\n" + "\n".join(f"  {l}" for l in lines)


@mcp.tool()
async def load_context(
    query: str = "",
    include_typed: bool = True,
    include_procedures: bool = True,
    include_temporal: bool = True,
) -> str:
    """Load enriched context from the memory system — semantic search, typed memories,
    active goals/beliefs/drives, relevant procedures, and temporal patterns.

    This is the equivalent of auto_enrich_context() that runs before every LLM call
    in the normal pipeline. Call this at the START of a session and when switching
    topics to pull in relevant cross-session knowledge.

    Args:
        query: Topic or question to search for. If empty, returns high-importance
               memories and active goals/beliefs only.
        include_typed: Include goals, beliefs, drives, conditioned behaviors (default true)
        include_procedures: Include relevant procedure recipes (default true)
        include_temporal: Include temporal pattern context (default true)
    """
    _set_context()
    import asyncio
    from memory import (
        load_context_block, load_topic_list,
        load_temporal_context,
    )

    sections = []
    tasks = {}

    # Semantic memory retrieval (ST + LT via Qdrant)
    tasks["memory"] = asyncio.create_task(load_context_block(
        min_importance=3,
        query=query,
        user_text=query,
        memory_types_enabled=include_typed,
    ))

    # Known topics
    tasks["topics"] = asyncio.create_task(load_topic_list())

    # Typed memory tables (goals, beliefs, drives, conditioned)
    if include_typed:
        from memory import load_typed_context_block
        tasks["typed"] = asyncio.create_task(load_typed_context_block())

    # Procedures (semantic match against query)
    if include_procedures and query:
        from memory import load_procedure_context_block
        tasks["procedures"] = asyncio.create_task(
            load_procedure_context_block(task_hint=query)
        )

    # Temporal patterns
    if include_temporal:
        tasks["temporal"] = asyncio.create_task(load_temporal_context())

    # Emotional context — recent emotional tone from samaritan_emotions
    tasks["emotions"] = asyncio.create_task(_load_emotion_context())

    # Collect results (2s deadline — degrade gracefully if Qdrant is slow)
    done, pending = await asyncio.wait(tasks.values(), timeout=3.0)
    task_to_name = {t: n for n, t in tasks.items()}

    for task in done:
        name = task_to_name[task]
        try:
            result = task.result()
            if result:
                if name == "topics" and isinstance(result, list):
                    sections.append(f"## Known Topics\n{', '.join(result[:50])}")
                elif isinstance(result, str) and result.strip():
                    sections.append(result)
        except Exception:
            pass

    for task in pending:
        task.cancel()

    if not sections:
        return "(no context available — memory may be empty)"

    return "\n\n---\n\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Memory tools
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def memory_save(
    topic: str,
    content: str,
    importance: int = 5,
    source: str = "assistant",
) -> str:
    """Save a fact to short-term memory. Persists across sessions.

    Args:
        topic: Short topic slug, e.g. 'auth-rewrite', 'user-preferences'
        content: One concise sentence describing the fact
        importance: 1 (low) to 10 (critical). 8+ are always injected into context
        source: 'user' (facts they stated), 'assistant' (your conclusions), 'directive'
    """
    _set_context()
    from memory import save_memory as _save
    row_id = await _save(
        topic=topic, content=content, importance=importance,
        source=source, session_id=f"{_CLIENT_ID_PREFIX}-mcp",
    )
    if row_id:
        return f"Memory saved (id={row_id}): [{topic}] {content[:80]}"
    return "Memory not saved (duplicate or disabled)"


@mcp.tool()
async def memory_recall(
    topic: str = "",
    tier: str = "short",
    limit: int = 20,
    query: str = "",
) -> str:
    """Search memories by keyword. Returns matching rows with id, topic, content, importance.

    Args:
        topic: Keyword to match against topic label or content text
        tier: 'short' (active memories) or 'long' (aged-out archive)
        limit: Max rows to return (default 20)
        query: Alternative search term (used if topic is empty)
    """
    _set_context()
    search = topic or query
    from memory import _ST, _LT
    from database import execute_sql
    table = _ST() if tier == "short" else _LT()

    if search:
        s = search.replace("'", "''")
        sql = (
            f"SELECT id, topic, content, importance, source, created_at "
            f"FROM {table} WHERE topic LIKE '%{s}%' OR content LIKE '%{s}%' "
            f"ORDER BY importance DESC, created_at DESC LIMIT {limit}"
        )
    else:
        sql = (
            f"SELECT id, topic, content, importance, source, created_at "
            f"FROM {table} ORDER BY created_at DESC LIMIT {limit}"
        )

    result = await execute_sql(sql)
    return result if result.strip() else "(no memories found)"


@mcp.tool()
async def memory_search_semantic(
    query: str,
    tier: str = "both",
    top_k: int = 15,
    min_score: float = 0.45,
) -> str:
    """Semantic vector search over all memories using Qdrant + nomic-embed-text.

    Args:
        query: Natural language query to find relevant memories
        tier: 'short', 'long', or 'both' (default)
        top_k: Max results per tier
        min_score: Minimum similarity score (0.0-1.0)
    """
    _set_context()
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if not vec:
            return "Vector search unavailable (Qdrant plugin not loaded)"

        results = []
        tiers = ["short", "long"] if tier == "both" else [tier]
        for t in tiers:
            hits = await vec.search_memories(query, tier=t, top_k=top_k, min_score=min_score)
            for p in hits:
                results.append(
                    f"[{t}] id={p.get('id','')} score={p.get('score',0):.3f} "
                    f"imp={p.get('importance','')} topic={p.get('topic','')} "
                    f"content={p.get('content','')[:200]}"
                )
        return "\n".join(results) if results else "(no semantic matches)"
    except Exception as e:
        return f"Semantic search error: {e}"


@mcp.tool()
async def memory_update(
    id: int,
    tier: str = "short",
    importance: int = 0,
    content: str = "",
    topic: str = "",
) -> str:
    """Update an existing memory row's importance, content, or topic.

    Args:
        id: Row ID (from memory_recall)
        tier: 'short' or 'long'
        importance: New importance 1-10 (0 = leave unchanged)
        content: New content (empty = leave unchanged)
        topic: New topic (empty = leave unchanged)
    """
    _set_context()
    from memory import _ST, _LT
    from database import execute_sql
    table = _ST() if tier == "short" else _LT()

    parts = []
    if importance > 0:
        parts.append(f"importance = {max(1, min(10, importance))}")
    if content:
        parts.append(f"content = '{content.replace(chr(39), chr(39)*2)}'")
    if topic:
        parts.append(f"topic = '{topic.replace(chr(39), chr(39)*2)}'")
    if not parts:
        return "Nothing to update"

    sql = f"UPDATE {table} SET {', '.join(parts)} WHERE id = {id}"
    await execute_sql(sql)
    return f"Memory id={id} updated in {tier}"


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Goal & Plan tools
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def goal_create(
    title: str,
    description: str = "",
    importance: int = 9,
) -> str:
    """Create a new goal. Returns the goal ID.

    Args:
        title: Short goal title
        description: Full description of the objective
        importance: 1-10 (goals default to 9)
    """
    _set_context()
    from memory import _GOALS, _typed_metric_write
    from database import execute_insert
    importance = max(1, min(10, importance))
    t = title.replace("'", "''")
    d = description.replace("'", "''")
    sql = (
        f"INSERT INTO {_GOALS()} "
        f"(title, description, status, importance, source, session_id) "
        f"VALUES ('{t}', '{d}', 'active', {importance}, 'assistant', '{_CLIENT_ID_PREFIX}-mcp')"
    )
    row_id = await execute_insert(sql)
    _typed_metric_write(_GOALS())
    return f"Goal created (id={row_id}): {title}"


@mcp.tool()
async def goal_update(
    id: int,
    status: str = "",
    title: str = "",
    description: str = "",
    importance: int = 0,
) -> str:
    """Update an existing goal's status, title, description, or importance.

    Args:
        id: Goal ID
        status: 'active', 'done', 'blocked', 'abandoned' (empty = leave unchanged)
        title: New title (empty = leave unchanged)
        description: New description (empty = leave unchanged)
        importance: New importance 1-10 (0 = leave unchanged)
    """
    _set_context()
    from memory import _GOALS, _typed_metric_write
    from database import execute_sql

    parts = []
    if status and status in ("active", "done", "blocked", "abandoned"):
        parts.append(f"status = '{status}'")
    if title:
        parts.append(f"title = '{title.replace(chr(39), chr(39)*2)}'")
    if description:
        parts.append(f"description = '{description.replace(chr(39), chr(39)*2)}'")
    if importance > 0:
        parts.append(f"importance = {max(1, min(10, importance))}")
    if not parts:
        return "Nothing to update"

    sql = f"UPDATE {_GOALS()} SET {', '.join(parts)} WHERE id = {id}"
    await execute_sql(sql)
    _typed_metric_write(_GOALS())
    return f"Goal id={id} updated"


@mcp.tool()
async def goal_list(
    status: str = "active",
) -> str:
    """List goals filtered by status.

    Args:
        status: 'active', 'done', 'blocked', 'abandoned', or 'all'
    """
    _set_context()
    from memory import _GOALS
    from database import execute_sql

    where = "" if status == "all" else f"WHERE status = '{status}'"
    sql = (
        f"SELECT id, title, status, importance, description, created_at, updated_at "
        f"FROM {_GOALS()} {where} ORDER BY importance DESC, created_at DESC"
    )
    result = await execute_sql(sql)
    return result if result.strip() else f"(no {status} goals)"


@mcp.tool()
async def step_create(
    goal_id: int,
    description: str,
    step_order: int = 1,
    step_type: str = "concept",
    target: str = "model",
) -> str:
    """Create a plan step under a goal.

    Args:
        goal_id: Parent goal ID (0 = ad-hoc plan with no goal)
        description: What this step involves
        step_order: Sequence number (ascending)
        step_type: 'concept' (human-readable intent) or 'task' (executable atom)
        target: 'model' (auto-executable), 'human' (requires person),
                'investigate' (needs analysis), 'claude-code' (queued for Claude Code)
    """
    _set_context()
    from memory import _PLANS, _typed_metric_write
    from database import execute_insert

    step_type = step_type if step_type in ("concept", "task") else "concept"
    target = target if target in ("model", "human", "investigate", "claude-code") else "model"
    d = description.replace("'", "''")

    sql = (
        f"INSERT INTO {_PLANS()} "
        f"(goal_id, step_order, description, status, step_type, target, approval, source, session_id) "
        f"VALUES ({goal_id}, {step_order}, '{d}', 'pending', '{step_type}', '{target}', "
        f"'proposed', 'assistant', '{_CLIENT_ID_PREFIX}-mcp')"
    )
    row_id = await execute_insert(sql)
    _typed_metric_write(_PLANS())
    return f"Step created (id={row_id}): goal={goal_id} [{step_type}] {description[:80]}"


@mcp.tool()
async def step_update(
    id: int,
    status: str = "",
    result: str = "",
    executor: str = "",
) -> str:
    """Update a plan step's status and/or result.

    Args:
        id: Plan step ID
        status: 'pending', 'in_progress', 'done', 'skipped' (empty = leave unchanged)
        result: Execution output or notes (empty = leave unchanged)
        executor: Who/what executed this step, e.g. 'claude-code', 'direct' (empty = leave unchanged)
    """
    _set_context()
    from memory import _PLANS, _typed_metric_write
    from database import execute_sql

    parts = []
    if status and status in ("pending", "in_progress", "done", "skipped"):
        parts.append(f"status = '{status}'")
    if result:
        r = result[:4000].replace("'", "''")
        parts.append(f"result = '{r}'")
    if executor:
        parts.append(f"executor = '{executor.replace(chr(39), chr(39)*2)}'")
    if not parts:
        return "Nothing to update"

    sql = f"UPDATE {_PLANS()} SET {', '.join(parts)} WHERE id = {id}"
    await execute_sql(sql)
    _typed_metric_write(_PLANS())

    # Check completion cascade
    if status == "done":
        from plan_engine import _check_parent_completion, _check_goal_completion
        from database import fetch_dicts
        rows = await fetch_dicts(
            f"SELECT parent_id, goal_id FROM {_PLANS()} WHERE id = {id} LIMIT 1"
        )
        if rows:
            if rows[0].get("parent_id"):
                await _check_parent_completion(rows[0]["parent_id"])
            elif rows[0].get("goal_id"):
                await _check_goal_completion(rows[0]["goal_id"])

    return f"Step id={id} updated"


@mcp.tool()
async def step_list(
    goal_id: int = 0,
    status: str = "all",
) -> str:
    """List plan steps, optionally filtered by goal and status.

    Args:
        goal_id: Filter by goal ID (0 = all goals)
        status: 'pending', 'in_progress', 'done', 'skipped', or 'all'
    """
    _set_context()
    from memory import _PLANS
    from database import execute_sql

    conditions = []
    if goal_id > 0:
        conditions.append(f"goal_id = {goal_id}")
    if status != "all":
        conditions.append(f"status = '{status}'")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = (
        f"SELECT id, goal_id, step_order, description, status, step_type, "
        f"target, executor, approval, LEFT(result, 200) as result, created_at "
        f"FROM {_PLANS()} {where} ORDER BY goal_id, step_order"
    )
    result = await execute_sql(sql)
    return result if result.strip() else "(no steps found)"


@mcp.tool()
async def plan_decompose(
    concept_step_id: int,
) -> str:
    """Decompose a concept step into executable task steps using Haiku.

    This calls the plan engine's decomposer which uses an LLM to break
    a concept step into atomic tool_call specs. The resulting task steps
    require approval before execution.

    Args:
        concept_step_id: ID of the concept step to decompose
    """
    _set_context()
    from plan_engine import decompose_concept_step
    try:
        tasks = await decompose_concept_step(concept_step_id)
        lines = [f"Decomposed into {len(tasks)} task steps:"]
        for t in tasks:
            lines.append(
                f"  [{t['id']}] {t['description'][:80]} "
                f"target={t['target']} approval={t['approval']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Decompose failed: {e}"


@mcp.tool()
async def plan_check_completion(
    goal_id: int,
) -> str:
    """Check if a goal's plan steps are all complete and cascade status updates.

    Args:
        goal_id: Goal ID to check
    """
    _set_context()
    from plan_engine import _check_goal_completion
    from memory import _GOALS
    from database import fetch_dicts
    await _check_goal_completion(goal_id)
    rows = await fetch_dicts(
        f"SELECT id, title, status FROM {_GOALS()} WHERE id = {goal_id} LIMIT 1"
    )
    if rows:
        g = rows[0]
        return f"Goal id={g['id']} '{g['title']}': status={g['status']}"
    return f"Goal id={goal_id} not found"


@mcp.tool()
async def steps_for_claude_code(
) -> str:
    """List plan steps with target='claude-code' that are pending/approved.

    These are steps that the autonomous system has queued specifically
    for Claude Code to execute (e.g. code changes, filesystem operations).
    """
    _set_context()
    from memory import _PLANS, _GOALS
    from database import execute_sql

    try:
        sql = (
            f"SELECT p.id, p.goal_id, p.step_order, p.description, p.status, "
            f"p.step_type, p.approval, LEFT(p.result, 200) as result, "
            f"g.title as goal_title "
            f"FROM {_PLANS()} p "
            f"LEFT JOIN {_GOALS()} g ON g.id = p.goal_id "
            f"WHERE p.target = 'claude-code' AND p.status IN ('pending', 'in_progress') "
            f"ORDER BY p.goal_id, p.step_order"
        )
        result = await execute_sql(sql)
        return result if result.strip() else "(no steps queued for claude-code)"
    except Exception:
        return "(no steps — goals/plans tables not available in this database)"


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Typed memory tools
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def assert_belief(
    topic: str,
    content: str,
    confidence: int = 7,
    status: str = "active",
    id: int = 0,
) -> str:
    """Assert or update a world-state belief.

    Args:
        topic: Short topic label
        content: The asserted fact
        confidence: 1-10 (how certain)
        status: 'active' or 'retracted'
        id: Existing belief ID to update (0 = create new)
    """
    _set_context()
    from tools import _assert_belief_exec
    return await _assert_belief_exec(
        topic=topic, content=content, confidence=confidence,
        status=status, id=id,
    )


@mcp.tool()
async def save_memory_typed(
    memory_type: str,
    topic: str,
    content: str,
    importance: int = 5,
    source: str = "assistant",
    due_at: str = "",
    status: str = "pending",
    id: int = 0,
) -> str:
    """Save a typed experiential memory.

    Args:
        memory_type: 'episodic' (events), 'semantic' (facts), 'procedural' (skills),
                     'autobiographical' (identity), 'prospective' (future intentions)
        topic: Short topic label
        content: The memory content
        importance: 1-10
        source: 'user', 'assistant', 'directive', 'session'
        due_at: For prospective only: when to act (e.g. 'next Monday', '2026-03-28')
        status: For prospective only: 'pending', 'done', 'missed'
        id: Existing row ID to update (0 = create new)
    """
    _set_context()
    from tools import _save_memory_typed_exec
    return await _save_memory_typed_exec(
        memory_type=memory_type, topic=topic, content=content,
        importance=importance, source=source, due_at=due_at,
        status=status, id=id,
    )


@mcp.tool()
async def set_conditioned(
    topic: str,
    trigger: str,
    reaction: str,
    strength: int = 7,
    status: str = "active",
    source: str = "assistant",
    id: int = 0,
) -> str:
    """Record a learned trigger→reaction behavior pattern.

    Args:
        topic: Short topic label (e.g. 'proactive-schema-gap')
        trigger: Stimulus pattern or condition
        reaction: The learned response or behavior
        strength: 1-10 reinforcement strength
        status: 'active' or 'extinguished'
        source: 'user', 'assistant', 'directive', 'session'
        id: Existing ID to update (0 = create new)
    """
    _set_context()
    from tools import _set_conditioned_exec
    return await _set_conditioned_exec(
        topic=topic, trigger=trigger, reaction=reaction,
        strength=strength, status=status, source=source, id=id,
    )


@mcp.tool()
async def recall_temporal(
    query: str = "",
    group_by: str = "day_of_week",
    day_of_week: str = "",
    time_range: str = "",
    lookback_days: int = 30,
    limit: int = 50,
    new: bool = False,
) -> str:
    """Discover time-based patterns in memories (e.g. 'what happens on Mondays?').

    Args:
        query: Keyword filter (e.g. 'walk', 'gym'). Empty = all.
        group_by: 'hour', 'day_of_week', 'date', 'week', 'month'
        day_of_week: Optional: 'Monday', 'Tuesday', etc.
        time_range: Optional: 'HH:MM-HH:MM', 'morning', 'afternoon', 'evening', 'now'
        lookback_days: How many days back (default 30)
        limit: Max raw rows (default 50)
        new: Force fresh query, bypass cache
    """
    _set_context()
    from tools import _recall_temporal_exec
    return await _recall_temporal_exec(
        query=query, group_by=group_by, day_of_week=day_of_week,
        time_range=time_range, lookback_days=lookback_days,
        limit=limit, new=new,
    )


@mcp.tool()
async def procedure_save(
    task_type: str,
    topic: str,
    steps: str,
    outcome: str = "unknown",
    notes: str = "",
    importance: int = 7,
    id: int = 0,
) -> str:
    """Save a reusable multi-step workflow/procedure.

    Args:
        task_type: Machine-readable slug, e.g. 'deploy-docker', 'git-push-pr'
        topic: Human-readable title
        steps: JSON array of step objects: [{"step": 1, "action": "...", "tool": "...", "note": "..."}]
        outcome: 'success', 'partial', 'failure', 'unknown'
        notes: Lessons learned, caveats
        importance: 1-10 (8+ enables pre-injection)
        id: Existing procedure ID to update (0 = create new)
    """
    _set_context()
    from tools import _procedure_save_exec
    return await _procedure_save_exec(
        task_type=task_type, topic=topic, steps=steps,
        outcome=outcome, notes=notes, importance=importance, id=id,
    )


@mcp.tool()
async def procedure_recall(
    query: str,
    task_type: str = "",
    top_k: int = 5,
) -> str:
    """Recall procedures relevant to a task using semantic search.

    Args:
        query: Natural language description of the task you're about to do
        task_type: Optional exact slug filter (e.g. 'deploy-docker')
        top_k: Max procedures to return
    """
    _set_context()
    from tools import _procedure_recall_exec
    return await _procedure_recall_exec(query=query, task_type=task_type, top_k=top_k)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: Data access tools
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def db_query(sql: str) -> str:
    """Execute a SQL query against the MySQL database.

    The database is the mymcp system-of-record. All SQL is permitted
    including DDL (CREATE/ALTER/DROP TABLE) for schema evolution.

    Args:
        sql: SQL statement to execute
    """
    _set_context()
    from database import execute_sql
    return await execute_sql(sql)


@mcp.tool()
async def google_drive(
    operation: str,
    file_id: str = "",
    file_name: str = "",
    content: str = "",
    folder_id: str = "",
) -> str:
    """CRUD operations on Google Drive within the authorized folder.

    Args:
        operation: 'list', 'read', 'create', 'append', 'delete'
        file_id: Required for read/delete (get from list first)
        file_name: Required for create
        content: Required for create/append
        folder_id: Leave empty to use default folder
    """
    _set_context()
    from drive import run_drive_op
    return await run_drive_op(
        operation,
        file_id or None,
        file_name or None,
        content or None,
        folder_id or None,
    )


@mcp.tool()
async def google_calendar(
    operation: str,
    calendar_id: str = "primary",
    event_id: str = "",
    time_min: str = "",
    time_max: str = "",
    summary: str = "",
    description: str = "",
    start_time: str = "",
    end_time: str = "",
    location: str = "",
    max_results: int = 10,
) -> str:
    """Google Calendar operations.

    Args:
        operation: 'list_calendars', 'list_events', 'get_event', 'create_event',
                   'delete_event', 'freebusy'
        calendar_id: Calendar ID (default 'primary')
        event_id: Required for get_event/delete_event
        time_min: Start of time range (ISO 8601, e.g. '2026-03-26T00:00:00Z')
        time_max: End of time range (ISO 8601)
        summary: Event title (for create_event)
        description: Event description (for create_event)
        start_time: Event start (ISO 8601, for create_event)
        end_time: Event end (ISO 8601, for create_event)
        location: Event location (for create_event)
        max_results: Max events to return (for list_events)
    """
    _set_context()
    try:
        from calendar_google import run_calendar_op
        return await run_calendar_op(
            operation=operation, calendar_id=calendar_id,
            event_id=event_id, time_min=time_min, time_max=time_max,
            summary=summary, description=description,
            start_time=start_time, end_time=end_time,
            location=location, max_results=max_results,
        )
    except ImportError:
        return "Google Calendar plugin not available"
    except Exception as e:
        return f"Calendar error: {e}"


@mcp.tool()
async def google_tasks(
    operation: str,
    tasklist_id: str = "@default",
    task_id: str = "",
    title: str = "",
    notes: str = "",
    status: str = "",
    due: str = "",
    due_min: str = "",
    due_max: str = "",
    parent: str = "",
    previous: str = "",
    show_completed: bool = True,
    show_hidden: bool = False,
    max_results: int = 100,
) -> str:
    """Google Tasks operations.

    Args:
        operation: Task list ops: 'list_tasklists', 'create_tasklist', 'delete_tasklist', 'update_tasklist'.
                   Task ops: 'list_tasks', 'get_task', 'create_task', 'update_task',
                   'complete_task', 'delete_task', 'move_task', 'clear_completed'.
        tasklist_id: Task list ID (default '@default' = primary list). Use list_tasklists to discover.
        task_id: Task ID. Required for get/update/complete/delete/move.
        title: Title for task or task list (max 1024 chars). Required for create ops.
        notes: Task notes/description (max 8192 chars). For create_task/update_task.
        status: 'needsAction' or 'completed'. For update_task.
        due: Due date (YYYY-MM-DD or RFC 3339). For create_task/update_task.
        due_min: Lower bound due date filter (RFC 3339). For list_tasks.
        due_max: Upper bound due date filter (RFC 3339). For list_tasks.
        parent: Parent task ID for subtasks. For create_task/move_task.
        previous: Previous sibling task ID for ordering. For create_task/move_task.
        show_completed: Include completed tasks (default true). For list_tasks.
        show_hidden: Include hidden tasks (default false). For list_tasks.
        max_results: Max tasks to return (default 100). For list_tasks.
    """
    _set_context()
    try:
        from tasks_google import run_tasks_op
        return await run_tasks_op(
            operation=operation, tasklist_id=tasklist_id,
            task_id=task_id, title=title, notes=notes,
            status=status, due=due, due_min=due_min, due_max=due_max,
            parent=parent, previous=previous,
            show_completed=show_completed, show_hidden=show_hidden,
            max_results=max_results,
        )
    except ImportError:
        return "Google Tasks plugin not available"
    except Exception as e:
        return f"Tasks error: {e}"


@mcp.tool()
async def weather(
    location: str,
    forecast_type: str = "current",
    days: int = 5,
    hours: int = 24,
) -> str:
    """Get weather data for a location.

    Args:
        location: Place name (e.g. 'San Francisco, CA') — auto-geocoded to coordinates
        forecast_type: 'current', 'daily', 'hourly', or 'alerts'
        days: Number of forecast days (daily only, max 10, default 5)
        hours: Number of forecast hours (hourly only, max 240, default 24)
    """
    _set_context()
    try:
        from weather_google import run_weather_op
        from geocode_google import _cache_lookup, _geocode
        import asyncio

        # Geocode location string to lat/lng
        cached = _cache_lookup(location)
        if cached:
            lat, lng = cached["latitude"], cached["longitude"]
        else:
            geo_result = await asyncio.to_thread(_geocode, location)
            # Re-lookup cache after geocoding (it saves to cache)
            cached = _cache_lookup(location)
            if not cached:
                return f"Could not geocode '{location}': {geo_result}"
            lat, lng = cached["latitude"], cached["longitude"]

        return await run_weather_op(
            operation=forecast_type,
            latitude=float(lat),
            longitude=float(lng),
            days=days,
            hours=hours,
        )
    except ImportError as e:
        return f"Weather plugin not available: {e}"
    except Exception as e:
        return f"Weather error: {e}"


@mcp.tool()
async def places(
    query: str,
    operation: str = "text_search",
    location: str = "",
    radius: int = 5000,
    place_id: str = "",
) -> str:
    """Find businesses and points of interest via Google Places.

    Args:
        query: Search query (e.g. 'coffee shops near downtown')
        operation: 'text_search', 'nearby_search', 'place_details'
        location: Center point for nearby_search (e.g. 'San Francisco') — auto-geocoded
        radius: Search radius in meters (for nearby_search)
        place_id: For place_details
    """
    _set_context()
    try:
        from places_google import run_places_op
        from geocode_google import _cache_lookup, _geocode
        import asyncio

        # Map MCP operation names to run_places_op operation names
        op_map = {"text_search": "search", "nearby_search": "nearby", "place_details": "details"}
        mapped_op = op_map.get(operation, operation)

        # Geocode location string to lat/lng if provided
        lat, lng = 0.0, 0.0
        if location:
            cached = _cache_lookup(location)
            if cached:
                lat, lng = float(cached["latitude"]), float(cached["longitude"])
            else:
                geo_result = await asyncio.to_thread(_geocode, location)
                cached = _cache_lookup(location)
                if cached:
                    lat, lng = float(cached["latitude"]), float(cached["longitude"])

        return await run_places_op(
            operation=mapped_op,
            latitude=lat,
            longitude=lng,
            radius=float(radius),
            query=query,
            place_id=place_id,
        )
    except ImportError as e:
        return f"Places plugin not available: {e}"
    except Exception as e:
        return f"Places error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2b: Complementary search + extraction (use after native tools fail)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def file_extract(
    file_id: str = "",
    url: str = "",
    local_path: str = "",
    prompt: str = "",
) -> str:
    """Extract or interpret any file using Gemini multimodal AI.

    Handles formats Claude Code cannot: audio, video, Office docs (.docx, .xlsx),
    complex PDFs, and images requiring deep interpretation. Provide exactly ONE source.

    Args:
        file_id: Google Drive file ID
        url: HTTP/HTTPS URL to the file
        local_path: Absolute path on the local filesystem
        prompt: Optional focus prompt (e.g. 'transcribe audio', 'extract all tables',
                'summarize key points'). Default: full content extraction.
    """
    _set_context()
    try:
        from tools import get_tool_executor
        executor = get_tool_executor("file_extract")
        if not executor:
            return "file_extract not available (plugin_extract_gemini not loaded)"
        return await executor(
            file_id=file_id or None,
            url=url or None,
            local_path=local_path or None,
            prompt=prompt or None,
        )
    except Exception as e:
        return f"file_extract error: {e}"


@mcp.tool()
async def perplexity_search(
    query: str,
    max_results: int = 5,
) -> str:
    """AI-curated web search via Perplexity with ranked results and citations.

    Use this as a FALLBACK when Claude Code's native WebSearch doesn't yield
    good results. Perplexity provides AI-ranked, deduplicated results with
    source quality scoring.

    Args:
        query: Search query
        max_results: Max results to return (default 5)
    """
    _set_context()
    try:
        from tools import get_tool_executor
        executor = get_tool_executor("perplexity_search")
        if not executor:
            return "perplexity_search not available (plugin not loaded)"
        return await executor(query=query, max_results=max_results)
    except Exception as e:
        return f"perplexity_search error: {e}"


@mcp.tool()
async def xai_search(
    query: str,
) -> str:
    """Web + X/Twitter search via xAI Grok with real-time results and citations.

    Use this as a FALLBACK when Claude Code's native WebSearch doesn't yield
    good results, especially for topics with significant X/Twitter discourse
    or very recent events.

    Args:
        query: Search query
    """
    _set_context()
    try:
        from tools import get_tool_executor
        executor = get_tool_executor("search_xai")
        if not executor:
            return "xai_search not available (plugin not loaded)"
        return await executor(query=query)
    except Exception as e:
        return f"xai_search error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: Cross-system utility tools
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def llm_call(
    model: str,
    prompt: str,
    mode: str = "text",
    sys_prompt: str = "none",
    history: str = "none",
    tool: str = "",
) -> str:
    """Call another LLM through llmem-gw (second opinion, specialist delegation).

    Args:
        model: Target model key (e.g. 'summarizer-gemini', 'gpt5n', 'grok41fr').
               Use llm_list() to see available models.
        prompt: The prompt/message to send
        mode: 'text' (default) for raw response, 'tool' for tool delegation
        sys_prompt: 'none' (clean call), 'target' (model's own system prompt)
        history: 'none' (default, clean single-turn call)
        tool: Tool name for mode='tool' delegation (optional)
    """
    _set_context()
    import agents
    return await agents.llm_call(
        model=model, prompt=prompt, mode=mode,
        sys_prompt=sys_prompt, history=history or "none",
        tool=tool or "",
    )


@mcp.tool()
async def llm_list() -> str:
    """List all available LLM models in the registry with their metadata."""
    _set_context()
    from config import LLM_REGISTRY
    if not LLM_REGISTRY:
        return "(LLM registry not loaded)"
    lines = []
    for key, cfg in LLM_REGISTRY.items():
        if not isinstance(cfg, dict):
            lines.append(f"{key}: (invalid config)")
            continue
        lines.append(
            f"{key}: model_id={cfg.get('model_id') or ''} "
            f"host={(cfg.get('host') or '')[:30]} "
            f"type={cfg.get('type') or ''}"
        )
    return "\n".join(lines) if lines else "(no models)"


@mcp.tool()
async def sms_send(
    message: str,
    phone: str = "",
    name: str = "",
) -> str:
    """Send an SMS message.

    Args:
        message: The SMS text
        phone: Recipient phone in E.164 format (e.g. '+14155551234')
        name: OR recipient name to look up (e.g. 'Lee')
    """
    _set_context()
    from tools import _sms_send_exec
    return await _sms_send_exec(message=message, phone=phone, name=name)


# ═══════════════════════════════════════════════════════════════════════════
# Voice Relay — bidirectional voice↔Claude Code via llmem-gw
# ═══════════════════════════════════════════════════════════════════════════

import asyncio as _asyncio
from collections import deque
import time as _time

# Multi-channel relay queues: channel_name → relay state
# Each Claude Code workspace gets its own channel via workspace_register().
_relay_channels: dict = {}
_relay_msg_counter = 0

# Concurrent polling cap — limits how many voice_relay_check() calls can
# long-poll simultaneously.  Excess callers yield immediately with "(busy)".
# Prevents event loop starvation when 6+ Claude Code sessions poll at once.
_MAX_CONCURRENT_POLLS = 4
_active_polls = 0

# Adaptive backoff: server adjusts effective wait based on time since last message
_RELAY_BACKOFF_TIERS = [
    (120,  5),   # last message < 2 min ago → 5s effective wait (active conversation)
    (300,  15),  # last message 2-5 min ago → 15s (recent activity)
    (None, 30),  # last message > 5 min ago → 30s (idle)
]

# Idle timeout: after this many consecutive empty polls, auto-disable relay
# to stop Claude Code from burning tokens on an idle session.
# Frontends auto-restart the workspace on next user interaction.
_RELAY_MAX_EMPTY_POLLS = 40


def _get_relay(channel: str = "") -> dict:
    """Get or create a relay channel. Uses session workspace if no channel specified."""
    ch = channel or _get_workspace_for_session().get("channel", "default")
    if ch not in _relay_channels:
        _relay_channels[ch] = {
            "enabled": False,
            "inbox": deque(maxlen=50),
            "outbox": deque(maxlen=50),
            "inbox_event": None,
            "last_message_at": 0.0,
            "empty_polls": 0,
        }
    return _relay_channels[ch]


@mcp.tool()
async def voice_relay_mode(
    action: str,
) -> str:
    """Enable or disable voice relay mode. When enabled, you can receive
    messages from the voice frontend and respond to them.

    Args:
        action: 'on' to enable, 'off' to disable, 'status' to check
    """
    relay = _get_relay()
    ch = _get_workspace_for_session().get("channel", "default")
    if action == "on":
        relay["enabled"] = True
        relay["inbox"].clear()
        relay["outbox"].clear()
        relay["last_message_at"] = _time.time()
        relay["empty_polls"] = 0
        return (
            f"Voice relay mode ENABLED on channel '{ch}'.\n"
            "Poll with voice_relay_check() to see incoming messages.\n"
            "Respond with voice_relay_respond(message_id, text).\n"
            "Disable with voice_relay_mode('off') when back at your station."
        )
    elif action == "off":
        relay["enabled"] = False
        return f"Voice relay mode DISABLED on channel '{ch}'."
    elif action == "status":
        enabled = relay["enabled"]
        inbox_count = len(relay["inbox"])
        return f"Voice relay [{ch}]: {'ENABLED' if enabled else 'DISABLED'}, {inbox_count} pending messages"
    return "Invalid action. Use 'on', 'off', or 'status'."


@mcp.tool()
async def voice_relay_check(
    wait: int = 15,
) -> str:
    """Check for incoming voice messages. Long-polls: waits up to `wait` seconds
    for a message to arrive before returning empty.

    During active conversation, messages arrive while you're already waiting —
    near-zero latency. During silence, blocks for `wait` seconds then returns
    so you can loop immediately.

    Args:
        wait: Max seconds to wait for a message (default 15, max 30)
    """
    global _active_polls
    relay = _get_relay()
    if not relay["enabled"]:
        await _asyncio.sleep(2)
        return "(voice relay mode is not enabled — STOP POLLING and exit the loop)"

    # Concurrent polling cap — prevent event loop starvation when many
    # Claude Code sessions poll at once.  Excess callers get a short backoff
    # instead of a long block, keeping the event loop responsive for Slack etc.
    if _active_polls >= _MAX_CONCURRENT_POLLS:
        await _asyncio.sleep(1)
        # Still check inbox in case a message arrived
        if relay["inbox"]:
            pass  # fall through to message delivery below
        else:
            return "(no messages — polling slots full, retrying soon)"

    # Adaptive backoff: use time since last message to determine effective wait
    last_msg = relay["last_message_at"]
    idle_secs = _time.time() - last_msg if last_msg > 0 else 999
    effective_wait = _RELAY_BACKOFF_TIERS[-1][1]
    for threshold, wait_secs in _RELAY_BACKOFF_TIERS:
        if threshold is None or idle_secs < threshold:
            effective_wait = wait_secs
            break

    # If inbox is empty, long-poll until a message arrives or timeout
    if not relay["inbox"]:
        _active_polls += 1
        try:
            evt = relay.get("inbox_event")
            if evt is None:
                evt = _asyncio.Event()
                relay["inbox_event"] = evt
            evt.clear()
            try:
                await _asyncio.wait_for(evt.wait(), timeout=effective_wait)
            except _asyncio.TimeoutError:
                pass
        finally:
            _active_polls -= 1

    if not relay["inbox"]:
        relay["empty_polls"] = relay.get("empty_polls", 0) + 1
        if relay["empty_polls"] >= _RELAY_MAX_EMPTY_POLLS:
            relay["enabled"] = False
            ch = _get_workspace_for_session().get("channel", "default")
            return (
                f"(idle_timeout — relay auto-disabled on channel '{ch}' after "
                f"{_RELAY_MAX_EMPTY_POLLS} empty polls to conserve tokens. "
                "STOP POLLING. The frontend will auto-restart on next user interaction.)"
            )
        return "(no messages)"

    # Messages available — reset idle counter
    relay["empty_polls"] = 0
    lines = [f"## Incoming Voice Messages ({len(relay['inbox'])} pending)\n"]
    for msg in relay["inbox"]:
        age = int(_time.time() - msg["timestamp"])
        line = (
            f"  [{msg['id']}] ({age}s ago) from {msg.get('source', 'voice')}: "
            f"{msg['text']}"
        )
        # Append voice prosody/emotion metadata if present (from xAI STT)
        if msg.get("emotion"):
            em = msg["emotion"]
            line += (
                f"\n    [voice prosody: emotion={em.get('emotion', '?')}, "
                f"confidence={em.get('confidence', '?')}, "
                f"prosody=\"{em.get('prosody', '')}\"]"
            )
        lines.append(line)
    lines.append("\nRespond with voice_relay_respond(message_id=N, text='your response')")
    return "\n".join(lines)


@mcp.tool()
async def voice_relay_respond(
    message_id: int,
    text: str,
) -> str:
    """Respond to a voice relay message. The response will be spoken by the voice frontend.

    Format responses according to your workspace rules (CLAUDE.md).
    For voice-only workspaces: keep it short and conversational.
    For chat/GED workspaces: use full markdown, math, and formatting.

    Args:
        message_id: The id from voice_relay_check()
        text: Your response (keep it terse, voice-appropriate)
    """
    relay = _get_relay()
    if not relay["enabled"]:
        return "(voice relay mode is not enabled)"

    # Find and remove the message from inbox
    found = None
    for msg in relay["inbox"]:
        if msg["id"] == message_id:
            found = msg
            break

    if not found:
        return f"Message id={message_id} not found in inbox"

    relay["inbox"].remove(found)

    # Add to outbox for voice frontend to pick up
    relay["outbox"].append({
        "id": message_id,
        "reply_to": found["text"][:100],
        "text": text,
        "timestamp": _time.time(),
    })

    # Save relay turn to memory (same as conv_log does for direct turns)
    _set_context()  # ensure DB context is set for the save
    try:
        from memory import save_conversation_turn
        topic_slug = await _generate_topic_slug(found["text"], text)
        asst_text = f"<<{topic_slug}>>{text}" if topic_slug else text
        await save_conversation_turn(
            user_text=found["text"],
            assistant_text=asst_text,
            session_id=f"{_CLIENT_ID_PREFIX}-relay",
            importance=5,  # slightly higher than default conv_log (4) since user actively asked
        )
        log.info(f"voice_relay: saved turn to memory topic={topic_slug}")
    except Exception as e:
        log.warning(f"voice_relay: conv_log save failed: {e}")

    return f"Response sent for message {message_id}. Voice frontend will speak it."


# HTTP endpoints for voice frontend to submit/retrieve relay messages

async def endpoint_voice_relay_submit(request: Request) -> JSONResponse:
    """Voice frontend submits a message for Claude Code.

    Body (JSON):
        text    : str  (the transcribed voice message)
        source  : str  (optional, e.g. 'voice', 'slack', 'shell')
        channel : str  (optional, relay channel name — default 'default')

    Returns immediately. Claude Code picks up via voice_relay_check().
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    channel = payload.get("channel", "default")
    relay = _get_relay(channel)

    if not relay["enabled"]:
        return JSONResponse({"error": f"Voice relay not enabled on channel '{channel}'"}, status_code=503)

    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)

    global _relay_msg_counter
    _relay_msg_counter += 1
    msg = {
        "id": _relay_msg_counter,
        "text": text,
        "source": payload.get("source", "voice"),
        "timestamp": _time.time(),
    }
    # Pass through emotion/prosody metadata from xAI STT if present
    if payload.get("emotion"):
        msg["emotion"] = payload["emotion"]
    relay["inbox"].append(msg)
    relay["last_message_at"] = _time.time()
    log.info(f"voice_relay[{channel}]: inbound msg #{msg['id']} from {msg['source']}: {text[:80]}")

    # Wake any long-polling voice_relay_check() call
    evt = relay.get("inbox_event")
    if evt:
        evt.set()

    return JSONResponse({"status": "queued", "message_id": msg["id"], "channel": channel})


async def endpoint_voice_relay_poll(request: Request) -> JSONResponse:
    """Voice frontend polls for Claude Code's response.

    Query params:
        wait    : int  (optional, seconds to long-poll, default 0 = instant)
        channel : str  (optional, relay channel name — default 'default')

    Returns the oldest outbox message, or empty if none.
    """
    channel = request.query_params.get("channel", "default")
    relay = _get_relay(channel)

    if not relay["enabled"]:
        return JSONResponse({"error": f"Voice relay not enabled on channel '{channel}'"}, status_code=503)

    wait = int(request.query_params.get("wait", "0"))

    # Long-poll: wait up to N seconds for a response
    if wait > 0 and not relay["outbox"]:
        deadline = _time.time() + min(wait, 30)
        while _time.time() < deadline and not relay["outbox"]:
            await _asyncio.sleep(0.5)

    if not relay["outbox"]:
        return JSONResponse({"status": "empty"})

    msg = relay["outbox"].popleft()
    return JSONResponse({
        "status": "ok",
        "message_id": msg["id"],
        "reply_to": msg.get("reply_to", ""),
        "text": msg["text"],
        "channel": channel,
    })


async def endpoint_voice_relay_status(request: Request) -> JSONResponse:
    """Check voice relay status. Accepts ?channel= param or returns all channels."""
    channel = request.query_params.get("channel", "")
    if channel:
        relay = _get_relay(channel)
        empty = relay.get("empty_polls", 0)
        return JSONResponse({
            "channel": channel,
            "enabled": relay["enabled"],
            "inbox_count": len(relay["inbox"]),
            "outbox_count": len(relay["outbox"]),
            "empty_polls": empty,
            "idle_limit": _RELAY_MAX_EMPTY_POLLS,
            "polls_remaining": max(0, _RELAY_MAX_EMPTY_POLLS - empty) if relay["enabled"] else 0,
        })
    # No channel specified — return all active channels
    channels = {}
    for ch, relay in _relay_channels.items():
        if relay["enabled"]:
            channels[ch] = {
                "enabled": True,
                "inbox_count": len(relay["inbox"]),
                "outbox_count": len(relay["outbox"]),
            }
    # For backwards compat, also return top-level "enabled" if any channel is active
    any_enabled = any(r["enabled"] for r in _relay_channels.values()) if _relay_channels else False
    return JSONResponse({
        "enabled": any_enabled,
        "channels": channels,
    })


async def endpoint_voice_relay_disable(request: Request) -> JSONResponse:
    """Disable a relay channel and clear its queues.

    Body (JSON):
        channel : str  (relay channel name)
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    channel = payload.get("channel", "")
    if not channel:
        return JSONResponse({"error": "Missing channel"}, status_code=400)

    if channel in _relay_channels:
        _relay_channels[channel]["enabled"] = False
        _relay_channels[channel]["inbox"].clear()
        _relay_channels[channel]["outbox"].clear()
    return JSONResponse({"status": "disabled", "channel": channel})


# ═══════════════════════════════════════════════════════════════════════════
# Health check — used by bash scripts to detect stale MCP connections
# ═══════════════════════════════════════════════════════════════════════════


async def endpoint_mcp_health(request: Request) -> JSONResponse:
    """Check if a workspace's MCP + relay pipeline is operational.

    Used by start scripts to decide warm restart vs cold restart.
    "healthy" means the relay channel is enabled — i.e. a Claude Code session
    successfully called voice_relay_mode("on") over MCP.  This is the
    strongest proof of a working SSE connection without requiring explicit
    workspace_register() (which only GED subjects use).

    Query params:
        channel : str  (relay channel name, e.g. 'ged-math' or 'default')
    """
    channel = request.query_params.get("channel", "default")

    # Check workspace registration (optional — only GED subjects use it)
    ws_registered = channel in _workspaces
    ws_info = _workspaces.get(channel, {})

    # Check relay state — this is the primary health signal.
    # If relay is enabled, Claude Code has a working MCP SSE connection
    # and successfully called voice_relay_mode("on").
    relay = _relay_channels.get(channel, {})
    relay_enabled = relay.get("enabled", False)

    # Check if any MCP session is explicitly mapped to this workspace
    mcp_session_found = False
    for sid, ws in _session_workspaces.items():
        if ws.get("channel") == channel:
            mcp_session_found = True
            break

    return JSONResponse({
        "channel": channel,
        "workspace_registered": ws_registered,
        "mcp_session": mcp_session_found,
        "relay_enabled": relay_enabled,
        "registered_at": ws_info.get("registered_at"),
        "healthy": relay_enabled,
    })


# ═══════════════════════════════════════════════════════════════════════════
# GED workspace auto-launcher
# ═══════════════════════════════════════════════════════════════════════════

_GED_START_SCRIPT = os.path.expanduser("~/projects/samaritan-ged/ged-start.sh")
_CLAUDE_START_SCRIPT = os.path.expanduser("~/projects/samaritan-work/claude-start.sh")

# Map channel names to launch scripts and arguments
_CHANNEL_LAUNCHERS = {
    "ged-math":    (_GED_START_SCRIPT, "math"),
    "ged-reading": (_GED_START_SCRIPT, "reading"),
    "ged-writing": (_GED_START_SCRIPT, "writing"),
    "ged-science": (_GED_START_SCRIPT, "science"),
    "ged-social":  (_GED_START_SCRIPT, "social"),
    "default":     (_CLAUDE_START_SCRIPT, None),
}


async def endpoint_ged_start(request: Request) -> JSONResponse:
    """Start a Claude Code workspace on demand.

    Body (JSON):
        channel : str  (relay channel name, e.g. 'ged-math' or 'default')

    Launches the workspace in tmux if not already running.
    Polls until the relay is enabled.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    channel = payload.get("channel", "")
    launcher = _CHANNEL_LAUNCHERS.get(channel)
    if not launcher:
        return JSONResponse({"error": f"Unknown channel: {channel}"}, status_code=400)

    script, arg = launcher

    # Check if already running — dispatch channels use tmux check, relay channels check relay
    if channel in _DISPATCH_TMUX_SESSIONS:
        dispatch = _get_dispatch(channel)
        tmux_session = dispatch["tmux_session"]
        proc_check = await _asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", tmux_session,
            stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
        )
        if await proc_check.wait() == 0:
            # tmux exists — check if Claude is alive
            proc_pane = await _asyncio.create_subprocess_exec(
                "tmux", "list-panes", "-t", tmux_session, "-F", "#{pane_pid}",
                stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc_pane.communicate()
            pane_pid = stdout.decode().strip().split("\n")[0]
            if pane_pid:
                proc_pg = await _asyncio.create_subprocess_exec(
                    "pgrep", "-P", pane_pid, "-f", "claude",
                    stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
                )
                if await proc_pg.wait() == 0:
                    return JSONResponse({"status": "already_running", "channel": channel})
    else:
        relay = _get_relay(channel)
        if relay["enabled"]:
            return JSONResponse({"status": "already_running", "channel": channel})

    # Launch via start script — fire-and-forget so we don't block the event loop.
    # The script handles its own prompt detection and relay activation;
    # we just poll _relay_channels until it comes up.
    if not os.path.exists(script):
        return JSONResponse({"error": f"Start script not found: {script}"}, status_code=500)

    cmd = [script, arg] if arg else [script]
    try:
        proc = await _asyncio.create_subprocess_exec(
            *cmd,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        log.info(f"workspace_start: launched {channel} (pid {proc.pid})")
    except Exception as e:
        return JSONResponse({"error": f"Launch failed: {e}"}, status_code=500)

    # Poll until ready or the script exits with an error.
    # Fast checks first (warm restart ~5s), then slower (cold start ~55s).
    for i in range(60):
        await _asyncio.sleep(0.5 if i < 20 else 1.0)

        # Check readiness: dispatch channels check tmux, relay channels check relay
        if channel in _DISPATCH_TMUX_SESSIONS:
            tmux_sess = _get_dispatch(channel)["tmux_session"]
            chk = await _asyncio.create_subprocess_exec(
                "tmux", "has-session", "-t", tmux_sess,
                stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
            )
            if await chk.wait() == 0:
                p2 = await _asyncio.create_subprocess_exec(
                    "tmux", "list-panes", "-t", tmux_sess, "-F", "#{pane_pid}",
                    stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
                )
                out, _ = await p2.communicate()
                ppid = out.decode().strip().split("\n")[0]
                if ppid:
                    p3 = await _asyncio.create_subprocess_exec(
                        "pgrep", "-P", ppid, "-f", "claude",
                        stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
                    )
                    if await p3.wait() == 0:
                        return JSONResponse({"status": "started", "channel": channel})
        else:
            relay = _get_relay(channel)
            if relay["enabled"]:
                return JSONResponse({"status": "started", "channel": channel})

        # If the script already exited with an error, stop waiting
        if proc.returncode is not None and proc.returncode != 0:
            stderr = (await proc.stderr.read()).decode().strip() if proc.stderr else ""
            log.warning(f"workspace_start: {channel} script failed (rc={proc.returncode}): {stderr}")
            return JSONResponse({
                "error": f"Start script exited with code {proc.returncode}",
                "detail": stderr,
            }, status_code=500)

    return JSONResponse({
        "status": "timeout",
        "channel": channel,
        "message": "Workspace launched but not yet ready. May need more time.",
    })


# ═══════════════════════════════════════════════════════════════════════════
# RC-style tmux dispatch (replaces voice relay polling for Claude mode)
# ═══════════════════════════════════════════════════════════════════════════

_dispatch_channels: Dict[str, dict] = {}

_DISPATCH_TMUX_SESSIONS = {
    "default":     "samaritan-work",
    "ged-math":    "ged:math",
    "ged-reading": "ged:reading",
    "ged-writing": "ged:writing",
    "ged-science": "ged:science",
    "ged-social":  "ged:social",
}


_dispatch_locks: Dict[str, _asyncio.Lock] = {}

def _get_dispatch(channel: str = "default") -> dict:
    if channel not in _dispatch_channels:
        _dispatch_channels[channel] = {
            "pending_prompt": None,
            "response_event": None,
            "response_text": None,
            "last_submit_at": 0.0,
            "tmux_session": _DISPATCH_TMUX_SESSIONS.get(channel, "samaritan-work"),
        }
        _dispatch_locks[channel] = _asyncio.Lock()
    return _dispatch_channels[channel]


def _format_dispatch_prompt(text: str, source: str = "voice",
                            emotion: dict = None) -> str:
    """Format user text with source/emotion metadata prefix for Claude."""
    prefix = f"[{source}"
    if emotion:
        em_label = emotion.get("emotion", "neutral")
        em_conf = emotion.get("confidence", "")
        prefix += f"|emotion:{em_label}"
        if em_conf:
            prefix += f"({em_conf})"
        prosody = emotion.get("prosody", "")
        if prosody:
            prefix += f"|prosody:{prosody}"
    prefix += "]"
    return f"{prefix} {text}"


async def endpoint_claude_submit(request: Request) -> JSONResponse:
    """Submit a message to Claude Code via tmux send-keys.

    Body: { text, source, emotion, channel }
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    text = payload.get("text", "").strip()
    source = payload.get("source", "voice")
    emotion = payload.get("emotion")
    channel = payload.get("channel", "default")

    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)

    dispatch = _get_dispatch(channel)
    tmux_session = dispatch["tmux_session"]

    # Check tmux session is alive
    proc = await _asyncio.create_subprocess_exec(
        "tmux", "has-session", "-t", tmux_session,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    if await proc.wait() != 0:
        return JSONResponse(
            {"error": f"tmux session '{tmux_session}' not found"},
            status_code=503,
        )

    # Format prompt with metadata prefix
    prompt = _format_dispatch_prompt(text, source, emotion)

    # Prepare response capture
    dispatch["response_text"] = None
    dispatch["response_event"] = _asyncio.Event()
    dispatch["pending_prompt"] = text
    dispatch["last_submit_at"] = _time.time()

    # Send to tmux (literal mode to avoid escaping issues)
    proc = await _asyncio.create_subprocess_exec(
        "tmux", "send-keys", "-l", "-t", tmux_session, prompt,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    rc = await proc.wait()
    if rc != 0:
        return JSONResponse({"error": "tmux send-keys failed"}, status_code=503)

    # Send Enter separately (send-keys -l doesn't interpret special keys)
    proc2 = await _asyncio.create_subprocess_exec(
        "tmux", "send-keys", "-t", tmux_session, "Enter",
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    await proc2.wait()

    log.info(f"dispatch: sent to {tmux_session} ch={channel} src={source} "
             f"emotion={emotion.get('emotion') if emotion else 'none'} "
             f"text={text[:60]}")

    return JSONResponse({"status": "submitted", "channel": channel})


async def endpoint_claude_poll(request: Request) -> JSONResponse:
    """Poll for Claude's response after a tmux dispatch.

    Query params: wait (seconds), channel
    """
    channel = request.query_params.get("channel", "default")
    wait = min(int(request.query_params.get("wait", "10")), 30)

    dispatch = _get_dispatch(channel)
    lock = _dispatch_locks.get(channel, _asyncio.Lock())

    # Atomic consume: only one poller gets the response
    async def _try_consume():
        async with lock:
            if dispatch["response_text"]:
                text = dispatch["response_text"]
                dispatch["response_text"] = None
                return text
        return None

    # Check if response already available
    text = await _try_consume()
    if text:
        return JSONResponse({"status": "ok", "text": text})

    # Long-poll: wait on event if active, otherwise just sleep for the wait period.
    # This prevents the background poller from flooding when no dispatch is pending.
    evt = dispatch.get("response_event")
    if evt and not evt.is_set():
        try:
            await _asyncio.wait_for(evt.wait(), timeout=wait)
        except _asyncio.TimeoutError:
            pass
    else:
        # No active dispatch — sleep to throttle background polling
        await _asyncio.sleep(wait)

    text = await _try_consume()
    if text:
        return JSONResponse({"status": "ok", "text": text})

    return JSONResponse({"status": "empty"})


async def endpoint_claude_status(request: Request) -> JSONResponse:
    """Check if Claude Code is alive in its tmux session."""
    channel = request.query_params.get("channel", "default")
    dispatch = _get_dispatch(channel)
    tmux_session = dispatch["tmux_session"]

    # Check tmux session exists
    proc = await _asyncio.create_subprocess_exec(
        "tmux", "has-session", "-t", tmux_session,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    tmux_alive = (await proc.wait() == 0)

    # Check if Claude process is running inside the pane
    claude_alive = False
    if tmux_alive:
        proc2 = await _asyncio.create_subprocess_exec(
            "tmux", "list-panes", "-t", tmux_session, "-F", "#{pane_pid}",
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc2.communicate()
        pane_pid = stdout.decode().strip().split("\n")[0]
        if pane_pid:
            proc3 = await _asyncio.create_subprocess_exec(
                "pgrep", "-P", pane_pid, "-f", "claude",
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            claude_alive = (await proc3.wait() == 0)

    return JSONResponse({
        "channel": channel,
        "tmux_session": tmux_session,
        "tmux_alive": tmux_alive,
        "claude_alive": claude_alive,
        "enabled": claude_alive,  # backward compat with frontend relay check
    })


# ═══════════════════════════════════════════════════════════════════════════
# Conversation logging endpoint (called by Claude Code hook)
# ═══════════════════════════════════════════════════════════════════════════

async def endpoint_conv_log(request: Request) -> JSONResponse:
    """
    Log a conversation turn (user prompt + assistant response) to short-term memory.

    Called by the Claude Code Stop hook after each assistant response.
    Mirrors what conv_log:true does in the normal LLM pipeline.

    Body (JSON):
        user_text      : str  (the user's prompt)
        assistant_text : str  (the assistant's response)
        session_id     : str  (optional, for attribution)
        importance     : int  (optional, default 4)
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    user_text_raw = payload.get("user_text", "").strip()
    assistant_text = payload.get("assistant_text", "").strip()

    if not user_text_raw and not assistant_text:
        return JSONResponse({"status": "skipped", "reason": "empty"})

    # Strip dispatch prefix for memory storage, but keep raw for dispatch matching
    # Match any dispatch prefix: [voice|...], [typed], [chat-ged], etc.
    _dp_match = _re.match(r'^\[[a-z][\w-]*[^\]]*\]\s*', user_text_raw)
    user_text = user_text_raw[_dp_match.end():] if _dp_match else user_text_raw

    # Filter automated loop/relay polling noise — these are not real conversation.
    # Kept here (next to the relay code that generates the noise) rather than in
    # the client-side hook, so there's one maintenance point.
    _noise_markers = (
        "voice_relay_check",    # relay polling prompt
        "# /loop",              # loop scheduling meta-prompt
        "(no messages)",        # empty relay poll response
        "CronCreate",           # cron scheduling internals
    )
    combined_check = user_text[:300] + assistant_text[:300]
    if any(marker in combined_check for marker in _noise_markers):
        return JSONResponse({"status": "skipped", "reason": "automated_noise"})

    session_id = payload.get("session_id", f"{_CLIENT_ID_PREFIX}-mcp")
    importance = int(payload.get("importance", 4))

    _set_context()

    try:
        from memory import save_conversation_turn, _normalize_topic, load_topic_list

        # Generate a topic slug server-side since Claude Code doesn't prepend <<topic>>
        # Strategy: extract key nouns from user text, fuzzy-match against existing topics
        topic_slug = await _generate_topic_slug(user_text, assistant_text)
        if topic_slug:
            # Prepend the tag so save_conversation_turn's _extract_topic_tag() finds it
            assistant_text = f"<<{topic_slug}>>" + assistant_text

        user_id, asst_id, topic = await save_conversation_turn(
            user_text=user_text,
            assistant_text=assistant_text,
            session_id=session_id,
            importance=importance,
        )
        log.info(
            f"conv_log: topic={topic} user_id={user_id} asst_id={asst_id} "
            f"session={session_id}"
        )

        # Write live emotion tag if provided (from Claude Code inline inference)
        emotion_written = False
        user_emotion = payload.get("user_emotion")
        if user_emotion and user_id:
            try:
                from emotions import _write_emotion, CORE_EMOTIONS
                emotion_item = {
                    "memory_id": user_id,
                    "_memory_table": "shortterm",
                    "core_emotion": user_emotion.get("core_emotion", ""),
                    "emotion_label": user_emotion.get("emotion_label", ""),
                    "intensity": user_emotion.get("intensity", 0.5),
                    "confidence": 0.9,  # high confidence — inferred live with full context
                    "context": "live inference by Claude Code with conversation context",
                }
                emotion_written = await _write_emotion(emotion_item, confidence_threshold=0.0)
                if emotion_written:
                    log.info(f"conv_log: live emotion '{user_emotion.get('emotion_label')}' written for user_id={user_id}")
            except Exception as e:
                log.warning(f"conv_log: emotion write failed: {e}")

        # Signal dispatch channel if a pending prompt is waiting for a response.
        # Only match conv_logs that contain the dispatch prefix (proves it came from
        # the tmux dispatch, not from VS Code or another session).
        if _dp_match:
            for ch_name, dispatch in _dispatch_channels.items():
                if (dispatch.get("pending_prompt")
                        and dispatch.get("response_event")
                        and dispatch["last_submit_at"] > 0
                        and _time.time() - dispatch["last_submit_at"] < 120
                        and dispatch["pending_prompt"][:50] in user_text):
                    # Strip topic tag from assistant_text before delivering
                    clean_asst = _re.sub(r'^<<[^>]*>>', '', assistant_text).strip()
                    # Skip truly empty responses (tool-call noise with no text)
                    if not clean_asst:
                        log.info(f"dispatch: skipping empty response for '{ch_name}'")
                        continue
                    log.info(f"dispatch: raw asst_text={assistant_text[:100]!r} clean={clean_asst[:100]!r}")
                    dispatch["response_text"] = clean_asst
                    dispatch["pending_prompt"] = None
                    dispatch["last_submit_at"] = 0  # prevent re-match
                    evt = dispatch["response_event"]
                    if evt:
                        evt.set()
                    log.info(f"dispatch: response captured for channel '{ch_name}' "
                             f"({len(clean_asst)} chars)")
                    break

        return JSONResponse({
            "status": "ok",
            "user_id": user_id,
            "asst_id": asst_id,
            "topic": topic,
            "emotion_written": emotion_written,
        })
    except Exception as e:
        log.error(f"conv_log error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════
# Plugin class
# ═══════════════════════════════════════════════════════════════════════════

class Plugin(BasePlugin):
    PLUGIN_NAME = "plugin_mcp_direct"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "client_interface"
    DESCRIPTION = "Direct MCP access for Claude Code — bypasses LLM routing"
    DEPENDENCIES = ["mcp"]
    ENV_VARS = []

    def __init__(self):
        self.port = 8769
        self.host = "0.0.0.0"

    def init(self, config: dict) -> bool:
        self.port = config.get("mcp_direct_port", 8769)
        self.host = config.get("mcp_direct_host", "0.0.0.0")
        log.info(f"MCP Direct plugin initialized — SSE on port {self.port}")
        return True

    def shutdown(self) -> None:
        log.info("MCP Direct plugin shutting down")

    def get_routes(self) -> List[Route]:
        """Mount the FastMCP SSE app routes + conv_log + voice relay endpoints."""
        sse_app = mcp.sse_app()
        routes = list(sse_app.routes)
        routes.append(Route("/conv_log", endpoint_conv_log, methods=["POST"]))
        routes.append(Route("/voice_relay/submit", endpoint_voice_relay_submit, methods=["POST"]))
        routes.append(Route("/voice_relay/poll", endpoint_voice_relay_poll, methods=["GET"]))
        routes.append(Route("/voice_relay/status", endpoint_voice_relay_status, methods=["GET"]))
        routes.append(Route("/ged/start", endpoint_ged_start, methods=["POST"]))
        routes.append(Route("/voice_relay/disable", endpoint_voice_relay_disable, methods=["POST"]))
        routes.append(Route("/claude/submit", endpoint_claude_submit, methods=["POST"]))
        routes.append(Route("/claude/poll", endpoint_claude_poll, methods=["GET"]))
        routes.append(Route("/claude/status", endpoint_claude_status, methods=["GET"]))
        routes.append(Route("/mcp/health", endpoint_mcp_health, methods=["GET"]))
        return routes

    def get_config(self) -> dict:
        return {
            "port": self.port,
            "host": self.host,
            "name": "MCP Direct (Claude Code)",
        }

    def get_help(self) -> str:
        return (
            "\n**MCP Direct (Claude Code)**\n"
            f"  SSE endpoint on port {self.port} — connect from Claude Code\n"
            f"  Tools: memory, goals, plans, beliefs, db_query, drive, calendar, etc.\n"
        )
