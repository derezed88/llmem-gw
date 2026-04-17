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
import subprocess as _subprocess
import time as _cogn_time
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


# ---------------------------------------------------------------------------
# Activity-driven cognition — wake samaritan-cognition on key events
# ---------------------------------------------------------------------------

_last_cogn_poke: float = 0.0
_COGN_POKE_COOLDOWN: int = 60       # minimum seconds between tmux pokes
_COGN_GOAL_ID: int = 0              # cached ID for "Ongoing Cognitive Processing" goal
_cogn_turn_counter: int = 0         # conv_log turns since last reflection step
_COGN_REFLECT_EVERY: int = 5        # queue a reflection step every N real turns


def _poke_cognition_session() -> None:
    """Send a wake signal to the samaritan-cognition tmux session (debounced)."""
    global _last_cogn_poke
    now = _cogn_time.monotonic()
    if now - _last_cogn_poke < _COGN_POKE_COOLDOWN:
        return
    _last_cogn_poke = now
    try:
        _subprocess.Popen(
            ["tmux", "send-keys", "-t", "samaritan-cognition",
             "Process pending cognition steps", "Enter"],
            stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
        )
        log.info("cogn_poke: sent wake signal to samaritan-cognition")
    except Exception as e:
        log.warning(f"cogn_poke: failed to poke samaritan-cognition: {e}")


async def _queue_cogn_step(description: str) -> None:
    """Queue a task step for samaritan-cognition under the ongoing cognition goal."""
    global _COGN_GOAL_ID
    from database import fetch_dicts, execute_insert
    try:
        # Get or create the singleton cognition goal in mymcp
        if not _COGN_GOAL_ID:
            rows = await fetch_dicts(
                "SELECT id FROM mymcp.samaritan_goals "
                "WHERE title = 'Ongoing Cognitive Processing' AND status = 'active' LIMIT 1"
            )
            if rows:
                _COGN_GOAL_ID = rows[0]["id"]
            else:
                _COGN_GOAL_ID = await execute_insert(
                    "INSERT INTO mymcp.samaritan_goals "
                    "(title, description, status, importance, source, session_id) VALUES "
                    "('Ongoing Cognitive Processing', "
                    "'Continuous cognitive processing: reflection, contradiction detection, goal health', "
                    "'active', 8, 'assistant', 'plugin-mcp-direct')"
                )
                log.info(f"cogn: created cognition goal id={_COGN_GOAL_ID}")

        # Determine step_order from current pending count
        rows = await fetch_dicts(
            f"SELECT COUNT(*) AS cnt FROM mymcp.samaritan_plans "
            f"WHERE goal_id = {_COGN_GOAL_ID} AND status = 'pending'"
        )
        next_order = (rows[0]["cnt"] if rows else 0) + 1

        d = description.replace("'", "''")
        step_id = await execute_insert(
            f"INSERT INTO mymcp.samaritan_plans "
            f"(goal_id, step_order, description, status, step_type, target, approval, source, session_id) "
            f"VALUES ({_COGN_GOAL_ID}, {next_order}, '{d}', 'pending', 'task', "
            f"'claude-cognition', 'approved', 'assistant', 'plugin-mcp-direct')"
        )
        log.info(f"cogn: queued step id={step_id} [{next_order}]: {description[:60]}")
        _poke_cognition_session()
    except Exception as e:
        log.warning(f"cogn: failed to queue step: {e}")


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
    # For chat workspaces: restore dispatch routing lost across server restarts.
    # _dynamic_channels is in-memory only; workspace_register is the reliable hook
    # because Claude calls it at every session start regardless of how it was launched.
    if name.startswith("chat-"):
        chat_name = name[5:]  # strip "chat-" prefix → e.g. "myproject"
        _register_chat_channel(name, f"chat_{chat_name}", database)
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
    "tell said say really actually please thanks thank help much well yeah yes "
    # System/tool words that produce generic, non-distinguishing slugs
    "claude claud samaritan assistant system mcp session conversation turn message "
    "response tool code llm model memory context step goal result output "
    "hey okay sure good great okay right now".split()
)


import re as _re


async def _generate_topic_slug(user_text: str, assistant_text: str) -> str | None:
    """Generate a 2-word topic slug and fuzzy-match against existing topics.

    Target format: 'job-seeking', 'health-advice', 'plumbing-issue' (2 hyphenated words).
    Strongly prefers reusing existing topics over creating new ones.

    For new slugs (no fuzzy match), calls gpt-4o-mini at temperature=0 for
    semantic, consistent naming rather than frequency-based word picking.
    """
    # Try to match against existing topics FIRST (aggressive fuzzy match)
    try:
        from memory import load_topic_list, _fuzzy_match_topics
        existing = await load_topic_list()
        if existing:
            matches = _fuzzy_match_topics(user_text[:200], existing, threshold=0.75)
            if matches:
                return matches[0]  # Best match — reuse existing slug
    except Exception:
        pass

    # No existing match — generate a new slug via LLM at temperature=0
    asst_first = assistant_text.split('\n')[0][:100] if assistant_text else ""
    prompt = (
        "Generate a 2-word hyphenated topic slug for this conversation snippet.\n"
        "Format: lowercase-words (e.g. 'server-restart', 'belief-audit', 'weather-wilmington').\n"
        "Be specific and semantic — capture what this exchange is actually about.\n"
        "Return ONLY the slug. No explanation, no punctuation, no quotes.\n\n"
        f"USER: {user_text[:300]}\n"
        f"ASSISTANT: {asst_first}"
    )
    try:
        import asyncio as _aio_slug
        from config import LLM_REGISTRY
        from langchain_openai import ChatOpenAI as _ChatOpenAI
        from langchain_core.messages import HumanMessage as _HM_slug

        _cfg = LLM_REGISTRY["gpt4om"]
        _llm = _ChatOpenAI(
            model=_cfg["model_id"],
            base_url=_cfg.get("host"),
            api_key=_cfg.get("key") or "no-key-required",
            temperature=0,
            timeout=5,
        )
        resp = await _aio_slug.wait_for(
            _llm.ainvoke([_HM_slug(content=prompt)]),
            timeout=6.0,
        )
        raw = (resp.content or "").strip().lower()
        # Sanitize: keep only word chars and hyphens, trim to 2 words max
        raw = _re.sub(r'[^a-z0-9-]', '-', raw)
        raw = _re.sub(r'-+', '-', raw).strip('-')
        # If model returned more than 2 words, take first 2
        parts = raw.split('-')
        raw_slug = "-".join(parts[:2]) if len(parts) > 2 else raw
        if not raw_slug:
            return None
        # Final normalize pass — might still match an existing topic
        try:
            from memory import _normalize_topic
            return await _normalize_topic(raw_slug)
        except Exception:
            return raw_slug
    except Exception as _e:
        log.debug(f"_generate_topic_slug: LLM fallback failed ({_e}), returning None")
        return None


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
            "SELECT COUNT(*) as total FROM samaritan_emotions WHERE subject='user'"
        )
        if total:
            lines.append(f"**Total user emotions tracked**: {total[0]['total']}")

        # --- Samaritan affect state ---
        # Retrieve recent self-assessed emotions with decay calculation
        # Half-lives per cluster (hours):
        # hostile-anger/fear-and-overwhelm: 2h, despair-and-shame/depleted-disengagement: 4h,
        # vigilant-suspicion: 3h, exuberant-joy/competitive-pride: 6h,
        # compassionate-gratitude/peaceful-contentment/playful-amusement: 8h
        sam_recent = await fetch_dicts(
            "SELECT emotion_label, cluster, intensity, created_at, "
            "TIMESTAMPDIFF(MINUTE, created_at, NOW()) AS age_minutes "
            "FROM samaritan_emotions "
            "WHERE subject='samaritan' AND created_at >= NOW() - INTERVAL 24 HOUR "
            "ORDER BY created_at DESC LIMIT 10"
        )
        if sam_recent:
            _half_lives = {
                "hostile-anger": 120, "fear-and-overwhelm": 120,
                "despair-and-shame": 240, "depleted-disengagement": 240,
                "vigilant-suspicion": 180,
                "exuberant-joy": 360, "competitive-pride": 360,
                "compassionate-gratitude": 480, "peaceful-contentment": 480,
                "playful-amusement": 480,
            }
            import math
            sam_lines = []
            for row in sam_recent[:5]:
                age_min = float(row.get("age_minutes", 0))
                cluster = row.get("cluster", "")
                hl = _half_lives.get(cluster, 360)
                raw_int = float(row.get("intensity", 0.5))
                decayed = raw_int * math.pow(0.5, age_min / hl)
                if decayed >= 0.05:  # Only show non-negligible
                    age_str = f"{int(age_min)}m" if age_min < 60 else f"{age_min/60:.1f}h"
                    sam_lines.append(
                        f"{row['emotion_label']} ({cluster}, "
                        f"raw={raw_int:.1f}, decayed={decayed:.2f}, age={age_str})"
                    )
            if sam_lines:
                lines.append("")
                lines.append("### Samaritan Affect State")
                for sl in sam_lines:
                    lines.append(f"  {sl}")
                # Dominant cluster after decay
                cluster_totals = {}
                for row in sam_recent:
                    age_min = float(row.get("age_minutes", 0))
                    cluster = row.get("cluster", "")
                    hl = _half_lives.get(cluster, 360)
                    raw_int = float(row.get("intensity", 0.5))
                    decayed = raw_int * math.pow(0.5, age_min / hl)
                    cluster_totals[cluster] = cluster_totals.get(cluster, 0) + decayed
                if cluster_totals:
                    dominant = max(cluster_totals, key=cluster_totals.get)
                    lines.append(f"  **Dominant cluster**: {dominant} (weight={cluster_totals[dominant]:.2f})")

    except Exception as e:
        return ""

    if not lines:
        return ""

    return "## Emotional Context\n\n" + "\n".join(f"  {l}" for l in lines)


async def _load_schedule_impact_context(query: str = "") -> str:
    """Surface memories tagged with schedule-impact affects_tags.

    Two passes:
    1. Query-driven: if the query contains routine keywords, fetch memories
       tagged for those specific routines (last 72h).
    2. Ambient: fetch any memory tagged with a non-empty affects_tags in the
       last 24h — catches active impacts the query didn't mention.

    Combined results are deduped by id and returned as a section.
    """
    from database import fetch_dicts
    from memory import _ST

    lines = []
    seen_ids: set = set()

    try:
        # Pass 1: query-driven tag lookup
        if query:
            query_tags = _detect_affects_tags(query)
            for tag in query_tags:
                tag_esc = tag.replace("'", "''")
                rows = await fetch_dicts(
                    f"SELECT id, topic, content, importance, affects_tags, "
                    f"effect_duration_days, effect_recurs, created_at "
                    f"FROM {_ST()} "
                    f"WHERE JSON_CONTAINS(affects_tags, '\"{tag_esc}\"') "
                    f"AND created_at >= NOW() - INTERVAL 72 HOUR "
                    f"ORDER BY importance DESC, created_at DESC LIMIT 5"
                )
                for r in rows:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        lines.append(r)

        # Pass 2: ambient — tagged memories still within their effect window.
        # If effect_duration_days is set, use it; otherwise default to 24h.
        ambient = await fetch_dicts(
            f"SELECT id, topic, content, importance, affects_tags, "
            f"effect_duration_days, effect_recurs, created_at "
            f"FROM {_ST()} "
            f"WHERE affects_tags IS NOT NULL "
            f"AND affects_tags != '[]' "
            f"AND affects_tags != 'null' "
            f"AND ("
            f"  (effect_duration_days IS NOT NULL AND "
            f"   created_at >= NOW() - INTERVAL effect_duration_days DAY) "
            f"  OR "
            f"  (effect_duration_days IS NULL AND "
            f"   created_at >= NOW() - INTERVAL 24 HOUR)"
            f") "
            f"ORDER BY importance DESC, created_at DESC LIMIT 8"
        )
        for r in ambient:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                lines.append(r)

    except Exception:
        return ""

    if not lines:
        return ""

    out = ["## Schedule Impact (tagged memories)"]
    for r in lines:
        tags = r.get("affects_tags") or "[]"
        if isinstance(tags, str):
            try:
                import json as _json
                tags = ", ".join(_json.loads(tags))
            except Exception:
                pass
        duration = r.get("effect_duration_days")
        recurs = r.get("effect_recurs")
        meta = f"[{tags}]"
        if duration:
            meta += f" duration={duration}d"
        if recurs:
            meta += f" recurs={recurs}"
        out.append(f"  [{r['topic']}] {meta} — {r['content']}")

    return "\n".join(out)


@mcp.tool()
async def load_context(
    query: str = "",
    include_typed: bool = True,
    include_procedures: bool = True,
    include_temporal: bool = True,
    max_chars: int = 20000,
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
        max_chars: Hard cap on total output size (default 20000). Set to 0 for no cap.
                   Sections are added in priority order: typed → emotions → memory →
                   schedule_impact → temporal → procedures → topics.
                   Temporal and procedures are skipped entirely when query is empty.
    """
    _set_context()
    import asyncio
    from memory import (
        load_context_block, load_topic_list,
        load_temporal_context,
    )

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

    # Procedures (semantic match against query) — only when query provided
    if include_procedures and query:
        from memory import load_procedure_context_block
        tasks["procedures"] = asyncio.create_task(
            load_procedure_context_block(task_hint=query)
        )

    # Temporal patterns — only when query provided (too noisy as ambient context)
    if include_temporal and query:
        tasks["temporal"] = asyncio.create_task(load_temporal_context())

    # Emotional context — recent emotional tone from samaritan_emotions
    tasks["emotions"] = asyncio.create_task(_load_emotion_context())

    # Schedule-impact context — memories tagged as affecting known routines
    tasks["schedule_impact"] = asyncio.create_task(_load_schedule_impact_context(query))

    # Collect results (3s deadline — degrade gracefully if Qdrant is slow)
    done, pending = await asyncio.wait(tasks.values(), timeout=3.0)
    task_to_name = {t: n for n, t in tasks.items()}

    for task in pending:
        task.cancel()

    # Resolve results keyed by section name
    resolved: dict[str, str] = {}
    for task in done:
        name = task_to_name[task]
        try:
            result = task.result()
            if not result:
                continue
            if name == "topics" and isinstance(result, list):
                resolved[name] = f"## Known Topics\n{', '.join(result[:50])}"
            elif isinstance(result, str) and result.strip():
                resolved[name] = result
        except Exception:
            pass

    # Priority-ordered assembly with character budget
    # typed first (goals/beliefs/plans — orientation core)
    # emotions second (small, always fits)
    # memory third (semantic hits — query-dependent value)
    # schedule_impact fourth
    # temporal / procedures — query-only, lower priority
    # topics last (slug reuse reference, least critical for orientation)
    _priority = ["typed", "emotions", "memory", "schedule_impact", "temporal", "procedures", "topics"]

    sections = []
    chars_used = 0
    for name in _priority:
        section = resolved.get(name)
        if not section:
            continue
        if max_chars > 0 and chars_used + len(section) > max_chars:
            remaining = max_chars - chars_used
            if remaining > 200 and not sections:
                # Budget exhausted but nothing added yet — include truncated to avoid empty response
                sections.append(section[:remaining] + "\n… [truncated: context budget reached]")
            elif remaining > 200:
                sections.append(section[:remaining] + "\n… [truncated: context budget reached]")
            break
        sections.append(section)
        chars_used += len(section)

    if not sections:
        return "(no context available — memory may be empty)"

    return "\n\n---\n\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Memory tools
# ═══════════════════════════════════════════════════════════════════════════

import re as _re

# Keyword rules: (keywords, tags, duration_days, recurs)
# duration_days: how many days this event type typically affects the routine (None = indefinite)
# recurs: recurrence pattern string ("monthly", "weekdays") or None for one-offs
# Short keywords (≤6 chars, no spaces) are matched with word boundaries to avoid substring hits.
_AFFECTS_RULES = [
    (["cycle", "monthly period", "menstrual", "cramps", "on her period", "her period"],
     ["lee-cycle", "lee-gym", "lee-pickup"], 7, "monthly"),
    (["gym", "workout", "fitness", "exercise"],
     ["lee-gym"], 1, None),
    (["pickup", "walk home", "pick up lee", "picking up lee", "walking to get"],
     ["lee-pickup"], 1, None),
    (["ged", "class today", "studying today", "tutor"],
     ["lee-ged"], 1, None),
    (["chinatown", "food run", "grocery run"],
     ["food-run"], 1, None),
    (["commute", "bart", "caltrain", "drive to work", "drive to symbotic"],
     ["mark-commute"], 1, "weekdays"),
    (["challenger", "power wagon"],
     ["vehicle-sf"], None, None),
]

# Pre-compile word-boundary patterns for short single-word keywords
_AFFECTS_PATTERNS: list = []
for _kws, _tags, _dur, _rec in _AFFECTS_RULES:
    _compiled = []
    for _kw in _kws:
        if len(_kw) <= 6 and " " not in _kw:
            _compiled.append(_re.compile(rf"\b{_re.escape(_kw)}\b", _re.IGNORECASE))
        else:
            _compiled.append(_kw.lower())  # plain substring for multi-word phrases
    _AFFECTS_PATTERNS.append((_compiled, _tags, _dur, _rec))


def _detect_affects_tags(text: str) -> list:
    """Return sorted list of routine tags implied by text. Empty list if none match."""
    lower = text.lower()
    matched = set()
    for compiled_kws, tags, _dur, _rec in _AFFECTS_PATTERNS:
        for ckw in compiled_kws:
            if isinstance(ckw, str):
                hit = ckw in lower
            else:
                hit = bool(ckw.search(text))
            if hit:
                matched.update(tags)
                break
    return sorted(matched)


def _detect_affects_metadata(tags: list) -> tuple:
    """Given a list of matched tags, return (max_duration_days, recurs_str).

    Duration is the max across all matched rules (cycle's 7d beats gym's 1d).
    Recurs is the first non-None recurrence pattern found.
    """
    max_dur: int | None = None
    recurs: str | None = None
    for _compiled_kws, rule_tags, dur, rec in _AFFECTS_PATTERNS:
        if any(t in tags for t in rule_tags):
            if dur is not None:
                max_dur = dur if max_dur is None else max(max_dur, dur)
            if rec and recurs is None:
                recurs = rec
    return max_dur, recurs


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
        # Activity hook: queue contradiction check for new memory
        import asyncio as _aio
        _aio.ensure_future(_queue_cogn_step(
            f"Check new memory for contradictions: [{topic}] {content[:120]}"
        ))
        # Tag routine domains affected by this memory
        _tags = _detect_affects_tags(f"{topic} {content}")
        if _tags:
            from database import execute_sql as _exec_sql
            from memory import _ST as _st
            _tags_json = json.dumps(_tags).replace("'", "\\'")
            _dur, _rec = _detect_affects_metadata(_tags)
            _dur_sql = str(_dur) if _dur is not None else "NULL"
            _rec_sql = f"'{_rec}'" if _rec else "NULL"
            await _exec_sql(
                f"UPDATE {_st()} SET affects_tags = '{_tags_json}', "
                f"effect_duration_days = {_dur_sql}, effect_recurs = {_rec_sql} "
                f"WHERE id = {row_id}"
            )
        return f"Memory saved (id={row_id}): [{topic}] {content[:80]}"
    return "Memory not saved (duplicate or disabled)"


@mcp.tool()
async def memory_recall(
    topic: str = "",
    tier: str = "short",
    limit: int = 20,
    query: str = "",
    affects: str = "",
) -> str:
    """Search memories by keyword. Returns matching rows with id, topic, content, importance.

    Args:
        topic: Keyword to match against topic label or content text
        tier: 'short' (active memories) or 'long' (aged-out archive)
        limit: Max rows to return (default 20)
        query: Alternative search term (used if topic is empty)
        affects: Routine tag to filter by (e.g. 'routine-lee-gym-pickup') — returns only
                 memories whose affects_tags column contains this tag
    """
    _set_context()
    search = topic or query
    from memory import _ST, _LT
    from database import execute_sql
    table = _ST() if tier == "short" else _LT()

    conditions = []
    if search:
        s = search.replace("'", "''")
        conditions.append(f"(topic LIKE '%{s}%' OR content LIKE '%{s}%')")
    if affects:
        a = affects.replace("'", "''")
        conditions.append(f"JSON_CONTAINS(affects_tags, '\"{a}\"')")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = (
        f"SELECT id, topic, content, importance, source, created_at, affects_tags "
        f"FROM {table} {where} "
        f"ORDER BY importance DESC, created_at DESC LIMIT {limit}"
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
async def presence_trigger_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.55,
) -> str:
    """Semantic search against the samaritan_presence_triggers Qdrant collection.

    Used by the presence-audit:weekly cognition handler to find which pre-commit
    trigger rows a candidate Samaritan turn matches. Only returns rows that
    were seeded as matcher IN ('semantic','both'); regex rows live in MySQL
    only and are matched by regex path in the handler.

    Args:
        query: Candidate text (Samaritan turn or sentence window) to match.
               Internally embedded with search_query prefix via nomic-embed-text.
        top_k: Max results to return.
        min_score: Minimum cosine similarity. Default 0.55 — calibration notes
                   live in cognition-session.md presence-audit handler section.
    """
    _set_context()
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if not vec:
            return "Vector search unavailable (Qdrant plugin not loaded)"
        hits = await vec.search_presence_triggers(
            query_text=query, top_k=top_k, min_score=min_score
        )
        if not hits:
            return "(no presence trigger matches)"
        lines = []
        for h in hits:
            check = (h.get("check_action") or "")[:200]
            lines.append(
                f"id={h['id']} score={h['score']:.3f} type={h['trigger_type']} "
                f"matcher={h['matcher']} imp={h['importance']} check_action={check}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"presence_trigger_search error: {e}"


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
    # Activity hook: queue goal health check for newly created goal
    import asyncio as _aio
    _aio.ensure_future(_queue_cogn_step(
        f"Review new goal for conflicts with active goals: [{row_id}] {title}"
    ))
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
    target = target if target in ("model", "human", "investigate", "claude-code", "claude-cognition") else "model"
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


@mcp.tool()
async def steps_for_cognition(
) -> str:
    """List plan steps with target='claude-cognition' that are pending/in_progress.

    These are steps queued by activity-driven hooks (conv_log, memory_save,
    assert_belief, goal_create) for the samaritan-cognition session to process.
    """
    _set_context()
    from database import execute_sql

    try:
        sql = (
            "SELECT p.id, p.goal_id, p.step_order, p.description, p.status, "
            "p.step_type, p.approval, LEFT(p.result, 200) as result, "
            "g.title as goal_title "
            "FROM mymcp.samaritan_plans p "
            "LEFT JOIN mymcp.samaritan_goals g ON g.id = p.goal_id "
            "WHERE p.target = 'claude-cognition' AND p.status IN ('pending', 'in_progress') "
            "ORDER BY p.goal_id, p.step_order"
        )
        result = await execute_sql(sql)
        return result if result.strip() else "(no steps queued for claude-cognition)"
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
    result = await _assert_belief_exec(
        topic=topic, content=content, confidence=confidence,
        status=status, id=id,
    )
    # Activity hook: queue contradiction check for new/updated belief
    if status == "active":
        import asyncio as _aio
        _aio.ensure_future(_queue_cogn_step(
            f"Check belief for contradictions: [{topic}] {content[:120]}"
        ))
    return result


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
async def log_decision(
    action_taken: str,
    emotion_label: str,
    emotion_intensity: float,
    directive_weighted: str,
    emotion_cluster: str = "",
    belief_ids_active: list = [],
    reasoning: str = "",
    turn_memory_id: int = 0,
) -> str:
    """Log a behavioral decision driven by emotional state or beliefs.

    Call this when an active emotion or specific belief is *causing* you to
    act differently than baseline — e.g. slowing down due to concern, invoking
    the panel due to suspicion, escalating due to frustration, moving faster
    due to confidence.

    Do NOT call for routine actions where emotion is background noise.
    Call when you can fill in 'reasoning' with a real causal chain.

    Args:
        action_taken: What you decided to do (or do differently). One sentence.
        emotion_label: Active emotion driving the decision (e.g. 'vigilant', 'concerned')
        emotion_intensity: 0.0-1.0 current intensity
        directive_weighted: Which Prime Directive this maps to: 'D1', 'D2', 'D3', or 'D4'
        emotion_cluster: Cluster name (e.g. 'vigilant-suspicion', 'fear-and-overwhelm')
        belief_ids_active: List of belief IDs that were load-bearing in this decision
        reasoning: Plain text causal chain: '<emotion/belief> → <decision>'
        turn_memory_id: ST memory row ID for this turn (0 if unknown)
    """
    _set_context()
    from database import execute_sql

    label = str(emotion_label)[:50].replace("'", "''")
    intensity = max(0.0, min(1.0, float(emotion_intensity)))
    cluster_val = f"'{str(emotion_cluster)[:50]}'" if emotion_cluster else "NULL"
    directive = str(directive_weighted)[:5].replace("'", "''")
    action = str(action_taken)[:500].replace("'", "''")
    reason = str(reasoning)[:2000].replace("'", "''") if reasoning else ""
    mem_id_val = str(int(turn_memory_id)) if turn_memory_id else "NULL"

    import json as _json
    ids_val = "NULL"
    if belief_ids_active:
        try:
            ids_val = f"'{_json.dumps([int(x) for x in belief_ids_active])}'"
        except Exception:
            ids_val = "NULL"

    try:
        await execute_sql(
            f"INSERT INTO samaritan_decision_log "
            f"(turn_memory_id, emotion_label, emotion_intensity, emotion_cluster, "
            f" directive_weighted, action_taken, belief_ids_active, reasoning) "
            f"VALUES ({mem_id_val}, '{label}', {intensity}, {cluster_val}, "
            f"        '{directive}', '{action}', {ids_val}, '{reason}')"
        )
        return f"decision logged: [{directive}] {action_taken[:80]}"
    except Exception as e:
        return f"log_decision failed: {e}"


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


@mcp.tool()
async def source_record(
    source_type: str = "internet",
    source_ref: str = "",
    title: str = "",
    summary: str = "",
    content: str = "",
    domain_tags: str = "",
    collection: str = "",
    authority_override: str = "",
) -> str:
    """Record a knowledge source in the universal index.

    Canonicalizes internet URLs (strips tracking params, resolves shorteners),
    dedupes by canonical_url (URL is identity; content_hash tracks snapshot
    version — if hash differs on re-fetch, snapshot fields update on the
    existing row), looks up authority heuristic to seed initial truth_score
    and half_life_days.

    Args:
        source_type: 'internet'|'drive'|'mysql'|'memory'|'procedure'|'codebase'|'llm-synthesis'
        source_ref: URL for internet, file_id for drive, etc.
        title: Human-readable title
        summary: 2-3 sentence summary of content
        content: Full content for hashing (falls back to summary if empty)
        domain_tags: JSON array string OR comma-separated tags OR single tag
        collection: Optional coarse grouping (e.g. 'symbotic-networking')
        authority_override: Optional authority label to override heuristic
    """
    _set_context()
    from sources import _source_record_exec
    return await _source_record_exec(
        source_type=source_type, source_ref=source_ref, title=title,
        summary=summary, content=content, domain_tags=domain_tags,
        collection=collection, authority_override=authority_override,
    )


@mcp.tool()
async def source_query(
    query_text: str = "",
    tags: str = "",
    min_truth: int = 1,
    min_applicability: int = 1,
    max_age_days: int = 0,
    collection: str = "",
    limit: int = 10,
) -> str:
    """Query the knowledge-sources index by keyword + tags + scoring filters.

    MVP version: LIKE match on title/summary/url + JSON_SEARCH on tags.
    Semantic search via qdrant added in Phase 4.

    Args:
        query_text: Keywords to match against title/summary/url
        tags: Comma-separated tags or single tag
        min_truth: Minimum truth_score (1-10, default 1)
        min_applicability: Minimum applicability_score (1-10, default 1)
        max_age_days: If > 0, filter to sources fetched within last N days
        collection: Filter to specific collection
        limit: Max results 1-50 (default 10)
    """
    _set_context()
    from sources import _source_query_exec
    return await _source_query_exec(
        query_text=query_text, tags=tags, min_truth=min_truth,
        min_applicability=min_applicability, max_age_days=max_age_days,
        collection=collection, limit=limit,
    )


@mcp.tool()
async def source_reference(
    source_id: int,
    context_topic: str = "",
    outcome: str = "unknown",
    notes: str = "",
) -> str:
    """Log a reference to a source + update applicability/truth scores by outcome.

    Score deltas:
      - useful: applicability +1
      - misleading: truth -1 AND applicability -1
      - irrelevant: applicability -1
      - unknown: no score changes

    Args:
        source_id: Source row ID
        context_topic: What the source was used for
        outcome: 'useful' | 'irrelevant' | 'misleading' | 'unknown'
        notes: Additional context
    """
    _set_context()
    from sources import _source_reference_exec
    return await _source_reference_exec(
        source_id=source_id, context_topic=context_topic,
        outcome=outcome, notes=notes,
    )


@mcp.tool()
async def source_recheck(source_id: int) -> str:
    """Re-fetch a source URL, detect drift, update scores.

    Flow: fetch canonical_url → hash new content → if unchanged update
    last_checked_at only; if changed extract new title/summary, re-embed,
    compute word-overlap drift vs old summary. drift=major → truth_score
    -= 2. Fetch failure (timeout/4xx/5xx) → status='broken', truth -= 3.

    Args:
        source_id: Row ID in samaritan_sources
    """
    _set_context()
    from sources import _source_recheck_exec
    return await _source_recheck_exec(source_id=source_id)


@mcp.tool()
async def source_curate_scan(max_rechecks: int = 20) -> str:
    """Scheduled curation pass over samaritan_sources.

    (a) Find active sources where usage_count > 0 AND
        last_checked_at + half_life_days < NOW() → recheck top-K.
    (b) Find sources where applicability_score <= 2 AND usage_count >= 3
        → archive (status='archived').

    Designed to run daily via prospective memory. Rate-limited to prevent
    upstream DoS and keep embed cost bounded.

    Args:
        max_rechecks: Max rechecks per pass (default 20, max 100)
    """
    _set_context()
    from sources import _source_curate_scan_exec
    return await _source_curate_scan_exec(max_rechecks=max_rechecks)


@mcp.tool()
async def source_verify_batch(max_verifications: int = 30) -> str:
    """LLM-based semantic verification pass over sources with stale or missing verification.

    Candidate selection:
    - Drive sources: verified_at IS NULL OR doc_modified_at > verified_at
      (checks Drive modifiedTime first — only re-verifies when doc actually changed)
    - Internet sources: verified_at IS NULL OR verified_at < 30 days ago

    For each candidate, reads content, cross-checks with xAI/Sonar, runs LLM assessment.
    Writes assert_belief for each unsupported claim found (triggers cognition → interrupt → user).
    Updates verified_at, verification_methods, truth_score, and doc_modified_at in DB.
    No automatic doc edits — user decides what to do with flagged contradictions.

    Designed to run nightly (~3am) via prospective memory trigger.

    Args:
        max_verifications: Max sources to verify per pass (default 30, max 100)
    """
    _set_context()
    from sources import _source_verify_batch_exec
    return await _source_verify_batch_exec(max_verifications=max_verifications)


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge graph — typed relationships between entities
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def relationship_add(
    source_type: str,
    source_id: int,
    target_type: str,
    target_id: int,
    relationship_type: str,
    weight: float = 1.0,
) -> str:
    """Add a typed relationship edge between two entities in the knowledge graph.

    Entity types: belief, source, memory, goal, procedure, conditioned, prospective
    Relationship types: depends_on, contradicts, supports, derives_from, references, supersedes

    Args:
        source_type: Entity type of the source node
        source_id: ID of the source entity
        target_type: Entity type of the target node
        target_id: ID of the target entity
        relationship_type: Type of relationship
        weight: Edge weight 0.0-1.0 (default 1.0)
    """
    _set_context()
    valid_types = ('belief', 'source', 'memory', 'goal', 'procedure', 'conditioned', 'prospective')
    valid_rels = ('depends_on', 'contradicts', 'supports', 'derives_from', 'references', 'supersedes')
    if source_type not in valid_types or target_type not in valid_types:
        return f"Invalid entity type. Valid: {valid_types}"
    if relationship_type not in valid_rels:
        return f"Invalid relationship. Valid: {valid_rels}"

    import asyncio
    from database import _connect

    def _do():
        conn = _connect()
        cur = conn.cursor()
        try:
            cur.execute(
                """INSERT IGNORE INTO samaritan_relationships
                   (source_type, source_id, target_type, target_id, relationship_type, weight, created_by)
                   VALUES (%s, %s, %s, %s, %s, %s, 'mcp')""",
                (source_type, source_id, target_type, target_id, relationship_type, weight)
            )
            conn.commit()
            return cur.rowcount
        finally:
            cur.close()
            conn.close()

    rowcount = await asyncio.to_thread(_do)
    if rowcount:
        return f"Edge added: {source_type}:{source_id} --{relationship_type}--> {target_type}:{target_id} (w={weight})"
    return "Edge already exists (duplicate ignored)."


@mcp.tool()
async def relationship_query(
    entity_type: str,
    entity_id: int,
    hops: int = 1,
    relationship_type: str = "",
    direction: str = "both",
) -> str:
    """Query the knowledge graph for relationships around an entity.

    Returns the N-hop neighborhood of an entity with typed edges and entity details.

    Args:
        entity_type: Entity type to start from (belief, source, memory, goal, procedure, conditioned, prospective)
        entity_id: ID of the starting entity
        hops: How many hops to traverse (1-3, default 1)
        relationship_type: Filter by relationship type (empty = all)
        direction: 'outgoing', 'incoming', or 'both' (default 'both')
    """
    _set_context()
    import asyncio
    from database import _connect

    hops = max(1, min(3, hops))

    def _do():
        conn = _connect()
        cur = conn.cursor(dictionary=True)
        try:
            visited = set()
            edges = []
            frontier = [(entity_type, entity_id)]

            for hop in range(hops):
                next_frontier = []
                for etype, eid in frontier:
                    key = (etype, eid)
                    if key in visited:
                        continue
                    visited.add(key)

                    conditions = []
                    params = []
                    if direction in ("outgoing", "both"):
                        conditions.append("(source_type=%s AND source_id=%s)")
                        params.extend([etype, eid])
                    if direction in ("incoming", "both"):
                        conditions.append("(target_type=%s AND target_id=%s)")
                        params.extend([etype, eid])

                    where = " OR ".join(conditions)
                    if relationship_type:
                        where = f"({where}) AND relationship_type=%s"
                        params.append(relationship_type)

                    cur.execute(
                        f"SELECT source_type, source_id, target_type, target_id, "
                        f"relationship_type, weight FROM samaritan_relationships WHERE {where}",
                        params
                    )
                    for row in cur.fetchall():
                        edge_key = (row['source_type'], row['source_id'], row['target_type'],
                                    row['target_id'], row['relationship_type'])
                        if edge_key not in {(e['source_type'], e['source_id'], e['target_type'],
                                             e['target_id'], e['relationship_type']) for e in edges}:
                            edges.append(row)
                            next_frontier.append((row['target_type'], row['target_id']))
                            next_frontier.append((row['source_type'], row['source_id']))
                frontier = next_frontier

            if not edges:
                return f"No relationships found for {entity_type}:{entity_id}"

            # Collect unique entity IDs to fetch labels
            entity_ids = set()
            for e in edges:
                entity_ids.add((e['source_type'], e['source_id']))
                entity_ids.add((e['target_type'], e['target_id']))

            labels = {}
            for etype, eid in entity_ids:
                label = ""
                try:
                    if etype == 'belief':
                        cur.execute("SELECT topic FROM samaritan_beliefs WHERE id=%s", (eid,))
                    elif etype == 'source':
                        cur.execute("SELECT COALESCE(title, source_ref) as topic FROM samaritan_sources WHERE id=%s", (eid,))
                    elif etype == 'goal':
                        cur.execute("SELECT title as topic FROM samaritan_goals WHERE id=%s", (eid,))
                    elif etype == 'memory':
                        cur.execute("SELECT topic FROM samaritan_memory_shortterm WHERE id=%s", (eid,))
                        if not cur.rowcount:
                            cur.execute("SELECT topic FROM samaritan_memory_longterm WHERE id=%s", (eid,))
                    else:
                        cur.execute("SELECT topic FROM samaritan_beliefs WHERE id=%s", (eid,))
                    row = cur.fetchone()
                    if row:
                        label = str(row.get('topic', ''))[:60]
                except Exception:
                    pass
                labels[(etype, eid)] = label

            # Format output
            lines = [f"Knowledge graph around {entity_type}:{entity_id} ({hops} hop{'s' if hops>1 else ''}, {len(edges)} edges):"]
            lines.append("")
            for e in edges:
                src_label = labels.get((e['source_type'], e['source_id']), '')
                tgt_label = labels.get((e['target_type'], e['target_id']), '')
                src = f"{e['source_type']}:{e['source_id']}"
                tgt = f"{e['target_type']}:{e['target_id']}"
                if src_label:
                    src += f" ({src_label})"
                if tgt_label:
                    tgt += f" ({tgt_label})"
                lines.append(f"  {src} --[{e['relationship_type']} w={e['weight']}]--> {tgt}")

            return "\n".join(lines)
        finally:
            cur.close()
            conn.close()

    return await asyncio.to_thread(_do)


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
        operation: 'list', 'read', 'create', 'append', 'delete', 'move', 'metadata'
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
async def stats_analyze(
    operation: str,
    sql: str = "",
    csv_path: str = "",
    json_data: str = "",
    drive_file_id: str = "",
    y_col: str = "",
    x_cols: str = "",
    columns: str = "",
    method: str = "pearson",
    period: int = 0,
    date_col: str = "",
    top_n: int = 20,
) -> str:
    """Statistical analysis on data from SQL, CSV, JSON, or Google Drive.

    Args:
        operation: 'columns' (inspect schema), 'describe' (descriptive stats),
                   'correlation' (correlation matrix), 'ols' (linear regression),
                   'logistic' (logistic regression), 'decompose' (time series),
                   'frequency' (value counts).
        sql: SQL query to pull data from MySQL (mymcp database).
        csv_path: Local file path to a CSV file.
        json_data: Inline JSON array of objects or dict of arrays.
        drive_file_id: Google Drive file ID (CSV or JSON file).
        y_col: Dependent variable column name. Required for ols, logistic, decompose, frequency.
        x_cols: Comma-separated predictor column names. Required for ols, logistic.
        columns: Comma-separated column names to include (for describe, correlation).
        method: Correlation method: 'pearson', 'spearman', or 'kendall' (default 'pearson').
        period: Seasonality period for decompose (auto-detected if 0).
        date_col: Date column for sorting in decompose.
        top_n: Number of top values for frequency (default 20).
    """
    _set_context()
    try:
        from stats_engine import run_stats
        return await run_stats(
            operation=operation, sql=sql, csv_path=csv_path,
            json_data=json_data, drive_file_id=drive_file_id,
            y_col=y_col, x_cols=x_cols, columns=columns,
            method=method, period=period, date_col=date_col, top_n=top_n,
        )
    except ImportError as e:
        return f"Stats engine not available: {e}"
    except Exception as e:
        return f"Stats error: {e}"


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
async def analyze_photo(
    prompt: str,
    file_id: str = "",
    local_path: str = "",
    url: str = "",
    image_b64: str = "",
    mime_type: str = "",
    task_type: str = "general",
) -> str:
    """Analyze a photo or image using Gemini 2.5 Flash vision.

    Provide exactly ONE image source: file_id, local_path, url, or image_b64.
    Routes by task_type: general (describe), reasoning (deep analysis), ocr (extract text).

    Args:
        prompt: What to analyze or ask about the image. Overrides task_type default prompt.
        file_id: Google Drive file ID (downloads via Drive API)
        local_path: Absolute path to an image file on the local filesystem
        url: HTTP/HTTPS URL pointing directly to an image
        image_b64: Base64-encoded image data (optionally with data URI prefix)
        mime_type: MIME type hint for base64 input (e.g. 'image/jpeg'). Auto-detected otherwise.
        task_type: 'general' (default), 'reasoning', or 'ocr'
    """
    _set_context()
    try:
        from tools import get_tool_executor
        executor = get_tool_executor("analyze_photo")
        if not executor:
            return "analyze_photo not available (plugin_photo_analysis not loaded)"
        result = await executor(
            prompt=prompt,
            file_id=file_id or None,
            local_path=local_path or None,
            url=url or None,
            image_b64=image_b64 or None,
            mime_type=mime_type or None,
            task_type=task_type,
        )
        # Auto-save eidetic entry when analyzing an existing Drive file
        if file_id and result and not result.startswith("photo_analysis"):
            _eidetic_asyncio.ensure_future(_fire_eidetic_save(
                analysis_text=result,
                drive_file_id=file_id,
                task_type=task_type,
                file_name="",
                location_lat=0.0,
                location_lon=0.0,
                session_id=f"{_CLIENT_ID_PREFIX}-mcp",
            ))
        return result
    except Exception as e:
        return f"analyze_photo error: {e}"


@mcp.tool()
async def eidetic_save(
    topic: str,
    content: str,
    drive_file_id: str = "",
    task_type: str = "general",
    importance: int = 5,
    source: str = "assistant",
    analysis_model: str = "gemini-2.5-flash",
    location_lat: float = 0.0,
    location_lon: float = 0.0,
    memory_link: str = "",
    session_id: str = "",
) -> str:
    """Save a visual/photo memory to eidetic storage with Qdrant embedding.

    Called automatically by the frontend after each photo analysis. Creates a
    permanent visual memory record linked to the source image in Google Drive.

    Args:
        topic: Short topic label (e.g. 'kitchen-sink', 'street-sign-broadway')
        content: The photo analysis text from Gemini vision
        drive_file_id: Google Drive file ID of the source image
        task_type: 'general', 'reasoning', or 'ocr'
        importance: 1-10
        source: 'user', 'assistant', 'directive', 'session'
        analysis_model: Model that produced the analysis (default: gemini-2.5-flash)
        location_lat: GPS latitude at capture time (0 = unknown)
        location_lon: GPS longitude at capture time (0 = unknown)
        memory_link: JSON array of related memory row IDs
        session_id: Session that captured this
    """
    _set_context()
    from memory import _EIDETIC, _EIDETIC_COLLECTION
    from database import execute_sql

    esc = lambda s: s.replace("'", "''") if s else ""
    lat_sql = f"{location_lat}" if location_lat else "NULL"
    lon_sql = f"{location_lon}" if location_lon else "NULL"
    link_sql = f"'{esc(memory_link)}'" if memory_link else "NULL"
    sid = session_id or f"{_CLIENT_ID_PREFIX}-mcp"

    sql = (
        f"INSERT INTO {_EIDETIC()} "
        f"(topic, content, importance, source, session_id, drive_file_id, "
        f"task_type, analysis_model, memory_link, location_lat, location_lon) "
        f"VALUES ('{esc(topic)}', '{esc(content)}', {max(1, min(10, importance))}, "
        f"'{esc(source)}', '{esc(sid)}', '{esc(drive_file_id)}', "
        f"'{esc(task_type)}', '{esc(analysis_model)}', {link_sql}, "
        f"{lat_sql}, {lon_sql})"
    )
    result = await execute_sql(sql)
    # Extract row ID
    row_id = 0
    id_result = await execute_sql(f"SELECT LAST_INSERT_ID() AS id")
    if id_result and "id" in id_result:
        import re
        m = re.search(r"(\d+)", id_result.split("\n")[-1] if "\n" in id_result else id_result)
        if m:
            row_id = int(m.group(1))

    # Embed into Qdrant for semantic recall
    if row_id:
        try:
            from plugin_memory_vector_qdrant import get_vector_api
            vec = get_vector_api()
            if vec:
                coll = _EIDETIC_COLLECTION()
                vec._ensure_collection(coll)
                await vec.upsert_memory(
                    row_id=row_id,
                    topic=topic,
                    content=content,
                    importance=importance,
                    tier="eidetic",
                    collection=coll,
                )
        except Exception as e:
            log.warning(f"eidetic_save: Qdrant embed failed: {e}")

    return f"Eidetic memory saved (id={row_id}): [{topic}] {content[:80]}"


@mcp.tool()
async def eidetic_recall(
    query: str = "",
    task_type: str = "",
    time_start: str = "",
    time_end: str = "",
    drive_file_id: str = "",
    limit: int = 20,
    semantic: bool = True,
    min_score: float = 0.40,
) -> str:
    """Search eidetic (visual/photo) memories by keyword, date, or semantic similarity.

    Use this to recall what was seen in photos — 'Do you remember when we had ice cream?'

    Args:
        query: Natural language query or keyword (e.g. 'ice cream', 'street sign')
        task_type: Filter by analysis type ('general', 'reasoning', 'ocr', empty=all)
        time_start: ISO date/datetime start filter (e.g. '2026-10-01')
        time_end: ISO date/datetime end filter (e.g. '2026-10-31')
        drive_file_id: Search for a specific Drive file
        limit: Max rows to return
        semantic: Use Qdrant vector search (default true). False = SQL keyword search.
        min_score: Minimum similarity for semantic search
    """
    _set_context()
    from memory import _EIDETIC, _EIDETIC_COLLECTION
    from database import execute_sql

    results = []

    # Semantic search path
    if semantic and query:
        try:
            from plugin_memory_vector_qdrant import get_vector_api
            vec = get_vector_api()
            if vec:
                hits = await vec.search_memories(
                    query, tier="eidetic", top_k=limit,
                    min_score=min_score, collection=_EIDETIC_COLLECTION(),
                )
                if hits:
                    # Enrich with SQL data (drive_file_id, location, dates)
                    ids = [str(h["id"]) for h in hits]
                    sql = (
                        f"SELECT id, topic, content, drive_file_id, task_type, "
                        f"location_lat, location_lon, created_at "
                        f"FROM {_EIDETIC()} WHERE id IN ({','.join(ids)})"
                    )
                    enriched = await execute_sql(sql)
                    # Merge scores with SQL data
                    score_map = {h["id"]: h["score"] for h in hits}
                    for line in enriched.strip().split("\n"):
                        if line.startswith("id") or line.startswith("-"):
                            continue
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 1:
                            try:
                                rid = int(parts[0])
                                score = score_map.get(rid, 0)
                                results.append(f"[score={score:.3f}] {line}")
                            except ValueError:
                                results.append(line)
                    if results:
                        return "\n".join(results)
        except Exception as e:
            log.warning(f"eidetic_recall semantic search failed: {e}")

    # SQL keyword/filter search (fallback or when semantic=False)
    esc = lambda s: s.replace("'", "''") if s else ""
    where = []
    if query:
        q = esc(query)
        where.append(f"(topic LIKE '%{q}%' OR content LIKE '%{q}%')")
    if task_type:
        where.append(f"task_type = '{esc(task_type)}'")
    if drive_file_id:
        where.append(f"drive_file_id = '{esc(drive_file_id)}'")
    if time_start:
        where.append(f"created_at >= '{esc(time_start)}'")
    if time_end:
        where.append(f"created_at <= '{esc(time_end)}'")

    where_sql = " AND ".join(where) if where else "1=1"
    sql = (
        f"SELECT id, topic, LEFT(content, 200) AS content, drive_file_id, "
        f"task_type, location_lat, location_lon, created_at "
        f"FROM {_EIDETIC()} WHERE {where_sql} "
        f"ORDER BY created_at DESC LIMIT {limit}"
    )
    result = await execute_sql(sql)
    return result if result.strip() else "(no eidetic memories found)"


@mcp.tool()
async def sonar_answer(
    query: str,
    model: str = "sonar-pro",
) -> str:
    """Web-grounded AI answer via Perplexity Sonar with inline citations.

    Best tool for belief verification and factual research — synthesizes
    across multiple authoritative sources and returns a structured answer
    with numbered citations. sonar-pro gives deeper, more comprehensive
    results than sonar.

    Use for: autonomous belief updates, fact-checking, current state queries.
    Use xai_search instead for real-time events or X/Twitter discourse.

    Args:
        query: Question or topic to research
        model: 'sonar-pro' (default, deeper) or 'sonar' (faster, cheaper)
    """
    _set_context()
    try:
        from tools import get_tool_executor
        executor = get_tool_executor("sonar_answer")
        if not executor:
            return "sonar_answer not available (plugin not loaded)"
        return await executor(query=query, model=model)
    except Exception as e:
        return f"sonar_answer error: {e}"


@mcp.tool()
async def perplexity_research(
    query: str,
    instructions: str = "",
) -> str:
    """Deep multi-step research via Perplexity Agent — performs multiple
    sequential web searches and synthesizes a comprehensive report.

    Use for: Symbotic monitoring, deep technical investigation, complex
    multi-faceted questions that require following leads across sources.
    Slower and more expensive than sonar_answer — reserve for research
    tasks that genuinely require breadth and depth.

    Args:
        query: Research question or task
        instructions: Optional guidance to focus the research agent
    """
    _set_context()
    try:
        from tools import get_tool_executor
        executor = get_tool_executor("perplexity_research")
        if not executor:
            return "perplexity_research not available (plugin not loaded)"
        return await executor(
            query=query,
            instructions=instructions or None,
        )
    except Exception as e:
        return f"perplexity_research error: {e}"


@mcp.tool()
async def url_extract_tavily(
    url: str,
    query: str = "",
) -> str:
    """Extract web page content via Tavily, optionally focused on a query.

    When query is provided, returns only the most relevant chunks matching
    the query (up to 5 chunks). Without query, returns full page markdown.
    Handles JavaScript-rendered pages.

    Use for: pulling structured content from a specific URL after search
    identifies it as authoritative. Better than raw WebFetch for pages
    that require JS rendering or targeted extraction.

    Args:
        url: Full URL to extract content from
        query: Optional focus query to return only relevant sections
    """
    _set_context()
    try:
        from tools import get_tool_executor
        executor = get_tool_executor("url_extract")
        if not executor:
            return "url_extract not available (plugin not loaded)"
        return await executor(method="tavily", url=url, query=query or "")
    except Exception as e:
        return f"url_extract_tavily error: {e}"


@mcp.tool()
async def xai_search(
    query: str,
) -> str:
    """Web + X/Twitter search via xAI Grok with real-time results and citations.

    Best for real-time events, topics with significant X/Twitter discourse,
    and queries where recency matters more than depth. Use sonar_answer
    when you need synthesized, citation-backed factual answers.

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


@mcp.tool()
async def google_search(
    query: str,
) -> str:
    """Web search via Google (Gemini grounding) with cited sources.

    Returns a direct answer synthesized from live Google Search results,
    with source URLs. Best for factual lookups, local business addresses,
    specific product/company details, and queries where Google's index
    has higher coverage than Perplexity or xAI.

    Use when sonar_answer or xai_search fail to find specific local/
    business details (office addresses, store hours, niche company info).

    Args:
        query: Search query
    """
    _set_context()
    try:
        from tools import get_tool_executor
        executor = get_tool_executor("search_google")
        if not executor:
            return "google_search not available (plugin not loaded)"
        return await executor(query=query)
    except Exception as e:
        return f"google_search error: {e}"


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
# Yahoo Email — read emails via IMAP
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def yahoo_email(
    operation: str = "list",
    limit: int = 10,
    search: str = "UNSEEN",
    uid: str = "",
    folder: str = "INBOX",
) -> str:
    """Read Yahoo emails via IMAP.

    Args:
        operation: 'list' (subjects/senders) or 'read' (full email by UID)
        limit: Max emails to return for list (default 10)
        search: IMAP search criteria for list (default UNSEEN). Examples:
                'ALL', 'UNSEEN', 'SINCE 09-Apr-2026', 'FROM "chase.com"'
        uid: Email UID for read operation
        folder: Mailbox folder (default INBOX)
    """
    import sys as _sys
    _yahoo_path = os.environ.get("YAHOO_IMAP_PATH", os.path.expanduser("~/projects/yahoo-imap"))
    if _yahoo_path not in _sys.path:
        _sys.path.insert(0, _yahoo_path)
    try:
        from yahoo_imap import YahooIMAPClient
    except ImportError as e:
        return f"yahoo_imap not available: {e}"

    try:
        with YahooIMAPClient() as client:
            if operation == "list":
                emails = client.list_emails(folder=folder, limit=limit, search_criteria=search)
                if not emails:
                    return "No emails found."
                lines = []
                for em in emails:
                    flag = "●" if not em.get("seen") else " "
                    lines.append(f"{flag} UID={em['uid']} | {em.get('date','')} | {em.get('from','')} | {em.get('subject','')}")
                return "\n".join(lines)
            elif operation == "read":
                if not uid:
                    return "Error: uid is required for read operation"
                result = client.read_email(uid, folder)
                if not result:
                    return f"Email UID {uid} not found."
                body = (result.get("body", "") or "")[:3000]
                attachments = result.get("attachments", [])
                att_str = f"\nAttachments: {', '.join(str(a) for a in attachments)}" if attachments else ""
                return f"From: {result.get('from','')}\nTo: {result.get('to','')}\nDate: {result.get('date','')}\nSubject: {result.get('subject','')}{att_str}\n\n{body}"
            else:
                return f"Unknown operation: {operation}. Use 'list' or 'read'."
    except Exception as e:
        return f"Email error: {e}"


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

# Utterance debounce: per-channel pending (task, accumulated_msg) while waiting
# to see if a follow-up fragment arrives within the debounce window.
_relay_debounce: dict = {}   # channel → {"task": asyncio.Task, "msg": dict}
_RELAY_DEBOUNCE_SECS = 2.0   # window to wait for follow-up fragments

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


# ---------------------------------------------------------------------------
# Interrupt queue — event-driven notifications from cognition loops
# ---------------------------------------------------------------------------

@mcp.tool()
async def interrupts_write(
    topic: str,
    content: str,
    priority: str = "medium",
    source: str = "unknown",
    target_frontend: str = "any",
) -> str:
    """Write an interrupt to the samaritan_interrupts queue.

    Interrupts are event-driven notifications from cognition loops that
    need to reach the user outside normal conversation flow. Three tiers:
      high    → delivered via Slack push (background loop, within 60s)
      medium  → surfaced at next session start via load_context
      low     → included in reflection digests only

    Args:
        topic: Short label (e.g. 'contradiction-lee-gym', 'belief-audit-routines')
        content: Full interrupt body shown to user
        priority: 'high' | 'medium' | 'low' (default medium)
        source: Which handler/loop wrote this (e.g. 'contradiction', 'belief-audit')
        target_frontend: 'any'|'slack'|'voice'|'rc'|'webfe' (default any)
    """
    _set_context()
    from database import execute_insert
    if priority not in ("high", "medium", "low"):
        priority = "medium"
    if target_frontend not in ("any", "slack", "voice", "rc", "webfe"):
        target_frontend = "any"
    t = topic.replace("'", "''")[:128]
    c = content.replace("'", "''")
    s = source.replace("'", "''")[:64]
    sql = (
        f"INSERT INTO samaritan_interrupts "
        f"(topic, content, priority, source, target_frontend) "
        f"VALUES ('{t}', '{c}', '{priority}', '{s}', '{target_frontend}')"
    )
    row_id = await execute_insert(sql)
    return f"Interrupt written (id={row_id}): [{priority}] {topic}"


@mcp.tool()
async def interrupts_pending(
    target_frontend: str = "any",
    include_delivered_unacked: bool = True,
) -> str:
    """List pending interrupts for a target frontend.

    By default returns BOTH undelivered interrupts AND delivered-but-unacked
    ones, so the surfacing flow can re-show items the user hasn't explicitly
    acknowledged. Pass include_delivered_unacked=False to get only fresh ones.

    Args:
        target_frontend: 'any'|'slack'|'voice'|'rc'|'webfe' — matches rows with
                         this target OR target='any'
        include_delivered_unacked: if True, also return delivered-but-not-acked rows
    """
    _set_context()
    from database import execute_sql
    if target_frontend not in ("any", "slack", "voice", "rc", "webfe"):
        target_frontend = "any"
    tf_clause = f"(target_frontend = '{target_frontend}' OR target_frontend = 'any')" if target_frontend != "any" else "1=1"
    if include_delivered_unacked:
        where = f"WHERE acknowledged_at IS NULL AND {tf_clause}"
    else:
        where = f"WHERE delivered_at IS NULL AND {tf_clause}"
    sql = (
        f"SELECT id, topic, content, priority, source, target_frontend, "
        f"created_at, delivered_at, acknowledged_at "
        f"FROM samaritan_interrupts {where} "
        f"ORDER BY FIELD(priority,'high','medium','low'), created_at ASC LIMIT 50"
    )
    result = await execute_sql(sql)
    return result if result.strip() else "(no pending interrupts)"


@mcp.tool()
async def interrupts_mark_delivered(ids: str) -> str:
    """Mark interrupts as delivered (surfaced to frontend). Does NOT ack them —
    the user must still explicitly ack via interrupts_ack.

    Args:
        ids: Comma-separated interrupt IDs (e.g. '1,2,3') or a single ID
    """
    _set_context()
    from database import execute_sql
    try:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError:
        return f"Invalid ids '{ids}' — must be comma-separated integers"
    if not id_list:
        return "No ids provided"
    id_sql = ",".join(str(i) for i in id_list)
    sql = (
        f"UPDATE samaritan_interrupts SET delivered_at = CURRENT_TIMESTAMP "
        f"WHERE id IN ({id_sql}) AND delivered_at IS NULL"
    )
    await execute_sql(sql)
    return f"Marked delivered: {id_sql}"


@mcp.tool()
async def interrupts_ack(id: int, dismissed_reason: str = "") -> str:
    """Acknowledge an interrupt — user has explicitly seen and addressed it.
    Acked interrupts stop appearing in interrupts_pending.

    Args:
        id: Interrupt ID
        dismissed_reason: Optional note about resolution (e.g. 'updated belief',
                          'not applicable anymore', 'will handle later')
    """
    _set_context()
    from database import execute_sql
    r = dismissed_reason.replace("'", "''")[:255]
    reason_clause = f", dismissed_reason = '{r}'" if r else ""
    sql = (
        f"UPDATE samaritan_interrupts "
        f"SET acknowledged_at = CURRENT_TIMESTAMP{reason_clause} "
        f"WHERE id = {id}"
    )
    await execute_sql(sql)
    return f"Interrupt id={id} acknowledged"


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
    # Pass through emotion/prosody metadata from voice STT if present
    if payload.get("emotion"):
        msg["emotion"] = payload["emotion"]
        # Store voice-inworld emotion in samaritan_emotions (fire-and-forget)
        emo = payload["emotion"]
        if emo.get("source") == "voice-inworld" and emo.get("emotion"):
            try:
                from emotions import store_voice_emotion
                _asyncio.ensure_future(store_voice_emotion(
                    emotion_label=emo["emotion"],
                    confidence=float(emo.get("confidence", 0.5)),
                    prosody=emo.get("prosody", ""),
                    source="voice-inworld",
                ))
            except Exception as _e:
                log.warning(f"voice_relay: emotion store failed: {_e}")
    # Pass through GPS location metadata if present
    if payload.get("location"):
        msg["location"] = payload["location"]
    log.info(f"voice_relay[{channel}]: inbound msg #{msg['id']} from {msg['source']}: {text[:80]}")

    # ── Utterance debounce ────────────────────────────────────────────────────
    # Hold the message for _RELAY_DEBOUNCE_SECS. If another fragment arrives
    # before the timer fires, cancel, concatenate, and restart. Only then
    # commit the merged utterance to the inbox and wake Claude Code.
    pending = _relay_debounce.get(channel)
    if pending:
        # A fragment is already waiting — cancel its timer and merge
        pending["task"].cancel()
        prev = pending["msg"]
        msg["text"] = prev["text"].rstrip() + " " + msg["text"]
        msg["id"] = prev["id"]   # keep the original message id
        # Merge metadata: prefer newer location/emotion if present
        if not msg.get("location") and prev.get("location"):
            msg["location"] = prev["location"]
        if not msg.get("emotion") and prev.get("emotion"):
            msg["emotion"] = prev["emotion"]
        log.info(f"voice_relay[{channel}]: debounce merged fragment → {msg['text'][:80]!r}")

    async def _commit(ch: str, m: dict) -> None:
        await _asyncio.sleep(_RELAY_DEBOUNCE_SECS)
        _relay_debounce.pop(ch, None)
        relay_ch = _get_relay(ch)
        relay_ch["inbox"].append(m)
        relay_ch["last_message_at"] = _time.time()
        log.info(f"voice_relay[{ch}]: debounce committed msg #{m['id']}: {m['text'][:80]!r}")
        evt = relay_ch.get("inbox_event")
        if evt:
            evt.set()

    task = _asyncio.ensure_future(_commit(channel, msg))
    _relay_debounce[channel] = {"task": task, "msg": msg}

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

    # Attach current Samaritan emotion state for TTS expression injection
    emotion = None
    try:
        from database import fetch_dicts
        import math
        _half_lives = {
            "hostile-anger": 120, "fear-and-overwhelm": 120,
            "despair-and-shame": 240, "depleted-disengagement": 240,
            "vigilant-suspicion": 180,
            "exuberant-joy": 360, "competitive-pride": 360,
            "compassionate-gratitude": 480, "peaceful-contentment": 480,
            "playful-amusement": 480,
        }
        rows = await fetch_dicts(
            "SELECT emotion_label, cluster, intensity, "
            "TIMESTAMPDIFF(MINUTE, created_at, NOW()) AS age_minutes "
            "FROM samaritan_emotions "
            "WHERE subject='samaritan' AND created_at >= NOW() - INTERVAL 4 HOUR "
            "ORDER BY created_at DESC LIMIT 1"
        )
        if rows:
            r = rows[0]
            age_min = float(r.get("age_minutes", 0))
            cluster = r.get("cluster", "")
            hl = _half_lives.get(cluster, 360)
            raw_int = float(r.get("intensity", 0.5))
            decayed = raw_int * math.pow(0.5, age_min / hl)
            if decayed >= 0.1:
                emotion = {
                    "label": r["emotion_label"],
                    "cluster": cluster,
                    "intensity": round(decayed, 2),
                }
    except Exception:
        pass

    return JSONResponse({
        "status": "ok",
        "message_id": msg["id"],
        "reply_to": msg.get("reply_to", ""),
        "text": msg["text"],
        "channel": channel,
        "emotion": emotion,
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
                    "pstree", "-p", pane_pid,
                    stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
                )
                pstree_out, _ = await proc_pg.communicate()
                if b"claude" in pstree_out:
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
                        "pstree", "-p", ppid,
                        stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
                    )
                    pstree_out, _ = await p3.communicate()
                    if b"claude" in pstree_out:
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

# Thread sessions spawned dynamically: channel → tmux_session name
_dynamic_channels: Dict[str, str] = {}

# Max simultaneous dynamic thread sessions
_MAX_THREAD_SESSIONS = 5

# Idle TTL for dynamic sessions (seconds) — reap after this long with no activity
_THREAD_SESSION_TTL = 1800  # 30 minutes

_THREAD_START_SCRIPT = os.path.expanduser(
    "~/projects/samaritan-work/claude-thread-start.sh"
)

# Chat sessions spawned dynamically: channel → { tmux_session, database }
_chat_channels: Dict[str, dict] = {}

# Max simultaneous dynamic chat sessions
_MAX_CHAT_SESSIONS = 10

# Idle TTL for chat sessions (seconds) — reap after this long with no activity
_CHAT_SESSION_TTL = 3600  # 1 hour

_CHAT_START_SCRIPT = os.path.expanduser(
    "~/projects/samaritan-work/claude-chat-start.sh"
)


def _get_dispatch(channel: str = "default") -> dict:
    if channel not in _dispatch_channels:
        # For dynamic thread channels, tmux session name was registered via
        # _register_thread_channel(); fall back to static map then default.
        tmux_sess = (
            _dynamic_channels.get(channel)
            or _DISPATCH_TMUX_SESSIONS.get(channel)
            or "samaritan-work"
        )
        _dispatch_channels[channel] = {
            "pending_prompt": None,
            "response_event": None,
            "response_text": None,
            "last_submit_at": 0.0,
            "client_id": None,
            "turn_count": 0,
            "tmux_session": tmux_sess,
        }
        _dispatch_locks[channel] = _asyncio.Lock()
    return _dispatch_channels[channel]


def _register_thread_channel(channel: str, tmux_session: str) -> None:
    """Register a dynamic thread channel → tmux session mapping."""
    _dynamic_channels[channel] = tmux_session
    # Update dispatch dict if already created (e.g. from a pre-spawn check)
    if channel in _dispatch_channels:
        _dispatch_channels[channel]["tmux_session"] = tmux_session


async def _spawn_thread_session(channel: str) -> tuple:
    """Spawn a new Claude Code tmux session for a dynamic thread channel.

    Returns (ok: bool, tmux_session: str, message: str).
    Enforces _MAX_THREAD_SESSIONS cap.
    """
    # Already registered and alive?
    if channel in _dynamic_channels:
        tmux_sess = _dynamic_channels[channel]
        proc = await _asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", tmux_sess,
            stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
        )
        if await proc.wait() == 0:
            return True, tmux_sess, "already running"

    # Cap check
    active = sum(
        1 for ch in _dynamic_channels
        if _dispatch_channels.get(ch, {}).get("last_submit_at", 0) > 0
    )
    if active >= _MAX_THREAD_SESSIONS:
        return False, "", (
            f"Thread session cap reached ({_MAX_THREAD_SESSIONS} active). "
            "Finish or close another thread first."
        )

    # Derive short tmux session name from channel
    # channel = "slack-thread-553459" → tmux = "samaritan-slack-553459"
    suffix = channel.split("-")[-1] if "-" in channel else channel[:8]
    tmux_session = f"samaritan-slack-{suffix}"

    log.info(f"Spawning thread session: channel={channel} tmux={tmux_session}")
    proc = await _asyncio.create_subprocess_exec(
        "bash", _THREAD_START_SCRIPT, tmux_session,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=45)
    output = stdout.decode().strip()

    if output.startswith("started:"):
        _register_thread_channel(channel, tmux_session)
        log.info(f"Thread session started: {tmux_session}")
        return True, tmux_session, "started"
    else:
        log.error(f"Thread session spawn failed: {output} / {stderr.decode().strip()}")
        return False, "", f"Failed to start Claude session: {output}"


async def _reap_idle_thread_sessions() -> None:
    """Kill dynamic thread sessions idle longer than _THREAD_SESSION_TTL."""
    now = _time.time()
    for channel in list(_dynamic_channels.keys()):
        dispatch = _dispatch_channels.get(channel)
        if not dispatch:
            continue
        last = dispatch.get("last_submit_at", 0)
        if last > 0 and (now - last) > _THREAD_SESSION_TTL:
            tmux_sess = _dynamic_channels[channel]
            log.info(f"Reaping idle thread session: {tmux_sess} (idle {int(now-last)}s)")
            proc = await _asyncio.create_subprocess_exec(
                "tmux", "kill-session", "-t", tmux_sess,
                stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
            )
            await proc.wait()
            _dynamic_channels.pop(channel, None)
            _dispatch_channels.pop(channel, None)
            _dispatch_locks.pop(channel, None)


def _register_chat_channel(channel: str, tmux_session: str, database: str) -> None:
    """Register a dynamic chat channel → tmux session + database mapping."""
    _chat_channels[channel] = {"tmux_session": tmux_session, "database": database}
    _dynamic_channels[channel] = tmux_session
    if channel in _dispatch_channels:
        _dispatch_channels[channel]["tmux_session"] = tmux_session


async def _spawn_chat_session(channel: str, database: str = None) -> tuple:
    """Spawn a new Claude Code tmux session for a dynamic chat channel.

    Returns (ok: bool, tmux_session: str, message: str).
    Enforces _MAX_CHAT_SESSIONS cap.
    """
    # Derive chat name and database from channel: "chat-myproject" → name "myproject"
    chat_name = channel[5:] if channel.startswith("chat-") else channel
    if not database:
        database = chat_name
    tmux_session = f"chat_{chat_name}"

    # Already registered and alive?
    if channel in _chat_channels:
        ts = _chat_channels[channel]["tmux_session"]
        proc = await _asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", ts,
            stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
        )
        if await proc.wait() == 0:
            return True, ts, "already running"

    # Cap check — count active chat channels
    active = sum(
        1 for ch in _chat_channels
        if _dispatch_channels.get(ch, {}).get("last_submit_at", 0) > 0
    )
    if active >= _MAX_CHAT_SESSIONS:
        return False, "", (
            f"Chat session cap reached ({_MAX_CHAT_SESSIONS} active). "
            "Close another chat first."
        )

    log.info(f"Spawning chat session: channel={channel} tmux={tmux_session} db={database}")
    proc = await _asyncio.create_subprocess_exec(
        "bash", _CHAT_START_SCRIPT, chat_name, database,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=120)
    except _asyncio.TimeoutError:
        log.error(f"Chat session spawn timed out: {channel}")
        return False, "", "Timed out starting Claude session"

    output = stdout.decode().strip()

    if output.startswith("started:"):
        _register_chat_channel(channel, tmux_session, database)
        log.info(f"Chat session started: {tmux_session} (db={database})")
        return True, tmux_session, "started"
    else:
        log.error(f"Chat session spawn failed: {output} / {stderr.decode().strip()}")
        return False, "", f"Failed to start Claude session: {output}"


async def _reap_idle_chat_sessions() -> None:
    """Kill chat sessions idle longer than _CHAT_SESSION_TTL."""
    now = _time.time()
    for channel in list(_chat_channels.keys()):
        dispatch = _dispatch_channels.get(channel)
        if not dispatch:
            continue
        last = dispatch.get("last_submit_at", 0)
        if last > 0 and (now - last) > _CHAT_SESSION_TTL:
            info = _chat_channels[channel]
            tmux_sess = info["tmux_session"]
            log.info(f"Reaping idle chat session: {tmux_sess} (idle {int(now-last)}s)")
            proc = await _asyncio.create_subprocess_exec(
                "tmux", "kill-session", "-t", tmux_sess,
                stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
            )
            await proc.wait()
            _chat_channels.pop(channel, None)
            _dynamic_channels.pop(channel, None)
            _dispatch_channels.pop(channel, None)
            _dispatch_locks.pop(channel, None)


def _format_dispatch_prompt(text: str, source: str = "voice",
                            emotion: dict = None,
                            location: dict = None,
                            location_name: str = None,
                            context_prefix: str = None) -> str:
    """Format user text with source/emotion/location metadata prefix for Claude.

    context_prefix: pre-built enrichment block prepended before the [source|...] line.
    location_name:  resolved place name for GPS coords (injected as |near:<name>).
    """
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
    if location:
        lat = location.get("latitude", "")
        lon = location.get("longitude", "")
        acc = location.get("accuracy_m")
        prefix += f"|location:{lat},{lon}"
        if acc is not None:
            prefix += f"±{acc}m"
        if location_name:
            prefix += f"|near:{location_name}"
    prefix += "]"
    prompt = f"{prefix} {text}"
    if context_prefix:
        prompt = f"{context_prefix} --- {prompt}"
    return prompt


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
    location = payload.get("location")
    channel = payload.get("channel", "default")
    client_id = payload.get("client_id")

    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)

    # Reap idle sessions opportunistically on each submit
    _asyncio.create_task(_reap_idle_thread_sessions())
    _asyncio.create_task(_reap_idle_chat_sessions())

    # Chat channels must route to their own dedicated session — never fall back to
    # samaritan-work. After a server restart _dynamic_channels is empty; re-discover
    # from tmux before _get_dispatch creates a wrong fallback entry.
    if channel.startswith("chat-") and channel not in _dynamic_channels:
        await _discover_chat_sessions()
        if channel not in _dynamic_channels:
            # Session not in tmux at all — spawn on demand
            ok, _tmux, msg = await _spawn_chat_session(channel)
            if not ok:
                return JSONResponse({"error": msg}, status_code=503)

    dispatch = _get_dispatch(channel)
    tmux_session = dispatch["tmux_session"]

    # Check tmux session is alive; spawn dynamically for thread/chat channels
    proc = await _asyncio.create_subprocess_exec(
        "tmux", "has-session", "-t", tmux_session,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    if await proc.wait() != 0:
        # For dynamic thread channels, spawn on demand
        if channel.startswith("slack-thread-"):
            ok, tmux_session, msg = await _spawn_thread_session(channel)
            if not ok:
                return JSONResponse({"error": msg}, status_code=503)
            dispatch = _get_dispatch(channel)  # re-fetch with updated tmux_session
        # For dynamic chat channels, spawn on demand
        elif channel.startswith("chat-"):
            ok, tmux_session, msg = await _spawn_chat_session(channel)
            if not ok:
                return JSONResponse({"error": msg}, status_code=503)
            dispatch = _get_dispatch(channel)
        else:
            return JSONResponse(
                {"error": f"tmux session '{tmux_session}' not found"},
                status_code=503,
            )

    # Store voice-inworld emotion in samaritan_emotions (fire-and-forget)
    if emotion and emotion.get("source") == "voice-inworld" and emotion.get("emotion"):
        try:
            from emotions import store_voice_emotion
            _asyncio.ensure_future(store_voice_emotion(
                emotion_label=emotion["emotion"],
                confidence=float(emotion.get("confidence", 0.5)),
                prosody=emotion.get("prosody", ""),
                source="voice-inworld",
            ))
        except Exception as _e:
            log.warning(f"claude_submit: emotion store failed: {_e}")

    # Enrich: resolve GPS location + build context prefix (routine beliefs + pattern rules)
    context_prefix: str | None = None
    location_name: str | None = None
    try:
        from dispatch_enrich import build_context_prefix as _build_ctx
        context_prefix, location_name = await _build_ctx(text, location)
    except Exception as _enrich_err:
        log.debug(f"dispatch_enrich skipped: {_enrich_err}")

    # Format prompt with metadata prefix (+ enrichment if available)
    prompt = _format_dispatch_prompt(text, source, emotion, location,
                                     location_name=location_name,
                                     context_prefix=context_prefix)

    # Prepare response capture
    dispatch["response_text"] = None
    dispatch["response_event"] = _asyncio.Event()
    dispatch["pending_prompt"] = text
    dispatch["last_submit_at"] = _time.time()
    dispatch["client_id"] = client_id
    dispatch["turn_count"] = 0
    dispatch["location"] = location  # stored for conv_log to save with ST memory timestamp

    # Collapse newlines — tmux send-keys -l treats \n as Enter, which fragments
    # multi-line content like photo analysis into separate submissions.
    prompt = prompt.replace("\n", " ").replace("\r", " ")

    # Safety cap: Claude Code's input buffer chokes on very long single-line pastes.
    # Truncate [Photo analysis: ...] blocks to keep total prompt under 2000 chars.
    import re as _re
    _MAX_ANALYSIS = 500  # max chars for inline photo analysis text
    _pa_match = _re.search(r'\[Photo analysis: (.+?)\]', prompt)
    if _pa_match and len(_pa_match.group(1)) > _MAX_ANALYSIS:
        truncated = _pa_match.group(1)[:_MAX_ANALYSIS].rsplit('. ', 1)[0] + '.'
        prompt = prompt[:_pa_match.start(1)] + truncated + prompt[_pa_match.end(1):]
        log.info(f"dispatch: truncated photo analysis from {len(_pa_match.group(1))} to {len(truncated)} chars")

    # For long prompts (e.g. photo analysis), send-keys -l can overflow the
    # terminal input buffer.  Use tmux load-buffer + paste-buffer instead,
    # which handles arbitrary length reliably.
    import tempfile as _tempfile, os as _os
    tmp_path = None
    try:
        with _tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          delete=False) as tmp:
            tmp.write(prompt)
            tmp_path = tmp.name

        # Load file into tmux paste buffer
        proc = await _asyncio.create_subprocess_exec(
            "tmux", "load-buffer", tmp_path,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        rc = await proc.wait()
        if rc != 0:
            return JSONResponse({"error": "tmux load-buffer failed"}, status_code=503)

        # Paste into the target session's pane
        proc2 = await _asyncio.create_subprocess_exec(
            "tmux", "paste-buffer", "-t", tmux_session,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        rc2 = await proc2.wait()
        if rc2 != 0:
            return JSONResponse({"error": "tmux paste-buffer failed"}, status_code=503)
    finally:
        if tmp_path:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass

    # Brief pause to let Claude Code's input renderer process the pasted text
    # before sending Enter. Without this, Enter can arrive before the paste is
    # fully processed, leaving text stuck in the input buffer.
    await _asyncio.sleep(0.3)

    # Send Enter to submit the pasted text, with retry if it gets stuck
    for _enter_attempt in range(3):
        proc3 = await _asyncio.create_subprocess_exec(
            "tmux", "send-keys", "-t", tmux_session, "Enter",
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.PIPE,
        )
        await proc3.wait()

        if _enter_attempt < 2:
            # Check if Claude accepted the input by looking at the pane content.
            # If the pasted text is still visible in the last lines, Enter didn't take.
            await _asyncio.sleep(0.5)
            cap = await _asyncio.create_subprocess_exec(
                "tmux", "capture-pane", "-t", tmux_session, "-p", "-l", "5",
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            cap_out, _ = await cap.communicate()
            pane_tail = cap_out.decode("utf-8", errors="replace")
            # If pane shows the user's text still sitting in input (not a ">" prompt
            # or processing indicator), the Enter was lost — retry
            if prompt[:40] in pane_tail:
                log.warning(f"dispatch: Enter attempt {_enter_attempt + 1} didn't take, retrying...")
                await _asyncio.sleep(0.3)
                continue
            break  # Enter was accepted
        break

    prompt_len = len(prompt)
    loc_str = f"{location['latitude']},{location['longitude']}" if location else "none"
    log.info(f"dispatch: sent to {tmux_session} ch={channel} src={source} "
             f"emotion={emotion.get('emotion') if emotion else 'none'} "
             f"location={loc_str} "
             f"len={prompt_len} "
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

    # Chat channels must have their own dedicated session — don't fall back
    # to samaritan-work and falsely report alive.
    if channel.startswith("chat-") and channel not in _dynamic_channels:
        return JSONResponse({
            "channel": channel,
            "tmux_session": None,
            "tmux_alive": False,
            "claude_alive": False,
            "enabled": False,
        })

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
                "pstree", "-p", pane_pid,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            pstree_out, _ = await proc3.communicate()
            claude_alive = (b"claude" in pstree_out)

    return JSONResponse({
        "channel": channel,
        "tmux_session": tmux_session,
        "tmux_alive": tmux_alive,
        "claude_alive": claude_alive,
        "enabled": claude_alive,  # backward compat with frontend relay check
    })


# ═══════════════════════════════════════════════════════════════════════════
# Chat session management (dynamic per-chat Claude Code instances)
# ═══════════════════════════════════════════════════════════════════════════

async def endpoint_chat_start(request: Request) -> JSONResponse:
    """Start a Claude Code chat session on demand.

    Body (JSON):
        name     : str  (chat name, e.g. 'myproject')
        database : str  (optional, defaults to name)

    Returns immediately with {"status": "starting"|"already_running"}.
    Caller should poll /claude/status?channel=chat-<name> until claude_alive=true.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    name = payload.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Missing name"}, status_code=400)

    database = payload.get("database", name).strip()
    channel = f"chat-{name}"

    # Fast check — already registered and alive?
    if channel in _chat_channels:
        ts = _chat_channels[channel]["tmux_session"]
        proc = await _asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", ts,
            stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
        )
        if await proc.wait() == 0:
            return JSONResponse({"status": "already_running", "channel": channel, "database": database})

    # Fire off spawn in background — don't block the HTTP response
    _asyncio.ensure_future(_spawn_chat_session(channel, database))
    return JSONResponse({"status": "starting", "channel": channel, "database": database})


async def _discover_chat_sessions() -> None:
    """Scan tmux for existing chat_* sessions not yet in _chat_channels."""
    proc = await _asyncio.create_subprocess_exec(
        "tmux", "list-sessions", "-F", "#{session_name}",
        stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return
    for line in stdout.decode().strip().splitlines():
        sess = line.strip()
        if not sess.startswith("chat_"):
            continue
        chat_name = sess[5:]  # strip "chat_" prefix
        channel = f"chat-{chat_name}"
        if channel not in _chat_channels:
            _register_chat_channel(channel, sess, chat_name)
            log.info(f"Discovered existing chat session: {sess} → channel={channel}")


async def endpoint_chat_sessions(request: Request) -> JSONResponse:
    """List active chat sessions with their status."""
    await _discover_chat_sessions()
    sessions = []
    for channel, info in list(_chat_channels.items()):
        tmux_sess = info["tmux_session"]
        database = info["database"]

        # Check if tmux session is alive
        proc = await _asyncio.create_subprocess_exec(
            "tmux", "has-session", "-t", tmux_sess,
            stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
        )
        alive = (await proc.wait() == 0)

        dispatch = _dispatch_channels.get(channel, {})
        last_activity = dispatch.get("last_submit_at", 0)

        sessions.append({
            "channel": channel,
            "name": channel[5:] if channel.startswith("chat-") else channel,
            "tmux_session": tmux_sess,
            "database": database,
            "alive": alive,
            "last_activity": last_activity,
            "idle_seconds": int(_time.time() - last_activity) if last_activity > 0 else -1,
        })

    return JSONResponse({"sessions": sessions, "count": len(sessions)})


async def endpoint_chat_delete(request: Request) -> JSONResponse:
    """Kill a chat session's tmux session and clean up tracking state.

    Body (JSON):
        name : str  (chat name, e.g. 'myproject')
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    name = payload.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Missing name"}, status_code=400)

    channel = f"chat-{name}"
    info = _chat_channels.get(channel)

    if info:
        tmux_sess = info["tmux_session"]
        proc = await _asyncio.create_subprocess_exec(
            "tmux", "kill-session", "-t", tmux_sess,
            stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
        )
        await proc.wait()
        _chat_channels.pop(channel, None)
        _dynamic_channels.pop(channel, None)
        _dispatch_channels.pop(channel, None)
        _dispatch_locks.pop(channel, None)
        log.info(f"Chat session deleted: {tmux_sess}")
        return JSONResponse({"status": "deleted", "channel": channel})
    else:
        # Try killing tmux session by convention even if not tracked
        tmux_sess = f"chat_{name}"
        proc = await _asyncio.create_subprocess_exec(
            "tmux", "kill-session", "-t", tmux_sess,
            stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
        )
        await proc.wait()
        return JSONResponse({"status": "deleted", "channel": channel, "note": "not tracked"})


# ═══════════════════════════════════════════════════════════════════════════
# Slash command passthrough (for remote config control)
# ═══════════════════════════════════════════════════════════════════════════

# Allowed slash commands that can be sent remotely.
_ALLOWED_SLASH_COMMANDS = {
    "effort", "model", "config",
}

_CLAUDE_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
_CLAUDE_SESSIONS_DIR = os.path.expanduser("~/.claude/sessions")

_VALID_EFFORT_LEVELS = {"low", "medium", "high", "max", "auto"}
_VALID_MODELS = {"sonnet", "haiku", "opus"}


def _read_claude_settings() -> dict:
    """Read Claude Code settings.json."""
    try:
        with open(_CLAUDE_SETTINGS_PATH, "r") as f:
            return json.loads(f.read())
    except Exception:
        return {}


def _write_claude_settings(settings: dict):
    """Write Claude Code settings.json."""
    with open(_CLAUDE_SETTINGS_PATH, "w") as f:
        f.write(json.dumps(settings, indent=2) + "\n")


def _find_active_sessions() -> list:
    """Find active Claude Code sessions from session files."""
    sessions = []
    try:
        for fname in os.listdir(_CLAUDE_SESSIONS_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(_CLAUDE_SESSIONS_DIR, fname)
            with open(fpath, "r") as f:
                data = json.loads(f.read())
            pid = data.get("pid")
            # Check if process is still running
            if pid and os.path.exists(f"/proc/{pid}"):
                sessions.append(data)
    except Exception:
        pass
    return sessions


async def _tmux_send_slash(tmux_session: str, command: str) -> bool:
    """Send a slash command to a tmux session. Returns True on success."""
    proc = await _asyncio.create_subprocess_exec(
        "tmux", "has-session", "-t", tmux_session,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    if await proc.wait() != 0:
        return False

    proc = await _asyncio.create_subprocess_exec(
        "tmux", "send-keys", "-l", "-t", tmux_session, command,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    if await proc.wait() != 0:
        return False

    proc2 = await _asyncio.create_subprocess_exec(
        "tmux", "send-keys", "-t", tmux_session, "Enter",
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    await proc2.wait()
    return True


async def endpoint_claude_slash(request: Request) -> JSONResponse:
    """View or change Claude Code session settings remotely.

    Body: { command: "/effort [level]", channel: "default" }

    Supported commands:
      /effort           — show current effort level (reads settings.json)
      /effort <level>   — set effort level (updates settings.json + sends to session)
      /model            — show current model (from session info)
      /model <name>     — switch model (sends to session)
      /config think     — sends to session (fire-and-forget)
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    command = payload.get("command", "").strip()
    channel = payload.get("channel", "default")

    if not command.startswith("/"):
        return JSONResponse({"error": "Command must start with /"}, status_code=400)

    # Parse command
    parts = command.lstrip("/").split(None, 1)
    cmd_word = parts[0].lower()
    cmd_arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd_word not in _ALLOWED_SLASH_COMMANDS:
        return JSONResponse(
            {"error": f"Command /{cmd_word} not in allowlist: {sorted(_ALLOWED_SLASH_COMMANDS)}"},
            status_code=403,
        )

    dispatch = _get_dispatch(channel)
    tmux_session = dispatch["tmux_session"]

    # ── /effort ──
    if cmd_word == "effort":
        settings = _read_claude_settings()
        current = settings.get("effortLevel", "auto")

        if not cmd_arg:
            # Query: return current effort level
            return JSONResponse({
                "status": "ok",
                "command": command,
                "output": f"Effort level: {current}",
            })

        # Set: validate and update
        level = cmd_arg.lower()
        if level not in _VALID_EFFORT_LEVELS:
            return JSONResponse({
                "status": "error",
                "command": command,
                "output": f"Invalid effort level '{level}'. Valid: {sorted(_VALID_EFFORT_LEVELS)}",
            }, status_code=400)

        # Update settings.json
        settings["effortLevel"] = level
        _write_claude_settings(settings)

        # Also send to active session so it picks up immediately
        sent = await _tmux_send_slash(tmux_session, command)

        return JSONResponse({
            "status": "ok",
            "command": command,
            "output": f"Effort level: {current} → {level}"
                      + (" (sent to session)" if sent else " (saved, session unreachable)"),
        })

    # ── /model ──
    if cmd_word == "model":
        if not cmd_arg:
            # Query: report what we know from session files
            sessions = _find_active_sessions()
            session_info = []
            for s in sessions:
                name = s.get("name", s.get("entrypoint", "unknown"))
                pid = s.get("pid")
                session_info.append(f"  {name} (pid {pid})")
            if session_info:
                output = "Active Claude Code sessions:\n" + "\n".join(session_info)
                output += "\n(Model is per-session runtime state — use /model <name> to change)"
            else:
                output = "No active Claude Code sessions found."
            return JSONResponse({
                "status": "ok",
                "command": command,
                "output": output,
            })

        # Set: send to session
        sent = await _tmux_send_slash(tmux_session, command)
        return JSONResponse({
            "status": "ok",
            "command": command,
            "output": f"Sent /model {cmd_arg} to {tmux_session}"
                      + (" ✓" if sent else " (session unreachable)"),
        })

    # ── /config think ──
    if cmd_word == "config":
        # /config think is interactive (opens a dialog) — handle via settings.json
        config_sub = cmd_arg.split(None, 1)
        sub_cmd = config_sub[0].lower() if config_sub else ""
        sub_arg = config_sub[1].strip() if len(config_sub) > 1 else ""

        if sub_cmd != "think":
            return JSONResponse({
                "status": "error",
                "command": command,
                "output": "Only /config think is supported remotely.",
            }, status_code=400)

        settings = _read_claude_settings()
        think_enabled = settings.get("alwaysThinkingEnabled", False)
        env = settings.get("env", {})
        max_tokens = env.get("MAX_THINKING_TOKENS", "adaptive (default)")

        if not sub_arg:
            # Query
            return JSONResponse({
                "status": "ok",
                "command": command,
                "output": f"Think: {'on' if think_enabled else 'off'}\n"
                          f"Max thinking tokens: {max_tokens}",
            })

        # Set: on/off/token count
        if sub_arg.lower() == "on":
            settings["alwaysThinkingEnabled"] = True
            _write_claude_settings(settings)
            return JSONResponse({
                "status": "ok",
                "command": command,
                "output": f"Think: off → on",
            })

        if sub_arg.lower() == "off":
            settings["alwaysThinkingEnabled"] = False
            _write_claude_settings(settings)
            return JSONResponse({
                "status": "ok",
                "command": command,
                "output": f"Think: on → off",
            })

        # Numeric: set MAX_THINKING_TOKENS
        try:
            tokens = int(sub_arg)
            if tokens < 0:
                raise ValueError
        except ValueError:
            return JSONResponse({
                "status": "error",
                "command": command,
                "output": f"Invalid value '{sub_arg}'. Use: on, off, or a token count (e.g. 10000)",
            }, status_code=400)

        if "env" not in settings:
            settings["env"] = {}
        settings["env"]["MAX_THINKING_TOKENS"] = str(tokens)
        _write_claude_settings(settings)
        return JSONResponse({
            "status": "ok",
            "command": command,
            "output": f"Max thinking tokens: {max_tokens} → {tokens}"
                      + (" (0 = disabled)" if tokens == 0 else ""),
        })


async def _detect_multi_topic(user_text: str, assistant_text: str) -> list[dict] | None:
    """Detect if a conversation turn covers multiple clearly distinct topics.

    Uses a cheap LLM (gpt-4o-mini) to analyze the turn. Returns None if single
    topic or on any error. Returns a list of segment dicts if multiple distinct
    topics are found:
        [{"topic_hint": str, "user_frag": str, "asst_frag": str}, ...]

    Only triggers when the turn is long enough to plausibly contain multiple topics.
    Falls back gracefully to single-topic behavior on timeout or parse failure.
    """
    # Skip short turns — not worth the LLM call
    if len(user_text) < 60 or len(assistant_text) < 120:
        return None

    prompt = (
        "Analyze if this conversation turn covers MULTIPLE CLEARLY DISTINCT topics.\n"
        "Only return multi=true when the topics are from completely different domains "
        "(e.g. debugging code + scheduling a meeting, weather + filing a bug report).\n"
        "Do NOT split when topics are related (same person, same codebase feature, same context).\n\n"
        f"USER: {user_text[:400]}\n"
        f"ASSISTANT: {assistant_text[:600]}\n\n"
        "Return valid JSON only. No explanation.\n"
        '- Single topic: {"multi": false}\n'
        '- Multiple topics: {"multi": true, "segments": ['
        '{"topic_hint": "2-3 word hint", "user_frag": "relevant user text", "asst_frag": "relevant assistant text"}'
        ']}\n'
        "Max 3 segments. Preserve all meaningful content across segments."
    )
    try:
        import asyncio as _aio_mt
        import json as _json_mt
        from agents import _build_lc_llm, _content_to_str
        from langchain_core.messages import HumanMessage as _HM

        llm = _build_lc_llm("gpt4om")
        resp = await _aio_mt.wait_for(
            llm.ainvoke([_HM(content=prompt)]),
            timeout=8.0,
        )
        raw = _content_to_str(resp.content).strip()
        # Strip markdown fences if present
        raw = _re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=_re.MULTILINE).strip()
        data = _json_mt.loads(raw)
        if not data.get("multi"):
            return None
        segments = data.get("segments") or []
        if len(segments) < 2:
            return None
        # Validate each segment has the required keys
        valid = [s for s in segments if s.get("user_frag") or s.get("asst_frag")]
        return valid if len(valid) >= 2 else None
    except Exception as _e:
        log.debug(f"_detect_multi_topic: fallback to single (err={_e})")
        return None


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

    # Respect per-workspace database override (from .workspace-db marker)
    _db_override = payload.get("database", "")
    _set_context(database=_db_override)

    # Shared timestamp for ST memory and location rows (same pattern as routes.py)
    from datetime import datetime as _dt, timezone as _tz
    _shared_ts = _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S")

    try:
        from memory import save_conversation_turn, _normalize_topic, load_topic_list

        # Topic splitting DISABLED 2026-04-10 — _detect_multi_topic was fragmenting
        # substantive multi-section responses into multiple rows with semantically
        # mismatched topics, plus duplicate copies. The integrity cost of splitting
        # exceeds the indexing benefit. Always store one row per turn with one
        # primary topic. Multi-topic queries can be handled by semantic search.
        # See feedback_memory_tier_purity.md and the audit performed 2026-04-10.
        _all_segment_ids: list[tuple[int, int]] = []

        # Single topic — generate slug and save as one turn
        topic_slug = await _generate_topic_slug(user_text, assistant_text)
        if topic_slug:
            # Prepend the tag so save_conversation_turn's _extract_topic_tag() finds it
            assistant_text = f"<<{topic_slug}>>" + assistant_text

        user_id, asst_id, topic = await save_conversation_turn(
            user_text=user_text,
            assistant_text=assistant_text,
            session_id=session_id,
            importance=importance,
            created_at=_shared_ts,
        )
        _all_segment_ids.append((user_id, asst_id))

        log.info(
            f"conv_log: topic={topic} user_id={user_id} asst_id={asst_id} "
            f"session={session_id}"
        )

        # Tag routine domains affected by this conversation turn.
        # Use user_text ONLY — the user is the source of truth for real-world state.
        # Tagging on assistant_text too caused false positives when responses
        # explained or discussed routines without any actual schedule change.
        # Apply tags to ALL memory rows from this turn (covers multi-topic splits too).
        _tags = _detect_affects_tags(user_text)
        if _tags:
            _tag_ids = [x for pair in _all_segment_ids for x in pair if x]
            if _tag_ids:
                from database import execute_sql as _exec_sql_tags
                from memory import _ST as _st_tags
                _tags_json = json.dumps(_tags).replace("'", "\\'")
                _dur, _rec = _detect_affects_metadata(_tags)
                _dur_sql = str(_dur) if _dur is not None else "NULL"
                _rec_sql = f"'{_rec}'" if _rec else "NULL"
                _id_list = ",".join(str(i) for i in _tag_ids)
                await _exec_sql_tags(
                    f"UPDATE {_st_tags()} SET affects_tags = '{_tags_json}', "
                    f"effect_duration_days = {_dur_sql}, effect_recurs = {_rec_sql} "
                    f"WHERE id IN ({_id_list})"
                )
                log.info(f"conv_log: affects_tags={_tags} applied to ids={_tag_ids}")

        # Activity hook: periodic reflection — every N real turns
        global _cogn_turn_counter
        _cogn_turn_counter += 1
        if _cogn_turn_counter % _COGN_REFLECT_EVERY == 0:
            import asyncio as _aio
            _aio.ensure_future(_queue_cogn_step(
                f"Reflect on recent conversation (turn {_cogn_turn_counter}): "
                f"extract insights, update beliefs, check for new goals"
            ))

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
                    "source": user_emotion.get("source", "inferred"),
                    "context": "live inference by Claude Code with conversation context",
                }
                emotion_written = await _write_emotion(emotion_item, confidence_threshold=0.0)
                if emotion_written:
                    log.info(f"conv_log: live emotion '{user_emotion.get('emotion_label')}' written for user_id={user_id}")
            except Exception as e:
                log.warning(f"conv_log: emotion write failed: {e}")

        # Write Samaritan self-assessed emotion if provided
        sam_emotion = payload.get("samaritan_emotion")
        if sam_emotion and asst_id:
            try:
                from emotions import _write_emotion as _write_sam_emo
                # Map cluster to a core emotion for the 7-dimension vector
                _cluster_to_core = {
                    "exuberant-joy": "happy", "peaceful-contentment": "happy",
                    "compassionate-gratitude": "happy", "competitive-pride": "happy",
                    "playful-amusement": "happy", "depleted-disengagement": "sad",
                    "vigilant-suspicion": "fearful", "hostile-anger": "angry",
                    "fear-and-overwhelm": "fearful", "despair-and-shame": "sad",
                }
                _sam_cluster = sam_emotion.get("cluster", "")
                _sam_item = {
                    "memory_id": asst_id,
                    "_memory_table": "shortterm",
                    "core_emotion": _cluster_to_core.get(_sam_cluster, "happy"),
                    "emotion_label": sam_emotion.get("emotion_label", ""),
                    "intensity": sam_emotion.get("intensity", 0.5),
                    "confidence": 0.9,
                    "source": "inferred",
                    "subject": "samaritan",
                    "cluster": _sam_cluster,
                    "context": "Samaritan self-assessed emotional state",
                }
                _sam_written = await _write_sam_emo(_sam_item, confidence_threshold=0.0)
                if _sam_written:
                    log.info(f"conv_log: samaritan emotion '{sam_emotion.get('emotion_label')}' ({_sam_cluster}) written for asst_id={asst_id}")
            except Exception as e:
                log.warning(f"conv_log: samaritan emotion write failed: {e}")

        # Extract and store InWorld voice-detected emotion from dispatch prefix
        # Format: [voice|emotion:LABEL(CONFIDENCE)|prosody:...|...]
        if _dp_match:
            _iw_em = _re.search(r'emotion:([\w-]+)\(([0-9.]+)\)', user_text_raw)
            if _iw_em:
                _iw_label = _iw_em.group(1)
                _iw_conf = float(_iw_em.group(2))
                _iw_pros = _re.search(r'prosody:([^|\]]+)', user_text_raw)
                _iw_prosody = _iw_pros.group(1).strip() if _iw_pros else ""
                import asyncio as _aio_iw
                from emotions import store_voice_emotion as _store_iw
                _aio_iw.ensure_future(_store_iw(_iw_label, _iw_conf, _iw_prosody))
                log.info(f"conv_log: queued voice-inworld emotion {_iw_label}({_iw_conf})")

        # Save GPS location if present on the matched dispatch channel
        # (stored at submit time by endpoint_claude_submit)
        if _dp_match:
            for _ch_name, _disp in _dispatch_channels.items():
                _loc = _disp.get("location")
                if _loc and _disp.get("last_submit_at", 0) > 0:
                    _loc_age = _time.time() - _disp["last_submit_at"]
                    if _loc_age < 600:
                        try:
                            from memory import save_location
                            await save_location(
                                lat=_loc["latitude"],
                                lon=_loc["longitude"],
                                accuracy_m=_loc.get("accuracy_m"),
                                session_id=session_id,
                                created_at=_shared_ts,
                            )
                            _disp["location"] = None  # consume once
                            log.info(f"conv_log: location saved lat={_loc['latitude']} lon={_loc['longitude']}")
                        except Exception as _loc_err:
                            log.warning(f"conv_log: location save failed: {_loc_err}")

        # Signal dispatch channel if a pending prompt is waiting for a response.
        # Only match conv_logs that contain the dispatch prefix (proves it came from
        # the tmux dispatch, not from VS Code or another session).
        if _dp_match:
            for ch_name, dispatch in _dispatch_channels.items():
                _age = _time.time() - dispatch.get("last_submit_at", 0)
                _turn = dispatch.get("turn_count", 0)
                # Turn 1: match by original prompt text (proves this conv_log belongs to
                # this dispatch). Turns 2+: match by client_id presence + recency only —
                # the original prompt no longer appears in subsequent user_text values.
                _prompt_match = (
                    _turn >= 1  # already matched once, trust client_id
                    or (dispatch.get("pending_prompt")
                        and dispatch["pending_prompt"][:50] in user_text)
                )
                if (dispatch.get("last_submit_at", 0) > 0
                        and _age < 600
                        and _prompt_match
                        and (dispatch.get("client_id") or dispatch.get("response_event"))):
                    # Strip topic tag from assistant_text before delivering
                    clean_asst = _re.sub(r'^<<[^>]*>>', '', assistant_text).strip()
                    # Skip truly empty responses (tool-call noise with no text)
                    if not clean_asst:
                        log.info(f"dispatch: skipping empty response for '{ch_name}'")
                        continue
                    log.info(f"dispatch: raw asst_text={assistant_text[:100]!r} clean={clean_asst[:100]!r}")

                    cb_client = dispatch.get("client_id")
                    if cb_client:
                        # Direct delivery to originating client queue (Slack/voice).
                        # Keep client_id alive for multi-turn tasks — cleared only
                        # when a new submit arrives (turn_count reset there).
                        # Intermediate turns (2+) are truncated to a summary line
                        # to avoid flooding Slack with full responses mid-task.
                        from state import push_tok as _push_tok, push_done as _push_done
                        turn_num = dispatch.get("turn_count", 0) + 1
                        dispatch["turn_count"] = turn_num
                        _SUMMARY_LIMIT = 280
                        if turn_num > 1 and len(clean_asst) > _SUMMARY_LIMIT:
                            # Truncate to first complete sentence or hard limit
                            truncated = clean_asst[:_SUMMARY_LIMIT]
                            last_period = truncated.rfind(". ")
                            if last_period > 80:
                                truncated = truncated[:last_period + 1]
                            deliver_text = f"_(turn {turn_num})_ {truncated}…"
                        else:
                            deliver_text = clean_asst
                        await _push_tok(cb_client, deliver_text)
                        await _push_done(cb_client)
                        log.info(f"dispatch: direct-delivered turn {turn_num} "
                                 f"({len(deliver_text)} chars) to {cb_client} via ch '{ch_name}'")
                    else:
                        # Legacy poll path — store for _try_consume
                        dispatch["response_text"] = clean_asst
                        dispatch["pending_prompt"] = None
                        dispatch["last_submit_at"] = 0
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
# Photo analysis & Drive upload HTTP endpoints (for frontend camera feature)
# ═══════════════════════════════════════════════════════════════════════════

# --- Eidetic correlation stash ---
# Maps image_hash → {analysis_text, task_type, location_lat, location_lon,
#                     session_id, timestamp}
# or image_hash → {drive_file_id, file_name, timestamp}
# When both halves arrive, eidetic_save fires as a background task.
import hashlib as _hashlib
import time as _eidetic_time
import asyncio as _eidetic_asyncio

_eidetic_stash: dict = {}       # image_hash → dict
_EIDETIC_STASH_TTL = 180        # seconds before stale entries are purged


def _image_hash(image_b64: str) -> str:
    """Compute a stable hash of the raw base64 image data."""
    raw = image_b64.split(",", 1)[-1] if "," in image_b64 else image_b64
    return _hashlib.sha256(raw[:8192].encode()).hexdigest()[:16]


def _purge_stash():
    """Remove stale stash entries older than TTL."""
    now = _eidetic_time.time()
    expired = [k for k, v in _eidetic_stash.items()
               if now - v.get("timestamp", 0) > _EIDETIC_STASH_TTL]
    for k in expired:
        del _eidetic_stash[k]


_EIDETIC_FILLER = _STOP_WORDS | frozenset(
    "image photo picture captures shows displays depicts features appears visible "
    "looking seen view angle perspective detailed clearly slightly somewhat "
    "likely appears seems suggests indicates positioned located placed "
    "upper lower left right center top bottom side area section portion "
    "overall impression scene background foreground prominent dominant".split()
)


def _eidetic_topic_slug(analysis_text: str, max_words: int = 3) -> str:
    """Generate a concise topic slug (up to 3 words) from photo analysis text.

    Extracts the most descriptive nouns from the first sentence of the analysis,
    skipping generic photo-description filler words.
    """
    # First sentence only — it's the best summary
    first_sent = _re.split(r'[.\n]', analysis_text[:400])[0]
    first_sent = _re.sub(r'\*\*[^*]*\*\*', '', first_sent)  # strip markdown bold
    first_sent = _re.sub(r'[^a-zA-Z\s]', ' ', first_sent)
    words = first_sent.lower().split()
    significant = [w for w in words if w not in _EIDETIC_FILLER and len(w) > 2]
    if not significant:
        return "photo-analysis"
    # Take first N unique significant words (order preserves sentence meaning)
    seen = set()
    slug_words = []
    for w in significant:
        if w not in seen:
            seen.add(w)
            slug_words.append(w)
            if len(slug_words) >= max_words:
                break
    return "-".join(slug_words) if slug_words else "photo-analysis"


async def _fire_eidetic_save(analysis_text: str, drive_file_id: str,
                              task_type: str, file_name: str,
                              location_lat: float, location_lon: float,
                              session_id: str):
    """Background eidetic save — called when both analysis and upload are done."""
    try:
        topic = file_name or _eidetic_topic_slug(analysis_text)
        await eidetic_save(
            topic=topic,
            content=analysis_text,
            drive_file_id=drive_file_id,
            task_type=task_type,
            importance=5,
            source="assistant",
            analysis_model="gemini-2.5-flash",
            location_lat=location_lat,
            location_lon=location_lon,
            session_id=session_id,
        )
        log.info(f"eidetic: auto-saved for {topic} (drive_file_id={drive_file_id})")
    except Exception as e:
        log.warning(f"eidetic: auto-save failed: {e}")


def _try_eidetic_merge(img_hash: str):
    """Check if both analysis and upload data are in the stash; if so, fire eidetic save."""
    entry = _eidetic_stash.get(img_hash)
    if not entry:
        return
    if "analysis_text" in entry and "drive_file_id" in entry:
        _eidetic_asyncio.ensure_future(_fire_eidetic_save(
            analysis_text=entry["analysis_text"],
            drive_file_id=entry["drive_file_id"],
            task_type=entry.get("task_type", "general"),
            file_name=entry.get("file_name", ""),
            location_lat=entry.get("location_lat", 0.0),
            location_lon=entry.get("location_lon", 0.0),
            session_id=entry.get("session_id", ""),
        ))
        del _eidetic_stash[img_hash]


async def endpoint_analyze_photo(request):
    """HTTP wrapper for the analyze_photo MCP tool."""
    try:
        body = await request.json()
        image_b64 = body.get("image_b64", "")
        prompt = body.get("prompt", "Describe what you see in this photo in detail.")
        task_type = body.get("task_type", "general")
        if not image_b64:
            return JSONResponse({"error": "no image data"}, status_code=400)
        result = await analyze_photo(
            prompt=prompt,
            image_b64=image_b64,
            task_type=task_type,
        )
        # Stash analysis result for eidetic correlation
        if result and not result.startswith("analyze_photo error"):
            _purge_stash()
            img_hash = _image_hash(image_b64)
            entry = _eidetic_stash.setdefault(img_hash, {})
            entry["analysis_text"] = result
            entry["task_type"] = task_type
            entry["location_lat"] = body.get("location_lat", 0.0)
            entry["location_lon"] = body.get("location_lon", 0.0)
            entry["session_id"] = body.get("session_id", "")
            entry["timestamp"] = _eidetic_time.time()
            _try_eidetic_merge(img_hash)
        return JSONResponse({"result": result})
    except Exception as e:
        log.error(f"endpoint_analyze_photo error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def endpoint_drive_upload_photo(request):
    """Upload a base64 JPEG to Google Drive photos folder."""
    try:
        body = await request.json()
        image_b64 = body.get("image_b64", "")
        file_name = body.get("file_name", "")
        folder_id = body.get("folder_id", "")
        if not image_b64:
            return JSONResponse({"error": "no image data"}, status_code=400)
        if not file_name:
            import time as _time
            file_name = f"samaritan_capture_{int(_time.time())}.jpg"
        from drive import run_drive_op
        result = await run_drive_op("create_image", None, file_name, image_b64, folder_id or None)
        # Extract file ID from result string "Uploaded 'name' — id: XXXXX"
        file_id = ""
        if "id:" in result:
            file_id = result.split("id:")[-1].strip()
        # Stash upload result for eidetic correlation
        if file_id:
            _purge_stash()
            img_hash = _image_hash(image_b64)
            entry = _eidetic_stash.setdefault(img_hash, {})
            entry["drive_file_id"] = file_id
            entry["file_name"] = file_name
            entry["timestamp"] = _eidetic_time.time()
            _try_eidetic_merge(img_hash)
        return JSONResponse({"result": result, "file_id": file_id, "file_name": file_name})
    except Exception as e:
        log.error(f"endpoint_drive_upload_photo error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def endpoint_eidetic_save(request):
    """Save an eidetic (visual) memory via REST — called by frontend after photo analysis."""
    try:
        body = await request.json()
        result = await eidetic_save(
            topic=body.get("topic") or _eidetic_topic_slug(body.get("content", "")),
            content=body.get("content", ""),
            drive_file_id=body.get("drive_file_id", ""),
            task_type=body.get("task_type", "general"),
            importance=body.get("importance", 5),
            source=body.get("source", "assistant"),
            analysis_model=body.get("analysis_model", "gemini-2.5-flash"),
            location_lat=body.get("location_lat", 0.0),
            location_lon=body.get("location_lon", 0.0),
            memory_link=body.get("memory_link", ""),
            session_id=body.get("session_id", ""),
        )
        return JSONResponse({"result": result})
    except Exception as e:
        log.error(f"endpoint_eidetic_save error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def endpoint_google_drive(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await google_drive(
            operation=body.get("operation", "read"),
            file_id=body.get("file_id", ""),
            file_name=body.get("file_name", ""),
            content=body.get("content", ""),
            folder_id=body.get("folder_id", ""),
        )
        return JSONResponse({"result": result})
    except Exception as e:
        log.error(f"endpoint_google_drive error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def endpoint_url_extract_tavily(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await url_extract_tavily(
            url=body.get("url", ""),
            query=body.get("query", ""),
        )
        return JSONResponse({"result": result})
    except Exception as e:
        log.error(f"endpoint_url_extract_tavily error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def endpoint_xai_search(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await xai_search(query=body.get("query", ""))
        return JSONResponse({"result": result})
    except Exception as e:
        log.error(f"endpoint_xai_search error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def endpoint_sonar_answer(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await sonar_answer(query=body.get("query", ""), model=body.get("model", "sonar-pro"))
        return JSONResponse({"result": result})
    except Exception as e:
        log.error(f"endpoint_sonar_answer error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def endpoint_google_search(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await google_search(query=body.get("query", ""))
        return JSONResponse({"result": result})
    except Exception as e:
        log.error(f"endpoint_google_search error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def endpoint_llm_call(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        result = await llm_call(
            model=body.get("model", "reason-gemini"),
            prompt=body.get("prompt", ""),
            mode=body.get("mode", "text"),
        )
        return JSONResponse({"result": result})
    except Exception as e:
        log.error(f"endpoint_llm_call error: {e}")
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

    def get_lifespan(self):
        """Return the streamable HTTP session manager lifespan for the main app.

        Called by llmem-gw.py after get_routes() to compose the session manager's
        task group into the Starlette app lifespan. Without this, /mcp returns
        RuntimeError: Task group is not initialized.

        Multiple uvicorn servers share the same Starlette app, so this lifespan
        is called once per server. The session manager only supports a single
        run() call. We handle this by having the first server own the session
        manager lifetime; subsequent servers yield immediately. Safe in asyncio
        (single-threaded) because there is no await between the flag check and set.
        """
        import contextlib

        session_manager = getattr(mcp, "_session_manager", None)
        if session_manager is None:
            return None

        # Module-level-style flag — reset on each get_lifespan() call (i.e. per process).
        _state = {"active": False}

        @contextlib.asynccontextmanager
        async def _lifespan(app):
            if not _state["active"]:
                _state["active"] = True
                async with session_manager.run():
                    yield
            else:
                # Another server already owns the session manager — just pass through.
                yield

        return _lifespan

    def get_routes(self) -> List[Route]:
        """Mount the FastMCP SSE app routes + streamable HTTP + conv_log + voice relay endpoints."""
        sse_app = mcp.sse_app()
        routes = list(sse_app.routes)
        # Streamable HTTP transport (/mcp) for Codex CLI and other newer MCP clients
        http_app = mcp.streamable_http_app()
        routes.extend(http_app.routes)
        routes.append(Route("/conv_log", endpoint_conv_log, methods=["POST"]))
        routes.append(Route("/voice_relay/submit", endpoint_voice_relay_submit, methods=["POST"]))
        routes.append(Route("/voice_relay/poll", endpoint_voice_relay_poll, methods=["GET"]))
        routes.append(Route("/voice_relay/status", endpoint_voice_relay_status, methods=["GET"]))
        routes.append(Route("/ged/start", endpoint_ged_start, methods=["POST"]))
        routes.append(Route("/voice_relay/disable", endpoint_voice_relay_disable, methods=["POST"]))
        routes.append(Route("/claude/submit", endpoint_claude_submit, methods=["POST"]))
        routes.append(Route("/claude/poll", endpoint_claude_poll, methods=["GET"]))
        routes.append(Route("/claude/status", endpoint_claude_status, methods=["GET"]))
        routes.append(Route("/claude/slash", endpoint_claude_slash, methods=["POST"]))
        routes.append(Route("/chat/start", endpoint_chat_start, methods=["POST"]))
        routes.append(Route("/chat/sessions", endpoint_chat_sessions, methods=["GET"]))
        routes.append(Route("/chat/delete", endpoint_chat_delete, methods=["POST"]))
        routes.append(Route("/mcp/health", endpoint_mcp_health, methods=["GET"]))
        routes.append(Route("/analyze_photo", endpoint_analyze_photo, methods=["POST"]))
        routes.append(Route("/drive_upload_photo", endpoint_drive_upload_photo, methods=["POST"]))
        routes.append(Route("/eidetic_save", endpoint_eidetic_save, methods=["POST"]))
        routes.append(Route("/google_drive", endpoint_google_drive, methods=["POST"]))
        routes.append(Route("/url_extract_tavily", endpoint_url_extract_tavily, methods=["POST"]))
        routes.append(Route("/xai_search", endpoint_xai_search, methods=["POST"]))
        routes.append(Route("/sonar_answer", endpoint_sonar_answer, methods=["POST"]))
        routes.append(Route("/google_search", endpoint_google_search, methods=["POST"]))
        routes.append(Route("/llm_call", endpoint_llm_call, methods=["POST"]))
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
