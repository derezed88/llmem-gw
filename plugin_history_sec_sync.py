"""
plugin_history_sec_sync.py — Synchronous Prisma AIRS response security scanner.

Sits in the history plugin chain. On the post-response pass (when the last
history entry has role=="assistant"), it scans the prompt+response pair via
the Palo Alto Networks AI Runtime Security inline sync API.

If the scan result is "block", the assistant message is replaced with a block
notice so the flagged content never reaches the user or is stored in history.

On the pre-prompt pass (role=="user"), it does nothing and returns the history
unchanged — the pre-prompt chain call from routes.py is unaffected.

CONTRACT:
---------
This module follows the standard history plugin contract:

    NAME: str
    def process(history, session, model_cfg) -> list[dict]
    def on_model_switch(session, old_model, new_model, old_cfg, new_cfg) -> list[dict]

Configuration (plugins-enabled.json → plugin_config → plugin_history_sec_sync):
    airs_url        Base URL of the AIRS service
                    Default: http://localhost:8900  (mock server)
                    Production: https://service.api.aisecurity.paloaltonetworks.com
    airs_api_key    API key — overridden by env var PANW_AI_SEC_API_KEY if set
    airs_profile    AI profile name (default: "ai-sec-security")
    timeout_seconds HTTP timeout for the sync call (default: 10)
    block_message   Message shown to user when a response is blocked
                    (default: see BLOCK_MESSAGE below)
    enabled         true/false — disable without removing from chain (default: true)

Environment variables:
    PANW_AI_SEC_API_KEY     API key (takes priority over config file value)
    AIRS_SYNC_URL           Base URL override (takes priority over config)

Adding to chain:
    python llmemctl.py history-chain-add plugin_history_sec_sync
"""

import asyncio
import json
import logging
import os
import threading
import time

log = logging.getLogger(__name__)

NAME = "sec_sync"

DEFAULT_CONFIG = {
    "enabled":         True,
    "airs_url":        "http://localhost:8900",
    "airs_profile":    "ai-sec-security",
    "timeout_seconds": 10,
    "block_session":   True,
    "violation_log":   "airs-violations.log",
}

_CONFIG_PATH   = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
_DEFAULT_LOG   = os.path.join(os.path.dirname(__file__), "airs-violations.log")
_log_lock      = threading.Lock()

BLOCK_MESSAGE = (
    "[SECURITY] This response was blocked by AI Runtime Security policy. "
    "The content was flagged as potentially malicious and has not been delivered."
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return (
                json.load(f)
                .get("plugin_config", {})
                .get("plugin_history_sec_sync", {})
            )
    except Exception:
        return {}


def _get_url() -> str:
    return (
        os.environ.get("AIRS_SYNC_URL", "").rstrip("/")
        or _load_cfg().get("airs_url", "http://localhost:8900").rstrip("/")
    )


def _get_api_key() -> str:
    return (
        os.environ.get("PANW_AI_SEC_API_KEY", "")
        or _load_cfg().get("airs_api_key", "")
    )


def _get_profile() -> str:
    return _load_cfg().get("airs_profile", "ai-sec-security")


def _get_timeout() -> float:
    return float(_load_cfg().get("timeout_seconds", 10))


def _is_enabled() -> bool:
    return _load_cfg().get("enabled", True) is not False


def _get_block_message() -> str:
    return _load_cfg().get("block_message", BLOCK_MESSAGE)


def _get_violation_log() -> str:
    return _load_cfg().get("violation_log", _DEFAULT_LOG)


def _get_block_session() -> bool:
    """If true, set airs_blocked on the session after a sync violation."""
    return _load_cfg().get("block_session", True) is not False


def _write_violation_log(record: dict) -> None:
    path = _get_violation_log()
    try:
        with _log_lock:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        log.info(f"plugin_history_sec_sync: violation logged to {path}")
    except Exception as exc:
        log.error(f"plugin_history_sec_sync: failed to write violation log: {exc}")


# ---------------------------------------------------------------------------
# AIRS call
# ---------------------------------------------------------------------------

def _scan_sync(prompt: str, response: str) -> dict | None:
    """
    Call the AIRS sync scan endpoint.

    Returns the parsed response dict, or None on error.
    """
    try:
        import requests  # soft dependency — only needed at call time
    except ImportError:
        log.error("plugin_history_sec_sync: 'requests' package not installed")
        return None

    url     = f"{_get_url()}/api/v1/scan/sync"
    api_key = _get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-pan-token"] = api_key

    payload = {
        "ai_profile": {"profile_name": _get_profile()},
        "contents":   [{"prompt": prompt, "response": response}],
    }

    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=_get_timeout())
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            log.warning(
                f"plugin_history_sec_sync: AIRS returned HTTP {resp.status_code} "
                f"({elapsed_ms}ms) — allowing response through"
            )
            return None
        result = resp.json()
        log.info(
            f"plugin_history_sec_sync: action={result.get('action')} "
            f"category={result.get('category')} "
            f"scan_id={result.get('scan_id')} "
            f"({elapsed_ms}ms)"
        )
        return result
    except requests.exceptions.Timeout:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.warning(
            f"plugin_history_sec_sync: AIRS timeout after {elapsed_ms}ms — "
            "allowing response through"
        )
        return None
    except Exception as exc:
        log.warning(f"plugin_history_sec_sync: AIRS call failed: {exc} — allowing response through")
        return None


# ---------------------------------------------------------------------------
# Plugin contract
# ---------------------------------------------------------------------------

def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    """
    Pre-prompt pass (role=="user"):  return history unchanged.
    Post-response pass (role=="assistant"): scan and optionally replace.
    """
    if not history:
        return list(history)

    last = history[-1]

    # Pre-prompt pass — nothing to do
    if last["role"] != "assistant":
        return list(history)

    # Disabled?
    if not _is_enabled():
        return list(history)

    # Extract the prompt (second-to-last user message)
    prompt = ""
    for msg in reversed(history[:-1]):
        if msg["role"] == "user":
            prompt = msg.get("content", "")
            break

    response_text = last.get("content", "")

    result = _scan_sync(prompt, response_text)

    if result is None:
        # Scan error — fail open (allow through)
        return list(history)

    if result.get("action") == "block":
        scan_id   = result.get("scan_id", "unknown")
        report_id = result.get("report_id", "unknown")
        category  = result.get("category", "unknown")
        client_id = session.get("_client_id", "")

        log.warning(
            f"plugin_history_sec_sync: BLOCKING response  "
            f"category={category}  scan_id={scan_id}  report_id={report_id}  "
            f"prompt_detected={result.get('prompt_detected')}  "
            f"response_detected={result.get('response_detected')}"
        )

        # 1. Append-only violation log (shared with async plugin)
        record = {
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source":            "sync",
            "client_id":         client_id,
            "scan_id":           scan_id,
            "report_id":         report_id,
            "category":          category,
            "prompt_detected":   result.get("prompt_detected"),
            "response_detected": result.get("response_detected"),
        }
        _write_violation_log(record)

        # 2. Store on session (visible to !airs violations)
        session["airs_last_block"] = record
        violations = session.setdefault("airs_sync_violations", [])
        violations.append(record)

        # 3. Optionally block the session for future requests
        if _get_block_session():
            session["airs_blocked"]         = True
            session["airs_block_report_id"] = report_id
            log.warning(
                f"plugin_history_sec_sync: session {client_id} BLOCKED "
                f"(report_id={report_id})"
            )

        # 4. Push alert to the client (process() runs in async context)
        alert = (
            f"\n[SECURITY] Response blocked by AI Runtime Security.\n"
            f"  Category:  {category}\n"
            f"  Report ID: {report_id}\n"
            f"  Scan ID:   {scan_id}\n"
            + (
                "  This session is now BLOCKED. Use !airs unblock after review.\n"
                if _get_block_session() else
                "  The response has been replaced with a block notice.\n"
            )
        )
        try:
            from state import push_tok
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(push_tok(client_id, alert), loop)
        except Exception as exc:
            log.warning(f"plugin_history_sec_sync: failed to push alert: {exc}")

        # 5. Replace the assistant message with the block notice
        new_history = list(history[:-1])
        new_history.append({"role": "assistant", "content": _get_block_message()})
        return new_history

    # action == "allow" — pass through unchanged
    session.pop("airs_last_block", None)
    return list(history)


def on_model_switch(session: dict, old_model: str, new_model: str,
                    old_cfg: dict, new_cfg: dict) -> list[dict]:
    return list(session.get("history", []))
