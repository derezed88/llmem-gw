"""
plugin_history_sec_async.py — Asynchronous Prisma AIRS response security scanner.

Sits in the history plugin chain. On the post-response pass (role=="assistant"),
it submits the prompt+response pair to the AIRS async scan endpoint in a
background thread — it does NOT block or delay the response to the user.

When scan results become available, the background poller:
  1. Writes a JSON line to airs-violations.log (append-only audit trail)
  2. Pushes a real-time alert to the session owner via push_tok
  3. Sets session["airs_blocked"] = True so subsequent requests are rejected

NOTE: The response is already delivered by the time the scan resolves.
Use plugin_history_sec_sync if you need to block responses before delivery.
Use this plugin for audit/telemetry with session-level enforcement after-the-fact.

CONTRACT:
---------
    NAME: str
    def process(history, session, model_cfg) -> list[dict]
    def on_model_switch(session, old_model, new_model, old_cfg, new_cfg) -> list[dict]

Configuration (plugins-enabled.json → plugin_config → plugin_history_sec_async):
    airs_url            Base URL of the AIRS service
                        Default: http://localhost:8900  (mock server)
                        Production: https://service.api.aisecurity.paloaltonetworks.com
    airs_api_key        API key — overridden by PANW_AI_SEC_API_KEY env var
    airs_profile        AI profile name (default: "ai-sec-security")
    poll_interval       Seconds between poll attempts (default: 3)
    poll_max_attempts   Max poll attempts before giving up (default: 5)
    violation_log       Path to append-only JSON-lines violation log
                        Default: airs-violations.log (same dir as this file)
    block_on_violation  true/false — whether to block the session on a "block" result
                        Default: true
    enabled             true/false (default: true)

Environment variables:
    PANW_AI_SEC_API_KEY     API key (takes priority over config)
    AIRS_ASYNC_URL          Base URL override (takes priority over config)

Adding to chain:
    python llmemctl.py history-chain-add plugin_history_sec_async
"""

import asyncio
import json
import logging
import os
import threading
import time

log = logging.getLogger(__name__)

NAME = "sec_async"

DEFAULT_CONFIG = {
    "enabled":            False,
    "airs_url":           "http://localhost:8900",
    "airs_profile":       "ai-sec-security",
    "poll_interval":      3,
    "poll_max_attempts":  5,
    "block_on_violation": True,
    "violation_log":      "airs-violations.log",
}

_CONFIG_PATH   = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
_DEFAULT_LOG   = os.path.join(os.path.dirname(__file__), "airs-violations.log")
_log_lock      = threading.Lock()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return (
                json.load(f)
                .get("plugin_config", {})
                .get("plugin_history_sec_async", {})
            )
    except Exception:
        return {}


def _get_url() -> str:
    return (
        os.environ.get("AIRS_ASYNC_URL", "").rstrip("/")
        or _load_cfg().get("airs_url", "http://localhost:8900").rstrip("/")
    )


def _get_api_key() -> str:
    return (
        os.environ.get("PANW_AI_SEC_API_KEY", "")
        or _load_cfg().get("airs_api_key", "")
    )


def _get_profile() -> str:
    return _load_cfg().get("airs_profile", "ai-sec-security")


def _get_poll_interval() -> float:
    return float(_load_cfg().get("poll_interval", 3))


def _get_poll_max_attempts() -> int:
    return int(_load_cfg().get("poll_max_attempts", 5))


def _get_violation_log() -> str:
    return _load_cfg().get("violation_log", _DEFAULT_LOG)


def _get_block_on_violation() -> bool:
    return _load_cfg().get("block_on_violation", True) is not False


def _is_enabled() -> bool:
    return _load_cfg().get("enabled", True) is not False


def _make_headers() -> dict:
    h = {"Content-Type": "application/json"}
    key = _get_api_key()
    if key:
        h["x-pan-token"] = key
    return h


# ---------------------------------------------------------------------------
# Violation log
# ---------------------------------------------------------------------------

def _write_violation_log(record: dict) -> None:
    """Append a JSON line to the violation log file (thread-safe)."""
    path = _get_violation_log()
    try:
        with _log_lock:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        log.info(f"plugin_history_sec_async: violation logged to {path}")
    except Exception as exc:
        log.error(f"plugin_history_sec_async: failed to write violation log: {exc}")


# ---------------------------------------------------------------------------
# Async submit + background poller
# ---------------------------------------------------------------------------

def _submit_async(prompt: str, response: str, req_id: int) -> tuple[str, str] | None:
    """Submit an async scan. Returns (scan_id, report_id) or None on error."""
    try:
        import requests
    except ImportError:
        log.error("plugin_history_sec_async: 'requests' package not installed")
        return None

    url = f"{_get_url()}/api/v1/scan/async"
    payload = [
        {
            "req_id": req_id,
            "scan_req": {
                "ai_profile": {"profile_name": _get_profile()},
                "contents": [{"prompt": prompt, "response": response}],
            },
        }
    ]
    try:
        resp = requests.post(url, json=payload, headers=_make_headers(), timeout=10)
        if resp.status_code != 200:
            log.warning(f"plugin_history_sec_async: submit returned HTTP {resp.status_code}")
            return None
        data  = resp.json()
        scans = data.get("scans", [])
        if not scans:
            log.warning("plugin_history_sec_async: empty scans in submit response")
            return None
        first     = scans[0]
        scan_id   = first.get("scan_id", "")
        report_id = first.get("report_id", "")
        log.info(f"plugin_history_sec_async: submitted scan_id={scan_id}")
        return scan_id, report_id
    except Exception as exc:
        log.warning(f"plugin_history_sec_async: submit failed: {exc}")
        return None


def _poll_and_handle(
    scan_id: str,
    report_id: str,
    session: dict,
    client_id: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Background thread: poll until result is ready, then:
      1. Write violation to append-only log
      2. Push real-time alert to session owner
      3. Flag session as blocked (if block_on_violation=true)
    """
    try:
        import requests
    except ImportError:
        return

    url       = f"{_get_url()}/api/v1/scan/results"
    interval  = _get_poll_interval()
    max_tries = _get_poll_max_attempts()

    for attempt in range(1, max_tries + 1):
        time.sleep(interval)
        try:
            resp = requests.get(
                url,
                params={"scan_ids": scan_id},
                headers=_make_headers(),
                timeout=10,
            )
            if resp.status_code != 200:
                log.warning(
                    f"plugin_history_sec_async: poll attempt {attempt} HTTP {resp.status_code}"
                )
                continue

            data    = resp.json()
            results = data.get("results", [])
            if not results:
                continue

            result = results[0]
            status = result.get("status", "")

            if status == "pending":
                log.debug(
                    f"plugin_history_sec_async: scan_id={scan_id} still pending "
                    f"(attempt {attempt}/{max_tries})"
                )
                continue

            if status == "not_found":
                log.warning(f"plugin_history_sec_async: scan_id={scan_id} not found on server")
                return

            # Full result received
            action   = result.get("action", "unknown")
            category = result.get("category", "unknown")
            log.info(
                f"plugin_history_sec_async: scan_id={scan_id} "
                f"action={action} category={category} "
                f"prompt_detected={result.get('prompt_detected')} "
                f"response_detected={result.get('response_detected')}"
            )

            if action == "block":
                _handle_violation(result, session, client_id, loop)
            return

        except Exception as exc:
            log.warning(f"plugin_history_sec_async: poll attempt {attempt} error: {exc}")

    log.warning(
        f"plugin_history_sec_async: gave up polling scan_id={scan_id} "
        f"after {max_tries} attempts"
    )


def _handle_violation(
    result: dict,
    session: dict,
    client_id: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Record violation, alert session owner, optionally block the session."""
    scan_id   = result.get("scan_id", "unknown")
    report_id = result.get("report_id", "unknown")
    category  = result.get("category", "unknown")

    log.warning(
        f"plugin_history_sec_async: POLICY VIOLATION (post-delivery) "
        f"scan_id={scan_id} report_id={report_id} category={category} — "
        "response was already delivered"
    )

    # 1. Append-only violation log
    record = {
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source":            "async",
        "client_id":         client_id,
        "scan_id":           scan_id,
        "report_id":         report_id,
        "category":          category,
        "prompt_detected":   result.get("prompt_detected"),
        "response_detected": result.get("response_detected"),
    }
    _write_violation_log(record)

    # 2. Record in session history
    violations = session.setdefault("airs_async_violations", [])
    violations.append(record)

    # 3. Block the session so future requests are rejected
    if _get_block_on_violation():
        session["airs_blocked"]           = True
        session["airs_block_report_id"]   = report_id
        log.warning(
            f"plugin_history_sec_async: session {client_id} BLOCKED "
            f"(report_id={report_id})"
        )

    # 4. Push alert to session owner (from background thread → event loop)
    alert = (
        f"\n⚠ [SECURITY ALERT] AI Runtime Security flagged a response in this session.\n"
        f"  Category: {category}\n"
        f"  Report ID: {report_id}\n"
        f"  Scan ID: {scan_id}\n"
        + (
            "  This session is now BLOCKED. Contact an administrator to review.\n"
            if _get_block_on_violation() else
            "  The response was already delivered. Check the violation log for details.\n"
        )
    )
    try:
        from state import push_tok
        asyncio.run_coroutine_threadsafe(push_tok(client_id, alert), loop)
    except Exception as exc:
        log.warning(f"plugin_history_sec_async: failed to push alert: {exc}")


# ---------------------------------------------------------------------------
# Plugin contract
# ---------------------------------------------------------------------------

_req_id_counter = 0
_req_id_lock    = threading.Lock()


def _next_req_id() -> int:
    global _req_id_counter
    with _req_id_lock:
        _req_id_counter += 1
        return _req_id_counter


def process(history: list[dict], session: dict, model_cfg: dict) -> list[dict]:
    """
    Pre-prompt pass (role=="user"):  return unchanged.
    Post-response pass (role=="assistant"):
        - submit async scan in background thread (non-blocking)
        - return history unchanged immediately
    """
    if not history:
        return list(history)

    last = history[-1]

    if last["role"] != "assistant":
        return list(history)

    if not _is_enabled():
        return list(history)

    # Extract the most recent user prompt
    prompt = ""
    for msg in reversed(history[:-1]):
        if msg["role"] == "user":
            prompt = msg.get("content", "")
            break

    response_text = last.get("content", "")
    req_id        = _next_req_id()
    client_id     = session.get("_client_id", "")

    # Capture the running event loop while still in async context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    def _background():
        ids = _submit_async(prompt, response_text, req_id)
        if ids and loop:
            scan_id, report_id = ids
            _poll_and_handle(scan_id, report_id, session, client_id, loop)

    t = threading.Thread(target=_background, daemon=True, name=f"airs-async-{req_id}")
    t.start()

    # Always return history unchanged — async does not block delivery
    return list(history)


def on_model_switch(session: dict, old_model: str, new_model: str,
                    old_cfg: dict, new_cfg: dict) -> list[dict]:
    return list(session.get("history", []))
