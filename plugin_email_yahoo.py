"""
Yahoo Email Triage Plugin for llmem-gw

Phase 1: Scan → Classify → Log → Notify (no destructive actions)
Phase 2: Rules-based auto-handling (delete, spam, archive, folder)

Timer polls INBOX every N minutes for unseen emails. Each email is:
1. Checked against rules table (fast, no LLM cost)
2. If no rule matches, classified by cheap LLM
3. Decision logged to triage table
4. Notifications fired for important emails

Commands:
    !email                 — show recent triage decisions
    !email rules           — list active rules
    !email rules add ...   — add a rule
    !email rules disable N — disable rule
    !email rules delete N  — delete rule
    !email rules stats     — hit count summary
    !email scan            — trigger immediate scan
    !email review          — show unreviewed decisions
    !email approve N       — mark decision as reviewed/correct
    !email override N <action> — override a decision
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from typing import Dict, Any, List

from plugin_loader import BasePlugin
from config import log

# Import the IMAP client from the yahoo-imap project
_YAHOO_IMAP_PATH = os.environ.get(
    "YAHOO_IMAP_PATH",
    os.path.expanduser("~/projects/yahoo-imap"),
)
if _YAHOO_IMAP_PATH not in sys.path:
    sys.path.insert(0, _YAHOO_IMAP_PATH)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    """Load email triage config from plugins-enabled.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")
    try:
        with open(path) as f:
            return json.load(f).get("plugin_config", {}).get("email_yahoo", {})
    except Exception:
        return {}


_DEFAULT_CFG = {
    "enabled": True,
    "scan_interval_m": 15,
    "scan_limit": 50,            # max unseen emails per scan
    "classify_model": None,      # None = use model_roles["email_classifier"]
    "body_preview_chars": 500,   # how much body to send to LLM
    "auto_execute": False,       # Phase 2: actually perform actions
    "notify_classifications": ["notify"],  # which classifications fire notifications
}


def _effective_cfg() -> dict:
    cfg = _cfg()
    return {k: cfg.get(k, v) for k, v in _DEFAULT_CFG.items()}


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _RULES() -> str:
    from database import get_tables_for_model
    return get_tables_for_model().get("email_rules", "samaritan_email_rules")


def _TRIAGE() -> str:
    from database import get_tables_for_model
    return get_tables_for_model().get("email_triage", "samaritan_email_triage")


# ---------------------------------------------------------------------------
# Stats (in-memory, since restart)
# ---------------------------------------------------------------------------

_stats = {
    "scans": 0,
    "emails_scanned": 0,
    "rules_matched": 0,
    "llm_classified": 0,
    "notifications_sent": 0,
    "last_scan_at": None,
    "last_scan_duration_s": None,
    "last_error": None,
}

_runtime_enabled = None  # None = use config; True/False = override
_wake_event = None


def trigger_now():
    global _wake_event
    if _wake_event:
        _wake_event.set()


def set_enabled(val):
    global _runtime_enabled
    _runtime_enabled = val
    if val:
        trigger_now()


def is_enabled() -> bool:
    if _runtime_enabled is not None:
        return _runtime_enabled
    return _effective_cfg().get("enabled", True)


def get_stats() -> dict:
    return dict(_stats)


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

async def _load_rules() -> list:
    """Load enabled rules sorted by priority."""
    from database import fetch_dicts
    return await fetch_dicts(
        f"SELECT * FROM {_RULES()} WHERE enabled = 1 ORDER BY priority, id"
    ) or []


def _match_rule(email_data: dict, rules: list) -> dict | None:
    """Check email against rules. Returns first matching rule or None."""
    sender = email_data.get("from", "")
    domain = ""
    m = re.search(r"@([\w.-]+)", sender)
    if m:
        domain = m.group(1).lower()
    subject = email_data.get("subject", "")
    body = email_data.get("body_preview", "")

    for rule in rules:
        match_type = rule.get("match_type", "")
        match_value = rule.get("match_value", "")
        match_mode = rule.get("match_mode", "contains")

        # Select the field to match against
        if match_type == "sender":
            field = sender.lower()
        elif match_type == "domain":
            field = domain
        elif match_type == "subject":
            field = subject.lower()
        elif match_type == "body":
            field = body.lower()
        elif match_type == "header":
            field = (sender + " " + subject).lower()
        else:
            continue

        # Apply match mode
        target = match_value.lower()
        matched = False
        if match_mode == "exact":
            matched = field == target
        elif match_mode == "contains":
            matched = target in field
        elif match_mode == "regex":
            try:
                matched = bool(re.search(match_value, field, re.IGNORECASE))
            except re.error:
                continue

        if matched:
            return rule

    return None


async def _record_rule_hit(rule_id: int):
    """Increment hit count and update last_hit_at for a rule."""
    from database import execute_sql
    await execute_sql(
        f"UPDATE {_RULES()} SET hit_count = hit_count + 1, "
        f"last_hit_at = NOW() WHERE id = {rule_id}"
    )


# ---------------------------------------------------------------------------
# Action execution (Phase 2)
# ---------------------------------------------------------------------------

async def _execute_action(
    client,          # open YahooIMAPClient
    uid: str,
    action: str,
    folder_name: str | None = None,
    also_notify: bool = False,
    email_data: dict | None = None,
) -> tuple[str, str | None]:
    """Execute a triage action against the mailbox.

    Returns (action_taken, moved_to_folder).
    action_taken is one of: moved, deleted, spam_marked, archived, notified, skipped, error:<msg>

    Supported actions:
      delete            — permanently delete
      spam              — move to Spam / Bulk Mail
      archive           — move to Archive
      folder            — move to folder_name (required)
      notify            — fire notification only, no move
      notify+folder     — notify AND move to folder_name (required)
      unsubscribe       — notify only (manual action needed)
      skip              — no-op
    """
    moved_to = None
    try:
        if action in ("delete",):
            client.delete_email(uid, folder="INBOX")
            return "deleted", None

        if action in ("spam",):
            # Yahoo uses "Bulk Mail" for spam folder
            dest = "Bulk Mail"
            client.move_email(uid, source="INBOX", destination=dest)
            return "spam_marked", dest

        if action in ("archive",):
            client.move_email(uid, source="INBOX", destination="Archive")
            return "archived", "Archive"

        if action in ("folder", "notify+folder"):
            if not folder_name:
                return "error:folder_name_required", None
            client.move_email(uid, source="INBOX", destination=folder_name)
            moved_to = folder_name
            return "moved", moved_to

        if action in ("notify", "unsubscribe", "skip"):
            return "notified" if action == "notify" else "skipped", None

    except Exception as e:
        log.warning(f"_execute_action uid={uid} action={action}: {e}")
        return f"error:{str(e)[:100]}", None

    return "skipped", None


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
You are an email triage classifier. Classify this email into exactly ONE category.

Categories:
- delete: Obvious junk, expired promotions, automated notifications with no value
- spam: Unsolicited commercial email, phishing attempts, scams
- archive: Newsletters, receipts, confirmations worth keeping but not urgent
- notify: Important email that the user should know about (job-related, financial, personal, from known contacts)
- skip: Uncertain — leave for manual review

Classify based on sender reputation, subject line, and content. Err toward 'skip' when uncertain.
{examples}
EMAIL:
From: {sender}
Subject: {subject}
Date: {date}
Body preview:
{body}

Respond with ONLY valid JSON (no markdown, no explanation):
{{"classification": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}
"""

_EXAMPLES_HEADER = """
Known classification patterns (follow these for similar emails):
{lines}

"""


async def _build_few_shot_examples() -> str:
    """Build one-shot example block from active rules.

    Each rule becomes one example line showing the pattern and expected action.
    Skips rules with action='skip' (uninformative) and disabled rules.
    """
    try:
        rules = await _load_rules()
        if not rules:
            return ""
        lines = []
        for r in rules:
            action = r.get("action", "")
            if action == "skip":
                continue
            match_type = r.get("match_type", "")
            match_value = r.get("match_value", "")
            folder = r.get("folder_name")
            also_notify = r.get("also_notify")
            action_label = action
            if also_notify and folder:
                action_label = f"notify (move to {folder})"
            elif folder:
                action_label = f"folder:{folder}"
            lines.append(f"- {match_type} '{match_value}' → {action_label}")
        if not lines:
            return ""
        return _EXAMPLES_HEADER.format(lines="\n".join(lines))
    except Exception:
        return ""


async def _classify_email(email_data: dict) -> dict:
    """Classify an email using a cheap LLM. Returns {classification, confidence, reasoning}."""
    from agents import llm_call

    cfg = _effective_cfg()
    try:
        from config import get_model_role
        model = cfg.get("classify_model") or get_model_role("email_classifier")
    except (KeyError, Exception):
        model = "summarizer-anthropic"  # cheap fallback

    examples = await _build_few_shot_examples()
    prompt = _CLASSIFY_PROMPT.format(
        examples=examples,
        sender=email_data.get("from", ""),
        subject=email_data.get("subject", ""),
        date=email_data.get("date", ""),
        body=email_data.get("body_preview", "")[:cfg.get("body_preview_chars", 500)],
    )

    try:
        result = await llm_call(
            model=model, prompt=prompt, mode="text",
            sys_prompt="none", history="none",
        )
        # Parse JSON response
        raw = result.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        parsed = json.loads(raw.strip())
        return {
            "classification": parsed.get("classification", "skip"),
            "confidence": float(parsed.get("confidence", 0.5)),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        log.warning(f"email classify failed: {e}")
        return {"classification": "skip", "confidence": 0.0, "reasoning": f"LLM error: {e}"}


# ---------------------------------------------------------------------------
# Triage scan
# ---------------------------------------------------------------------------

async def _already_triaged(email_uid: str) -> bool:
    """Check if we've already processed this email."""
    from database import execute_sql
    result = await execute_sql(
        f"SELECT 1 FROM {_TRIAGE()} WHERE email_uid = '{email_uid}' LIMIT 1"
    )
    return "1" in result.strip()


async def run_scan(search_criteria: str = "UNSEEN") -> dict:
    """Scan INBOX for emails matching search_criteria, classify, log, notify.

    search_criteria: IMAP search string (default: "UNSEEN").
      Use "SINCE DD-Mon-YYYY" for date-range backfill, "ALL" for everything.
    """
    from database import execute_sql, execute_insert, set_db_override
    from state import current_client_id

    set_db_override("mymcp")
    current_client_id.set("__email_triage__")

    cfg = _effective_cfg()
    scan_limit = cfg.get("scan_limit", 50)
    results = {"scanned": 0, "rules_matched": 0, "llm_classified": 0,
               "notifications": 0, "errors": 0, "skipped_already": 0}

    try:
        from yahoo_imap import YahooIMAPClient
    except ImportError as e:
        return {"error": f"yahoo_imap not available: {e}"}

    try:
        with YahooIMAPClient() as client:
            emails = client.list_emails(
                folder="INBOX", limit=scan_limit, search_criteria=search_criteria
            )
            if not emails:
                return {**results, "scanned": 0}

            rules = await _load_rules()

            for em in emails:
                uid = em.get("uid", "")

                # Skip if already triaged
                if await _already_triaged(uid):
                    results["skipped_already"] += 1
                    continue

                results["scanned"] += 1

                # Extract sender domain
                sender = em.get("from", "")
                domain = ""
                dm = re.search(r"@([\w.-]+)", sender)
                if dm:
                    domain = dm.group(1).lower()

                # Read body preview for classification
                try:
                    full = client.read_email(uid, "INBOX")
                    body_preview = (full.get("body", "") or "")[:cfg.get("body_preview_chars", 500)]
                except Exception:
                    body_preview = ""

                email_data = {
                    "from": sender,
                    "subject": em.get("subject", ""),
                    "date": em.get("date", ""),
                    "body_preview": body_preview,
                }

                # Try rules first
                matched_rule = _match_rule(email_data, rules)
                if matched_rule:
                    classification = matched_rule["action"]
                    confidence = 1.0
                    reasoning = f"Matched rule #{matched_rule['id']}: {matched_rule.get('rule_name', '')}"
                    rule_id = matched_rule["id"]
                    await _record_rule_hit(rule_id)
                    results["rules_matched"] += 1
                else:
                    # LLM classification
                    cl = await _classify_email(email_data)
                    classification = cl["classification"]
                    confidence = cl["confidence"]
                    reasoning = cl["reasoning"]
                    rule_id = None
                    results["llm_classified"] += 1

                # Log to triage table
                try:
                    s_sender = sender.replace("'", "''")[:255]
                    s_domain = domain.replace("'", "''")[:100]
                    s_subject = em.get("subject", "").replace("'", "''")[:500]
                    s_body = body_preview.replace("'", "''")
                    s_reasoning = reasoning.replace("'", "''")

                    rule_sql = f"'{rule_id}'" if rule_id else "NULL"
                    await execute_insert(
                        f"INSERT INTO {_TRIAGE()} "
                        f"(email_uid, folder, sender, sender_domain, subject, "
                        f"body_preview, classification, confidence, action_taken, "
                        f"matched_rule_id, llm_reasoning) "
                        f"VALUES ('{uid}', 'INBOX', '{s_sender}', '{s_domain}', "
                        f"'{s_subject}', '{s_body}', '{classification}', {confidence}, "
                        f"'logged', {rule_sql}, '{s_reasoning}')"
                    )
                except Exception as e:
                    log.warning(f"email triage log failed: {e}")
                    results["errors"] += 1

                # Determine folder_name and also_notify from matched rule (if any)
                rule_folder = matched_rule.get("folder_name") if matched_rule else None
                rule_also_notify = bool(matched_rule.get("also_notify")) if matched_rule else False

                # Notify: fire for notify classifications OR when rule has also_notify set
                notify_classes = cfg.get("notify_classifications", ["notify"])
                should_notify = (
                    classification in notify_classes
                    or "notify+" in classification
                    or rule_also_notify
                )
                if should_notify:
                    try:
                        import notifier
                        await notifier.fire_event(
                            "email_important",
                            f"📧 {sender}: {em.get('subject', '(no subject)')}\n"
                            f"Classification: {classification} ({confidence:.0%})\n"
                            f"{reasoning}",
                        )
                        await execute_sql(
                            f"UPDATE {_TRIAGE()} SET notified = TRUE "
                            f"WHERE email_uid = '{uid}' ORDER BY id DESC LIMIT 1"
                        )
                        results["notifications"] += 1
                    except Exception as e:
                        log.warning(f"email notify failed: {e}")

                # Execute action if auto_execute is enabled
                if cfg.get("auto_execute", False):
                    try:
                        exec_action = classification
                        exec_folder = rule_folder
                        action_taken, moved_to = await _execute_action(
                            client, uid, exec_action,
                            folder_name=exec_folder,
                            also_notify=rule_also_notify,
                            email_data=email_data,
                        )
                        moved_sql = f"'{moved_to.replace(chr(39), chr(39)*2)}'" if moved_to else "NULL"
                        await execute_sql(
                            f"UPDATE {_TRIAGE()} SET action_taken = '{action_taken}', "
                            f"moved_to_folder = {moved_sql} "
                            f"WHERE email_uid = '{uid}' ORDER BY id DESC LIMIT 1"
                        )
                        if action_taken.startswith("error:"):
                            results["errors"] += 1
                    except Exception as e:
                        log.warning(f"email execute failed uid={uid}: {e}")
                        results["errors"] += 1

    except Exception as e:
        log.error(f"email scan error: {e}")
        return {**results, "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Background timer task
# ---------------------------------------------------------------------------

async def email_triage_task() -> None:
    """Long-running asyncio task. Polls INBOX every scan_interval_m minutes."""
    from timer_registry import register_timer, timer_start, timer_end, timer_sleep

    global _wake_event
    _wake_event = asyncio.Event()

    cfg = _effective_cfg()
    interval_m = cfg.get("scan_interval_m", 15)
    register_timer("email_triage", f"{interval_m}m")

    # Initial delay — let system warm up
    timer_sleep("email_triage", 60)
    await asyncio.sleep(60)

    while True:
        try:
            if not is_enabled():
                await asyncio.sleep(300)
                continue

            cfg = _effective_cfg()
            interval_m = cfg.get("scan_interval_m", 15)
            register_timer("email_triage", f"{interval_m}m")

            t0 = timer_start("email_triage")
            results = await run_scan()

            _stats["scans"] += 1
            _stats["emails_scanned"] += results.get("scanned", 0)
            _stats["rules_matched"] += results.get("rules_matched", 0)
            _stats["llm_classified"] += results.get("llm_classified", 0)
            _stats["notifications_sent"] += results.get("notifications", 0)
            _stats["last_scan_at"] = time.time()
            _stats["last_scan_duration_s"] = round(time.time() - t0, 1) if t0 else None
            if results.get("error"):
                _stats["last_error"] = results["error"]

            timer_end("email_triage", t0)
            log.info(
                f"email_triage scan complete: scanned={results.get('scanned', 0)} "
                f"rules={results.get('rules_matched', 0)} llm={results.get('llm_classified', 0)} "
                f"notify={results.get('notifications', 0)} errors={results.get('errors', 0)}"
            )
        except Exception as e:
            log.error(f"email_triage_task error: {e}")
            _stats["last_error"] = str(e)[:200]

        # Sleep with wake support
        sleep_sec = max(60, interval_m * 60)
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=sleep_sec)
            _wake_event.clear()
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# !email command handlers
# ---------------------------------------------------------------------------

async def _cmd_email(args: str) -> str:
    """Handle !email commands."""
    from database import execute_sql, execute_insert, fetch_dicts
    from state import current_client_id
    from database import set_db_override

    set_db_override("mymcp")

    parts = args.strip().split(maxsplit=1) if args.strip() else []
    sub = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    # !email (no args) — show recent triage decisions
    if not sub:
        rows = await fetch_dicts(
            f"SELECT id, email_uid, sender, subject, classification, confidence, "
            f"action_taken, matched_rule_id, reviewed, created_at "
            f"FROM {_TRIAGE()} ORDER BY id DESC LIMIT 20"
        ) or []
        if not rows:
            return "No triage decisions yet. Run !email scan to start."
        lines = ["## Recent Email Triage (last 20)\n"]
        for r in rows:
            rev = "✓" if r.get("reviewed") else " "
            rule = f" rule#{r['matched_rule_id']}" if r.get("matched_rule_id") else ""
            lines.append(
                f"  [{rev}] #{r['id']} {r.get('classification','?')} "
                f"({r.get('confidence', 0):.0%}){rule} "
                f"— {r.get('sender', '?')[:30]}: {r.get('subject', '')[:50]}"
            )
        lines.append(f"\n{len(rows)} shown. !email review = unreviewed only")
        return "\n".join(lines)

    # !email scan [since N] — trigger immediate scan
    # "since N" fetches emails from the last N days (e.g. "!email scan since 3")
    if sub == "scan":
        criteria = "UNSEEN"
        if rest.startswith("since "):
            try:
                import datetime
                days = int(rest.split()[1])
                since_dt = datetime.date.today() - datetime.timedelta(days=days)
                criteria = f"SINCE {since_dt.strftime('%d-%b-%Y')}"
            except (IndexError, ValueError):
                return "Usage: !email scan since <days>  (e.g. !email scan since 3)"
        results = await run_scan(search_criteria=criteria)
        return (
            f"Scan complete ({criteria}): {results.get('scanned', 0)} emails, "
            f"{results.get('rules_matched', 0)} rules matched, "
            f"{results.get('llm_classified', 0)} LLM classified, "
            f"{results.get('notifications', 0)} notifications\n"
            f"{'Error: ' + results['error'] if results.get('error') else ''}"
        )

    # !email review — unreviewed decisions
    if sub == "review":
        rows = await fetch_dicts(
            f"SELECT id, sender, subject, classification, confidence, "
            f"llm_reasoning, matched_rule_id "
            f"FROM {_TRIAGE()} WHERE reviewed = FALSE ORDER BY id DESC LIMIT 30"
        ) or []
        if not rows:
            return "All decisions reviewed. ✓"
        lines = [f"## Unreviewed Triage Decisions ({len(rows)})\n"]
        for r in rows:
            rule = f" [rule#{r['matched_rule_id']}]" if r.get("matched_rule_id") else ""
            lines.append(
                f"  #{r['id']} **{r.get('classification','?')}** "
                f"({r.get('confidence', 0):.0%}){rule}\n"
                f"    {r.get('sender', '?')[:40]}: {r.get('subject', '')[:60]}\n"
                f"    Reason: {r.get('llm_reasoning', '')[:80]}"
            )
        lines.append("\n!email approve <id> | !email override <id> <action>")
        return "\n".join(lines)

    # !email approve N
    if sub == "approve":
        try:
            tid = int(rest)
            await execute_sql(
                f"UPDATE {_TRIAGE()} SET reviewed = TRUE WHERE id = {tid}"
            )
            return f"Triage #{tid} marked as reviewed/correct."
        except ValueError:
            return "Usage: !email approve <triage_id>"

    # !email override N <action>
    if sub == "override":
        p = rest.split(maxsplit=1)
        if len(p) < 2:
            return "Usage: !email override <triage_id> <delete|spam|archive|notify|skip>"
        try:
            tid = int(p[0])
            action = p[1].lower()
            valid = ("delete", "spam", "archive", "folder", "notify", "unsubscribe", "skip")
            if action not in valid:
                return f"Invalid action. Valid: {', '.join(valid)}"
            await execute_sql(
                f"UPDATE {_TRIAGE()} SET user_override = '{action}', reviewed = TRUE "
                f"WHERE id = {tid}"
            )
            return f"Triage #{tid} overridden to '{action}' and marked reviewed."
        except ValueError:
            return "Usage: !email override <triage_id> <action>"

    # !email auto on|off — toggle auto_execute in plugins-enabled.json
    if sub == "auto":
        import json as _json
        plugins_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins-enabled.json")
        try:
            with open(plugins_path) as f:
                pcfg = _json.load(f)
            email_cfg = pcfg.setdefault("plugin_config", {}).setdefault("email_yahoo", {})
            if rest in ("on", "true", "1"):
                email_cfg["auto_execute"] = True
                with open(plugins_path, "w") as f:
                    _json.dump(pcfg, f, indent=4)
                return "Auto-execute ON — actions will be performed on next scan."
            elif rest in ("off", "false", "0"):
                email_cfg["auto_execute"] = False
                with open(plugins_path, "w") as f:
                    _json.dump(pcfg, f, indent=4)
                return "Auto-execute OFF — triage logged only, no mailbox changes."
            else:
                current = email_cfg.get("auto_execute", False)
                return f"Auto-execute is currently {'ON' if current else 'OFF'}. Use: !email auto on|off"
        except Exception as e:
            return f"Error toggling auto_execute: {e}"

    # !email notify <on|off|status> [session_prefix]
    # Subscribe/unsubscribe a session prefix to email_important notifications.
    # Persisted in notifier.json — survives restarts and session reconnects.
    # Default prefix = current session's client_id prefix (e.g. "slack-", "samaritan-voice")
    # Examples:
    #   !email notify on              — subscribe current session
    #   !email notify on slack-       — subscribe all slack sessions
    #   !email notify off             — unsubscribe current session prefix
    #   !email notify status          — show current subscriptions
    if sub == "notify":
        try:
            import notifier as _notifier
            from state import current_client_id as _cid
        except Exception as e:
            return f"Notifier unavailable: {e}"

        notify_parts = rest.strip().split(maxsplit=1) if rest.strip() else []
        notify_sub = notify_parts[0].lower() if notify_parts else "status"
        # Prefix arg — or derive from current session
        if len(notify_parts) > 1:
            prefix = notify_parts[1].strip()
        else:
            cid = _cid.get() or ""
            # Derive prefix: keep up to and including first hyphen segment
            # e.g. "slack-abc123" → "slack-", "samaritan-voice-xyz" → "samaritan-voice-"
            parts = cid.split("-", 2)
            prefix = "-".join(parts[:2]) + "-" if len(parts) >= 2 else cid

        if notify_sub in ("on", "enable"):
            msg = _notifier.add_target_prefix(
                prefix, events=["email_important"], quiet_minutes=0
            )
            return f"{msg}\nEmail notifications will be delivered to sessions matching '{prefix}'."

        if notify_sub in ("off", "disable"):
            # Only remove email_important from this prefix's subscriptions, not full target
            for t in _notifier._targets:
                if t.get("client_id_prefix") == prefix:
                    t["events"].discard("email_important")
                    if not t["events"]:
                        _notifier._targets.remove(t)
                    _notifier._save()
                    return f"Email notifications disabled for prefix '{prefix}'."
            return f"No notification target found for prefix '{prefix}'."

        # status
        lines = ["## Email Notification Targets\n"]
        found = False
        for t in _notifier._targets:
            if "email_important" in t.get("events", set()):
                pfx = t.get("client_id_prefix") or f"session:{t['session_id']}"
                lines.append(f"  {pfx}  (quiet={t['quiet_minutes']}m)")
                found = True
        if not found:
            lines.append("  None. Use: !email notify on")
        lines.append(f"\nCurrent session prefix: '{prefix}'")
        return "\n".join(lines)

    # !email execute [N] — retroactively execute actions on logged triage entries
    # Optional N limits how many to process (default 50)
    if sub == "execute":
        try:
            limit = int(rest) if rest.strip().isdigit() else 50
        except ValueError:
            limit = 50
        rows = await fetch_dicts(
            f"SELECT id, email_uid, classification, matched_rule_id "
            f"FROM {_TRIAGE()} WHERE action_taken = 'logged' LIMIT {limit}"
        ) or []
        if not rows:
            return "No pending triage entries (action_taken='logged') to execute."

        # Load rules for folder_name lookup
        rules_map = {r["id"]: r for r in (await _load_rules())}

        executed = errors = 0
        try:
            from yahoo_imap import YahooIMAPClient
            with YahooIMAPClient() as client:
                for row in rows:
                    uid = row["email_uid"]
                    classification = row["classification"]
                    rule = rules_map.get(row.get("matched_rule_id")) if row.get("matched_rule_id") else None
                    folder_name = rule.get("folder_name") if rule else None
                    also_notify = bool(rule.get("also_notify")) if rule else False
                    action_taken, moved_to = await _execute_action(
                        client, uid, classification,
                        folder_name=folder_name,
                        also_notify=also_notify,
                    )
                    moved_sql = f"'{moved_to.replace(chr(39), chr(39)*2)}'" if moved_to else "NULL"
                    await execute_sql(
                        f"UPDATE {_TRIAGE()} SET action_taken = '{action_taken}', "
                        f"moved_to_folder = {moved_sql} WHERE id = {row['id']}"
                    )
                    if action_taken.startswith("error:"):
                        errors += 1
                    else:
                        executed += 1
        except Exception as e:
            return f"Execute failed: {e}"
        return f"Execute complete: {executed} actioned, {errors} errors (of {len(rows)} entries)"

    # !email stats
    if sub == "stats":
        s = get_stats()
        last = time.strftime("%H:%M:%S", time.localtime(s["last_scan_at"])) if s["last_scan_at"] else "never"
        lines = [
            "## Email Triage Stats (since restart)\n",
            f"  Scans: {s['scans']}",
            f"  Emails scanned: {s['emails_scanned']}",
            f"  Rules matched: {s['rules_matched']}",
            f"  LLM classified: {s['llm_classified']}",
            f"  Notifications: {s['notifications_sent']}",
            f"  Last scan: {last}  ({s['last_scan_duration_s']}s)",
            f"  Enabled: {is_enabled()}",
        ]
        if s["last_error"]:
            lines.append(f"  Last error: {s['last_error']}")
        return "\n".join(lines)

    # !email on/off
    if sub == "on":
        set_enabled(True)
        trigger_now()
        return "Email triage timer enabled."
    if sub == "off":
        set_enabled(False)
        return "Email triage timer disabled."

    # ── Rules subcommands ──────────────────────────────────────

    if sub == "rules":
        rparts = rest.strip().split(maxsplit=2) if rest.strip() else []
        rsub = rparts[0].lower() if rparts else ""

        # !email rules — list all
        if not rsub:
            rows = await fetch_dicts(
                f"SELECT * FROM {_RULES()} ORDER BY enabled DESC, priority, id"
            ) or []
            if not rows:
                return "No email rules defined.\n!email rules add <match_type>:<match_value> <action>"
            lines = ["## Email Rules\n"]
            for r in rows:
                status = "ON" if r.get("enabled") else "off"
                action_label = r.get("action", "?")
                if r.get("also_notify") and r.get("folder_name"):
                    action_label = f"notify+folder:{r['folder_name']}"
                elif r.get("folder_name"):
                    action_label = f"folder:{r['folder_name']}"
                lines.append(
                    f"  [{status}] #{r['id']} p={r.get('priority', 50)} "
                    f"{r.get('match_type', '?')}:{r.get('match_mode', '?')}:"
                    f"'{r.get('match_value', '')}' → **{action_label}** "
                    f"(hits={r.get('hit_count', 0)})"
                )
                if r.get("notes"):
                    lines.append(f"       {r['notes']}")
            return "\n".join(lines)

        # !email rules add domain:groupon.com delete [note text]
        # Compound: !email rules add domain:chase.com notify+folder:Bills [notes]
        if rsub == "add":
            rest2 = " ".join(rparts[1:]) if len(rparts) > 1 else ""
            add_parts = rest2.split(maxsplit=2)
            if len(add_parts) < 2:
                return (
                    "Usage: !email rules add <match>:<value> <action> [notes]\n"
                    "Actions: delete, spam, archive, notify, skip, folder:<name>, notify+folder:<name>\n"
                    "  Folder names with spaces: use quotes — folder:\"bills paid\" or notify+folder:\"bills paid\"\n"
                    "Examples:\n"
                    "  !email rules add domain:groupon.com delete\n"
                    "  !email rules add domain:chase.com notify+folder:Bills\n"
                    "  !email rules add domain:progressive.com notify+folder:\"bills paid\"\n"
                    "  !email rules add sender:recruiter@symbotic.com notify\n"
                    "  !email rules add domain:usps.com archive\n"
                    "  !email rules add body:\"win a prize\" spam"
                )
            match_spec = add_parts[0]
            action_spec = add_parts[1].lower()
            notes = add_parts[2] if len(add_parts) > 2 else ""

            # Parse compound action: notify+folder:Bills, folder:"My Bills", notify+folder:"bills paid"
            # Folder names with spaces must be quoted: folder:"bills paid"
            folder_name = None
            also_notify = False
            if action_spec.startswith("notify+folder:"):
                raw_folder = action_spec.split(":", 1)[1].strip('"\'')
                # Also consume quoted folder name that may have been split into notes
                if not raw_folder and notes:
                    quoted = notes.split(maxsplit=1)
                    raw_folder = quoted[0].strip('"\'')
                    notes = quoted[1] if len(quoted) > 1 else ""
                folder_name = raw_folder
                action = "folder"
                also_notify = True
            elif action_spec.startswith("folder:"):
                raw_folder = action_spec.split(":", 1)[1].strip('"\'')
                if not raw_folder and notes:
                    quoted = notes.split(maxsplit=1)
                    raw_folder = quoted[0].strip('"\'')
                    notes = quoted[1] if len(quoted) > 1 else ""
                folder_name = raw_folder
                action = "folder"
            else:
                action = action_spec

            valid_actions = ("delete", "spam", "archive", "folder", "notify", "unsubscribe", "skip")
            if action not in valid_actions:
                return f"Invalid action '{action}'. Valid: {', '.join(valid_actions)}"

            if action == "folder" and not folder_name:
                return "folder action requires a name: folder:<name> or notify+folder:<name>"

            if ":" not in match_spec:
                return "Match must be type:value (e.g. domain:groupon.com)"

            match_type, match_value = match_spec.split(":", 1)
            valid_types = ("sender", "domain", "subject", "body", "header")
            if match_type not in valid_types:
                return f"Invalid match type '{match_type}'. Valid: {', '.join(valid_types)}"

            s_val = match_value.strip("'\"").replace("'", "''")
            s_notes = notes.replace("'", "''") if notes else ""
            s_folder = folder_name.replace("'", "''") if folder_name else ""
            action_label = f"notify+folder:{folder_name}" if also_notify else (f"folder:{folder_name}" if folder_name else action)
            name = f"{match_type}-{s_val[:30]}-{action_label[:20]}"

            folder_sql = f"'{s_folder}'" if s_folder else "NULL"
            row_id = await execute_insert(
                f"INSERT INTO {_RULES()} "
                f"(rule_name, match_type, match_value, match_mode, action, folder_name, also_notify, source, notes) "
                f"VALUES ('{name}', '{match_type}', '{s_val}', 'contains', "
                f"'{action}', {folder_sql}, {1 if also_notify else 0}, 'user', '{s_notes}')"
            )
            return f"Rule #{row_id} created: {match_type} contains '{match_value}' → {action_label}"

        # !email rules disable N
        if rsub == "disable":
            try:
                rid = int(rparts[1]) if len(rparts) > 1 else 0
                if not rid:
                    return "Usage: !email rules disable <rule_id>"
                await execute_sql(
                    f"UPDATE {_RULES()} SET enabled = FALSE WHERE id = {rid}"
                )
                return f"Rule #{rid} disabled."
            except (ValueError, IndexError):
                return "Usage: !email rules disable <rule_id>"

        # !email rules enable N
        if rsub == "enable":
            try:
                rid = int(rparts[1]) if len(rparts) > 1 else 0
                await execute_sql(
                    f"UPDATE {_RULES()} SET enabled = TRUE WHERE id = {rid}"
                )
                return f"Rule #{rid} enabled."
            except (ValueError, IndexError):
                return "Usage: !email rules enable <rule_id>"

        # !email rules delete N
        if rsub == "delete":
            try:
                rid = int(rparts[1]) if len(rparts) > 1 else 0
                await execute_sql(f"DELETE FROM {_RULES()} WHERE id = {rid}")
                return f"Rule #{rid} deleted."
            except (ValueError, IndexError):
                return "Usage: !email rules delete <rule_id>"

        # !email rules stats
        if rsub == "stats":
            rows = await fetch_dicts(
                f"SELECT id, rule_name, action, hit_count, last_hit_at "
                f"FROM {_RULES()} WHERE hit_count > 0 ORDER BY hit_count DESC LIMIT 20"
            ) or []
            if not rows:
                return "No rule hits yet."
            lines = ["## Rule Hit Stats\n"]
            for r in rows:
                lines.append(
                    f"  #{r['id']} {r.get('rule_name', '')}: "
                    f"{r.get('hit_count', 0)} hits, last={r.get('last_hit_at', 'never')}"
                )
            return "\n".join(lines)

        return f"Unknown rules subcommand: {rsub}"

    return (
        "Usage: !email [scan|review|approve|override|stats|on|off|rules]\n"
        "  !email           — show recent triage\n"
        "  !email scan      — trigger immediate scan\n"
        "  !email review    — show unreviewed decisions\n"
        "  !email approve N — mark as correct\n"
        "  !email override N <action> — correct a decision\n"
        "  !email stats     — scan statistics\n"
        "  !email on/off    — enable/disable timer\n"
        "  !email rules     — manage rules\n"
    )


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class Plugin(BasePlugin):
    PLUGIN_NAME = "plugin_email_yahoo"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Yahoo email triage: scan, classify, log, notify"
    DEPENDENCIES = []
    ENV_VARS = ["YAHOO_EMAIL", "YAHOO_APP_PASSWORD"]

    def init(self, config: dict) -> bool:
        log.info("Email Yahoo triage plugin initialized")
        return True

    def shutdown(self) -> None:
        log.info("Email Yahoo triage plugin shutting down")

    def get_commands(self) -> Dict[str, Any]:
        return {"email": _cmd_email}

    def get_help(self) -> str:
        return (
            "\n**Email Triage (Yahoo)**\n"
            "  !email           — recent triage decisions\n"
            "  !email scan      — scan inbox now\n"
            "  !email review    — unreviewed decisions\n"
            "  !email rules     — manage classification rules\n"
            "  !email stats     — scan statistics\n"
            "  !email on/off    — enable/disable timer\n"
        )
