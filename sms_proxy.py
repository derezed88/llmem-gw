#!/usr/bin/env python3
"""
macOS SMS Proxy Client for llmem-gw

Monitors the macOS Messages database for incoming SMS/iMessage,
forwards them to the llmem-gw plugin_sms_proxy endpoint (which fires
notifier events to all subscribed sessions), and polls for outbound
reply messages queued via !sms reply commands.

Requirements:
  - macOS with Messages app signed into iCloud (same Apple ID as iPhone)
  - Full Disk Access granted to Terminal/Python in System Settings > Privacy
  - pip install aiohttp

Usage:
  python sms_proxy.py [--config sms_proxy_config.json]

Config (sms_proxy_config.json):
  {
    "llmem_gw_url": "http://192.168.10.111:8767",
    "poll_interval": 2,
    "outbound_poll_interval": 3,
    "allowed_numbers": [],
    "blocked_numbers": [],
    "max_message_age": 60
  }
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sms_proxy")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "llmem_gw_url": "http://192.168.10.111:8767",
    "poll_interval": 2,            # seconds between Messages DB polls
    "allowed_numbers": [],         # empty = allow all
    "blocked_numbers": [],
    "max_message_age": 60,         # ignore messages older than N seconds
}

MESSAGES_DB = os.path.expanduser("~/Library/Messages/chat.db")


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
        log.info(f"Config loaded from {path}")
    else:
        log.info("Using default config (no config file found)")
    return cfg


# ---------------------------------------------------------------------------
# Messages DB reader (read-only copy to avoid locking)
# ---------------------------------------------------------------------------

def _copy_db(dest: str) -> str:
    """Copy chat.db to a temp location to avoid locking the live DB."""
    shutil.copy2(MESSAGES_DB, dest)
    for ext in ("-wal", "-shm"):
        src = MESSAGES_DB + ext
        if os.path.exists(src):
            shutil.copy2(src, dest + ext)
    return dest


def get_recent_messages(since_rowid: int, db_copy_path: str) -> list[dict]:
    """Read messages newer than since_rowid from a snapshot of chat.db."""
    _copy_db(db_copy_path)
    conn = sqlite3.connect(db_copy_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT
                m.ROWID as rowid,
                m.date as date,
                m.text as text,
                m.is_from_me as is_from_me,
                h.id as phone,
                h.service as service
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.ROWID > ?
              AND m.is_from_me = 0
              AND m.text IS NOT NULL
              AND m.text != ''
            ORDER BY m.ROWID ASC
        """, (since_rowid,)).fetchall()
        return [
            {
                "rowid": r["rowid"],
                "date": r["date"],
                "text": r["text"],
                "phone": r["phone"] or "unknown",
                "service": r["service"] or "SMS",
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_max_rowid(db_copy_path: str) -> int:
    """Get the current maximum ROWID from the messages DB."""
    _copy_db(db_copy_path)
    conn = sqlite3.connect(db_copy_path, timeout=5)
    try:
        row = conn.execute("SELECT MAX(ROWID) FROM message").fetchone()
        return row[0] or 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# macOS date conversion
# ---------------------------------------------------------------------------

APPLE_EPOCH_OFFSET = 978307200  # 2001-01-01 00:00:00 UTC

def apple_date_to_unix(apple_date: int) -> float:
    """Convert Apple's nanosecond timestamp to Unix epoch seconds."""
    if apple_date is None:
        return 0.0
    if apple_date > 1_000_000_000_000:
        return (apple_date / 1_000_000_000) + APPLE_EPOCH_OFFSET
    return apple_date + APPLE_EPOCH_OFFSET


# ---------------------------------------------------------------------------
# AppleScript SMS sender
# ---------------------------------------------------------------------------

def send_imessage(phone: str, text: str) -> bool:
    """Send a message via macOS Messages.app using AppleScript."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{phone}" of targetService
        send "{escaped}" to targetBuddy
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            log.info(f"Sent reply to {phone} ({len(text)} chars)")
            return True
        else:
            log.error(f"AppleScript send failed: {result.stderr.strip()}")
            return _send_sms_fallback(phone, text)
    except subprocess.TimeoutExpired:
        log.error("AppleScript send timed out")
        return False
    except Exception as e:
        log.error(f"AppleScript send error: {e}")
        return False


def _send_sms_fallback(phone: str, text: str) -> bool:
    """Fallback: try sending via SMS service if iMessage fails."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = SMS
        set targetBuddy to participant "{phone}" of targetService
        send "{escaped}" to targetBuddy
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            log.info(f"Sent reply via SMS fallback to {phone}")
            return True
        log.error(f"SMS fallback also failed: {result.stderr.strip()}")
        return False
    except Exception as e:
        log.error(f"SMS fallback error: {e}")
        return False


# ---------------------------------------------------------------------------
# llmem-gw SMS plugin client
# ---------------------------------------------------------------------------

try:
    import aiohttp
except ImportError:
    log.error("aiohttp required: pip install aiohttp")
    sys.exit(1)


class SmsPluginClient:
    """Async client for llmem-gw /sms/* plugin endpoints."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def post_inbound(self, phone: str, text: str) -> bool:
        """Send an incoming SMS to the plugin."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/sms/inbound",
                json={"phone": phone, "text": text},
            ) as resp:
                if resp.status == 200:
                    log.info(f"Inbound SMS delivered to plugin ({phone})")
                    return True
                elif resp.status == 503:
                    log.warning("SMS relay is disabled on server")
                    return False
                else:
                    log.error(f"Inbound POST failed: {resp.status}")
                    return False
        except Exception as e:
            log.error(f"Inbound POST error: {e}")
            return False

    async def ack_outbound(self, ids: list[int]) -> bool:
        """Acknowledge sent outbound messages."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/sms/ack",
                json={"ids": ids},
            ) as resp:
                return resp.status == 200
        except Exception as e:
            log.error(f"Outbound ACK error: {e}")
            return False

    async def health(self) -> dict | None:
        """Check plugin health."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/sms/health") as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Core proxy
# ---------------------------------------------------------------------------

class SmsProxy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = SmsPluginClient(cfg["llmem_gw_url"])
        self.db_copy_path = "/tmp/sms_proxy_chat.db"
        self.last_rowid = 0

    def _is_allowed(self, phone: str) -> bool:
        if phone in self.cfg["blocked_numbers"]:
            return False
        if self.cfg["allowed_numbers"]:
            return phone in self.cfg["allowed_numbers"]
        return True

    def _is_recent(self, apple_date: int) -> bool:
        msg_time = apple_date_to_unix(apple_date)
        age = time.time() - msg_time
        return age < self.cfg["max_message_age"]

    async def _poll_inbound(self):
        """Poll Messages DB for new incoming SMS and forward to plugin."""
        while True:
            try:
                messages = get_recent_messages(self.last_rowid, self.db_copy_path)
                for msg in messages:
                    phone = msg["phone"]
                    text = msg["text"]

                    if not self._is_allowed(phone):
                        log.debug(f"Blocked message from {phone}")
                    elif not self._is_recent(msg["date"]):
                        log.debug(f"Skipping old message (rowid {msg['rowid']})")
                    else:
                        log.info(f"New SMS from {phone}: {text[:80]}...")
                        await self.client.post_inbound(phone, text)
                    self.last_rowid = max(self.last_rowid, msg["rowid"])

            except sqlite3.OperationalError as e:
                log.warning(f"DB read error (will retry): {e}")
            except Exception as e:
                log.error(f"Inbound poll error: {e}", exc_info=True)

            await asyncio.sleep(self.cfg["poll_interval"])

    async def _stream_outbound(self):
        """Stream outbound replies via SSE and send via AppleScript.

        Holds a persistent connection to llmem-gw /sms/outbound/stream.
        Server pushes messages as SSE events; we send immediately and ACK.
        Reconnects with exponential backoff on disconnect.
        Connection direction is always macOS → llmem-gw (NAT/firewall safe).
        """
        stream_url = f"{self.cfg['llmem_gw_url']}/sms/outbound/stream"
        backoff = 5

        while True:
            try:
                log.info("SMS outbound: connecting to SSE stream")
                session = await self.client._get_session()
                async with session.get(
                    stream_url,
                    headers={"Accept": "text/event-stream"},
                    timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10),
                ) as resp:
                    if resp.status != 200:
                        log.error(f"SMS outbound: SSE rejected HTTP {resp.status}, retry in {backoff}s")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue

                    log.info("SMS outbound: SSE stream connected")
                    backoff = 5  # reset on successful connect

                    while True:
                        line_bytes = await resp.content.readline()
                        if not line_bytes:
                            log.warning("SMS outbound: SSE stream closed by server")
                            break

                        line = line_bytes.decode("utf-8").rstrip("\r\n")

                        if not line or line.startswith(":"):
                            continue  # blank line (event boundary) or heartbeat comment

                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                msg = json.loads(data_str)
                            except json.JSONDecodeError:
                                log.warning(f"SMS outbound: bad JSON in SSE event: {data_str}")
                                continue

                            phone = msg.get("phone", "")
                            text = msg.get("text", "")
                            msg_id = msg.get("id")
                            if not phone or not text:
                                continue

                            log.info(f"Outbound reply → {phone}: {text[:80]}...")
                            if send_imessage(phone, text):
                                await self.client.ack_outbound([msg_id])

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"SMS outbound SSE error: {e}", exc_info=True)

            log.info(f"SMS outbound: reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def run(self):
        """Main entry point — runs inbound and outbound loops concurrently."""
        log.info("SMS Proxy starting")
        log.info(f"  llmem-gw: {self.cfg['llmem_gw_url']}")
        log.info(f"  Inbound poll: {self.cfg['poll_interval']}s")
        log.info(f"  Outbound: SSE stream (event-driven)")
        log.info(f"  Messages DB: {MESSAGES_DB}")

        if not os.path.exists(MESSAGES_DB):
            log.error(f"Messages database not found: {MESSAGES_DB}")
            log.error("This script must run on macOS with Messages app configured.")
            return

        # Check plugin health
        health = await self.client.health()
        if health:
            log.info(f"  Plugin health: {health}")
        else:
            log.warning("  Plugin not reachable — will retry on first message")

        # Start from current max rowid
        self.last_rowid = get_max_rowid(self.db_copy_path)
        log.info(f"Starting from ROWID {self.last_rowid}")

        try:
            await asyncio.gather(
                self._poll_inbound(),
                self._stream_outbound(),
            )
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            await self.client.close()
            for ext in ("", "-wal", "-shm"):
                p = self.db_copy_path + ext
                if os.path.exists(p):
                    os.remove(p)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="macOS SMS Proxy for llmem-gw")
    parser.add_argument("--config", "-c", default="sms_proxy_config.json",
                        help="Path to config JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load_config(args.config)
    proxy = SmsProxy(cfg)
    asyncio.run(proxy.run())


if __name__ == "__main__":
    main()
