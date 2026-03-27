# iOS / macOS SMS Proxy

Bridge iPhone SMS and iMessage to llmem-gw via the macOS Messages app. Incoming texts are forwarded to all matching AI sessions in real time; outbound replies are sent back through Messages.app using AppleScript.

## Architecture

```
iPhone (SMS/iMessage)
    │  iCloud sync
    ▼
macOS Messages.app ──► ~/Library/Messages/chat.db
    │                         │
    │  AppleScript send       │  SQLite poll (read-only copy)
    │◄────────────────────────┤
    │                         ▼
    │                   sms_proxy.py  (macOS client)
    │                         │
    │                         │  HTTP
    │                         ▼
    │                   llmem-gw  (Linux server)
    │                   ├─ POST /sms/inbound   ← new SMS arrives
    │                   ├─ GET  /sms/outbound  → proxy polls for replies
    │                   ├─ POST /sms/ack       ← proxy confirms delivery
    │                   ├─ GET  /sms/notifications → webfe polls
    │                   └─ GET  /sms/health
    │                         │
    │                         ▼
    │                   notifier + session push
    │                   (samaritan-voice, samaritan-reasoning, etc.)
    │
    └──── outbound reply sent via Messages.app
```

### Components

| Component | Runs on | File |
|---|---|---|
| **sms_proxy.py** | macOS (same Apple ID as iPhone) | `sms_proxy.py` |
| **plugin_sms_proxy** | llmem-gw server (Linux) | `plugin_sms_proxy.py` |
| **Config** | macOS client | `sms_proxy_config.json` |

## Prerequisites

### macOS client machine
- macOS with Messages.app signed into the same Apple ID as the iPhone
- **Text Message Forwarding** enabled on iPhone (Settings > Messages > Text Message Forwarding > select Mac)
- **Full Disk Access** granted to Terminal (or whichever app runs Python) in System Settings > Privacy & Security > Full Disk Access
- Python 3.11+ with `aiohttp` installed

### llmem-gw server
- `plugin_sms_proxy` listed in `plugins-enabled.json`
- Network connectivity from macOS client to the llmem-gw host

## Configuration

### Client config (`sms_proxy_config.json`)

```json
{
    "llmem_gw_url": "http://192.168.10.111:8767",
    "poll_interval": 2,
    "outbound_poll_interval": 3,
    "allowed_numbers": [],
    "blocked_numbers": [],
    "max_message_age": 60
}
```

| Key | Default | Description |
|---|---|---|
| `llmem_gw_url` | `http://192.168.10.111:8767` | llmem-gw base URL |
| `poll_interval` | `2` | Seconds between Messages DB polls for new incoming SMS |
| `outbound_poll_interval` | `3` | Seconds between polls for outbound replies |
| `allowed_numbers` | `[]` | Whitelist of phone numbers (empty = allow all) |
| `blocked_numbers` | `[]` | Blacklist of phone numbers |
| `max_message_age` | `60` | Ignore messages older than N seconds (prevents replaying old texts on startup) |

### Server config (`plugins-enabled.json`)

```json
{
    "plugin_sms_proxy": {
        "enabled": true,
        "relay_enabled": true,
        "notify_models": ["samaritan-voice*", "samaritan-reasoning"]
    }
}
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Load the plugin |
| `relay_enabled` | `true` | Accept inbound SMS (can be toggled at runtime with `!sms enable/disable`) |
| `notify_models` | `[]` | Model name patterns for auto-push notifications. Trailing `*` is a wildcard. |

### Contact name resolution

The plugin looks up sender phone numbers in the `person` MySQL table (`phone` column, added by migration `013_person_phone.sql`). If a match is found, the notification uses the person's nickname or full name instead of the raw number.

## Running the proxy

On the macOS machine:

```bash
# Install dependency
pip install aiohttp

# Copy files from llmem-gw
# sms_proxy.py, sms_proxy_config.json

# Run (foreground)
python sms_proxy.py

# Run with custom config path
python sms_proxy.py --config /path/to/sms_proxy_config.json

# Verbose logging
python sms_proxy.py -v
```

On startup the proxy:
1. Checks plugin health via `GET /sms/health`
2. Reads the current max ROWID from `chat.db` (skips all existing messages)
3. Starts two concurrent loops: inbound poll + outbound poll

## Commands

From any llmem-gw session:

| Command | Description |
|---|---|
| `!sms` | Show status and recent messages |
| `!sms reply <phone> <msg>` | Queue a reply to be sent via the macOS proxy |
| `!sms history [N]` | Show last N messages (default 10) |
| `!sms enable` | Enable SMS relay at runtime |
| `!sms disable` | Disable SMS relay at runtime |

## Testing

### 1. Verify plugin is loaded

```bash
curl http://localhost:8767/sms/health
```

Expected:
```json
{"status": "ok", "enabled": true, "proxy_connected": false, "inbox_count": 0, "outbound_pending": 0}
```

`proxy_connected` becomes `true` once the macOS client starts polling (heartbeat via `/sms/outbound`).

### 2. Simulate an inbound SMS (no macOS needed)

```bash
curl -X POST http://localhost:8767/sms/inbound \
  -H "Content-Type: application/json" \
  -d '{"phone": "+14155551234", "text": "Hello from test"}'
```

Expected:
```json
{"status": "ok", "id": 1, "auto_notified": 0}
```

`auto_notified` > 0 if any active sessions match `notify_models`.

### 3. Queue an outbound reply

From a session: `!sms reply +14155551234 Got your message`

Or via curl:
```bash
# Check the outbound queue (this is what the macOS proxy polls)
curl http://localhost:8767/sms/outbound
```

### 4. End-to-end test with macOS proxy

1. Start llmem-gw with the plugin enabled
2. On the Mac, run `python sms_proxy.py -v`
3. Send a text to the iPhone from another phone
4. Watch the proxy log for `New SMS from ...` and `Inbound SMS delivered to plugin`
5. In an active voice/reasoning session, the SMS notification should appear
6. Reply with `!sms reply <phone> <message>` and confirm delivery in the proxy log

### 5. Verify Full Disk Access

If the proxy logs `DB read error` or `Permission denied`, Full Disk Access is not granted. Go to System Settings > Privacy & Security > Full Disk Access and add your terminal application.

## How it works

### Inbound flow
1. `sms_proxy.py` copies `~/Library/Messages/chat.db` to `/tmp/` (avoids locking the live DB)
2. Queries for messages with ROWID > last seen, `is_from_me = 0`, non-empty text
3. Filters by allowed/blocked numbers and max age
4. POSTs each message to `POST /sms/inbound`
5. Server-side plugin fires `sms_received` notifier event and pushes directly to all sessions matching `notify_models`

### Outbound flow
1. User runs `!sms reply <phone> <message>` in any session
2. Plugin queues the message in memory
3. macOS proxy polls `GET /sms/outbound` every few seconds
4. Proxy sends via AppleScript (`tell application "Messages"` > iMessage, with SMS fallback)
5. On success, proxy calls `POST /sms/ack` with the message IDs to clear the queue

### Notification delivery
- **Auto-push**: sessions whose model matches `notify_models` patterns receive SMS notifications directly via `push_notif()`
- **Notifier events**: `sms_received` event fires for any explicitly subscribed sessions
- **Polling**: webfe clients can poll `GET /sms/notifications?client_id=<id>` for buffered notifications
