# Plugin: plugin_client_slack

Slack bidirectional client. The agent joins Slack via Socket Mode — no public endpoint required.

## What it provides

- Receives messages via Slack Socket Mode (WebSocket, no inbound port needed)
- Sends responses via Slack Web API
- Each Slack thread maps to a unique agent session
- Health check endpoint: `GET /slack/health` on port 8766
- All `!commands` work as in shell.py

## Session behavior

- Session ID format: `slack-<channel_id>-<thread_ts>`
- Each thread = isolated session with its own history and model
- Tool access controlled per-model via `llm_tools` in `llm-models.json`

## Prerequisites

1. Create a Slack app at api.slack.com
2. Enable Socket Mode
3. Add Bot Token Scopes: `chat:write`, `channels:history`, `app_mentions:read`
4. Install app to workspace
5. Copy Bot Token (`xoxb-...`) and App Token (`xapp-...`)

## Dependencies

```bash
pip install slack-sdk>=3.0 aiohttp>=3.0
```

## Environment variables

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

## Enable

```bash
python llmemctl.py enable plugin_client_slack
```
