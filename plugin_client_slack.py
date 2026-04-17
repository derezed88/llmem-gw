"""
Slack Client Interface Plugin for MCP Agent

Provides bidirectional Slack integration with asymmetric transport:
- Receives messages from Slack via Socket Mode (WebSocket, no public endpoint needed)
- Routes to LLM agent for processing
- Sends responses back to Slack via the Web API (chat.postMessage, authenticated via SLACK_BOT_TOKEN)
- Thread-aware conversations (Slack threads map to agent sessions)
- Full support for !commands like other clients
"""

import os
import re
import json
import asyncio
import time
from typing import List, Dict, Optional
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from plugin_loader import BasePlugin

# Slack SDK imports
try:
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.socket_mode.aiohttp import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.socket_mode.response import SocketModeResponse
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False

# Import agent infrastructure
from config import log
from state import sessions, get_queue, push_tok, push_done, active_tasks, cancel_active_task
from routes import process_request


class SlackClientPlugin(BasePlugin):
    """Slack bidirectional client interface plugin."""

    PLUGIN_NAME = "plugin_client_slack"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "client_interface"
    DESCRIPTION = "Slack bidirectional client with Socket Mode and webhook support"
    DEPENDENCIES = ["slack-sdk>=3.0", "aiohttp>=3.0"]
    ENV_VARS = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"]

    def __init__(self):
        self.enabled = False
        self.slack_port = 8766
        self.slack_host = "0.0.0.0"
        self.inter_turn_timeout = 30.0
        self.slack_client: Optional[AsyncWebClient] = None
        self.socket_client: Optional[SocketModeClient] = None

        # Map Slack thread_ts to agent session_id and channel
        # Format: {"thread_ts_or_channel_ts": {"session_id": "sess_xyz", "channel_id": "C123"}}
        self.slack_sessions: Dict[str, Dict[str, str]] = {}

        # One active consumer task per client_id — prevents queue bleed when
        # rapid messages arrive before the previous consumer has exited.
        self._consumer_tasks: Dict[str, asyncio.Task] = {}

        # Background task for Socket Mode listener
        self._socket_task: Optional[asyncio.Task] = None

    def init(self, config: dict) -> bool:
        """Initialize Slack client plugin."""
        if not SLACK_SDK_AVAILABLE:
            log.error("Slack SDK not available. Install with: pip install slack-sdk aiohttp")
            return False

        try:
            # Get configuration
            self.slack_port = config.get('slack_port', 8766)
            self.slack_host = config.get('slack_host', '0.0.0.0')
            self.inter_turn_timeout = float(
                config.get("inter_turn_timeout",
                os.getenv("SLACK_INTER_TURN_TIMEOUT", "30"))
            )

            # Get Slack credentials from environment
            bot_token = os.getenv("SLACK_BOT_TOKEN")
            app_token = os.getenv("SLACK_APP_TOKEN")

            # Optional: fixed channel for proactive interrupt notifications.
            # When set, _deliver_notification posts there as a new top-level
            # message (no thread_ts) instead of replying to the active session thread.
            self.notification_channel = os.getenv("SLACK_NOTIFICATION_CHANNEL", "")

            if not bot_token:
                log.error("SLACK_BOT_TOKEN not found in .env")
                return False

            if not app_token:
                log.error("SLACK_APP_TOKEN not found in .env")
                return False

            # Initialize Slack Web API client (for metadata, not used for sending)
            self.slack_client = AsyncWebClient(token=bot_token)

            # Initialize Socket Mode client for receiving events
            self.socket_client = SocketModeClient(
                app_token=app_token,
                web_client=self.slack_client
            )

            # Register Socket Mode event handlers
            self.socket_client.socket_mode_request_listeners.append(
                self._handle_socket_mode_request
            )

            self.enabled = True

            log.info("Slack client plugin initialized (Socket Mode)")
            log.info(f"  Bot token: {bot_token[:10]}...")
            log.info(f"  App token: {app_token[:10]}...")

            # Register notifier delivery hook — posts notifications directly to
            # Slack channels, bypassing the SSE queue (which has no persistent
            # consumer for Slack sessions).
            try:
                import notifier as _notifier
                _notifier.register_delivery_hook("slack-", self._deliver_notification)
            except Exception as e:
                log.warning(f"Slack: notifier hook registration failed: {e}")

            # Start Socket Mode listener in background
            self._socket_task = asyncio.create_task(self._run_socket_mode())

            return True

        except Exception as e:
            log.error(f"Slack client plugin init failed: {e}", exc_info=True)
            return False

    def shutdown(self) -> None:
        """Cleanup Slack client resources."""
        self.enabled = False

        # Stop Socket Mode listener
        if self._socket_task:
            self._socket_task.cancel()
            self._socket_task = None

        # Disconnect Socket Mode client
        if self.socket_client:
            try:
                asyncio.create_task(self.socket_client.close())
            except Exception as e:
                log.error(f"Error closing Socket Mode client: {e}")
            self.socket_client = None

        self.slack_client = None
        self.slack_sessions.clear()
        log.info("Slack client plugin shutdown")

    def get_routes(self) -> List[Route]:
        """Return Starlette routes for Slack client (health/status only - events via Socket Mode)."""
        return [
            Route("/slack/health", self._handle_health, methods=["GET"]),
            Route("/slack/status", self._handle_status, methods=["GET"]),
        ]

    def get_config(self) -> dict:
        """Return plugin configuration for server startup."""
        return {
            "port": self.slack_port,
            "host": self.slack_host,
            "name": "Slack client"
        }

    # =========================================================================
    # Socket Mode listener
    # =========================================================================

    async def _run_socket_mode(self) -> None:
        """Run Socket Mode client to listen for Slack events."""
        try:
            log.info("Starting Slack Socket Mode listener...")
            await self.socket_client.connect()
            log.info("Slack Socket Mode connected successfully!")
        except Exception as e:
            log.error(f"Socket Mode connection error: {e}", exc_info=True)

    async def _handle_socket_mode_request(
        self,
        client: SocketModeClient,
        request: SocketModeRequest
    ) -> None:
        """
        Handle incoming Socket Mode requests (events from Slack).

        This is called automatically by the Socket Mode client when events arrive.
        """
        try:
            # Acknowledge the request immediately
            response = SocketModeResponse(envelope_id=request.envelope_id)
            await client.send_socket_mode_response(response)

            # Process the event payload
            if request.type == "events_api":
                event = request.payload.get("event", {})
                await self._process_slack_event(event)
            elif request.type == "slash_commands":
                # Future: handle slash commands if needed
                log.debug("Received slash command (not yet implemented)")
            else:
                log.debug(f"Unhandled Socket Mode request type: {request.type}")

        except Exception as e:
            log.error(f"Error handling Socket Mode request: {e}", exc_info=True)

    # =========================================================================
    # Slack mrkdwn de-formatting
    # =========================================================================

    # Slack auto-formats URLs, mentions, and channels in event text payloads.
    # e.g. "bash new-node.sh" → "bash <http://new-node.sh|new-node.sh>"
    # (.sh is a valid TLD so Slack auto-links it)
    # This corrupts shell commands sent via !tmux exec and similar paths.
    _SLACK_LINK_RE = re.compile(
        r'<'
        r'(?:https?://[^|>]+)\|'   # URL before the pipe — discard
        r'([^>]+)'                  # display text after the pipe — keep
        r'>'
    )
    _SLACK_BARE_URL_RE = re.compile(
        r'<(https?://[^|>]+)>'      # bare URL with no pipe — keep URL
    )
    _SLACK_CHANNEL_RE = re.compile(
        r'<#([A-Z0-9]+)(?:\|([^>]+))?>'  # channel: <#C123|name> or <#C123>
    )

    @classmethod
    def _deformat_slack_text(cls, text: str) -> str:
        """Strip Slack mrkdwn auto-formatting from incoming message text.

        Converts:
          <http://new-node.sh|new-node.sh>  →  new-node.sh
          <https://example.com>              →  https://example.com
          <#C123ABC|general>                 →  #general
          <#C123ABC>                         →  #C123ABC
        User mentions (<@U123>) are left intact — handled separately.
        """
        # URL with display text: keep display text
        text = cls._SLACK_LINK_RE.sub(r'\1', text)
        # Bare URL (no pipe): keep URL itself
        text = cls._SLACK_BARE_URL_RE.sub(r'\1', text)
        # Channel references: use name if available
        def _channel_repl(m):
            return f"#{m.group(2)}" if m.group(2) else f"#{m.group(1)}"
        text = cls._SLACK_CHANNEL_RE.sub(_channel_repl, text)
        # Slack encodes &, <, > as HTML entities in event payloads
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        # Slack auto-converts straight quotes to curly/smart quotes — normalize back
        text = text.replace("\u2018", "'").replace("\u2019", "'")   # left/right single quote
        text = text.replace("\u201c", '"').replace("\u201d", '"')   # left/right double quote
        return text

    # =========================================================================
    # Event processing
    # =========================================================================

    async def _process_slack_event(self, event: dict) -> None:
        """Process individual Slack event (message, app_mention, etc.)."""
        event_type = event.get("type")

        # Ignore bot messages to prevent loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            log.debug("Ignoring bot message")
            return

        # Handle message events
        if event_type == "message":
            await self._handle_message_event(event)
        elif event_type == "app_mention":
            await self._handle_app_mention_event(event)
        else:
            log.debug(f"Ignoring event type: {event_type}")

    async def _handle_message_event(self, event: dict) -> None:
        """Handle message event from Slack."""
        channel_id = event.get("channel")
        user_id = event.get("user")
        raw_text = event.get("text", "").strip()
        thread_ts = event.get("thread_ts") or event.get("ts")  # Use thread or message ts
        message_ts = event.get("ts")

        if not raw_text or not channel_id:
            log.debug("Ignoring empty message or missing channel")
            return

        # Strip Slack auto-formatting (auto-linked URLs, channels, etc.)
        text = self._deformat_slack_text(raw_text)

        log.info(f"Slack message: channel={channel_id}, user={user_id}, thread={thread_ts}")
        log.debug(f"Message text (raw): {raw_text}")
        log.debug(f"Message text (clean): {text}")

        # Process the message through the agent
        await self._process_user_message(channel_id, thread_ts, user_id, text, message_ts)

    async def _handle_app_mention_event(self, event: dict) -> None:
        """Handle app_mention event (when bot is @mentioned)."""
        channel_id = event.get("channel")
        user_id = event.get("user")
        raw_text = event.get("text", "").strip()
        thread_ts = event.get("thread_ts") or event.get("ts")
        message_ts = event.get("ts")

        if not raw_text or not channel_id:
            return

        # Strip Slack auto-formatting first (auto-linked URLs, channels, etc.)
        text = self._deformat_slack_text(raw_text)

        # Remove the bot mention from text (e.g., "<@U12345> hello" -> "hello")
        # Slack mentions are in format <@U12345>
        words = text.split()
        cleaned_words = [w for w in words if not (w.startswith("<@") and w.endswith(">"))]
        cleaned_text = " ".join(cleaned_words).strip()

        log.info(f"Slack app_mention: channel={channel_id}, user={user_id}, thread={thread_ts}")
        log.debug(f"Cleaned text (raw): {raw_text}")
        log.debug(f"Cleaned text: {cleaned_text}")

        # Process through agent
        await self._process_user_message(channel_id, thread_ts, user_id, cleaned_text, message_ts)

    # =========================================================================
    # Agent integration
    # =========================================================================

    async def _process_user_message(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        text: str,
        message_ts: str
    ) -> None:
        """
        Process user message through the agent.

        Maps Slack thread to agent session and routes message.
        """
        # Create unique client_id for this Slack thread
        client_id = f"slack-{channel_id}-{thread_ts}"

        # Get or create session mapping
        if thread_ts not in self.slack_sessions:
            self.slack_sessions[thread_ts] = {
                "session_id": client_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "thread_ts": thread_ts
            }
            log.info(f"Created new Slack session: {client_id}")

        session_info = self.slack_sessions[thread_ts]

        # Cancel any previous consumer for this client_id before starting a new one.
        # Without this, rapid messages spawn multiple consumers that race on the same
        # queue — causing responses from conversation N to appear during conversation N+1.
        old_task = self._consumer_tasks.get(client_id)
        if old_task and not old_task.done():
            old_task.cancel()

        # Also cancel any in-flight process_request for this client and drain its queue,
        # mirroring what endpoint_submit does.  Without this the old agent task keeps
        # running after the consumer is cancelled and its tok/done items pile up in the
        # queue; the next consumer then picks them up as if they were responses to the
        # new message.  We await so the drain completes before the new consumer starts.
        await cancel_active_task(client_id)

        task = asyncio.create_task(self._consume_agent_responses(client_id, channel_id, thread_ts))
        self._consumer_tasks[client_id] = task

        # Submit message to agent (same as shell.py does via /submit)
        # This will trigger process_request which handles !commands and LLM
        payload = {
            "client_id": client_id,
            "text": text,
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts
        }

        # Process through agent and register the task so the next incoming message
        # can cancel it via cancel_active_task (same pattern as endpoint_submit).
        pr_task = asyncio.create_task(process_request(client_id, text, payload))
        active_tasks[client_id] = pr_task

    # Two-pass regexes to strip push_tok bracket status lines from agent output.
    #
    # Pass 1 — "owned body" tags: bracket lines that own the body lines immediately
    # following them.  Two sub-cases:
    #   a) Tag line ends with ":" — e.g. "[agent_call ◀] url:\nbody text"
    #   b) Tag line contains "◀" (close/result tag) — e.g. "[search ddgs ◀]\nresults…"
    # Both the tag line and its body are removed because the LLM always writes its
    # own summary of the result afterwards.
    _STATUS_OWNED_RE = re.compile(
        r'\[(?:agent_call|tool_call|llm_call'
        r'|db|search\s+\w+|sysprompt|sysinfo|drive)[^\]]*\]'
        r'[^\n]*(?::|◀)[^\n]*\n'
        r'(?:(?!\[)[^\n]+\n?)*',
        re.MULTILINE,
    )
    # Pass 2 — standalone bracket tag lines (▶ progress lines, bare ◀, ✗ errors,
    # [Max iterations], [RATE LIMITED], [REJECTED], [catcher], [context], etc.)
    _STATUS_STANDALONE_RE = re.compile(
        r'^\[(?:agent_call|tool_call|llm_call'
        r'|db|search\s+\w+|sysprompt|sysinfo|drive|catcher|context'
        r'|RATE\s+LIMITED|REJECTED|Max\s+iterations)[^\]]*\][^\n]*\n?',
        re.MULTILINE,
    )

    @classmethod
    def _filter_status_lines(cls, text: str) -> str:
        """Remove push_tok status bracket lines (and their owned bodies) from agent output."""
        filtered = cls._STATUS_OWNED_RE.sub('', text)
        filtered = cls._STATUS_STANDALONE_RE.sub('', filtered)
        # Collapse runs of blank lines that may be left behind
        filtered = re.sub(r'\n{3,}', '\n\n', filtered)
        return filtered.strip()

    async def _update_slack_message(
        self, channel_id: str, message_ts: str, text: str
    ) -> None:
        """Edit an existing Slack message in-place via chat.update."""
        if not self.slack_client:
            return
        try:
            await self.slack_client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text=text,
            )
        except Exception as e:
            log.warning(f"Slack chat.update failed: {e}")

    async def _consume_agent_responses(
        self,
        client_id: str,
        channel_id: str,
        thread_ts: str
    ) -> None:
        """
        Consume responses from agent queue and send to Slack.

        Similar to how shell.py consumes via SSE stream.

        Multi-turn behaviour: after each agent turn completes (done event) the
        accumulated text is posted to Slack immediately so the user sees progress.
        Accumulation then resets for the next turn.  The final turn's post is the
        last message the user sees — no silent 2-minute wait.

        Heartbeat: while waiting for the first token (e.g. during a long agent_call),
        a "_(working...)_" message is posted and edited in-place every 30 seconds so
        the user knows the system is alive.  The heartbeat stops the moment real
        content arrives; its final state is left in the thread as a breadcrumb.
        """
        if not self.slack_client:
            log.error("Slack client not initialized")
            return

        # Get the queue for this client
        queue = await get_queue(client_id)

        # Accumulate tokens for the current turn
        response_parts: List[str] = []
        turn_index = 0  # which agent turn we are on (0-based)

        # Multi-turn agents (tool call → LLM response) emit multiple "done" signals.
        # After each "done" we post the turn's output immediately, then wait a short
        # grace period for the next turn to start.
        # If nothing arrives within the grace period, the conversation is truly finished.
        FIRST_TIMEOUT = 300.0   # max wait for first token; raised to cover 5-turn agent_call
        HEARTBEAT_INTERVAL = 30.0  # how often to update the "working" message
        INTER_TURN_TIMEOUT = self.inter_turn_timeout  # configurable via plugins-enabled.json

        timeout = FIRST_TIMEOUT
        received_done = False

        # Heartbeat state
        heartbeat_ts: Optional[str] = None   # ts of the posted "working" message
        heartbeat_task: Optional[asyncio.Task] = None
        start_time = time.monotonic()
        first_token_received = False

        async def _start_heartbeat() -> None:
            """Post an initial 'working' message and then edit it every 30s.

            Waits HEARTBEAT_INTERVAL before posting so fast responses never
            see the heartbeat at all — it only appears during long waits.
            """
            nonlocal heartbeat_ts
            try:
                # Initial delay — if content arrives quickly, we never post
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                resp = await self.slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="_(working…)_",
                )
                heartbeat_ts = resp.get("ts")
                while True:
                    await asyncio.sleep(HEARTBEAT_INTERVAL)
                    elapsed = int(time.monotonic() - start_time)
                    mins, secs = divmod(elapsed, 60)
                    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                    await _update_heartbeat(f"_(working… {elapsed_str})_")
            except asyncio.CancelledError:
                pass  # normal — cancelled when real content arrives
            except Exception as e:
                log.warning(f"Heartbeat error: {e}")

        async def _update_heartbeat(text: str) -> None:
            if heartbeat_ts:
                await _update_slack_message(channel_id, heartbeat_ts, text)

        async def _stop_heartbeat(final_text: Optional[str] = None) -> None:
            """Cancel the heartbeat task and optionally set a final message."""
            nonlocal heartbeat_task
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            heartbeat_task = None
            if final_text and heartbeat_ts:
                await _update_heartbeat(final_text)

        # Local alias so inner functions can call instance method without self
        async def _update_slack_message(ch: str, ts: str, text: str) -> None:
            await self._update_slack_message(ch, ts, text)

        # ts of the first turn's Slack message — kept so we can retroactively
        # prepend "_(turn 1)_" when a second turn arrives, without labeling
        # single-turn responses at all.
        first_turn_ts: Optional[str] = None
        first_turn_text: Optional[str] = None

        async def _flush_turn(label: str) -> None:
            """Post the current turn's accumulated text to Slack and reset."""
            nonlocal response_parts, turn_index, first_turn_ts, first_turn_text
            if response_parts:
                # push_tok encodes newlines as \\n literals; restore them before
                # filtering so the line-anchored regexes work correctly.
                turn_text = self._filter_status_lines(
                    "".join(response_parts).replace("\\n", "\n")
                )
                if turn_text:
                    if turn_index == 0:
                        # First turn: post without a label for now.  If a second
                        # turn arrives we'll retroactively edit this message to add
                        # "_(turn 1)_" so single-turn responses stay clean.
                        first_turn_ts = await self._send_slack_message(channel_id, thread_ts, turn_text)
                        first_turn_text = turn_text
                    else:
                        if turn_index == 1 and first_turn_ts and first_turn_text:
                            # Second turn arriving — retroactively label turn 1
                            await _update_slack_message(
                                channel_id, first_turn_ts,
                                f"_(turn 1)_\n{first_turn_text}"
                            )
                        # Post this turn with its label
                        await self._send_slack_message(
                            channel_id, thread_ts,
                            f"_(turn {turn_index + 1})_\n{turn_text}"
                        )
                    log.info(f"Slack: posted {label} for {client_id} (turn {turn_index + 1})")
                    turn_index += 1  # only advance on visible output
            response_parts = []

        cancelled = False
        try:
            # Start heartbeat immediately — covers the initial wait before any token
            # arrives (e.g. during a long blocking agent_call sequence).
            heartbeat_task = asyncio.create_task(_start_heartbeat())

            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                    received_done = False  # reset: activity means we're still in a turn
                except asyncio.TimeoutError:
                    if received_done:
                        # Grace period expired after a "done" — conversation finished.
                        # The turn was already flushed on the done event; nothing left to do.
                        await _stop_heartbeat()
                        break
                    else:
                        # No response at all within FIRST_TIMEOUT
                        log.warning(f"Slack consumer timeout waiting for response from {client_id}")
                        await _stop_heartbeat("_(timed out)_")
                        # Flush whatever we have (may be empty)
                        await _flush_turn("timeout flush")
                        break

                item_type = item.get("t")

                if item_type == "tok":
                    # First real token — stop heartbeat before appending content
                    if not first_token_received:
                        first_token_received = True
                        await _stop_heartbeat("_(done working)_")
                    # Token/text data
                    response_parts.append(item["d"])
                    timeout = FIRST_TIMEOUT  # reset to long timeout while active

                elif item_type == "done":
                    # One turn complete — post immediately, then wait for next turn.
                    # If the agent task is still running, use FIRST_TIMEOUT so a slow
                    # LLM thinking between turns (e.g. after a tool call) doesn't cause
                    # the consumer to exit before the final response arrives.
                    await _stop_heartbeat()
                    await _flush_turn("turn complete")
                    received_done = True
                    agent_task = active_tasks.get(client_id)
                    timeout = FIRST_TIMEOUT if (agent_task and not agent_task.done()) else INTER_TURN_TIMEOUT
                    # Start heartbeat for next inter-turn wait
                    first_token_received = False
                    heartbeat_ts = None
                    heartbeat_task = asyncio.create_task(_start_heartbeat())

                elif item_type == "err":
                    # Error occurred — append and flush immediately
                    await _stop_heartbeat("_(error)_")
                    error_msg = item.get("d", "Unknown error")
                    response_parts.append(f"\n\n⚠️ Error: {error_msg}")
                    await _flush_turn("error")
                    break

                elif item_type == "notif":
                    # Async notification — post directly, don't interrupt current turn
                    notif_text = item.get("d", "").replace("\\n", "\n")
                    await self._send_slack_message(channel_id, thread_ts, notif_text)

                elif item_type == "progress":
                    # Progress update — update the heartbeat message in-place
                    # so the user sees "Still working… <stage>" like voice does.
                    progress_text = item.get("d", "Still working…")
                    if heartbeat_ts:
                        await _update_heartbeat(f"_{progress_text}_")
                    elif not first_token_received:
                        # No heartbeat message yet — post one now
                        try:
                            resp = await self.slack_client.chat_postMessage(
                                channel=channel_id,
                                thread_ts=thread_ts,
                                text=f"_{progress_text}_",
                            )
                            heartbeat_ts = resp.get("ts")
                        except Exception as e:
                            log.warning(f"Progress post error: {e}")

                elif item_type == "flush":
                    # Intermediate flush between tool iterations — ignore on Slack
                    # (shell.py uses this to clear its reply buffer mid-turn)
                    pass

                elif item_type == "gate":
                    # Gate request - inform user that approval is needed
                    await _stop_heartbeat()
                    gate_data = item.get("d", {})
                    tool_name = gate_data.get("tool_name", "unknown")
                    gate_notice = (
                        f"🔒 Gate approval required for tool: `{tool_name}`\n"
                        f"(Approval must be provided via shell.py client)"
                    )
                    await self._send_slack_message(channel_id, thread_ts, gate_notice)
                    timeout = FIRST_TIMEOUT  # gate may take a while
                    received_done = False
                    # Continue listening for more responses

        except asyncio.CancelledError:
            # Cancelled by a new incoming message — do NOT drain the queue.
            # The new consumer will pick up whatever process_request already
            # put there for the next message.
            cancelled = True
            await _stop_heartbeat()
            raise
        except Exception as e:
            log.error(f"Error consuming agent responses for {client_id}: {e}", exc_info=True)
            await _stop_heartbeat("_(error)_")
        finally:
            # Drain stale queue items on normal or error exit — but NOT on cancel,
            # because the cancelling message's process_request may have already
            # put its tok+done into the queue for the new consumer to pick up.
            if not cancelled:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

    @staticmethod
    def _close_open_backticks(text: str) -> str:
        """
        Ensure backtick spans are closed at the end of a message chunk.

        Slack truncates rendering when a backtick code span is left open.
        Handles triple-backtick (```) code blocks and single-backtick spans.
        """
        # Check for open triple-backtick code block first
        triple_count = text.count('```')
        if triple_count % 2 != 0:
            return text + '\n```'

        # Check for open single-backtick span (not inside a triple block).
        # Walk the string tracking state so embedded backticks don't count.
        in_triple = False
        in_single = False
        i = 0
        while i < len(text):
            if text[i:i+3] == '```':
                in_triple = not in_triple
                i += 3
                continue
            if not in_triple and text[i] == '`':
                in_single = not in_single
            i += 1

        if in_single:
            return text + '`'
        return text

    async def _deliver_notification(self, client_id: str, msg: str) -> bool:
        """Notifier delivery hook — post notification directly to Slack.

        client_id format: slack-{channel_id}-{thread_ts}

        If SLACK_NOTIFICATION_CHANNEL is configured, posts there as a new
        top-level message (no thread_ts) so notifications are always visible
        regardless of which session thread is currently active.

        Returns True if the message was posted successfully.
        """
        if not self.slack_client:
            return False
        if self.notification_channel:
            # Post to dedicated notification channel — not threaded into
            # whatever session happened to be last active.
            ts = await self._send_slack_message(self.notification_channel, "", msg)
            return ts is not None
        # Fallback: thread into the active session (legacy behavior)
        parts = client_id.split("-", 2)
        if len(parts) < 3:
            log.warning(f"Slack notifier: can't parse client_id: {client_id}")
            return False
        channel_id = parts[1]
        thread_ts = parts[2]
        ts = await self._send_slack_message(channel_id, thread_ts, msg)
        return ts is not None

    async def _send_slack_message(self, channel_id: str, thread_ts: str, text: str) -> Optional[str]:
        """
        Send message to Slack channel/thread via Web API (chat.postMessage).

        Authenticated via SLACK_BOT_TOKEN. Supports threaded replies and
        automatic chunking for messages exceeding Slack's ~4000 char limit.

        Returns the Slack message ts of the first posted chunk, or None on error.
        The ts can be used with _update_slack_message() to retroactively edit the message.
        """
        if not self.slack_client:
            log.error("Slack client not initialized")
            return None

        try:
            # Clean up text for Slack rendering
            # Slack uses actual newlines, not \n literals
            cleaned_text = text.replace('\\n', '\n')

            # Slack has a ~4000 character limit for messages
            # Split into chunks if needed
            max_chunk_size = 3500
            first_ts: Optional[str] = None

            # Only include thread_ts when non-empty — omitting it posts as a
            # new top-level message (used for notification channel delivery).
            thread_kwargs = {"thread_ts": thread_ts} if thread_ts else {}

            if len(cleaned_text) <= max_chunk_size:
                # Close any unclosed backtick spans before sending
                cleaned_text = self._close_open_backticks(cleaned_text)
                resp = await self.slack_client.chat_postMessage(
                    channel=channel_id,
                    text=cleaned_text,
                    **thread_kwargs
                )
                first_ts = resp.get("ts")
                log.info(f"Sent Slack message to {channel_id}/{thread_ts or 'top-level'} ({len(cleaned_text)} chars)")
            else:
                # Split into chunks
                chunks = [cleaned_text[i:i+max_chunk_size] for i in range(0, len(cleaned_text), max_chunk_size)]
                log.info(f"Splitting message into {len(chunks)} chunks")

                for i, chunk in enumerate(chunks):
                    prefix = f"(Part {i+1}/{len(chunks)})\n" if len(chunks) > 1 else ""
                    # Close any unclosed backtick spans at chunk boundary
                    chunk = self._close_open_backticks(chunk)
                    resp = await self.slack_client.chat_postMessage(
                        channel=channel_id,
                        text=prefix + chunk,
                        **thread_kwargs
                    )
                    if i == 0:
                        first_ts = resp.get("ts")
                    # Small delay between chunks to avoid rate limits
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.5)

            return first_ts

        except Exception as e:
            log.error(f"Error sending Slack message: {e}", exc_info=True)
            return None

    # =========================================================================
    # Status endpoints
    # =========================================================================

    async def _handle_health(self, request: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse({
            "status": "healthy",
            "plugin": self.PLUGIN_NAME,
            "version": self.PLUGIN_VERSION,
            "enabled": self.enabled,
            "active_sessions": len(self.slack_sessions)
        })

    async def _handle_status(self, request: Request) -> JSONResponse:
        """Status endpoint showing active Slack sessions."""
        sessions_list = [
            {
                "thread_ts": thread_ts,
                "session_id": info["session_id"],
                "channel_id": info["channel_id"],
                "user_id": info.get("user_id", "unknown")
            }
            for thread_ts, info in self.slack_sessions.items()
        ]

        return JSONResponse({
            "plugin": self.PLUGIN_NAME,
            "active_sessions": len(self.slack_sessions),
            "sessions": sessions_list
        })
