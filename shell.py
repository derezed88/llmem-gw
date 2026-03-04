"""
__shell__.py - AIOps Console Client  (Stage 4: Google Search + Buffer Scrolling)

Input model:
  The input area is a single flat string that visually wraps across
  `input_lines` screen rows.  There are no "logical rows" — the buffer
  is just text; the display chunks it by (terminal_width - 2) characters
  and always scrolls to keep the cursor visible.

Key bindings — input:
  Enter / F5 / Ctrl+G  — submit immediately
  Backspace / Delete   — delete char before cursor
  Left / Right         — move cursor one character
  Option+Left / Right  — move cursor one word (bash-style)
  Up / Down            — move cursor one visual row up/down
  Home / Ctrl+A        — jump to start
  End  / Ctrl+E        — jump to end

Key bindings — output buffer scrolling:
  PgUp  / Ctrl+B       — scroll output up one page
  PgDn  / Ctrl+F       — scroll output down one page
  Ctrl+End             — jump to bottom (latest output)
  Mouse wheel up/down  — scroll output 3 lines (if mouse capture is ON)

Special client-side commands:
  !input_lines <n>              — resize input area (1-20 rows)
  !mouse [on|off]               — toggle mouse capture (OFF allows text selection)
  !session                      — list all active sessions (shows current)
  !session <ID> attach          — switch to a different session (immediate)
  !session <ID> delete          — delete a session from server
  !exit / !quit                 — exit the shell

Server-side commands (forwarded):
  !model                        — list available models
  !model <key>                  — switch active LLM
  !reset                        — clear conversation history
  !db <sql>                     — run SQL directly (no LLM)
  !help                         — full help
"""

import asyncio
import curses
import json
import locale
import os
import re
import sys
import uuid
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_HOST         = os.getenv("AISVC_HOST", "http://127.0.0.1:8765")
SUBMIT_URL          = f"{SERVER_HOST}/submit"
STREAM_URL          = f"{SERVER_HOST}/stream"
DEFAULT_INPUT_LINES = 3
BORDER_CHAR         = "─"

# ---------------------------------------------------------------------------
# Session Persistence — per-tty so each terminal window has its own session
# ---------------------------------------------------------------------------

def _session_file_for_tty() -> str:
    """
    Return a session-file path that is unique to the current terminal device.
    Uses /dev/pts/N (or /dev/ttyN) sanitised into a filename so that two
    shell.py instances running in different terminal windows never share an ID.
    Falls back to a generic name when stdin is not a real tty (e.g. piped).
    """
    try:
        tty = os.ttyname(sys.stdin.fileno())          # e.g. "/dev/pts/3"
        safe = tty.lstrip("/").replace("/", "_")       # "dev_pts_3"
        return f".aiops_session_{safe}"
    except OSError:
        return ".aiops_session_id"                     # fallback (piped/non-tty)

SESSION_FILE = _session_file_for_tty()

def load_or_create_client_id() -> str:
    """Load CLIENT_ID from the tty-specific file, or create a new one."""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                client_id = f.read().strip()
                if client_id:
                    return client_id
        except Exception:
            pass
    client_id = str(uuid.uuid4())
    save_client_id(client_id)
    return client_id

def save_client_id(client_id: str):
    """Save CLIENT_ID to the tty-specific session file."""
    with open(SESSION_FILE, 'w') as f:
        f.write(client_id)

CLIENT_ID = load_or_create_client_id()

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.output_lines: list[str] = []

        # ---- flat input model ----
        self.input_text: str       = ""    # entire input as one string
        self.input_cursor_pos: int = 0     # byte offset into input_text
        self.input_lines: int      = DEFAULT_INPUT_LINES  # visual rows available

        self.status: str           = "Ready"
        self.lock                  = asyncio.Lock()
        self.redraw_event          = asyncio.Event()
        self.running: bool         = True
        self.current_model: str    = "unknown"
        # Output buffer scroll  (0 = pinned to bottom; N = scrolled up N display-lines)
        self.output_scroll: int    = 0
        # Mouse capture state
        self.mouse_enabled: bool   = False
        # Session switching
        self.session_switch_requested: bool = False
        self.new_session_id: str | None = None
        # Input history (bash-style Up/Down traversal)
        self.input_history: list[str] = []   # oldest first
        self.history_idx: int = -1           # -1 = not browsing; 0..n-1 = index into history
        self.history_draft: str = ""         # saved draft while browsing
        # Gate state — set when server sends a gate event; cleared after response
        self.gate_pending: bool = False      # True = waiting for user y/n

    async def append_output(self, text: str):
        async with self.lock:
            for part in text.split("\n"):
                self.output_lines.append(part)
        self.redraw_event.set()

    async def set_status(self, text: str):
        async with self.lock:
            self.status = text
        self.redraw_event.set()

    async def scroll_by(self, delta: int, output_rows: int):
        """Adjust scroll offset. delta>0 scrolls up (older), delta<0 scrolls down (newer)."""
        async with self.lock:
            # Build the full display list length so we can clamp properly.
            # (Rough estimate: each line may wrap; use line count as lower bound.)
            max_scroll = max(0, len(self.output_lines) - max(1, output_rows))
            self.output_scroll = max(0, min(max_scroll, self.output_scroll + delta))
        self.redraw_event.set()

    async def scroll_to_bottom(self):
        async with self.lock:
            self.output_scroll = 0
        self.redraw_event.set()


state = AppState()

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

async def submit_to_server(text: str):
    payload = {
        "client_id":     CLIENT_ID,
        "text":          text,
        "default_model": state.current_model,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(SUBMIT_URL, json=payload)
            if resp.status_code != 200:
                await state.append_output(
                    f"[ERROR] Server returned {resp.status_code}: {resp.text}"
                )
    except httpx.ConnectError:
        await state.append_output(
            f"[ERROR] Cannot connect to AISvc at {SERVER_HOST}. Is the server running?"
        )
    except Exception as exc:
        await state.append_output(f"[ERROR] Submit failed: {exc}")


async def gate_respond(approved: bool):
    """Send gate Y/N response to the server."""
    gate_url = f"{SERVER_HOST}/gate_respond"
    payload = {"client_id": CLIENT_ID, "approved": approved}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(gate_url, json=payload)
            if resp.status_code != 200:
                await state.append_output(
                    f"[GATE ERROR] Server returned {resp.status_code}: {resp.text}"
                )
    except Exception as exc:
        await state.append_output(f"[GATE ERROR] gate_respond failed: {exc}")


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------

async def fetch_sessions() -> list[dict]:
    """Fetch raw session list from server. Returns [] on error."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SERVER_HOST}/sessions")
            if resp.status_code == 200:
                return resp.json().get("sessions", [])
    except Exception:
        pass
    return []

async def sync_model_from_server():
    """Fetch and update current_model from the server's session state."""
    try:
        sessions = await fetch_sessions()
        for s in sessions:
            if s["client_id"] == CLIENT_ID:
                model = s.get("model", "")
                if model:
                    async with state.lock:
                        state.current_model = model
                return
    except Exception:
        pass

async def resolve_session_id(token: str) -> str | None:
    """
    Resolve a session token to a full session ID.
    If token is an integer, look it up as a shorthand ID via the server.
    Otherwise treat it as a full session ID and return as-is.
    Returns None if shorthand not found.
    """
    try:
        shorthand = int(token)
        sessions = await fetch_sessions()
        for s in sessions:
            if s.get("shorthand_id") == shorthand:
                return s["client_id"]
        return None
    except ValueError:
        return token

def _fmt_k(n: int) -> str:
    """Format an integer compactly: 1234 -> '1.2k', 123456 -> '123k', 999 -> '999'."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


async def list_sessions():
    """Fetch and display all sessions from server."""
    try:
        sessions = await fetch_sessions()
        if not sessions:
            await state.append_output("\n[shell] No active sessions found.\n")
            return
        await state.append_output("\nActive sessions:")
        for s in sessions:
            current = " (current)" if s["client_id"] == CLIENT_ID else ""
            shorthand = s.get("shorthand_id", "?")
            cid = s["client_id"]
            model = s["model"]
            history = s["history_length"]
            char_k = s.get("history_chars", 0)
            tok_est = s.get("history_token_est", 0)
            peer_ip = s.get("peer_ip")
            ip_str = f", ip={peer_ip}" if peer_ip else ""
            size_str = f" (~{char_k:,} chars, ~{tok_est:,} tok est)"
            await state.append_output(
                f"  ID [{shorthand}] {cid}: model={model}, history={history} msgs{size_str}{ip_str}{current}"
            )
            in_total = s.get("tokens_in_total", 0)
            out_total = s.get("tokens_out_total", 0)
            in_last = s.get("tokens_in_last")
            out_last = s.get("tokens_out_last")
            if in_total == 0 and out_total == 0:
                await state.append_output("  tokens: no LLM calls yet (or provider doesn't report usage)")
            else:
                last_str = f"last: in={_fmt_k(in_last)} out={_fmt_k(out_last or 0)} | " if in_last is not None else ""
                await state.append_output(
                    f"  tokens: {last_str}total: in={_fmt_k(in_total)} out={_fmt_k(out_total)}"
                )
        await state.append_output("")
    except Exception as exc:
        await state.append_output(f"[ERROR] Failed to list sessions: {exc}")

async def delete_session(session_id: str):
    """Delete a session from the server."""
    global CLIENT_ID
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.delete(f"{SERVER_HOST}/session/{session_id}")
            if resp.status_code == 200:
                await state.append_output(f"[shell] Session {session_id[:8]}... deleted.")
                # If deleting current session, generate new ID
                if session_id == CLIENT_ID:
                    CLIENT_ID = str(uuid.uuid4())
                    save_client_id(CLIENT_ID)
                    await state.append_output(f"[shell] New session created: {CLIENT_ID[:8]}...")
            elif resp.status_code == 404:
                await state.append_output(f"[ERROR] Session {session_id[:8]}... not found.")
            else:
                await state.append_output(f"[ERROR] Failed to delete session: {resp.text}")
    except Exception as exc:
        await state.append_output(f"[ERROR] Delete failed: {exc}")

async def attach_session(session_id: str):
    """Switch to a different session."""
    global CLIENT_ID
    if session_id == CLIENT_ID:
        await state.append_output("[shell] Already attached to this session.")
        return

    # Verify session exists and grab its model
    target_model = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SERVER_HOST}/sessions?client_id={session_id}")
            if resp.status_code == 200:
                data = resp.json()
                sessions = data.get("sessions", [])
                if not sessions:
                    await state.append_output(f"[ERROR] Session {session_id[:8]}... not found.")
                    return
                target_model = sessions[0].get("model")
            else:
                await state.append_output(f"[ERROR] Failed to verify session: {resp.text}")
                return
    except Exception as exc:
        await state.append_output(f"[ERROR] Failed to verify session: {exc}")
        return

    if target_model:
        async with state.lock:
            state.current_model = target_model

    # Initiate switch
    await state.append_output(f"[shell] Switching to session {session_id[:8]}...")
    save_client_id(session_id)

    async with state.lock:
        state.session_switch_requested = True
        state.new_session_id = session_id
        state.output_lines.clear()  # Clear output for new session

    # SSE listener will pick up the switch request and reconnect


# ---------------------------------------------------------------------------
# SSE listener
# ---------------------------------------------------------------------------

async def sse_listener():
    """
    Persistent SSE connection.  Decodes \\n → real newlines in token data.
    Handles: token stream, done, error.
    """
    global CLIENT_ID
    params          = {"client_id": CLIENT_ID}
    current_reply: list[str] = []
    current_event   = "message"
    gate_buf: list[str] = []   # accumulates data: lines for a multi-line gate event

    while state.running:
        # Check for session switch request
        async with state.lock:
            if state.session_switch_requested:
                CLIENT_ID = state.new_session_id
                state.session_switch_requested = False
                state.new_session_id = None
                await state.append_output(f"\n[shell] Reconnecting to session {CLIENT_ID[:8]}...\n")

        # Update params with current CLIENT_ID (may have changed)
        params = {"client_id": CLIENT_ID}

        try:
            await state.set_status("Connecting…")
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", STREAM_URL, params=params) as resp:
                    await state.set_status("Connected")

                    async for line in resp.aiter_lines():
                        if not state.running:
                            return

                        line = line.strip()

                        if not line:
                            # Blank line = SSE event dispatch boundary.
                            # If we were accumulating a gate event, dispatch it now.
                            if gate_buf:
                                decoded_gate = "\n".join(gate_buf)
                                gate_buf.clear()
                                await state.append_output(f"\n{'='*60}")
                                await state.append_output(decoded_gate)
                                await state.append_output(f"{'='*60}\n")
                                async with state.lock:
                                    state.gate_pending = True
                                await state.set_status("GATE: type y/yes to allow, anything else to deny")
                            current_event = "message"
                            continue

                        # event: field — peel one optional space per SSE spec
                        if line.startswith("event:"):
                            ev = line[6:]
                            current_event = (ev[1:] if ev.startswith(" ") else ev).strip()
                            if current_event in ("done", "flush"):
                                if current_reply:
                                    await state.append_output("")
                                    current_reply.clear()
                                gate_buf.clear()
                                current_event = "message"
                            continue

                        if not line.startswith("data:"):
                            continue

                        # Peel one optional space only (preserve internal spaces)
                        raw = line[5:]
                        if raw.startswith(" "):
                            raw = raw[1:]
                        if not raw:
                            continue

                        # ---- model change ----------------------------------
                        if current_event == "model":
                            model_name = raw.strip()
                            if model_name:
                                async with state.lock:
                                    state.current_model = model_name
                                await state.set_status(f"Connected  model={model_name}")
                            current_event = "message"
                            continue

                        # ---- gate prompt ----------------------------------
                        # Accumulate all data: lines; dispatch on blank line above.
                        if current_event == "gate":
                            gate_buf.append(raw.replace("\\n", "\n"))
                            continue

                        # ---- error ----------------------------------------
                        if current_event == "error":
                            try:
                                err = json.loads(raw)
                                await state.append_output(
                                    f"\n[SERVER ERROR] {err.get('error', raw)}"
                                )
                            except Exception:
                                await state.append_output(f"\n[SERVER ERROR] {raw}")
                            current_event = "message"
                            continue

                        # ---- regular token ---------------------------------
                        decoded = raw.replace("\\n", "\n")
                        current_reply.append(decoded)

                        async with state.lock:
                            parts = decoded.split("\n")
                            for i, part in enumerate(parts):
                                if i == 0:
                                    if state.output_lines:
                                        state.output_lines[-1] += part
                                    else:
                                        state.output_lines.append(part)
                                else:
                                    state.output_lines.append(part)
                        state.redraw_event.set()

        except httpx.ConnectError:
            await state.set_status("Server offline — retrying in 3 s…")
            await asyncio.sleep(3)
        except Exception as exc:
            if state.running:
                await state.set_status(f"SSE error: {exc} — retrying in 3 s…")
                await asyncio.sleep(3)


# ---------------------------------------------------------------------------
# User input handler
# ---------------------------------------------------------------------------

async def user_input_handler(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    # Gate response intercept — if a gate is pending, this input is the Y/N answer
    async with state.lock:
        gate_active = state.gate_pending
    if gate_active:
        approved = stripped.lower() in ("y", "yes")
        async with state.lock:
            state.gate_pending = False
        label = "ALLOWED" if approved else "DENIED"
        await state.append_output(f"[GATE] {label}")
        await gate_respond(approved)
        await state.set_status(f"Connected  model={state.current_model}")
        return True

    if stripped.startswith("!"):
        parts = stripped[1:].split(maxsplit=1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("exit", "quit"):
            state.running = False
            return False

        if cmd == "input_lines":
            try:
                n = int(arg)
                if 1 <= n <= 20:
                    async with state.lock:
                        state.input_lines = n
                    await state.append_output(f"[shell] Input area resized to {n} lines.")
                else:
                    await state.append_output("[shell] !input_lines: value must be 1–20.")
            except ValueError:
                await state.append_output("[shell] Usage: !input_lines <integer>")
            return True

        if cmd == "mouse":
            target = arg.lower()
            if target == "on":
                async with state.lock:
                    state.mouse_enabled = True
                await state.append_output("[shell] Mouse capture ON (scrolling enabled, selection disabled).")
            elif target == "off":
                async with state.lock:
                    state.mouse_enabled = False
                await state.append_output("[shell] Mouse capture OFF (scrolling disabled, selection enabled).")
            else:
                # Toggle
                async with state.lock:
                    state.mouse_enabled = not state.mouse_enabled
                    new_mode = state.mouse_enabled
                await state.append_output(f"[shell] Mouse capture {'ON' if new_mode else 'OFF'}.")
            return True

        if cmd == "session":
            ts = datetime.now().strftime("%H:%M:%S")
            await state.append_output(f"\n[{ts}] You: {stripped}")
            await state.append_output("")
            if not arg:
                # List all sessions
                await list_sessions()
            else:
                parts = arg.split(maxsplit=1)
                token = parts[0]
                action = parts[1] if len(parts) > 1 else "attach"

                # Resolve shorthand integer → full UUID
                session_id = await resolve_session_id(token)
                if session_id is None:
                    await state.append_output(f"[shell] Session ID [{token}] not found.")
                    return True

                if action == "delete":
                    await delete_session(session_id)
                elif action == "attach":
                    await attach_session(session_id)
                else:
                    await state.append_output(f"[shell] Unknown action: {action}. Use 'attach' or 'delete'.")
            return True

        if cmd == "model" and arg:
            # Forward to server — the server will push a "model" SSE event back
            ts = datetime.now().strftime("%H:%M:%S")
            await state.append_output(f"\n[{ts}] You: {stripped}")
            await state.append_output("")
            await submit_to_server(stripped)
            return True

    ts = datetime.now().strftime("%H:%M:%S")
    await state.append_output(f"\n[{ts}] You: {stripped}")
    await state.append_output("")   # blank line so LLM response starts fresh
    await submit_to_server(stripped)
    return True


# ---------------------------------------------------------------------------
# Curses rendering
# ---------------------------------------------------------------------------

def _chunk_input(text: str, usable: int) -> list[str]:
    """
    Split flat input text into display chunks of `usable` width.
    Always returns at least one element (empty string if text is empty).
    """
    if usable <= 0:
        return [text] if text else [""]
    if not text:
        return [""]
    chunks = []
    while text:
        chunks.append(text[:usable])
        text = text[usable:]
    return chunks


def _cursor_to_screen(pos: int, usable: int, input_lines: int,
                      border_row: int) -> tuple[int, int]:
    """
    Convert flat cursor position → (screen_row, screen_col).
    Also returns start_chunk (scroll offset) so the cursor chunk is visible.
    """
    if usable <= 0:
        return border_row + 1, 2
    cur_chunk     = pos // usable
    cur_col_in_c  = pos % usable
    # Scroll: ensure cur_chunk is within [start_chunk, start_chunk + input_lines)
    start_chunk   = max(0, cur_chunk - (input_lines - 1))
    screen_row    = border_row + 1 + (cur_chunk - start_chunk)
    screen_col    = 2 + cur_col_in_c          # 2 = len("> " or "  ")
    return screen_row, screen_col, start_chunk


def _draw(stdscr, snap: dict):
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()

    input_lines = snap["input_lines"]
    border_row  = max_y - input_lines - 1
    output_rows = border_row

    if output_rows < 1 or border_row < 1:
        stdscr.refresh()
        return

    # ---- Output area -------------------------------------------------------
    display: list[str] = []
    for line in snap["output_lines"]:
        if not line:
            display.append("")
            continue
        while len(line) > max_x:
            display.append(line[:max_x])
            line = line[max_x:]
        display.append(line)

    total_display = len(display)
    scroll        = snap["output_scroll"]

    if scroll == 0:
        # Pinned to bottom — show newest lines
        visible = display[-output_rows:] if total_display >= output_rows else display
    else:
        # Scrolled up — anchor is 'scroll' lines from the bottom
        end   = max(0, total_display - scroll)
        start = max(0, end - output_rows)
        visible = display[start:end]

    for row, line in enumerate(visible):
        if row >= output_rows:
            break
        try:
            stdscr.addstr(row, 0, line[:max_x])
        except Exception:
            pass

    # ---- Separator ---------------------------------------------------------
    base_status = snap["status"]
    if scroll > 0:
        lines_above = max(0, total_display - output_rows - scroll)
        base_status = f"{base_status}  [↑{scroll} scrolled — ↑{lines_above} more above]"
    elif total_display > output_rows:
        lines_above = total_display - output_rows
        base_status = f"{base_status}  [↑{lines_above} above — PgUp to scroll]"
    status_str = f" {base_status} "
    attr = curses.A_BOLD

    fill_len = max(0, max_x - len(status_str))
    sep_line = (status_str + BORDER_CHAR * fill_len)[:max_x]
    try:
        stdscr.addstr(border_row, 0, sep_line, attr)
    except Exception:
        pass

    # ---- Input area --------------------------------------------------------
    usable = max_x - 2         # 2 chars for "> " / "  " prefix

    text      = snap["input_text"]
    pos       = snap["input_cursor_pos"]
    chunks    = _chunk_input(text, usable)

    # Scroll offset so cursor chunk is always visible
    cur_chunk    = pos // usable if usable > 0 else 0
    start_chunk  = max(0, cur_chunk - (input_lines - 1))

    for i in range(input_lines):
        row = border_row + 1 + i
        if row >= max_y:
            break
        chunk_idx = start_chunk + i
        chunk     = chunks[chunk_idx] if chunk_idx < len(chunks) else ""
        prefix    = "> " if (start_chunk == 0 and i == 0) else "  "
        try:
            stdscr.addstr(row, 0, (prefix + chunk)[:max_x])
        except Exception:
            pass

    # Cursor
    col_in_chunk = pos % usable if usable > 0 else 0
    screen_row   = border_row + 1 + (cur_chunk - start_chunk)
    screen_col   = 2 + col_in_chunk
    try:
        stdscr.move(
            min(screen_row, max_y - 1),
            min(screen_col, max_x - 1),
        )
    except Exception:
        pass

    stdscr.refresh()


def _snapshot(st: AppState) -> dict:
    return {
        "output_lines":     list(st.output_lines),
        "output_scroll":    st.output_scroll,
        "input_text":       st.input_text,
        "input_cursor_pos": st.input_cursor_pos,
        "input_lines":      st.input_lines,
        "status":           st.status,
    }


# ---------------------------------------------------------------------------
# Word-movement helpers (bash-style: stop at word boundary)
# ---------------------------------------------------------------------------

def _word_left(text: str, pos: int) -> int:
    """Return position after moving one word left (Option+Left / Alt+b)."""
    p = pos
    while p > 0 and not text[p - 1].isalnum() and text[p - 1] != '_':
        p -= 1
    while p > 0 and (text[p - 1].isalnum() or text[p - 1] == '_'):
        p -= 1
    return p


def _word_right(text: str, pos: int) -> int:
    """Return position after moving one word right (Option+Right / Alt+f)."""
    n = len(text)
    p = pos
    while p < n and not text[p].isalnum() and text[p] != '_':
        p += 1
    while p < n and (text[p].isalnum() or text[p] == '_'):
        p += 1
    return p


# ---------------------------------------------------------------------------
# Input loop
# ---------------------------------------------------------------------------

async def input_loop(stdscr):
    stdscr.nodelay(True)
    curses.cbreak()
    stdscr.keypad(True)
    curses.noecho()

    # Track local mouse state to detect changes from !mouse command
    current_mouse_mode = True

    # Enable mouse events for wheel scrolling initially
    try:
        curses.mousemask(
            curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION
        )
    except Exception:
        pass

    loop = asyncio.get_event_loop()

    def _do_submit():
        text = state.input_text.strip()
        if not text:
            return

        async def _task():
            async with state.lock:
                state.input_text       = ""
                state.input_cursor_pos = 0

            # Normalize line endings, then join soft-wrapped lines.
            # A soft wrap is a newline where the preceding char is not a space
            # (terminal wrapped mid-word). A hard paragraph break has a blank
            # line (two consecutive newlines) or the preceding char is a space.
            normalized = text.replace('\r\n', '\n').replace('\r', '\n')

            # Join only true mid-word terminal soft-wraps.
            # Conditions for joining line N with line N+1:
            #   1. Line N ends with an alphanumeric char (cut mid-word)
            #   2. Line N+1 starts with a lowercase letter (word continues)
            #   3. Line N does NOT start with '!' (never glue command lines)
            # This handles "servic\ne" → "service" while keeping Keep-pasted
            # command blocks (which start new words on each line) separate.
            def _join_soft_wraps(text: str) -> str:
                lines = text.split('\n')
                out = []
                i = 0
                while i < len(lines):
                    line = lines[i]
                    while (
                        i + 1 < len(lines)
                        and line
                        and line[-1].isalnum()
                        and lines[i + 1]
                        and lines[i + 1][0].islower()
                        and not line.lstrip().startswith('!')
                    ):
                        i += 1
                        line = line + lines[i]   # no space: glue mid-word
                    out.append(line)
                    i += 1
                return '\n'.join(out)

            joined = _join_soft_wraps(normalized)

            # Now split on blank lines (paragraph breaks) for multi-paragraph input
            paragraphs = re.split(r'\n{2,}', joined)
            for para in paragraphs:
                para = para.strip()
                if para:
                    # Add to history (avoid duplicate of last entry)
                    async with state.lock:
                        if not state.input_history or state.input_history[-1] != para:
                            state.input_history.append(para)
                        state.history_idx = -1
                        state.history_draft = ""
                    await user_input_handler(para)

        loop.create_task(_task())

    while state.running:
        # ---- Sync mouse state ----------------------------------------------
        # If the user toggled mouse mode via "!mouse off", we must disable
        # the mask so the terminal performs native selection.
        if state.mouse_enabled != current_mouse_mode:
            current_mouse_mode = state.mouse_enabled
            if current_mouse_mode:
                try:
                    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
                    sys.stdout.write("\033[?1003h")
                    sys.stdout.flush()
                except Exception: pass
            else:
                try:
                    curses.mousemask(0)
                    sys.stdout.write("\033[?1003l")
                    sys.stdout.flush()
                except Exception: pass

        try:
            ch = stdscr.getch()
        except Exception:
            ch = -1

        if ch == -1:
            await asyncio.sleep(0.02)
            continue

        # ---- Escape sequence handling (Option/Alt + arrow = word movement) -
        # ESC (27) starts multi-byte sequences.  Read the rest with a brief
        # halfdelay so we don't block but still capture what follows quickly.
        if ch == 27:
            curses.halfdelay(1)   # 0.1 s timeout for subsequent bytes
            seq = [27]
            try:
                while True:
                    nc = stdscr.getch()
                    if nc == -1:
                        break
                    seq.append(nc)
                    if len(seq) >= 6:
                        break
            except Exception:
                pass
            curses.cbreak()       # restore non-blocking mode
            stdscr.nodelay(True)

            seq_bytes = bytes(seq)
            # Option+Left:  \x1b[1;3D  or  \x1bb  (macOS Terminal / iTerm2)
            # Option+Right: \x1b[1;3C  or  \x1bf
            if seq_bytes in (b'\x1b[1;3D', b'\x1bb', b'\x1b\x1b[D'):
                async with state.lock:
                    state.input_cursor_pos = _word_left(state.input_text, state.input_cursor_pos)
                state.redraw_event.set()
                continue
            elif seq_bytes in (b'\x1b[1;3C', b'\x1bf', b'\x1b\x1b[C'):
                async with state.lock:
                    state.input_cursor_pos = _word_right(state.input_text, state.input_cursor_pos)
                state.redraw_event.set()
                continue
            # Unrecognised escape sequence — discard silently
            continue

        # ---- Detect paste: collect multiple rapid characters --------------
        # If characters are arriving rapidly (paste), collect them all first
        paste_buffer = []
        if ch != -1:
            paste_buffer.append(ch)
            # Check if more characters are immediately available (paste scenario)
            try:
                while True:
                    next_ch = stdscr.getch()
                    if next_ch == -1:
                        break
                    paste_buffer.append(next_ch)
                    if len(paste_buffer) > 1000:  # Safety limit
                        break
            except Exception:
                pass

        # If we collected multiple characters, it's likely a paste
        if len(paste_buffer) > 5:  # Threshold for paste detection
            # Convert to string and handle newlines
            paste_text = ''.join(chr(c) if 32 <= c <= 126 or c in (10, 13) else '' for c in paste_buffer)
            # Add to current input
            async with state.lock:
                current_text = state.input_text
                pos = state.input_cursor_pos
                state.input_text = current_text[:pos] + paste_text + current_text[pos:]
                state.input_cursor_pos = pos + len(paste_text)

            # If paste contains newlines, trigger submission
            if '\n' in paste_text or '\r' in paste_text:
                loop.create_task(state.scroll_to_bottom())
                _do_submit()

            state.redraw_event.set()
            continue

        # Process single character normally
        ch = paste_buffer[0] if paste_buffer else -1
        if ch == -1:
            continue

        # ---- Compute output rows for scroll page size ----------------------
        max_y, max_x = stdscr.getmaxyx()
        output_rows  = max(1, max_y - state.input_lines - 1)
        page_size    = max(1, output_rows - 1)

        # ---- Scroll output buffer ------------------------------------------
        if ch == curses.KEY_PPAGE or ch == 2:      # PgUp / Ctrl+B
            loop.create_task(state.scroll_by(page_size, output_rows))
            state.redraw_event.set()
            continue

        if ch == curses.KEY_NPAGE or ch == 6:      # PgDn / Ctrl+F
            loop.create_task(state.scroll_by(-page_size, output_rows))
            state.redraw_event.set()
            continue

        if ch == 533 or ch == 566:                 # Ctrl+End (terminal-dependent codes)
            loop.create_task(state.scroll_to_bottom())
            state.redraw_event.set()
            continue

        # ---- Mouse events --------------------------------------------------
        if ch == curses.KEY_MOUSE:
            try:
                _, mx, my, mz, bstate = curses.getmouse()
                # Button 4 = wheel up, Button 5 = wheel down
                # (some terminals report as BUTTON4_PRESSED / BUTTON5_PRESSED)
                scroll_up   = bstate & 0x00080000   # BUTTON4_PRESSED
                scroll_down = bstate & 0x00800000   # BUTTON5_PRESSED
                if scroll_up:
                    loop.create_task(state.scroll_by(3, output_rows))
                    state.redraw_event.set()
                elif scroll_down:
                    loop.create_task(state.scroll_by(-3, output_rows))
                    state.redraw_event.set()
            except Exception:
                pass
            continue

        # ---- Normal mode ---------------------------------------------------
        # Compute usable width for Up/Down movement
        usable = max(1, max_x - 2)

        async with state.lock:
            text = state.input_text
            pos  = state.input_cursor_pos

        # Submit
        if ch in (curses.KEY_ENTER, 10, 13, curses.KEY_F5, 7):
            # Any input submission jumps to bottom
            loop.create_task(state.scroll_to_bottom())
            _do_submit()

        # Backspace / Delete
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if pos > 0:
                async with state.lock:
                    state.input_text       = text[:pos-1] + text[pos:]
                    state.input_cursor_pos = pos - 1

        elif ch == curses.KEY_DC:   # Delete key — delete char after cursor
            if pos < len(text):
                async with state.lock:
                    state.input_text = text[:pos] + text[pos+1:]
                # cursor stays

        # Left
        elif ch == curses.KEY_LEFT:
            if pos > 0:
                async with state.lock:
                    state.input_cursor_pos = pos - 1

        # Right
        elif ch == curses.KEY_RIGHT:
            if pos < len(text):
                async with state.lock:
                    state.input_cursor_pos = pos + 1

        # Up — history: previous entry
        elif ch == curses.KEY_UP:
            async with state.lock:
                hist = state.input_history
                if hist:
                    if state.history_idx == -1:
                        # Save current draft before entering history
                        state.history_draft = state.input_text
                        state.history_idx = len(hist) - 1
                    elif state.history_idx > 0:
                        state.history_idx -= 1
                    state.input_text = hist[state.history_idx]
                    state.input_cursor_pos = len(state.input_text)

        # Down — history: next entry (or restore draft)
        elif ch == curses.KEY_DOWN:
            async with state.lock:
                if state.history_idx != -1:
                    hist = state.input_history
                    if state.history_idx < len(hist) - 1:
                        state.history_idx += 1
                        state.input_text = hist[state.history_idx]
                    else:
                        # Past the end — restore draft
                        state.history_idx = -1
                        state.input_text = state.history_draft
                        state.history_draft = ""
                    state.input_cursor_pos = len(state.input_text)

        # Home / Ctrl+A
        elif ch in (curses.KEY_HOME, 1):
            async with state.lock:
                state.input_cursor_pos = 0

        # End / Ctrl+E
        elif ch in (curses.KEY_END, 5):
            async with state.lock:
                state.input_cursor_pos = len(state.input_text)

        # Printable ASCII — typing cancels history browsing
        elif 32 <= ch <= 126:
            async with state.lock:
                state.history_idx   = -1
                state.history_draft = ""
                state.input_text       = text[:pos] + chr(ch) + text[pos:]
                state.input_cursor_pos = pos + 1

        state.redraw_event.set()

    state.running = False


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------

async def render_loop(stdscr):
    while state.running:
        await state.redraw_event.wait()
        state.redraw_event.clear()
        snap = _snapshot(state)
        try:
            _draw(stdscr, snap)
        except Exception:
            pass
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main(stdscr):
    curses.curs_set(1)

    await state.append_output(
        "AIOps Shell  |  PgUp/PgDn=scroll  !mouse=selection toggle  !help=commands  !exit=quit"
    )
    await state.append_output(
        f"Session: {CLIENT_ID[:8]}…   Server: {SERVER_HOST}"
    )
    await state.append_output("")

    await asyncio.gather(
        sse_listener(),
        input_loop(stdscr),
        render_loop(stdscr),
    )


def main():
    # Set locale so curses handles UTF-8 and non-ASCII characters correctly.
    locale.setlocale(locale.LC_ALL, "")

    # Enable xterm-style extended mouse reporting (button 4/5 = wheel up/down)
    # Must be written BEFORE curses.wrapper takes over stdout.
    try:
        import sys
        sys.stdout.write("\033[?1003h")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        curses.wrapper(lambda stdscr: asyncio.run(async_main(stdscr)))
    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal: disable extended mouse reporting
        try:
            import sys
            sys.stdout.write("\033[?1003l")
            sys.stdout.flush()
        except Exception:
            pass
        print("\nAIOps Shell exited.")


if __name__ == "__main__":
    main()