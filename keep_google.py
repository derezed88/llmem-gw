"""
Google Keep API operations.

Provides note listing, retrieval, creation, and deletion.
Uses a separate token file (token_keep.json) from Drive/Calendar to keep scopes isolated.

API reference: https://developers.google.com/workspace/keep/api/reference/rest
"""

import os
import json
import asyncio

from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import (
    log,
    KEEP_TOKEN_FILE,
    KEEP_CREDS_FILE,
    KEEP_SCOPES,
)

_keep_service = None


def _get_keep_service():
    """Build or return cached Keep API v1 service."""
    global _keep_service
    creds = None

    if os.path.exists(KEEP_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            KEEP_TOKEN_FILE, KEEP_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GAuthRequest())
        else:
            if not os.path.exists(KEEP_CREDS_FILE):
                raise FileNotFoundError(
                    "Missing 'credentials.json'. Download from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                KEEP_CREDS_FILE, KEEP_SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(KEEP_TOKEN_FILE, "w") as fh:
            fh.write(creds.to_json())

    _keep_service = build("keep", "v1", credentials=creds)
    return _keep_service


def _format_note(note: dict) -> str:
    """Format a single note for display."""
    lines = [f"Note: {note.get('name', '?')}"]
    title = note.get("title", "")
    if title:
        lines.append(f"  Title: {title}")

    # Trashed status
    if note.get("trashed"):
        lines.append(f"  Status: TRASHED (at {note.get('trashTime', '?')})")

    lines.append(f"  Created: {note.get('createTime', '?')}")
    lines.append(f"  Updated: {note.get('updateTime', '?')}")

    # Body content
    body = note.get("body", {})
    text_content = body.get("text", {})
    list_content = body.get("list", {})

    if text_content and text_content.get("text"):
        text = text_content["text"]
        if len(text) > 2000:
            text = text[:2000] + "..."
        lines.append(f"  Content: {text}")

    if list_content and list_content.get("listItems"):
        lines.append("  Checklist:")
        for item in list_content["listItems"][:50]:
            checked = "[x]" if item.get("checked") else "[ ]"
            item_text = item.get("text", {}).get("text", "")
            lines.append(f"    {checked} {item_text}")
            # Nested items (one level only)
            for child in item.get("childListItems", []):
                child_checked = "[x]" if child.get("checked") else "[ ]"
                child_text = child.get("text", {}).get("text", "")
                lines.append(f"      {child_checked} {child_text}")

    # Attachments
    attachments = note.get("attachments", [])
    if attachments:
        lines.append(f"  Attachments: {len(attachments)}")
        for att in attachments[:10]:
            lines.append(f"    - {att.get('name', '?')} ({', '.join(att.get('mimeType', []))})")

    return "\n".join(lines)


def _list_notes(filter_str: str = "", page_size: int = 25) -> str:
    """List notes with optional filter."""
    svc = _get_keep_service()

    kwargs = {"pageSize": min(page_size, 100)}
    if filter_str:
        kwargs["filter"] = filter_str

    all_notes = []
    page_token = None

    while True:
        if page_token:
            kwargs["pageToken"] = page_token
        result = svc.notes().list(**kwargs).execute()
        notes = result.get("notes", [])
        all_notes.extend(notes)
        page_token = result.get("nextPageToken")
        if not page_token or len(all_notes) >= page_size:
            break

    if not all_notes:
        return "No notes found."

    lines = [f"Notes ({len(all_notes)}):"]
    for note in all_notes:
        title = note.get("title", "(untitled)")
        name = note.get("name", "?")
        trashed = " [TRASHED]" if note.get("trashed") else ""
        updated = note.get("updateTime", "?")

        # Preview body
        body = note.get("body", {})
        text_content = body.get("text", {})
        list_content = body.get("list", {})
        preview = ""
        if text_content and text_content.get("text"):
            preview = text_content["text"][:80].replace("\n", " ")
        elif list_content and list_content.get("listItems"):
            count = len(list_content["listItems"])
            checked = sum(1 for i in list_content["listItems"] if i.get("checked"))
            preview = f"[checklist: {checked}/{count} done]"

        line = f"  - {title}{trashed}  (id: {name}, updated: {updated})"
        if preview:
            line += f"\n    {preview}"
        lines.append(line)

    return "\n".join(lines)


def _get_note(note_name: str) -> str:
    """Get detailed info about a single note."""
    svc = _get_keep_service()
    # Accept bare ID or full resource name
    if not note_name.startswith("notes/"):
        note_name = f"notes/{note_name}"
    note = svc.notes().get(name=note_name).execute()
    return _format_note(note)


def _create_note(
    title: str = "",
    text: str = "",
    list_items: str = "",
) -> str:
    """Create a new note.

    Args:
        title: Note title (max 1000 chars).
        text: Plain text body (max 20000 chars). Mutually exclusive with list_items.
        list_items: Checklist items, one per line. Prefix with [x] for checked.
                    Indent with 2+ spaces for nested items.
    """
    svc = _get_keep_service()

    note_body = {}
    if title:
        note_body["title"] = title[:1000]

    if list_items:
        items = []
        for raw_line in list_items.split("\n"):
            if not raw_line.strip():
                continue
            # Detect nesting (2+ leading spaces)
            is_nested = raw_line.startswith("  ") and len(items) > 0
            line = raw_line.strip()
            # Detect checked status
            checked = False
            if line.startswith("[x]") or line.startswith("[X]"):
                checked = True
                line = line[3:].strip()
            elif line.startswith("[ ]"):
                line = line[3:].strip()

            item = {
                "text": {"text": line[:1000]},
                "checked": checked,
            }

            if is_nested and items:
                # Add as child of last top-level item
                if "childListItems" not in items[-1]:
                    items[-1]["childListItems"] = []
                items[-1]["childListItems"].append(item)
            else:
                items.append(item)

        note_body["body"] = {"list": {"listItems": items}}
    elif text:
        note_body["body"] = {"text": {"text": text[:20000]}}

    created = svc.notes().create(body=note_body).execute()
    return f"Created note: {created.get('title', '(untitled)')} (id: {created.get('name', '?')})"


def _delete_note(note_name: str) -> str:
    """Delete a note. Requires OWNER role."""
    svc = _get_keep_service()
    if not note_name.startswith("notes/"):
        note_name = f"notes/{note_name}"
    svc.notes().delete(name=note_name).execute()
    return f"Deleted note: {note_name}"


# --- async dispatcher ---

async def run_keep_op(
    operation: str,
    note_id: str = "",
    title: str = "",
    text: str = "",
    list_items: str = "",
    filter_str: str = "",
    page_size: int = 25,
) -> str:
    """Dispatch Keep operations."""
    op = (operation or "").strip().lower()

    try:
        if op == "list":
            return await asyncio.to_thread(_list_notes, filter_str, page_size)

        elif op == "get":
            if not note_id:
                return "Error: note_id required for 'get'."
            return await asyncio.to_thread(_get_note, note_id)

        elif op == "create":
            if not title and not text and not list_items:
                return "Error: at least one of title, text, or list_items required for 'create'."
            return await asyncio.to_thread(_create_note, title, text, list_items)

        elif op == "delete":
            if not note_id:
                return "Error: note_id required for 'delete'."
            return await asyncio.to_thread(_delete_note, note_id)

        else:
            return (
                "Error: unknown operation. "
                "Valid: list | get | create | delete"
            )

    except Exception as exc:
        log.exception("Keep op '%s' failed", op)
        return f"Keep error ({op}): {exc}"
