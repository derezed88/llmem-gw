"""
Google Tasks API operations.

Provides task list management, task CRUD, and task reordering.
Uses a separate token file (token_tasks.json) from Drive/Calendar to keep scopes isolated.

API reference: https://developers.google.com/workspace/tasks/reference/rest
"""

import os
import asyncio

from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import (
    log,
    TASKS_TOKEN_FILE,
    TASKS_CREDS_FILE,
    TASKS_SCOPES,
)

_tasks_service = None


def _get_tasks_service():
    """Build or return cached Tasks API v1 service."""
    global _tasks_service
    creds = None

    if os.path.exists(TASKS_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            TASKS_TOKEN_FILE, TASKS_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GAuthRequest())
        else:
            if not os.path.exists(TASKS_CREDS_FILE):
                raise FileNotFoundError(
                    "Missing 'credentials.json'. Download from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                TASKS_CREDS_FILE, TASKS_SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TASKS_TOKEN_FILE, "w") as fh:
            fh.write(creds.to_json())

    _tasks_service = build("tasks", "v1", credentials=creds)
    return _tasks_service


# ── Task Lists ──────────────────────────────────────────────

def _list_tasklists() -> str:
    """List all task lists."""
    svc = _get_tasks_service()
    result = svc.tasklists().list(maxResults=100).execute()
    items = result.get("items", [])
    if not items:
        return "No task lists found."
    lines = ["Task Lists:"]
    for tl in items:
        lines.append(
            f"  - {tl.get('title', '(untitled)')}  (id: {tl['id']}, updated: {tl.get('updated', '?')})"
        )
    return "\n".join(lines)


def _create_tasklist(title: str) -> str:
    """Create a new task list."""
    svc = _get_tasks_service()
    created = svc.tasklists().insert(body={"title": title[:1024]}).execute()
    return f"Created task list: {created.get('title', '?')} (id: {created['id']})"


def _delete_tasklist(tasklist_id: str) -> str:
    """Delete a task list."""
    svc = _get_tasks_service()
    svc.tasklists().delete(tasklist=tasklist_id).execute()
    return f"Deleted task list: {tasklist_id}"


def _update_tasklist(tasklist_id: str, title: str) -> str:
    """Update (rename) a task list."""
    svc = _get_tasks_service()
    updated = svc.tasklists().patch(
        tasklist=tasklist_id, body={"title": title[:1024]}
    ).execute()
    return f"Updated task list: {updated.get('title', '?')} (id: {updated['id']})"


# ── Tasks ───────────────────────────────────────────────────

def _format_task(task: dict, indent: int = 2) -> str:
    """Format a single task for display."""
    prefix = " " * indent
    status_icon = "[x]" if task.get("status") == "completed" else "[ ]"
    title = task.get("title", "(untitled)")
    line = f"{prefix}{status_icon} {title}"

    parts = []
    if task.get("due"):
        parts.append(f"due: {task['due'][:10]}")
    if task.get("status") == "completed" and task.get("completed"):
        parts.append(f"completed: {task['completed'][:10]}")
    parts.append(f"id: {task['id']}")

    line += f"  ({', '.join(parts)})"

    if task.get("notes"):
        notes = task["notes"]
        if len(notes) > 200:
            notes = notes[:200] + "..."
        line += f"\n{prefix}  Notes: {notes}"

    if task.get("webViewLink"):
        line += f"\n{prefix}  Link: {task['webViewLink']}"

    return line


def _list_tasks(
    tasklist_id: str = "@default",
    show_completed: bool = True,
    show_hidden: bool = False,
    due_min: str = "",
    due_max: str = "",
    max_results: int = 100,
) -> str:
    """List tasks in a task list."""
    svc = _get_tasks_service()

    kwargs = {
        "tasklist": tasklist_id,
        "maxResults": min(max_results, 100),
        "showCompleted": show_completed,
        "showHidden": show_hidden,
    }
    if due_min:
        kwargs["dueMin"] = due_min
    if due_max:
        kwargs["dueMax"] = due_max

    all_tasks = []
    page_token = None

    while True:
        if page_token:
            kwargs["pageToken"] = page_token
        result = svc.tasks().list(**kwargs).execute()
        items = result.get("items", [])
        all_tasks.extend(items)
        page_token = result.get("nextPageToken")
        if not page_token or len(all_tasks) >= max_results:
            break

    if not all_tasks:
        return f"No tasks found in list '{tasklist_id}'."

    # Group by parent for hierarchy display
    top_level = []
    children = {}
    for task in all_tasks:
        parent = task.get("parent")
        if parent:
            children.setdefault(parent, []).append(task)
        else:
            top_level.append(task)

    lines = [f"Tasks ({len(all_tasks)}):"]
    for task in top_level:
        lines.append(_format_task(task))
        for child in children.get(task["id"], []):
            lines.append(_format_task(child, indent=6))

    return "\n".join(lines)


def _get_task(tasklist_id: str, task_id: str) -> str:
    """Get detailed info about a single task."""
    svc = _get_tasks_service()
    task = svc.tasks().get(tasklist=tasklist_id, task=task_id).execute()

    lines = [
        f"Task: {task.get('title', '(untitled)')}",
        f"  Status: {task.get('status', '?')}",
        f"  Updated: {task.get('updated', '?')}",
    ]
    if task.get("due"):
        lines.append(f"  Due: {task['due'][:10]}")
    if task.get("completed"):
        lines.append(f"  Completed: {task['completed'][:10]}")
    if task.get("notes"):
        notes = task["notes"]
        if len(notes) > 2000:
            notes = notes[:2000] + "..."
        lines.append(f"  Notes: {notes}")
    if task.get("parent"):
        lines.append(f"  Parent: {task['parent']}")
    if task.get("webViewLink"):
        lines.append(f"  Link: {task['webViewLink']}")
    lines.append(f"  ID: {task['id']}")

    return "\n".join(lines)


def _create_task(
    tasklist_id: str = "@default",
    title: str = "",
    notes: str = "",
    due: str = "",
    parent: str = "",
    previous: str = "",
) -> str:
    """Create a new task."""
    svc = _get_tasks_service()

    body = {}
    if title:
        body["title"] = title[:1024]
    if notes:
        body["notes"] = notes[:8192]
    if due:
        # API expects RFC 3339 but only uses date portion
        if "T" not in due:
            due = due + "T00:00:00.000Z"
        body["due"] = due

    kwargs = {"tasklist": tasklist_id, "body": body}
    if parent:
        kwargs["parent"] = parent
    if previous:
        kwargs["previous"] = previous

    created = svc.tasks().insert(**kwargs).execute()
    return (
        f"Created task: {created.get('title', '(untitled)')} "
        f"(id: {created['id']}, list: {tasklist_id})"
    )


def _update_task(
    tasklist_id: str,
    task_id: str,
    title: str = "",
    notes: str = "",
    status: str = "",
    due: str = "",
) -> str:
    """Update an existing task."""
    svc = _get_tasks_service()

    # Fetch current to merge
    current = svc.tasks().get(tasklist=tasklist_id, task=task_id).execute()

    if title:
        current["title"] = title[:1024]
    if notes is not None and notes != "":
        current["notes"] = notes[:8192]
    if status in ("needsAction", "completed"):
        current["status"] = status
        if status == "needsAction":
            current.pop("completed", None)
    if due:
        if "T" not in due:
            due = due + "T00:00:00.000Z"
        current["due"] = due

    updated = svc.tasks().update(
        tasklist=tasklist_id, task=task_id, body=current
    ).execute()
    return (
        f"Updated task: {updated.get('title', '?')} "
        f"(status: {updated.get('status', '?')}, id: {updated['id']})"
    )


def _delete_task(tasklist_id: str, task_id: str) -> str:
    """Delete a task."""
    svc = _get_tasks_service()
    svc.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
    return f"Deleted task: {task_id}"


def _complete_task(tasklist_id: str, task_id: str) -> str:
    """Mark a task as completed."""
    return _update_task(tasklist_id, task_id, status="completed")


def _move_task(
    tasklist_id: str,
    task_id: str,
    parent: str = "",
    previous: str = "",
) -> str:
    """Move/reorder a task."""
    svc = _get_tasks_service()
    kwargs = {"tasklist": tasklist_id, "task": task_id}
    if parent:
        kwargs["parent"] = parent
    if previous:
        kwargs["previous"] = previous
    moved = svc.tasks().move(**kwargs).execute()
    return f"Moved task: {moved.get('title', '?')} (id: {moved['id']})"


def _clear_completed(tasklist_id: str) -> str:
    """Clear all completed tasks from a list."""
    svc = _get_tasks_service()
    svc.tasks().clear(tasklist=tasklist_id).execute()
    return f"Cleared completed tasks from list: {tasklist_id}"


# --- async dispatcher ---

async def run_tasks_op(
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
    """Dispatch Tasks operations."""
    op = (operation or "").strip().lower()

    try:
        # Task list operations
        if op == "list_tasklists":
            return await asyncio.to_thread(_list_tasklists)

        elif op == "create_tasklist":
            if not title:
                return "Error: title required for 'create_tasklist'."
            return await asyncio.to_thread(_create_tasklist, title)

        elif op == "delete_tasklist":
            if not tasklist_id or tasklist_id == "@default":
                return "Error: tasklist_id required for 'delete_tasklist'."
            return await asyncio.to_thread(_delete_tasklist, tasklist_id)

        elif op == "update_tasklist":
            if not tasklist_id or tasklist_id == "@default":
                return "Error: tasklist_id required for 'update_tasklist'."
            if not title:
                return "Error: title required for 'update_tasklist'."
            return await asyncio.to_thread(_update_tasklist, tasklist_id, title)

        # Task operations
        elif op == "list_tasks":
            return await asyncio.to_thread(
                _list_tasks, tasklist_id, show_completed, show_hidden,
                due_min, due_max, max_results,
            )

        elif op == "get_task":
            if not task_id:
                return "Error: task_id required for 'get_task'."
            return await asyncio.to_thread(_get_task, tasklist_id, task_id)

        elif op == "create_task":
            if not title:
                return "Error: title required for 'create_task'."
            return await asyncio.to_thread(
                _create_task, tasklist_id, title, notes, due, parent, previous,
            )

        elif op == "update_task":
            if not task_id:
                return "Error: task_id required for 'update_task'."
            return await asyncio.to_thread(
                _update_task, tasklist_id, task_id, title, notes, status, due,
            )

        elif op == "complete_task":
            if not task_id:
                return "Error: task_id required for 'complete_task'."
            return await asyncio.to_thread(_complete_task, tasklist_id, task_id)

        elif op == "delete_task":
            if not task_id:
                return "Error: task_id required for 'delete_task'."
            return await asyncio.to_thread(_delete_task, tasklist_id, task_id)

        elif op == "move_task":
            if not task_id:
                return "Error: task_id required for 'move_task'."
            return await asyncio.to_thread(
                _move_task, tasklist_id, task_id, parent, previous,
            )

        elif op == "clear_completed":
            return await asyncio.to_thread(_clear_completed, tasklist_id)

        else:
            return (
                "Error: unknown operation. "
                "Valid: list_tasklists | create_tasklist | delete_tasklist | update_tasklist | "
                "list_tasks | get_task | create_task | update_task | complete_task | "
                "delete_task | move_task | clear_completed"
            )

    except Exception as exc:
        log.exception("Tasks op '%s' failed", op)
        return f"Tasks error ({op}): {exc}"
