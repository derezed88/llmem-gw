"""
Google Tasks Plugin for MCP Agent

Provides google_tasks tool for operations on Google Tasks.
Operations: list_tasklists, create_tasklist, delete_tasklist, update_tasklist,
            list_tasks, get_task, create_task, update_task, complete_task,
            delete_task, move_task, clear_completed
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from tasks_google import run_tasks_op


class _GoogleTasksArgs(BaseModel):
    operation: Literal[
        "list_tasklists", "create_tasklist", "delete_tasklist", "update_tasklist",
        "list_tasks", "get_task", "create_task", "update_task", "complete_task",
        "delete_task", "move_task", "clear_completed",
    ] = Field(
        description=(
            "Operation to perform. "
            "Task list ops: list_tasklists, create_tasklist, delete_tasklist, update_tasklist. "
            "Task ops: list_tasks, get_task, create_task, update_task, complete_task, "
            "delete_task, move_task, clear_completed."
        )
    )
    tasklist_id: Optional[str] = Field(
        default="@default",
        description="Task list ID (default '@default' = user's primary list). Use list_tasklists to discover IDs."
    )
    task_id: Optional[str] = Field(
        default="",
        description="Task ID. Required for get/update/complete/delete/move operations."
    )
    title: Optional[str] = Field(
        default="",
        description="Title for task or task list (max 1024 chars). Required for create operations."
    )
    notes: Optional[str] = Field(
        default="",
        description="Task notes/description (max 8192 chars). For create_task/update_task."
    )
    status: Optional[str] = Field(
        default="",
        description="Task status: 'needsAction' or 'completed'. For update_task."
    )
    due: Optional[str] = Field(
        default="",
        description="Due date (RFC 3339 or YYYY-MM-DD). For create_task/update_task."
    )
    due_min: Optional[str] = Field(
        default="",
        description="Lower bound for due date filter (RFC 3339). For list_tasks."
    )
    due_max: Optional[str] = Field(
        default="",
        description="Upper bound for due date filter (RFC 3339). For list_tasks."
    )
    parent: Optional[str] = Field(
        default="",
        description="Parent task ID for subtasks. For create_task/move_task."
    )
    previous: Optional[str] = Field(
        default="",
        description="Previous sibling task ID for ordering. For create_task/move_task."
    )
    show_completed: Optional[bool] = Field(
        default=True,
        description="Include completed tasks in list (default true). For list_tasks."
    )
    show_hidden: Optional[bool] = Field(
        default=False,
        description="Include hidden tasks in list (default false). For list_tasks."
    )
    max_results: Optional[int] = Field(
        default=100,
        description="Max tasks to return (default 100, max 100). For list_tasks."
    )


async def google_tasks_executor(
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
    """Execute Google Tasks operation."""
    return await run_tasks_op(
        operation,
        tasklist_id or "@default",
        task_id or "",
        title or "",
        notes or "",
        status or "",
        due or "",
        due_min or "",
        due_max or "",
        parent or "",
        previous or "",
        show_completed if show_completed is not None else True,
        show_hidden if show_hidden is not None else False,
        max_results or 100,
    )


class GoogleTasksPlugin(BasePlugin):
    """Google Tasks plugin."""

    PLUGIN_NAME = "plugin_tasks_google"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Google Tasks management (task lists, tasks, subtasks, completion)"
    DEPENDENCIES = ["google-auth", "google-auth-oauthlib", "google-api-python-client"]
    ENV_VARS = []

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize Google Tasks plugin."""
        try:
            import os
            from config import TASKS_CREDS_FILE, TASKS_TOKEN_FILE

            if not os.path.exists(TASKS_CREDS_FILE):
                print("Google Tasks plugin: credentials.json not found")
                return False

            if not os.path.exists(TASKS_TOKEN_FILE):
                print(
                    f"Google Tasks plugin: {TASKS_TOKEN_FILE} not found. "
                    "Run tasks_google_auth.py to authorize."
                )
                return False

            self.enabled = True
            return True
        except Exception as e:
            print(f"Google Tasks plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=google_tasks_executor,
                    name="google_tasks",
                    description=(
                        "Operations on Google Tasks. "
                        "Task list ops: list_tasklists, create_tasklist, delete_tasklist, update_tasklist. "
                        "Task ops: list_tasks (with due date filters), get_task, "
                        "create_task (with title, notes, due date, parent for subtasks), "
                        "update_task, complete_task, delete_task, "
                        "move_task (reorder/nest), clear_completed. "
                        "Use list_tasklists first to discover list IDs."
                    ),
                    args_schema=_GoogleTasksArgs,
                )
            ]
        }
