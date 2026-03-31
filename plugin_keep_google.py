"""
Google Keep Plugin for MCP Agent

Provides google_keep tool for operations on Google Keep notes.
Operations: list, get, create, delete
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from keep_google import run_keep_op


class _GoogleKeepArgs(BaseModel):
    operation: Literal["list", "get", "create", "delete"] = Field(
        description="Operation to perform: list, get, create, delete"
    )
    note_id: Optional[str] = Field(
        default="",
        description="Note resource name (e.g. 'notes/abc123'). Required for get/delete."
    )
    title: Optional[str] = Field(
        default="",
        description="Note title (max 1000 chars). Used for create."
    )
    text: Optional[str] = Field(
        default="",
        description="Plain text body (max 20000 chars). For create. Mutually exclusive with list_items."
    )
    list_items: Optional[str] = Field(
        default="",
        description=(
            "Checklist items, one per line. For create. "
            "Prefix with [x] for checked, [ ] for unchecked. "
            "Indent with 2+ spaces for nested items."
        )
    )
    filter_str: Optional[str] = Field(
        default="",
        description=(
            "Filter for list operation. Uses Google AIP filtering syntax. "
            "Filter by createTime, updateTime, trashTime, or trashed. "
            "Example: 'trashed = false' or 'updateTime > \"2026-01-01T00:00:00Z\"'"
        )
    )
    page_size: Optional[int] = Field(
        default=25,
        description="Max notes to return for list (default 25, max 100)"
    )


async def google_keep_executor(
    operation: str,
    note_id: str = "",
    title: str = "",
    text: str = "",
    list_items: str = "",
    filter_str: str = "",
    page_size: int = 25,
) -> str:
    """Execute Google Keep operation."""
    return await run_keep_op(
        operation,
        note_id or "",
        title or "",
        text or "",
        list_items or "",
        filter_str or "",
        page_size or 25,
    )


class GoogleKeepPlugin(BasePlugin):
    """Google Keep notes plugin."""

    PLUGIN_NAME = "plugin_keep_google"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Google Keep note management (list, get, create, delete)"
    DEPENDENCIES = ["google-auth", "google-auth-oauthlib", "google-api-python-client"]
    ENV_VARS = []

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize Google Keep plugin."""
        try:
            import os
            from config import KEEP_CREDS_FILE, KEEP_TOKEN_FILE

            if not os.path.exists(KEEP_CREDS_FILE):
                print("Google Keep plugin: credentials.json not found")
                return False

            if not os.path.exists(KEEP_TOKEN_FILE):
                print(
                    f"Google Keep plugin: {KEEP_TOKEN_FILE} not found. "
                    "Run keep_google_auth.py to authorize."
                )
                return False

            self.enabled = True
            return True
        except Exception as e:
            print(f"Google Keep plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=google_keep_executor,
                    name="google_keep",
                    description=(
                        "Operations on Google Keep notes. "
                        "Operations: list (all notes with optional filter), "
                        "get (single note by ID), "
                        "create (new note with title, text body, or checklist), "
                        "delete (remove note by ID — requires OWNER role). "
                        "Use list first to discover note IDs."
                    ),
                    args_schema=_GoogleKeepArgs,
                )
            ]
        }
