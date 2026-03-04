"""
MySQL Database Plugin for MCP Agent

Provides db_query tool for executing SQL queries against MySQL database.
Includes per-table read/write gate management.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from database import execute_sql


class _DbQueryArgs(BaseModel):
    sql: str = Field(description="SQL query to execute")


import re as _re
import json as _json
import os as _os

def _load_table_prefix() -> str:
    """Read managed_table_prefix from db-config.json, defaulting to empty string."""
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "db-config.json")
    try:
        with open(path) as f:
            return _json.load(f).get("managed_table_prefix", "")
    except Exception:
        return ""

_TABLE_PREFIX = _load_table_prefix()

async def db_query_executor(sql: str) -> str:
    """Execute SQL query. If a prefixed table is not found, retries with prefix stripped."""
    try:
        return await execute_sql(sql)
    except Exception as exc:
        if "1146" in str(exc) and _TABLE_PREFIX and _TABLE_PREFIX in sql:
            fixed = _re.sub(rf'\b{_re.escape(_TABLE_PREFIX)}(\w+)\b', r'\1', sql)
            if fixed != sql:
                try:
                    return await execute_sql(fixed)
                except Exception as exc2:
                    return f"[db_query error] {exc2}"
        return f"[db_query error] {exc}"


class MysqlPlugin(BasePlugin):
    """MySQL database query plugin."""

    PLUGIN_NAME = "plugin_database_mysql"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "MySQL database query tool with per-table gates"
    DEPENDENCIES = ["mysql-connector-python>=8.0"]
    ENV_VARS = ["MYSQL_USER", "MYSQL_PASS"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize MySQL plugin."""
        try:
            from database import _connect
            conn = _connect()
            if not conn.is_connected():
                conn.close()
                return False
            conn.close()
            self.enabled = True
            return True
        except Exception as e:
            print(f"MySQL plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup MySQL connections."""
        # Per-call connection model — nothing to clean up globally.
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        """Return MySQL tool definitions in LangChain StructuredTool format."""
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=db_query_executor,
                    name="db_query",
                    description="Execute a SQL query against the mymcp MySQL database.",
                    args_schema=_DbQueryArgs,
                )
            ]
        }
