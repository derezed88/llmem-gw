import json
import os
import asyncio
import re
import mysql.connector
#from .config import log
from config import log

def _load_database_name() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db-config.json")
    try:
        with open(path) as f:
            return json.load(f).get("database", "")
    except FileNotFoundError:
        log.warning("db-config.json not found — database name not set")
        return ""
    except Exception as e:
        log.warning(f"db-config.json load failed: {e}")
        return ""

_DATABASE = _load_database_name()

def _connect() -> mysql.connector.MySQLConnection:
    return mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
        database=_DATABASE,
    )

def _run_sql(sql: str) -> str:
    # Fresh connection per call — mysql.connector is not thread-safe with a
    # shared connection when multiple asyncio.to_thread calls run concurrently.
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        if cursor.description:
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            if not rows:
                return "(no rows)"
            col_widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
                          for i, c in enumerate(cols)]
            fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
            sep = "-+-".join("-" * w for w in col_widths)
            lines = [fmt.format(*cols), sep]
            for row in rows:
                lines.append(fmt.format(*[str(v) for v in row]))
            return "\n".join(lines)
        else:
            conn.commit()
            return f"OK — rows affected: {cursor.rowcount}"
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        raise exc
    finally:
        cursor.close()
        try:
            conn.close()
        except Exception:
            pass

async def execute_sql(sql: str) -> str:
    return await asyncio.to_thread(_run_sql, sql)

def _run_insert(sql: str) -> int:
    """Run an INSERT and return lastrowid within the same connection."""
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
        return cursor.lastrowid or 0
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        raise exc
    finally:
        cursor.close()
        try:
            conn.close()
        except Exception:
            pass

async def execute_insert(sql: str) -> int:
    """Execute an INSERT and return the new row id (same-connection, avoids LAST_INSERT_ID race)."""
    return await asyncio.to_thread(_run_insert, sql)

def extract_table_names(sql: str) -> list[str]:
    u = sql.upper()
    patterns = [
        r"\bFROM\s+(\w+)",
        r"\bJOIN\s+(\w+)",
        r"\bINTO\s+(\w+)",
        r"\bUPDATE\s+(\w+)",
        r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
        r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)",
        r"\bDELETE\s+FROM\s+(\w+)",
        r"\bTRUNCATE\s+(?:TABLE\s+)?(\w+)",
    ]
    tables = set()
    for p in patterns:
        for m in re.finditer(p, u):
            tables.add(m.group(1).lower())
    return list(tables)