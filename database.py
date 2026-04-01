import json
import os
import asyncio
import re
import contextvars
import mysql.connector
#from .config import log
from config import log

def _load_db_config() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db-config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("db-config.json not found — database name not set")
        return {}
    except Exception as e:
        log.warning(f"db-config.json load failed: {e}")
        return {}

_db_cfg = _load_db_config()
_DB_DEFAULT    = _db_cfg.get("database", "")
_DB_TABLES     = _db_cfg.get("tables", {})   # db_name -> {memory_shortterm, ...}
_DB_META       = _db_cfg.get("meta", {})     # db_name -> {source: "switch"|"config", created_at: ...}

# Context variable — set per-request so all DB calls in that request use the
# correct database without threading model_key through every call site.
_active_model_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_active_model_key", default=""
)

# Per-session database override — when set, bypasses model-based DB routing.
_active_db_override: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_active_db_override", default=""
)

def set_model_context(model_key: str) -> None:
    """Set the active model key for DB routing in the current async context."""
    _active_model_key.set(model_key or "")

def set_db_override(db_name: str) -> None:
    """Set a session-level database override that bypasses model-based routing."""
    _active_db_override.set(db_name or "")

def get_db_override() -> str:
    """Return the current db override, or empty string if none."""
    return _active_db_override.get()

def get_database_for_model(model_key: str | None = None) -> str:
    """Return the database name for a model key from LLM_REGISTRY, or None if not configured.

    Session-level db override (set via !db switch) takes precedence.
    """
    # Session-level override wins
    override = _active_db_override.get()
    if override:
        return override
    from config import LLM_REGISTRY
    key = model_key or _active_model_key.get()
    if not key:
        return _DB_DEFAULT
    return LLM_REGISTRY.get(key, {}).get("database") or _DB_DEFAULT

def get_tables_for_model(model_key: str | None = None) -> dict:
    """Return the table name map for the active model's database.

    Falls back to the first table set defined, then bare logical names.
    """
    db = get_database_for_model(model_key)
    if db in _DB_TABLES:
        return _DB_TABLES[db]
    # Fallback: use first defined table set (should not normally happen)
    if _DB_TABLES:
        return next(iter(_DB_TABLES.values()))
    return {}

def _connect() -> mysql.connector.MySQLConnection:
    return mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
        database=get_database_for_model(),
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

def _fetch_dicts(sql: str) -> list[dict]:
    """Run a SELECT and return rows as list of dicts (no text formatting)."""
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        if not cursor.description:
            return []
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()
        try:
            conn.close()
        except Exception:
            pass

async def fetch_dicts(sql: str) -> list[dict]:
    """Async wrapper for _fetch_dicts — returns list of dicts, pipe-safe."""
    return await asyncio.to_thread(_fetch_dicts, sql)

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

# ---------------------------------------------------------------------------
# Database lifecycle: create / delete / list
# ---------------------------------------------------------------------------

# Protected databases that cannot be deleted via !db delete
_PROTECTED_DBS: list[str] = _db_cfg.get("protected_databases", [])

def get_protected_databases() -> list[str]:
    """Return list of database names that cannot be deleted via !db delete."""
    return list(_PROTECTED_DBS)

def list_databases() -> list[str]:
    """Return all database names registered in db-config."""
    return list(_DB_TABLES.keys())

def list_managed_databases() -> list[str]:
    """Return databases flagged as managed (for background tasks like auto-review).
    Falls back to protected_databases, then all databases."""
    managed = _db_cfg.get("managed_databases")
    if managed:
        return [db for db in managed if db in _DB_TABLES]
    protected = _db_cfg.get("protected_databases")
    if protected:
        return [db for db in protected if db in _DB_TABLES]
    return list(_DB_TABLES.keys())

def get_db_meta(db_name: str) -> dict:
    """Return metadata for a database entry: {source, created_at}.
    source is 'switch' (created via !db switch) or 'config' (pre-defined in db-config.json).
    """
    return _DB_META.get(db_name, {"source": "config"})

def list_user_databases() -> list[str]:
    """Return databases created via !db switch (source='switch'), not pre-defined in config."""
    return [db for db in _DB_TABLES if _DB_META.get(db, {}).get("source") == "switch"]

def get_model_databases() -> set[str]:
    """Return the set of database names referenced by any model in llm-models.json."""
    from config import LLM_REGISTRY
    return {cfg["database"] for cfg in LLM_REGISTRY.values() if cfg.get("database")}

def _generate_table_map(prefix: str) -> dict:
    """Generate a full table name map using the given prefix."""
    return {
        "collection": f"{prefix}memory",
        "memory_shortterm": f"{prefix}memory_shortterm",
        "memory_longterm": f"{prefix}memory_longterm",
        "chat_summaries": f"{prefix}chat_summaries",
        "goals": f"{prefix}goals",
        "plans": f"{prefix}plans",
        "beliefs": f"{prefix}beliefs",
        "episodic": f"{prefix}episodic",
        "semantic": f"{prefix}semantic",
        "procedural": f"{prefix}procedural",
        "autobiographical": f"{prefix}autobiographical",
        "prospective": f"{prefix}prospective",
        "conditioned": f"{prefix}conditioned",
        "drives": f"{prefix}drives",
        "proc_collection": f"{prefix}procedures",
        "temporal": f"{prefix}temporal",
        "tool_stats": f"{prefix}tool_stats",
        "cognition": f"{prefix}cognition",
        "eidetic": f"{prefix}memory_eidetic",
        "eidetic_collection": f"{prefix}eidetic",
    }

def _get_create_tables_sql(prefix: str) -> str:
    """Return the full SQL to create all cognitive/memory tables with the given prefix."""
    return f"""
-- Core memory tables
CREATE TABLE IF NOT EXISTS `{prefix}memory_shortterm` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    DEFAULT 5 COMMENT '1=low 5=med 10=critical',
    `source`      ENUM('session','user','directive','assistant') DEFAULT 'session',
    `type`        ENUM('context','goal','plan','belief','episodic','semantic','procedural','autobiographical','prospective','conditioned') NOT NULL DEFAULT 'context',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `last_accessed` TIMESTAMP   DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance` DESC),
    KEY `idx_created` (`created_at`),
    KEY `idx_type` (`type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}memory_longterm` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    DEFAULT 5,
    `source`      ENUM('session','user','directive','assistant') DEFAULT 'session',
    `type`        ENUM('context','goal','plan','belief','episodic','semantic','procedural','autobiographical','prospective','conditioned') NOT NULL DEFAULT 'context',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `shortterm_id` INT(11)      DEFAULT NULL COMMENT 'original shortterm row id',
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `aged_at`     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `last_accessed` TIMESTAMP   DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance` DESC),
    KEY `idx_type` (`type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}chat_summaries` (
    `id`           INT(11)      NOT NULL AUTO_INCREMENT,
    `session_id`   VARCHAR(255) DEFAULT NULL,
    `summary`      TEXT         NOT NULL,
    `message_count` INT(11)     DEFAULT 0,
    `model_used`   VARCHAR(255) DEFAULT NULL,
    `created_at`   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_session` (`session_id`),
    KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Cognitive typed memory tables
CREATE TABLE IF NOT EXISTS `{prefix}goals` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `title`       VARCHAR(255)  NOT NULL,
    `description` TEXT          NOT NULL,
    `status`      ENUM('active','done','blocked','abandoned') NOT NULL DEFAULT 'active',
    `importance`  TINYINT(4)    NOT NULL DEFAULT 9,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'user',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `childof`     TEXT          DEFAULT NULL,
    `parentof`    TEXT          DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `attempt_count`  INT        NOT NULL DEFAULT 0,
    `failure_count`  INT        NOT NULL DEFAULT 0,
    `abandon_reason` TEXT       DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_status` (`status`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}plans` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `goal_id`     INT(11)       NOT NULL,
    `step_order`  INT(11)       NOT NULL DEFAULT 1,
    `description` TEXT          NOT NULL,
    `status`      ENUM('pending','in_progress','done','skipped') NOT NULL DEFAULT 'pending',
    `step_type`   ENUM('concept','task') NOT NULL DEFAULT 'concept',
    `parent_id`   INT           DEFAULT NULL,
    `target`      ENUM('model','human','investigate') NOT NULL DEFAULT 'model',
    `executor`    VARCHAR(64)   DEFAULT NULL,
    `tool_call`   TEXT          DEFAULT NULL,
    `result`      TEXT          DEFAULT NULL,
    `approval`    ENUM('proposed','approved','rejected','auto') NOT NULL DEFAULT 'proposed',
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_goal` (`goal_id`),
    KEY `idx_goal_order` (`goal_id`, `step_order`),
    KEY `idx_parent` (`parent_id`),
    KEY `idx_exec_queue` (`step_type`, `status`, `approval`, `target`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}beliefs` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `confidence`  TINYINT(4)    NOT NULL DEFAULT 7,
    `status`      ENUM('active','retracted') NOT NULL DEFAULT 'active',
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}episodic` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    NOT NULL DEFAULT 5,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}semantic` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    NOT NULL DEFAULT 5,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}procedural` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `task_type`   VARCHAR(120)  NOT NULL DEFAULT '',
    `content`     TEXT          NOT NULL,
    `steps`       MEDIUMTEXT    DEFAULT NULL,
    `outcome`     ENUM('unknown','success','partial','failure') NOT NULL DEFAULT 'unknown',
    `run_count`   SMALLINT      NOT NULL DEFAULT 1,
    `success_count` SMALLINT    NOT NULL DEFAULT 0,
    `notes`       TEXT          DEFAULT NULL,
    `last_run_at` TIMESTAMP     DEFAULT NULL,
    `importance`  TINYINT(4)    NOT NULL DEFAULT 5,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'assistant',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance`),
    KEY `idx_task_type` (`task_type`),
    KEY `idx_outcome` (`outcome`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}autobiographical` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `importance`  TINYINT(4)    NOT NULL DEFAULT 7,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'user',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}prospective` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `content`     TEXT          NOT NULL,
    `due_at`      VARCHAR(255)  DEFAULT NULL,
    `status`      ENUM('pending','done','missed') NOT NULL DEFAULT 'pending',
    `importance`  TINYINT(4)    NOT NULL DEFAULT 7,
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'user',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_status` (`status`),
    KEY `idx_importance` (`importance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}conditioned` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `topic`       VARCHAR(255)  NOT NULL,
    `trigger`     TEXT          NOT NULL,
    `reaction`    TEXT          NOT NULL,
    `strength`    TINYINT(4)    NOT NULL DEFAULT 5,
    `status`      ENUM('active','extinguished') NOT NULL DEFAULT 'active',
    `source`      ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'user',
    `session_id`  VARCHAR(255)  DEFAULT NULL,
    `memory_link` TEXT          DEFAULT NULL,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_topic` (`topic`),
    KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}drives` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `name`        VARCHAR(64)   NOT NULL,
    `description` TEXT          NOT NULL,
    `value`       FLOAT         NOT NULL DEFAULT 0.5,
    `baseline`    FLOAT         NOT NULL DEFAULT 0.5,
    `decay_rate`  FLOAT         NOT NULL DEFAULT 0.05,
    `source`      ENUM('system','user','reflection') NOT NULL DEFAULT 'system',
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}temporal` (
    `id`          INT(11)       NOT NULL AUTO_INCREMENT,
    `source`      ENUM('explicit','inferred') NOT NULL DEFAULT 'explicit',
    `query_key`   VARCHAR(255)  NOT NULL,
    `query_params` JSON         NOT NULL,
    `result`      MEDIUMTEXT    NOT NULL,
    `hit_count`   INT           NOT NULL DEFAULT 0,
    `created_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_query_key` (`query_key`),
    KEY `idx_source` (`source`),
    KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}tool_stats` (
    `id`            INT(11)      NOT NULL AUTO_INCREMENT,
    `model`         VARCHAR(100) NOT NULL,
    `tool_name`     VARCHAR(100) NOT NULL,
    `call_count`    INT          NOT NULL DEFAULT 0,
    `success_count` INT          NOT NULL DEFAULT 0,
    `error_count`   INT          NOT NULL DEFAULT 0,
    `first_called`  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    `last_called`   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_model_tool` (`model`, `tool_name`),
    KEY `idx_last_called` (`last_called`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}emotions` (
    `id`            INT(11)      NOT NULL AUTO_INCREMENT,
    `memory_table`  ENUM('shortterm','longterm','episodic') NULL,
    `memory_id`     INT(11)      NULL,
    `angry`         DECIMAL(3,2) DEFAULT 0.00,
    `sad`           DECIMAL(3,2) DEFAULT 0.00,
    `disgusted`     DECIMAL(3,2) DEFAULT 0.00,
    `happy`         DECIMAL(3,2) DEFAULT 0.00,
    `surprised`     DECIMAL(3,2) DEFAULT 0.00,
    `bad`           DECIMAL(3,2) DEFAULT 0.00,
    `fearful`       DECIMAL(3,2) DEFAULT 0.00,
    `emotion_label` VARCHAR(50)  NULL,
    `intensity`     DECIMAL(3,2) DEFAULT 0.50,
    `confidence`    DECIMAL(3,2) DEFAULT 0.50,
    `source`        ENUM('inferred','stated','corrected') DEFAULT 'inferred',
    `context`       TEXT         NULL,
    `created_at`    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_created` (`created_at`),
    KEY `idx_memory` (`memory_table`, `memory_id`),
    KEY `idx_label` (`emotion_label`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `{prefix}cognition` (
    `id`            INT(11)      NOT NULL AUTO_INCREMENT,
    `origin`        ENUM(
                        'reflection',
                        'goal_health',
                        'self_model',
                        'prospective',
                        'tool_log',
                        'tool_failure',
                        'summary'
                    ) NOT NULL,
    `topic`         VARCHAR(255) NOT NULL DEFAULT '',
    `content`       TEXT         NOT NULL,
    `importance`    TINYINT      NOT NULL DEFAULT 5,
    `source`        ENUM('session','user','directive','assistant') NOT NULL DEFAULT 'session',
    `session_id`    VARCHAR(255) NOT NULL DEFAULT '',
    `last_accessed` TIMESTAMP    NULL DEFAULT NULL,
    `created_at`    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_origin`          (`origin`),
    KEY `idx_origin_topic`    (`origin`, `topic`),
    KEY `idx_importance`      (`importance`),
    KEY `idx_created`         (`created_at`),
    KEY `idx_last_accessed`   (`last_accessed`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def _create_database_sync(db_name: str, prefix: str) -> str:
    """Create a new MySQL database and all cognitive tables. Returns status message."""
    conn = mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
    )
    cursor = conn.cursor()
    try:
        # Create the database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
    except Exception as e:
        cursor.close()
        conn.close()
        return f"ERROR creating database: {e}"
    finally:
        cursor.close()
        conn.close()

    # Now connect to the new database and create tables
    conn2 = mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
        database=db_name,
    )
    cursor2 = conn2.cursor()
    sql = _get_create_tables_sql(prefix)
    created = []
    errors = []
    for raw_stmt in sql.split(";"):
        # Strip comment-only lines from top of each chunk (same as apply.py)
        lines = [l for l in raw_stmt.splitlines() if l.strip() and not l.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if not stmt:
            continue
        try:
            cursor2.execute(stmt)
            # Extract table name from CREATE TABLE statement
            m = re.search(r"CREATE TABLE IF NOT EXISTS `(\w+)`", stmt, re.IGNORECASE)
            if m:
                created.append(m.group(1))
        except Exception as e:
            errors.append(str(e))
    conn2.commit()
    cursor2.close()
    conn2.close()

    # Register in _DB_TABLES at runtime
    import datetime as _dt
    table_map = _generate_table_map(prefix)
    _DB_TABLES[db_name] = table_map
    _DB_META[db_name] = {"source": "switch", "created_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}

    # Persist to db-config.json
    _save_db_config()

    # Create matching Qdrant collections (fire-and-forget; non-fatal if Qdrant unavailable)
    qdrant_created = []
    try:
        from plugin_memory_vector_qdrant import get_vector_api
        vec = get_vector_api()
        if vec:
            qdrant_created = vec.ensure_collections(
                collection=table_map["collection"],
                proc_collection=table_map["proc_collection"],
            )
    except Exception as _qe:
        log.warning(f"create_database: Qdrant collection setup skipped: {_qe}")

    parts = [f"Database '{db_name}' ready — {len(created)} tables created (prefix: {prefix})"]
    if qdrant_created:
        parts.append(f"  Qdrant collections created: {', '.join(qdrant_created)}")
    if errors:
        parts.append(f"  Errors: {'; '.join(errors)}")
    return "\n".join(parts)


async def create_database(db_name: str, prefix: str) -> str:
    """Async wrapper for database creation."""
    return await asyncio.to_thread(_create_database_sync, db_name, prefix)


def _delete_database_sync(db_name: str) -> str:
    """Drop a MySQL database entirely. Returns status message."""
    if db_name in _PROTECTED_DBS:
        return f"ERROR: Database '{db_name}' is protected and cannot be deleted."

    # Capture Qdrant collection names before removing from _DB_TABLES
    table_map = _DB_TABLES.get(db_name, {})
    qdrant_collection = table_map.get("collection", "")
    qdrant_proc_collection = table_map.get("proc_collection", "")

    conn = mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
    )
    cursor = conn.cursor()
    try:
        cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
        conn.commit()
    except Exception as e:
        return f"ERROR dropping database: {e}"
    finally:
        cursor.close()
        conn.close()

    # Delete matching Qdrant collections (non-fatal if Qdrant unavailable)
    qdrant_deleted = []
    if qdrant_collection or qdrant_proc_collection:
        try:
            from plugin_memory_vector_qdrant import get_vector_api
            vec = get_vector_api()
            if vec and qdrant_collection:
                qdrant_deleted = vec.delete_collections(
                    collection=qdrant_collection,
                    proc_collection=qdrant_proc_collection or f"{db_name}_procedures",
                )
        except Exception as _qe:
            log.warning(f"delete_database: Qdrant collection deletion skipped: {_qe}")

    # Remove from runtime config
    _DB_TABLES.pop(db_name, None)
    _DB_META.pop(db_name, None)
    _save_db_config()

    parts = [f"Database '{db_name}' deleted."]
    if qdrant_deleted:
        parts.append(f"  Qdrant collections deleted: {', '.join(qdrant_deleted)}")
    return "\n".join(parts)


async def delete_database(db_name: str) -> str:
    """Async wrapper for database deletion."""
    return await asyncio.to_thread(_delete_database_sync, db_name)


def database_exists(db_name: str) -> bool:
    """Check if a database exists in MySQL."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASS"),
        )
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES LIKE %s", (db_name,))
        exists = cursor.fetchone() is not None
        cursor.close()
        conn.close()
        return exists
    except Exception:
        return False


def _save_db_config() -> None:
    """Persist the current _DB_TABLES and _PROTECTED_DBS to db-config.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db-config.json")
    # Filter out non-serializable values (auto_enrich lists are fine)
    cfg = {
        "tables": _DB_TABLES,
        "protected_databases": _PROTECTED_DBS,
        "meta": _DB_META,
    }
    try:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        log.info(f"db-config.json saved ({len(_DB_TABLES)} databases)")
    except Exception as e:
        log.warning(f"Failed to save db-config.json: {e}")


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