#!/usr/bin/env python3
"""
Apply a SQL migration to all configured databases, substituting {prefix}.
Usage: python migrations/apply.py 001_memory_types.sql [--dry-run]
"""
import sys, os, json, re, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mysql.connector

DB_CONF_PATH = os.path.join(os.path.dirname(__file__), '..', 'db-config.json')

# DB connection credentials — reads from environment or falls back to defaults
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('MYSQL_USER', os.environ.get('DB_USER', 'root'))
DB_PASS = os.environ.get('MYSQL_PASS', os.environ.get('DB_PASS', ''))

# Map db-config key → actual MySQL database name
DB_NAME_MAP = {
    'mymcp': 'mymcp',
    'qwen':  'qwen',
    'test':  'test',
    'default':    'default',
    'testchat001': 'testchat001',
    'leegeneral': 'leegeneral',
    'gedmath':    'gedmath',
    'gedreading': 'gedreading',
    'gedwriting': 'gedwriting',
    'gedscience': 'gedscience',
    'gedsocial':  'gedsocial',
}

# Map db-config key → table prefix
def get_prefix(tables: dict) -> str:
    """Derive prefix from memory_shortterm table name."""
    st = tables.get('memory_shortterm', '')
    if st.endswith('memory_shortterm'):
        return st[: -len('memory_shortterm')]
    return ''


def split_statements(sql: str) -> list[str]:
    """Split SQL on semicolons, strip leading comments, skip empty chunks."""
    statements = []
    for stmt in sql.split(';'):
        # Strip comment-only lines from the top of each chunk
        lines = [l for l in stmt.splitlines() if l.strip() and not l.strip().startswith('--')]
        body = '\n'.join(lines).strip()
        if body:
            statements.append(body)
    return statements


def apply_migration(sql_path: str, dry_run: bool = False):
    with open(DB_CONF_PATH) as f:
        cfg = json.load(f)

    with open(sql_path) as f:
        template = f.read()

    for db_key, tables in cfg['tables'].items():
        db_name = DB_NAME_MAP.get(db_key)
        if not db_name:
            print(f"[SKIP] Unknown db key: {db_key}")
            continue

        prefix = get_prefix(tables)
        if not prefix:
            print(f"[SKIP] Could not determine prefix for {db_key}")
            continue

        sql = template.replace('{prefix}', prefix)
        statements = split_statements(sql)

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Applying to {db_key} (db={db_name}, prefix={prefix})")

        if dry_run:
            for i, stmt in enumerate(statements, 1):
                first_line = stmt.split('\n')[0][:80]
                print(f"  [{i}] {first_line}")
            continue

        try:
            conn = mysql.connector.connect(
                host=DB_HOST, user=DB_USER, password=DB_PASS, database=db_name
            )
            cur = conn.cursor()
            for stmt in statements:
                try:
                    cur.execute(stmt)
                    print(f"  OK: {stmt.split(chr(10))[0][:80]}")
                except mysql.connector.Error as e:
                    print(f"  ERR: {e}")
                    print(f"       Statement: {stmt[:120]}")
            conn.commit()
            cur.close()
            conn.close()
            print(f"  Done: {db_key}")
        except mysql.connector.Error as e:
            print(f"  CONNECT ERROR for {db_key}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply SQL migration to all databases')
    parser.add_argument('sql_file', help='Path to SQL migration file')
    parser.add_argument('--dry-run', action='store_true', help='Print statements without executing')
    args = parser.parse_args()

    sql_path = args.sql_file
    if not os.path.isabs(sql_path):
        sql_path = os.path.join(os.path.dirname(__file__), sql_path)

    apply_migration(sql_path, dry_run=args.dry_run)
