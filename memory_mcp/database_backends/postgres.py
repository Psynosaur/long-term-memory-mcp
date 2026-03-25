"""
PostgreSQL database backend.

Implements the DatabaseBackend interface against a PostgreSQL
connection (shared with the pgvector vector backend).

Handles SQL dialect translation so that memory_system.py can use
the same SQLite-flavoured SQL for both backends:
  - ``?`` placeholders → ``%s``
  - ``INSERT OR REPLACE`` → ``INSERT ... ON CONFLICT DO UPDATE``
  - PRAGMAs → no-ops
  - ``CURRENT_TIMESTAMP`` → ``NOW()``
  - ``LIKE`` for JSON tag search stays the same (works in Postgres too)

Requirements:
    pip install psycopg[binary]
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional, Sequence

from .base import DatabaseBackend, DatabaseCursor, DatabaseRow

logger = logging.getLogger(__name__)

# Regex to find ``?`` placeholders.
# NOTE: This is safe because all SQL in memory_system.py uses ``?``
# exclusively as parameter placeholders, never inside string literals.
# If that changes, switch to a proper SQL parser (e.g., sqlglot).
_QMARK_RE = re.compile(r"\?")

# Known primary keys for each table.  This avoids inferring PK from
# column order, which would be fragile if columns are reordered.
_TABLE_PK = {
    "memories": "id",
    "memory_stats": "key",
}


def _translate_sql(sql: str) -> Optional[str]:
    """Translate SQLite SQL dialect to PostgreSQL.

    Only handles the subset of SQLite-isms used by memory_system.py:
      - PRAGMA statements → skipped (returned as None)
      - INSERT OR REPLACE → INSERT ... ON CONFLICT (pk) DO UPDATE
      - ? placeholders → %s

    Limitations (by design — the input SQL is tightly controlled):
      - Does not handle ? inside string literals or comments
      - Does not handle schema-qualified table names
      - Does not handle multi-statement INSERT OR REPLACE

    Returns None for statements that should be skipped (PRAGMAs).
    """
    # 1. Skip PRAGMAs entirely (handled by lifecycle methods)
    stripped = sql.strip()
    if stripped.upper().startswith("PRAGMA"):
        return None  # signal to caller: skip this statement

    # 2. INSERT OR REPLACE → INSERT ... ON CONFLICT
    #    Pattern: INSERT OR REPLACE INTO <table> (col, ...) VALUES (...)
    if "INSERT OR REPLACE" in sql.upper():
        sql = re.sub(
            r"INSERT\s+OR\s+REPLACE\s+INTO",
            "INSERT INTO",
            sql,
            flags=re.IGNORECASE,
        )
        # Extract the table name to look up the known PK
        table_match = re.search(r"INSERT\s+INTO\s+(\w+)\s*\(", sql, re.IGNORECASE)
        if not table_match:
            raise ValueError(
                f"Cannot parse INSERT OR REPLACE statement for Postgres translation: {sql[:100]}"
            )

        table_name = table_match.group(1).lower()
        pk_col = _TABLE_PK.get(table_name)
        if pk_col is None:
            raise ValueError(
                f"No known primary key for table '{table_name}'. "
                f"Add it to _TABLE_PK in postgres.py. SQL: {sql[:100]}"
            )

        # Extract column names for the ON CONFLICT SET clause
        cols_match = re.search(r"INSERT\s+INTO\s+\w+\s*\(([^)]+)\)", sql, re.IGNORECASE)
        if not cols_match:
            raise ValueError(
                f"Cannot extract column list from INSERT statement: {sql[:100]}"
            )

        cols = [c.strip() for c in cols_match.group(1).split(",")]
        non_pk = [c for c in cols if c != pk_col]
        sql = sql.rstrip().rstrip(";")
        if non_pk:
            set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk)
            sql += f" ON CONFLICT ({pk_col}) DO UPDATE SET {set_clause}"
        else:
            sql += f" ON CONFLICT ({pk_col}) DO NOTHING"

    # 3. CURRENT_TIMESTAMP works in Postgres too — no change needed

    # 4. Replace ? placeholders with %s
    sql = _QMARK_RE.sub("%s", sql)

    return sql


class PostgresDatabase(DatabaseBackend):
    """
    PostgreSQL metadata store.

    Creates the ``memories`` and ``memory_stats`` tables in the same
    database used by the pgvector backend.

    Args:
        host, port, database, user, password: Connection parameters.
            Defaults mirror the pgvector backend defaults and
            fall back to PG* environment variables.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self._host = host or os.environ.get("PGHOST", "localhost")
        self._port = port or int(os.environ.get("PGPORT", "5433"))
        self._database = database or os.environ.get("PGDATABASE", "memories")
        self._user = user or os.environ.get("PGUSER", "memory_user")
        self._password = password or os.environ.get("PGPASSWORD", "memory_pass")
        self._conn = None

    # ── lifecycle ───────────────────────────────────────────────

    def initialize(self) -> None:
        try:
            import psycopg
        except ImportError as e:
            raise ImportError(
                "Postgres database backend requires 'psycopg[binary]'. "
                "Install with:  pip install 'psycopg[binary]'"
            ) from e

        conninfo = (
            f"host={self._host} port={self._port} dbname={self._database} "
            f"user={self._user} password={self._password}"
        )
        self._conn = psycopg.connect(conninfo, autocommit=False)

        # Create schema
        self._create_schema()

        logger.info(
            "Postgres database initialized: %s@%s:%s/%s",
            self._user,
            self._host,
            self._port,
            self._database,
        )

    def _create_schema(self) -> None:
        """Create the memories and memory_stats tables if they don't exist."""
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT,
                    importance INTEGER DEFAULT 5,
                    memory_type TEXT DEFAULT 'conversation',
                    metadata TEXT,
                    content_hash TEXT,
                    created_at TEXT DEFAULT (NOW()::text),
                    updated_at TEXT DEFAULT (NOW()::text),
                    last_accessed TEXT DEFAULT (NOW()::text),
                    token_count INTEGER DEFAULT 0,
                    shared INTEGER DEFAULT 0
                );
            """)

            # Indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed);"
            )

            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT (NOW()::text)
                );
            """)

        self._conn.commit()

        # Column migrations — safe to re-run (IF NOT EXISTS)
        with self._conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE memories ADD COLUMN IF NOT EXISTS shared INTEGER DEFAULT 0"
            )
        self._conn.commit()

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            try:
                self._conn.commit()
            except Exception:
                pass
            self._conn.close()
            logger.info("Postgres database connection closed")
        self._conn = None

    # ── SQL execution ───────────────────────────────────────────

    def execute(self, sql: str, params: Sequence = ()) -> DatabaseCursor:
        translated = _translate_sql(sql)

        # PRAGMA or skip → return empty cursor
        if translated is None:
            return DatabaseCursor([])

        with self._conn.cursor() as cur:
            cur.execute(translated, params if params else None)

            # If this was a SELECT/RETURNING, fetch results
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = [DatabaseRow(columns, row) for row in cur.fetchall()]
                return DatabaseCursor(rows)

            return DatabaseCursor([])

    def executescript(self, sql: str) -> None:
        """Execute a multi-statement SQL script.

        Translates each statement individually and skips PRAGMAs.
        """
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        with self._conn.cursor() as cur:
            for stmt in statements:
                translated = _translate_sql(stmt)
                if translated is not None and translated.strip():
                    cur.execute(translated)
        self._conn.commit()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    # ── maintenance ─────────────────────────────────────────────

    def checkpoint(self) -> None:
        # Postgres WAL is automatic — no-op
        pass

    def integrity_check(self) -> Optional[str]:
        """Run basic Postgres health check."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return None
        except Exception as e:
            return str(e)

    # ── info ────────────────────────────────────────────────────

    def storage_size_bytes(self) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_total_relation_size('memories') + "
                    "pg_total_relation_size('memory_stats')"
                )
                row = cur.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    @property
    def backend_name(self) -> str:
        return "postgres"

    @property
    def is_postgres(self) -> bool:
        return True
