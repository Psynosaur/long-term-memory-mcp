"""
SQLite database backend.

Thin wrapper around sqlite3.Connection that implements the
DatabaseBackend interface.  This is the default backend — it
preserves the exact behaviour of the original inline SQLite code
in memory_system.py.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

from .base import DatabaseBackend, DatabaseCursor, DatabaseRow

logger = logging.getLogger(__name__)


class SQLiteDatabase(DatabaseBackend):
    """
    Embedded SQLite database with WAL journaling.

    Args:
        db_path: Path to the ``memories.db`` file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── lifecycle ───────────────────────────────────────────────

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row

        # WAL mode + safety settings
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA wal_autocheckpoint=500;")
        self._conn.execute("PRAGMA synchronous=FULL;")
        self._conn.commit()

        logger.info("SQLite database initialized at %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.commit()
            except Exception:
                pass
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception as e:
                logger.warning("WAL checkpoint on close failed: %s", e)
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── SQL execution ───────────────────────────────────────────

    def execute(self, sql: str, params: Sequence = ()) -> DatabaseCursor:
        raw_cursor = self._conn.execute(sql, params)

        # sqlite3.Row already supports dict-like access, but we wrap
        # it in DatabaseRow for a consistent interface across backends.
        columns = (
            [desc[0] for desc in raw_cursor.description]
            if raw_cursor.description
            else []
        )
        rows = [DatabaseRow(columns, tuple(row)) for row in raw_cursor.fetchall()]
        return DatabaseCursor(rows)

    def executescript(self, sql: str) -> None:
        self._conn.executescript(sql)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    # ── maintenance ─────────────────────────────────────────────

    def checkpoint(self) -> None:
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self._conn.commit()
        except Exception as e:
            logger.warning("WAL checkpoint failed: %s", e)

    def integrity_check(self) -> Optional[str]:
        try:
            result = self._conn.execute("PRAGMA integrity_check;").fetchone()
            if result and result[0] != "ok":
                return result[0]
            return None
        except Exception as e:
            return str(e)

    # ── info ────────────────────────────────────────────────────

    def storage_size_bytes(self) -> int:
        if not self._db_path.exists():
            return 0
        size = self._db_path.stat().st_size
        # Include WAL file if present
        wal = Path(str(self._db_path) + "-wal")
        if wal.exists():
            size += wal.stat().st_size
        return size

    @property
    def backend_name(self) -> str:
        return "sqlite"

    @property
    def is_postgres(self) -> bool:
        return False

    @property
    def db_path(self) -> Path:
        """Expose the database file path for backup operations."""
        return self._db_path
