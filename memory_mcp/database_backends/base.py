"""
Abstract base class for database (metadata) backends.

Every database backend must implement this interface so that
RobustMemorySystem can swap between SQLite and PostgreSQL
without changing the core memory logic.

The interface mirrors the subset of the Python DB-API 2.0 that
memory_system.py actually uses, plus lifecycle helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class DatabaseRow:
    """Dict-like row wrapper compatible with sqlite3.Row access patterns.

    Supports both ``row["column"]`` and ``row[0]`` access styles.
    """

    def __init__(self, columns: List[str], values: tuple) -> None:
        self._columns = columns
        self._values = values
        self._map = dict(zip(columns, values))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return self._map[key]

    def __contains__(self, key):
        return key in self._map

    def get(self, key, default=None):
        return self._map.get(key, default)

    def keys(self):
        return self._map.keys()

    def values(self):
        return self._map.values()

    def items(self):
        return self._map.items()

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"DatabaseRow({self._map})"


class DatabaseCursor:
    """Minimal cursor interface returned by DatabaseBackend.execute()."""

    def __init__(self, rows: List[DatabaseRow]) -> None:
        self._rows = rows
        self._pos = 0

    def fetchone(self) -> Optional[DatabaseRow]:
        if self._pos < len(self._rows):
            row = self._rows[self._pos]
            self._pos += 1
            return row
        return None

    def fetchall(self) -> List[DatabaseRow]:
        remaining = self._rows[self._pos :]
        self._pos = len(self._rows)
        return remaining


class DatabaseBackend(ABC):
    """
    Interface for structured metadata storage (memories table, memory_stats).

    Implementations must translate SQL for their engine (e.g. ``?`` → ``%s``
    for Postgres) inside ``execute()``.
    """

    # ── lifecycle ───────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """Create/open storage, ensure tables and indexes exist."""

    @abstractmethod
    def close(self) -> None:
        """Release connections and resources."""

    # ── SQL execution ───────────────────────────────────────────

    @abstractmethod
    def execute(self, sql: str, params: Sequence = ()) -> DatabaseCursor:
        """Execute a single SQL statement with optional params.

        The implementation must handle placeholder translation
        (``?`` in the source SQL → engine-native placeholder).
        """

    @abstractmethod
    def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements separated by ``;``.

        Used for schema creation (CREATE TABLE IF NOT EXISTS ...).
        """

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""

    @abstractmethod
    def rollback(self) -> None:
        """Roll back the current transaction."""

    # ── maintenance ─────────────────────────────────────────────

    @abstractmethod
    def checkpoint(self) -> None:
        """Flush buffered writes to durable storage.

        For SQLite this is ``PRAGMA wal_checkpoint(TRUNCATE)``.
        For Postgres this is a no-op (WAL is automatic).
        """

    @abstractmethod
    def integrity_check(self) -> Optional[str]:
        """Run a database integrity check.

        Returns None if OK, or an error string if problems found.
        """

    # ── info ────────────────────────────────────────────────────

    @abstractmethod
    def storage_size_bytes(self) -> int:
        """Return approximate on-disk size of the metadata store."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name, e.g. 'sqlite' or 'postgres'."""

    @property
    @abstractmethod
    def is_postgres(self) -> bool:
        """True if this is a Postgres backend (affects backup strategy, etc.)."""
