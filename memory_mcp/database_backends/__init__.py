"""
Database backend abstraction for the long-term memory system.

Provides a pluggable interface for structured metadata storage:
  - sqlite (default): Embedded SQLite with WAL mode
  - postgres: PostgreSQL (shared with pgvector backend)
"""

from .base import DatabaseBackend, DatabaseRow, DatabaseCursor
from .sqlite import SQLiteDatabase

__all__ = ["DatabaseBackend", "DatabaseRow", "DatabaseCursor", "SQLiteDatabase"]
