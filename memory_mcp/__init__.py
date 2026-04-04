"""
Long-term memory MCP package.

A robust, persistent memory system for AI companions.
Supports pluggable vector backends (ChromaDB default, pgvector optional)
and pluggable database backends (SQLite default, PostgreSQL optional).
"""

from .config import (
    DATA_FOLDER,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_CONFIG,
    EMBEDDING_MODEL_PRESETS,
)
from .models import MemoryRecord, SearchResult, Result
from .memory_system import RobustMemorySystem
from .mcp_tools import register_tools, jsonify_result
from .vector_backends.base import VectorBackend
from .database_backends.base import DatabaseBackend
from .audit import AuditLogger

__all__ = [
    "DATA_FOLDER",
    "CHROMA_COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_CONFIG",
    "EMBEDDING_MODEL_PRESETS",
    "MemoryRecord",
    "SearchResult",
    "Result",
    "RobustMemorySystem",
    "register_tools",
    "jsonify_result",
    "VectorBackend",
    "DatabaseBackend",
    "AuditLogger",
]
