"""
Long-term memory MCP package.

A robust, persistent memory system for AI companions.
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
]
