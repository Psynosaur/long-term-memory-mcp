"""
Long-term memory MCP package.

A robust, persistent memory system for AI companions.
"""

from .config import DATA_FOLDER
from .models import MemoryRecord, SearchResult, Result
from .memory_system import RobustMemorySystem
from .mcp_tools import register_tools, jsonify_result

__all__ = [
    "DATA_FOLDER",
    "MemoryRecord",
    "SearchResult",
    "Result",
    "RobustMemorySystem",
    "register_tools",
    "jsonify_result",
]
