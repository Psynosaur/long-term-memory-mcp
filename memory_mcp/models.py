"""
Data models for the long-term memory system.

Contains all dataclass definitions for memory records, search results, and operation results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class MemoryRecord:
    """
    Structured memory record.

    Attributes:
        id (str)
        title (str)
        content (str)
        timestamp (datetime)
        tags (List[str])
        importance (int): 1–10 scale
        memory_type (str): conversation, fact, preference, event, etc.
        metadata (Dict[str, Any])
        shared_with (List[str]): Peer UUIDs this memory is visible to.
            []        = private
            ["*"]     = broadcast to all discovered peers
            ["uuid1"] = specific peer(s) only
    """

    id: str
    title: str
    content: str
    timestamp: datetime
    tags: List[str]
    importance: int
    memory_type: str
    metadata: Dict[str, Any]
    shared_with: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """
    Search result with relevance score.

    Attributes:
        record (MemoryRecord): The matched memory record.
        relevance_score (float): Similarity or match score.
        match_type (str): Type of match (e.g., "semantic", "exact", "metadata").
    """

    record: MemoryRecord
    relevance_score: float
    match_type: str  # semantic, exact, metadata


@dataclass
class Result:
    """
    Standard result container for memory operations.

    Attributes:
        success (bool): Whether the operation succeeded.
        reason (str, optional): Explanation when the operation fails.
        data (list of dict, optional): Operation-specific data, such as
            memory objects, statistics, or search results.
    """

    success: bool
    reason: Optional[str] = None
    data: Optional[List[Dict]] = None
