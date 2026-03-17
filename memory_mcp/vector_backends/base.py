"""
Abstract base class for vector storage backends.

Every vector backend must implement this interface so that
RobustMemorySystem can swap between ChromaDB, pgvector, etc.
without any changes to the core memory logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VectorQueryResult:
    """Single result row returned by a vector similarity query."""

    id: str
    distance: float  # raw distance (cosine); caller converts to similarity
    document: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorBackend(ABC):
    """
    Interface that every vector storage engine must implement.

    Methods mirror the ChromaDB collection API surface that
    RobustMemorySystem actually uses, abstracted to plain Python types.
    """

    # ── lifecycle ───────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """
        Create / open the underlying storage and ensure the collection
        (or table) exists.  Called once during system startup.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources (connections, file handles, etc.)."""

    # ── write ───────────────────────────────────────────────────

    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Insert one or more vectors with their documents and metadata.
        Raises on failure so the caller can roll back SQLite.
        """

    @abstractmethod
    def update(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Update existing vectors (used when memory content changes).
        Raises on failure.
        """

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""

    # ── read ────────────────────────────────────────────────────

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> List[VectorQueryResult]:
        """
        Return the *n_results* nearest neighbours for *query_embedding*,
        ordered by ascending distance (closest first).
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored vectors."""

    @abstractmethod
    def get(
        self,
        ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch vectors by IDs (or all, up to *limit*).

        Returns a dict with keys: ids, documents, metadatas,
        and optionally embeddings.
        """

    # ── maintenance ─────────────────────────────────────────────

    @abstractmethod
    def reset_collection(self) -> None:
        """
        Drop and recreate the collection / table.
        Used by rebuild_vector_index().
        """

    @abstractmethod
    def persist(self) -> None:
        """
        Flush writes to durable storage if the backend buffers.
        No-op for backends that auto-persist (e.g. pgvector).
        """

    # ── info ────────────────────────────────────────────────────

    @abstractmethod
    def storage_size_bytes(self) -> int:
        """Return approximate on-disk size in bytes."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name, e.g. 'chromadb' or 'pgvector'."""
