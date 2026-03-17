"""
ChromaDB vector storage backend.

This is the default backend — it wraps the existing ChromaDB integration
that was previously inline in memory_system.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .base import VectorBackend, VectorQueryResult
from ..config import CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)


class ChromaBackend(VectorBackend):
    """
    Embedded ChromaDB vector store with on-disk persistence.

    Args:
        db_folder: Directory that contains (or will contain) the ``chroma_db/``
                   sub-directory.  Typically ``<data_folder>/memory_db``.
    """

    def __init__(self, db_folder: Path) -> None:
        self._db_folder = Path(db_folder)
        self._chroma_path = str(self._db_folder / "chroma_db")
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    # ── lifecycle ───────────────────────────────────────────────

    def initialize(self) -> None:
        self._client = chromadb.PersistentClient(
            path=self._chroma_path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={
                "description": "Long-term memory for AI companion",
                "hnsw:space": "cosine",
            },
            embedding_function=None,  # embeddings passed manually
        )
        logger.info("ChromaDB backend initialized at %s", self._chroma_path)

    def close(self) -> None:
        # ChromaDB PersistentClient doesn't need explicit close — just
        # drop references so the GC can clean up.
        self._collection = None
        self._client = None

    # ── write ───────────────────────────────────────────────────

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._try_persist()

    def update(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self._collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._try_persist()

    def delete(self, ids: List[str]) -> None:
        self._collection.delete(ids=ids)

    # ── read ────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> List[VectorQueryResult]:
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if not raw["ids"][0]:
            return []

        results: List[VectorQueryResult] = []
        for i, mid in enumerate(raw["ids"][0]):
            results.append(
                VectorQueryResult(
                    id=mid,
                    distance=raw["distances"][0][i],
                    document=raw["documents"][0][i] if raw.get("documents") else None,
                    metadata=raw["metadatas"][0][i] if raw.get("metadatas") else None,
                )
            )
        return results

    def count(self) -> int:
        return self._collection.count()

    def get(
        self,
        ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        kwargs: Dict[str, Any] = {"include": include}
        if ids is not None:
            kwargs["ids"] = ids
        if limit is not None:
            kwargs["limit"] = limit

        raw = self._collection.get(**kwargs)
        return {
            "ids": raw.get("ids", []),
            "documents": raw.get("documents", []),
            "metadatas": raw.get("metadatas", []),
            "embeddings": raw.get("embeddings", []) if include_embeddings else [],
        }

    # ── maintenance ─────────────────────────────────────────────

    def reset_collection(self) -> None:
        """Drop and recreate the ChromaDB collection."""
        try:
            self._client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception as e:
            logger.warning("ChromaDB drop collection warning: %s", e)

        self._collection = None
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={
                "description": "Long-term memory for AI companion",
                "hnsw:space": "cosine",
            },
            embedding_function=None,
        )

    def persist(self) -> None:
        self._try_persist()

    # ── info ────────────────────────────────────────────────────

    def storage_size_bytes(self) -> int:
        chroma_dir = self._db_folder / "chroma_db"
        if not chroma_dir.exists():
            return 0
        return sum(f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file())

    @property
    def backend_name(self) -> str:
        return "chromadb"

    # ── internal ────────────────────────────────────────────────

    def _try_persist(self) -> None:
        """Call persist() if the client version supports it."""
        try:
            if hasattr(self._client, "persist"):
                self._client.persist()
        except Exception as e:
            logger.warning("ChromaDB persist warning: %s", e)
