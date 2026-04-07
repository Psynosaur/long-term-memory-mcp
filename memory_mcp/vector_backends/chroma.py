"""
ChromaDB vector storage backend.

This is the default backend — it wraps the existing ChromaDB integration
that was previously inline in memory_system.py.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
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

    # ── read-only / lock-free access ─────────────────────────────

    @classmethod
    def get_vectors_readonly(
        cls,
        db_folder: Path,
    ) -> Dict[str, Any]:
        """Read vectors directly from ChromaDB's SQLite WAL without opening a
        PersistentClient.

        This avoids Windows mandatory file locking that causes the visualizer
        to stall indefinitely when the MCP server already holds the DB open via
        its own PersistentClient.

        The approach:
          - Opens ``chroma.sqlite3`` with ``?mode=ro`` (read-only, shared lock
            only — compatible with an existing writer).
          - Replays ``embeddings_queue``: takes the latest ``seq_id`` per ``id``
            and excludes operation=2 (delete).  This gives the current live set
            of embeddings without needing the HNSW index.

        Returns a dict with the same shape as :meth:`get` with
        ``include_embeddings=True``:
            ``{"ids": [...], "documents": [...], "metadatas": [...],
               "embeddings": [[float, ...], ...]}``

        Raises:
            FileNotFoundError: if ``chroma.sqlite3`` does not exist at the
                expected path.
            sqlite3.OperationalError: if the file cannot be opened (e.g.
                exclusive lock held by another process at the OS level).
        """
        chroma_sqlite = Path(db_folder) / "chroma_db" / "chroma.sqlite3"
        if not chroma_sqlite.exists():
            raise FileNotFoundError(f"ChromaDB SQLite file not found: {chroma_sqlite}")

        uri = "file:" + str(chroma_sqlite).replace("\\", "/") + "?mode=ro"
        con = sqlite3.connect(uri, uri=True, check_same_thread=False)
        try:
            sql = """
                SELECT q.id, q.vector, q.encoding, q.metadata
                FROM embeddings_queue q
                INNER JOIN (
                    SELECT id, MAX(seq_id) AS max_seq
                    FROM embeddings_queue
                    GROUP BY id
                ) latest ON q.id = latest.id AND q.seq_id = latest.max_seq
                WHERE q.operation != 2
                ORDER BY q.seq_id
            """
            rows = con.execute(sql).fetchall()
        finally:
            con.close()

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for mem_id, vec_bytes, encoding, meta_json in rows:
            ids.append(mem_id)

            # Decode vector bytes
            if vec_bytes and encoding and encoding.upper() == "FLOAT32":
                n = len(vec_bytes) // 4
                embedding = list(struct.unpack(f"{n}f", vec_bytes))
            else:
                embedding = []
            embeddings.append(embedding)

            # Decode metadata JSON
            meta: Dict[str, Any] = {}
            document = ""
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                    # ChromaDB stores the document text under "chroma:document"
                    document = meta.pop("chroma:document", "")
                except (json.JSONDecodeError, TypeError):
                    pass
            metadatas.append(meta)
            documents.append(document)

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }
