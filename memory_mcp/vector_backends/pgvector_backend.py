"""
pgvector (PostgreSQL) vector storage backend.

Stores embeddings in a PostgreSQL table with the pgvector extension.
All vectors and metadata live in a single ``memory_vectors`` table
alongside the existing SQLite metadata store.

Requirements:
    pip install psycopg[binary] pgvector

Connection parameters can be supplied explicitly or via the standard
PG* environment variables (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import VectorBackend, VectorQueryResult

logger = logging.getLogger(__name__)

# Table / index names
_TABLE = "memory_vectors"
_INDEX = "memory_vectors_embedding_idx"


class PgvectorBackend(VectorBackend):
    """
    PostgreSQL + pgvector vector store.

    Args:
        host: Postgres host (default from PGHOST or ``localhost``).
        port: Postgres port (default from PGPORT or ``5433``).
        database: Database name (default from PGDATABASE or ``memories``).
        user: Username (default from PGUSER or ``memory_user``).
        password: Password (default from PGPASSWORD or ``memory_pass``).
        dimensions: Embedding dimensions (must match the embedding model).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dimensions: int = 384,
    ) -> None:
        self._host = host or os.environ.get("PGHOST", "localhost")
        self._port = port or int(os.environ.get("PGPORT", "5433"))
        self._database = database or os.environ.get("PGDATABASE", "memories")
        self._user = user or os.environ.get("PGUSER", "memory_user")
        self._password = password or os.environ.get("PGPASSWORD", "memory_pass")
        self._dimensions = dimensions
        self._conn = None  # psycopg connection

    # ── lifecycle ───────────────────────────────────────────────

    def initialize(self) -> None:
        # Late imports so users who don't use pgvector don't need
        # psycopg / pgvector installed at all.
        try:
            import psycopg  # noqa: F811
            from pgvector.psycopg import register_vector  # noqa: F811
        except ImportError as e:
            raise ImportError(
                "pgvector backend requires 'psycopg[binary]' and 'pgvector' packages. "
                "Install with:  pip install 'psycopg[binary]' pgvector"
            ) from e

        conninfo = (
            f"host={self._host} port={self._port} dbname={self._database} "
            f"user={self._user} password={self._password}"
        )
        self._conn = psycopg.connect(conninfo, autocommit=True)

        # Enable the pgvector extension (idempotent)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Register pgvector types with psycopg so we can use Python lists
        register_vector(self._conn)

        # Create the vectors table if it doesn't exist
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                id          TEXT PRIMARY KEY,
                embedding   vector({self._dimensions}),
                document    TEXT,
                metadata    JSONB DEFAULT '{{}}'::jsonb
            );
        """)

        # Create HNSW index for cosine distance (idempotent)
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {_INDEX}
            ON {_TABLE}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        logger.info(
            "pgvector backend initialized: %s@%s:%s/%s (dims=%d)",
            self._user,
            self._host,
            self._port,
            self._database,
            self._dimensions,
        )

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("pgvector connection closed")
        self._conn = None

    # ── write ───────────────────────────────────────────────────

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        with self._conn.cursor() as cur:
            for mid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                cur.execute(
                    f"""
                    INSERT INTO {_TABLE} (id, embedding, document, metadata)
                    VALUES (%s, %s::vector, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        document  = EXCLUDED.document,
                        metadata  = EXCLUDED.metadata
                    """,
                    (mid, str(emb), doc, json.dumps(meta)),
                )

    def update(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        # Upsert semantics — same as add with ON CONFLICT
        self.add(ids, embeddings, documents, metadatas)

    def delete(self, ids: List[str]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {_TABLE} WHERE id = ANY(%s)",
                (ids,),
            )

    # ── read ────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> List[VectorQueryResult]:
        with self._conn.cursor() as cur:
            # Use cosine distance operator (<=>)
            # Cast to ::vector explicitly so psycopg sends the right type
            cur.execute(
                f"""
                SELECT id, embedding <=> %s::vector AS distance, document, metadata
                FROM {_TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (str(query_embedding), str(query_embedding), n_results),
            )
            rows = cur.fetchall()

        results: List[VectorQueryResult] = []
        for row in rows:
            meta = row[3] if row[3] else {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            results.append(
                VectorQueryResult(
                    id=row[0],
                    distance=float(row[1]),
                    document=row[2],
                    metadata=meta,
                )
            )
        return results

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {_TABLE}")
            return cur.fetchone()[0]

    def get(
        self,
        ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        emb_col = ", embedding" if include_embeddings else ""
        query = f"SELECT id, document, metadata{emb_col} FROM {_TABLE}"
        params: list = []

        if ids is not None:
            query += " WHERE id = ANY(%s)"
            params.append(ids)

        query += " ORDER BY id"

        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        result: Dict[str, Any] = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }
        for row in rows:
            result["ids"].append(row[0])
            result["documents"].append(row[1])
            meta = row[2] if row[2] else {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            result["metadatas"].append(meta)
            if include_embeddings:
                # pgvector returns numpy array or list depending on driver
                emb = row[3]
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                result["embeddings"].append(emb)

        return result

    # ── maintenance ─────────────────────────────────────────────

    def reset_collection(self) -> None:
        """Truncate the vectors table (fast, keeps structure)."""
        self._conn.execute(f"TRUNCATE TABLE {_TABLE};")
        logger.info("pgvector table truncated")

    def persist(self) -> None:
        # Postgres auto-persists (WAL); nothing to do.
        pass

    # ── info ────────────────────────────────────────────────────

    def storage_size_bytes(self) -> int:
        """Approximate table + index size from pg_total_relation_size."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_total_relation_size(%s)",
                    (_TABLE,),
                )
                row = cur.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    @property
    def backend_name(self) -> str:
        return "pgvector"
