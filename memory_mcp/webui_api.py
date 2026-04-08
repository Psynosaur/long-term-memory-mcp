"""
WebUI REST API for the Long-Term Memory system.

Exposes all memory manager features over HTTP so the browser-based
WebUI can perform CRUD, search, stats, backup, export, migration,
comparison, and vector inspection without a second Python process.

The FastAPI app is created via create_app() and intended to be mounted
in a daemon thread alongside the existing FastMCP server so that both
share the same RobustMemorySystem instance (and the same embedding model
loaded only once).

All routes are under /api/v1/ prefix.

Usage (from server.py):
    from memory_mcp.webui_api import create_app
    webui_app = create_app(memory_system, identity, sharing_mgr)
    # Run in a daemon thread on a separate port (default 8666)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── FastAPI imports ────────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, HTTPException, Query, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "WebUI requires fastapi. Install with: pip install fastapi"
    ) from _e


# ── Pydantic request / response models ────────────────────────────────────────


class MemoryCreate(BaseModel):
    title: str
    content: str
    memory_type: str = "conversation"
    importance: int = Field(default=5, ge=1, le=10)
    tags: List[str] = []
    shared_with: List[str] = []
    file_paths: List[str] = []


class MemoryUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    memory_type: Optional[str] = None
    importance: Optional[int] = Field(default=None, ge=1, le=10)
    tags: Optional[List[str]] = None
    shared_with: Optional[List[str]] = None


class MigrateRequest(BaseModel):
    direction: str  # "sqlite_to_sqlite" | "sqlite_to_pg" | "pg_to_sqlite"
    # Used for sqlite_to_sqlite direction
    source_db_path: Optional[str] = None
    source_chroma_path: Optional[str] = None
    memory_ids: Optional[List[str]] = None
    skip_duplicates: bool = True
    migrate_vectors: bool = True
    # Used for pg directions
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    pg_database: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None


class MigratePreviewRequest(BaseModel):
    direction: str  # "sqlite_to_sqlite" | "sqlite_to_pg" | "pg_to_sqlite"
    source_db_path: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    pg_database: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    limit: int = 200


class TestConnectionRequest(BaseModel):
    pg_host: str
    pg_port: int = 5433
    pg_database: str = "memories"
    pg_user: str = "memory_user"
    pg_password: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

_INTERNAL_METADATA_KEYS = {
    "last_decay_at"
}  # reinforcement_accum is now hoisted to top-level


def _clean_memory(obj: dict) -> dict:
    """Normalize a memory dict for JSON output.

    - Normalises timestamp to ISO string
    - Hoists reinforcement_accum from metadata to a top-level field so the
      frontend can display and sort by it without parsing JSON
    - Strips purely-internal metadata keys (last_decay_at)
    """
    obj = dict(obj)
    ts = obj.get("timestamp")
    if isinstance(ts, datetime):
        obj["timestamp"] = ts.isoformat()

    meta = obj.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}

    if isinstance(meta, dict):
        # Hoist reinforcement_accum as a named top-level field
        if "reinforcement_accum" in meta:
            obj["reinforcement_accum"] = meta["reinforcement_accum"]
        # Strip internal-only keys for the display metadata dict
        stripped = {k: v for k, v in meta.items() if k not in _INTERNAL_METADATA_KEYS}
        if stripped:
            obj["metadata"] = stripped
        else:
            obj.pop("metadata", None)
    return obj


def _result_to_response(result) -> dict:
    """Convert a Result dataclass to a JSON-serializable dict."""
    out: dict = {"success": result.success}
    if result.reason is not None:
        out["reason"] = result.reason
    if result.data is not None:
        out["data"] = [_clean_memory(item) for item in result.data]
    return out


def _require_success(result, status_code: int = 500) -> dict:
    """Raise HTTPException if result.success is False, otherwise return cleaned data."""
    if not result.success:
        raise HTTPException(
            status_code=status_code, detail=result.reason or "Unknown error"
        )
    return _result_to_response(result)


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(
    memory_system,
    identity=None,
    sharing_mgr=None,
) -> FastAPI:
    """
    Create and configure the FastAPI WebUI application.

    Args:
        memory_system: Shared RobustMemorySystem instance (embedding model already loaded).
        identity: Optional NodeIdentity instance for /api/v1/identity.
        sharing_mgr: Optional NetworkSharingManager for /api/v1/peers.

    Returns:
        Configured FastAPI application ready to be served by uvicorn.
    """
    app = FastAPI(
        title="Long-Term Memory Manager",
        description="WebUI REST API for the Long-Term Memory MCP system",
        version="1.0.0",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
    )

    # Allow the Vite dev server (port 5173) during development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── /api/v1/config ────────────────────────────────────────────────────────

    @app.get("/api/v1/config")
    def get_config():
        """Return backend type, embedding model info, and data folder."""
        from .config import EMBEDDING_MODEL, EMBEDDING_MODEL_CONFIG, DATA_FOLDER

        return {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_model_config": EMBEDDING_MODEL_CONFIG,
            "database_backend": getattr(memory_system.db, "backend_name", "unknown"),
            "vector_backend": getattr(
                memory_system.vector_backend, "backend_name", "unknown"
            ),
            "data_folder": str(DATA_FOLDER),
        }

    # ── /api/v1/identity ──────────────────────────────────────────────────────

    @app.get("/api/v1/identity")
    def get_identity():
        """Return this node's UUID and username."""
        if identity is None:
            return {"node_uuid": None, "username": None}
        return {
            "node_uuid": identity.node_uuid,
            "username": identity.username,
            "created_at": identity.created_at,
        }

    # ── /api/v1/peers ─────────────────────────────────────────────────────────

    @app.get("/api/v1/peers")
    def get_peers():
        """Return all currently discovered LAN peers."""
        if sharing_mgr is None:
            return {"peers": []}
        return {"peers": sharing_mgr.get_known_peers()}

    # ── /api/v1/stats ─────────────────────────────────────────────────────────

    @app.get("/api/v1/stats")
    def get_stats():
        """
        Extended statistics including token counts and type breakdowns.
        Augments get_statistics() with token/importance data from the DB.
        """
        result = memory_system.get_statistics()
        if not result.success:
            raise HTTPException(status_code=500, detail=result.reason)
        data = dict(result.data[0]) if result.data else {}

        # Add token stats and per-type token breakdown via direct DB query
        try:
            cursor = memory_system.db.execute(
                """
                SELECT
                    SUM(token_count) as total_tokens,
                    AVG(token_count) as avg_tokens
                FROM memories
                """
            )
            row = cursor.fetchone()
            data["total_tokens"] = row["total_tokens"] or 0
            data["avg_tokens"] = round(row["avg_tokens"] or 0)

            cursor = memory_system.db.execute(
                """
                SELECT memory_type,
                       COUNT(*) as count,
                       SUM(token_count) as tokens
                FROM memories
                GROUP BY memory_type
                ORDER BY count DESC
                """
            )
            data["type_token_breakdown"] = {
                r["memory_type"]: {"count": r["count"], "tokens": r["tokens"] or 0}
                for r in cursor.fetchall()
            }
        except Exception as e:
            logger.warning("Could not fetch token stats: %s", e)

        return {"success": True, "data": data}

    # ── /api/v1/memories ──────────────────────────────────────────────────────

    @app.get("/api/v1/memories")
    def list_memories(
        q: Optional[str] = Query(
            default=None, description="Full-text search query (semantic)"
        ),
        type: Optional[str] = Query(default=None, description="Filter by memory_type"),
        min_importance: Optional[int] = Query(default=None, ge=1, le=10),
        tags: Optional[str] = Query(
            default=None, description="Comma-separated tag names"
        ),
        date_from: Optional[str] = Query(default=None),
        date_to: Optional[str] = Query(default=None),
        sort: Optional[str] = Query(
            default="importance DESC, timestamp DESC",
            description="Sort order.",
        ),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        search_type: str = Query(
            default="structured", description="'structured' or 'semantic'"
        ),
    ):
        """
        List or search memories with server-side paging.

        Returns: { success, total, offset, limit, data[] }
        - total: total matching rows (for building page controls)
        - offset/limit: echo back for the client
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        if search_type == "semantic" and q and q.strip():
            # Semantic search: fetch up to limit results from the vector index.
            # offset is applied post-fetch (vector search has no native offset).
            result = memory_system.search_semantic(
                q.strip(), limit=min(limit + offset, 200)
            )
            if not result.success:
                raise HTTPException(status_code=500, detail=result.reason)
            all_data = result.data or []
            total = len(all_data)
            paged = all_data[offset : offset + limit]
            return {
                "success": True,
                "total": total,
                "offset": offset,
                "limit": limit,
                "data": paged,
            }

        # ── Structured / text search ──────────────────────────────────────────
        # Step 1: get the total matching count (cheap COUNT query)
        total = _count_matching(
            memory_system,
            q=q.strip() if q else None,
            memory_type=type,
            importance_min=min_importance,
            tags=tag_list,
            date_from=date_from,
            date_to=date_to,
        )

        # Step 2: fetch one page
        if q and q.strip():
            result = _search_text_structured(
                memory_system,
                q=q.strip(),
                memory_type=type,
                importance_min=min_importance,
                tags=tag_list,
                date_from=date_from,
                date_to=date_to,
                order_by=sort,
                limit=limit,
                offset=offset,
            )
        else:
            result = _search_structured_paged(
                memory_system,
                memory_type=type,
                importance_min=min_importance,
                tags=tag_list,
                date_from=date_from,
                date_to=date_to,
                order_by=sort,
                limit=limit,
                offset=offset,
            )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.reason)

        return {
            "success": True,
            "total": total,
            "offset": offset,
            "limit": limit,
            "data": [_clean_memory(item) for item in (result.data or [])],
        }

    @app.get("/api/v1/memories/{memory_id}")
    def get_memory(memory_id: str):
        """Fetch a single memory by ID including full metadata."""
        try:
            cursor = memory_system.db.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        if not row:
            raise HTTPException(status_code=404, detail="Memory not found")

        return _row_to_dict(row)

    @app.post("/api/v1/memories", status_code=201)
    def create_memory(body: MemoryCreate):
        """Create a new memory."""
        result = memory_system.remember(
            title=body.title,
            content=body.content,
            tags=body.tags,
            importance=body.importance,
            memory_type=body.memory_type,
            shared_with=body.shared_with,
            file_paths=body.file_paths,
        )
        return _require_success(result, status_code=500)

    @app.patch("/api/v1/memories/{memory_id}")
    def update_memory(memory_id: str, body: MemoryUpdate):
        """Partially update a memory. Only provided fields are changed."""
        result = memory_system.update_memory(
            memory_id=memory_id,
            title=body.title,
            content=body.content,
            tags=body.tags,
            importance=body.importance,
            memory_type=body.memory_type,
            shared_with=body.shared_with,
        )
        if not result.success:
            status = 404 if "not found" in (result.reason or "").lower() else 500
            raise HTTPException(status_code=status, detail=result.reason)
        return _result_to_response(result)

    @app.delete("/api/v1/memories/{memory_id}")
    def delete_memory(memory_id: str):
        """Permanently delete a memory by ID."""
        result = memory_system.delete_memory(memory_id)
        if not result.success:
            status = 404 if "not found" in (result.reason or "").lower() else 500
            raise HTTPException(status_code=status, detail=result.reason)
        return _result_to_response(result)

    # ── /api/v1/backup ────────────────────────────────────────────────────────

    @app.post("/api/v1/backup")
    def create_backup():
        """Trigger an immediate backup of the memory database."""
        result = memory_system.create_backup()
        return _require_success(result)

    # ── /api/v1/export ────────────────────────────────────────────────────────

    @app.get("/api/v1/export")
    def export_memories():
        """
        Stream all memories as a JSON file download.
        Equivalent to the Tkinter GUI 'Export' button.
        """
        try:
            cursor = memory_system.db.execute(
                "SELECT * FROM memories ORDER BY timestamp ASC"
            )
            rows = cursor.fetchall()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        memories = [_row_to_dict(row) for row in rows]
        payload = json.dumps(
            {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_memories": len(memories),
                "memories": memories,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"memories_export_{ts}.json"
        return Response(
            content=payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ── /api/v1/vectors ───────────────────────────────────────────────────────

    @app.get("/api/v1/vectors/stats")
    def get_vector_stats():
        """
        Vector backend statistics: count, configured dimensions, backend name,
        and a mismatch warning if stored dims differ from configured dims.
        """
        from .config import EMBEDDING_MODEL, EMBEDDING_MODEL_CONFIG

        try:
            count = memory_system.vector_backend.count()
            backend = memory_system.vector_backend.backend_name
            configured_dims = EMBEDDING_MODEL_CONFIG["dimensions"]

            # Probe actual stored dimensions by fetching one vector
            actual_dims = None
            try:
                sample = memory_system.vector_backend.get(
                    limit=1, include_embeddings=True
                )
                embeddings = sample.get("embeddings") or []
                if embeddings and embeddings[0]:
                    actual_dims = len(embeddings[0])
            except Exception:
                pass

            dims_match = (
                actual_dims == configured_dims if actual_dims is not None else None
            )

            return {
                "backend": backend,
                "count": count,
                "embedding_model": EMBEDDING_MODEL,
                "configured_dims": configured_dims,
                "actual_dims": actual_dims,
                "dims_match": dims_match,
                "dims_warning": (
                    f"Stored dims ({actual_dims}) != configured dims ({configured_dims}). "
                    "Run rebuild_vectors to fix."
                    if dims_match is False
                    else None
                ),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/vectors/{memory_id}")
    def get_vector(memory_id: str):
        """
        Retrieve and analyze the stored vector for a specific memory.
        Returns the first 20 dimensions plus min/max/mean/L2-norm stats.
        """
        try:
            data = memory_system.vector_backend.get(
                ids=[memory_id], include_embeddings=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        ids = data.get("ids") or []
        if not ids or memory_id not in ids:
            raise HTTPException(status_code=404, detail="Vector not found")

        idx = ids.index(memory_id)
        embeddings = data.get("embeddings") or []
        embedding = embeddings[idx] if idx < len(embeddings) else None

        documents = data.get("documents") or []
        document = documents[idx] if idx < len(documents) else None

        metadatas = data.get("metadatas") or []
        metadata = metadatas[idx] if idx < len(metadatas) else None

        stats = _vector_stats(embedding) if embedding else None

        return {
            "id": memory_id,
            "document_preview": document[:200] if document else None,
            "metadata": metadata,
            "dimensions": len(embedding) if embedding else None,
            "first_20": embedding[:20] if embedding else None,
            "stats": stats,
        }

    @app.post("/api/v1/vectors/rebuild")
    def rebuild_vectors():
        """Rebuild the entire vector index from the database."""
        result = memory_system.rebuild_vector_index()
        return _require_success(result)

    # ── /api/v1/migrate ───────────────────────────────────────────────────────

    @app.post("/api/v1/migrate/test-connection")
    def test_pg_connection(body: TestConnectionRequest):
        """Test a PostgreSQL connection and return basic stats."""
        try:
            import psycopg

            conn = psycopg.connect(
                host=body.pg_host,
                port=body.pg_port,
                dbname=body.pg_database,
                user=body.pg_user,
                password=body.pg_password,
                connect_timeout=5,
            )
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'memories')"
                )
                has_memories = cur.fetchone()[0]

                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'memory_vectors')"
                )
                has_vectors = cur.fetchone()[0]

                memory_count = 0
                vector_count = 0
                if has_memories:
                    cur.execute("SELECT COUNT(*) FROM memories")
                    memory_count = cur.fetchone()[0]
                if has_vectors:
                    cur.execute("SELECT COUNT(*) FROM memory_vectors")
                    vector_count = cur.fetchone()[0]
            conn.close()
            return {
                "success": True,
                "has_memories_table": has_memories,
                "has_vectors_table": has_vectors,
                "memory_count": memory_count,
                "vector_count": vector_count,
            }
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="psycopg not installed. Run: pip install 'psycopg[binary]'",
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/api/v1/migrate/preview")
    def preview_migration(body: MigratePreviewRequest):
        """Preview source memories before committing to a migration."""
        try:
            memories = _preview_source(memory_system, body)
            return {"success": True, "count": len(memories), "memories": memories}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/migrate")
    def run_migration(body: MigrateRequest):
        """
        Execute a memory migration.

        Directions:
          - sqlite_to_sqlite: migrate from a source SQLite file into the active DB.
          - sqlite_to_pg: migrate from the active SQLite into a Postgres instance.
          - pg_to_sqlite: migrate from a Postgres instance into the active SQLite.
        """
        try:
            if body.direction == "sqlite_to_sqlite":
                return _migrate_sqlite_to_sqlite(memory_system, body)
            elif body.direction == "sqlite_to_pg":
                return _migrate_sqlite_to_pg(memory_system, body)
            elif body.direction == "pg_to_sqlite":
                return _migrate_pg_to_sqlite(memory_system, body)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown direction: {body.direction}. "
                    "Use sqlite_to_sqlite, sqlite_to_pg, or pg_to_sqlite.",
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── /api/v1/compare ───────────────────────────────────────────────────────

    @app.get("/api/v1/compare")
    def compare_backends(
        pg_host: str = Query(default="localhost"),
        pg_port: int = Query(default=5433),
        pg_database: str = Query(default="memories"),
        pg_user: str = Query(default="memory_user"),
        pg_password: str = Query(default=""),
    ):
        """
        Compare the active SQLite database against a Postgres instance.
        Returns four classified lists: only_in_sqlite, only_in_pg, modified, identical.
        Also returns vector counts from each side.
        """
        try:
            return _compare_backends(
                memory_system,
                pg_host=pg_host,
                pg_port=pg_port,
                pg_database=pg_database,
                pg_user=pg_user,
                pg_password=pg_password,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Static files (React SPA) ───────────────────────────────────────────────

    _webui_dist = Path(__file__).parent.parent / "webui" / "dist"
    if _webui_dist.exists():
        # Serve the built frontend at root; the API is already under /api/v1/
        app.mount("/", StaticFiles(directory=str(_webui_dist), html=True), name="webui")
    else:

        @app.get("/")
        def webui_not_built():
            return {
                "message": "WebUI not built yet.",
                "instructions": "Run: cd webui && npm install && npm run build",
                "api_docs": "/api/v1/docs",
            }

    return app


# ── Helper: text search via SQL LIKE ──────────────────────────────────────────


_ALLOWED_ORDERS = {
    "importance DESC, timestamp DESC",
    "timestamp DESC",
    "timestamp DESC, importance DESC",
    "timestamp ASC",
    "importance DESC",
    "last_accessed DESC",
    "last_accessed ASC",
    # reinforcement_accum lives inside the metadata JSON column
    "json_extract(metadata, '$.reinforcement_accum') DESC, importance DESC",
    "json_extract(metadata, '$.reinforcement_accum') ASC, importance DESC",
}


def _build_where(
    q: Optional[str],
    memory_type: Optional[str],
    importance_min: Optional[int],
    tags: Optional[List[str]],
    date_from: Optional[str],
    date_to: Optional[str],
) -> tuple:
    """Return (conditions_list, params_list) for a WHERE clause."""
    conditions: List[str] = []
    params: List[Any] = []

    if q:
        conditions.append("(title LIKE ? OR content LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    if memory_type:
        conditions.append("memory_type = ?")
        params.append(memory_type)
    if importance_min:
        conditions.append("importance >= ?")
        params.append(importance_min)
    if date_from:
        conditions.append("timestamp >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("timestamp <= ?")
        params.append(date_to)
    if tags:
        tag_conds = [f"tags LIKE ?" for _ in tags]
        params += [f'%"{t}"%' for t in tags]
        conditions.append(f"({' OR '.join(tag_conds)})")

    return conditions, params


def _count_matching(
    memory_system,
    q: Optional[str],
    memory_type: Optional[str],
    importance_min: Optional[int],
    tags: Optional[List[str]],
    date_from: Optional[str],
    date_to: Optional[str],
) -> int:
    """Return the total number of rows matching the given filters."""
    try:
        conditions, params = _build_where(
            q, memory_type, importance_min, tags, date_from, date_to
        )
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor = memory_system.db.execute(
            f"SELECT COUNT(*) as n FROM memories {where}", params
        )
        row = cursor.fetchone()
        return row["n"] if row else 0
    except Exception:
        return 0


def _search_structured_paged(
    memory_system,
    memory_type: Optional[str],
    importance_min: Optional[int],
    tags: Optional[List[str]],
    date_from: Optional[str],
    date_to: Optional[str],
    order_by: Optional[str],
    limit: int,
    offset: int,
):
    """Structured search with LIMIT/OFFSET paging (no free-text filter)."""
    from .models import Result

    try:
        conditions, params = _build_where(
            None, memory_type, importance_min, tags, date_from, date_to
        )
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        effective_order = (
            order_by
            if order_by in _ALLOWED_ORDERS
            else "importance DESC, timestamp DESC"
        )
        query = f"SELECT * FROM memories {where} ORDER BY {effective_order} LIMIT ? OFFSET ?"
        params += [limit, offset]
        cursor = memory_system.db.execute(query, params)
        rows = cursor.fetchall()
        return Result(success=True, data=[_row_to_dict(r) for r in rows])
    except Exception as e:
        return Result(success=False, reason=str(e))


def _search_text_structured(
    memory_system,
    q: str,
    memory_type: Optional[str],
    importance_min: Optional[int],
    tags: Optional[List[str]],
    date_from: Optional[str],
    date_to: Optional[str],
    order_by: Optional[str],
    limit: int,
    offset: int = 0,
):
    """
    SQL LIKE search on title + content with optional filters and paging.
    """
    from .models import Result

    try:
        conditions, params = _build_where(
            q, memory_type, importance_min, tags, date_from, date_to
        )
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        effective_order = (
            order_by
            if order_by in _ALLOWED_ORDERS
            else "importance DESC, timestamp DESC"
        )
        query = f"SELECT * FROM memories {where} ORDER BY {effective_order} LIMIT ? OFFSET ?"
        params += [limit, offset]
        cursor = memory_system.db.execute(query, params)
        rows = cursor.fetchall()
        return Result(success=True, data=[_row_to_dict(r) for r in rows])
    except Exception as e:
        return Result(success=False, reason=str(e))


# ── Helper: row → dict ────────────────────────────────────────────────────────


def _row_to_dict(row) -> dict:
    """Convert a raw DB row (dict-like) to a clean JSON-serializable dict."""
    obj: dict = dict(row)
    # Decode JSON-stored fields
    for field in ("tags", "shared_with", "metadata"):
        val = obj.get(field)
        if isinstance(val, str):
            try:
                obj[field] = json.loads(val)
            except Exception:
                pass
        elif val is None:
            obj[field] = [] if field in ("tags", "shared_with") else {}

    # Normalize metadata — strip internal keys
    meta = obj.get("metadata")
    if isinstance(meta, dict):
        stripped = {k: v for k, v in meta.items() if k not in _INTERNAL_METADATA_KEYS}
        obj["metadata"] = stripped if stripped else {}

    return obj


# ── Helper: vector stats ──────────────────────────────────────────────────────


def _vector_stats(embedding: List[float]) -> dict:
    """Compute min/max/mean/L2-norm for a float vector."""
    if not embedding:
        return {}
    try:
        import math

        n = len(embedding)
        min_v = min(embedding)
        max_v = max(embedding)
        mean_v = sum(embedding) / n
        l2 = math.sqrt(sum(x * x for x in embedding))
        return {
            "dimensions": n,
            "min": round(min_v, 6),
            "max": round(max_v, 6),
            "mean": round(mean_v, 6),
            "l2_norm": round(l2, 6),
        }
    except Exception:
        return {}


# ── Migration helpers ─────────────────────────────────────────────────────────


def _preview_source(memory_system, body: "MigratePreviewRequest") -> List[dict]:
    """Return up to body.limit memory summaries from the specified source."""
    limit = body.limit

    if body.direction == "sqlite_to_sqlite":
        if not body.source_db_path:
            raise HTTPException(status_code=400, detail="source_db_path is required")
        result = memory_system.list_source_memories(body.source_db_path, limit=limit)
        if not result.success:
            raise HTTPException(status_code=500, detail=result.reason)
        return result.data or []

    elif body.direction in ("sqlite_to_pg", "pg_to_sqlite"):
        # For sqlite_to_pg the SOURCE is sqlite (active db) — just return recent memories.
        # For pg_to_sqlite the SOURCE is postgres — query it directly.
        if body.direction == "sqlite_to_pg":
            result = memory_system.search_structured(
                limit=limit, order_by="timestamp DESC"
            )
            if not result.success:
                raise HTTPException(status_code=500, detail=result.reason)
            return [
                {
                    k: v
                    for k, v in m.items()
                    if k
                    in ("id", "title", "memory_type", "importance", "tags", "timestamp")
                }
                for m in (result.data or [])
            ]
        else:
            # pg_to_sqlite: query postgres
            conn = _pg_connect(body)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, title, memory_type, importance, tags FROM memories "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (limit,),
                    )
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in rows]
            finally:
                conn.close()

    raise HTTPException(status_code=400, detail=f"Unknown direction: {body.direction}")


def _pg_connect(body):
    """Open a psycopg connection from request body fields."""
    try:
        import psycopg
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="psycopg not installed. Run: pip install 'psycopg[binary]'",
        )
    try:
        return psycopg.connect(
            host=body.pg_host or "localhost",
            port=body.pg_port or 5433,
            dbname=body.pg_database or "memories",
            user=body.pg_user or "memory_user",
            password=body.pg_password or "",
            connect_timeout=10,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PG connection failed: {e}")


def _migrate_sqlite_to_sqlite(memory_system, body: "MigrateRequest") -> dict:
    """Migrate from a source SQLite file into the active database."""
    if not body.source_db_path:
        raise HTTPException(status_code=400, detail="source_db_path is required")

    memory_id_list = body.memory_ids if body.memory_ids else None
    result = memory_system.migrate_memories(
        source_db_path=body.source_db_path,
        source_chroma_path=body.source_chroma_path,
        memory_ids=memory_id_list,
        skip_duplicates=body.skip_duplicates,
    )
    return _require_success(result)


def _migrate_sqlite_to_pg(memory_system, body: "MigrateRequest") -> dict:
    """
    Migrate from the active SQLite/ChromaDB into a Postgres/pgvector instance.
    Creates the target schema if it does not exist.
    """
    import psycopg as _pg  # noqa: F401 — will raise ImportError if missing

    conn = _pg_connect(body)
    try:
        with conn.cursor() as cur:
            # Ensure pgvector extension and tables exist
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            from .config import EMBEDDING_MODEL_CONFIG

            dims = EMBEDDING_MODEL_CONFIG["dimensions"]
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMPTZ,
                    tags JSONB DEFAULT '[]',
                    importance INTEGER DEFAULT 5,
                    memory_type TEXT DEFAULT 'conversation',
                    metadata JSONB DEFAULT '{}',
                    content_hash TEXT,
                    last_accessed TIMESTAMPTZ,
                    token_count INTEGER DEFAULT 0,
                    shared_with JSONB DEFAULT '[]',
                    updated_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    document TEXT,
                    metadata JSONB,
                    embedding vector({dims})
                )
                """
            )
            conn.commit()

        # Fetch all source memories from SQLite
        cursor = memory_system.db.execute("SELECT * FROM memories")
        rows = cursor.fetchall()

        migrated = skipped = errors = vectors_migrated = 0
        with conn.cursor() as cur:
            for row in rows:
                try:
                    # Check for duplicate
                    if body.skip_duplicates:
                        cur.execute(
                            "SELECT 1 FROM memories WHERE content_hash = %s",
                            (row["content_hash"],),
                        )
                        if cur.fetchone():
                            skipped += 1
                            continue

                    cur.execute(
                        """
                        INSERT INTO memories
                        (id, title, content, timestamp, tags, importance, memory_type,
                         metadata, content_hash, last_accessed, token_count, shared_with,
                         updated_at, created_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                            title=EXCLUDED.title, content=EXCLUDED.content,
                            timestamp=EXCLUDED.timestamp, tags=EXCLUDED.tags,
                            importance=EXCLUDED.importance,
                            memory_type=EXCLUDED.memory_type,
                            metadata=EXCLUDED.metadata,
                            content_hash=EXCLUDED.content_hash,
                            last_accessed=EXCLUDED.last_accessed,
                            token_count=EXCLUDED.token_count,
                            shared_with=EXCLUDED.shared_with,
                            updated_at=EXCLUDED.updated_at
                        """,
                        (
                            row["id"],
                            row["title"],
                            row["content"],
                            row["timestamp"],
                            row["tags"],
                            row["importance"],
                            row["memory_type"],
                            row["metadata"],
                            row["content_hash"],
                            row.get("last_accessed"),
                            row.get("token_count") or 0,
                            row.get("shared_with", "[]"),
                            row.get("updated_at"),
                            row.get("created_at"),
                        ),
                    )
                    migrated += 1

                    # Migrate vector if requested
                    if body.migrate_vectors:
                        try:
                            vec_data = memory_system.vector_backend.get(
                                ids=[row["id"]], include_embeddings=True
                            )
                            vids = vec_data.get("ids") or []
                            if row["id"] in vids:
                                idx = vids.index(row["id"])
                                emb = (vec_data.get("embeddings") or [])[idx]
                                doc = (vec_data.get("documents") or [None])[idx]
                                meta = (vec_data.get("metadatas") or [None])[idx]
                                if emb:
                                    cur.execute(
                                        """
                                        INSERT INTO memory_vectors (id, document, metadata, embedding)
                                        VALUES (%s, %s, %s, %s)
                                        ON CONFLICT (id) DO UPDATE
                                        SET embedding=EXCLUDED.embedding,
                                            document=EXCLUDED.document,
                                            metadata=EXCLUDED.metadata
                                        """,
                                        (
                                            row["id"],
                                            doc,
                                            json.dumps(meta) if meta else None,
                                            emb,
                                        ),
                                    )
                                    vectors_migrated += 1
                        except Exception as ve:
                            logger.warning(
                                "Could not migrate vector for %s: %s", row["id"], ve
                            )

                    conn.commit()
                except Exception as re:
                    conn.rollback()
                    errors += 1
                    logger.error("Row migration error for %s: %s", row.get("id"), re)

        return {
            "success": True,
            "data": {
                "total_found": len(rows),
                "migrated": migrated,
                "skipped_duplicates": skipped,
                "errors": errors,
                "vectors_migrated": vectors_migrated,
            },
        }
    finally:
        conn.close()


def _migrate_pg_to_sqlite(memory_system, body: "MigrateRequest") -> dict:
    """Migrate from Postgres into the active SQLite/ChromaDB."""
    conn = _pg_connect(body)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM memories")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]

        migrated = skipped = errors = vectors_migrated = 0

        for row in rows:
            try:
                if body.skip_duplicates:
                    dup = memory_system.db.execute(
                        "SELECT 1 FROM memories WHERE content_hash = ?",
                        (row.get("content_hash"),),
                    ).fetchone()
                    if dup:
                        skipped += 1
                        continue

                # Normalize JSON fields
                for f in ("tags", "shared_with", "metadata"):
                    v = row.get(f)
                    if v is None:
                        row[f] = "[]" if f in ("tags", "shared_with") else "{}"
                    elif not isinstance(v, str):
                        row[f] = json.dumps(v)

                memory_system.db.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (id, title, content, timestamp, tags, importance, memory_type,
                     metadata, content_hash, last_accessed, token_count, shared_with,
                     updated_at, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["id"],
                        row["title"],
                        row["content"],
                        str(row.get("timestamp", "")),
                        row["tags"],
                        row.get("importance", 5),
                        row.get("memory_type", "conversation"),
                        row["metadata"],
                        row.get("content_hash"),
                        str(row.get("last_accessed", "")),
                        row.get("token_count") or 0,
                        row["shared_with"],
                        str(row.get("updated_at", "")),
                        str(row.get("created_at", "")),
                    ),
                )
                memory_system.db.commit()
                migrated += 1

                # Re-embed and add to ChromaDB
                if body.migrate_vectors:
                    try:
                        text = f"{row['title']}\n{row['content']}"
                        emb = memory_system.embedding_model.encode(text).tolist()
                        memory_system.vector_backend.add(
                            ids=[row["id"]],
                            embeddings=[emb],
                            documents=[text],
                            metadatas=[
                                {
                                    "title": row["title"],
                                    "memory_type": row.get("memory_type", ""),
                                }
                            ],
                        )
                        vectors_migrated += 1
                    except Exception as ve:
                        logger.warning(
                            "Vector add failed for %s: %s", row.get("id"), ve
                        )

            except Exception as re:
                memory_system.db.rollback()
                errors += 1
                logger.error("Row migration error for %s: %s", row.get("id"), re)

        return {
            "success": True,
            "data": {
                "total_found": len(rows),
                "migrated": migrated,
                "skipped_duplicates": skipped,
                "errors": errors,
                "vectors_migrated": vectors_migrated,
            },
        }
    finally:
        conn.close()


# ── Compare helper ────────────────────────────────────────────────────────────


def _compare_backends(
    memory_system,
    pg_host: str,
    pg_port: int,
    pg_database: str,
    pg_user: str,
    pg_password: str,
) -> dict:
    """Diff active SQLite vs a Postgres instance."""
    try:
        import psycopg
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="psycopg not installed. Run: pip install 'psycopg[binary]'",
        )

    # Read SQLite side
    try:
        cursor = memory_system.db.execute(
            "SELECT id, title, memory_type, importance, content_hash FROM memories"
        )
        sqlite_rows = {r["id"]: dict(r) for r in cursor.fetchall()}
        sqlite_vec_count = memory_system.vector_backend.count()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQLite read error: {e}")

    # Read Postgres side
    try:
        conn = psycopg.connect(
            host=pg_host,
            port=pg_port,
            dbname=pg_database,
            user=pg_user,
            password=pg_password,
            connect_timeout=5,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, memory_type, importance, content_hash FROM memories"
            )
            cols = [d[0] for d in cur.description]
            pg_rows = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}

            pg_vec_count = 0
            try:
                cur.execute("SELECT COUNT(*) FROM memory_vectors")
                pg_vec_count = cur.fetchone()[0]
            except Exception:
                pass
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PG connection failed: {e}")

    all_ids = set(sqlite_rows) | set(pg_rows)
    only_sqlite, only_pg, modified, identical = [], [], [], []

    for mid in all_ids:
        in_s = mid in sqlite_rows
        in_p = mid in pg_rows
        if in_s and not in_p:
            only_sqlite.append(sqlite_rows[mid])
        elif in_p and not in_s:
            only_pg.append(pg_rows[mid])
        else:
            h_s = sqlite_rows[mid].get("content_hash") or ""
            h_p = pg_rows[mid].get("content_hash") or ""
            if h_s and h_p and h_s == h_p:
                identical.append(sqlite_rows[mid])
            else:
                row = dict(sqlite_rows[mid])
                row["hash_sqlite"] = h_s
                row["hash_pg"] = h_p
                modified.append(row)

    return {
        "sqlite_memory_count": len(sqlite_rows),
        "sqlite_vector_count": sqlite_vec_count,
        "pg_memory_count": len(pg_rows),
        "pg_vector_count": pg_vec_count,
        "only_in_sqlite": only_sqlite,
        "only_in_pg": only_pg,
        "modified": modified,
        "identical": identical,
        "summary": {
            "only_sqlite": len(only_sqlite),
            "only_pg": len(only_pg),
            "modified": len(modified),
            "identical": len(identical),
        },
    }
