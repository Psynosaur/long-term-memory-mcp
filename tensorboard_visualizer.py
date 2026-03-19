#!/usr/bin/env python3
"""
TensorBoard Embedding Projector for Long-Term Memory MCP.

Exports memory embeddings to TensorBoard format and launches the
Embedding Projector — a rich interactive tool with built-in PCA,
t-SNE, UMAP, custom projections, nearest-neighbour search, and
metadata-based filtering/colouring.

Only requires ``tensorboard`` (no TensorFlow).

Usage:
    python tensorboard_visualizer.py
    python tensorboard_visualizer.py --vector-backend pgvector
    python tensorboard_visualizer.py --port 6006
    python tensorboard_visualizer.py --logdir runs/my_session
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Reuse vector-loading helpers from the existing visualiser
# ---------------------------------------------------------------------------


def _load_vectors(
    backend_type: str = "chromadb",
    pg_host: Optional[str] = None,
    pg_port: Optional[int] = None,
    pg_database: Optional[str] = None,
    pg_user: Optional[str] = None,
    pg_password: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
    """Load all vectors and metadata from the specified backend.

    Returns (embeddings, metadatas, ids).
    """
    from memory_mcp.config import DATA_FOLDER, EMBEDDING_MODEL_CONFIG

    if backend_type == "pgvector":
        from memory_mcp.vector_backends.pgvector_backend import PgvectorBackend

        backend = PgvectorBackend(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password or os.environ.get("PGPASSWORD"),
            dimensions=EMBEDDING_MODEL_CONFIG["dimensions"],
        )
    else:
        from memory_mcp.vector_backends.chroma import ChromaBackend

        backend = ChromaBackend(db_folder=Path(DATA_FOLDER) / "memory_db")

    initialized = False
    try:
        backend.initialize()
        initialized = True
        count = backend.count()
        if count == 0:
            print("No vectors found in the database.")
            sys.exit(0)

        print(f"Loading {count} vectors from {backend.backend_name}...")
        t0 = time.perf_counter()

        data = backend.get(include_embeddings=True)

        ids = data["ids"]
        metadatas = data.get("metadatas") or [{} for _ in range(len(ids))]
        raw_embeddings = data.get("embeddings")
        if raw_embeddings is None or (
            hasattr(raw_embeddings, "__len__") and len(raw_embeddings) == 0
        ):
            print("Backend returned no embeddings.")
            sys.exit(1)

        embeddings = np.array(raw_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            print(f"Expected 2D embeddings array, got shape {embeddings.shape}")
            sys.exit(1)
        elapsed = time.perf_counter() - t0
        print(
            f"Loaded {embeddings.shape[0]} vectors ({embeddings.shape[1]}D) "
            f"in {elapsed:.2f}s"
        )
        return embeddings, metadatas, ids
    finally:
        if initialized:
            backend.close()


def _load_memory_texts(
    ids: List[str],
    backend_type: str = "chromadb",
    pg_host: Optional[str] = None,
    pg_port: Optional[int] = None,
    pg_database: Optional[str] = None,
    pg_user: Optional[str] = None,
    pg_password: Optional[str] = None,
) -> Dict[str, str]:
    """Load full title + content from the memories table.

    Returns {memory_id: "title\\ncontent"}.
    """
    from memory_mcp.config import DATA_FOLDER

    texts: Dict[str, str] = {}

    if not ids:
        return texts

    if backend_type == "pgvector":
        try:
            from memory_mcp.database_backends.postgres import PostgresDatabase

            db = PostgresDatabase(
                host=pg_host,
                port=pg_port,
                database=pg_database,
                user=pg_user,
                password=pg_password or os.environ.get("PGPASSWORD"),
            )
            db.initialize()
            try:
                cur = db.execute(
                    "SELECT id, title, content FROM memories WHERE id = ANY(%s)",
                    [list(ids)],
                )
                for row in cur.fetchall():
                    mid = row["id"]
                    title = row["title"] or ""
                    content = row["content"] or ""
                    texts[mid] = f"{title}\n{content}" if content else title
            finally:
                db.close()
        except Exception as exc:
            print(f"Warning: could not load memory texts from Postgres: {exc}")
    else:
        try:
            import sqlite3

            db_path = Path(DATA_FOLDER) / "memory_db" / "memories.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                try:
                    placeholders = ",".join("?" for _ in ids)
                    cur = conn.execute(
                        f"SELECT id, title, content FROM memories "
                        f"WHERE id IN ({placeholders})",
                        list(ids),
                    )
                    for row in cur.fetchall():
                        mid, title, content = row[0], row[1] or "", row[2] or ""
                        texts[mid] = f"{title}\n{content}" if content else title
                finally:
                    conn.close()
        except Exception as exc:
            print(f"Warning: could not load memory texts from SQLite: {exc}")

    return texts


def _clean_tag(tag: str) -> str:
    """Strip quotes, brackets from a single tag string."""
    return tag.strip().strip("\"'[]")


def _parse_tags(raw: Any) -> List[str]:
    """Parse tags from metadata (may be JSON array string or comma-separated)."""
    if not raw:
        return []
    s = str(raw)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [cleaned for t in parsed if (cleaned := _clean_tag(str(t)))]
    except (json.JSONDecodeError, TypeError):
        pass
    return [cleaned for t in s.split(",") if (cleaned := _clean_tag(t))]


def _sanitize_tsv(s: str) -> str:
    """Remove tab and newline characters to prevent TSV field corruption."""
    return s.replace("\t", " ").replace("\n", " ").replace("\r", "")


# ---------------------------------------------------------------------------
# TensorBoard export
# ---------------------------------------------------------------------------


def write_tensorboard_embeddings(
    logdir: str,
    embeddings: np.ndarray,
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    memory_texts: Dict[str, str],
    tensor_name: str = "memory_embeddings",
) -> None:
    """Write embedding TSV, metadata TSV, and projector_config.pbtxt.

    The TensorBoard Embedding Projector reads these files directly
    without needing TensorFlow checkpoints.
    """
    logdir_path = Path(logdir)
    logdir_path.mkdir(parents=True, exist_ok=True)

    n, d = embeddings.shape
    print(f"Writing {n} embeddings ({d}D) to {logdir_path}/")

    # ── Vectors TSV ─────────────────────────────────────────────
    tensors_path = logdir_path / "tensors.tsv"
    t0 = time.perf_counter()
    with open(tensors_path, "w") as f:
        for i in range(n):
            line = "\t".join(f"{v:.6f}" for v in embeddings[i])
            f.write(line + "\n")
    elapsed = time.perf_counter() - t0
    print(
        f"  tensors.tsv ({tensors_path.stat().st_size / 1024:.0f} KB, {elapsed:.2f}s)"
    )

    # ── Metadata TSV ────────────────────────────────────────────
    # Columns: id, title, memory_type, importance, tags, content_preview
    metadata_path = logdir_path / "metadata.tsv"
    with open(metadata_path, "w") as f:
        f.write("id\ttitle\tmemory_type\timportance\ttags\tcontent_preview\n")
        for i, mid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            title = _sanitize_tsv(str(meta.get("title", "untitled")))
            mem_type = _sanitize_tsv(str(meta.get("memory_type", "unknown")))
            importance = _sanitize_tsv(str(meta.get("importance", "5")))
            tags_raw = meta.get("tags", "")
            tags_str = _sanitize_tsv(", ".join(_parse_tags(tags_raw)))

            # Content preview (first 200 chars of full text)
            full_text = memory_texts.get(mid, title)
            preview = _sanitize_tsv(full_text[:200])

            f.write(
                f"{mid}\t{title}\t{mem_type}\t{importance}\t{tags_str}\t{preview}\n"
            )

    print(f"  metadata.tsv ({metadata_path.stat().st_size / 1024:.0f} KB)")

    # ── Projector config ────────────────────────────────────────
    config_path = logdir_path / "projector_config.pbtxt"
    config = f"""embeddings {{
  tensor_name: "{tensor_name}"
  tensor_path: "tensors.tsv"
  metadata_path: "metadata.tsv"
}}
"""
    config_path.write_text(config)
    print(f"  projector_config.pbtxt")

    print(f"Export complete: {n} vectors, {d} dimensions.")


# ---------------------------------------------------------------------------
# TensorBoard launch
# ---------------------------------------------------------------------------


def launch_tensorboard(
    logdir: str, port: int = 6006, open_browser: bool = True
) -> None:
    """Start TensorBoard pointing at *logdir* and open the Projector tab."""
    try:
        from tensorboard import program
    except ImportError:
        print(
            "tensorboard is not installed.\n"
            "Install with:  pip install tensorboard\n"
            "Or:            pip install '.[tensorboard]'"
        )
        sys.exit(1)

    print(f"\nLaunching TensorBoard on port {port}...")
    print(f"Log directory: {logdir}")

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--port", str(port)])

    url = tb.launch()
    projector_url = f"{url}#projector"

    print(f"\nTensorBoard running at: {url}")
    print(f"Embedding Projector at: {projector_url}")
    print("\nPress Ctrl+C to stop.\n")

    if open_browser:
        time.sleep(1.5)
        webbrowser.open(projector_url)

    # Keep the process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="TensorBoard Embedding Projector for Long-Term Memory MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The TensorBoard Embedding Projector provides:\n"
            "  - PCA, t-SNE, UMAP, and custom linear projections\n"
            "  - Nearest-neighbour search by point or text query\n"
            "  - Colour by any metadata column\n"
            "  - Filter and isolate points by label\n"
            "  - 2D and 3D views with smooth animations\n"
            "\n"
            "Only requires: pip install tensorboard\n"
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port for TensorBoard (default: 6006)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: data/tensorboard_logs)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Write TSV files but don't launch TensorBoard",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the browser automatically",
    )

    # Backend args (mirror server.py / vector_visualizer.py)
    parser.add_argument(
        "--vector-backend",
        choices=["chromadb", "pgvector"],
        default="chromadb",
        help="Vector storage backend (default: chromadb)",
    )
    parser.add_argument("--pg-host", type=str, default=None)
    parser.add_argument("--pg-port", type=int, default=None)
    parser.add_argument("--pg-database", type=str, default=None)
    parser.add_argument("--pg-user", type=str, default=None)
    parser.add_argument(
        "--pg-password",
        type=str,
        default=None,
        help="Database password (prefer PGPASSWORD env var instead)",
    )

    args = parser.parse_args()

    # Default logdir
    if args.logdir is None:
        from memory_mcp.config import DATA_FOLDER

        args.logdir = str(Path(DATA_FOLDER) / "tensorboard_logs")

    # Load embeddings
    backend_cfg = {
        "backend_type": args.vector_backend,
        "pg_host": args.pg_host,
        "pg_port": args.pg_port,
        "pg_database": args.pg_database,
        "pg_user": args.pg_user,
        "pg_password": args.pg_password,
    }

    embeddings, metadatas, ids = _load_vectors(**backend_cfg)

    # Load full texts for content preview
    memory_texts = _load_memory_texts(ids, **backend_cfg)

    # Fill in any missing texts from metadata title
    for i, mid in enumerate(ids):
        if mid not in memory_texts or not memory_texts[mid].strip():
            meta = metadatas[i] if i < len(metadatas) else {}
            memory_texts[mid] = str(meta.get("title", ""))

    # Export
    write_tensorboard_embeddings(
        logdir=args.logdir,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
        memory_texts=memory_texts,
    )

    if args.export_only:
        print(f"\nFiles written to {args.logdir}/")
        print(f"Launch manually with:  tensorboard --logdir {args.logdir}")
        return

    # Launch TensorBoard
    launch_tensorboard(
        logdir=args.logdir,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
