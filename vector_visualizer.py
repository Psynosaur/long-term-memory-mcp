#!/usr/bin/env python3
"""
3D Vector Space Visualizer for Long-Term Memory MCP.

Renders memory embeddings as an interactive 3D scatter plot using Plotly.
High-dimensional vectors (384/768D) are reduced to 3D via PCA, t-SNE, or UMAP.

Points are coloured by memory_type or importance.  Hover over any point to
see its title, type, importance, and memory ID.

Supports both ChromaDB and pgvector backends.

Usage:
    python vector_visualizer.py
    python vector_visualizer.py --method tsne
    python vector_visualizer.py --method umap
    python vector_visualizer.py --vector-backend pgvector
    python vector_visualizer.py --colour-by importance
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("visualizer")


def _setup_logging() -> None:
    """Configure logging to both stderr and a file in data/logs/."""
    log_dir = Path(__file__).resolve().parent / "data" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path(__file__).resolve().parent
    log_file = log_dir / "visualizer.log"
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler — always DEBUG
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    # Stderr handler — INFO and above
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)


_setup_logging()


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _require(module_name: str, install_hint: str):
    """Import a module or exit with a helpful install message."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError:
        log.critical(f"Missing required package: {module_name}")
        log.critical(f"Install with: {install_hint}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# Plotly-friendly hex colours for memory_type categories
TYPE_COLOURS: Dict[str, str] = {
    "conversation": "#4286F4",  # blue
    "fact": "#38BE70",  # green
    "preference": "#F4B042",  # amber
    "event": "#DC4E4E",  # red
    "task": "#A85ED6",  # purple
    "ephemeral": "#999999",  # grey
}
DEFAULT_COLOUR = "#808080"


def _importance_hex(importance: int) -> str:
    """Map importance 1-10 to a red-yellow-green hex colour."""
    t = max(0.0, min(1.0, (importance - 1) / 9.0))
    if t < 0.5:
        s = t * 2
        r, g, b = 1.0, s, 0.0
    else:
        s = (t - 0.5) * 2
        r, g, b = 1.0 - s, 1.0, 0.0
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_vectors(
    backend_type: str = "chromadb",
    pg_host: Optional[str] = None,
    pg_port: Optional[int] = None,
    pg_database: Optional[str] = None,
    pg_user: Optional[str] = None,
    pg_password: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str], List[str]]:
    """
    Load all vectors and metadata from the specified backend.

    Returns:
        embeddings: (N, D) float32 array
        metadatas:  list of metadata dicts per vector
        ids:        list of memory IDs
        documents:  list of document text per vector
    """
    from memory_mcp.config import DATA_FOLDER, EMBEDDING_MODEL_CONFIG

    if backend_type == "pgvector":
        from memory_mcp.vector_backends.pgvector_backend import PgvectorBackend

        backend = PgvectorBackend(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password,
            dimensions=EMBEDDING_MODEL_CONFIG["dimensions"],
        )

        initialized = False
        try:
            backend.initialize()
            initialized = True
            count = backend.count()
            if count == 0:
                log.warning("No vectors found in the database.")
                sys.exit(0)

            log.info(f"Loading {count} vectors from {backend.backend_name}...")
            t0 = time.perf_counter()

            data = backend.get(include_embeddings=True)

            ids = data["ids"]
            metadatas = data.get("metadatas") or [{} for _ in range(len(ids))]
            documents = data.get("documents") or ["" for _ in range(len(ids))]
            raw_embeddings = data.get("embeddings")
            if raw_embeddings is None or (
                hasattr(raw_embeddings, "__len__") and len(raw_embeddings) == 0
            ):
                log.warning(
                    "Backend returned no embeddings (try include_embeddings=True)."
                )
                sys.exit(1)

            embeddings = np.array(raw_embeddings, dtype=np.float32)

            if not (len(ids) == len(metadatas) == embeddings.shape[0]):
                log.critical(
                    f"Length mismatch: ids={len(ids)}, metadatas={len(metadatas)}, "
                    f"embeddings={embeddings.shape[0]}"
                )
                sys.exit(1)

            elapsed = time.perf_counter() - t0
            log.info(
                f"Loaded {embeddings.shape[0]} vectors ({embeddings.shape[1]}D) "
                f"in {elapsed:.2f}s"
            )

            return embeddings, metadatas, ids, documents
        finally:
            if initialized:
                backend.close()

    else:
        # ChromaDB — use a lock-free read-only path that reads directly from
        # the SQLite WAL.  This avoids Windows mandatory file locking that
        # causes the visualizer to stall when the MCP server already holds
        # ChromaDB's PersistentClient open.
        from memory_mcp.vector_backends.chroma import ChromaBackend

        db_folder = Path(DATA_FOLDER) / "memory_db"
        log.info("Loading vectors via ChromaDB read-only SQLite path...")
        t0 = time.perf_counter()

        data = ChromaBackend.get_vectors_readonly(db_folder)

        ids = data["ids"]
        metadatas = data.get("metadatas") or [{} for _ in range(len(ids))]
        documents = data.get("documents") or ["" for _ in range(len(ids))]
        raw_embeddings = data.get("embeddings")

        if not ids:
            log.warning("No vectors found in the database.")
            sys.exit(0)

        if raw_embeddings is None or (
            hasattr(raw_embeddings, "__len__") and len(raw_embeddings) == 0
        ):
            log.warning("Backend returned no embeddings.")
            sys.exit(1)

        embeddings = np.array(raw_embeddings, dtype=np.float32)

        if not (len(ids) == len(metadatas) == embeddings.shape[0]):
            log.critical(
                f"Length mismatch: ids={len(ids)}, metadatas={len(metadatas)}, "
                f"embeddings={embeddings.shape[0]}"
            )
            sys.exit(1)

        elapsed = time.perf_counter() - t0
        log.info(
            f"Loaded {embeddings.shape[0]} vectors ({embeddings.shape[1]}D) "
            f"in {elapsed:.2f}s"
        )

        return embeddings, metadatas, ids, documents


def _load_memory_texts(
    ids: List[str],
    backend_type: str = "chromadb",
    pg_host: Optional[str] = None,
    pg_port: Optional[int] = None,
    pg_database: Optional[str] = None,
    pg_user: Optional[str] = None,
    pg_password: Optional[str] = None,
) -> Dict[str, str]:
    """Query the memories table directly for full title + content.

    The vector store's ``document`` field is often just the title or
    truncated.  This function reads the actual database so word
    expansion has access to the complete text.

    Returns:
        {memory_id: "title\\ncontent", ...}
    """
    from memory_mcp.config import DATA_FOLDER

    texts: Dict[str, str] = {}

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

            log.info(
                f"Loaded full text for {len(texts)}/{len(ids)} memories from Postgres."
            )
        except Exception as exc:
            log.warning(f"Warning: could not load memory texts from Postgres: {exc}")
            import traceback

            traceback.print_exc()
    else:
        # SQLite (default / chromadb mode)
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

                log.info(
                    f"Loaded full text for {len(texts)}/{len(ids)} memories from SQLite."
                )
        except Exception as exc:
            log.warning(f"Warning: could not load memory texts from SQLite: {exc}")
            import traceback

            traceback.print_exc()

    return texts


# ---------------------------------------------------------------------------
# Dimensionality reduction  384/768D -> 3D
# ---------------------------------------------------------------------------


def reduce_pca(embeddings: np.ndarray) -> np.ndarray:
    """PCA reduction to 3D. Fast, preserves global variance."""
    from sklearn.decomposition import PCA

    n, d = embeddings.shape
    n_components = min(3, n, d)
    if n_components < 3:
        log.warning(
            f"Warning: only {n} vectors / {d} dims — PCA limited to {n_components}D"
        )

    log.info(f"Running PCA ({d}D -> {n_components}D)...")
    t0 = time.perf_counter()
    reducer = PCA(n_components=n_components, random_state=42)
    result = reducer.fit_transform(embeddings)

    if n_components < 3:
        pad = np.zeros((n, 3 - n_components), dtype=result.dtype)
        result = np.hstack([result, pad])

    variance = sum(reducer.explained_variance_ratio_) * 100
    elapsed = time.perf_counter() - t0
    log.info(f"PCA done in {elapsed:.2f}s (explains {variance:.1f}% variance)")
    return result


def reduce_tsne(embeddings: np.ndarray) -> np.ndarray:
    """t-SNE reduction to 3D. Slower, best for cluster visualisation."""
    from sklearn.manifold import TSNE

    n, d = embeddings.shape
    if n < 4:
        log.warning(
            f"Warning: t-SNE needs at least 4 points (have {n}), falling back to PCA"
        )
        return reduce_pca(embeddings)

    perplexity = min(30.0, max(5.0, n / 4.0))

    log.info(f"Running t-SNE (perplexity={perplexity:.0f}, this may take a while)...")
    t0 = time.perf_counter()

    if d > 50:
        from sklearn.decomposition import PCA

        pca_dim = min(50, n, d)
        embeddings = PCA(n_components=pca_dim, random_state=42).fit_transform(
            embeddings
        )
        log.info(f"  Pre-reduced to {embeddings.shape[1]}D with PCA")

    reducer = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    result = reducer.fit_transform(embeddings)
    elapsed = time.perf_counter() - t0
    log.info(f"t-SNE done in {elapsed:.2f}s")
    return result


def reduce_umap(embeddings: np.ndarray) -> np.ndarray:
    """UMAP reduction to 3D. Fast, preserves global + local structure."""
    umap_mod = _require("umap", "pip install umap-learn")

    n = embeddings.shape[0]
    n_neighbors = min(15, max(2, n - 1))

    log.info(f"Running UMAP (n_neighbors={n_neighbors})...")
    t0 = time.perf_counter()
    reducer = umap_mod.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    result = reducer.fit_transform(embeddings)
    elapsed = time.perf_counter() - t0
    log.info(f"UMAP done in {elapsed:.2f}s")
    return result


REDUCERS = {
    "pca": reduce_pca,
    "tsne": reduce_tsne,
    "umap": reduce_umap,
}


# ---------------------------------------------------------------------------
# Plotly visualisation
# ---------------------------------------------------------------------------


def _build_figure(
    points_3d: np.ndarray,
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    colour_by: str = "memory_type",
    method: str = "pca",
    raw_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Any, int, Dict[str, List[int]], List[str]]:
    """
    Build the Plotly Figure with scatter traces + hidden tag-line traces.

    Returns:
        fig:            plotly.graph_objects.Figure
        n_scatter:      number of scatter (point) traces
        tag_to_indices: {tag_name: [point indices]}
        sorted_tags:    alphabetically sorted tag names
    """
    go = _require("plotly.graph_objects", "pip install plotly")

    n = points_3d.shape[0]

    # ── Build per-point data grouped by type ────────────────────
    type_groups: Dict[str, Dict[str, list]] = {}

    for i, meta in enumerate(metadatas):
        title = str(meta.get("title", ids[i]))
        mtype = str(meta.get("memory_type", "unknown")).lower()
        try:
            imp = int(meta.get("importance", 5))
        except (ValueError, TypeError):
            imp = 5

        if mtype not in type_groups:
            type_groups[mtype] = dict(
                x=[],
                y=[],
                z=[],
                titles=[],
                types=[],
                importances=[],
                mem_ids=[],
                sizes=[],
                imp_colours=[],
            )

        grp = type_groups[mtype]
        grp["x"].append(points_3d[i, 0])
        grp["y"].append(points_3d[i, 1])
        grp["z"].append(points_3d[i, 2])
        grp["sizes"].append(5 + imp * 1.2)
        grp["titles"].append(title)
        grp["types"].append(mtype)
        grp["importances"].append(imp)
        grp["mem_ids"].append(ids[i])
        grp["imp_colours"].append(_importance_hex(imp))

    # ── Axis labels showing absolute bounds ─────────────────────
    axis_labels = ["PC1", "PC2", "PC3"]
    axis_ticks: Dict[str, dict] = {}

    if raw_bounds is not None:
        rmin, rmax = raw_bounds
        for ax_idx, ax_key in enumerate(["xaxis", "yaxis", "zaxis"]):
            lo, hi = float(rmin[ax_idx]), float(rmax[ax_idx])
            axis_labels[ax_idx] = f"Dim {ax_idx + 1}  [{lo:.2f} .. {hi:.2f}]"
            tick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
            tick_text = [f"{lo + t * (hi - lo):.2f}" for t in tick_vals]
            axis_ticks[ax_key] = dict(
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(color="rgb(140,140,150)", size=10),
            )

    # ── Hover template — dark background, white text ──────────────
    _hover_tpl = (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:#aaa'>Type:</span> %{customdata[1]}<br>"
        "<span style='color:#aaa'>Importance:</span> %{customdata[2]}<br>"
        "<span style='color:#aaa'>ID:</span> %{customdata[3]}"
        "<extra></extra>"
    )

    # ── Create one trace per memory_type for legend isolation ───
    traces = []
    for mtype in sorted(type_groups.keys(), key=lambda t: -len(type_groups[t]["x"])):
        grp = type_groups[mtype]
        col = TYPE_COLOURS.get(mtype, DEFAULT_COLOUR)
        count = len(grp["x"])

        if colour_by == "importance":
            marker_color = grp["imp_colours"]
        else:
            marker_color = col

        customdata = list(
            zip(grp["titles"], grp["types"], grp["importances"], grp["mem_ids"])
        )

        traces.append(
            go.Scatter3d(
                x=grp["x"],
                y=grp["y"],
                z=grp["z"],
                mode="markers",
                marker=dict(
                    size=grp["sizes"],
                    color=marker_color,
                    opacity=0.9,
                    line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
                ),
                customdata=customdata,
                hovertemplate=_hover_tpl,
                hoverlabel=dict(
                    bgcolor="rgba(20, 20, 30, 0.95)",
                    bordercolor="rgb(80, 80, 90)",
                    font=dict(color="rgb(240, 240, 240)", size=13),
                ),
                name=f"{mtype} ({count})",
                legendgroup=mtype,
                showlegend=True,
            )
        )

    n_scatter = len(traces)

    # ── Pre-compute tag line traces (all hidden initially) ──────
    from collections import defaultdict

    tag_to_indices: Dict[str, List[int]] = defaultdict(list)

    def _clean_tag(raw: str) -> str:
        """Strip quotes, brackets, and whitespace from a tag value."""
        return raw.strip().strip("\"'[]").strip().lower()

    for i, meta in enumerate(metadatas):
        raw_tags = meta.get("tags", "")
        tags: List[str] = []
        if isinstance(raw_tags, (list, tuple)):
            tags = [_clean_tag(str(t)) for t in raw_tags if str(t).strip()]
        elif isinstance(raw_tags, str) and raw_tags.strip():
            # Tags may be stored as JSON array string: '["foo", "bar"]'
            # or as comma-separated string: 'foo, bar'
            import json

            stripped = raw_tags.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        tags = [_clean_tag(str(t)) for t in parsed if str(t).strip()]
                except (json.JSONDecodeError, ValueError):
                    # Fallback: strip brackets and split
                    stripped = stripped.strip("[]")
                    tags = [_clean_tag(t) for t in stripped.split(",") if t.strip()]
            else:
                tags = [_clean_tag(t) for t in stripped.split(",") if t.strip()]
        # Drop empty strings after cleaning
        tags = [t for t in tags if t]
        for tag in tags:
            tag_to_indices[tag].append(i)

    # Filter: need >= 2 points, drop tags covering > 50% of points
    max_group = int(n * 0.5) if n > 10 else n
    tag_to_indices = {
        tag: indices
        for tag, indices in tag_to_indices.items()
        if 2 <= len(indices) <= max_group
    }

    _line_colours = [
        "rgb(100,160,255)",
        "rgb(100,220,140)",
        "rgb(255,180,80)",
        "rgb(220,100,100)",
        "rgb(180,120,240)",
        "rgb(140,200,220)",
    ]

    sorted_tags = sorted(tag_to_indices.keys())

    for tidx, tag in enumerate(sorted_tags):
        indices = tag_to_indices[tag]
        line_col = _line_colours[tidx % len(_line_colours)]

        lx: List[Optional[float]] = []
        ly: List[Optional[float]] = []
        lz: List[Optional[float]] = []
        for j in range(len(indices)):
            for k in range(j + 1, len(indices)):
                p1, p2 = indices[j], indices[k]
                lx.extend([points_3d[p1, 0], points_3d[p2, 0], None])
                ly.extend([points_3d[p1, 1], points_3d[p2, 1], None])
                lz.extend([points_3d[p1, 2], points_3d[p2, 2], None])

        traces.append(
            go.Scatter3d(
                x=lx,
                y=ly,
                z=lz,
                mode="lines",
                line=dict(color=line_col, width=1.5),
                opacity=0.25,
                name=f"tag: {tag} ({len(indices)})",
                showlegend=False,
                hoverinfo="skip",
                visible=False,
            )
        )

    # ── Build axis dicts ────────────────────────────────────────
    def _axis(idx: int) -> dict:
        ax_key = ["xaxis", "yaxis", "zaxis"][idx]
        base: Dict[str, Any] = dict(
            backgroundcolor="rgb(13, 13, 20)",
            gridcolor="rgb(50, 50, 60)",
            title=dict(
                text=axis_labels[idx],
                font=dict(color="rgb(180,180,190)", size=12),
            ),
            showticklabels=True,
        )
        if ax_key in axis_ticks:
            base.update(axis_ticks[ax_key])
        else:
            base["tickfont"] = dict(color="rgb(140,140,150)", size=10)
        return base

    # ── Layout ──────────────────────────────────────────────────
    layout = go.Layout(
        title=dict(
            text=f"Memory Vector Space — {n} points — {method.upper()}",
            font=dict(color="white", size=18),
            x=0.5,
        ),
        scene=dict(
            bgcolor="rgb(13, 13, 20)",
            xaxis=_axis(0),
            yaxis=_axis(1),
            zaxis=_axis(2),
            # Prevent camera / zoom reset when legend items are toggled
            # or traces are added / removed via Patch().
            uirevision="scene-lock",
        ),
        paper_bgcolor="rgb(13, 13, 20)",
        plot_bgcolor="rgb(13, 13, 20)",
        autosize=True,
        # Preserve camera / zoom / pan across Patch() updates.
        # As long as this value stays constant, Plotly keeps the
        # user's current viewpoint when traces are added or removed.
        uirevision="memory-viz",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            font=dict(color="white", size=12),
            bgcolor="rgba(30,30,40,0.8)",
            bordercolor="rgba(80,80,90,0.5)",
            borderwidth=1,
            itemsizing="constant",
        ),
    )

    fig = go.Figure(data=traces, layout=layout)

    # ── Console summary ─────────────────────────────────────────
    type_counts = {t: len(g["x"]) for t, g in type_groups.items()}
    log.info(f"\n{'=' * 60}")
    log.info(f"Memory Vector Space  |  {n} points  |  {method.upper()}")
    log.info(f"Coloured by: {colour_by}")
    if colour_by == "memory_type":
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            log.info(f"  {t:15s} : {count:4d}")
    else:
        log.info("  1-3: red  |  4-6: yellow  |  7-10: green")
    if sorted_tags:
        log.info(f"Tag lines available: {len(sorted_tags)} tags (searchable dropdown)")
    if raw_bounds is not None:
        rmin, rmax = raw_bounds
        log.info(
            f"Bounds:  X [{rmin[0]:.3f}, {rmax[0]:.3f}]  "
            f"Y [{rmin[1]:.3f}, {rmax[1]:.3f}]  "
            f"Z [{rmin[2]:.3f}, {rmax[2]:.3f}]"
        )
    log.info(f"{'=' * 60}")

    return fig, n_scatter, dict(tag_to_indices), sorted_tags


# ---------------------------------------------------------------------------
# Dash app — interactive tag-line dropdown with text search
# ---------------------------------------------------------------------------

# Dark CSS for the Dash container
_DARK_CSS = """
    /* ── Global dark theme ──────────────────────────────────── */
    html, body, #root, #react-entry-point, ._dash-loading,
    #_dash-app-content {
        background-color: #0d0d14 !important;
        color: #e0e0e0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
        margin: 0; padding: 0;
        height: 100%; overflow: hidden;
    }
    .tag-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 16px;
        background: #1a1a28;
        border-bottom: 1px solid #2a2a3a;
        height: 38px;
        box-sizing: border-box;
        flex-shrink: 0;
    }
    .tag-bar label {
        color: #aaa;
        font-size: 13px;
        white-space: nowrap;
    }
    .tag-dropdown {
        flex: 1;
        min-width: 250px;
        max-width: 500px;
    }
    .query-box {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-left: 8px;
    }
    .sim-bar-panel {
        position: fixed;
        bottom: 12px;
        left: 12px;
        background: rgba(18, 18, 30, 0.92);
        border: 1px solid #3a3a4a;
        border-radius: 6px;
        padding: 8px 10px 4px 10px;
        font-family: 'SF Mono', 'Consolas', 'Courier New', monospace;
        font-size: 11px;
        color: #b0b0c0;
        z-index: 1000;
        pointer-events: auto;
        width: 320px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.5);
    }
    .sim-bar-panel .stats-title {
        color: #d0d0e0;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 4px;
        border-bottom: 1px solid #2a2a3a;
        padding-bottom: 4px;
    }
    .query-input {
        background-color: #1e1e2e !important;
        border: 1px solid #3a3a4a !important;
        color: #e0e0e0 !important;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 13px;
        width: 220px;
        outline: none;
    }
    .query-input:focus {
        border-color: #5a5a7a !important;
    }
    .query-input::placeholder {
        color: #666;
    }
    .query-words-btn {
        background-color: #2a2a3a !important;
        border: 1px solid #3a3a4a !important;
        color: #999 !important;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 12px;
        cursor: pointer;
        white-space: nowrap;
        transition: background-color 0.15s, color 0.15s, border-color 0.15s;
    }
    .query-words-btn:hover {
        background-color: #3a3a5a !important;
        color: #e0e0e0 !important;
        border-color: #5a5a7a !important;
    }
    .query-words-btn.active {
        background-color: #3a4a6a !important;
        border-color: #6a8aba !important;
        color: #c0d0f0 !important;
    }
    .query-limit-label {
        color: #888;
        font-size: 12px;
        white-space: nowrap;
        margin-left: 4px;
    }
    .query-limit {
        background-color: #1e1e2e !important;
        border: 1px solid #3a3a4a !important;
        color: #e0e0e0 !important;
        border-radius: 4px;
        padding: 5px 6px;
        font-size: 13px;
        width: 52px;
        outline: none;
        text-align: center;
        -moz-appearance: textfield;
    }
    .query-limit:focus {
        border-color: #5a5a7a !important;
    }
    .query-limit::-webkit-outer-spin-button,
    .query-limit::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    /* ── Radix-based Dash Dropdown: dark theme ─────────────── */
    /* Dash ≥ 2.x uses Radix UI primitives, NOT react-select.
       DOM structure:
         .dash-dropdown-wrapper
           button.dash-dropdown  (trigger)
             span.dash-dropdown-grid-container.dash-dropdown-trigger
               span.dash-dropdown-value (selected items / placeholder)
               svg.dash-dropdown-trigger-icon
           div[data-radix-popper-content-wrapper]  (floating popup portal)
             div[data-radix-select-content / role="listbox"]
               search input
               "Select All / Deselect All" row
               option items (role="option")
    */

    /* ── Trigger button ────────────────────────────────────── */
    .dash-dropdown-wrapper {
        background-color: transparent !important;
    }
    button.dash-dropdown,
    .dash-dropdown-wrapper button {
        background-color: #1e1e2e !important;
        border: 1px solid #3a3a4a !important;
        color: #e0e0e0 !important;
        border-radius: 4px;
    }
    button.dash-dropdown:hover,
    button.dash-dropdown:focus {
        border-color: #5a5a7a !important;
        outline: none !important;
    }

    /* Trigger inner grid / text */
    .dash-dropdown-grid-container,
    .dash-dropdown-trigger {
        color: #e0e0e0 !important;
    }
    /* Placeholder text */
    .dash-dropdown-placeholder {
        color: #666 !important;
    }
    /* Selected value text */
    .dash-dropdown-value {
        color: #e0e0e0 !important;
    }
    /* Chevron/trigger icon */
    .dash-dropdown-trigger-icon {
        fill: #888 !important;
        color: #888 !important;
    }

    /* ── Selected tag chips inside trigger ──────────────────── */
    .dash-dropdown-label,
    .dash-dropdown-tag {
        background-color: #3a3a5a !important;
        border-color: #5a5a7a !important;
        color: #e0e0e0 !important;
    }
    /* Remove "x" button on chips */
    .dash-dropdown-close,
    button.dash-dropdown-close {
        color: #aaa !important;
    }
    .dash-dropdown-close:hover,
    button.dash-dropdown-close:hover {
        color: #ff6666 !important;
    }

    /* ── Floating popup (Radix popper portal) ──────────────── */
    div[data-radix-popper-content-wrapper] {
        z-index: 9999 !important;
    }
    /* The content div inside the popper */
    div[data-radix-popper-content-wrapper] > div,
    div[data-radix-popper-content-wrapper] > div > div,
    div[data-radix-popper-content-wrapper] [role="listbox"],
    div[data-radix-popper-content-wrapper] [data-radix-select-content] {
        background-color: #1e1e2e !important;
        border: 1px solid #3a3a4a !important;
        color: #e0e0e0 !important;
        border-radius: 4px;
    }

    /* ── Search input inside popup ─────────────────────────── */
    div[data-radix-popper-content-wrapper] input,
    div[data-radix-popper-content-wrapper] input[type="text"],
    div[data-radix-popper-content-wrapper] input[type="search"],
    div[data-radix-popper-content-wrapper] input[role="searchbox"],
    div[data-radix-popper-content-wrapper] input[aria-autocomplete] {
        background-color: #262638 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a3a4a !important;
        border-radius: 3px;
        outline: none !important;
        caret-color: #e0e0e0 !important;
    }
    div[data-radix-popper-content-wrapper] input:focus {
        border-color: #5a5a7a !important;
    }
    div[data-radix-popper-content-wrapper] input::placeholder {
        color: #666 !important;
    }

    /* ── "Select All / Deselect All" row ───────────────────── */
    div[data-radix-popper-content-wrapper] a,
    div[data-radix-popper-content-wrapper] [role="button"],
    div[data-radix-popper-content-wrapper] button:not(.dash-dropdown) {
        color: #8888cc !important;
        background-color: transparent !important;
    }
    div[data-radix-popper-content-wrapper] a:hover,
    div[data-radix-popper-content-wrapper] [role="button"]:hover {
        color: #aaaaee !important;
    }

    /* ── Option items ──────────────────────────────────────── */
    div[data-radix-popper-content-wrapper] [role="option"],
    div[data-radix-popper-content-wrapper] [data-radix-select-item],
    div[data-radix-popper-content-wrapper] li,
    div[data-radix-popper-content-wrapper] .dash-dropdown-option {
        background-color: #1e1e2e !important;
        color: #d0d0d8 !important;
        border-color: transparent !important;
    }
    div[data-radix-popper-content-wrapper] [role="option"]:hover,
    div[data-radix-popper-content-wrapper] [role="option"][data-highlighted],
    div[data-radix-popper-content-wrapper] [data-radix-select-item]:hover,
    div[data-radix-popper-content-wrapper] .dash-dropdown-option:hover {
        background-color: #2e2e45 !important;
        color: #fff !important;
    }
    /* Selected/checked option */
    div[data-radix-popper-content-wrapper] [role="option"][aria-selected="true"],
    div[data-radix-popper-content-wrapper] [role="option"][data-state="checked"],
    div[data-radix-popper-content-wrapper] [data-radix-select-item][data-state="checked"] {
        background-color: #2a2a4a !important;
        color: #fff !important;
    }

    /* ── Checkboxes inside options ──────────────────────────── */
    div[data-radix-popper-content-wrapper] input[type="checkbox"] {
        accent-color: #6c63ff;
    }

    /* ── Labels, spans, divs inside popup — force dark ─────── */
    div[data-radix-popper-content-wrapper] span,
    div[data-radix-popper-content-wrapper] label {
        color: #d0d0d8 !important;
    }
    div[data-radix-popper-content-wrapper] div {
        background-color: #1e1e2e !important;
    }

    /* ── No results message ────────────────────────────────── */
    div[data-radix-popper-content-wrapper] [class*="noOptions"],
    div[data-radix-popper-content-wrapper] [class*="no-results"] {
        background-color: #1e1e2e !important;
        color: #888 !important;
    }

    /* ── Scrollbar inside popup ─────────────────────────────── */
    div[data-radix-popper-content-wrapper] ::-webkit-scrollbar {
        width: 8px;
    }
    div[data-radix-popper-content-wrapper] ::-webkit-scrollbar-track {
        background: #1e1e2e;
    }
    div[data-radix-popper-content-wrapper] ::-webkit-scrollbar-thumb {
        background: #3a3a5a;
        border-radius: 4px;
    }

    /* ── Counter badge on trigger ───────────────────────────── */
    .tag-selector-value-count {
        color: #aaa !important;
    }

    /* ── Reduction method dropdown ──────────────────────────── */
    .method-dropdown {
        width: 90px;
        flex-shrink: 0;
    }

    /* ── Date picker ─────────────────────────────────────────── */
    .date-picker-box {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-shrink: 0;
    }
    .date-picker-box label {
        color: #aaa;
        font-size: 13px;
        white-space: nowrap;
    }
    .SingleDatePickerInput,
    .SingleDatePickerInput__withBorder {
        background-color: #1e1e2e !important;
        border: 1px solid #3a3a4a !important;
        border-radius: 4px !important;
    }
    .DateInput,
    .DateInput_input {
        background-color: #1e1e2e !important;
        color: #e0e0e0 !important;
        font-size: 12px !important;
        border-bottom: none !important;
        width: 90px !important;
        padding: 4px 6px !important;
    }
    .DateInput_input::placeholder { color: #555 !important; }
    .DayPicker, .DayPicker__withBorder, .DayPicker_transitionContainer,
    .CalendarMonthGrid, .CalendarMonth, .CalendarMonthGrid_month__horizontal {
        background-color: #1e1e2e !important;
        color: #e0e0e0 !important;
    }
    .CalendarMonth_caption, .CalendarMonth_caption strong {
        color: #d0d0e0 !important;
    }
    .DayPickerNavigation_button {
        border: 1px solid #3a3a4a !important;
        background-color: #1e1e2e !important;
    }
    .DayPickerNavigation_svg { fill: #aaa !important; }
    .DayPickerNavigation_button:hover { background-color: #2e2e45 !important; }
    .CalendarDay__default {
        background-color: #1e1e2e !important;
        color: #c0c0d0 !important;
        border: 1px solid #2a2a3a !important;
    }
    .CalendarDay__default:hover {
        background-color: #2e2e45 !important;
        color: #fff !important;
    }
    .CalendarDay__selected, .CalendarDay__selected:hover {
        background-color: #3a4a6a !important;
        border-color: #6a8aba !important;
        color: #fff !important;
    }
    .CalendarDay__today { color: #F4B042 !important; font-weight: bold !important; }
    .CalendarDay__outside,
    .CalendarDay__blocked_out_of_range,
    .CalendarDay__blocked_out_of_range:hover {
        background-color: #181826 !important;
        color: #444 !important;
        cursor: default !important;
    }
    .DayPickerKeyboardShortcuts_buttonReset,
    .DayPickerKeyboardShortcuts_show { display: none !important; }
    .date-clear-btn {
        background-color: #2a2a3a !important;
        border: 1px solid #3a3a4a !important;
        color: #888 !important;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 11px;
        cursor: pointer;
        white-space: nowrap;
        line-height: 1;
    }
    .date-clear-btn:hover {
        background-color: #3a3a5a !important;
        color: #e0e0e0 !important;
        border-color: #5a5a7a !important;
    }

    /* ── Loading spinner overlay ─────────────────────────────── */
    .graph-loading-wrapper {
        height: calc(100vh - 38px);
        position: relative;
    }
    /* Keep the graph visible underneath — override Dash default hide */
    .graph-loading-wrapper > div[class*="dash-loading"] {
        position: absolute !important;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 999;
        pointer-events: none;
    }
    .graph-loading-wrapper > div[class*="dash-loading"] > div {
        visibility: visible !important;
        opacity: 1 !important;
    }
    /* The spinner itself — centered on top of the graph */
    .graph-loading-wrapper .dash-spinner,
    .graph-loading-wrapper ._dash-loading-callback {
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 1000;
    }
    /* Prevent Dash from hiding graph children during load */
    .graph-loading-wrapper .dash-graph--pending,
    .graph-loading-wrapper [data-dash-is-loading="true"] > .js-plotly-plot {
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* ── Cluster stats overlay panel ─────────────────────────── */
    .cluster-stats-panel {
        position: fixed;
        bottom: 12px;
        right: 12px;
        background: rgba(18, 18, 30, 0.92);
        border: 1px solid #3a3a4a;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: 'SF Mono', 'Consolas', 'Courier New', monospace;
        font-size: 11px;
        color: #b0b0c0;
        line-height: 1.6;
        z-index: 1000;
        pointer-events: auto;
        min-width: 280px;
        max-width: 360px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.5);
    }
    .cluster-stats-panel .stats-title {
        color: #d0d0e0;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
        border-bottom: 1px solid #2a2a3a;
        padding-bottom: 4px;
    }
    .cluster-stats-panel .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 1px 0;
    }
    .cluster-stats-panel .stat-label {
        color: #888;
    }
    .cluster-stats-panel .stat-value {
        color: #c8c8e0;
        font-weight: 500;
    }
    .cluster-stats-panel .stat-value.good {
        color: #38BE70;
    }
    .cluster-stats-panel .stat-value.moderate {
        color: #F4B042;
    }
    .cluster-stats-panel .stat-value.bad {
        color: #DC4E4E;
    }
    .cluster-stats-panel .stat-info {
        display: inline-block;
        margin-left: 4px;
        color: #555;
        font-size: 10px;
        cursor: help;
        vertical-align: middle;
        line-height: 1;
    }
    .cluster-stats-panel .stat-info:hover {
        color: #aaa;
    }
    .cluster-stats-panel .stat-warning {
        margin-top: 6px;
        padding-top: 4px;
        border-top: 1px solid #2a2a3a;
        font-size: 10px;
        color: #999;
        font-style: italic;
    }
"""


# Module-level cache for the PCA projector + normalisation bounds so the
# word-expansion callback can project new 384D vectors into the same 3D
# space without re-fitting.  Updated each time _fetch_and_build() runs.
_projector_cache: Dict[str, Any] = {}

# Cache the raw high-D embeddings and their IDs for semantic query.
# Populated by _fetch_and_build() on every page load.
_embeddings_cache: Dict[str, Any] = {}


def _project_new_vectors(high_d: np.ndarray) -> np.ndarray:
    """Project new high-D vectors into the current 3D unit-cube space.

    Uses the PCA + normalisation bounds cached by the last
    ``_fetch_and_build()`` call.  Returns (N, 3) float32.
    """
    cache = _projector_cache
    if "pca" not in cache:
        raise RuntimeError("No projector available — call _fetch_and_build first.")

    pca = cache["pca"]
    raw_min = cache["raw_min"]
    span = cache["span"]

    pts = pca.transform(high_d)
    # Pad to 3D if PCA returned fewer components
    if pts.shape[1] < 3:
        pad = np.zeros((pts.shape[0], 3 - pts.shape[1]), dtype=pts.dtype)
        pts = np.hstack([pts, pad])
    # Normalise into the same unit-cube
    pts = (pts - raw_min) / span
    return pts.astype(np.float32)


# ---------------------------------------------------------------------------
# Cluster / density metrics for the 384D embedding space
# ---------------------------------------------------------------------------


def _compute_cluster_stats(embeddings: np.ndarray) -> Dict[str, Any]:
    """Compute density and clustering metrics on the raw high-D embeddings.

    All metrics are computed in the original embedding space (e.g. 384D),
    NOT in the projected 3D space.  This gives an honest picture of how
    tightly packed the vectors are and therefore how reliable top-K
    nearest-neighbour searches will be.

    Returns a dict with:
        n_vectors:        number of vectors
        dimensions:       embedding dimensionality
        avg_cosine_sim:   mean pairwise cosine similarity (0-1, higher = denser)
        std_cosine_sim:   std dev of pairwise similarities
        min_cosine_sim:   minimum pairwise similarity
        max_cosine_sim:   maximum pairwise similarity
        hopkins:          Hopkins statistic (0.5 = uniform, >0.7 = clusterable)
        silhouette:       silhouette score from KMeans auto-clustering (-1 to 1)
        n_clusters:       number of clusters used for silhouette
        nn_gap_ratio:     median ratio of (dist_rank2 - dist_rank1) / dist_rank1
                          (higher = clearer separation between nearest and 2nd nearest)
        top_k_warning:    human-readable assessment of top-K reliability
    """
    n, d = embeddings.shape
    stats: Dict[str, Any] = {"n_vectors": n, "dimensions": d}

    if n < 3:
        stats.update(
            {
                "avg_cosine_sim": None,
                "std_cosine_sim": None,
                "min_cosine_sim": None,
                "max_cosine_sim": None,
                "hopkins": None,
                "silhouette": None,
                "n_clusters": None,
                "nn_gap_ratio": None,
                "top_k_warning": "Too few vectors for analysis",
            }
        )
        return stats

    t0 = time.perf_counter()

    # ── Pairwise cosine similarity ───────────────────────────────
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms
    # (N, N) similarity matrix — use float32 to keep memory reasonable
    sim_matrix = normed @ normed.T
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[triu_idx]

    stats["avg_cosine_sim"] = round(float(np.mean(pairwise_sims)), 4)
    stats["std_cosine_sim"] = round(float(np.std(pairwise_sims)), 4)
    stats["min_cosine_sim"] = round(float(np.min(pairwise_sims)), 4)
    stats["max_cosine_sim"] = round(float(np.max(pairwise_sims)), 4)

    # ── Hopkins statistic (clustering tendency) ──────────────────
    # Sample m points from the dataset.  For each, find the nearest
    # neighbour distance.  Also generate m random points in the data's
    # bounding box and find their nearest real-point distance.
    # H = sum(rand_nn) / (sum(rand_nn) + sum(data_nn))
    # H ~ 0.5 => uniform, H > 0.7 => clusterable.
    try:
        from sklearn.neighbors import NearestNeighbors

        m = min(50, n // 2)  # sample size
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, size=m, replace=False)
        sample = embeddings[sample_idx]

        # Fit NN on all points
        nn = NearestNeighbors(n_neighbors=2, metric="cosine")
        nn.fit(embeddings)
        # Distance to nearest neighbour for sampled real points
        # (k=2 because the point itself is distance 0)
        dists_real, _ = nn.kneighbors(sample)
        data_nn = dists_real[:, 1]  # second column = nearest OTHER point

        # Generate random points in the bounding box
        mins = embeddings.min(axis=0)
        maxs = embeddings.max(axis=0)
        random_pts = rng.uniform(mins, maxs, size=(m, d)).astype(np.float32)
        dists_rand, _ = nn.kneighbors(random_pts)
        rand_nn = dists_rand[:, 0]  # nearest real point to each random point

        sum_rand = float(np.sum(rand_nn))
        sum_data = float(np.sum(data_nn))
        hopkins = sum_rand / (sum_rand + sum_data) if (sum_rand + sum_data) > 0 else 0.5
        stats["hopkins"] = round(hopkins, 4)
    except Exception as exc:
        log.warning("Hopkins statistic failed: %s", exc)
        stats["hopkins"] = None

    # ── Silhouette score (auto-clustered via KMeans) ─────────────
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Try a few k values, pick the one with the best silhouette
        best_sil = -2.0
        best_k = 2
        max_k = min(10, n - 1)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            labels = km.fit_predict(embeddings)
            sil = silhouette_score(
                embeddings, labels, metric="cosine", sample_size=min(300, n)
            )
            if sil > best_sil:
                best_sil = sil
                best_k = k
        stats["silhouette"] = round(float(best_sil), 4)
        stats["n_clusters"] = best_k
    except Exception as exc:
        log.warning("Silhouette score failed: %s", exc)
        stats["silhouette"] = None
        stats["n_clusters"] = None

    # ── Nearest-neighbour gap ratio ──────────────────────────────
    # For each point, compute distance to rank-1 and rank-2 neighbours.
    # gap_ratio = (d2 - d1) / d1.  High ratio means rank-1 is clearly
    # closer than rank-2, so top-K retrieval is discriminative.
    try:
        from sklearn.neighbors import NearestNeighbors as _NN3

        nn3 = _NN3(n_neighbors=3, metric="cosine")
        nn3.fit(embeddings)
        dists_all, _ = nn3.kneighbors(embeddings)
        d1 = dists_all[:, 1]  # nearest other
        d2 = dists_all[:, 2]  # 2nd nearest
        # Avoid division by zero
        safe_d1 = np.maximum(d1, 1e-10)
        gap_ratios = (d2 - d1) / safe_d1
        stats["nn_gap_ratio"] = round(float(np.median(gap_ratios)), 4)
    except Exception as exc:
        log.warning("NN gap ratio failed: %s", exc)
        stats["nn_gap_ratio"] = None

    # ── Top-K reliability assessment ─────────────────────────────
    avg_sim = stats.get("avg_cosine_sim", 0) or 0
    hopkins_val = stats.get("hopkins", 0.5) or 0.5
    sil_val = stats.get("silhouette", 0) or 0
    gap = stats.get("nn_gap_ratio", 0) or 0

    if avg_sim > 0.85:
        warning = "Dense: vectors tightly packed, top-K results may be noisy"
    elif avg_sim > 0.7:
        warning = "Moderate: some discrimination, threshold tuning recommended"
    elif avg_sim > 0.5:
        warning = "Good: reasonable separation for top-K retrieval"
    else:
        warning = "Excellent: well-separated vectors, top-K highly reliable"

    if hopkins_val < 0.55:
        warning += " | No natural clusters"
    elif hopkins_val > 0.75:
        warning += " | Strong clustering tendency"

    stats["top_k_warning"] = warning

    elapsed = time.perf_counter() - t0
    log.info(
        "Cluster stats computed in %.2fs: avg_sim=%.4f, hopkins=%.4f, "
        "silhouette=%.4f (k=%s), nn_gap=%.4f",
        elapsed,
        stats.get("avg_cosine_sim", 0),
        stats.get("hopkins", 0),
        stats.get("silhouette", 0),
        stats.get("n_clusters"),
        stats.get("nn_gap_ratio", 0),
    )
    return stats


def _build_stats_panel_children() -> list:
    """Build Dash HTML children for the cluster stats overlay panel.

    Reads from ``_embeddings_cache`` (populated by ``_fetch_and_build``).
    Returns a list of ``html.Div`` elements ready to drop into the panel.
    """
    from dash import html as _html

    emb = _embeddings_cache.get("embeddings")
    if emb is None or len(emb) < 3:
        return [_html.Div("Not enough vectors for analysis", className="stat-warning")]

    try:
        stats = _compute_cluster_stats(emb)
    except Exception as exc:
        log.error("Cluster stats computation failed: %s", exc, exc_info=True)
        return [_html.Div(f"Stats error: {exc}", className="stat-warning")]

    def _row(label: str, value, css_class: str = "", tooltip: str = "") -> Any:
        label_children: list = [_html.Span(label, className="stat-label")]
        if tooltip:
            label_children.append(
                _html.Span(
                    "ⓘ",
                    className="stat-info",
                    title=tooltip,
                )
            )
        return _html.Div(
            className="stat-row",
            children=[
                _html.Span(children=label_children),
                _html.Span(
                    str(value) if value is not None else "n/a",
                    className=f"stat-value {css_class}".strip(),
                ),
            ],
        )

    # Colour-code the average similarity
    avg_sim = stats.get("avg_cosine_sim")
    if avg_sim is not None:
        if avg_sim > 0.85:
            sim_class = "bad"
        elif avg_sim > 0.7:
            sim_class = "moderate"
        else:
            sim_class = "good"
    else:
        sim_class = ""

    # Colour-code the silhouette score
    sil = stats.get("silhouette")
    if sil is not None:
        if sil > 0.4:
            sil_class = "good"
        elif sil > 0.15:
            sil_class = "moderate"
        else:
            sil_class = "bad"
    else:
        sil_class = ""

    # Colour-code Hopkins
    hop = stats.get("hopkins")
    if hop is not None:
        if hop > 0.75:
            hop_class = "good"
        elif hop > 0.6:
            hop_class = "moderate"
        else:
            hop_class = "bad"
    else:
        hop_class = ""

    # Colour-code NN gap
    gap = stats.get("nn_gap_ratio")
    if gap is not None:
        if gap > 0.3:
            gap_class = "good"
        elif gap > 0.1:
            gap_class = "moderate"
        else:
            gap_class = "bad"
    else:
        gap_class = ""

    rows = [
        _row("Vectors", f"{stats['n_vectors']} x {stats['dimensions']}D"),
        _row(
            "Avg Cosine Sim",
            avg_sim,
            sim_class,
            tooltip=(
                "Average cosine similarity between all memory pairs.\n"
                "Range: 0.0 (completely different) to 1.0 (identical).\n"
                "Low (<0.55): diverse, well-spread memories — good for retrieval.\n"
                "Moderate (0.55–0.70): normal for a focused project.\n"
                "High (>0.70): memories are too similar — semantic search may\n"
                "  struggle to discriminate; consider pruning duplicates."
            ),
        ),
        _row(
            "Sim Range",
            f"{stats.get('min_cosine_sim', '?')} .. {stats.get('max_cosine_sim', '?')}",
            tooltip=(
                "Min and max pairwise cosine similarity across all memories.\n"
                "A wide range (e.g. 0.3 .. 0.98) means good topic diversity.\n"
                "Values near 0.98+ indicate near-duplicate memories that may\n"
                "  be candidates for merging or deletion.\n"
                "Values near 1.0 = identical content stored twice."
            ),
        ),
        _row(
            "Std Dev",
            stats.get("std_cosine_sim"),
            tooltip=(
                "Standard deviation of pairwise cosine similarity.\n"
                "High (>0.10): large spread — memories cover very different topics.\n"
                "Low (<0.05): tight cluster — memories are all about similar things,\n"
                "  which can hurt recall precision for unrelated queries."
            ),
        ),
        _row(
            "Hopkins Stat",
            hop,
            hop_class,
            tooltip=(
                "Hopkins statistic: measures clustering tendency of the vector space.\n"
                "Range: 0.5 to 1.0.\n"
                "~0.5: points are uniformly distributed (no structure).\n"
                "0.6–0.75: moderate clustering tendency.\n"
                ">0.75: strong clustering — memories group into distinct topics.\n"
                "  This is GOOD: it means semantic search can isolate relevant clusters."
            ),
        ),
        _row(
            "Silhouette",
            f"{sil} (k={stats.get('n_clusters', '?')})" if sil is not None else None,
            sil_class,
            tooltip=(
                "Silhouette score: how well-separated the clusters are (k = auto-detected).\n"
                "Range: -1.0 to 1.0.\n"
                ">0.4: clusters are well-separated — topic boundaries are clear.\n"
                "0.15–0.40: overlapping clusters — memories blend across topics.\n"
                "<0.15: poor separation — the k clusters are not meaningful.\n"
                "Low score with high Hopkins = structure exists but clusters overlap;\n"
                "  try more specific tags or split broad memories into focused ones."
            ),
        ),
        _row(
            "NN Gap Ratio",
            gap,
            gap_class,
            tooltip=(
                "Nearest-neighbour gap ratio: how isolated each memory is from its neighbours.\n"
                "Range: 0.0 to 1.0.\n"
                ">0.30: good separation — each memory has clear breathing room.\n"
                "0.10–0.30: moderate — neighbours are close but still distinguishable.\n"
                "<0.10: memories are tightly packed — top-K retrieval will return\n"
                "  near-identical neighbours, reducing result diversity.\n"
                "Low values often indicate duplicate or near-duplicate memories."
            ),
        ),
    ]

    # Explained variance — how much of the full 384D structure the 3D view captures
    pca_var = _embeddings_cache.get("pca_explained_variance")
    if pca_var is not None:
        if pca_var >= 50:
            var_class = "good"
        elif pca_var >= 25:
            var_class = "moderate"
        else:
            var_class = "bad"
        rows.append(
            _row(
                "3D Captures",
                f"{pca_var:.1f}%",
                var_class,
                tooltip=(
                    "How much of the full 384D variance the 3D projection captures.\n"
                    "This tells you how much to trust the visual layout.\n"
                    ">50%: the 3D view is a reasonable representation.\n"
                    "25–50%: moderate — clusters are visible but distances are approximate.\n"
                    "<25%: the 3D view is misleading — use cosine similarity (query bar)\n"
                    "  for ground-truth distances rather than visual proximity.\n"
                    "Low % is normal for diverse, multi-topic memory sets."
                ),
            )
        )

    warning_text = stats.get("top_k_warning", "")
    if warning_text:
        rows.append(_html.Div(warning_text, className="stat-warning"))

    return rows


def _fetch_and_build(
    backend_type: str,
    method: str,
    colour_by: str,
    pg_host: Optional[str] = None,
    pg_port: Optional[int] = None,
    pg_database: Optional[str] = None,
    pg_user: Optional[str] = None,
    pg_password: Optional[str] = None,
) -> Tuple[
    Any,
    int,
    Dict[str, List[int]],
    List[str],
    List[Dict[str, Any]],
    List[str],
    Dict[str, str],
]:
    """
    Full pipeline: load vectors -> reduce -> normalise -> build figure.

    Called on every page load so the visualisation always reflects the
    current state of the database.

    Returns:
        fig, n_scatter, tag_to_indices, sorted_tags, metadatas, ids
    """
    embeddings, metadatas, ids, documents = load_vectors(
        backend_type=backend_type,
        pg_host=pg_host,
        pg_port=pg_port,
        pg_database=pg_database,
        pg_user=pg_user,
        pg_password=pg_password,
    )

    # Always fit a PCA on the raw embeddings for word-vector projection,
    # regardless of which reduction method the user chose for display.
    from sklearn.decomposition import PCA as _PCA

    n, d = embeddings.shape
    n_comp = min(3, n, d)
    pca_proj = _PCA(n_components=n_comp, random_state=42)
    pca_proj.fit(embeddings)

    # Run the user's chosen reduction method
    reducer = REDUCERS[method]
    points_3d = reducer(embeddings)

    if not np.isfinite(points_3d).all():
        log.warning(
            "Warning: dimensionality reduction produced non-finite values, "
            "falling back to PCA."
        )
        points_3d = reduce_pca(embeddings)

    raw_min = points_3d.min(axis=0).copy()
    raw_max = points_3d.max(axis=0).copy()
    span = raw_max - raw_min
    span[np.isclose(span, 0)] = 1.0
    points_3d = (points_3d - raw_min) / span

    # For PCA method the display PCA == the projector PCA.
    # For t-SNE/UMAP we use the dedicated PCA fitted above so that
    # word vectors land in approximately the same region as their parent
    # (not perfectly, but spatially consistent).
    if method == "pca":
        # Re-use the fitted PCA from the reduction so the projections
        # are identical, not merely approximately equal.
        pca_proj = _PCA(n_components=n_comp, random_state=42)
        pca_proj.fit(embeddings)  # same result (deterministic)

    _projector_cache["pca"] = pca_proj
    _projector_cache["raw_min"] = raw_min
    _projector_cache["span"] = span

    # Store explained variance of the 3-component projection.
    # For PCA method, use the fitted reducer directly.
    # For t-SNE/UMAP, use the dedicated pca_proj (always fitted on raw embeddings).
    try:
        _embeddings_cache["pca_explained_variance"] = float(
            sum(pca_proj.explained_variance_ratio_) * 100
        )
    except Exception:
        _embeddings_cache["pca_explained_variance"] = None

    # Cache raw embeddings + ids for semantic query (cosine similarity
    # in the original 384D space, not the projected 3D space).
    _embeddings_cache["embeddings"] = embeddings  # (N, 384) float32
    _embeddings_cache["ids"] = ids

    fig, n_scatter, tag_to_indices, sorted_tags = _build_figure(
        points_3d,
        metadatas,
        ids,
        colour_by=colour_by,
        method=method,
        raw_bounds=(raw_min, raw_max),
    )

    # Load full memory texts from the memories table (not the vector
    # store's document field which may be incomplete).
    memory_texts = _load_memory_texts(
        ids,
        backend_type=backend_type,
        pg_host=pg_host,
        pg_port=pg_port,
        pg_database=pg_database,
        pg_user=pg_user,
        pg_password=pg_password,
    )

    return fig, n_scatter, tag_to_indices, sorted_tags, metadatas, ids, memory_texts


def run_dash_app(
    backend_cfg: Dict[str, Any],
    method: str,
    colour_by: str,
    port: int = 8050,
) -> None:
    """
    Launch a Dash app that fetches live data on every page load.

    ``backend_cfg`` carries the vector-backend connection parameters so
    each refresh can re-query the database.
    """
    dash = _require("dash", "pip install dash")
    from dash import dcc, html, Input, Output, State, Patch, no_update, callback_context

    app = dash.Dash(
        __name__,
        title="Memory Vector Space",
        update_title=None,
    )

    app.index_string = (
        """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>"""
        + _DARK_CSS
        + """</style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
    <script>
    // Force-dark Radix popper popups that Dash renders with inline styles.
    // CSS handles most cases, but Radix applies inline background-color on
    // the popper wrapper and some children that override our stylesheet.
    (function() {
        var BG   = '#1e1e2e';
        var BG2  = '#262638';
        var FG   = '#d0d0d8';
        var BDR  = '#3a3a4a';
        var LINK = '#8888cc';

        function isLight(el) {
            var cs = window.getComputedStyle(el);
            var bg = cs.backgroundColor;
            if (!bg || bg === 'transparent' || bg === 'rgba(0, 0, 0, 0)') return false;
            var m = bg.match(/\\d+/g);
            if (m && m.length >= 3) {
                var r = parseInt(m[0]), g = parseInt(m[1]), b = parseInt(m[2]);
                return (r > 180 && g > 180 && b > 180);
            }
            return false;
        }

        function isDark(colorStr) {
            if (!colorStr) return false;
            var m = colorStr.match(/\\d+/g);
            if (m && m.length >= 3) {
                var r = parseInt(m[0]), g = parseInt(m[1]), b = parseInt(m[2]);
                return (r < 80 && g < 80 && b < 80);
            }
            return false;
        }

        function darkifyPopper(root) {
            if (!root || !root.querySelectorAll) return;
            // Force dark bg on all containers
            root.querySelectorAll('div, span, label, li, ul').forEach(function(el) {
                if (isLight(el)) {
                    el.style.backgroundColor = BG;
                }
            });
            // Inputs (search box)
            root.querySelectorAll('input').forEach(function(el) {
                el.style.backgroundColor = BG2;
                el.style.color = FG;
                el.style.borderColor = BDR;
                el.style.caretColor = FG;
                el.style.outline = 'none';
            });
            // Force light text on dark-text elements
            root.querySelectorAll('span, label, div, a, button').forEach(function(el) {
                var cs = window.getComputedStyle(el);
                if (isDark(cs.color)) {
                    el.style.color = FG;
                }
            });
            // Links / buttons (Select All, Deselect All)
            root.querySelectorAll('a').forEach(function(el) {
                el.style.color = LINK;
            });
            // Also force the root wrapper itself
            if (isLight(root)) {
                root.style.backgroundColor = BG;
            }
        }

        // Watch for Radix popper wrappers being added to the DOM
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mut) {
                mut.addedNodes.forEach(function(node) {
                    if (node.nodeType !== 1) return;
                    // Radix renders: div[data-radix-popper-content-wrapper]
                    if (node.hasAttribute && node.hasAttribute('data-radix-popper-content-wrapper')) {
                        darkifyPopper(node);
                        setTimeout(function() { darkifyPopper(node); }, 20);
                        setTimeout(function() { darkifyPopper(node); }, 150);
                    }
                    // Or check descendants (sometimes an outer div wraps it)
                    var poppers = node.querySelectorAll && node.querySelectorAll('[data-radix-popper-content-wrapper]');
                    if (poppers) {
                        poppers.forEach(function(p) {
                            darkifyPopper(p);
                            setTimeout(function() { darkifyPopper(p); }, 20);
                            setTimeout(function() { darkifyPopper(p); }, 150);
                        });
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // Catch popper open on click (Radix may reuse existing popper nodes)
        document.addEventListener('click', function(e) {
            var btn = e.target.closest && e.target.closest('button.dash-dropdown');
            if (btn) {
                [20, 80, 200].forEach(function(ms) {
                    setTimeout(function() {
                        document.querySelectorAll('[data-radix-popper-content-wrapper]').forEach(darkifyPopper);
                    }, ms);
                });
            }
        });
    })();
    </script>
</body>
</html>"""
    )

    # ── Layout ──────────────────────────────────────────────────
    # dcc.Location fires its callback on every page load / refresh.
    # dcc.Store holds per-session state (n_scatter, sorted_tags,
    # metadatas/ids for word expansion, expanded memory set).
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="figure-meta", storage_type="memory"),
            dcc.Store(id="expanded-memories", data=[], storage_type="memory"),
            dcc.Store(id="query-active", data=False, storage_type="memory"),
            dcc.Store(id="query-match-ids", data=[], storage_type="memory"),
            dcc.Store(id="query-expanded-ids", data=[], storage_type="memory"),
            dcc.Store(id="lines-visible", data=True, storage_type="memory"),
            dcc.Store(id="word-paths-visible", data=False, storage_type="memory"),
            dcc.Store(id="words-visible", data=True, storage_type="memory"),
            dcc.Store(id="date-picker-active", data=False, storage_type="memory"),
            html.Div(
                className="tag-bar",
                children=[
                    html.Label("Tag Lines:"),
                    html.Div(
                        className="tag-dropdown",
                        children=[
                            dcc.Dropdown(
                                id="tag-selector",
                                options=[],
                                value=[],
                                multi=True,
                                searchable=True,
                                placeholder="Type to search tags...",
                                clearable=True,
                            ),
                        ],
                    ),
                    html.Div(
                        className="query-box",
                        children=[
                            dcc.Input(
                                id="query-input",
                                type="text",
                                placeholder="Semantic search...",
                                debounce=True,
                                n_submit=0,
                                className="query-input",
                            ),
                            html.Span("Limit:", className="query-limit-label"),
                            dcc.Input(
                                id="query-limit",
                                type="number",
                                value=10,
                                min=1,
                                max=100,
                                className="query-limit",
                            ),
                            html.Button(
                                "Clear",
                                id="clear-search-btn",
                                n_clicks=0,
                                className="query-words-btn",
                                title="Clear semantic search and query-expanded words",
                            ),
                            html.Button(
                                "Query Words",
                                id="query-expand-words-btn",
                                n_clicks=0,
                                className="query-words-btn",
                                title="Expand word vectors for query matches",
                            ),
                            html.Button(
                                "Words",
                                id="toggle-words-visible-btn",
                                n_clicks=0,
                                className="query-words-btn active",
                                title="Show / hide all word projections",
                            ),
                            html.Button(
                                "Reset",
                                id="reset-words-btn",
                                n_clicks=0,
                                className="query-words-btn",
                                title="Collapse and remove all expanded words",
                            ),
                            html.Button(
                                "Lines",
                                id="toggle-lines-btn",
                                n_clicks=0,
                                className="query-words-btn active",
                                title="Show / hide all connection lines",
                            ),
                            html.Button(
                                "Word Paths",
                                id="toggle-word-paths-btn",
                                n_clicks=0,
                                className="query-words-btn",
                                title="Show thick starburst lines through shared words",
                            ),
                            html.Span("Min:", className="query-limit-label"),
                            dcc.Input(
                                id="word-path-min-input",
                                type="number",
                                value=2,
                                min=2,
                                max=50,
                                debounce=True,
                                className="query-limit",
                            ),
                        ],
                    ),
                    html.Span(
                        "  |  ",
                        style={"color": "#444", "marginLeft": "8px"},
                    ),
                    html.Div(
                        className="date-picker-box",
                        children=[
                            html.Label("Date:"),
                            dcc.DatePickerSingle(
                                id="date-picker",
                                placeholder="Filter by date",
                                display_format="YYYY-MM-DD",
                                clearable=True,
                                with_portal=False,
                                first_day_of_week=1,
                                style={"fontSize": "12px"},
                            ),
                            html.Button(
                                "Clear Date",
                                id="clear-date-btn",
                                n_clicks=0,
                                className="date-clear-btn",
                                title="Clear date filter",
                            ),
                        ],
                    ),
                    html.Span(
                        "  |  ",
                        style={"color": "#444", "marginLeft": "8px"},
                    ),
                    html.Div(
                        className="method-dropdown",
                        children=[
                            dcc.Dropdown(
                                id="reduction-method",
                                options=[
                                    {"label": "PCA", "value": "pca"},
                                    {"label": "t-SNE", "value": "tsne"},
                                    {"label": "UMAP", "value": "umap"},
                                ],
                                value=method,
                                multi=False,
                                searchable=False,
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Span(
                        "  |  Click point to expand words",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginLeft": "12px",
                            "whiteSpace": "nowrap",
                        },
                    ),
                ],
            ),
            dcc.Loading(
                id="graph-loading",
                type="circle",
                color="#6a8aba",
                delay_show=300,
                overlay_style={"visibility": "visible", "opacity": 0.5},
                className="graph-loading-wrapper",
                children=[
                    dcc.Graph(
                        id="scatter-3d",
                        figure={},
                        style={"height": "calc(100vh - 38px)"},
                        responsive=True,
                        config=dict(
                            displayModeBar=True,
                            displaylogo=False,
                            scrollZoom=True,
                        ),
                    ),
                ],
            ),
            html.Div(
                id="cluster-stats",
                className="cluster-stats-panel",
                children=[
                    html.Div("Vector Space Density", className="stats-title"),
                    html.Div("Loading...", id="cluster-stats-body"),
                ],
            ),
            html.Div(
                id="sim-bar-panel",
                className="sim-bar-panel",
                style={"display": "none"},
                children=[
                    html.Div("Query Similarity (384D)", className="stats-title"),
                    dcc.Graph(
                        id="sim-bar-chart",
                        config={"displayModeBar": False},
                        style={"height": "200px"},
                    ),
                ],
            ),
        ]
    )

    # ── Helper for debug logging ──────────────────────────────────
    def _trace_inventory(fig_data) -> str:
        """One-line summary of all traces in the figure for debug logs."""
        groups: Dict[str, int] = {}
        for t in fig_data or []:
            lg = t.get("legendgroup", "<none>")
            groups[lg] = groups.get(lg, 0) + 1
        return ", ".join(f"{k}:{v}" for k, v in groups.items()) or "(empty)"

    # ── Callback 1: page load -> fetch fresh data ───────────────
    @app.callback(
        Output("scatter-3d", "figure"),
        Output("tag-selector", "options"),
        Output("tag-selector", "value"),
        Output("figure-meta", "data"),
        Output("expanded-memories", "data"),
        Output("cluster-stats-body", "children"),
        Output("query-active", "data", allow_duplicate=True),
        Output("query-match-ids", "data", allow_duplicate=True),
        Output("query-expanded-ids", "data", allow_duplicate=True),
        Output("date-picker", "min_date_allowed"),
        Output("date-picker", "max_date_allowed"),
        Output("date-picker", "date"),
        Output("date-picker-active", "data", allow_duplicate=True),
        Input("url", "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def on_page_load(_pathname):
        """Re-fetch vectors from the database on every page load."""
        log.debug("CB1 on_page_load: triggered")
        # Clear word registry on refresh (no expanded memories yet)
        _word_registry.clear()
        _word_positions.clear()
        _parent_positions.clear()

        # Reset dedup counters so buttons work after refresh
        global _words_last_click, _lines_last_click, _word_paths_last_click
        _words_last_click = 0
        _lines_last_click = 0
        _word_paths_last_click = 0

        fig, n_scatter, tag_to_indices, sorted_tags, metadatas, ids, memory_texts = (
            _fetch_and_build(
                backend_type=backend_cfg["backend_type"],
                method=method,
                colour_by=colour_by,
                pg_host=backend_cfg.get("pg_host"),
                pg_port=backend_cfg.get("pg_port"),
                pg_database=backend_cfg.get("pg_database"),
                pg_user=backend_cfg.get("pg_user"),
                pg_password=backend_cfg.get("pg_password"),
            )
        )

        tag_options = [
            {"label": f"{tag}  ({len(tag_to_indices[tag])})", "value": tag}
            for tag in sorted_tags
        ]

        # memory_texts is a Dict[str, str] loaded directly from the
        # memories table (full title + content).  Fall back to metadata
        # title for any IDs not found in the DB query.
        for mid in ids:
            if mid not in memory_texts or not memory_texts[mid].strip():
                idx = ids.index(mid)
                meta_i = metadatas[idx] if idx < len(metadatas) else {}
                memory_texts[mid] = str(meta_i.get("title", ""))

        # ── Extract dates from metadatas for the date picker ────
        # Build id -> ISO date string (YYYY-MM-DD) from the timestamp field.
        # Also pre-build id_to_coords for O(1) lookup in the date-pick callback.
        def _parse_date(ts_raw: Any) -> Optional[str]:
            """Parse any timestamp representation to 'YYYY-MM-DD', or None."""
            if not ts_raw:
                return None
            ts_str = str(ts_raw).strip()
            if not ts_str:
                return None
            # Try ISO-8601 / datetime string (most common)
            try:
                from datetime import datetime as _dt

                return (
                    _dt.fromisoformat(ts_str.replace("Z", "+00:00")).date().isoformat()
                )
            except (ValueError, TypeError):
                pass
            # Try Unix epoch (int or float)
            try:
                from datetime import datetime as _dt

                return _dt.utcfromtimestamp(float(ts_str)).date().isoformat()
            except (ValueError, TypeError):
                pass
            # Fallback: take first 10 chars if they look like YYYY-MM-DD
            if len(ts_str) >= 10 and ts_str[4] == "-" and ts_str[7] == "-":
                return ts_str[:10]
            return None

        id_to_date: Dict[str, str] = {}
        id_to_coords: Dict[str, Dict[str, Any]] = {}
        all_dates: List[str] = []
        for i, mid in enumerate(ids):
            meta_i = metadatas[i] if i < len(metadatas) else {}
            date_str = _parse_date(meta_i.get("timestamp"))
            if date_str:
                id_to_date[mid] = date_str
                all_dates.append(date_str)

        if all_dates:
            sorted_dates = sorted(set(all_dates))
            min_date = sorted_dates[0]
            max_date = sorted_dates[-1]
        else:
            min_date = None
            max_date = None

        meta = {
            "n_scatter": n_scatter,
            "sorted_tags": sorted_tags,
            "memory_texts": memory_texts,
            "id_to_date": id_to_date,
        }

        # Compute cluster / density stats from cached 384D embeddings
        stats_children = _build_stats_panel_children()

        log.debug(
            "CB1 on_page_load: done — n_scatter=%d, tags=%d, texts=%d, dates=%d "
            "date_range=[%s..%s], fig traces=%d, expanded=[], dedup reset",
            n_scatter,
            len(sorted_tags),
            len(memory_texts),
            len(id_to_date),
            min_date,
            max_date,
            len(fig.get("data", [])) if isinstance(fig, dict) else -1,
        )
        return (
            fig,
            tag_options,
            [],
            meta,
            [],
            stats_children,
            False,
            [],
            [],
            min_date,
            max_date,
            None,
            False,
        )

    # ── Callback 2: tag selection -> toggle line visibility ─────
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Input("tag-selector", "value"),
        State("figure-meta", "data"),
        prevent_initial_call=True,
    )
    def update_tag_lines(selected_tags, meta):
        log.debug("CB2 update_tag_lines: selected=%s", selected_tags)
        if not meta:
            return no_update

        if selected_tags is None:
            selected_tags = []

        n_scatter = meta["n_scatter"]
        s_tags = meta["sorted_tags"]

        # Use Patch() to surgically update only the tag-line trace
        # visibility without touching the camera, zoom, or any other
        # trace state (legend toggles, word expansions).
        patched = Patch()
        selected_set = set(selected_tags)
        for tidx, tag in enumerate(s_tags):
            i = n_scatter + tidx
            patched["data"][i]["visible"] = tag in selected_set

        return patched

    # ── Callback 2b: date picker -> highlight memories for that date ─
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("date-picker-active", "data"),
        Input("date-picker", "date"),
        State("scatter-3d", "figure"),
        State("figure-meta", "data"),
        State("date-picker-active", "data"),
        prevent_initial_call=True,
    )
    def on_date_pick(selected_date, current_fig, meta, was_active):
        """Overlay highlight rings on all memories from the selected date."""
        log.debug(
            "CB2b on_date_pick: date=%s, was_active=%s, fig_traces=%d",
            selected_date,
            was_active,
            len(current_fig.get("data", [])) if current_fig else 0,
        )
        go_mod = _require("plotly.graph_objects", "pip install plotly")

        if not current_fig or not meta:
            return no_update, no_update

        fig_data = current_fig.get("data", [])
        patched = Patch()

        # Always scan and remove any existing date-highlight traces,
        # regardless of was_active (guards against state desync on refresh).
        existing_highlight_indices = [
            i
            for i, t in enumerate(fig_data)
            if t.get("legendgroup", "") == "date-highlight"
        ]
        for i in reversed(existing_highlight_indices):
            del patched["data"][i]

        # If date cleared, just remove highlights
        if not selected_date:
            return patched, False

        # Parse the selected date robustly to YYYY-MM-DD
        try:
            from datetime import datetime as _dt

            target_date = (
                _dt.fromisoformat(str(selected_date).replace("Z", "+00:00"))
                .date()
                .isoformat()
            )
        except (ValueError, TypeError):
            target_date = str(selected_date)[:10]

        id_to_date = meta.get("id_to_date", {})
        n_scatter = meta.get("n_scatter", 0)

        # Find memory IDs that match the selected date
        match_ids = {mid for mid, d in id_to_date.items() if d == target_date}

        if not match_ids:
            log.info("CB2b: no memories on %s", target_date)
            return patched, False

        # Build highlight coordinate lists by scanning scatter traces.
        # Guard n_scatter against out-of-bounds access.
        safe_n_scatter = min(n_scatter, len(fig_data))
        hi_x: List[float] = []
        hi_y: List[float] = []
        hi_z: List[float] = []
        hi_customdata: List[List] = []
        hi_hover: List[str] = []

        for t_idx in range(safe_n_scatter):
            trace = fig_data[t_idx]
            xs = trace.get("x") or []
            ys = trace.get("y") or []
            zs = trace.get("z") or []
            cds = trace.get("customdata") or []
            for p_idx in range(len(xs)):
                cd = cds[p_idx] if p_idx < len(cds) else []
                mem_id = cd[3] if len(cd) > 3 else ""
                if mem_id not in match_ids:
                    continue
                try:
                    px = float(xs[p_idx])
                    py = float(ys[p_idx])
                    pz = float(zs[p_idx])
                except (TypeError, ValueError):
                    continue
                title = cd[0] if len(cd) > 0 else ""
                mtype = cd[1] if len(cd) > 1 else ""
                imp = cd[2] if len(cd) > 2 else 5
                hi_x.append(px)
                hi_y.append(py)
                hi_z.append(pz)
                hi_customdata.append([title, mtype, imp, mem_id])
                hi_hover.append(
                    f"<b>{title}</b><br>"
                    f"<span style='color:#aaa'>Date:</span> {target_date}<br>"
                    f"<span style='color:#aaa'>Type:</span> {mtype}<br>"
                    f"<span style='color:#aaa'>Importance:</span> {imp}<br>"
                    f"<span style='color:#aaa'>ID:</span> {mem_id}"
                    "<extra></extra>"
                )

        if not hi_x:
            return patched, False

        highlight_trace = go_mod.Scatter3d(
            x=hi_x,
            y=hi_y,
            z=hi_z,
            mode="markers",
            marker=dict(
                size=18,
                color="rgba(0,0,0,0)",
                symbol="circle",
                opacity=1.0,
                line=dict(width=2.5, color="rgb(255, 220, 50)"),
            ),
            customdata=hi_customdata,
            hovertemplate=hi_hover,
            hoverlabel=dict(
                bgcolor="rgba(30, 25, 5, 0.95)",
                bordercolor="rgb(220, 200, 50)",
                font=dict(color="rgb(255, 240, 180)", size=13),
            ),
            name=f"date: {target_date} ({len(hi_x)})",
            legendgroup="date-highlight",
            showlegend=True,
            visible=True,
        )

        patched["data"].append(highlight_trace.to_plotly_json())

        log.info("CB2b: highlighted %d memories on %s", len(hi_x), target_date)
        return patched, True

    # ── Callback 2c: clear date btn -> remove date highlights ───
    @app.callback(
        Output("date-picker", "date", allow_duplicate=True),
        Input("clear-date-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_clear_date(_n_clicks):
        """Clear the date picker value (triggers on_date_pick with None)."""
        return None

    # ── Callback 3: click memory point -> expand/collapse words ─
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("expanded-memories", "data", allow_duplicate=True),
        Input("scatter-3d", "clickData"),
        State("scatter-3d", "figure"),
        State("figure-meta", "data"),
        State("expanded-memories", "data"),
        State("lines-visible", "data"),
        State("word-paths-visible", "data"),
        State("word-path-min-input", "value"),
        prevent_initial_call=True,
    )
    def toggle_word_expansion(
        click_data,
        current_fig,
        meta,
        expanded,
        lines_visible,
        paths_visible,
        wp_threshold,
    ):
        log.debug(
            "CB3 toggle_word_expansion: expanded=%s, lines_visible=%s, "
            "paths_visible=%s, fig_traces=%d, inventory=[%s]",
            expanded,
            lines_visible,
            paths_visible,
            len(current_fig.get("data", [])) if current_fig else 0,
            _trace_inventory(current_fig.get("data")) if current_fig else "no fig",
        )
        if not click_data or not meta or not current_fig:
            log.debug(
                "CB3: early exit (click=%s, meta=%s, fig=%s)",
                bool(click_data),
                bool(meta),
                bool(current_fig),
            )
            return no_update, no_update

        # Extract memory ID from the clicked point's customdata
        point = click_data.get("points", [{}])[0]
        customdata = point.get("customdata")
        if not customdata or len(customdata) < 4:
            return no_update, no_update

        mem_id = customdata[3]  # [title, type, importance, id]
        if not mem_id:
            log.debug("CB3: no mem_id in customdata")
            return no_update, no_update

        log.debug(
            "CB3: clicked mem_id=%s, in_expanded=%s",
            mem_id[:12],
            mem_id in (expanded or []),
        )

        if expanded is None:
            expanded = []

        n_scatter = meta["n_scatter"]
        n_tag_traces = len(meta["sorted_tags"])
        n_base = n_scatter + n_tag_traces

        # Use Patch() so we only mutate trace data, never touch
        # the camera / zoom / layout state.
        patched = Patch()

        # Always remove existing word-path traces first (they'll be
        # rebuilt from the updated registry after expand/collapse).
        path_indices = [
            i
            for i, trace in enumerate(current_fig["data"])
            if trace.get("legendgroup", "") == "word-paths"
        ]

        # ── Collapse: remove word traces for this memory ────────
        if mem_id in expanded:
            new_expanded = [m for m in expanded if m != mem_id]
            group_key = f"words:{mem_id}"
            word_indices = [
                i
                for i, trace in enumerate(current_fig["data"])
                if trace.get("legendgroup", "") == group_key
            ]
            # Delete all (word + path) indices highest-first
            for i in sorted(set(word_indices + path_indices), reverse=True):
                del patched["data"][i]

            _unregister_words(mem_id)

            # Rebuild word-path traces from updated registry
            wp_visible = paths_visible if paths_visible is not None else False
            path_traces = _build_word_path_traces(visible=wp_visible)
            for t in path_traces:
                patched["data"].append(t)

            log.debug(
                "CB3 COLLAPSE: mem_id=%s, removed %d word + %d path traces, "
                "rebuilt %d path traces, new_expanded=%s",
                mem_id[:12],
                len(word_indices),
                len(path_indices),
                len(path_traces),
                new_expanded,
            )
            return patched, new_expanded

        # ── Expand: generate word embeddings and project ────────
        memory_texts = meta.get("memory_texts", {})
        text = memory_texts.get(mem_id, "")
        if not text.strip():
            return no_update, no_update

        # Remove old word-path traces before appending new ones
        for i in sorted(path_indices, reverse=True):
            del patched["data"][i]

        try:
            _threshold = int(wp_threshold) if wp_threshold else 2
            word_traces = _make_word_traces(
                mem_id,
                text,
                point,
                n_base,
                expanded,
                lines_visible=lines_visible if lines_visible is not None else True,
                filter_threshold=_threshold if paths_visible else 0,
            )
        except Exception as exc:
            log.error(f"Word expansion failed for {mem_id}: {exc}", exc_info=True)
            return no_update, no_update

        if not word_traces:
            return no_update, no_update

        # Append word traces via Patch — camera stays put
        for trace in word_traces:
            patched["data"].append(trace)

        # Rebuild word-path traces with updated registry
        wp_visible = paths_visible if paths_visible is not None else False
        path_traces = _build_word_path_traces(visible=wp_visible, min_shared=_threshold)
        for t in path_traces:
            patched["data"].append(t)

        new_expanded = expanded + [mem_id]
        log.debug(
            "CB3 EXPAND: mem_id=%s, %d word traces, %d path traces, "
            "new_expanded=%d items",
            mem_id[:12],
            len(word_traces),
            len(path_traces),
            len(new_expanded),
        )
        return patched, new_expanded

    # ── Callback 4: semantic query -> embed, project, draw ──────
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("query-active", "data"),
        Output("query-match-ids", "data"),
        Output("sim-bar-chart", "figure"),
        Output("sim-bar-panel", "style"),
        Input("query-input", "n_submit"),
        State("query-input", "value"),
        State("query-limit", "value"),
        State("scatter-3d", "figure"),
        State("figure-meta", "data"),
        State("query-active", "data"),
        State("lines-visible", "data"),
        prevent_initial_call=True,
    )
    def on_semantic_query(
        _n_submit, query_text, query_limit, current_fig, meta, was_active, lines_visible
    ):
        log.debug(
            "CB4 on_semantic_query: n_submit=%s, query=%r, limit=%s, "
            "was_active=%s, lines_visible=%s, fig_traces=%d, inventory=[%s]",
            _n_submit,
            query_text,
            query_limit,
            was_active,
            lines_visible,
            len(current_fig.get("data", [])) if current_fig else 0,
            _trace_inventory(current_fig.get("data")) if current_fig else "no fig",
        )
        if not meta or not current_fig:
            log.debug(
                "CB4: early exit (meta=%s, fig=%s)", bool(meta), bool(current_fig)
            )
            return no_update, no_update, no_update, no_update, no_update

        patched = Patch()

        # ── Remove any previous query traces ────────────────────
        if was_active:
            to_remove = [
                i
                for i, trace in enumerate(current_fig["data"])
                if trace.get("legendgroup", "") == "semantic-query"
            ]
            log.debug("CB4: removing %d old query traces", len(to_remove))
            for i in reversed(to_remove):
                del patched["data"][i]

        # If the input was cleared, just remove the old traces
        if not query_text or not query_text.strip():
            log.debug("CB4: query cleared, returning active=False, match_ids=[]")
            return patched, False, [], no_update, {"display": "none"}

        # Sanitise limit
        try:
            n_results = max(1, min(100, int(query_limit or 10)))
        except (TypeError, ValueError):
            n_results = 10

        # ── Embed the query text ────────────────────────────────
        try:
            query_traces, matched_ids, sim_data = _make_query_traces(
                query_text.strip(),
                current_fig,
                meta,
                n_results=n_results,
                lines_visible=lines_visible if lines_visible is not None else True,
            )
        except Exception as exc:
            log.error(f"Semantic query failed: {exc}", exc_info=True)
            return patched, False, [], no_update, {"display": "none"}

        if not query_traces:
            log.debug("CB4: no query traces returned, active=False")
            return patched, False, [], no_update, {"display": "none"}

        for trace in query_traces:
            patched["data"].append(trace)

        # ── Build the similarity bar chart (384D ground truth) ──
        go = _require("plotly.graph_objects", "pip install plotly")
        titles = [d["title"] for d in sim_data]
        sims_vals = [d["sim"] for d in sim_data]
        # Colour bars green→red based on sim value
        bar_colours = [
            f"rgba({max(0, int(255 * (1 - s)))}, {int(255 * s)}, 80, 0.85)"
            for s in sims_vals
        ]
        bar_fig = go.Figure(
            go.Bar(
                x=sims_vals,
                y=[
                    f"#{i + 1} {t[:35]}{'…' if len(t) > 35 else ''}"
                    for i, t in enumerate(titles)
                ],
                orientation="h",
                marker_color=bar_colours,
                hovertemplate="%{y}<br>cosine sim: %{x:.4f}<extra></extra>",
                hoverlabel=dict(
                    bgcolor="rgba(18,18,30,0.95)",
                    font=dict(color="#e0e0f0", size=11),
                ),
            )
        )
        bar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=8, t=4, b=4),
            xaxis=dict(
                range=[0, 1],
                tickfont=dict(color="#888", size=9),
                gridcolor="#2a2a3a",
                zerolinecolor="#2a2a3a",
            ),
            yaxis=dict(
                tickfont=dict(color="#aaa", size=9),
                autorange="reversed",
            ),
            font=dict(color="#b0b0c0"),
            showlegend=False,
        )

        log.debug(
            "CB4 DONE: %d query traces appended, %d match_ids=%s",
            len(query_traces),
            len(matched_ids),
            [m[:12] for m in matched_ids[:5]],
        )
        return patched, True, matched_ids, bar_fig, {"display": "block"}

    # ── Callback 5: expand/collapse words for query matches ─────
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("expanded-memories", "data", allow_duplicate=True),
        Output("query-expand-words-btn", "className"),
        Output("query-expanded-ids", "data", allow_duplicate=True),
        Input("query-expand-words-btn", "n_clicks"),
        State("scatter-3d", "figure"),
        State("figure-meta", "data"),
        State("query-match-ids", "data"),
        State("expanded-memories", "data"),
        State("lines-visible", "data"),
        State("word-paths-visible", "data"),
        State("word-path-min-input", "value"),
        State("query-expanded-ids", "data"),
        prevent_initial_call=True,
    )
    def toggle_query_words(
        _n_clicks,
        current_fig,
        meta,
        match_ids,
        expanded,
        lines_visible,
        paths_visible,
        wp_threshold,
        prev_query_expanded,
    ):
        log.debug(
            "CB5 toggle_query_words: n_clicks=%s, match_ids=%s, expanded=%s, "
            "lines_visible=%s, paths_visible=%s, fig_traces=%d, inventory=[%s]",
            _n_clicks,
            len(match_ids) if match_ids else None,
            expanded,
            lines_visible,
            paths_visible,
            len(current_fig.get("data", [])) if current_fig else 0,
            _trace_inventory(current_fig.get("data")) if current_fig else "no fig",
        )
        # ── Server-side dedup: Dash fires this callback twice per
        #    physical click (allow_duplicate=True triggers it).
        #    Both fires carry the same n_clicks value, so the second
        #    invocation sees the updated module var and bails out.
        global _words_last_click
        with _words_click_lock:
            if _n_clicks is not None and _n_clicks == _words_last_click:
                log.debug(
                    "CB5: dedup skip (n_clicks=%s == last=%s)",
                    _n_clicks,
                    _words_last_click,
                )
                return no_update, no_update, no_update, no_update
            _words_last_click = _n_clicks if _n_clicks is not None else 0
            log.debug("CB5: dedup passed, updated last_click=%s", _words_last_click)

        if not meta or not current_fig or not match_ids:
            log.debug(
                "CB5: early exit (meta=%s, fig=%s, match_ids=%s)",
                bool(meta),
                bool(current_fig),
                match_ids,
            )
            return no_update, no_update, no_update, no_update

        if expanded is None:
            expanded = []

        n_scatter = meta["n_scatter"]
        n_tag_traces = len(meta["sorted_tags"])
        n_base = n_scatter + n_tag_traces

        # Determine expand vs collapse by checking which match IDs
        # actually have word traces in the figure (ground truth).
        # Don't rely on `expanded` store — it may be stale due to
        # Dash double-firing the callback.
        groups_present = set()
        for trace in current_fig.get("data", []):
            lg = trace.get("legendgroup", "")
            if lg.startswith("words:"):
                groups_present.add(lg.replace("words:", "", 1))

        already_in_fig = [mid for mid in match_ids if mid in groups_present]

        # Identify existing word-path trace indices for removal
        path_indices = [
            i
            for i, trace in enumerate(current_fig.get("data", []))
            if trace.get("legendgroup", "") == "word-paths"
        ]

        log.debug(
            "CB5: groups_present=%s, already_in_fig=%d, path_traces=%d, deciding %s",
            groups_present,
            len(already_in_fig),
            len(path_indices),
            "COLLAPSE" if already_in_fig else "EXPAND",
        )

        patched = Patch()

        if already_in_fig:
            # ── Collapse: remove word + path traces for all query matches ──
            new_expanded = [m for m in expanded if m not in match_ids]
            groups = {f"words:{mid}" for mid in match_ids}
            word_indices = [
                i
                for i, trace in enumerate(current_fig.get("data", []))
                if trace.get("legendgroup", "") in groups
            ]
            for i in sorted(set(word_indices + path_indices), reverse=True):
                del patched["data"][i]

            for mid in match_ids:
                _unregister_words(mid)

            # Rebuild word-path traces from updated registry
            wp_visible = paths_visible if paths_visible is not None else False
            path_traces = _build_word_path_traces(visible=wp_visible)
            for t in path_traces:
                patched["data"].append(t)

            log.debug(
                "CB5 COLLAPSE: removed %d word + %d path traces, "
                "rebuilt %d path traces, new_expanded=%d items",
                len(word_indices),
                len(path_indices),
                len(path_traces),
                len(new_expanded),
            )
            return patched, new_expanded, "query-words-btn", []

        # ── Expand: generate word traces for all matched memories ──
        memory_texts = meta.get("memory_texts", {})

        # Build id -> 3D point lookup from scatter traces
        id_to_point: Dict[str, Dict[str, Any]] = {}
        for t_idx in range(n_scatter):
            trace = current_fig["data"][t_idx]
            xs = trace.get("x", [])
            ys = trace.get("y", [])
            zs = trace.get("z", [])
            cds = trace.get("customdata", [])
            for p_idx in range(len(xs)):
                cd = cds[p_idx] if p_idx < len(cds) else []
                mem_id = cd[3] if len(cd) > 3 else ""
                if mem_id:
                    id_to_point[mem_id] = {
                        "x": xs[p_idx],
                        "y": ys[p_idx],
                        "z": zs[p_idx],
                    }

        # Remove old word-path traces before appending
        for i in sorted(path_indices, reverse=True):
            del patched["data"][i]

        new_expanded = list(expanded)
        expanded_count = 0
        skipped = {"already": 0, "no_text": 0, "no_point": 0, "empty": 0, "error": 0}
        _threshold = int(wp_threshold) if wp_threshold else 2
        for mid in match_ids:
            if mid in groups_present or mid in new_expanded:
                skipped["already"] += 1
                continue
            text = memory_texts.get(mid, "")
            if not text.strip():
                skipped["no_text"] += 1
                continue
            pt = id_to_point.get(mid)
            if pt is None:
                skipped["no_point"] += 1
                continue
            try:
                word_traces = _make_word_traces(
                    mid,
                    text,
                    pt,
                    n_base,
                    new_expanded,
                    lines_visible=lines_visible if lines_visible is not None else True,
                    filter_threshold=_threshold if paths_visible else 0,
                )
            except Exception as exc:
                skipped["error"] += 1
                log.error(f"Word expansion failed for {mid}: {exc}", exc_info=True)
                continue
            if not word_traces:
                skipped["empty"] += 1
                continue
            for trace in word_traces:
                patched["data"].append(trace)
            new_expanded.append(mid)
            expanded_count += 1

        log.info(
            "CB5 EXPAND: expanded=%d, skipped(%s), id_to_point=%d, word_registry=%d",
            expanded_count,
            ", ".join(f"{k}={v}" for k, v in skipped.items()),
            len(id_to_point),
            len(_word_registry),
        )

        if new_expanded == expanded:
            log.debug("CB5: nothing new expanded, returning no_update")
            return no_update, no_update, no_update, no_update

        # Rebuild word-path traces with updated registry
        wp_visible = paths_visible if paths_visible is not None else False
        path_traces = _build_word_path_traces(visible=wp_visible, min_shared=_threshold)
        for t in path_traces:
            patched["data"].append(t)

        log.debug(
            "CB5 EXPAND done: new_expanded=%d items, %d path traces",
            len(new_expanded),
            len(path_traces),
        )
        # Track which IDs were actually expanded by this query-words action
        # (not already expanded by click). Used by Clear Search.
        newly_expanded_by_query = [m for m in new_expanded if m not in expanded]
        # Merge with any previously query-expanded IDs still alive
        all_query_expanded = list(
            set((prev_query_expanded or []) + newly_expanded_by_query)
        )
        return patched, new_expanded, "query-words-btn active", all_query_expanded

    # ── Callback 6: toggle all connection lines on / off ────────
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("lines-visible", "data"),
        Output("toggle-lines-btn", "className"),
        Input("toggle-lines-btn", "n_clicks"),
        State("scatter-3d", "figure"),
        State("lines-visible", "data"),
        prevent_initial_call=True,
    )
    def toggle_lines(_n_clicks, current_fig, currently_visible):
        log.debug(
            "CB6 toggle_lines: n_clicks=%s, currently_visible=%s, "
            "fig_traces=%d, inventory=[%s]",
            _n_clicks,
            currently_visible,
            len(current_fig.get("data", [])) if current_fig else 0,
            _trace_inventory(current_fig.get("data")) if current_fig else "no fig",
        )
        # ── Server-side dedup (same pattern as Words button) ────
        global _lines_last_click
        with _lines_click_lock:
            if _n_clicks is not None and _n_clicks == _lines_last_click:
                log.debug(
                    "CB6: dedup skip (n_clicks=%s == last=%s)",
                    _n_clicks,
                    _lines_last_click,
                )
                return no_update, no_update, no_update
            _lines_last_click = _n_clicks if _n_clicks is not None else 0
            log.debug("CB6: dedup passed, updated last_click=%s", _lines_last_click)

        if not current_fig:
            log.debug("CB6: early exit (no fig)")
            return no_update, no_update, no_update

        # Flip the state
        new_visible = not currently_visible
        patched = Patch()

        line_count = 0
        for i, trace in enumerate(current_fig.get("data", [])):
            mode = trace.get("mode", "")
            lg = trace.get("legendgroup", "")

            is_line = mode == "lines" and (
                lg.startswith("words:")
                or lg == "semantic-query"
                or lg.startswith("tag:")
            )
            if is_line:
                patched["data"][i]["visible"] = new_visible
                line_count += 1

        log.debug(
            "CB6 DONE: toggled %d line traces, new visible=%s", line_count, new_visible
        )
        btn_class = "query-words-btn active" if new_visible else "query-words-btn"
        return patched, new_visible, btn_class

    # ── Callback 7: toggle word-path starburst lines ────────────
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("word-paths-visible", "data"),
        Output("toggle-word-paths-btn", "className"),
        Input("toggle-word-paths-btn", "n_clicks"),
        State("scatter-3d", "figure"),
        State("word-paths-visible", "data"),
        State("word-path-min-input", "value"),
        prevent_initial_call=True,
    )
    def toggle_word_paths(_n_clicks, current_fig, currently_visible, wp_threshold):
        _threshold = int(wp_threshold) if wp_threshold else 2
        log.debug(
            "CB7 toggle_word_paths: n_clicks=%s, currently_visible=%s, "
            "threshold=%d, fig_traces=%d, inventory=[%s]",
            _n_clicks,
            currently_visible,
            _threshold,
            len(current_fig.get("data", [])) if current_fig else 0,
            _trace_inventory(current_fig.get("data")) if current_fig else "no fig",
        )
        # ── Server-side dedup ────────────────────────────────────
        global _word_paths_last_click
        with _word_paths_click_lock:
            if _n_clicks is not None and _n_clicks == _word_paths_last_click:
                log.debug(
                    "CB7: dedup skip (n_clicks=%s == last=%s)",
                    _n_clicks,
                    _word_paths_last_click,
                )
                return no_update, no_update, no_update
            _word_paths_last_click = _n_clicks if _n_clicks is not None else 0
            log.debug(
                "CB7: dedup passed, updated last_click=%s", _word_paths_last_click
            )

        if not current_fig:
            log.debug("CB7: early exit (no fig)")
            return no_update, no_update, no_update

        new_visible = not currently_visible
        patched = Patch()

        # If turning ON and no word-path traces exist yet, build them
        has_paths = any(
            t.get("legendgroup", "") == "word-paths"
            for t in current_fig.get("data", [])
        )

        if new_visible and not has_paths:
            # Build and append fresh word-path traces
            path_traces = _build_word_path_traces(visible=True, min_shared=_threshold)
            if not path_traces:
                # No shared words — nothing to show, keep button inactive
                log.debug(
                    "CB7: no shared words at threshold=%d, nothing to build", _threshold
                )
                return no_update, False, "query-words-btn"
            for t in path_traces:
                patched["data"].append(t)
            log.debug("CB7: built %d new word-path traces", len(path_traces))
        elif new_visible and has_paths:
            # Threshold may have changed — remove old paths and rebuild
            for i in sorted(
                (
                    i
                    for i, t in enumerate(current_fig.get("data", []))
                    if t.get("legendgroup", "") == "word-paths"
                ),
                reverse=True,
            ):
                del patched["data"][i]
            path_traces = _build_word_path_traces(visible=True, min_shared=_threshold)
            for t in path_traces:
                patched["data"].append(t)
            log.debug(
                "CB7: rebuilt %d word-path traces with threshold=%d",
                len(path_traces),
                _threshold,
            )
        else:
            # Turning OFF — hide existing word-path traces
            path_count = 0
            for i, trace in enumerate(current_fig.get("data", [])):
                if trace.get("legendgroup", "") == "word-paths":
                    patched["data"][i]["visible"] = False
                    path_count += 1
            log.debug("CB7: hid %d word-path traces", path_count)

        # Apply / remove word filter on existing word scatter traces.
        # IMPORTANT: this uses pure index-based mutations (no del/append)
        # so it is safe to combine with the del+append above because
        # _apply_word_filter only touches word scatter traces (legendgroup
        # "words:*"), while the del+append above only touches "word-paths"
        # traces.  However, we must be careful: after del+append the
        # patched indices would be stale for "word-paths" traces, but
        # _apply_word_filter targets "words:*" traces whose indices are
        # unaffected by appends at the end.  The del operations above
        # shift indices, but only for traces AFTER the deleted ones.
        # Word scatter traces come BEFORE word-path traces in the data
        # array (word-path traces are always appended last), so their
        # indices are stable.
        _apply_word_filter(patched, current_fig, new_visible, _threshold)

        btn_class = "query-words-btn active" if new_visible else "query-words-btn"
        return patched, new_visible, btn_class

    # ── Callback 8: re-apply word filter when threshold changes ──
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Input("word-path-min-input", "value"),
        State("scatter-3d", "figure"),
        State("word-paths-visible", "data"),
        prevent_initial_call=True,
    )
    def on_threshold_change(new_threshold, current_fig, paths_visible):
        _threshold = int(new_threshold) if new_threshold else 2
        log.debug(
            "CB8 on_threshold_change: threshold=%d, paths_visible=%s",
            _threshold,
            paths_visible,
        )
        if not paths_visible or not current_fig:
            return no_update

        patched = Patch()

        # Remove old word-path traces and rebuild with new threshold
        path_indices = [
            i
            for i, t in enumerate(current_fig.get("data", []))
            if t.get("legendgroup", "") == "word-paths"
        ]
        for i in sorted(path_indices, reverse=True):
            del patched["data"][i]

        path_traces = _build_word_path_traces(visible=True, min_shared=_threshold)
        for t in path_traces:
            patched["data"].append(t)

        # Re-apply word filter with new threshold
        _apply_word_filter(patched, current_fig, True, _threshold)

        log.debug(
            "CB8 DONE: rebuilt %d path traces with threshold=%d",
            len(path_traces),
            _threshold,
        )
        return patched

    # ── Callback 9: toggle all word projections on / off ─────────
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("words-visible", "data"),
        Output("toggle-words-visible-btn", "className"),
        Input("toggle-words-visible-btn", "n_clicks"),
        State("scatter-3d", "figure"),
        State("words-visible", "data"),
        prevent_initial_call=True,
    )
    def toggle_all_words(_n_clicks, current_fig, currently_visible):
        log.debug(
            "CB9 toggle_all_words: n_clicks=%s, currently_visible=%s",
            _n_clicks,
            currently_visible,
        )
        if not current_fig:
            return no_update, no_update, no_update

        new_visible = not currently_visible
        patched = Patch()

        word_count = 0
        for i, trace in enumerate(current_fig.get("data", [])):
            lg = trace.get("legendgroup", "")
            if lg.startswith("words:"):
                patched["data"][i]["visible"] = new_visible
                word_count += 1

        log.debug("CB9: toggled %d word traces, visible=%s", word_count, new_visible)
        btn_class = "query-words-btn active" if new_visible else "query-words-btn"
        return patched, new_visible, btn_class

    # ── Callback 10: reset — collapse and remove ALL expanded words ─
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("expanded-memories", "data", allow_duplicate=True),
        Output("query-expanded-ids", "data", allow_duplicate=True),
        Output("word-paths-visible", "data", allow_duplicate=True),
        Output("toggle-word-paths-btn", "className", allow_duplicate=True),
        Input("reset-words-btn", "n_clicks"),
        State("scatter-3d", "figure"),
        prevent_initial_call=True,
    )
    def on_reset_words(_n_clicks, current_fig):
        log.debug(
            "CB10 on_reset_words: fig_traces=%d",
            len(current_fig.get("data", [])) if current_fig else 0,
        )
        if not current_fig:
            return (no_update,) * 5

        patched = Patch()

        # Find all word and word-path trace indices
        remove_indices = [
            i
            for i, t in enumerate(current_fig.get("data", []))
            if t.get("legendgroup", "").startswith("words:")
            or t.get("legendgroup", "") == "word-paths"
        ]

        if not remove_indices:
            log.debug("CB10: nothing to reset")
            return (no_update,) * 5

        for i in sorted(remove_indices, reverse=True):
            del patched["data"][i]

        # Clear all registries
        _word_registry.clear()
        _word_positions.clear()
        _parent_positions.clear()

        log.info(
            "CB10 RESET: removed %d word/path traces, registries cleared",
            len(remove_indices),
        )
        return patched, [], [], False, "query-words-btn"

    # ── Callback 11: clear semantic search and query-expanded words ─
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("query-active", "data", allow_duplicate=True),
        Output("query-match-ids", "data", allow_duplicate=True),
        Output("query-expanded-ids", "data", allow_duplicate=True),
        Output("expanded-memories", "data", allow_duplicate=True),
        Output("query-input", "value"),
        Output("sim-bar-panel", "style", allow_duplicate=True),
        Input("clear-search-btn", "n_clicks"),
        State("scatter-3d", "figure"),
        State("query-active", "data"),
        State("query-expanded-ids", "data"),
        State("expanded-memories", "data"),
        State("word-paths-visible", "data"),
        State("word-path-min-input", "value"),
        prevent_initial_call=True,
    )
    def on_clear_search(
        _n_clicks,
        current_fig,
        query_active,
        query_expanded_ids,
        expanded,
        paths_visible,
        wp_threshold,
    ):
        log.debug(
            "CB10 on_clear_search: query_active=%s, query_expanded=%s, "
            "expanded=%d, fig_traces=%d",
            query_active,
            query_expanded_ids,
            len(expanded or []),
            len(current_fig.get("data", [])) if current_fig else 0,
        )
        if not current_fig:
            return (no_update,) * 7

        patched = Patch()

        # ── Remove semantic-query traces (query point, match highlights, lines)
        query_indices = [
            i
            for i, t in enumerate(current_fig.get("data", []))
            if t.get("legendgroup", "") == "semantic-query"
        ]

        # ── Remove word traces for query-expanded memories only
        qe_ids = set(query_expanded_ids or [])
        word_groups = {f"words:{mid}" for mid in qe_ids}
        word_indices = [
            i
            for i, t in enumerate(current_fig.get("data", []))
            if t.get("legendgroup", "") in word_groups
        ]

        # ── Remove word-path traces (they'll be rebuilt)
        path_indices = [
            i
            for i, t in enumerate(current_fig.get("data", []))
            if t.get("legendgroup", "") == "word-paths"
        ]

        all_remove = sorted(
            set(query_indices + word_indices + path_indices), reverse=True
        )
        for i in all_remove:
            del patched["data"][i]

        # Unregister words for query-expanded memories
        for mid in qe_ids:
            _unregister_words(mid)

        # Update expanded list — remove query-expanded IDs
        new_expanded = [m for m in (expanded or []) if m not in qe_ids]

        # Rebuild word-path traces if any words remain
        _threshold = int(wp_threshold) if wp_threshold else 2
        wp_visible = paths_visible if paths_visible is not None else False
        if _word_registry:
            path_traces = _build_word_path_traces(
                visible=wp_visible, min_shared=_threshold
            )
            for t in path_traces:
                patched["data"].append(t)

        log.info(
            "CB10 CLEAR: removed %d query + %d word + %d path traces, "
            "unregistered %d memories, expanded %d -> %d",
            len(query_indices),
            len(word_indices),
            len(path_indices),
            len(qe_ids),
            len(expanded or []),
            len(new_expanded),
        )
        # Return: figure, query-active, query-match-ids, query-expanded-ids,
        #         expanded-memories, query-input value (clear the text box),
        #         sim-bar-panel style (hide it)
        return patched, False, [], [], new_expanded, "", {"display": "none"}

    # ── Callback 11: switch reduction method (PCA / t-SNE / UMAP) ─
    @app.callback(
        Output("scatter-3d", "figure", allow_duplicate=True),
        Output("tag-selector", "options", allow_duplicate=True),
        Output("tag-selector", "value", allow_duplicate=True),
        Output("figure-meta", "data", allow_duplicate=True),
        Output("expanded-memories", "data", allow_duplicate=True),
        Output("cluster-stats-body", "children", allow_duplicate=True),
        Input("reduction-method", "value"),
        prevent_initial_call=True,
    )
    def on_method_change(new_method):
        """Rebuild the entire figure with a different reduction method."""
        log.info("CB9 on_method_change: method=%s", new_method)
        if not new_method or new_method not in REDUCERS:
            return (no_update,) * 6

        # Clear word registries (expanded words use PCA projection
        # which is re-fit during _fetch_and_build)
        _word_registry.clear()
        _word_positions.clear()
        _parent_positions.clear()

        # Reset dedup counters
        global _words_last_click, _lines_last_click, _word_paths_last_click
        _words_last_click = 0
        _lines_last_click = 0
        _word_paths_last_click = 0

        fig, n_scatter, tag_to_indices, sorted_tags, metadatas, ids, memory_texts = (
            _fetch_and_build(
                backend_type=backend_cfg["backend_type"],
                method=new_method,
                colour_by=colour_by,
                pg_host=backend_cfg.get("pg_host"),
                pg_port=backend_cfg.get("pg_port"),
                pg_database=backend_cfg.get("pg_database"),
                pg_user=backend_cfg.get("pg_user"),
                pg_password=backend_cfg.get("pg_password"),
            )
        )

        tag_options = [
            {"label": f"{tag}  ({len(tag_to_indices[tag])})", "value": tag}
            for tag in sorted_tags
        ]

        for mid in ids:
            if mid not in memory_texts or not memory_texts[mid].strip():
                idx = ids.index(mid)
                meta_i = metadatas[idx] if idx < len(metadatas) else {}
                memory_texts[mid] = str(meta_i.get("title", ""))

        # Re-build id_to_date so date highlights work after method switch
        id_to_date: Dict[str, str] = {}
        for i, mid in enumerate(ids):
            meta_i = metadatas[i] if i < len(metadatas) else {}
            ts_raw = meta_i.get("timestamp")
            if ts_raw:
                ts_str = str(ts_raw).strip()
                # Use same robust ISO parse as CB1
                try:
                    from datetime import datetime as _dt

                    id_to_date[mid] = (
                        _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                        .date()
                        .isoformat()
                    )
                except (ValueError, TypeError):
                    if len(ts_str) >= 10 and ts_str[4] == "-" and ts_str[7] == "-":
                        id_to_date[mid] = ts_str[:10]

        meta = {
            "n_scatter": n_scatter,
            "sorted_tags": sorted_tags,
            "memory_texts": memory_texts,
            "id_to_date": id_to_date,
        }

        stats_children = _build_stats_panel_children()

        log.info(
            "CB9 on_method_change: done — method=%s, n_scatter=%d, fig traces=%d",
            new_method,
            n_scatter,
            len(fig.get("data", [])) if isinstance(fig, dict) else -1,
        )
        return fig, tag_options, [], meta, [], stats_children

    log.info(f"Starting Dash app on http://127.0.0.1:{port}/")
    log.info("Hover over points to see memory titles.")
    log.info("Use the tag dropdown (top) to search and enable tag connection lines.")
    log.info("Click a memory point to expand its word vectors.")
    log.info(
        "Type a query in the search box and press Enter to visualise semantic search."
    )
    log.info("Page refresh fetches fresh data from the database.")
    log.info("Press Ctrl+C to stop.\n")

    # Pre-load the embedding model in a background thread so the first
    # semantic query / word expansion doesn't stall for ~20 s.
    threading.Thread(target=_get_embedding_model, daemon=True).start()

    try:
        app.run(host="127.0.0.1", port=port, debug=False)
    except Exception:
        log.critical("Dash app.run() crashed:", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Word-vector expansion
# ---------------------------------------------------------------------------

# Registry: word -> set of memory IDs that contain it.
# Used to scale diamond size — shared words get bigger markers.
_word_registry: Dict[str, set] = {}

# Server-side dedup for the Words toggle button (Dash double-fire guard).
# Both fires of a single physical click share the same n_clicks value,
# so the second invocation sees the updated _words_last_click and bails.
# NOTE: This uses module-level state, which is correct for this single-user
# local tool (bound to 127.0.0.1).  For multi-user deployments, use
# per-session dcc.Store instead.
_words_click_lock = threading.Lock()
_words_last_click: int = 0

# Same dedup guard for the Lines toggle button.
_lines_click_lock = threading.Lock()
_lines_last_click: int = 0

_WORD_SIZE_BASE = 4
_WORD_SIZE_STEP = 2  # extra px per additional memory referencing the word
_WORD_SIZE_MAX = 14

# Word path registries — track 3D positions so we can draw starburst
# lines through shared words connecting their parent memories.
# word string -> (x, y, z) in the projected 3D space
_word_positions: Dict[str, Tuple[float, float, float]] = {}
# mem_id -> (x, y, z) parent memory point
_parent_positions: Dict[str, Tuple[float, float, float]] = {}

# Dedup guard for Word Paths toggle button (same pattern as Words/Lines).
_word_paths_click_lock = threading.Lock()
_word_paths_last_click: int = 0

# Word-path line styling
_WORD_PATH_WIDTH_BASE = 3  # line width for a word shared by 2 memories
_WORD_PATH_WIDTH_STEP = 2  # extra px per additional sharing memory
_WORD_PATH_WIDTH_MAX = 12
_WORD_PATH_COLOUR = "rgb(180, 230, 255)"  # bright cyan-white
_WORD_PATH_OPACITY = 0.55

# Colours for word clusters (one per expanded memory, cycling)
_WORD_COLOURS = [
    "rgba(120,180,255,0.7)",
    "rgba(120,230,160,0.7)",
    "rgba(255,200,100,0.7)",
    "rgba(240,120,120,0.7)",
    "rgba(200,140,255,0.7)",
    "rgba(160,220,240,0.7)",
]

# Cache the SentenceTransformer model so it's only loaded once
_st_model_cache: Dict[str, Any] = {}


def _get_embedding_model():
    """Load (or return cached) SentenceTransformer model."""
    if "model" not in _st_model_cache:
        from sentence_transformers import SentenceTransformer
        from memory_mcp.config import EMBEDDING_MODEL_CONFIG

        model_name = EMBEDDING_MODEL_CONFIG["model_name"]
        log.info(f"Loading embedding model for word expansion: {model_name}")
        _st_model_cache["model"] = SentenceTransformer(model_name)
    return _st_model_cache["model"]


def _tokenize_to_words(text: str, max_words: int = 60) -> List[str]:
    """Split text into meaningful word chunks for embedding.

    Filters out very short tokens (< 3 chars), stopwords, and
    deduplicates.  Returns up to *max_words* unique words.
    """
    import re

    _stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "like",
        "through",
        "after",
        "over",
        "between",
        "out",
        "up",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "and",
        "but",
        "or",
        "not",
        "no",
        "so",
        "if",
        "than",
        "too",
        "very",
        "just",
        "also",
        "then",
        "now",
        "here",
        "there",
        "when",
        "where",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "than",
        "which",
        "who",
        "whom",
    }

    raw = re.findall(r"[a-zA-Z0-9_\-]{3,}", text.lower())
    seen: set = set()
    words: List[str] = []
    for w in raw:
        if w in _stopwords or w in seen:
            continue
        # Skip pure numbers
        if w.isdigit():
            continue
        seen.add(w)
        words.append(w)
        if len(words) >= max_words:
            break
    return words


def _make_word_traces(
    mem_id: str,
    text: str,
    parent_point: Dict[str, Any],
    n_base: int,
    expanded: List[str],
    lines_visible: bool = True,
    filter_threshold: int = 0,
) -> List[Dict[str, Any]]:
    """Build Plotly trace dicts for the word vectors of a single memory.

    Registers each word in ``_word_registry`` so that shared words
    (referenced by multiple expanded memories) get larger diamond
    markers.

    When *filter_threshold* is >= 2, words not shared by at least that
    many memories are hidden (size 0, no text label).  This is used
    when the Word Paths view is active to de-clutter the visualisation.
    A value of 0 disables filtering entirely.

    Returns a list of trace dicts (scatter + lines) ready to be
    appended to ``fig["data"]``.
    """
    go = _require("plotly.graph_objects", "pip install plotly")

    words = _tokenize_to_words(text)
    if not words:
        log.debug("_make_word_traces(%s): no words after tokenize", mem_id[:12])
        return []

    log.debug(
        "_make_word_traces(%s): %d words, lines_visible=%s, filter_threshold=%d, encoding...",
        mem_id[:12],
        len(words),
        lines_visible,
        filter_threshold,
    )
    model = _get_embedding_model()
    word_embeddings = model.encode(words)
    word_embeddings = np.array(word_embeddings, dtype=np.float32)

    # Project into the same 3D space as the memory points
    pts_3d = _project_new_vectors(word_embeddings)

    # Register words and compute per-word marker sizes
    for w in words:
        _word_registry.setdefault(w, set()).add(mem_id)

    sizes = []
    display_words = []
    filtering = filter_threshold >= 2
    for w in words:
        n_refs = len(_word_registry.get(w, set()))
        if filtering and n_refs < filter_threshold:
            sizes.append(0)
            display_words.append("")
        else:
            sz = min(_WORD_SIZE_MAX, _WORD_SIZE_BASE + _WORD_SIZE_STEP * (n_refs - 1))
            sizes.append(sz)
            display_words.append(w)

    # Parent point coordinates
    px = parent_point.get("x", 0.0)
    py = parent_point.get("y", 0.0)
    pz = parent_point.get("z", 0.0)

    # Register positions for word-path starburst traces
    _parent_positions[mem_id] = (px, py, pz)
    for i, w in enumerate(words):
        _word_positions[w] = (
            float(pts_3d[i, 0]),
            float(pts_3d[i, 1]),
            float(pts_3d[i, 2]),
        )

    # Pick a colour for this expansion
    colour_idx = len(expanded) % len(_WORD_COLOURS)
    word_col = _WORD_COLOURS[colour_idx]
    # Extract RGB from "rgba(R,G,B,0.7)" for the line (trace-level opacity
    # is used instead of rgba alpha because Plotly WebGL ignores alpha on lines)
    _rgb_match = word_col.replace("rgba(", "").split(",")[:3]
    line_rgb = f"rgb({','.join(c.strip() for c in _rgb_match)})"

    group_key = f"words:{mem_id}"

    # ── Word scatter trace ──────────────────────────────────────
    hover_tpl = (
        "<b>%{customdata[0]}</b><br>shared by %{customdata[1]} memories<extra></extra>"
    )
    customdata = [[w, len(_word_registry.get(w, set()))] for w in words]

    word_scatter = go.Scatter3d(
        x=pts_3d[:, 0].tolist(),
        y=pts_3d[:, 1].tolist(),
        z=pts_3d[:, 2].tolist(),
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=word_col,
            symbol="diamond",
            opacity=0.8,
            line=dict(width=0.3, color="rgba(255,255,255,0.2)"),
        ),
        text=display_words,
        textposition="top center",
        textfont=dict(size=9, color="rgba(200,200,210,0.85)"),
        customdata=customdata,
        hovertemplate=hover_tpl,
        hoverlabel=dict(
            bgcolor="rgba(20, 20, 30, 0.95)",
            bordercolor="rgb(80, 80, 90)",
            font=dict(color="rgb(240, 240, 240)", size=12),
        ),
        name=f"words: {mem_id[:12]}...",
        legendgroup=group_key,
        showlegend=True,
        visible=True,
    )

    # ── Lines from each word to the parent memory point ─────────
    lx: List[Optional[float]] = []
    ly: List[Optional[float]] = []
    lz: List[Optional[float]] = []
    for i in range(len(words)):
        # Skip connector lines for words hidden by the filter
        if filtering and display_words[i] == "":
            continue
        lx.extend([px, float(pts_3d[i, 0]), None])
        ly.extend([py, float(pts_3d[i, 1]), None])
        lz.extend([pz, float(pts_3d[i, 2]), None])

    line_trace = go.Scatter3d(
        x=lx,
        y=ly,
        z=lz,
        mode="lines",
        line=dict(color=line_rgb, width=1),
        opacity=0.25,
        legendgroup=group_key,
        showlegend=False,
        hoverinfo="skip",
        visible=lines_visible,
    )

    # Return as plain dicts (Dash serializes go objects to dicts anyway,
    # but being explicit avoids edge cases).
    log.debug(
        "_make_word_traces(%s): returning 2 traces (scatter + lines), "
        "group=%s, lines_visible=%s",
        mem_id[:12],
        group_key,
        lines_visible,
    )
    return [word_scatter.to_plotly_json(), line_trace.to_plotly_json()]


def _unregister_words(mem_id: str) -> None:
    """Remove *mem_id* from the word registry (called on collapse).

    Also cleans up orphaned word positions and the parent position entry.
    """
    before = len(_word_registry)
    dead_keys = []
    for word, mem_ids in _word_registry.items():
        mem_ids.discard(mem_id)
        if not mem_ids:
            dead_keys.append(word)
    for k in dead_keys:
        del _word_registry[k]
        # Only remove position for words with NO remaining sharers
        # (dead_keys contains only fully-orphaned words).
        _word_positions.pop(k, None)
    _parent_positions.pop(mem_id, None)
    log.debug(
        "_unregister_words(%s): registry %d -> %d entries, removed %d dead keys, "
        "parent_positions=%d, word_positions=%d",
        mem_id[:12],
        before,
        len(_word_registry),
        len(dead_keys),
        len(_parent_positions),
        len(_word_positions),
    )


def _word_size(word: str) -> int:
    """Marker size for *word* based on how many memories reference it."""
    n = len(_word_registry.get(word, set()))
    return min(_WORD_SIZE_MAX, _WORD_SIZE_BASE + _WORD_SIZE_STEP * max(0, n - 1))


def _refresh_word_sizes(
    patched, current_fig: Dict[str, Any], skip_indices: Optional[set] = None
) -> None:
    """Patch marker sizes on all existing word scatter traces to reflect
    the current ``_word_registry`` counts.

    Only touches traces whose ``legendgroup`` starts with ``"words:"``.
    Within each trace, ``customdata[i][0]`` holds the word string.

    IMPORTANT: ``skip_indices`` must contain the indices of any traces
    being deleted in the same ``Patch()`` — mixing ``del`` with
    index-based mutations on the same index corrupts the Plotly figure
    client-side.
    """
    if skip_indices is None:
        skip_indices = set()
    for i, trace in enumerate(current_fig.get("data", [])):
        if i in skip_indices:
            continue
        lg = trace.get("legendgroup", "")
        if not lg.startswith("words:"):
            continue
        # Only update the scatter trace (has marker), skip line traces
        if trace.get("mode", "") != "markers+text":
            continue
        cds = trace.get("customdata", [])
        if not cds:
            continue
        new_sizes = []
        new_cds = []
        for cd in cds:
            word = cd[0] if cd else ""
            new_sizes.append(_word_size(word))
            new_cds.append([word, len(_word_registry.get(word, set()))])
        patched["data"][i]["marker"]["size"] = new_sizes
        patched["data"][i]["customdata"] = new_cds


def _build_word_path_traces(
    visible: bool = True, min_shared: int = 2
) -> List[Dict[str, Any]]:
    """Build starburst line traces through shared words.

    For every word in ``_word_registry`` that is referenced by at least
    *min_shared* expanded memories, draw a thick line from each parent
    memory's 3D position to the word's 3D position.  This creates a
    star/fan pattern centred on the shared word diamond, visually
    bridging the memories that share vocabulary.

    Line thickness scales with the number of memories sharing the word.

    Returns a list of Plotly trace dicts (may be empty if nothing is shared).
    All traces have ``legendgroup="word-paths"``.
    """
    go = _require("plotly.graph_objects", "pip install plotly")

    # Snapshot registries to avoid races with concurrent callbacks
    # (Dash runs callbacks in threads even in single-user mode).
    word_reg = dict(_word_registry)
    word_pos = dict(_word_positions)
    parent_pos = dict(_parent_positions)

    threshold = max(2, min_shared)  # never go below 2

    # Collect segments grouped by sharing count so we can use different
    # line widths.  Key = n_sharing, Value = list of (x, y, z) triples
    # with None separators.
    width_buckets: Dict[int, Dict[str, List[Optional[float]]]] = {}

    for word, mem_ids in word_reg.items():
        if len(mem_ids) < threshold:
            continue
        wp = word_pos.get(word)
        if wp is None:
            continue

        n_sharing = len(mem_ids)
        if n_sharing not in width_buckets:
            width_buckets[n_sharing] = {"x": [], "y": [], "z": []}
        bucket = width_buckets[n_sharing]

        for mid in mem_ids:
            pp = parent_pos.get(mid)
            if pp is None:
                continue
            # Line segment: parent -> word point -> None (break)
            bucket["x"].extend([pp[0], wp[0], None])
            bucket["y"].extend([pp[1], wp[1], None])
            bucket["z"].extend([pp[2], wp[2], None])

    if not width_buckets:
        log.debug(
            "_build_word_path_traces: no words shared by >=%d memories, returning empty",
            threshold,
        )
        return []

    min_sharing = min(width_buckets)
    traces = []
    for n_sharing, coords in sorted(width_buckets.items()):
        width = min(
            _WORD_PATH_WIDTH_MAX,
            max(1, _WORD_PATH_WIDTH_BASE + _WORD_PATH_WIDTH_STEP * (n_sharing - 2)),
        )
        trace = go.Scatter3d(
            x=coords["x"],
            y=coords["y"],
            z=coords["z"],
            mode="lines",
            line=dict(color=_WORD_PATH_COLOUR, width=width),
            opacity=_WORD_PATH_OPACITY,
            legendgroup="word-paths",
            name=f"word paths ({n_sharing} shared)",
            showlegend=(n_sharing == min_sharing),  # legend for thinnest only
            hoverinfo="skip",
            visible=visible,
        )
        traces.append(trace.to_plotly_json())

    n_shared = sum(1 for m in word_reg.values() if len(m) >= threshold)
    log.debug(
        "_build_word_path_traces: %d shared words (threshold=%d), "
        "%d width buckets, %d traces",
        n_shared,
        threshold,
        len(width_buckets),
        len(traces),
    )
    return traces


def _apply_word_filter(
    patched,  # Patch object
    current_fig,  # current figure dict
    filter_on: bool,
    threshold: int = 2,
) -> None:
    """Apply or remove the word filter on existing word scatter traces.

    When *filter_on* is True, iterate every word scatter trace
    (legendgroup starts with ``"words:"``), and for each point whose
    word (from ``customdata[i][0]``) is referenced by fewer than
    *threshold* memories, set ``marker.size`` to 0 and ``text`` to "".

    When *filter_on* is False, restore original sizes via
    ``_word_size()`` and text from ``customdata[i][0]``.

    Uses **pure index-based Patch mutations only** (no del/append)
    to avoid the Plotly Patch corruption bug.
    """
    threshold = max(2, threshold)
    data = current_fig.get("data", [])
    for i, trace in enumerate(data):
        lg = trace.get("legendgroup", "")
        if not lg.startswith("words:"):
            continue
        # Only process scatter traces (mode contains "markers"), not line traces
        mode = trace.get("mode", "")
        if "markers" not in mode:
            continue

        cd = trace.get("customdata", [])
        if not cd:
            continue

        new_sizes = []
        new_texts = []
        for pt_cd in cd:
            word = pt_cd[0] if pt_cd else ""
            n_refs = len(_word_registry.get(word, set()))
            if filter_on and n_refs < threshold:
                new_sizes.append(0)
                new_texts.append("")
            else:
                new_sizes.append(_word_size(word))
                new_texts.append(word)

        patched["data"][i]["marker"]["size"] = new_sizes
        patched["data"][i]["text"] = new_texts

    log.debug(
        "_apply_word_filter: filter_on=%s, threshold=%d, processed %d traces",
        filter_on,
        threshold,
        sum(
            1
            for t in data
            if t.get("legendgroup", "").startswith("words:")
            and "markers" in t.get("mode", "")
        ),
    )


# ---------------------------------------------------------------------------
# Semantic query visualisation
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector *a* (1, D) and matrix *b* (N, D).

    Returns (N,) float32 array of similarities in [-1, 1].
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return (a_norm @ b_norm.T).flatten()


def _adaptive_threshold(
    similarities: np.ndarray, min_relevance: float = 0.15
) -> Tuple[np.ndarray, float]:
    """Apply the same adaptive thresholding as the memory manager.

    Mirrors ``memory_system.search_semantic()`` logic:
      1. ``adaptive = max(0.12, min(0.35, top_sim - 0.08))``
      2. ``threshold = max(min_relevance, adaptive)``
      3. Keep all entries >= threshold.
      4. Top-1 fallback if nothing passes (and top_sim >= 0.08).

    Args:
        similarities: (N,) array of cosine similarities, **descending** order.
        min_relevance: floor relevance, same default as the memory manager.

    Returns:
        (selected_mask, threshold) — boolean mask over *similarities* and
        the computed threshold value.
    """
    if len(similarities) == 0:
        return np.array([], dtype=bool), min_relevance

    # Guard: similarities must be pre-sorted descending for correct threshold
    assert similarities[0] >= similarities[-1], (
        "_adaptive_threshold expects similarities in descending order, "
        f"got first={similarities[0]:.4f} last={similarities[-1]:.4f}"
    )

    top_sim = float(similarities[0])
    adaptive = max(0.12, min(0.35, top_sim - 0.08))
    threshold = max(min_relevance, adaptive)

    mask = similarities >= threshold

    # Top-1 fallback: if nothing passes but top candidate isn't garbage
    if not mask.any() and top_sim >= 0.08:
        mask[0] = True

    return mask, threshold


def _make_query_traces(
    query_text: str,
    current_fig: Dict[str, Any],
    meta: Dict[str, Any],
    n_results: int = 10,
    lines_visible: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    """Embed *query_text*, compute cosine similarity in the original 384D
    embedding space, project the query into 3D, and draw lines to the
    matching memories.

    Replicates the exact same search pipeline as the MCP
    ``search_memories`` tool (``memory_system.search_semantic``):

      1. Encode the query with the model-specific prefix.
      2. Retrieve the top *n_results* by cosine similarity
         (mirrors pgvector ``ORDER BY distance LIMIT n_results``).
      3. Apply adaptive thresholding on that candidate pool.
      4. Top-1 fallback if nothing passes the threshold.

    Returns (trace_dicts, matched_memory_ids, sim_data).
    sim_data is a list of {title, sim, id} dicts for the matched memories,
    in rank order, using ground-truth 384D cosine similarity.
    """
    go = _require("plotly.graph_objects", "pip install plotly")

    log.debug(
        "_make_query_traces: query=%r, n_results=%d, lines_visible=%s",
        query_text[:40],
        n_results,
        lines_visible,
    )

    # Check that raw embeddings are cached
    if "embeddings" not in _embeddings_cache:
        log.warning("Warning: no embeddings cache — page may need a refresh.")
        return [], [], []

    all_embeddings = _embeddings_cache["embeddings"]  # (N, 384)
    all_ids = _embeddings_cache["ids"]  # List[str]

    if len(all_ids) != all_embeddings.shape[0]:
        log.warning(
            f"Warning: cache inconsistency — {len(all_ids)} ids vs "
            f"{all_embeddings.shape[0]} embeddings. Refresh the page."
        )
        return [], [], []

    # Embed the query — apply the same prefix the memory manager uses
    # so the resulting embedding matches what pgvector/chromadb would see.
    from memory_mcp.config import EMBEDDING_MODEL_CONFIG

    query_prefix = EMBEDDING_MODEL_CONFIG.get("query_prefix", "").strip()
    prefixed_query = f"{query_prefix} {query_text}" if query_prefix else query_text

    model = _get_embedding_model()
    query_embedding = model.encode([prefixed_query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Cosine similarity in full 384D space (ground truth)
    sims = _cosine_similarity(query_embedding, all_embeddings)

    # ── Replicate the pgvector pipeline ──────────────────────
    # pgvector: ORDER BY distance LIMIT n_results → returns top-N
    # memory_system then converts to similarities and thresholds.
    sorted_indices = np.argsort(-sims)
    candidate_indices = sorted_indices[:n_results]
    candidate_sims = sims[candidate_indices]

    # Apply the same adaptive thresholding as the memory manager
    mask, threshold = _adaptive_threshold(candidate_sims)
    top_indices = candidate_indices[mask]

    if len(top_indices) == 0:
        log.info(
            f"Semantic query '{query_text}': no matches above adaptive "
            f"threshold {threshold:.3f} (limit={n_results})"
        )
        return [], [], []

    log.info(
        f"Semantic query '{query_text}': {len(top_indices)} matches "
        f"(limit={n_results}, best={sims[top_indices[0]]:.4f}, "
        f"threshold={threshold:.3f})"
    )

    # Project query into 3D
    q_3d = _project_new_vectors(query_embedding)
    qx, qy, qz = float(q_3d[0, 0]), float(q_3d[0, 1]), float(q_3d[0, 2])

    # Build a lookup: mem_id -> (x, y, z, title) from the scatter traces
    n_scatter = meta["n_scatter"]
    id_to_point: Dict[str, Tuple] = {}
    for t_idx in range(n_scatter):
        trace = current_fig["data"][t_idx]
        xs = trace.get("x", [])
        ys = trace.get("y", [])
        zs = trace.get("z", [])
        cds = trace.get("customdata", [])
        for p_idx in range(len(xs)):
            cd = cds[p_idx] if p_idx < len(cds) else []
            title = cd[0] if len(cd) > 0 else ""
            mem_id = cd[3] if len(cd) > 3 else ""
            if mem_id:
                id_to_point[mem_id] = (xs[p_idx], ys[p_idx], zs[p_idx], title)

    group_key = "semantic-query"

    # ── Query point marker ──────────────────────────────────────
    query_marker = go.Scatter3d(
        x=[qx],
        y=[qy],
        z=[qz],
        mode="markers+text",
        marker=dict(
            size=10,
            color="rgb(220, 40, 40)",
            symbol="cross",
            opacity=1.0,
            line=dict(width=1, color="rgba(255,255,255,0.6)"),
        ),
        text=[query_text[:40] + ("..." if len(query_text) > 40 else "")],
        textposition="top center",
        textfont=dict(size=11, color="rgb(255, 140, 140)"),
        hovertemplate=(
            "<b>Query</b><br>"
            "%{text}<br>"
            f"<b>{len(top_indices)}</b> / {n_results} "
            f"(threshold {threshold:.3f})<br>"
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor="rgba(60, 10, 10, 0.95)",
            bordercolor="rgb(220, 60, 60)",
            font=dict(color="rgb(255, 220, 220)", size=12),
        ),
        name=f"query: {query_text[:20]} ({len(top_indices)}/{n_results})",
        legendgroup=group_key,
        showlegend=True,
        visible=True,
    )

    # ── Lines from query to matching memories ───────────────────
    lx: List[Optional[float]] = []
    ly: List[Optional[float]] = []
    lz: List[Optional[float]] = []
    nn_x: List[float] = []
    nn_y: List[float] = []
    nn_z: List[float] = []
    nn_hover: List[str] = []
    nn_customdata: List[List] = []
    matched_ids: List[str] = []
    sim_data: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_indices):
        mid = all_ids[idx]
        sim = float(sims[idx])
        pt = id_to_point.get(mid)
        if pt is None:
            continue

        matched_ids.append(mid)
        px, py, pz, title = pt
        lx.extend([qx, px, None])
        ly.extend([qy, py, None])
        lz.extend([qz, pz, None])

        nn_x.append(float(px))
        nn_y.append(float(py))
        nn_z.append(float(pz))
        nn_hover.append(
            f"<b>#{rank + 1}</b> {title[:50]}<br>"
            f"cosine similarity: {sim:.4f}<br>"
            f"Click to expand words<br>"
            f"<extra></extra>"
        )
        # customdata format matches scatter traces: [title, type, importance, id]
        # type/importance unknown here so use placeholders — id is what matters for click
        nn_customdata.append([title, "", 5, mid])
        sim_data.append({"title": title, "sim": sim, "id": mid})

    line_trace = go.Scatter3d(
        x=lx,
        y=ly,
        z=lz,
        mode="lines",
        line=dict(color="rgb(220, 60, 60)", width=2),
        opacity=0.35,
        legendgroup=group_key,
        showlegend=False,
        hoverinfo="skip",
        visible=lines_visible,
    )

    # ── Diamond markers on matched memory points ────────────────
    nn_markers = go.Scatter3d(
        x=nn_x,
        y=nn_y,
        z=nn_z,
        mode="markers",
        marker=dict(
            size=8,
            color="rgb(220, 80, 80)",
            symbol="diamond",
            opacity=0.9,
            line=dict(width=1, color="rgba(255,255,255,0.5)"),
        ),
        customdata=nn_customdata,
        hovertemplate=nn_hover,
        hoverlabel=dict(
            bgcolor="rgba(60, 10, 10, 0.95)",
            bordercolor="rgb(220, 60, 60)",
            font=dict(color="rgb(255, 220, 220)", size=12),
        ),
        name=f"matches ({len(nn_x)})",
        legendgroup=group_key,
        showlegend=False,
        visible=True,
    )

    log.debug(
        "_make_query_traces: returning 3 traces (marker+lines+matches), "
        "%d matched_ids, lines_visible=%s",
        len(matched_ids),
        lines_visible,
    )
    return (
        [
            query_marker.to_plotly_json(),
            line_trace.to_plotly_json(),
            nn_markers.to_plotly_json(),
        ],
        matched_ids,
        sim_data,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="3D Vector Space Visualizer for Long-Term Memory MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reduction methods:\n"
            "  pca   Fast, preserves global variance (default)\n"
            "  tsne  Slow, best cluster separation\n"
            "  umap  Fast, preserves local + global structure "
            "(requires umap-learn)\n"
        ),
    )

    parser.add_argument(
        "--method",
        choices=list(REDUCERS.keys()),
        default="pca",
        help="Dimensionality reduction method (default: pca)",
    )
    parser.add_argument(
        "--colour-by",
        choices=["memory_type", "importance"],
        default="memory_type",
        help="Attribute to colour points by (default: memory_type)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Dash web app (default: 8050)",
    )

    # Backend args (mirror server.py)
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
    parser.add_argument("--pg-password", type=str, default=None)

    args = parser.parse_args()

    # Build backend config dict — passed into run_dash_app so each
    # page refresh can re-query the database.
    backend_cfg: Dict[str, Any] = {
        "backend_type": args.vector_backend,
        "pg_host": args.pg_host,
        "pg_port": args.pg_port,
        "pg_database": args.pg_database,
        "pg_user": args.pg_user,
        "pg_password": args.pg_password,
    }

    # Smoke-test: verify the backend is reachable before starting the
    # Dash server (exits with a clear error message if not).
    load_vectors(**backend_cfg)
    log.info("Backend OK — starting Dash server.\n")

    # Launch Dash app (data is re-fetched on every page load)
    try:
        run_dash_app(
            backend_cfg=backend_cfg,
            method=args.method,
            colour_by=args.colour_by,
            port=args.port,
        )
    except Exception:
        log.critical("run_dash_app() crashed:", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("Unhandled exception in main():", exc_info=True)
        raise
