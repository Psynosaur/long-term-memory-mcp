#!/usr/bin/env python3
"""
Embedding Model Benchmark — Compare retrieval quality between models.

Uses the live SQLite memory database as the corpus and tests both models
against a set of benchmark queries designed to cover the types of lookups
the memory system actually handles.

Usage:
    python benchmark_embeddings.py

Requirements:
    pip install sentence-transformers chromadb tabulate numpy
"""

import sqlite3
import time
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from memory_mcp.config import DATA_FOLDER, EMBEDDING_MODEL_PRESETS


# ── Configuration ──────────────────────────────────────────────────────────

# Models to benchmark (must be keys from EMBEDDING_MODEL_PRESETS)
MODELS_TO_BENCHMARK = [
    "all-MiniLM-L12-v2",
    "bge-small-en-v1.5",
    "snowflake-arctic-embed-s",
]

# Number of top results to consider for metrics
TOP_K = 5

# SQLite database path
DB_PATH = DATA_FOLDER / "memory_db" / "memories.db"

# ── Benchmark Queries ──────────────────────────────────────────────────────
# Each query has:
#   - query: the search string
#   - expected_keywords: words that SHOULD appear in relevant results
#   - category: what type of recall this tests
#
# These are designed to mimic real usage of the memory system.

BENCHMARK_QUERIES = [
    # ── Preference recall ──
    {
        "query": "Does the user prefer Docker or local development?",
        "expected_keywords": ["docker", "local", "prefer"],
        "category": "preference",
    },
    {
        "query": "What are the user's git commit preferences?",
        "expected_keywords": ["commit", "git", "never"],
        "category": "preference",
    },
    {
        "query": "How does the user connect to GitHub?",
        "expected_keywords": ["https", "ssh", "github", "git"],
        "category": "preference",
    },
    # ── Project architecture recall ──
    {
        "query": "What is the modular architecture of the memory system?",
        "expected_keywords": ["modular", "memory_mcp", "refactor", "architecture"],
        "category": "architecture",
    },
    {
        "query": "How are embeddings generated and stored?",
        "expected_keywords": ["embedding", "chromadb", "vector", "sentence"],
        "category": "architecture",
    },
    {
        "query": "What embedding model does the project use?",
        "expected_keywords": ["embedding", "model", "MiniLM", "bge"],
        "category": "architecture",
    },
    # ── Bug fix / solution recall ──
    {
        "query": "database corruption fixes and WAL checkpoint",
        "expected_keywords": ["wal", "checkpoint", "sqlite", "corruption", "safety"],
        "category": "bugfix",
    },
    {
        "query": "ChromaDB and SQLite record count mismatch",
        "expected_keywords": ["mismatch", "chromadb", "sqlite", "integrity", "rebuild"],
        "category": "bugfix",
    },
    # ── Cross-project recall ──
    {
        "query": "JWT token generation and authentication in Atreus",
        "expected_keywords": ["jwt", "token", "atreus", "auth"],
        "category": "cross-project",
    },
    {
        "query": "race conditions in token refresh",
        "expected_keywords": ["race", "token", "refresh", "condition"],
        "category": "cross-project",
    },
    {
        "query": "Drizzle ORM migration journal requirement",
        "expected_keywords": ["drizzle", "migration", "journal"],
        "category": "cross-project",
    },
    {
        "query": "database schema changes for user onboarding",
        "expected_keywords": ["schema", "onboarding", "user", "database"],
        "category": "cross-project",
    },
    # ── Fuzzy / natural language recall ──
    {
        "query": "What safety improvements were made to the memory system?",
        "expected_keywords": ["safety", "fix", "gui", "wal", "chromadb"],
        "category": "fuzzy",
    },
    {
        "query": "How do I configure the memory MCP server?",
        "expected_keywords": ["server", "mcp", "transport", "config"],
        "category": "fuzzy",
    },
    {
        "query": "Tell me about the GUI application for managing memories",
        "expected_keywords": ["gui", "manager", "memory", "tkinter"],
        "category": "fuzzy",
    },
]


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_memories(db_path: Path) -> List[Dict]:
    """Load all memories from SQLite."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, content, tags, memory_type, importance "
        "FROM memories ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def keyword_hit_rate(results: List[Dict], expected_keywords: List[str]) -> float:
    """Fraction of expected keywords found in top-K result texts."""
    if not expected_keywords:
        return 0.0
    combined = " ".join(
        f"{r['title']} {r['content']} {r['tags']}".lower() for r in results
    )
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return hits / len(expected_keywords)


def reciprocal_rank(results: List[Dict], expected_keywords: List[str]) -> float:
    """Mean Reciprocal Rank — how early the first relevant result appears.

    A result is considered relevant if it contains at least half the expected
    keywords.
    """
    threshold = max(1, len(expected_keywords) // 2)
    for rank, r in enumerate(results, 1):
        text = f"{r['title']} {r['content']} {r['tags']}".lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in text)
        if hits >= threshold:
            return 1.0 / rank
    return 0.0


# ── Benchmark Runner ────────────────────────────────────────────────────────


def benchmark_model(
    preset_key: str,
    memories: List[Dict],
) -> Dict:
    """Run all benchmark queries against a single model and return metrics."""

    preset = EMBEDDING_MODEL_PRESETS[preset_key]
    model_name = preset["model_name"]
    query_prefix = preset.get("query_prefix", "")

    print(f"\n{'=' * 70}")
    print(f"Loading model: {model_name}")
    print(f"{'=' * 70}")

    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    # ── Encode corpus ───────────────────────────────────────────────────
    print(f"  Encoding {len(memories)} memories...")
    corpus_texts = [f"{m['title']}\n{m['content']}" for m in memories]

    t0 = time.perf_counter()
    corpus_embeddings = model.encode(
        corpus_texts, show_progress_bar=True, batch_size=64
    )
    encode_time = time.perf_counter() - t0
    print(
        f"  Corpus encoded in {encode_time:.2f}s ({len(memories) / encode_time:.0f} mem/s)"
    )

    # ── Run queries ─────────────────────────────────────────────────────
    query_results = []
    query_times = []

    for bq in BENCHMARK_QUERIES:
        query = bq["query"]
        query_text = f"{query_prefix}{query}" if query_prefix else query

        t0 = time.perf_counter()
        query_emb = model.encode(query_text)
        query_time = time.perf_counter() - t0
        query_times.append(query_time)

        # Compute similarities
        sims = [cosine_similarity(query_emb, ce) for ce in corpus_embeddings]
        ranked_indices = np.argsort(sims)[::-1][:TOP_K]

        top_results = []
        top_sims = []
        for idx in ranked_indices:
            result = memories[idx].copy()
            result["similarity"] = sims[idx]
            top_results.append(result)
            top_sims.append(sims[idx])

        hit_rate = keyword_hit_rate(top_results, bq["expected_keywords"])
        mrr = reciprocal_rank(top_results, bq["expected_keywords"])

        query_results.append(
            {
                "query": query,
                "category": bq["category"],
                "hit_rate": hit_rate,
                "mrr": mrr,
                "top1_sim": top_sims[0] if top_sims else 0.0,
                "top5_avg_sim": statistics.mean(top_sims) if top_sims else 0.0,
                "query_time_ms": query_time * 1000,
                "top_result_title": top_results[0]["title"] if top_results else "",
            }
        )

    # ── Aggregate metrics ───────────────────────────────────────────────
    return {
        "preset_key": preset_key,
        "model_name": model_name,
        "dimensions": corpus_embeddings.shape[1],
        "load_time_s": load_time,
        "encode_time_s": encode_time,
        "encode_speed": len(memories) / encode_time,
        "avg_query_ms": statistics.mean(query_times) * 1000,
        "avg_hit_rate": statistics.mean(r["hit_rate"] for r in query_results),
        "avg_mrr": statistics.mean(r["mrr"] for r in query_results),
        "avg_top1_sim": statistics.mean(r["top1_sim"] for r in query_results),
        "avg_top5_sim": statistics.mean(r["top5_avg_sim"] for r in query_results),
        "query_results": query_results,
    }


# ── Reporting ───────────────────────────────────────────────────────────────


def _best(results: List[Dict], key: str, lower_is_better: bool = False) -> str:
    """Return the preset_key of the best model for a given metric."""
    if lower_is_better:
        return min(results, key=lambda r: r[key])["preset_key"]
    return max(results, key=lambda r: r[key])["preset_key"]


def print_summary(results: List[Dict]):
    """Print the comparison summary table."""
    print(f"\n\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}\n")

    headers = [
        "Metric",
        *[r["preset_key"] for r in results],
        "Winner",
    ]
    rows = [
        ("Dimensions", *[str(r["dimensions"]) for r in results], "-"),
        (
            "Model Load (s)",
            *[f"{r['load_time_s']:.2f}" for r in results],
            _best(results, "load_time_s", lower_is_better=True),
        ),
        (
            "Corpus Encode (s)",
            *[f"{r['encode_time_s']:.2f}" for r in results],
            _best(results, "encode_time_s", lower_is_better=True),
        ),
        (
            "Encode Speed (mem/s)",
            *[f"{r['encode_speed']:.0f}" for r in results],
            _best(results, "encode_speed"),
        ),
        (
            "Avg Query (ms)",
            *[f"{r['avg_query_ms']:.1f}" for r in results],
            _best(results, "avg_query_ms", lower_is_better=True),
        ),
        (
            "Avg Hit Rate",
            *[f"{r['avg_hit_rate']:.1%}" for r in results],
            _best(results, "avg_hit_rate"),
        ),
        (
            "Avg MRR",
            *[f"{r['avg_mrr']:.3f}" for r in results],
            _best(results, "avg_mrr"),
        ),
        (
            "Avg Top-1 Similarity",
            *[f"{r['avg_top1_sim']:.4f}" for r in results],
            _best(results, "avg_top1_sim"),
        ),
        (
            "Avg Top-5 Similarity",
            *[f"{r['avg_top5_sim']:.4f}" for r in results],
            _best(results, "avg_top5_sim"),
        ),
    ]

    print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))


def print_per_query(results: List[Dict]):
    """Print per-query comparison."""
    print(f"\n\n{'=' * 70}")
    print("PER-QUERY RESULTS")
    print(f"{'=' * 70}\n")

    for i, bq in enumerate(BENCHMARK_QUERIES):
        print(f"\n  [{bq['category'].upper()}] {bq['query']}")
        print(f"  Expected keywords: {bq['expected_keywords']}")

        for r in results:
            qr = r["query_results"][i]
            print(
                f"    {r['preset_key']:25s}  "
                f"hit={qr['hit_rate']:.0%}  "
                f"MRR={qr['mrr']:.3f}  "
                f"sim={qr['top1_sim']:.4f}  "
                f'top1="{qr["top_result_title"][:60]}"'
            )

        # Highlight winner
        hit_rates = [
            (r["preset_key"], r["query_results"][i]["hit_rate"]) for r in results
        ]
        mrrs = [(r["preset_key"], r["query_results"][i]["mrr"]) for r in results]
        best_hit = max(hit_rates, key=lambda x: x[1])
        best_mrr = max(mrrs, key=lambda x: x[1])

        # Check if all models tied
        all_same_hit = all(h[1] == hit_rates[0][1] for h in hit_rates)
        all_same_mrr = all(m[1] == mrrs[0][1] for m in mrrs)

        if all_same_hit and all_same_mrr:
            print("    >>> Tie")
        else:
            # Prefer MRR winner, fall back to hit rate winner
            winner = best_mrr[0] if not all_same_mrr else best_hit[0]
            print(f"    >>> Winner: {winner}")


def print_category_breakdown(results: List[Dict]):
    """Print metrics broken down by query category."""
    print(f"\n\n{'=' * 70}")
    print("CATEGORY BREAKDOWN")
    print(f"{'=' * 70}\n")

    categories = sorted(set(bq["category"] for bq in BENCHMARK_QUERIES))

    headers = ["Category", *[r["preset_key"] for r in results]]
    rows = []

    for cat in categories:
        row = [cat]
        for r in results:
            cat_results = [qr for qr in r["query_results"] if qr["category"] == cat]
            avg_mrr = statistics.mean(qr["mrr"] for qr in cat_results)
            avg_hit = statistics.mean(qr["hit_rate"] for qr in cat_results)
            row.append(f"MRR={avg_mrr:.3f}  hit={avg_hit:.0%}")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print(f"Memory database: {DB_PATH}")
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    memories = load_memories(DB_PATH)
    print(f"Loaded {len(memories)} memories from SQLite\n")

    all_results = []
    for preset_key in MODELS_TO_BENCHMARK:
        if preset_key not in EMBEDDING_MODEL_PRESETS:
            print(f"WARNING: Unknown preset '{preset_key}', skipping")
            continue
        result = benchmark_model(preset_key, memories)
        all_results.append(result)

    if len(all_results) >= 2:
        print_summary(all_results)
        print_category_breakdown(all_results)
        print_per_query(all_results)

        # Final verdict
        quality_keys = ["avg_hit_rate", "avg_mrr", "avg_top1_sim"]
        win_counts = {r["preset_key"]: 0 for r in all_results}
        for key in quality_keys:
            winner = max(all_results, key=lambda r: r[key])
            win_counts[winner["preset_key"]] += 1

        print(f"\n\n{'=' * 70}")
        print("VERDICT")
        print(f"{'=' * 70}")

        best_quality = max(win_counts, key=lambda k: win_counts[k])
        if win_counts[best_quality] > 0:
            print(
                f"\n  {best_quality} wins on retrieval quality "
                f"({win_counts[best_quality]}/{len(quality_keys)} quality metrics)"
            )
        else:
            print("\n  Tie on retrieval quality — check per-query results for nuance")

        # Show full win counts
        for name, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {wins}/{len(quality_keys)} quality wins")

        speed_winner = min(all_results, key=lambda r: r["avg_query_ms"])
        print(
            f"\n  {speed_winner['preset_key']} is fastest at query time "
            f"({speed_winner['avg_query_ms']:.1f}ms)"
        )

        encode_winner = max(all_results, key=lambda r: r["encode_speed"])
        print(
            f"  {encode_winner['preset_key']} is fastest at corpus encoding "
            f"({encode_winner['encode_speed']:.0f} mem/s)"
        )
        print()


if __name__ == "__main__":
    main()
