#!/usr/bin/env python3
"""
Benchmark: File-based context vs Memory-based context token usage.

Measures actual token consumption when using:
  A) A flat file containing the same information as stored memories
  B) Vectorized memory recall (semantic search returning only relevant results)
  C) Structured memory recall (tag/type-based filtering)

Key question: Does the already-vectorized memory have a token advantage
over a file of similar token size?

Answer: The advantage is in SELECTIVE RETRIEVAL, not compression.
This benchmark quantifies exactly how large that advantage is.
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone

import tiktoken

from memory_mcp.memory_system import RobustMemorySystem
from memory_mcp.config import DATA_FOLDER


# ─── Helpers ────────────────────────────────────────────────────────────


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    """Count tokens using cl100k_base (GPT-4 tokenizer)."""
    return len(enc.encode(text))


def format_tokens(n: int) -> str:
    """Format token count with comma separator."""
    return f"{n:,}"


def separator(title: str = "") -> str:
    if title:
        return f"\n{'─' * 70}\n  {title}\n{'─' * 70}"
    return f"{'─' * 70}"


# ─── Benchmark Functions ────────────────────────────────────────────────


def measure_file_approach(memories: list, enc: tiktoken.Encoding) -> dict:
    """
    Simulate reading a flat file that contains ALL memory content.

    This represents the "put everything in a context file" approach
    where the LLM reads the entire file to find relevant info.
    """
    # Build the file content as plain text (most token-efficient format)
    lines = []
    for mem in memories:
        tags_str = ", ".join(mem.get("tags", []))
        lines.append(f"## {mem['title']}")
        lines.append(
            f"Type: {mem['memory_type']} | Tags: {tags_str} | Importance: {mem['importance']}"
        )
        lines.append(mem["content"])
        lines.append("")  # blank line separator

    file_content = "\n".join(lines)
    file_tokens = count_tokens(file_content, enc)

    # Also measure a minimal plain-text version (just content, no metadata)
    plain_content = "\n\n".join(mem["content"] for mem in memories)
    plain_tokens = count_tokens(plain_content, enc)

    # And a JSON version (what you'd get from a tool reading a .json file)
    json_content = json.dumps(memories, indent=2, default=str)
    json_tokens = count_tokens(json_content, enc)

    return {
        "num_memories": len(memories),
        "file_markdown_tokens": file_tokens,
        "file_plaintext_tokens": plain_tokens,
        "file_json_tokens": json_tokens,
        "file_markdown_content": file_content,
    }


def measure_memory_recall(
    memory_system: RobustMemorySystem,
    query: str,
    limit: int,
    enc: tiktoken.Encoding,
) -> dict:
    """
    Measure token cost of a semantic search recall.

    This is what actually happens when the LLM calls search_memories():
    - The query is vectorized locally (0 LLM tokens)
    - ChromaDB returns matching IDs (0 LLM tokens)
    - Full memory records are fetched from SQLite
    - JSON result is serialized and returned to the LLM context
    """
    start = time.time()
    result = memory_system.search_semantic(query, limit)
    search_time_ms = (time.time() - start) * 1000

    if not result.success or not result.data:
        return {
            "query": query,
            "num_results": 0,
            "search_time_ms": search_time_ms,
            "total_tokens": 0,
            "content_tokens": 0,
            "metadata_tokens": 0,
            "overhead_ratio": 0,
        }

    # Build the exact JSON that gets returned to the LLM
    response = {"success": True, "data": result.data}
    response_json = json.dumps(response, default=str)
    total_tokens = count_tokens(response_json, enc)

    # Measure content tokens alone
    content_only = " ".join(mem["content"] for mem in result.data)
    content_tokens = count_tokens(content_only, enc)

    # Measure per-memory overhead
    per_memory_overhead = []
    for mem in result.data:
        # Full memory as JSON
        full_json = json.dumps(mem, default=str)
        full_tok = count_tokens(full_json, enc)

        # Content only
        content_tok = count_tokens(mem["content"], enc)

        # Title tokens
        title_tok = count_tokens(mem.get("title", ""), enc)

        per_memory_overhead.append(
            {
                "id": mem["id"],
                "title": mem.get("title", "")[:50],
                "full_tokens": full_tok,
                "content_tokens": content_tok,
                "title_tokens": title_tok,
                "metadata_overhead": full_tok - content_tok - title_tok,
            }
        )

    metadata_tokens = total_tokens - content_tokens
    overhead_ratio = metadata_tokens / max(total_tokens, 1)

    return {
        "query": query,
        "num_results": len(result.data),
        "search_time_ms": round(search_time_ms, 1),
        "total_tokens": total_tokens,
        "content_tokens": content_tokens,
        "metadata_tokens": metadata_tokens,
        "overhead_ratio": round(overhead_ratio, 3),
        "per_memory": per_memory_overhead,
        "raw_json": response_json,
    }


def measure_structured_recall(
    memory_system: RobustMemorySystem,
    tags: list = None,
    memory_type: str = None,
    limit: int = 20,
    enc: tiktoken.Encoding = None,
) -> dict:
    """
    Measure token cost of a structured (tag/type) search.
    """
    start = time.time()
    if tags:
        result = memory_system.search_structured(tags=tags, limit=limit)
    elif memory_type:
        result = memory_system.search_structured(memory_type=memory_type, limit=limit)
    else:
        result = memory_system.search_structured(limit=limit)
    search_time_ms = (time.time() - start) * 1000

    if not result.success or not result.data:
        return {
            "filter": tags or memory_type or "all",
            "num_results": 0,
            "total_tokens": 0,
        }

    response = {"success": True, "data": result.data}
    response_json = json.dumps(response, default=str)
    total_tokens = count_tokens(response_json, enc)

    content_only = " ".join(mem["content"] for mem in result.data)
    content_tokens = count_tokens(content_only, enc)

    return {
        "filter": str(tags or memory_type or "all"),
        "num_results": len(result.data),
        "search_time_ms": round(search_time_ms, 1),
        "total_tokens": total_tokens,
        "content_tokens": content_tokens,
        "metadata_tokens": total_tokens - content_tokens,
        "overhead_ratio": round(
            (total_tokens - content_tokens) / max(total_tokens, 1), 3
        ),
    }


# ─── Main Benchmark ────────────────────────────────────────────────────


def load_all_memories_raw(ms: RobustMemorySystem) -> list:
    """
    Load all memories directly from SQLite, skipping corrupt rows.
    This avoids the isoformat parsing issue in search_structured().
    """
    cursor = ms.sqlite_conn.execute(
        "SELECT id, title, content, timestamp, tags, importance, memory_type, metadata "
        "FROM memories ORDER BY importance DESC, timestamp DESC"
    )
    rows = cursor.fetchall()
    memories = []
    skipped = 0
    for row in rows:
        try:
            ts = row["timestamp"]
            if not ts:
                skipped += 1
                continue
            memories.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "timestamp": ts,
                    "tags": json.loads(row["tags"]),
                    "importance": row["importance"],
                    "memory_type": row["memory_type"],
                    "metadata": json.loads(row["metadata"]),
                    "match_type": "structured",
                }
            )
        except Exception:
            skipped += 1
    if skipped:
        print(f"       (Skipped {skipped} corrupt rows with invalid timestamps)")
    return memories


def run_benchmark():
    print("=" * 70)
    print("  TOKEN USAGE BENCHMARK")
    print("  File-Based Context vs Vectorized Memory Recall")
    print("=" * 70)

    # Initialize
    enc = tiktoken.get_encoding("cl100k_base")
    print("\n[1/5] Initializing memory system...")
    ms = RobustMemorySystem(DATA_FOLDER)

    # Get all memories from the database
    print("[2/5] Loading all memories from database...")
    all_memories = load_all_memories_raw(ms)
    if not all_memories:
        print("ERROR: No memories found in database. Cannot benchmark.")
        sys.exit(1)

    total_count = len(all_memories)
    print(f"       Found {total_count} memories in database.")

    # ─── Part A: Full file approach ─────────────────────────────────
    print(separator("PART A: File-Based Approach (reading entire file)"))

    file_stats = measure_file_approach(all_memories, enc)
    print(f"""
  Total memories:           {file_stats["num_memories"]}
  
  If stored as markdown:    {format_tokens(file_stats["file_markdown_tokens"])} tokens
  If stored as plain text:  {format_tokens(file_stats["file_plaintext_tokens"])} tokens
  If stored as JSON:        {format_tokens(file_stats["file_json_tokens"])} tokens
  
  These tokens ALL go into LLM context when reading the file,
  regardless of what the user actually asked about.
""")

    # ─── Part B: Semantic search approach ───────────────────────────
    print(separator("PART B: Semantic Search (vector-based recall)"))

    # Test with several realistic queries
    test_queries = [
        ("project architecture", 5),
        ("user preferences", 5),
        ("bug fixes and solutions", 5),
        ("database migration", 3),
        ("GUI manager interface", 3),
    ]

    semantic_results = []
    for query, limit in test_queries:
        result = measure_memory_recall(ms, query, limit, enc)
        semantic_results.append(result)

        print(f"""
  Query: "{query}" (limit={limit})
  Results: {result["num_results"]} memories returned
  Search time: {result["search_time_ms"]}ms (runs locally, 0 LLM tokens)
  
  Token breakdown:
    Total returned:   {format_tokens(result["total_tokens"])} tokens
    Content only:     {format_tokens(result["content_tokens"])} tokens  
    Metadata overhead:{format_tokens(result["metadata_tokens"])} tokens ({result["overhead_ratio"] * 100:.1f}% of total)
""")

        if result.get("per_memory"):
            print("  Per-memory detail:")
            print(f"    {'Title':<50} {'Full':>6} {'Content':>8} {'Overhead':>9}")
            print(f"    {'─' * 50} {'─' * 6} {'─' * 8} {'─' * 9}")
            for pm in result["per_memory"]:
                print(
                    f"    {pm['title']:<50} {pm['full_tokens']:>6} "
                    f"{pm['content_tokens']:>8} {pm['metadata_overhead']:>9}"
                )
            print()

    # ─── Part C: Structured search approach ─────────────────────────
    print(separator("PART C: Structured Search (tag/type filtering)"))

    structured_tests = [
        {"tags": ["preference"], "limit": 20},
        {"tags": ["project", "long-term-memory-mcp"], "limit": 10},
        {"memory_type": "fact", "limit": 10},
        {"tags": ["architecture"], "limit": 10},
    ]

    structured_results = []
    for test in structured_tests:
        result = measure_structured_recall(
            ms,
            tags=test.get("tags"),
            memory_type=test.get("memory_type"),
            limit=test["limit"],
            enc=enc,
        )
        structured_results.append(result)
        print(f"""
  Filter: {result["filter"]} (limit={test["limit"]})
  Results: {result["num_results"]}
  Total tokens:     {format_tokens(result["total_tokens"])}
  Content tokens:   {format_tokens(result["content_tokens"])}
  Metadata overhead:{format_tokens(result["metadata_tokens"])} ({result["overhead_ratio"] * 100:.1f}%)
""")

    # ─── Part D: Comparison Summary ─────────────────────────────────
    print(separator("PART D: Head-to-Head Comparison"))

    file_total = file_stats["file_json_tokens"]  # fairest comparison (JSON vs JSON)

    print(f"""
  BASELINE: All {total_count} memories as a JSON file
  ──────────────────────────────────────────────────
  File tokens (JSON):     {format_tokens(file_total)}
  File tokens (markdown): {format_tokens(file_stats["file_markdown_tokens"])}
  File tokens (plaintext):{format_tokens(file_stats["file_plaintext_tokens"])}
  
  This is the cost of "just read the whole file" every time the LLM
  needs ANY piece of information from the memory store.
  
  SEMANTIC SEARCH: Only relevant memories returned
  ──────────────────────────────────────────────────""")

    for sr in semantic_results:
        if sr["num_results"] == 0:
            continue
        savings = file_total - sr["total_tokens"]
        pct = (savings / max(file_total, 1)) * 100
        print(f"""
  Query: "{sr["query"]}"
    Returned:     {sr["num_results"]} of {total_count} memories
    Tokens used:  {format_tokens(sr["total_tokens"])}
    Tokens saved: {format_tokens(savings)} ({pct:.1f}% reduction vs full file)
    Search cost:  0 LLM tokens (local vector search)""")

    print(f"""
  
  STRUCTURED SEARCH: Filtered by tag/type
  ──────────────────────────────────────────────────""")

    for sr in structured_results:
        if sr["num_results"] == 0:
            continue
        savings = file_total - sr["total_tokens"]
        pct = (savings / max(file_total, 1)) * 100
        print(f"""
  Filter: {sr["filter"]}
    Returned:     {sr["num_results"]} of {total_count} memories
    Tokens used:  {format_tokens(sr["total_tokens"])}
    Tokens saved: {format_tokens(savings)} ({pct:.1f}% reduction vs full file)""")

    # ─── Part E: Metadata Overhead Analysis ─────────────────────────
    print(separator("PART E: Metadata Overhead Analysis"))

    # Calculate average overhead across all semantic results
    all_per_memory = []
    for sr in semantic_results:
        all_per_memory.extend(sr.get("per_memory", []))

    if all_per_memory:
        avg_full = sum(p["full_tokens"] for p in all_per_memory) / len(all_per_memory)
        avg_content = sum(p["content_tokens"] for p in all_per_memory) / len(
            all_per_memory
        )
        avg_title = sum(p["title_tokens"] for p in all_per_memory) / len(all_per_memory)
        avg_overhead = sum(p["metadata_overhead"] for p in all_per_memory) / len(
            all_per_memory
        )

        print(f"""
  Average per-memory token breakdown (across {len(all_per_memory)} recalled memories):
  
    Content:           {avg_content:>8.1f} tokens ({avg_content / avg_full * 100:.1f}%)
    Title:             {avg_title:>8.1f} tokens ({avg_title / avg_full * 100:.1f}%)
    Metadata overhead: {avg_overhead:>8.1f} tokens ({avg_overhead / avg_full * 100:.1f}%)
    ─────────────────────────────────
    Total per memory:  {avg_full:>8.1f} tokens
  
  Metadata fields contributing to overhead:
    - id (mem_XXXXXXXX_XXXXXXXXXXXXXXXX): ~15 tokens
    - timestamp (ISO 8601):               ~15 tokens
    - tags (JSON array):                  ~15-30 tokens
    - importance + memory_type:           ~5 tokens
    - metadata dict (reinforcement etc.): ~10-20 tokens
    - relevance_score + match_type:       ~8 tokens
    - JSON structure (keys, brackets):    ~20 tokens
    ─────────────────────────────────
    Estimated overhead per memory:        ~88-118 tokens
  
  Fields the LLM does NOT need but currently receives:
    - reinforcement_accum (internal bookkeeping)
    - last_decay_at (internal bookkeeping)  
    - content_hash (deduplication artifact)
""")

    # ─── Part F: The Vectorization Advantage ────────────────────────
    print(separator("PART F: The Vectorization Advantage"))

    # Calculate what percentage of the DB each semantic search needed
    if semantic_results:
        avg_results = sum(
            sr["num_results"] for sr in semantic_results if sr["num_results"] > 0
        ) / max(sum(1 for sr in semantic_results if sr["num_results"] > 0), 1)
        avg_tokens = sum(
            sr["total_tokens"] for sr in semantic_results if sr["num_results"] > 0
        ) / max(sum(1 for sr in semantic_results if sr["num_results"] > 0), 1)
        selectivity = avg_results / max(total_count, 1) * 100

        print(f"""
  Database size:           {total_count} memories
  Full file cost:          {format_tokens(file_total)} tokens (JSON)
  
  Avg semantic search:     {avg_results:.1f} memories returned ({selectivity:.1f}% of DB)
  Avg tokens per search:   {format_tokens(int(avg_tokens))}
  Avg savings per search:  {format_tokens(file_total - int(avg_tokens))} tokens
  
  ┌───────────────────────────────────────────────────────────────┐
  │  VERDICT                                                      │
  │                                                               │
  │  The vector index acts as a ZERO-TOKEN FILTER.                │
  │                                                               │
  │  • Embedding + cosine search runs on your CPU = 0 LLM tokens │
  │  • Only matching results are sent to the LLM context          │
  │  • Per-result cost is ~{avg_overhead:.0f} tokens higher than raw file text    │
  │    (metadata overhead), but you send FAR fewer results        │
  │                                                               │
  │  At {total_count} memories, semantic search uses ~{selectivity:.0f}% of       │
  │  what a full file read would cost.                            │
  │                                                               │
  │  Break-even point: The memory approach loses its advantage    │
  │  when you need to recall >{int(file_total / max(avg_full, 1))} memories at once         │
  │  (at which point the metadata overhead exceeds the savings).  │
  └───────────────────────────────────────────────────────────────┘
""")

    # ─── Scaling projections ────────────────────────────────────────
    print(separator("PART G: Scaling Projections"))
    print(
        """
  How token savings scale with database size:
  (assuming avg {avg_content:.0f} content tokens/memory, {avg_overhead:.0f} overhead tokens/memory)
  (semantic search returning 5 results)
  """.format(avg_content=avg_content, avg_overhead=avg_overhead)
        if all_per_memory
        else ""
    )

    if all_per_memory:
        per_mem_file_tokens = file_total / max(
            total_count, 1
        )  # average file tokens per memory
        search_return_count = 5

        print(
            f"  {'DB Size':<12} {'File Read':>12} {'Memory Recall':>14} {'Savings':>12} {'Reduction':>10}"
        )
        print(f"  {'─' * 12} {'─' * 12} {'─' * 14} {'─' * 12} {'─' * 10}")

        for db_size in [10, 50, 100, 250, 500, 1000, 5000]:
            file_cost = int(per_mem_file_tokens * db_size)
            # Memory cost = N returned * (avg_content + avg_overhead) + JSON wrapper
            memory_cost = int(search_return_count * avg_full + 10)  # +10 for wrapper
            saved = file_cost - memory_cost
            pct = (saved / max(file_cost, 1)) * 100

            print(
                f"  {db_size:<12} {format_tokens(file_cost):>12} "
                f"{format_tokens(memory_cost):>14} {format_tokens(saved):>12} {pct:>9.1f}%"
            )

    print(f"\n{'=' * 70}")
    print("  Benchmark complete.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_benchmark()
