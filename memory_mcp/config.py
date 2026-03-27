"""
Configuration module for the long-term memory MCP server.

Contains all configuration constants for decay, reinforcement, and storage.
"""

from pathlib import Path
import os
import warnings


# Data Storage Configuration
# You can override the data folder by setting the AI_COMPANION_DATA_DIR environment variable.
# Example (PowerShell): $env:AI_COMPANION_DATA_DIR = "D:\a.i. apps\long_term_memory_mcp\data"

DATA_FOLDER = Path(
    os.environ.get(
        "AI_COMPANION_DATA_DIR", str(Path.home() / "Documents" / "ai_companion_memory")
    )
)

# ChromaDB collection name — single source of truth for all modules
CHROMA_COLLECTION_NAME = "ai_companion_memories"


# ── Embedding Model Configuration ──────────────────────────────────────────
#
# Available model presets for the memory vector store.
# Set MEMORY_EMBEDDING_MODEL env var to switch models, or change the default below.
#
# After changing models you MUST rebuild the vector index because different
# models produce incompatible embedding spaces (even when dimensions match).
# Use the rebuild_vectors MCP tool or call memory_system.rebuild_vector_index().
#
# ┌──────────────────────────────┬──────┬────────────┬────────┬────────────┐
# │ Model                        │ Dims │ Max Tokens │ Params │ MTEB Avg   │
# ├──────────────────────────────┼──────┼────────────┼────────┼────────────┤
# │ all-MiniLM-L12-v2 (legacy)   │ 384  │ 256        │ 33M    │ ~57        │
# │ bge-small-en-v1.5 (default)  │ 384  │ 512        │ 33M    │ ~62        │
# │ all-MiniLM-L6-v2             │ 384  │ 256        │ 22M    │ ~56        │
# │ bge-base-en-v1.5             │ 768  │ 512        │ 109M   │ ~64        │
# │ all-mpnet-base-v2            │ 768  │ 384        │ 109M   │ ~58        │
# │ nomic-embed-text-v1.5        │ 768  │ 8192       │ 137M   │ ~62        │
# │ snowflake-arctic-embed-s     │ 384  │ 512        │ 33M    │ ~61        │
# └──────────────────────────────┴──────┴────────────┴────────┴────────────┘
#
# Recommended for memories: bge-small-en-v1.5
#   - Same 384 dimensions as the legacy model (no ChromaDB schema change)
#   - Higher retrieval quality (MTEB retrieval ~52 vs ~43)
#   - Double the token window (512 vs 256)
#   - Smaller on disk (~65MB vs ~120MB)
#
# To use sentence-transformers model names with the library:
#   - "BAAI/bge-small-en-v1.5"      -> requires "query: " prefix for queries
#   - "sentence-transformers/all-MiniLM-L12-v2"  -> no prefix needed
#   - "sentence-transformers/all-MiniLM-L6-v2"   -> no prefix needed

EMBEDDING_MODEL_PRESETS = {
    "bge-small-en-v1.5": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "max_tokens": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "description": "Best quality/size ratio for memories (recommended)",
    },
    "all-MiniLM-L12-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L12-v2",
        "dimensions": 384,
        "max_tokens": 256,
        "query_prefix": "",
        "description": "Legacy model — good general purpose, smaller token window",
    },
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "max_tokens": 256,
        "query_prefix": "",
        "description": "Fastest inference, lowest quality of the three small models",
    },
    "bge-base-en-v1.5": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768,
        "max_tokens": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "description": "Higher quality, 768 dims — requires vector rebuild + more RAM",
    },
    "all-mpnet-base-v2": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "max_tokens": 384,
        "query_prefix": "",
        "description": "Highest quality SBERT model, 768 dims",
    },
    "nomic-embed-text-v1.5": {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "dimensions": 768,
        "max_tokens": 8192,
        "query_prefix": "search_query: ",
        "description": "Best for documentation — 8192 token window, Matryoshka dims",
    },
    "snowflake-arctic-embed-s": {
        "model_name": "Snowflake/snowflake-arctic-embed-s",
        "dimensions": 384,
        "max_tokens": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "description": "Good alternative to BGE-small, same dimensions",
    },
}

# Active embedding model — override with MEMORY_EMBEDDING_MODEL env var
# e.g. export MEMORY_EMBEDDING_MODEL="all-MiniLM-L12-v2"
EMBEDDING_MODEL = os.environ.get("MEMORY_EMBEDDING_MODEL", "bge-small-en-v1.5")

# Resolve the active preset — warn loudly on invalid key so typos don't go unnoticed
if EMBEDDING_MODEL not in EMBEDDING_MODEL_PRESETS:
    warnings.warn(
        f"Unknown MEMORY_EMBEDDING_MODEL={EMBEDDING_MODEL!r}. "
        f"Valid options: {list(EMBEDDING_MODEL_PRESETS)}. "
        f"Falling back to 'bge-small-en-v1.5'.",
        stacklevel=2,
    )
    EMBEDDING_MODEL = (
        "bge-small-en-v1.5"  # correct the variable so downstream sees the real model
    )
EMBEDDING_MODEL_CONFIG = EMBEDDING_MODEL_PRESETS[EMBEDDING_MODEL]

# Validate that the resolved config has all required keys
_REQUIRED_CONFIG_KEYS = {"model_name", "dimensions", "max_tokens", "query_prefix"}
_missing = _REQUIRED_CONFIG_KEYS - set(EMBEDDING_MODEL_CONFIG)
if _missing:
    raise ValueError(
        f"EMBEDDING_MODEL_CONFIG for {EMBEDDING_MODEL!r} is missing keys: {_missing}. "
        f"Each preset must define: {_REQUIRED_CONFIG_KEYS}"
    )


# Lazy Decay Configuration
DECAY_ENABLED = True

# Half-life in days by memory_type (how fast each type fades)
DECAY_HALF_LIFE_DAYS_BY_TYPE = {
    "conversation": 45,
    "fact": 120,
    "preference": 90,
    "task": 30,
    "ephemeral": 10,
}
DECAY_HALF_LIFE_DAYS_DEFAULT = 60

# Minimum floor per type (never decay below this)
DECAY_MIN_IMPORTANCE_BY_TYPE = {
    "conversation": 2,
    "fact": 3,
    "preference": 2,
    "task": 1,
    "ephemeral": 1,
}
DECAY_MIN_IMPORTANCE_DEFAULT = 1

# Tags that prevent decay entirely
DECAY_PROTECT_TAGS = {"core", "identity", "pinned"}

# Writeback policy (to avoid churn)
DECAY_WRITEBACK_STEP = 0.5  # only persist if change >= 0.5
DECAY_MIN_INTERVAL_HOURS = 12  # don't write decay more often than this


# Reinforcement Configuration
REINFORCEMENT_ENABLED = True
REINFORCEMENT_STEP = 0.1  # amount per retrieval
REINFORCEMENT_WRITEBACK_STEP = 0.5  # write to DB when accumulated ≥ 0.5
REINFORCEMENT_MAX = 10  # cap importance


# ── Staleness Score Configuration ───────────────────────────────────────────
#
# Staleness is a STRUCTURAL signal — separate from importance decay.
# Decay tracks usage (importance falls when a memory isn't accessed).
# Staleness tracks age vs expected lifetime for the memory type.
#
# staleness_score = min(1.0, days_since_stored / expected_lifetime_days)
# 0.0 = fresh, 1.0 = fully stale
#
# Appended to every search result so the agent can see it.
# Memories with staleness_score > STALENESS_WARN_THRESHOLD and
# memory_type in STALENESS_WARN_TYPES are flagged at session start.
#
STALENESS_ENABLED = True

STALENESS_EXPECTED_LIFETIME_DAYS = {
    "conversation": 7,
    "fact": 30,  # architecture facts go stale fast
    "preference": 180,
    "task": 14,
    "event": 365,
    "ephemeral": 3,
}
STALENESS_EXPECTED_LIFETIME_DEFAULT = 60  # fallback for unknown types

# Threshold above which a memory is considered stale enough to flag
STALENESS_WARN_THRESHOLD = 0.8

# Only flag these types at session start — no need to warn about events/conversations
STALENESS_WARN_TYPES = {"fact", "preference"}


# ── Contradiction Detection Configuration ───────────────────────────────────
#
# When remember() is called with memory_type in CONTRADICTION_CHECK_TYPES,
# a semantic similarity search is run against existing memories of the same type.
# If a candidate exceeds CONTRADICTION_SIMILARITY_THRESHOLD the write is still
# performed but the result includes a warning so the agent can decide whether
# to update the existing memory instead.
#
CONTRADICTION_DETECTION_ENABLED = True

# Cosine similarity range for contradiction detection.
# - Below LOW: unrelated — ignore
# - Between LOW and HIGH: similar topic, possibly contradictory — warn
# - Above HIGH: near-identical content — already caught by the exact hash check above
CONTRADICTION_SIMILARITY_THRESHOLD = 0.75  # lower bound (was 0.85)
CONTRADICTION_SIMILARITY_UPPER = 0.98  # upper bound — above this = near-duplicate

# Only check these types — no value checking conversations or events
CONTRADICTION_CHECK_TYPES = {"fact", "preference"}
