"""
Configuration module for the long-term memory MCP server.

Contains all configuration constants for decay, reinforcement, and storage.
"""

from pathlib import Path
import os


# Data Storage Configuration
# You can override the data folder by setting the AI_COMPANION_DATA_DIR environment variable.
# Example (PowerShell): $env:AI_COMPANION_DATA_DIR = "D:\a.i. apps\long_term_memory_mcp\data"

DATA_FOLDER = Path(
    os.environ.get(
        "AI_COMPANION_DATA_DIR", str(Path.home() / "Documents" / "ai_companion_memory")
    )
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
