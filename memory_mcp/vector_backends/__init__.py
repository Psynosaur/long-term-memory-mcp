"""
Vector backend abstraction for the long-term memory system.

Provides a pluggable interface for vector storage engines.
Currently supported backends:
  - chromadb (default): Embedded ChromaDB with on-disk persistence
  - pgvector: PostgreSQL with the pgvector extension
"""

from .base import VectorBackend, VectorQueryResult

__all__ = ["VectorBackend", "VectorQueryResult"]
