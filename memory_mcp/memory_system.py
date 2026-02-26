"""
Core memory system module.

Contains the RobustMemorySystem class that manages hybrid storage
(SQLite + ChromaDB) and implements all memory operations.
"""

from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Dict, Any
import shutil
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
import logging
from logging.handlers import TimedRotatingFileHandler

# Third-party imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import tiktoken

# Local imports
from .models import MemoryRecord, SearchResult, Result
from .config import (
    DATA_FOLDER,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_CONFIG,
    DECAY_ENABLED,
    DECAY_HALF_LIFE_DAYS_BY_TYPE,
    DECAY_HALF_LIFE_DAYS_DEFAULT,
    DECAY_MIN_IMPORTANCE_BY_TYPE,
    DECAY_MIN_IMPORTANCE_DEFAULT,
    DECAY_PROTECT_TAGS,
    DECAY_WRITEBACK_STEP,
    DECAY_MIN_INTERVAL_HOURS,
    REINFORCEMENT_ENABLED,
    REINFORCEMENT_STEP,
    REINFORCEMENT_WRITEBACK_STEP,
    REINFORCEMENT_MAX,
)


class RobustMemorySystem:
    """
    Hybrid memory system combining:
    1. ChromaDB for semantic/vector search
    2. SQLite for structured queries and metadata
    3. JSON backup files for portability
    """

    def __init__(self, data_folder: Path = DATA_FOLDER):
        self.data_folder = Path(data_folder)
        self.db_folder = self.data_folder / "memory_db"
        self.backup_folder = self.data_folder / "memory_backups"
        self.sqlite_path = self.db_folder / "memories.db"

        # Create directories
        self.db_folder.mkdir(parents=True, exist_ok=True)
        self.backup_folder.mkdir(parents=True, exist_ok=True)

        # Predeclare attributes for linters/type checkers
        self.logger = None  # will be set in _setup_logging
        self.sqlite_conn = None  # will be set in _init_sqlite
        self.chroma_client: Optional[object] = None
        self.chroma_collection: Optional[object] = None
        self.embedding_model: Optional[object] = None
        self._query_prefix: str = ""  # set by _init_embeddings (e.g. BGE needs prefix)
        self.tokenizer: Optional[object] = None

        # Setup logging
        self._setup_logging()

        # Initialize components
        self._init_sqlite()
        self._init_chromadb()
        self._init_embeddings()
        self._init_tokenizer()

        # Perform integrity check on startup
        self._integrity_check()

    def _setup_logging(self):
        """Setup logging for debugging and monitoring"""
        log_file = self.data_folder / "memory_system.log"

        # Daily rotation, keep 30 days
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=30, utc=False
        )
        file_handler.setLevel(logging.INFO)  # Keep full INFO in the file

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only WARNING+ to console/stderr

        # Set format for both
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,  # Overall minimum level
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[file_handler, console_handler],
        )

        self.logger = logging.getLogger(__name__)

    def _init_sqlite(self):
        """Initialize SQLite database for structured data"""
        try:
            self.sqlite_conn = sqlite3.connect(
                str(self.sqlite_path), check_same_thread=False, timeout=30.0
            )
            self.sqlite_conn.row_factory = sqlite3.Row

            self.sqlite_conn.execute("PRAGMA journal_mode=WAL;")
            self.sqlite_conn.execute(
                "PRAGMA wal_autocheckpoint=500;"
            )  # checkpoint every 500 pgs
            # SAFETY: synchronous=FULL ensures WAL data reaches disk before
            # returning.  Prevents corruption on macOS sleep / sudden power loss.
            # (WAL mode defaults to NORMAL which can lose committed data.)
            self.sqlite_conn.execute("PRAGMA synchronous=FULL;")
            self.sqlite_conn.commit()

            # Create base tables (OK to have DEFAULT CURRENT_TIMESTAMP here for fresh DBs)
            self.sqlite_conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,  
                    timestamp TEXT NOT NULL,  
                    tags TEXT,  -- JSON array  
                    importance INTEGER DEFAULT 5,  
                    memory_type TEXT DEFAULT 'conversation',  
                    metadata TEXT,  -- JSON object  
                    content_hash TEXT,  
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,  
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,  
                    last_accessed TEXT DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER DEFAULT 0
                );  
      
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);  
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);  
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);  
                CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);  
                CREATE TABLE IF NOT EXISTS memory_stats (  
                    key TEXT PRIMARY KEY,  
                    value TEXT,  
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP  
                );  
            """
            )

            # Migration: ensure last_accessed exists in existing DBs
            cursor = self.sqlite_conn.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]

            if "last_accessed" not in columns:
                self.logger.info(
                    "Adding last_accessed column to existing memories table"
                )
                # 1) Add column WITHOUT default (avoids 'non-constant default' error)
                self.sqlite_conn.execute(
                    "ALTER TABLE memories ADD COLUMN last_accessed TEXT"
                )

                # 2) Backfill existing rows
                now_iso = datetime.now(timezone.utc).isoformat()
                self.sqlite_conn.execute(
                    "UPDATE memories SET last_accessed = COALESCE(created_at, ?)",
                    (now_iso,),
                )

                self.sqlite_conn.commit()

            # Now that the column exists, create its index
            self.sqlite_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed)"
            )

            # Migration: ensure token_count exists in existing DBs
            if "token_count" not in columns:
                self.logger.info("Adding token_count column to existing memories table")
                # Add column without default
                self.sqlite_conn.execute(
                    "ALTER TABLE memories ADD COLUMN token_count INTEGER"
                )

                # Backfill existing rows with 0 (will be recalculated on next access)
                self.sqlite_conn.execute(
                    "UPDATE memories SET token_count = 0 WHERE token_count IS NULL"
                )

                self.sqlite_conn.commit()

            # Schema version bookkeeping
            self.sqlite_conn.execute(
                "INSERT OR REPLACE INTO memory_stats (key, value, updated_at)"
                "VALUES ('schema_version', '1.1', CURRENT_TIMESTAMP)"
            )

            # Normalize last_backup to an ISO UTC string
            now_iso = datetime.now(timezone.utc).isoformat()
            self.sqlite_conn.execute(
                "INSERT OR REPLACE INTO memory_stats (key, value, updated_at)"
                "VALUES ('last_backup', ?, ?)",
                (now_iso, now_iso),
            )

            self.sqlite_conn.commit()
            self.logger.info("SQLite database initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize SQLite: %s", e)
            raise

    def _init_chromadb(self):
        """Initialize ChromaDB for vector storage"""
        try:
            # Use persistent storage
            chroma_path = str(self.db_folder / "chroma_db")

            self.chroma_client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Use a stable collection name and cosine space
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={
                    "description": "Long-term memory for AI companion",
                    "hnsw:space": "cosine",  # important for sentence embeddings
                },
                embedding_function=None,  # we pass embeddings manually
            )

            self.logger.info("ChromaDB initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize ChromaDB: %s", e)
            raise

    def _init_embeddings(self):
        """Initialize sentence transformer for embeddings.

        Uses the model configured in config.py (EMBEDDING_MODEL_CONFIG).
        Override with env var MEMORY_EMBEDDING_MODEL to switch models.
        After switching, run rebuild_vector_index() to re-embed all memories.
        """
        try:
            model_name = EMBEDDING_MODEL_CONFIG["model_name"]
            # Normalise the query prefix: strip whitespace so the separator
            # between prefix and query is always explicit and consistent.
            self._query_prefix = EMBEDDING_MODEL_CONFIG.get("query_prefix", "").strip()
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(
                "Embedding model '%s' (preset: %s, dims: %d) loaded successfully",
                model_name,
                EMBEDDING_MODEL,
                EMBEDDING_MODEL_CONFIG["dimensions"],
            )

        except Exception as e:
            self.logger.error("Failed to load embedding model: %s", e)
            # Fallback to a simpler approach if needed
            raise

    def _init_tokenizer(self):
        """Initialize tiktoken tokenizer for token counting"""
        try:
            # Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.logger.info("Tiktoken tokenizer initialized successfully")

        except Exception as e:
            self.logger.error("Failed to load tiktoken tokenizer: %s", e)
            # Fallback: tokenizer will be None, and _count_tokens will estimate
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            if self.tokenizer:
                return len(self.tokenizer.encode(text))
            else:
                # Fallback: rough estimation (words * 1.3)
                return int(len(text.split()) * 1.3)
        except Exception as e:
            self.logger.warning("Token counting failed, using estimation: %s", e)
            return int(len(text.split()) * 1.3)

    def _generate_id(self, content: str, timestamp: datetime) -> str:
        """Generate unique ID for memory record"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        time_hash = hashlib.sha256(timestamp.isoformat().encode()).hexdigest()[:8]
        return f"mem_{time_hash}_{content_hash}"

    def _content_hash(self, content: str) -> str:
        """Generate content hash for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _safe_parse_timestamp(
        self, raw: str, fallback_id: str = ""
    ) -> Optional[datetime]:
        """
        Parse an ISO timestamp string, returning None on failure.

        Logs a warning for corrupt rows so they can be identified and repaired,
        rather than crashing the entire search.
        """
        if not raw:
            self.logger.warning(
                "Empty timestamp for memory %s — skipping row", fallback_id
            )
            return None
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError) as exc:
            self.logger.warning(
                "Invalid timestamp %r for memory %s: %s — skipping row",
                raw,
                fallback_id,
                exc,
            )
            return None

    def _integrity_check(self):
        """Check data integrity of SQLite database and cross-check with ChromaDB"""
        try:
            # --- SQLite page-level integrity check ---
            # This detects corruption in the B-tree structure, free-list, and
            # WAL file.  Returns 'ok' on success or a list of problems.
            ic_result = self.sqlite_conn.execute("PRAGMA integrity_check;").fetchone()
            if ic_result and ic_result[0] != "ok":
                self.logger.error("SQLite integrity_check FAILED: %s", ic_result[0])
                # Surface the problem but don't raise — let the server start
                # so the user can attempt a backup or repair.
            else:
                self.logger.info("SQLite integrity_check passed")

            # --- Record-count cross-check ---
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM memories")
            sqlite_count = cursor.fetchone()[0]

            chroma_count = self.chroma_collection.count()

            self.logger.info(
                "Integrity check: SQLite=%s, ChromaDB=%s",
                sqlite_count,
                chroma_count,
            )

            if sqlite_count != chroma_count:
                self.logger.warning(
                    "Record count mismatch between SQLite (%d) and ChromaDB (%d). "
                    "Run rebuild_vector_index to resync.",
                    sqlite_count,
                    chroma_count,
                )

        except Exception as e:
            self.logger.error("Integrity check failed: %s", e)

    def remember(
        self,
        title: str,
        content: str,
        tags: List[str] = None,
        importance: int = 5,
        memory_type: str = "conversation",
        metadata: Dict[str, Any] = None,
    ) -> Result:
        """
        Store a new memory with both vector and structured storage
        """
        try:
            if not title or not content:
                return Result(success=False, reason="Title and content are required")

            # Validate importance
            importance = max(1, min(10, importance))

            # Prepare data
            timestamp = datetime.now(timezone.utc)
            tags = tags or []
            metadata = metadata or {}

            # Generate ID and hash
            memory_id = self._generate_id(content, timestamp)
            content_hash = self._content_hash(content)

            # Count tokens in content
            token_count = self._count_tokens(content)

            # Check for duplicates
            cursor = self.sqlite_conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (content_hash,)
            )
            if cursor.fetchone():
                return Result(success=False, reason="Duplicate content detected")

            # Create memory record
            record = MemoryRecord(
                id=memory_id,
                title=title,
                content=content,
                timestamp=timestamp,
                tags=tags,
                importance=importance,
                memory_type=memory_type,
                metadata=metadata,
            )

            # Store in SQLite FIRST, then ChromaDB.
            # Order matters: if we crash between the two writes, it's better
            # to have a SQLite row with no vector (detectable via count
            # mismatch and fixable with rebuild_vector_index) than an orphan
            # vector with no SQLite row (invisible and unfixable).
            self.sqlite_conn.execute(
                """
                INSERT INTO memories 
                (id, title, content, timestamp, tags, importance,
                 memory_type, metadata, content_hash, last_accessed, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.id,
                    record.title,
                    record.content,
                    record.timestamp.isoformat(),
                    json.dumps(record.tags),
                    record.importance,
                    record.memory_type,
                    json.dumps(record.metadata),
                    content_hash,
                    record.timestamp.isoformat(),  # Set last_accessed to creation time
                    token_count,
                ),
            )
            self.sqlite_conn.commit()

            # Generate embedding and store in ChromaDB
            # Combine title and content for better semantic search
            text_for_embedding = f"{title}\n{content}"
            embedding = self.embedding_model.encode(text_for_embedding).tolist()

            try:
                self.chroma_collection.add(
                    ids=[record.id],
                    embeddings=[embedding],
                    documents=[text_for_embedding],
                    metadatas=[
                        {
                            "title": title,
                            "timestamp": record.timestamp.isoformat(),
                            "importance": importance,
                            "memory_type": memory_type,
                            "tags": json.dumps(tags),
                        }
                    ],
                )

                try:
                    if hasattr(self.chroma_client, "persist"):
                        self.chroma_client.persist()
                except Exception as pe:
                    self.logger.warning("Chroma persist warning: %s", pe)

                # Debug: check what Chroma actually contains after add
                self.logger.info("Chroma after add: %s", self._debug_vector_index())
            except Exception as chroma_err:
                # ChromaDB write failed — roll back the SQLite insert so the
                # two stores stay in sync. The next call will retry both.
                self.logger.error(
                    "ChromaDB add failed for %s, rolling back SQLite insert: %s",
                    record.id,
                    chroma_err,
                )
                self.sqlite_conn.execute(
                    "DELETE FROM memories WHERE id = ?", (record.id,)
                )
                self.sqlite_conn.commit()
                return Result(
                    success=False,
                    reason=f"ChromaDB storage failed: {chroma_err}",
                )

            # Trigger backup if needed
            self._maybe_backup()

            self.logger.info("Memory stored successfully: %s", memory_id)
            rec = asdict(record)
            rec["timestamp"] = record.timestamp.isoformat()
            return Result(success=True, data=[rec])

        except Exception as e:
            self.logger.error("Failed to store memory: %s", e)
            self.sqlite_conn.rollback()
            return Result(success=False, reason=f"Storage error: {str(e)}")

    def search_semantic(
        self, query: str, limit: int = 10, min_relevance: float = 0.15
    ) -> Result:
        """
        Semantic search using vector similarity with adaptive thresholding + top-1 fallback.
        """
        try:
            if not query.strip():
                return Result(success=False, reason="Query cannot be empty")

            # Debug: check what Chroma contains before query
            self.logger.info(
                "Chroma before query: %s",
                self._debug_vector_index(),
            )

            # Generate query embedding (apply model-specific prefix if set,
            # always separated by a single space so prefixes don't run into the query)
            query_text = (
                f"{self._query_prefix} {query}" if self._query_prefix else query
            )
            query_embedding = self.embedding_model.encode(query_text).tolist()

            # Search ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            if not results["ids"][0]:
                return Result(success=True, data=[])

            ids = results["ids"][0]
            distances = results["distances"][0]
            similarities = [1.0 - d for d in distances]

            # Adaptive threshold anchored on top match, with clamps
            if similarities:
                top_sim = similarities[0]
                adaptive = max(0.12, min(0.35, top_sim - 0.08))
                threshold = max(min_relevance, adaptive)
            else:
                threshold = min_relevance

            self.logger.info("Adaptive threshold computed: %.3f", threshold)

            search_results = []
            now_iso = datetime.now(timezone.utc).isoformat()

            # First pass: collect those meeting threshold
            selected = [
                (mid, sim) for mid, sim in zip(ids, similarities) if sim >= threshold
            ]

            # FALLBACK: if none pass threshold, keep the top-1 candidate anyway
            if not selected and ids:
                if similarities[0] >= 0.08:  # only fallback if it's not total garbage
                    self.logger.info(
                        "No candidates passed threshold; using top-1 semantic fallback"
                    )
                    selected = [(ids[0], similarities[0])]
                else:
                    self.logger.info(
                        "No candidates passed and top-1 sim %.3f < 0.08, skipping fallback",
                        similarities[0],
                    )

            # Fetch selected rows and reinforce
            for i, (memory_id, relevance) in enumerate(selected):
                if i < 3:
                    self.logger.info(
                        "Candidate %d: relevance=%.3f, threshold=%.3f",
                        i,
                        relevance,
                        threshold,
                    )

                cursor = self.sqlite_conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()

                if not row:
                    continue

                # Lazy decay before reinforcement
                self._maybe_decay(row)
                self._maybe_reinforce(row)

                # Reinforcement
                self.sqlite_conn.execute(
                    "UPDATE memories SET last_accessed = ? WHERE id = ?",
                    (now_iso, memory_id),
                )

                ts = self._safe_parse_timestamp(row["timestamp"], row["id"])
                if ts is None:
                    continue

                record = MemoryRecord(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    timestamp=ts,
                    tags=json.loads(row["tags"]),
                    importance=row["importance"],
                    memory_type=row["memory_type"],
                    metadata=json.loads(row["metadata"]),
                )

                search_results.append(
                    SearchResult(
                        record=record,
                        relevance_score=relevance,
                        match_type="semantic"
                        if relevance >= threshold
                        else "semantic_fallback",
                    )
                )

            # Commit the last_accessed updates
            self.sqlite_conn.commit()

            # Sort by relevance
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Convert to dict format
            result_data = []
            for sr in search_results:
                result_dict = asdict(sr.record)
                result_dict["timestamp"] = sr.record.timestamp.isoformat()
                result_dict["relevance_score"] = sr.relevance_score
                result_dict["match_type"] = sr.match_type
                result_data.append(result_dict)

            self.logger.info(
                "Semantic search returned %d results {threshold=%.3f)",
                len(result_data),
                threshold,
            )
            return Result(success=True, data=result_data)

        except Exception as e:
            self.logger.error("Semantic search failed: %s", e)
            return Result(success=False, reason=f"Search error: {str(e)}")

    def search_structured(
        self,
        memory_type: str = None,
        tags: List[str] = None,
        importance_min: int = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 50,
    ) -> Result:
        """
        Structured search using SQL queries
        """
        try:
            conditions = []
            params = []

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
                # Search for any of the provided tags
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
                conditions.append(f"({' OR '.join(tag_conditions)})")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
                SELECT * FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """
            params.append(limit)

            cursor = self.sqlite_conn.execute(query, params)
            rows = cursor.fetchall()

            # REINFORCEMENT: Update last_accessed for retrieved memories
            now_iso = datetime.now(timezone.utc).isoformat()
            memory_ids = [row["id"] for row in rows]

            if memory_ids:
                placeholders = ",".join(["?" for _ in memory_ids])
                self.sqlite_conn.execute(
                    f"UPDATE memories SET last_accessed = ? WHERE id IN ({placeholders})",
                    [now_iso] + memory_ids,
                )
                self.sqlite_conn.commit()

            # Convert to MemoryRecord objects
            results = []
            for row in rows:
                self._maybe_decay(row)
                self._maybe_reinforce(row)

                ts = self._safe_parse_timestamp(row["timestamp"], row["id"])
                if ts is None:
                    continue

                record = MemoryRecord(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    timestamp=ts,
                    tags=json.loads(row["tags"]),
                    importance=row["importance"],
                    memory_type=row["memory_type"],
                    metadata=json.loads(row["metadata"]),
                )

                result_dict = asdict(record)
                result_dict["timestamp"] = record.timestamp.isoformat()
                result_dict["match_type"] = "structured"
                results.append(result_dict)

            # Batch-commit any decay/reinforcement writes from the loop above
            if rows:
                self.sqlite_conn.commit()

            self.logger.info(
                "Structured search returned %d results",
                len(results),
            )
            return Result(success=True, data=results)

        except Exception as e:
            self.logger.error("Structured search failed: %s", e)
            return Result(success=False, reason=f"Search error: {str(e)}")

    def get_recent(self, limit: int = 20, current_project: str = None) -> Result:
        """
        Get most recent memories, optionally filtered by project.

        Memories with memory_type="preference" are treated as first-class
        citizens: they are always fetched **on top of** the requested limit.
        The caller asks for ``limit`` recent memories and gets all
        preferences prepended for free.

        Args:
            limit: Maximum number of *recent* memories to return
                   (preferences are added on top of this).
            current_project: Optional project identifier to filter by.
                           If provided, only returns memories with this project tag.
        """
        try:
            # --- 1. Fetch ALL preference memories by memory_type (global, not project-scoped) ---
            # Uses memory_type="preference" (not tags) for strict matching.
            # This ensures preferences like "I prefer dark mode" are returned
            # even when current_project is set and the preference isn't tagged
            # with that project.
            pref_result = self.search_structured(memory_type="preference", limit=100)
            pref_items = (
                pref_result.data if pref_result.success and pref_result.data else []
            )

            # Collect preference IDs for deduplication
            pref_ids = {item["id"] for item in pref_items}

            # --- 2. Fetch regular recent memories (full limit) ---
            if current_project:
                recent_result = self.search_structured(
                    tags=[current_project], limit=limit
                )
            else:
                recent_result = self.search_structured(limit=limit)
            recent_items = (
                recent_result.data
                if recent_result.success and recent_result.data
                else []
            )

            # --- 3. Deduplicate: remove any recent items already in preferences ---
            deduped_recent = [
                item for item in recent_items if item["id"] not in pref_ids
            ]

            # --- 4. Combine: all preferences first, then up to limit recent ---
            combined = pref_items + deduped_recent[:limit]

            return Result(success=True, data=combined)

        except Exception as e:
            self.logger.error("get_recent failed: %s", e)
            return Result(success=False, reason=f"get_recent error: {str(e)}")

    def update_memory(
        self,
        memory_id: str,
        title: str = None,
        content: str = None,
        tags: List[str] = None,
        importance: int = None,
        memory_type: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Result:
        """
        Update or modify an existing memory by its unique ID.
        Also updates updated_at and last_accessed (treating edits as an access).
        """
        try:
            # Get existing record
            cursor = self.sqlite_conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()

            if not row:
                return Result(success=False, reason="Memory not found")

            # Prepare updates
            updates = []
            params = []

            if title is not None:
                updates.append("title = ?")
                params.append(title)

            if content is not None:
                updates.append("content = ?")
                params.append(content)
                updates.append("content_hash = ?")
                params.append(self._content_hash(content))
                # Recalculate token count
                updates.append("token_count = ?")
                params.append(self._count_tokens(content))

            if tags is not None:
                updates.append("tags = ?")
                params.append(json.dumps(tags))

            if importance is not None:
                importance = max(1, min(10, importance))
                updates.append("importance = ?")
                params.append(importance)

            if memory_type is not None:
                updates.append("memory_type = ?")
                params.append(memory_type)

            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))

            now_iso = datetime.now(timezone.utc).isoformat()
            updates.append("updated_at = ?")
            params.append(now_iso)
            updates.append("last_accessed = ?")
            params.append(now_iso)

            params.append(memory_id)

            # Update SQLite
            update_query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
            self.sqlite_conn.execute(update_query, params)

            # Update ChromaDB if content changed
            if content is not None or title is not None:
                # Get updated record
                cursor = self.sqlite_conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                updated_row = cursor.fetchone()

                # Re-generate embedding
                text_for_embedding = f"{updated_row['title']}\n{updated_row['content']}"
                embedding = self.embedding_model.encode(text_for_embedding).tolist()

                # Update ChromaDB
                self.chroma_collection.update(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[text_for_embedding],
                    metadatas=[
                        {
                            "title": updated_row["title"],
                            "timestamp": updated_row["timestamp"],
                            "importance": updated_row["importance"],
                            "memory_type": updated_row["memory_type"],
                            "tags": updated_row["tags"],
                        }
                    ],
                )

                try:
                    if hasattr(self.chroma_client, "persist"):
                        self.chroma_client.persist()
                except Exception as pe:
                    self.logger.warning("Chroma persist warning: %s", pe)

            self.sqlite_conn.commit()

            self.logger.info("Memory updated successfully: %s", memory_id)
            return Result(success=True, data=[{"id": memory_id, "updated": True}])

        except Exception as e:
            self.logger.error("Failed to update memory: %s", e)
            self.sqlite_conn.rollback()
            return Result(success=False, reason=f"Update error: {str(e)}")

    def delete_memory(self, memory_id: str) -> Result:
        """
        Delete a memory from both systems
        """
        try:
            # Check if exists
            cursor = self.sqlite_conn.execute(
                "SELECT title, content FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()

            if not row:
                return Result(success=False, reason="Memory not found")

            # Delete from SQLite
            self.sqlite_conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

            # Delete from ChromaDB
            self.chroma_collection.delete(ids=[memory_id])

            self.sqlite_conn.commit()

            self.logger.info("Memory deleted successfully: %s", memory_id)
            return Result(
                success=True,
                data=[{"id": memory_id, "title": row["title"], "deleted": True}],
            )

        except Exception as e:
            self.logger.error("Failed to delete memory: %s", e)
            self.sqlite_conn.rollback()
            return Result(success=False, reason=f"Delete error: {str(e)}")

    def get_statistics(self) -> Result:
        """Get memory system statistics"""
        try:
            cursor = self.sqlite_conn.execute(
                """
                SELECT 
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT memory_type) as memory_types,
                    AVG(importance) as avg_importance,
                    MIN(timestamp) as oldest_memory,
                    MAX(timestamp) as newest_memory
                FROM memories
            """
            )
            stats = cursor.fetchone()

            # Get memory type breakdown
            cursor = self.sqlite_conn.execute(
                """
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
                ORDER BY count DESC
            """
            )
            type_breakdown = {
                row["memory_type"]: row["count"] for row in cursor.fetchall()
            }

            # Get database sizes
            sqlite_size = (
                self.sqlite_path.stat().st_size if self.sqlite_path.exists() else 0
            )
            chroma_size = sum(
                f.stat().st_size
                for f in (self.db_folder / "chroma_db").rglob("*")
                if f.is_file()
            )

            result_data = {
                "total_memories": stats["total_memories"],
                "memory_types": stats["memory_types"],
                "avg_importance": round(stats["avg_importance"] or 0, 2),
                "oldest_memory": stats["oldest_memory"],
                "newest_memory": stats["newest_memory"],
                "type_breakdown": type_breakdown,
                "storage_size_mb": round((sqlite_size + chroma_size) / 1024 / 1024, 2),
                "sqlite_size_mb": round(sqlite_size / 1024 / 1024, 2),
                "chroma_size_mb": round(chroma_size / 1024 / 1024, 2),
            }

            return Result(success=True, data=[result_data])

        except Exception as e:
            self.logger.error("Failed to get statistics: %s", e)
            return Result(success=False, reason=f"Statistics error: {str(e)}")

    def _maybe_backup(self):
        """Trigger backup if conditions are met"""
        try:
            # Get last backup time (string)
            cursor = self.sqlite_conn.execute(
                "SELECT value FROM memory_stats WHERE key = 'last_backup'"
            )
            row = cursor.fetchone()

            last_backup = None
            if row and row["value"]:
                raw = row["value"]
                try:
                    # Handle common ISO formats and 'Z'
                    val = raw.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(val)
                except Exception:
                    dt = None

                if dt is not None:
                    # Normalize to timezone-aware (assume UTC if naive)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    last_backup = dt

            # If we couldn't parse, default far in the past to force a backup later
            if last_backup is None:
                last_backup = datetime(1970, 1, 1, tzinfo=timezone.utc)

            hours_since_backup = (
                datetime.now(timezone.utc) - last_backup
            ).total_seconds() / 3600.0

            # Backup every 24 hours or every 100 new memories
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]

            if hours_since_backup > 24 or (
                total_memories > 0 and total_memories % 100 == 0
            ):
                self.create_backup()

        except Exception as e:
            self.logger.error("Backup check failed: %s", e)

    def create_backup(self) -> Result:
        """Create a complete backup of the memory system"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"memory_backup_{timestamp}"
            backup_path = self.backup_folder / backup_name
            backup_path.mkdir(exist_ok=True)

            # ===== CHECKPOINT BEFORE BACKUP =====
            # This merges the WAL into the main .db file and truncates the WAL
            self.logger.warning("Starting backup: checkpointing WAL...")
            try:
                self.sqlite_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self.sqlite_conn.commit()
                self.logger.warning("WAL checkpoint completed")
            except Exception as e:
                self.logger.error("Checkpoint failed: %s", e)
                # Continue anyway; we'll copy all files as fallback

            # Backup SQLite database + WAL files
            # Copy main .db
            sqlite_backup = backup_path / "memories.db"
            shutil.copy2(self.sqlite_path, sqlite_backup)

            # Copy WAL and SHM if they exist (belt-and-suspenders)
            wal_path = Path(str(self.sqlite_path) + "-wal")
            shm_path = Path(str(self.sqlite_path) + "-shm")

            if wal_path.exists():
                shutil.copy2(wal_path, backup_path / "memories.db-wal")
                self.logger.info("Copied WAL file to backup")

            if shm_path.exists():
                shutil.copy2(shm_path, backup_path / "memories.db-shm")
                self.logger.info("Copied SHM file to backup")

            # Backup ChromaDB
            chroma_backup = backup_path / "chroma_db"
            if (self.db_folder / "chroma_db").exists():
                shutil.copytree(self.db_folder / "chroma_db", chroma_backup)

            # Export to JSON for portability
            cursor = self.sqlite_conn.execute(
                "SELECT * FROM memories ORDER BY timestamp"
            )
            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                memory_dict["tags"] = json.loads(memory_dict["tags"])
                memory_dict["metadata"] = json.loads(memory_dict["metadata"])
                memories.append(memory_dict)

            json_backup = backup_path / "memories_export.json"
            with open(json_backup, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "export_timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_memories": len(memories),
                        "memories": memories,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # Update backup timestamp
            now_iso = datetime.now(timezone.utc).isoformat()
            self.sqlite_conn.execute(
                "UPDATE memory_stats SET value = ?, updated_at = ? WHERE key = 'last_backup'",
                (now_iso, now_iso),
            )
            self.sqlite_conn.commit()

            # Clean old backups (keep last 10)
            backups = sorted(
                [d for d in self.backup_folder.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backups[10:]:
                shutil.rmtree(old_backup)

            self.logger.warning(
                "Backup created successfully: %s (memories: %d)",
                backup_name,
                len(memories),
            )
            return Result(
                success=True,
                data=[
                    {
                        "backup_name": backup_name,
                        "backup_path": str(backup_path),
                        "memories_backed_up": len(memories),
                    }
                ],
            )

        except Exception as e:
            self.logger.error("Backup failed: %s", e)
            return Result(success=False, reason=f"Backup error: {str(e)}")

    def close(self):
        """Clean shutdown of the memory system"""
        try:
            if hasattr(self, "sqlite_conn") and self.sqlite_conn:
                try:
                    self.sqlite_conn.commit()
                except Exception:
                    pass

                # WAL checkpoint: merge WAL into main db file before closing.
                # TRUNCATE mode resets the WAL to zero bytes so no stale WAL
                # is left behind (avoids data-in-WAL-but-not-in-db on restart).
                try:
                    self.sqlite_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                except Exception as e:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.warning("WAL checkpoint on close failed: %s", e)

                try:
                    self.sqlite_conn.close()
                except Exception:
                    pass
        except Exception:
            pass

        # ChromaDB client doesn't need explicit close; make GC-friendly
        try:
            if hasattr(self, "chroma_client"):
                self.chroma_client = None
        except Exception:
            pass

        # Optional: free embedding model reference
        try:
            if hasattr(self, "embedding_model"):
                self.embedding_model = None
        except Exception:
            pass

        try:
            if hasattr(self, "logger"):
                self.logger.info("Memory system closed successfully")
        except Exception:
            pass

    def _debug_vector_index(self, sample: int = 5):
        """Debug helper to inspect vector index contents"""
        try:
            count = self.chroma_collection.count()
            data = self.chroma_collection.get(
                include=["documents", "metadatas"], limit=sample
            )
            return {
                "count": count,
                "ids": data.get("ids", []),
                "have_documents": bool(data.get("documents")),
                "have_metadatas": bool(data.get("metadatas")),
            }
        except Exception as e:
            return {"error": str(e)}

    def rebuild_vector_index(self, batch_size: int = 128) -> Result:
        """
        Rebuilds the ChromaDB vector index from all SQLite memories.

        Clears the existing vector collection and re-embeds all memories
        from the SQLite database in batches to avoid memory issues.

        Args:
            batch_size (int, optional): Number of memories to process per
                batch. Defaults to 128.

        Returns:
            Result: Dictionary with the following keys:
                - success (bool): Whether the rebuild succeeded.
                - reason (str, optional): Error message if rebuild failed.
                - data (list, optional): Contains reindexed status and
                  total count of memories indexed.
        """
        try:
            # Wipe collection by deleting and recreating it.
            # Note: delete(where={}) is unreliable in some ChromaDB versions
            # and can leave orphaned vectors, so we drop the whole collection.
            try:
                self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            except Exception as e:
                self.logger.warning("Chroma drop collection warning: %s", e)

            # Invalidate the stale reference so that a failure in
            # get_or_create_collection leaves the system in a clearly
            # broken state rather than pointing at a deleted collection.
            self.chroma_collection = None

            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={
                    "description": "Long-term memory for AI companion",
                    "hnsw:space": "cosine",
                },
                embedding_function=None,
            )

            rows = self.sqlite_conn.execute(
                "SELECT id, title, content, timestamp, importance, memory_type, "
                "tags FROM memories ORDER BY timestamp ASC"
            ).fetchall()

            ids, embs, docs, metas = [], [], [], []
            for row in rows:
                text = f"{row['title']}\n{row['content']}"
                emb = self.embedding_model.encode(text).tolist()
                ids.append(row["id"])
                embs.append(emb)
                docs.append(text)
                metas.append(
                    {
                        "title": row["title"],
                        "timestamp": row["timestamp"],
                        "importance": row["importance"],
                        "memory_type": row["memory_type"],
                        "tags": row["tags"],
                    }
                )

                if len(ids) >= batch_size:
                    self.chroma_collection.add(
                        ids=ids, embeddings=embs, documents=docs, metadatas=metas
                    )
                    ids, embs, docs, metas = [], [], [], []

            if ids:
                self.chroma_collection.add(
                    ids=ids, embeddings=embs, documents=docs, metadatas=metas
                )

            try:
                if hasattr(self.chroma_client, "persist"):
                    self.chroma_client.persist()
            except Exception as pe:
                self.logger.warning("Chroma persist warning: %s", pe)

            return Result(
                success=True,
                data=[{"reindexed": True, "count": self.chroma_collection.count()}],
            )
        except Exception as e:
            self.logger.error("Reindex failed: %s", e)
            return Result(success=False, reason=str(e))

    def _parse_iso(self, iso_str: str) -> datetime:
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _days_since(self, iso_str: Optional[str]) -> float:
        if not iso_str:
            return 0.0
        try:
            then = self._parse_iso(iso_str)
            return max(
                0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0
            )
        except Exception:
            return 0.0

    def _get_half_life_days(self, memory_type: Optional[str]) -> float:
        return DECAY_HALF_LIFE_DAYS_BY_TYPE.get(
            memory_type or "", DECAY_HALF_LIFE_DAYS_DEFAULT
        )

    def _get_floor(self, memory_type: Optional[str]) -> int:
        return DECAY_MIN_IMPORTANCE_BY_TYPE.get(
            memory_type or "", DECAY_MIN_IMPORTANCE_DEFAULT
        )

    def _should_protect(self, tags_field) -> bool:
        try:
            tags = json.loads(tags_field) if isinstance(tags_field, str) else tags_field
            tags = tags or []
            return any(t in DECAY_PROTECT_TAGS for t in tags)
        except Exception:
            return False

    def _compute_decay_importance(
        self, importance: float, days_idle: float, half_life_days: float
    ) -> float:
        if half_life_days <= 0:
            return importance
        factor = 0.5 ** (days_idle / half_life_days)
        return importance * factor

    def _round_to_half(self, value: float) -> float:
        return round(value * 2.0) / 2.0

    def _maybe_decay(self, row) -> Optional[float]:
        """
        Lazily decays importance for a single row if conditions are met.
        Returns the new importance if updated, else None.
        Safe: never drops below type floor; respects protected tags.
        """
        if not DECAY_ENABLED:
            return None

        try:
            mem_id = row["id"]
            mem_type = row["memory_type"]
            importance = float(row["importance"])
            floor = self._get_floor(mem_type)

            # Skip if protected or already at/below floor
            if self._should_protect(row["tags"]) or importance <= floor:
                self.logger.info(
                    "Decay check: id=%s type=%s skipped (protected or floor reached)",
                    mem_id,
                    mem_type,
                )
                return None

            # Anchor idle time on last_accessed; fallback to timestamp
            last_accessed = (
                row["last_accessed"] if "last_accessed" in row.keys() else None
            )
            if not last_accessed:
                last_accessed = row["timestamp"]

            days_idle = self._days_since(last_accessed)
            half_life = self._get_half_life_days(mem_type)

            decayed = self._compute_decay_importance(importance, days_idle, half_life)
            decayed = max(floor, self._round_to_half(decayed))

            # If no meaningful change, just log and bail
            change = importance - decayed
            if change < DECAY_WRITEBACK_STEP:
                self.logger.info(
                    "Decay check: id=%s type=%s old=%s -> new=%s "
                    "(idle=%.1fd, half_life=%sd) [no decay applied]",
                    mem_id,
                    mem_type,
                    importance,
                    decayed,
                    days_idle,
                    half_life,
                )
                return None

            # Rate limit writes
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                meta = {}

            last_decay_at = meta.get("last_decay_at")
            hours_since_last = (
                self._days_since(last_decay_at) * 24 if last_decay_at else 1e9
            )
            if hours_since_last < DECAY_MIN_INTERVAL_HOURS:
                self.logger.info(
                    "Decay check: id=%s type=%s old=%s → would become %s "
                    "(idle=%.1fd), but last decay %.1fh ago [rate‑limited]",
                    mem_id,
                    mem_type,
                    importance,
                    decayed,
                    days_idle,
                    hours_since_last,
                )
                return None

            # Persist decay to DB (caller is responsible for commit)
            meta["last_decay_at"] = datetime.now(timezone.utc).isoformat()
            self.sqlite_conn.execute(
                "UPDATE memories SET importance = ?, metadata = ? WHERE id = ?",
                (decayed, json.dumps(meta), mem_id),
            )

            self.logger.info(
                "Lazy decay: id=%s type=%s old=%s new=%s idle_days=%.1f half_life=%s",
                mem_id,
                mem_type,
                importance,
                decayed,
                days_idle,
                half_life,
            )
            return decayed

        except Exception as e:
            self.logger.warning(
                "Lazy decay skipped for id=%s: %s",
                row.get("id", "UNKNOWN"),
                e,
            )
            return None

    def _maybe_reinforce(self, row) -> Optional[float]:
        """
        Apply reinforcement bump on access.

        Returns:
            float | None: New importance if updated (writeback occurred),
            otherwise None.
        """
        if not REINFORCEMENT_ENABLED:
            return None

        try:
            mem_id = row["id"]
            mem_type = row["memory_type"]
            importance = float(row["importance"])

            # Load metadata safely
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                meta = {}

            accum = meta.get("reinforcement_accum", 0.0) + REINFORCEMENT_STEP

            # If accumulated boost reaches threshold, persist
            if accum >= REINFORCEMENT_WRITEBACK_STEP:
                new_importance = min(
                    REINFORCEMENT_MAX, self._round_to_half(importance + accum)
                )
                meta["reinforcement_accum"] = 0.0  # reset accumulator

                self.sqlite_conn.execute(
                    "UPDATE memories SET importance = ?, metadata = ? WHERE id = ?",
                    (new_importance, json.dumps(meta), mem_id),
                )
                # Caller is responsible for commit (batched)

                self.logger.info(
                    "Reinforcement: id=%s type=%s old=%s new=%s (+%s)",
                    mem_id,
                    mem_type,
                    importance,
                    new_importance,
                    accum,
                )
                return new_importance

            # No writeback yet: just save accumulator (caller commits)
            meta["reinforcement_accum"] = accum
            self.sqlite_conn.execute(
                "UPDATE memories SET metadata = ? WHERE id = ?",
                (json.dumps(meta), mem_id),
            )

            self.logger.info(
                "Reinforcement accum: id=%s +%s, total=%.2f (not written yet)",
                mem_id,
                REINFORCEMENT_STEP,
                accum,
            )
            return None

        except Exception as e:
            self.logger.warning(
                "Reinforcement skipped for id=%s: %s", row.get("id", "UNKNOWN"), e
            )
            return None

    def list_source_memories(self, source_db_path: str, limit: int = 100) -> Result:
        """
        List memories from a source database for migration preview.

        Args:
            source_db_path: Path to the source SQLite database file
            limit: Maximum number of memories to list

        Returns:
            Result with list of memories from source database
        """
        try:
            source_path = Path(source_db_path)
            if not source_path.exists():
                return Result(
                    success=False, reason=f"Source database not found: {source_db_path}"
                )

            # Connect to source database
            source_conn = sqlite3.connect(str(source_path), check_same_thread=False)
            source_conn.row_factory = sqlite3.Row

            # Query memories
            cursor = source_conn.execute(
                """
                SELECT id, title, content, timestamp, tags, importance, 
                       memory_type, metadata, token_count, created_at
                FROM memories 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (limit,),
            )

            memories = []
            for row in cursor.fetchall():
                memory_dict = {
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"][:200] + "..."
                    if len(row["content"]) > 200
                    else row["content"],
                    "timestamp": row["timestamp"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "importance": row["importance"],
                    "memory_type": row["memory_type"],
                    "token_count": row["token_count"] or 0,
                }
                memories.append(memory_dict)

            source_conn.close()

            self.logger.info(f"Listed {len(memories)} memories from source database")
            return Result(success=True, data=memories)

        except Exception as e:
            self.logger.error(f"Failed to list source memories: {e}")
            return Result(
                success=False, reason=f"Error listing source memories: {str(e)}"
            )

    def migrate_memories(
        self,
        source_db_path: str,
        source_chroma_path: str = None,
        memory_ids: List[str] = None,
        skip_duplicates: bool = True,
    ) -> Result:
        """
        Migrate memories from a source database to the active database.
        Includes both SQLite records and ChromaDB vectors.

        Args:
            source_db_path: Path to the source SQLite database file
            source_chroma_path: Path to the source ChromaDB directory (optional, auto-detected if not provided)
            memory_ids: List of specific memory IDs to migrate (None = migrate all)
            skip_duplicates: If True, skip memories with duplicate content hashes

        Returns:
            Result with migration statistics
        """
        try:
            source_db_path = Path(source_db_path)
            if not source_db_path.exists():
                return Result(
                    success=False, reason=f"Source database not found: {source_db_path}"
                )

            # Auto-detect ChromaDB path if not provided
            if source_chroma_path is None:
                # Try to find chroma_db in the same directory as the SQLite DB
                source_chroma_path = source_db_path.parent / "chroma_db"
                if not source_chroma_path.exists():
                    self.logger.warning(
                        f"ChromaDB not found at {source_chroma_path}, vectors will not be migrated"
                    )
                    source_chroma_path = None
            else:
                source_chroma_path = Path(source_chroma_path)
                if not source_chroma_path.exists():
                    self.logger.warning(
                        f"ChromaDB path does not exist: {source_chroma_path}"
                    )
                    source_chroma_path = None

            # Connect to source database
            source_conn = sqlite3.connect(str(source_db_path), check_same_thread=False)
            source_conn.row_factory = sqlite3.Row

            # Build query for memories
            if memory_ids:
                placeholders = ",".join("?" * len(memory_ids))
                query = f"""
                    SELECT * FROM memories 
                    WHERE id IN ({placeholders})
                    ORDER BY timestamp ASC
                """
                cursor = source_conn.execute(query, memory_ids)
            else:
                cursor = source_conn.execute(
                    "SELECT * FROM memories ORDER BY timestamp ASC"
                )

            source_memories = cursor.fetchall()

            # Connect to source ChromaDB if available
            source_chroma_client = None
            source_chroma_collection = None
            if source_chroma_path:
                try:
                    source_chroma_client = chromadb.PersistentClient(
                        path=str(source_chroma_path),
                        settings=Settings(
                            anonymized_telemetry=False, allow_reset=False
                        ),
                    )
                    source_chroma_collection = source_chroma_client.get_collection(
                        CHROMA_COLLECTION_NAME
                    )
                    self.logger.info(
                        f"Connected to source ChromaDB at {source_chroma_path}"
                    )
                except Exception as e:
                    self.logger.warning(f"Could not connect to source ChromaDB: {e}")
                    source_chroma_collection = None

            # Migration statistics
            stats = {
                "total_found": len(source_memories),
                "migrated": 0,
                "skipped_duplicates": 0,
                "errors": 0,
                "vectors_migrated": 0,
            }

            # Migrate each memory
            for row in source_memories:
                try:
                    memory_id = row["id"]
                    content_hash = row["content_hash"]

                    # Check for duplicates if requested
                    if skip_duplicates and content_hash:
                        existing = self.sqlite_conn.execute(
                            "SELECT id FROM memories WHERE content_hash = ?",
                            (content_hash,),
                        ).fetchone()

                        if existing:
                            self.logger.info(f"Skipping duplicate memory: {memory_id}")
                            stats["skipped_duplicates"] += 1
                            continue

                    # Insert into SQLite
                    self.sqlite_conn.execute(
                        """
                        INSERT INTO memories 
                        (id, title, content, timestamp, tags, importance, memory_type, 
                         metadata, content_hash, created_at, updated_at, last_accessed, token_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            row["created_at"]
                            if "created_at" in row.keys()
                            else row["timestamp"],
                            row["updated_at"]
                            if "updated_at" in row.keys()
                            else row["timestamp"],
                            row["last_accessed"]
                            if "last_accessed" in row.keys()
                            else row["timestamp"],
                            row["token_count"] if "token_count" in row.keys() else 0,
                        ),
                    )

                    # Migrate vector from ChromaDB if available
                    if source_chroma_collection:
                        try:
                            # Get the vector from source ChromaDB
                            result = source_chroma_collection.get(
                                ids=[memory_id],
                                include=["embeddings", "documents", "metadatas"],
                            )

                            if result["ids"] and len(result["ids"]) > 0:
                                # Add to destination ChromaDB
                                self.chroma_collection.add(
                                    ids=result["ids"],
                                    embeddings=result["embeddings"],
                                    documents=result["documents"],
                                    metadatas=result["metadatas"],
                                )
                                stats["vectors_migrated"] += 1
                                self.logger.info(
                                    f"Migrated vector for memory: {memory_id}"
                                )
                            else:
                                self.logger.warning(
                                    f"No vector found for memory: {memory_id}"
                                )
                        except Exception as ve:
                            self.logger.warning(
                                f"Failed to migrate vector for {memory_id}: {ve}"
                            )

                    stats["migrated"] += 1
                    self.logger.info(f"Migrated memory: {memory_id} - {row['title']}")

                except Exception as e:
                    self.logger.error(f"Failed to migrate memory {row['id']}: {e}")
                    stats["errors"] += 1
                    continue

            # Commit all changes
            self.sqlite_conn.commit()

            # Close source connection
            source_conn.close()

            self.logger.info(
                f"Migration complete: {stats['migrated']} migrated, "
                f"{stats['skipped_duplicates']} skipped, {stats['errors']} errors, "
                f"{stats['vectors_migrated']} vectors migrated"
            )

            return Result(success=True, data=[stats])

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return Result(success=False, reason=f"Migration error: {str(e)}")
