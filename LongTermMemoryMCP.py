"""\
Robust Long-Term Memory System for AI Companions\
Hybrid approach: ChromaDB (vector) + SQLite (structured) + File backup

Features:

* Semantic search via embeddings

* Structured metadata queries

* Cross-platform compatibility (Windows/Ubuntu/macOS)

* Automatic backups and data integrity

* Migration-friendly exports

* Scalable to decades of conversations\
  """

from fastmcp import FastMCP
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Union
import tempfile
import shutil
import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
import logging
import asyncio
import sys
import atexit

# Third-party imports (will be installed)

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing required packages. Install with: pip install chromadb sentence-transformers")
    print(f"Error: {e}")
    sys.exit(1)

# Configuration  
# You can override the data folder by setting the AI_COMPANION_DATA_DIR environment variable.  
# Example (PowerShell): $env:AI_COMPANION_DATA_DIR = "D:\a.i. apps\long_term_memory_mcp\data"  
DATA_FOLDER = Path(os.environ.get("AI_COMPANION_DATA_DIR", str(Path.home() / ".ai_companion_memory")))  
  
@dataclass  
class MemoryRecord:  
    """Structured memory record"""  
    id: str  
    title: str  
    content: str  
    timestamp: datetime  
    tags: List[str]  
    importance: int  # 1-10 scale  
    memory_type: str  # conversation, fact, preference, event, etc.  
    metadata: Dict[str, Any]  
  
@dataclass  
class SearchResult:  
    """Search result with relevance score"""  
    record: MemoryRecord  
    relevance_score: float  
    match_type: str  # semantic, exact, metadata  
  
@dataclass  
class Result:  
    success: bool  
    reason: Optional[str] = None  
    data: Optional[List[Dict]] = None

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
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._init_sqlite()
        self._init_chromadb()
        self._init_embeddings()
        
        # Perform integrity check on startup
        self._integrity_check()

    def _setup_logging(self):
        """Setup logging for debugging and monitoring"""
        log_file = self.data_folder / "memory_system.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _init_sqlite(self):  
        """Initialize SQLite database for structured data"""  
        try:  
            self.sqlite_conn = sqlite3.connect(  
                str(self.sqlite_path),   
                check_same_thread=False,  
                timeout=30.0  
            )  
            self.sqlite_conn.row_factory = sqlite3.Row  
              
            # Create tables  
            self.sqlite_conn.executescript("""  
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
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP  
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
                  
                -- Store system metadata  
                INSERT OR REPLACE INTO memory_stats (key, value)   
                VALUES ('schema_version', '1.0');  
            """)  
              
            # Normalize last_backup to an ISO UTC string (avoid sqlite datetime('now') naive value)  
            now_iso = datetime.now(timezone.utc).isoformat()  
            self.sqlite_conn.execute(  
                "INSERT OR REPLACE INTO memory_stats (key, value, updated_at) VALUES ('last_backup', ?, ?)",  
                (now_iso, now_iso)  
            )  
              
            self.sqlite_conn.commit()  
            self.logger.info("SQLite database initialized successfully")  
              
        except Exception as e:  
            self.logger.error(f"Failed to initialize SQLite: {e}")  
            raise

    def _init_chromadb(self):
        """Initialize ChromaDB for vector storage"""
        try:
            # Use persistent storage
            chroma_path = str(self.db_folder / "chroma_db")
            
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="ai_companion_memories",
                metadata={"description": "Long-term memory for AI companion"}
            )
            
            self.logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _init_embeddings(self):
        """Initialize sentence transformer for embeddings"""
        try:
            # Use a good general-purpose model that works offline
            model_name = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Embedding model '{model_name}' loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a simpler approach if needed
            raise

    def _generate_id(self, content: str, timestamp: datetime) -> str:
        """Generate unique ID for memory record"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        time_hash = hashlib.sha256(timestamp.isoformat().encode()).hexdigest()[:8]
        return f"mem_{time_hash}_{content_hash}"

    def _content_hash(self, content: str) -> str:
        """Generate content hash for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _integrity_check(self):
        """Check data integrity between SQLite and ChromaDB"""
        try:
            # Count records in both systems
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM memories")
            sqlite_count = cursor.fetchone()[0]
            
            chroma_count = self.chroma_collection.count()
            
            self.logger.info(f"Integrity check: SQLite={sqlite_count}, ChromaDB={chroma_count}")
            
            if sqlite_count != chroma_count:
                self.logger.warning("Record count mismatch between SQLite and ChromaDB")
                # Could implement auto-repair here
            
        except Exception as e:
            self.logger.error(f"Integrity check failed: {e}")

    def remember(self, title: str, content: str, tags: List[str] = None, 
                importance: int = 5, memory_type: str = "conversation", 
                metadata: Dict[str, Any] = None) -> Result:
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
            
            # Check for duplicates
            cursor = self.sqlite_conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", 
                (content_hash,)
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
                metadata=metadata
            )
            
            # Store in SQLite
            self.sqlite_conn.execute("""
                INSERT INTO memories 
                (id, title, content, timestamp, tags, importance, memory_type, metadata, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.title,
                record.content,
                record.timestamp.isoformat(),
                json.dumps(record.tags),
                record.importance,
                record.memory_type,
                json.dumps(record.metadata),
                content_hash
            ))
            
            # Generate embedding and store in ChromaDB
            # Combine title and content for better semantic search
            text_for_embedding = f"{title}\n{content}"
            embedding = self.embedding_model.encode(text_for_embedding).tolist()
            
            self.chroma_collection.add(
                ids=[record.id],
                embeddings=[embedding],
                documents=[text_for_embedding],
                metadatas=[{
                    "title": title,
                    "timestamp": record.timestamp.isoformat(),
                    "importance": importance,
                    "memory_type": memory_type,
                    "tags": json.dumps(tags)
                }]
            )
            
            self.sqlite_conn.commit()
            
            # Trigger backup if needed
            self._maybe_backup()
            
            self.logger.info(f"Memory stored successfully: {memory_id}")
            rec = asdict(record)
            rec["timestamp"] = record.timestamp.isoformat()
            return Result(success=True, data=[rec])
            
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            self.sqlite_conn.rollback()
            return Result(success=False, reason=f"Storage error: {str(e)}")

    def search_semantic(self, query: str, limit: int = 10, 
                       min_relevance: float = 0.3) -> Result:
        """
        Semantic search using vector similarity
        """
        try:
            if not query.strip():
                return Result(success=False, reason="Query cannot be empty")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['ids'][0]:
                return Result(success=True, data=[])
            
            # Convert to SearchResult objects
            search_results = []
            for i, memory_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                relevance = 1.0 - distance  # Convert distance to similarity
                
                if relevance < min_relevance:
                    continue
                
                # Get full record from SQLite
                cursor = self.sqlite_conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    record = MemoryRecord(
                        id=row['id'],
                        title=row['title'],
                        content=row['content'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        tags=json.loads(row['tags']),
                        importance=row['importance'],
                        memory_type=row['memory_type'],
                        metadata=json.loads(row['metadata'])
                    )
                    
                    search_results.append(SearchResult(
                        record=record,
                        relevance_score=relevance,
                        match_type="semantic"
                    ))
            
            # Sort by relevance
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Convert to dict format
            result_data = []
            for sr in search_results:  
                result_dict = asdict(sr.record)  
                result_dict['timestamp'] = sr.record.timestamp.isoformat()  
                result_dict['relevance_score'] = sr.relevance_score  
                result_dict['match_type'] = sr.match_type  
                result_data.append(result_dict)
            
            self.logger.info(f"Semantic search returned {len(result_data)} results")
            return Result(success=True, data=result_data)
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return Result(success=False, reason=f"Search error: {str(e)}")

    def search_structured(self, memory_type: str = None, tags: List[str] = None,
                         importance_min: int = None, date_from: str = None,
                         date_to: str = None, limit: int = 50) -> Result:
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
            
            # Convert to MemoryRecord objects
            results = []
            for row in rows:
                record = MemoryRecord(
                    id=row['id'],
                    title=row['title'],
                    content=row['content'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    tags=json.loads(row['tags']),
                    importance=row['importance'],
                    memory_type=row['memory_type'],
                    metadata=json.loads(row['metadata'])
                )
                
                result_dict = asdict(record)  
                result_dict['timestamp'] = record.timestamp.isoformat()  # ← ADD THIS LINE  
                result_dict['match_type'] = 'structured'  
                results.append(result_dict)
            
            self.logger.info(f"Structured search returned {len(results)} results")
            return Result(success=True, data=results)
            
        except Exception as e:
            self.logger.error(f"Structured search failed: {e}")
            return Result(success=False, reason=f"Search error: {str(e)}")

    def get_recent(self, limit: int = 20) -> Result:
        """Get most recent memories"""
        return self.search_structured(limit=limit)

    def update_memory(self, memory_id: str, title: str = None, content: str = None,
                     tags: List[str] = None, importance: int = None,
                     metadata: Dict[str, Any] = None) -> Result:
        """
        Update an existing memory
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
            
            if tags is not None:
                updates.append("tags = ?")
                params.append(json.dumps(tags))
            
            if importance is not None:
                importance = max(1, min(10, importance))
                updates.append("importance = ?")
                params.append(importance)
            
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            updates.append("updated_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
            
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
                    metadatas=[{
                        "title": updated_row['title'],
                        "timestamp": updated_row['timestamp'],
                        "importance": updated_row['importance'],
                        "memory_type": updated_row['memory_type'],
                        "tags": updated_row['tags']
                    }]
                )
            
            self.sqlite_conn.commit()
            
            self.logger.info(f"Memory updated successfully: {memory_id}")
            return Result(success=True, data=[{"id": memory_id, "updated": True}])
            
        except Exception as e:
            self.logger.error(f"Failed to update memory: {e}")
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
            
            self.logger.info(f"Memory deleted successfully: {memory_id}")
            return Result(success=True, data=[{
                "id": memory_id, 
                "title": row['title'],
                "deleted": True
            }])
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            self.sqlite_conn.rollback()
            return Result(success=False, reason=f"Delete error: {str(e)}")

    def get_statistics(self) -> Result:
        """Get memory system statistics"""
        try:
            cursor = self.sqlite_conn.execute("""
                SELECT 
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT memory_type) as memory_types,
                    AVG(importance) as avg_importance,
                    MIN(timestamp) as oldest_memory,
                    MAX(timestamp) as newest_memory
                FROM memories
            """)
            stats = cursor.fetchone()
            
            # Get memory type breakdown
            cursor = self.sqlite_conn.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
                ORDER BY count DESC
            """)
            type_breakdown = {row['memory_type']: row['count'] for row in cursor.fetchall()}
            
            # Get database sizes
            sqlite_size = self.sqlite_path.stat().st_size if self.sqlite_path.exists() else 0
            chroma_size = sum(f.stat().st_size for f in (self.db_folder / "chroma_db").rglob("*") if f.is_file())
            
            result_data = {
                "total_memories": stats['total_memories'],
                "memory_types": stats['memory_types'],
                "avg_importance": round(stats['avg_importance'] or 0, 2),
                "oldest_memory": stats['oldest_memory'],
                "newest_memory": stats['newest_memory'],
                "type_breakdown": type_breakdown,
                "storage_size_mb": round((sqlite_size + chroma_size) / 1024 / 1024, 2),
                "sqlite_size_mb": round(sqlite_size / 1024 / 1024, 2),
                "chroma_size_mb": round(chroma_size / 1024 / 1024, 2)
            }
            
            return Result(success=True, data=[result_data])
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
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
            if row and row['value']:  
                raw = row['value']  
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
      
            hours_since_backup = (datetime.now(timezone.utc) - last_backup).total_seconds() / 3600.0  
      
            # Backup every 24 hours or every 100 new memories  
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM memories")  
            total_memories = cursor.fetchone()[0]  
      
            if hours_since_backup > 24 or (total_memories > 0 and total_memories % 100 == 0):  
                self.create_backup()  
      
        except Exception as e:  
            self.logger.error(f"Backup check failed: {e}")

    def create_backup(self) -> Result:
        """Create a complete backup of the memory system"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"memory_backup_{timestamp}"
            backup_path = self.backup_folder / backup_name
            backup_path.mkdir(exist_ok=True)
            
            # Backup SQLite database
            sqlite_backup = backup_path / "memories.db"
            shutil.copy2(self.sqlite_path, sqlite_backup)
            
            # Backup ChromaDB
            chroma_backup = backup_path / "chroma_db"
            if (self.db_folder / "chroma_db").exists():
                shutil.copytree(self.db_folder / "chroma_db", chroma_backup)
            
            # Export to JSON for portability
            cursor = self.sqlite_conn.execute("SELECT * FROM memories ORDER BY timestamp")
            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                memory_dict['tags'] = json.loads(memory_dict['tags'])
                memory_dict['metadata'] = json.loads(memory_dict['metadata'])
                memories.append(memory_dict)
            
            json_backup = backup_path / "memories_export.json"
            with open(json_backup, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_memories": len(memories),
                    "memories": memories
                }, f, indent=2, ensure_ascii=False)
            
            # Update backup timestamp
            self.sqlite_conn.execute(
                "UPDATE memory_stats SET value = ?, updated_at = ? WHERE key = 'last_backup'",
                (datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat())
            )
            self.sqlite_conn.commit()
            
            # Clean old backups (keep last 10)
            backups = sorted([d for d in self.backup_folder.iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
            for old_backup in backups[10:]:
                shutil.rmtree(old_backup)
            
            self.logger.info(f"Backup created successfully: {backup_name}")
            return Result(success=True, data=[{
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "memories_backed_up": len(memories)
            }])
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return Result(success=False, reason=f"Backup error: {str(e)}")

    def close(self):  
        """Clean shutdown of the memory system"""  
        try:  
            if hasattr(self, "sqlite_conn"):  
                try:  
                    self.sqlite_conn.commit()  
                except Exception:  
                    pass  
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


# Initialize the memory system  
memory_system = RobustMemorySystem()  
  
# FastMCP setup  
mcp = FastMCP("RobustMemory")  
      
def _jsonify_result(res: Result) -> dict:  
    out = {"success": res.success}  
    if res.reason is not None:  
        out["reason"] = res.reason  
    if res.data is not None:  
        data = []  
        for item in res.data:  
            # Ensure we have a plain dict to mutate safely  
            obj = dict(item)  
            # Normalize timestamp fields (top-level)  
            ts = obj.get("timestamp")  
            if isinstance(ts, datetime):  
                obj["timestamp"] = ts.isoformat()  
            # Normalize nested fields you might have added in searches  
            # (e.g., 'relevance_score', 'match_type' already JSON-safe)  
            data.append(obj)  
        out["data"] = data  
    return out  
  
@mcp.tool  
def remember(title: str, content: str, tags: str = "", importance: int = 5,  
             memory_type: str = "conversation") -> dict:  
    """  
    Store a new memory (fact, preference, event, or conversation snippet).  
  
    When to use:  
    - The user shares something to keep or says “remember this.”  
    - New personal details, preferences, events, instructions.  
  
    Args:  
    - title (str): Short title for the memory.  
    - content (str): Full text to store.  
    - tags (str, optional): Comma-separated tags, e.g., "personal, preference".  
    - importance (int, optional): 1–10 (default 5). Higher = more important.  
    - memory_type (str, optional): e.g., "conversation", "fact", "preference", "event".  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: [ {id, title, content, timestamp, tags, importance, memory_type, ...} ] }  
  
    Example triggers:  
    - “My birthday is July 4th.”  
    - “Remember that I prefer tea over coffee.”  
    - “Please save this: truck camping next weekend.”  
    """  
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []  
    res = memory_system.remember(title, content, tag_list, importance, memory_type)  
    return _jsonify_result(res) 
  
@mcp.tool  
def search_memories(query: str, search_type: str = "semantic", limit: int = 10) -> dict:  
    """  
    Search memories using natural language queries for general recall.  
  
    When to use:  
    - User asks about a specific fact, event, or detail from the past.  
    - General "what did you tell me about..." or "when is my..." queries.  
    - Default search when no specific category, tags, or dates are mentioned.  
  
    Args:  
    - query (str): Natural language search query.  
    - search_type (str, optional): "semantic" (default). Other types not fully implemented.  
    - limit (int, optional): Max results to return (default 10).  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: [ {id, title, content, timestamp, tags, relevance_score, match_type, ...} ] }  
  
    Example triggers:  
    - "When is my birthday?"  
    - "What did I tell you about my favorite color?"  
    - "Do you remember what I said about camping?"  
    """  
    if search_type == "semantic":  
        res = memory_system.search_semantic(query, limit)  
    else:  
        res = memory_system.search_structured(limit=limit)  
    return _jsonify_result(res)  
  
@mcp.tool  
def search_by_type(memory_type: str, limit: int = 20) -> dict:  
    """  
    Retrieve memories by category/type for organized recall.  
  
    When to use:  
    - User asks for a specific category of memories.  
    - Requests like "show me all my preferences" or "list my facts."  
    - When they want to see everything in a particular memory type.  
  
    Args:  
    - memory_type (str): Category to search for, e.g., "conversation", "fact", "preference", "event".  
    - limit (int, optional): Max results to return (default 20).  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: [ {id, title, content, timestamp, tags, memory_type, ...} ] }  
  
    Example triggers:  
    - "Show me all my preferences so far."  
    - "List the facts you know about me."  
    - "What events have we discussed?"  
    """  
    res = memory_system.search_structured(memory_type=memory_type, limit=limit)  
    return _jsonify_result(res) 
  
@mcp.tool  
def search_by_tags(tags: str, limit: int = 20) -> dict:  
    """  
    Find memories associated with specific tags for thematic recall.  
  
    When to use:  
    - User mentions specific tags or themes they want to find.  
    - Requests like "find everything tagged X" or "show me camping memories."  
    - When they want memories grouped by topic/theme rather than type.  
  
    Args:  
    - tags (str): Comma-separated tags to search for, e.g., "camping, truck" or "music, guitar".  
    - limit (int, optional): Max results to return (default 20).  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: [ {id, title, content, timestamp, tags, memory_type, ...} ] }  
  
    Example triggers:  
    - "Find everything tagged camping and truck."  
    - "Show me memories about music."  
    - "What do you have tagged as personal?"  
    """  
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]  
    res = memory_system.search_structured(tags=tag_list, limit=limit)  
    return _jsonify_result(res)  
  
@mcp.tool  
def get_recent_memories(limit: int = 20) -> dict:  
    """  
    Retrieve the most recently stored memories for timeline-based recall.  
  
    When to use:  
    - User asks about recent interactions or conversations.  
    - Time-based queries like "today," "last night," "recently," "yesterday."  
    - When they want to review what was discussed in the current or recent sessions.  
    - Use this instead of date ranges when no specific dates are mentioned.  
  
    Args:  
    - limit (int, optional): Max results to return (default 20).  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: [ {id, title, content, timestamp, tags, memory_type, ...} ] }  
  
    Example triggers:  
    - "What did we talk about today?"  
    - "What have we discussed recently?"  
    - "Remind me what we covered last night."  
    - "What's been happening lately?"  
    """  
    res = memory_system.get_recent(limit)  
    return _jsonify_result(res)  
  
@mcp.tool  
def update_memory(memory_id: str, title: str = None, content: str = None,  
                  tags: str = None, importance: int = None, memory_type: str = None) -> dict:  
    """  
    Update or modify an existing memory by its unique ID.  
  
    When to use:  
    - User wants to correct, change, or add details to a stored memory.  
    - Requests like "update that memory" or "change my favorite color to blue."  
    - Use this to change content, tags, importance, or type.  
  
    Args:  
    - memory_id (str): Unique ID of the memory to update.  
    - title (str, optional): New title.  
    - content (str, optional): New content.  
    - tags (str, optional): New comma-separated tags.  
    - importance (int, optional): New importance 1–10.  
    - memory_type (str, optional): New category, e.g., "fact", "preference", "event", "conversation".  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: [ {id, ...} ] }  
  
    Example triggers:  
    - "Change that to type 'preference' and tag it 'personal'."  
    - "Update the camping note to type 'event'."  
    """  
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None  
    res = memory_system.update_memory(  
        memory_id=memory_id,  
        title=title,  
        content=content,  
        tags=tag_list,  
        importance=importance,  
        memory_type=memory_type,  
    )  
    return _jsonify_result(res)
  
@mcp.tool  
def delete_memory(memory_id: str) -> dict:  
    """  
    Permanently delete a memory by its unique ID.  
  
    When to use:  
    - User explicitly asks you to forget or erase something.  
    - Requests like "forget my old phone number" or "delete that memory."  
    - Use for permanent removal rather than updating or downgrading importance.  
  
    Args:  
    - memory_id (str): Unique ID of the memory to delete.  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str }  
  
    Example triggers:  
    - "Please forget my old address."  
    - "Delete that memory about my ex."  
    - "Erase what I told you earlier about my school."  
    """  
    res = memory_system.delete_memory(memory_id)  
    return _jsonify_result(res)  
  
@mcp.tool  
def get_memory_stats() -> dict:  
    """  
    Retrieve statistics and information about the memory system.  
  
    When to use:  
    - User asks about memory system capacity, totals, or status.  
    - Questions about "how many memories" or system health.  
    - When they want to know storage details or usage metrics.  
  
    Args:  
    - None  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: { "total_memories": int, "by_type": {...}, "by_importance": {...}, "storage_info": {...}, ...} }  
  
    Example triggers:  
    - "How many memories do you have?"  
    - "What's your memory system status?"  
    - "Show me your storage stats."  
    - "How much have you remembered so far?"  
    """  
    res = memory_system.get_statistics()  
    return _jsonify_result(res)  
  
@mcp.tool  
def create_backup() -> dict:  
    """  
    Create a complete backup of the memory system right now.  
  
    When to use:  
    - User explicitly requests a backup or save operation.  
    - Before major changes or when they want to preserve current state.  
    - Only use when directly asked - automatic backups happen regularly.  
  
    Args:  
    - None  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: { "backup_path": str, "timestamp": str, "files_backed_up": [...], ...} }  
  
    Example triggers:  
    - "Make a backup now."  
    - "Save everything to backup."  
    - "Create a backup of my memories."  
    - "Back up the system."  
    """  
    res = memory_system.create_backup()  
    return _jsonify_result(res)  
  
@mcp.tool  
def search_by_date_range(date_from: str, date_to: str = None, limit: int = 50) -> dict:  
    """  
    Find memories stored within a specific date or date range.  
  
    When to use:  
    - User asks about discussions or events during a particular time window.  
    - Queries mentioning explicit dates ("on Sept 10th") or ranges ("between Sept 1 and Sept 15").  
    - Use this instead of recent-memory search when precise dates are provided.  
  
    Args:  
    - date_from (str): Start date/time in ISO format (e.g., "2025-09-01" or "2025-09-01T10:30:00Z").  
    - date_to (str, optional): End date/time in ISO format. Defaults to current UTC time if omitted.  
    - limit (int, optional): Max results to return (default 50).  
  
    Returns:  
    - dict: { "success": bool, "reason"?: str, "data"?: [ {id, title, content, timestamp, tags, memory_type, ...} ] }  
  
    Example triggers:  
    - "What did we discuss on September 10th?"  
    - "Show me everything between September 1 and 15."  
    - "What memories are there from last week?"  
    - "Pull up our conversations from August."  
    """  
    if date_to is None:  
        date_to = datetime.now(timezone.utc).isoformat()  
    res = memory_system.search_structured(date_from=date_from, date_to=date_to, limit=limit)  
    return _jsonify_result(res)  
  
# Cleanup on exit  
  
atexit.register(memory_system.close)

if __name__ == "__main__":
    try:
        # Default: stdio transport, port unused
        asyncio.run(mcp.run_stdio_async())
    except KeyboardInterrupt:
        print("\nShutting down memory system...")
        memory_system.close()
    except Exception as e:
        print(f"Error running MCP server: {e}")
        memory_system.close()
