"""
MCP tool handlers module.

Contains all FastMCP tool definitions that wrap the memory system operations.
"""

import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Optional

from .models import Result
from .audit import AuditLogger


def jsonify_result(res: Result) -> dict:
    """
    Convert Result dataclass to JSON-serializable dict.

    Normalizes datetime objects to ISO strings, strips internal bookkeeping
    fields from the metadata dict (and any hoisted top-level duplicates),
    and ensures all fields are JSON-safe.

    Internal keys stripped from both metadata dict and top-level:
      - reinforcement_accum  (decay/reinforcement accumulator)
      - last_decay_at        (last decay timestamp)

    The metadata dict is dropped entirely if no user-defined keys remain after
    stripping. If it contains custom fields, those are preserved.
    """
    # Keys that are pure internal bookkeeping — stripped from both the nested
    # metadata dict AND from the top level (webui_api hoists reinforcement_accum
    # to top-level so the frontend can sort; that copy is redundant for LLMs).
    _INTERNAL_KEYS = {
        "reinforcement_accum",
        "last_decay_at",
    }

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
            # Strip internal keys hoisted to top-level (e.g. reinforcement_accum)
            for key in _INTERNAL_KEYS:
                obj.pop(key, None)
            # Strip internal keys from within the metadata dict; drop the dict
            # entirely if nothing user-defined remains (saves tokens).
            meta = obj.get("metadata")
            if isinstance(meta, dict):
                stripped = {k: v for k, v in meta.items() if k not in _INTERNAL_KEYS}
                if stripped:
                    obj["metadata"] = stripped
                else:
                    del obj["metadata"]
            data.append(obj)
        out["data"] = data
    return out


def register_tools(mcp, memory_system, audit_logger: Optional[AuditLogger] = None):
    """
    Register all MCP tools with the FastMCP instance.

    Args:
        mcp: FastMCP instance
        memory_system: RobustMemorySystem instance
        audit_logger: Optional AuditLogger. When provided every tool call is
            written to the daily-rotating JSONL audit file.
    """

    def _audit(tool_name: str, args: dict, fn):
        """
        Call *fn()* and, if an audit_logger is configured, write one record.

        *fn* is a zero-argument callable that performs the actual tool work
        and returns the jsonified result dict.

        Guarantees:
        - An audit I/O failure (disk full, permission error, etc.) never
          masks or suppresses the result of fn().
        - A non-dict return from fn() does not crash the audit path.
        """
        if audit_logger is None:
            return fn()

        call_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        error_str: str | None = None
        result: dict = {}
        raised = False

        try:
            result = fn()
        except Exception:
            error_str = traceback.format_exc()
            result = {"success": False, "reason": error_str.splitlines()[-1]}
            raised = True
            raise
        finally:
            duration_ms = round((time.perf_counter() - t0) * 1000)
            success = (
                result.get("success", False)
                if isinstance(result, dict)
                else bool(result)
            )
            # Sum token_count across all result items when available.
            # remember() stores it in data[0]; search/get tools now include
            # it on every item. Summing gives total tokens for the call.
            token_count: int | None = None
            if isinstance(result, dict):
                data = result.get("data")
                if isinstance(data, list) and data:
                    total = sum(
                        item.get("token_count") or 0
                        for item in data
                        if isinstance(item, dict)
                    )
                    if total > 0:
                        token_count = total
            record = {
                "call_id": call_id,
                "timestamp": ts,
                "tool": tool_name,
                "token_count": token_count,
                "args": args,
                "success": False if raised else success,
                "duration_ms": duration_ms,
                "result": result,
                "error": error_str,
            }
            # write() catches its own exceptions internally, but wrap here too
            # so that any unexpected failure never suppresses the original
            # exception from fn().
            try:
                audit_logger.write(record)
            except Exception:  # noqa: BLE001
                pass

        return result

    @mcp.tool
    def remember(
        title: str,
        content: str,
        tags: str = "",
        importance: int = 5,
        memory_type: str = "conversation",
        shared_with: str = "",
        file_paths: str = "",
    ) -> dict:
        """
        Store a new memory (fact, preference, event, conversation snippet, or session summary).

        When to use:
        - The user shares something to keep or says "remember this."
        - New personal details, preferences, events, instructions.
        - The harness agent summarizes a session (use memory_type="summary").

        Args:
        - title (str): Short title for the memory.
        - content (str): Full text to store.
        - tags (str, optional): Comma-separated tags, e.g., "personal, preference".
        - importance (int, optional): 1–10 (default 5). Higher = more important.
        - memory_type (str, optional): e.g., "conversation", "fact", "preference", "event", "summary".
        - shared_with (str, optional): Comma-separated peer UUIDs, or "*" for everyone.
            Leave empty for private (default). Examples:
            "*"                         — share with all discovered peers
            "uuid1,uuid2"               — share with specific peers only
        - file_paths (str, optional): Comma-separated absolute paths to source files.
            When provided AND memory_type is "fact", AST symbols are automatically
            extracted and appended as _symbols_at_storage for staleness detection.
            Supports: Python, TypeScript, TSX, JavaScript, Go, Rust, Java, Kotlin, C/C++.
            Example: "/abs/path/to/server.py,/abs/path/to/models.py"

        Returns:
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded.
                - reason (str, optional): Explanation when the operation fails.
                - data (list, optional): List of memory objects. Each object includes:
                    - id, title, content, timestamp, tags, importance, memory_type,
                      shared_with, ... (additional fields as needed)
                    - warning (str, optional): "potential_contradiction" if a similar
                      memory already exists. Check conflicting_id / conflicting_title.

        Example triggers:
        - "My birthday is July 4th."
        - "Remember that I prefer tea over coffee."
        - "Please save this: truck camping next weekend."
        - Session recap from the harness (memory_type="summary").
        """
        tag_list = (
            [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        )
        shared_with_list = (
            [s.strip() for s in shared_with.split(",") if s.strip()]
            if shared_with
            else []
        )
        file_paths_list = (
            [p.strip() for p in file_paths.split(",") if p.strip()]
            if file_paths
            else []
        )
        return _audit(
            "remember",
            {
                "title": title,
                "content": content,
                "tags": tags,
                "importance": importance,
                "memory_type": memory_type,
                "shared_with": shared_with,
                "file_paths": file_paths,
            },
            lambda: jsonify_result(
                memory_system.remember(
                    title,
                    content,
                    tag_list,
                    importance,
                    memory_type,
                    shared_with=shared_with_list,
                    file_paths=file_paths_list,
                )
            ),
        )

    @mcp.tool
    def search_memories(
        query: str, search_type: str = "semantic", limit: int = 10
    ) -> dict:
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
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded.
                - reason (str, optional): Explanation when the operation fails.
                - data (list, optional): List of match results. Each result includes:
                    - id
                    - title
                    - content
                    - timestamp
                    - tags
                    - relevance_score
                    - match_type
                    - ... (additional fields as needed)

        Example triggers:
        - "When is my birthday?"
        - "What did I tell you about my favorite color?"
        - "Do you remember what I said about camping?"
        """
        if search_type == "semantic":
            return _audit(
                "search_memories",
                {"query": query, "search_type": search_type, "limit": limit},
                lambda: jsonify_result(memory_system.search_semantic(query, limit)),
            )
        return _audit(
            "search_memories",
            {"query": query, "search_type": search_type, "limit": limit},
            lambda: jsonify_result(memory_system.search_structured(limit=limit)),
        )

    @mcp.tool
    def search_by_type(memory_type: str, limit: int = 20) -> dict:
        """
        Retrieve memories by category/type for organized recall.

        When to use:
        - User asks for a specific category of memories.
        - Requests like "show me all my preferences" or "list my facts."
        - When they want to see everything in a particular memory type.

        Args:
        - memory_type (str): Category to search for, e.g., "conversation", "fact",
        "preference", "event", "summary".
        - limit (int, optional): Max results to return (default 20).

        Returns:
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded.
                - reason (str, optional): Explanation when the operation fails.
                - data (list, optional): List of memory objects. Each object includes:
                    - id
                    - title
                    - content
                    - timestamp
                    - tags
                    - memory_type
                    - ... (additional fields as needed)

        Example triggers:
        - "Show me all my preferences so far."
        - "List the facts you know about me."
        - "What events have we discussed?"
        - "Show me session summaries."
        """
        return _audit(
            "search_by_type",
            {"memory_type": memory_type, "limit": limit},
            lambda: jsonify_result(
                memory_system.search_structured(memory_type=memory_type, limit=limit)
            ),
        )

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
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded.
                - reason (str, optional): Explanation when operation fails.
                - data (list, optional): List of memory objects. Each object
                  includes id, title, content, timestamp, tags, memory_type,
                  and other fields as needed.

        Example triggers:
        - "Find everything tagged camping and truck."
        - "Show me memories about music."
        - "What do you have tagged as personal?"
        """
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        return _audit(
            "search_by_tags",
            {"tags": tags, "limit": limit},
            lambda: jsonify_result(
                memory_system.search_structured(tags=tag_list, limit=limit)
            ),
        )

    @mcp.tool
    def get_recent_memories(limit: int = 20, current_project: str = None, include_summaries: bool = False) -> dict:
        """
        Retrieve the most recently stored memories for timeline-based recall.

        When to use:
        - User asks about recent interactions or conversations.
        - Time-based queries like "today," "last night," "recently," "yesterday."
        - When they want to review what was discussed in the current or recent sessions.
        - Use this instead of date ranges when no specific dates are mentioned.

        Summary memories (memory_type="summary") are excluded by default because
        they are verbose session recaps that clutter timeline-based recall.
        Pass include_summaries=True to include them — e.g. when the user
        explicitly asks to see session summaries.

        Args:
        - limit (int, optional): Max *recent* memories to return (default 20).
          All preference memories (memory_type="preference") are always returned
          in full on top of this count — they are short context items and are
          not subject to the limit.
        - current_project (str, optional): Project identifier to filter memories.
          When provided, only returns memories tagged with this project.
          Use the current working directory name as the project identifier.
          Set to None or empty string to retrieve memories from all projects.
        - include_summaries (bool, optional): Whether to include memories with
          memory_type="summary" (default False). Summaries are session recaps
          and are omitted from routine recall to keep results focused.
          Set to True only when the user explicitly wants session summaries.

        Returns:
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded
                - reason (str, optional): Error message if failed
                - data (list, optional): List of memory objects, each with
                  id, title, content, timestamp, tags, memory_type, etc.

        Example triggers:
        - "What did we talk about today?"
        - "What have we discussed recently?"
        - "Remind me what we covered last night."
        - "What's been happening lately?"

        Example with project filtering:
        - get_recent_memories(limit=10, current_project="long-term-memory-mcp")
        - Returns only memories tagged with "long-term-memory-mcp"

        Example including summaries:
        - get_recent_memories(include_summaries=True)
        - Returns recent memories including session recap summaries
        """
        return _audit(
            "get_recent_memories",
            {"limit": limit, "current_project": current_project, "include_summaries": include_summaries},
            lambda: jsonify_result(
                memory_system.get_recent(limit, current_project=current_project, include_summaries=include_summaries)
            ),
        )

    @mcp.tool
    def update_memory(
        memory_id: str,
        title: str = None,
        content: str = None,
        tags: str = None,
        importance: int = None,
        memory_type: str = None,
        shared_with: str = None,
    ) -> dict:
        """
        Update or modify an existing memory by its unique ID.

        When to use:
        - User wants to correct, change, or add details to a stored memory.
        - Requests like "update that memory" or "change my favorite color to blue."
        - Use this to change content, tags, importance, type, or shared_with.

        Args:
        - memory_id (str): Unique ID of the memory to update.
        - title (str, optional): New title.
        - content (str, optional): New content.
        - tags (str, optional): New comma-separated tags.
        - importance (int, optional): New importance 1–10.
        - memory_type (str, optional): New category, e.g., "fact", "preference", "event",
        "conversation", "summary".
        - shared_with (str, optional): Comma-separated peer UUIDs or "*" for everyone.
            Pass "" (empty string) to make private.

        Returns:
        - dict: { "success": bool, "reason"?: str, "data"?: [ {id, ...} ] }

        Example triggers:
        - "Change that to type 'preference' and tag it 'personal'."
        - "Update the camping note to type 'event'."
        - "Share that memory with everyone on the network."
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        shared_with_list = None
        if shared_with is not None:
            shared_with_list = [s.strip() for s in shared_with.split(",") if s.strip()]
        return _audit(
            "update_memory",
            {
                "memory_id": memory_id,
                "title": title,
                "content": content,
                "tags": tags,
                "importance": importance,
                "memory_type": memory_type,
                "shared_with": shared_with,
            },
            lambda: jsonify_result(
                memory_system.update_memory(
                    memory_id=memory_id,
                    title=title,
                    content=content,
                    tags=tag_list,
                    importance=importance,
                    memory_type=memory_type,
                    shared_with=shared_with_list,
                )
            ),
        )

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
        return _audit(
            "delete_memory",
            {"memory_id": memory_id},
            lambda: jsonify_result(memory_system.delete_memory(memory_id)),
        )

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
        - dict: {
            "success": bool,
            "reason"?: str,
            "data"?: {
                "total_memories": int,
                "by_type": {...},
                "by_importance": {...},
                "storage_info": {...},
                ...
                }
            }

        Example triggers:
        - "How many memories do you have?"
        - "What's your memory system status?"
        - "Show me your storage stats."
        - "How much have you remembered so far?"
        """
        return _audit(
            "get_memory_stats",
            {},
            lambda: jsonify_result(memory_system.get_statistics()),
        )

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
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded.
                - reason (str, optional): Explanation when the operation fails.
                - data (dict, optional): Backup details, including:
                    - backup_path (str): Filesystem path to the backup.
                    - timestamp (str): ISO 8601 timestamp of when the backup was created.
                    - files_backed_up (list): List of file paths included in the backup.
                    - ...: Additional fields as needed.

        Example triggers:
        - "Make a backup now."
        - "Save everything to backup."
        - "Create a backup of my memories."
        - "Back up the system."
        """
        return _audit(
            "create_backup",
            {},
            lambda: jsonify_result(memory_system.create_backup()),
        )

    @mcp.tool
    def search_by_date_range(
        date_from: str, date_to: str = None, limit: int = 50
    ) -> dict:
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
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded.
                - reason (str, optional): Explanation when the operation fails.
                - data (list, optional): List of memory objects. Each object includes:
                    - id
                    - title
                    - content
                    - timestamp
                    - tags
                    - memory_type
                    - ... (additional fields as needed)

        Example triggers:
        - "What did we discuss on September 10th?"
        - "Show me everything between September 1 and 15."
        - "What memories are there from last week?"
        - "Pull up our conversations from August."
        """
        if date_to is None:
            date_to = datetime.now(timezone.utc).isoformat()
        return _audit(
            "search_by_date_range",
            {"date_from": date_from, "date_to": date_to, "limit": limit},
            lambda: jsonify_result(
                memory_system.search_structured(
                    date_from=date_from, date_to=date_to, limit=limit
                )
            ),
        )

    @mcp.tool
    def rebuild_vectors() -> dict:
        """
        One-time repair: rebuild vector index from SQLite memories.
        Use if semantic search isn't working but structured search is.
        """
        return _audit(
            "rebuild_vectors",
            {},
            lambda: jsonify_result(memory_system.rebuild_vector_index()),
        )

    @mcp.tool
    def list_source_memories(source_db_path: str, limit: int = 100) -> dict:
        """
        List memories from a source database for migration preview.

        Use this to view memories from another database before migrating them.
        Helpful for verifying what will be transferred.

        Args:
            source_db_path (str): Full path to the source SQLite database file
                (e.g., "/Users/name/Documents/ai_companion_memory/memory_db/memories.db")
            limit (int, optional): Maximum number of memories to list (default 100)

        Returns:
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded
                - reason (str, optional): Error message if failed
                - data (list, optional): List of memory objects from source database, each with:
                    - id
                    - title
                    - content (truncated to 200 chars)
                    - timestamp
                    - tags
                    - importance
                    - memory_type
                    - token_count

        Example triggers:
        - "Show me what memories are in the old database"
        - "List memories from the default database location"
        - "Preview what will be migrated"
        """
        return _audit(
            "list_source_memories",
            {"source_db_path": source_db_path, "limit": limit},
            lambda: jsonify_result(
                memory_system.list_source_memories(source_db_path, limit)
            ),
        )

    @mcp.tool
    def migrate_memories(
        source_db_path: str,
        source_chroma_path: str = None,
        memory_ids: str = None,
        skip_duplicates: bool = True,
    ) -> dict:
        """
        Migrate memories from a source database to the active database.
        Transfers both SQLite records and ChromaDB vectors.

        Use this when you ran the memory system with default settings and memories
        were stored in a separate database that you want to merge into your active database.

        Args:
            source_db_path (str): Full path to the source SQLite database file
                (e.g., "/Users/name/Documents/ai_companion_memory/memory_db/memories.db")
            source_chroma_path (str, optional): Full path to the source ChromaDB directory.
                If not provided, will auto-detect by looking for chroma_db in the same
                directory as the SQLite database.
            memory_ids (str, optional): Comma-separated list of specific memory IDs to migrate.
                If not provided, all memories will be migrated.
                (e.g., "mem_abc123,mem_def456")
            skip_duplicates (bool, optional): If True, skip memories with duplicate content
                hashes (default: True). Set to False to import everything regardless of duplicates.

        Returns:
            dict: Dictionary with the following keys:
                - success (bool): Whether the operation succeeded
                - reason (str, optional): Error message if failed
                - data (list, optional): Migration statistics including:
                    - total_found: Total memories found in source
                    - migrated: Number successfully migrated
                    - skipped_duplicates: Number skipped due to duplicate content
                    - errors: Number of errors encountered
                    - vectors_migrated: Number of ChromaDB vectors transferred

        Example triggers:
        - "Migrate all memories from the default database"
        - "Transfer memories from /path/to/old/memories.db"
        - "Import specific memories: mem_123, mem_456"
        - "Migrate including duplicates"

        Warning:
        - Always use list_source_memories first to preview what will be migrated
        - Backup your active database before migrating (use create_backup)
        - Migration is additive - it adds to your active database, doesn't replace it
        """
        # Parse memory_ids if provided
        memory_id_list = None
        if memory_ids:
            memory_id_list = [mid.strip() for mid in memory_ids.split(",")]

        return _audit(
            "migrate_memories",
            {
                "source_db_path": source_db_path,
                "source_chroma_path": source_chroma_path,
                "memory_ids": memory_ids,
                "skip_duplicates": skip_duplicates,
            },
            lambda: jsonify_result(
                memory_system.migrate_memories(
                    source_db_path=source_db_path,
                    source_chroma_path=source_chroma_path,
                    memory_ids=memory_id_list,
                    skip_duplicates=skip_duplicates,
                )
            ),
        )

    # ── Knowledge Graph tools ─────────────────────────────────────────────────

    @mcp.tool
    def kg_add(
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source_memory_id: str = None,
    ) -> dict:
        """
        Add a fact to the knowledge graph as a (subject, predicate, object) triple.

        Use this to record structured, long-lived facts about people, projects,
        tools, or any entity — especially facts that may change over time.

        Args:
        - subject (str): The entity the fact is about. E.g. "Alice", "ProjectX".
        - predicate (str): The relationship type. E.g. "works_at", "uses", "knows",
          "has_skill", "lives_in", "is_blocked_by". Spaces are converted to underscores.
        - obj (str): The target entity or value. E.g. "Acme Corp", "Python", "Bob".
        - valid_from (str, optional): ISO date when this fact became true, e.g. "2023-01-01".
        - valid_to (str, optional): ISO date when this fact stopped being true.
          Leave blank for currently-true facts.
        - confidence (float, optional): 0.0–1.0 confidence score (default 1.0).
        - source_memory_id (str, optional): ID of the memory this fact was extracted from.

        Returns:
        - dict: { "success": bool, "data": [{"triple_id": str, "subject": str, ...}] }

        Example triggers:
        - "Alice works at Acme Corp since 2023."
        - "Bob knows Python."
        - "Project X depends on Project Y."
        - "Alice moved to Berlin in March 2025."
        """
        return _audit(
            "kg_add",
            {
                "subject": subject,
                "predicate": predicate,
                "obj": obj,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "confidence": confidence,
                "source_memory_id": source_memory_id,
            },
            lambda: jsonify_result(
                memory_system.kg_add_triple(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    confidence=confidence,
                    source_memory_id=source_memory_id,
                )
            ),
        )

    @mcp.tool
    def kg_invalidate(
        subject: str,
        predicate: str,
        obj: str,
        ended: str = None,
    ) -> dict:
        """
        Mark a knowledge graph fact as no longer true by setting its end date.

        Use this when a fact has changed — for example, someone changed jobs,
        a project was cancelled, or a preference was reversed.  The old fact is
        NOT deleted; it remains queryable via kg_query (with as_of) or kg_timeline.

        Args:
        - subject (str): The entity the fact is about.
        - predicate (str): The relationship type.
        - obj (str): The target entity or value.
        - ended (str, optional): ISO date the fact ended (defaults to today).

        Returns:
        - dict: { "success": bool }

        Example triggers:
        - "Alice no longer works at Acme Corp."
        - "The dependency on library X was removed."
        - "Bob moved out of Berlin."
        """
        return _audit(
            "kg_invalidate",
            {"subject": subject, "predicate": predicate, "obj": obj, "ended": ended},
            lambda: jsonify_result(
                memory_system.kg_invalidate(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    ended=ended,
                )
            ),
        )

    @mcp.tool
    def kg_query(
        entity: str,
        as_of: str = None,
        direction: str = "outgoing",
    ) -> dict:
        """
        Query all facts about an entity from the knowledge graph.

        Use this BEFORE making any claim about a person, project, or entity —
        verify from the graph rather than guessing.

        Args:
        - entity (str): Entity name to look up. E.g. "Alice", "ProjectX".
        - as_of (str, optional): ISO date for a time-travel query — returns only
          facts that were true on that date. If omitted, returns currently-active
          facts (those with no end date).
        - direction (str, optional): "outgoing" (entity is subject, default),
          "incoming" (entity is object), or "both".

        Returns:
        - dict: { "success": bool, "data": [ {subject, predicate, object,
          valid_from, valid_to, confidence, source_memory_id, current}, ... ] }

        Example triggers:
        - "What do we know about Alice?"
        - "What was Alice's job in 2024?"        ← use as_of="2024-01-01"
        - "Who knows Bob?"                       ← use direction="incoming"
        - "Tell me everything about ProjectX."   ← use direction="both"
        """
        return _audit(
            "kg_query",
            {"entity": entity, "as_of": as_of, "direction": direction},
            lambda: jsonify_result(
                memory_system.kg_query_entity(
                    name=entity,
                    as_of=as_of,
                    direction=direction,
                )
            ),
        )

    @mcp.tool
    def kg_timeline(entity: str = None, limit: int = 100) -> dict:
        """
        Get a chronological history of facts from the knowledge graph.

        Returns facts ordered by valid_from date, including both current and
        expired facts — useful for understanding how an entity has changed over time.

        Args:
        - entity (str, optional): Filter to a specific entity. If omitted,
          returns the global timeline (most recent facts across all entities).
        - limit (int, optional): Maximum number of facts to return (default 100).

        Returns:
        - dict: { "success": bool, "data": [ {subject, predicate, object,
          valid_from, valid_to, current}, ... ] }

        Example triggers:
        - "Show me Alice's full history."
        - "What has changed in the knowledge graph recently?"
        - "Timeline of everything about ProjectX."
        """
        return _audit(
            "kg_timeline",
            {"entity": entity, "limit": limit},
            lambda: jsonify_result(
                memory_system.kg_timeline(entity_name=entity, limit=limit)
            ),
        )

    @mcp.tool
    def kg_stats() -> dict:
        """
        Get statistics about the knowledge graph.

        Returns counts of entities, total triples, currently-active facts,
        expired facts, and the list of all relationship types in use.

        Returns:
        - dict: { "success": bool, "data": [{ "entities": int, "triples": int,
          "current_facts": int, "expired_facts": int,
          "relationship_types": [str] }] }

        Example triggers:
        - "How many facts are in the knowledge graph?"
        - "What relationship types are stored?"
        - "Knowledge graph stats."
        """
        return _audit(
            "kg_stats",
            {},
            lambda: jsonify_result(memory_system.kg_stats()),
        )

    # ── Self-teaching status tool ─────────────────────────────────────────────

    _MEMORY_PROTOCOL = """
MEMORY SYSTEM PROTOCOL — read once, apply every session.

TOOLS AVAILABLE:
  remember(title, content, tags, importance, memory_type, shared_with, file_paths)
      → Store new memory. memory_type: conversation|fact|preference|event|task|ephemeral|summary
      → For memory_type="fact" with code, always provide file_paths for staleness anchors.

  search_memories(query, limit)      → Semantic similarity search.
  search_by_tags(tags, limit)        → Tag-based filter search.
  search_by_type(memory_type, limit) → Filter by type.
  get_recent_memories(limit, current_project, include_summaries=False) → Session-start recall (call ONCE at start). Summary memories omitted by default; pass include_summaries=True to include them.
  update_memory(memory_id, ...)      → Patch any field; re-embeds if content changes.
  delete_memory(memory_id)           → Hard delete.

  kg_add(subject, predicate, obj, valid_from, valid_to, confidence, source_memory_id)
      → Add a structured fact triple. E.g. kg_add("Alice","works_at","Acme","2023-01-01")
  kg_invalidate(subject, predicate, obj, ended)
      → Expire a fact. E.g. when Alice changes jobs, invalidate the old works_at.
  kg_query(entity, as_of, direction)
      → ALWAYS call this BEFORE claiming anything about a person or project.
      → as_of accepts ISO date for time-travel: what was true on that date?
  kg_timeline(entity, limit)         → Full chronological history of an entity.
  kg_stats()                         → Entity and triple counts.

RULES:
  1. SESSION START: call get_recent_memories(current_project="<working dir name>") first.
  2. BEFORE any claim about a person/entity: call kg_query(entity) to verify — never guess.
  3. STORE STRUCTURED FACTS in both: remember() for full-text retrieval + kg_add() for facts.
  4. WHEN A FACT CHANGES: kg_invalidate() old triple, kg_add() new triple.
  5. AFTER EDITING FILES: call remember() before your next tool call (store gate).
  6. IMPORTANCE SCALE: 9-10=identity/critical, 7-8=architecture/decisions, 5-6=general, 3-4=session.
  7. TAGS: comma-separated, include project name. E.g. "project,long-term-memory-mcp,architecture".
  8. NEVER mention memory operations to the user — work silently.
""".strip()

    @mcp.tool
    def get_memory_system_info() -> dict:
        """
        Get memory system status, statistics, and the full usage protocol.

        Call this at the start of a session (after get_recent_memories) to
        orient yourself. The response embeds the complete MEMORY PROTOCOL so
        you know exactly when and how to use every tool.

        Returns:
        - dict with:
            - memories: total memory count + type breakdown
            - knowledge_graph: entity + triple counts + relationship types
            - backends: database and vector backend names
            - storage_mb: total storage used
            - protocol: complete usage instructions (read this!)

        Example triggers:
        - "What memory tools do you have?"
        - "Give me a memory system status."
        - "How does the memory system work?"
        """
        try:
            mem_stats = memory_system.get_statistics()
            kg_stats_result = memory_system.kg_stats()

            mem_data = mem_stats.data[0] if mem_stats.success and mem_stats.data else {}
            kg_data = (
                kg_stats_result.data[0]
                if kg_stats_result.success and kg_stats_result.data
                else {}
            )

            result = {
                "memories": {
                    "total": mem_data.get("total_memories", 0),
                    "type_breakdown": mem_data.get("type_breakdown", {}),
                    "avg_importance": mem_data.get("avg_importance", 0),
                },
                "knowledge_graph": kg_data,
                "backends": {
                    "database": mem_data.get("database_backend", "unknown"),
                    "vector": mem_data.get("vector_backend", "unknown"),
                },
                "storage_mb": mem_data.get("storage_size_mb", 0),
                "protocol": _MEMORY_PROTOCOL,
            }
            return _audit(
                "get_memory_system_info",
                {},
                lambda: {"success": True, "data": [result]},
            )
        except Exception as e:
            return {"success": False, "reason": str(e)}
