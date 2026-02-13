"""
MCP tool handlers module.

Contains all FastMCP tool definitions that wrap the memory system operations.
"""

from datetime import datetime, timezone
from .models import Result


def jsonify_result(res: Result) -> dict:
    """
    Convert Result dataclass to JSON-serializable dict.

    Normalizes datetime objects to ISO strings and ensures all fields are JSON-safe.
    """
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


def register_tools(mcp, memory_system):
    """
    Register all MCP tools with the FastMCP instance.

    Args:
        mcp: FastMCP instance
        memory_system: RobustMemorySystem instance
    """

    @mcp.tool
    def remember(
        title: str,
        content: str,
        tags: str = "",
        importance: int = 5,
        memory_type: str = "conversation",
    ) -> dict:
        """
        Store a new memory (fact, preference, event, or conversation snippet).

        When to use:
        - The user shares something to keep or says "remember this."
        - New personal details, preferences, events, instructions.

        Args:
        - title (str): Short title for the memory.
        - content (str): Full text to store.
        - tags (str, optional): Comma-separated tags, e.g., "personal, preference".
        - importance (int, optional): 1–10 (default 5). Higher = more important.
        - memory_type (str, optional): e.g., "conversation", "fact", "preference", "event".

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
                    - importance
                    - memory_type
                    - ... (additional fields as needed)

        Example triggers:
        - "My birthday is July 4th."
        - "Remember that I prefer tea over coffee."
        - "Please save this: truck camping next weekend."
        """
        tag_list = (
            [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        )
        res = memory_system.remember(title, content, tag_list, importance, memory_type)
        return jsonify_result(res)

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
            res = memory_system.search_semantic(query, limit)
        else:
            res = memory_system.search_structured(limit=limit)
        return jsonify_result(res)

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
        "preference", "event".
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
        """
        res = memory_system.search_structured(memory_type=memory_type, limit=limit)
        return jsonify_result(res)

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
        res = memory_system.search_structured(tags=tag_list, limit=limit)
        return jsonify_result(res)

    @mcp.tool
    def get_recent_memories(limit: int = 20, current_project: str = None) -> dict:
        """
        Retrieve the most recently stored memories for timeline-based recall.

        When to use:
        - User asks about recent interactions or conversations.
        - Time-based queries like "today," "last night," "recently," "yesterday."
        - When they want to review what was discussed in the current or recent sessions.
        - Use this instead of date ranges when no specific dates are mentioned.

        Args:
        - limit (int, optional): Max results to return (default 20).
        - current_project (str, optional): Project identifier to filter memories.
          When provided, only returns memories tagged with this project.
          Use the current working directory name as the project identifier.
          Set to None or empty string to retrieve memories from all projects.

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
        """
        res = memory_system.get_recent(limit, current_project=current_project)
        return jsonify_result(res)

    @mcp.tool
    def update_memory(
        memory_id: str,
        title: str = None,
        content: str = None,
        tags: str = None,
        importance: int = None,
        memory_type: str = None,
    ) -> dict:
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
        - memory_type (str, optional): New category, e.g., "fact", "preference", "event",
        "conversation".

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
        return jsonify_result(res)

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
        return jsonify_result(res)

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
        res = memory_system.get_statistics()
        return jsonify_result(res)

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
        res = memory_system.create_backup()
        return jsonify_result(res)

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
        res = memory_system.search_structured(
            date_from=date_from, date_to=date_to, limit=limit
        )
        return jsonify_result(res)

    @mcp.tool
    def rebuild_vectors() -> dict:
        """
        One-time repair: rebuild vector index from SQLite memories.
        Use if semantic search isn't working but structured search is.
        """
        res = memory_system.rebuild_vector_index()
        return jsonify_result(res)

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
        res = memory_system.list_source_memories(source_db_path, limit)
        return jsonify_result(res)

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

        res = memory_system.migrate_memories(
            source_db_path=source_db_path,
            source_chroma_path=source_chroma_path,
            memory_ids=memory_id_list,
            skip_duplicates=skip_duplicates,
        )
        return jsonify_result(res)
