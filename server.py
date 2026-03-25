#!/usr/bin/env python3
"""
Main entry point for the Long-Term Memory MCP Server.

This is a refactored version that uses a modular architecture with separate concerns:
- config.py: Configuration constants
- models.py: Data models (dataclasses)
- memory_system.py: Core RobustMemorySystem class
- mcp_tools.py: MCP tool handler registration
- vector_backends/: Pluggable vector storage (chromadb, pgvector)

Usage examples:
    # Default: ChromaDB backend, stdio transport
    python server.py

    # pgvector backend (first run auto-migrates from ChromaDB)
    python server.py --vector-backend pgvector \\
        --pg-host localhost --pg-port 5432 \\
        --pg-database memories --pg-user memory_user --pg-password secret

    # pgvector using PG* environment variables
    PGHOST=localhost PGDATABASE=memories python server.py --vector-backend pgvector

    # HTTP transport with pgvector
    python server.py --transport http --port 8000 --vector-backend pgvector
"""

import asyncio
import atexit
import argparse
import signal
import sys

# Third-party imports
try:
    from fastmcp import FastMCP
except ImportError as e:
    print("Missing required package. Install with: pip install fastmcp")
    print(f"Error: {e}")
    sys.exit(1)

# Local imports
from memory_mcp import RobustMemorySystem, register_tools
from memory_mcp.config import EMBEDDING_MODEL_CONFIG


def _build_vector_backend(args):
    """Construct the appropriate vector backend from CLI arguments.

    Returns None for chromadb (the default), which causes
    RobustMemorySystem to create its own ChromaBackend internally.
    For pgvector, returns an un-initialized PgvectorBackend instance.
    """
    if args.vector_backend == "chromadb":
        return None  # RobustMemorySystem defaults to ChromaBackend

    if args.vector_backend == "pgvector":
        try:
            from memory_mcp.vector_backends.pgvector_backend import PgvectorBackend
        except ImportError as e:
            print(
                "pgvector backend requires extra packages. Install with:\n"
                "  pip install 'psycopg[binary]' pgvector"
            )
            print(f"Error: {e}")
            sys.exit(1)

        return PgvectorBackend(
            host=args.pg_host,
            port=args.pg_port,
            database=args.pg_database,
            user=args.pg_user,
            password=args.pg_password,
            dimensions=EMBEDDING_MODEL_CONFIG["dimensions"],
        )

    # Shouldn't reach here because argparse validates choices
    print(f"Unknown vector backend: {args.vector_backend}")
    sys.exit(1)


def _build_database_backend(args):
    """Construct the appropriate database backend from CLI arguments.

    Returns None for chromadb mode (the default), which causes
    RobustMemorySystem to create its own SQLiteDatabase internally.
    For pgvector mode, returns an un-initialized PostgresDatabase that
    shares the same Postgres connection settings as the vector backend,
    consolidating structured data + vectors into a single database.
    """
    if args.vector_backend == "chromadb":
        return None  # RobustMemorySystem defaults to SQLiteDatabase

    if args.vector_backend == "pgvector":
        try:
            from memory_mcp.database_backends.postgres import PostgresDatabase
        except ImportError as e:
            print(
                "Postgres database backend requires 'psycopg[binary]'. "
                "Install with:  pip install 'psycopg[binary]'"
            )
            print(f"Error: {e}")
            sys.exit(1)

        return PostgresDatabase(
            host=args.pg_host,
            port=args.pg_port,
            database=args.pg_database,
            user=args.pg_user,
            password=args.pg_password,
        )

    return None


def _maybe_migrate_chromadb_to_pgvector(memory_system, args):
    """Auto-migrate vectors from ChromaDB to pgvector on first run.

    Checks whether the pgvector table is empty while a local ChromaDB
    database exists with vectors.  If so, performs a one-time migration
    by re-embedding all memories from the *database backend* (which is
    Postgres in pgvector mode) into the pgvector vector backend.

    IMPORTANT: This assumes structured data (memories table) has already
    been migrated to Postgres (e.g., via the GUI migration tool).  The
    rebuild_vector_index() call reads from the active database backend
    and generates embeddings for all rows it finds there.
    """
    if args.vector_backend != "pgvector":
        return

    # Only migrate if the pgvector table is empty (first run)
    if memory_system.vector_backend.count() > 0:
        return

    from pathlib import Path
    from memory_mcp.config import DATA_FOLDER

    chroma_dir = Path(DATA_FOLDER) / "memory_db" / "chroma_db"
    if not chroma_dir.exists():
        return

    # Check if there's anything in ChromaDB to migrate
    source = None
    try:
        from memory_mcp.vector_backends.chroma import ChromaBackend

        source = ChromaBackend(db_folder=Path(DATA_FOLDER) / "memory_db")
        source.initialize()
        source_count = source.count()

        if source_count == 0:
            return

        print(
            f"Migrating {source_count} vectors from ChromaDB to pgvector "
            f"(one-time operation)..."
        )

        # Rebuild the vector index — this re-embeds all rows from the
        # active database backend (Postgres) into the pgvector backend.
        result = memory_system.rebuild_vector_index()

        if result.success:
            new_count = "?"
            if result.data and len(result.data) > 0 and result.data[0]:
                new_count = result.data[0].get("count", "?")
            print(f"Migration complete: {new_count} vectors indexed in pgvector.")
        else:
            print(f"Migration failed: {result.reason}")
            print("You can retry with the rebuild_vectors MCP tool.")

    except Exception as e:
        print(f"Auto-migration from ChromaDB skipped: {e}")
        print("You can manually run rebuild_vectors later.")
    finally:
        if source is not None:
            try:
                source.close()
            except Exception:
                pass


def main():
    """Main entry point for the MCP server"""

    # ── Parse command-line arguments FIRST ──────────────────────
    parser = argparse.ArgumentParser(
        description="Long-Term Memory MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Vector backend examples:\n"
            "  %(prog)s                                    # ChromaDB (default)\n"
            "  %(prog)s --vector-backend pgvector           # pgvector with PG* env vars\n"
            "  %(prog)s --vector-backend pgvector \\\n"
            "      --pg-host localhost --pg-database memories\n"
        ),
    )

    # Transport arguments
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp/",
        help="URL path for HTTP transport (default: /mcp/)",
    )

    # Vector backend arguments
    parser.add_argument(
        "--vector-backend",
        type=str,
        choices=["chromadb", "pgvector"],
        default="chromadb",
        help="Vector storage backend (default: chromadb)",
    )
    parser.add_argument(
        "--pg-host",
        type=str,
        default=None,
        help="PostgreSQL host (default: PGHOST env or localhost)",
    )
    parser.add_argument(
        "--pg-port",
        type=int,
        default=None,
        help="PostgreSQL port (default: PGPORT env or 5433)",
    )
    parser.add_argument(
        "--pg-database",
        type=str,
        default=None,
        help="PostgreSQL database name (default: PGDATABASE env or 'memories')",
    )
    parser.add_argument(
        "--pg-user",
        type=str,
        default=None,
        help="PostgreSQL user (default: PGUSER env or 'memory_user')",
    )
    parser.add_argument(
        "--pg-password",
        type=str,
        default=None,
        help="PostgreSQL password (default: PGPASSWORD env or 'memory_pass')",
    )

    args = parser.parse_args()

    # ── Build backends ────────────────────────────────────────────
    vector_backend = _build_vector_backend(args)
    database_backend = _build_database_backend(args)

    # ── Initialize the memory system ────────────────────────────
    memory_system = RobustMemorySystem(
        vector_backend=vector_backend,
        database_backend=database_backend,
    )

    # Auto-migrate from ChromaDB on first pgvector run
    _maybe_migrate_chromadb_to_pgvector(memory_system, args)

    # Setup FastMCP
    mcp = FastMCP("RobustMemory")

    # Register all MCP tools
    register_tools(mcp, memory_system)

    # ── Shutdown helpers ────────────────────────────────────────
    _shutting_down = False

    def _graceful_shutdown(signum=None, frame=None):
        """Handle SIGTERM / SIGHUP by closing the memory system cleanly.

        This ensures the WAL is checkpointed and the SQLite connection is
        closed before the process exits — critical on macOS where the OS
        may send SIGTERM on sleep/logout.
        """
        nonlocal _shutting_down
        if _shutting_down:
            return  # Avoid re-entrancy
        _shutting_down = True

        sig_name = signal.Signals(signum).name if signum else "atexit"
        print(f"\nReceived {sig_name}, shutting down memory system...")
        memory_system.close()

    # Register cleanup on normal exit (atexit) and OS signals
    atexit.register(_graceful_shutdown)

    # SIGTERM: sent by launchd / systemd / Docker on stop
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    # SIGHUP:  sent when terminal is closed or SSH disconnects
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _graceful_shutdown)

    # ── Run the server ──────────────────────────────────────────
    backend_info = f"[{args.vector_backend}]"

    try:
        if args.transport == "http":
            print(
                f"Starting Long-Term Memory MCP Server {backend_info} "
                f"on http://{args.host}:{args.port}{args.path}"
            )
            # mcp.run() is synchronous in FastMCP 3.x — it calls anyio.run()
            # internally and blocks until the server stops.
            # Do NOT wrap in asyncio.run(): that expects a coroutine and would
            # receive None (the return value of the sync run()), raising
            # "a coroutine was expected, got None" on shutdown.
            mcp.run(transport="http", host=args.host, port=args.port, path=args.path)
        else:
            # stdio transport — run_async is a coroutine so asyncio.run() is correct
            asyncio.run(mcp.run_stdio_async(show_banner=False))
    except KeyboardInterrupt:
        print("\nShutting down memory system...")
        memory_system.close()
    except Exception as e:
        print(f"Error running MCP server: {e}")
        memory_system.close()


if __name__ == "__main__":
    main()
