# Robust Long-Term Memory MCP

A persistent, human-like memory system for AI companions, powered by pluggable backends for both structured storage and vector search. Designed for decades-long use, seamless recall across sessions, and automatic backups -- making your AI companion feel like a continuous, living persona. Now with biological behavior: time-based lazy decay and reinforcement by use.

---

## Features

- **Pluggable hybrid memory system**
  - **Database backends**: SQLite (default) or PostgreSQL
  - **Vector backends**: ChromaDB (default) or pgvector
  - **JSON backups** for portability
  - When using pgvector, both structured data and vectors live in a **single PostgreSQL database**

- **Cross-chat continuity**: memories persist beyond a single chat
- **Cross-model continuity**: swap models freely, the memory stays intact
- **Cross-machine portability**: move the database to another system and continue seamlessly
- **Automatic backups**: daily backups and after every 100 memories, pruned to keep the last 10
- **Invisible memory integration**: tools are hidden from the user; conversations feel natural
- **Desktop GUI** (`memory_manager_gui.py`): browse, search, migrate, and compare databases visually
- Human-like dynamics
  - Lazy Decay: importance decreases only when a memory is accessed after idle time
  - Reinforcement: frequent recall strengthens memory importance
  - Adaptive Semantic Threshold: balances precision/recall with a safe top-1 fallback

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Rotoslider/long-term-memory-mcp.git
cd long-term-memory-mcp
```

### 2. Install core dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `chromadb`, `sentence-transformers`, `fastmcp`, `tiktoken`, `numpy`, `torch` (`sqlite3` is built into Python).

### 3. (Optional) Install pgvector backend

If you want to use PostgreSQL + pgvector instead of (or alongside) ChromaDB:

```bash
pip install 'psycopg[binary]' pgvector
```

Or via the optional extra:

```bash
pip install '.[pgvector]'
```

### 4. (Optional) Faster HuggingFace downloads

```bash
pip install "huggingface_hub[hf_xet]"
```

---

## Project Structure

```
long-term-memory-mcp/
├── server.py                        # Main entry point
├── memory_mcp/                      # Core package
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # Configuration constants
│   ├── models.py                    # Data models (dataclasses)
│   ├── memory_system.py             # Core RobustMemorySystem class
│   ├── mcp_tools.py                 # MCP tool registration
│   ├── database_backends/           # Pluggable structured storage
│   │   ├── base.py                  # DatabaseBackend ABC
│   │   ├── sqlite.py                # SQLite implementation (default)
│   │   └── postgres.py              # PostgreSQL implementation
│   └── vector_backends/             # Pluggable vector storage
│       ├── base.py                  # VectorBackend ABC
│       ├── chroma.py                # ChromaDB implementation (default)
│       └── pgvector_backend.py      # pgvector implementation
├── memory_manager_gui.py            # Desktop GUI for memory management
├── docker-compose.yml               # pgvector Docker service
├── long_term_memory_mcp.py          # Legacy monolithic file (still functional)
├── pyproject.toml                   # Package config with optional [pgvector] extra
└── requirements.txt                 # Dependencies
```

---

## Architecture

### Default: SQLite + ChromaDB (no Docker required)

```
┌─────────────────────┐    ┌─────────────────────┐
│  SQLite (memories.db)│    │  ChromaDB (chroma_db/)│
│  Structured metadata │    │  Vector embeddings    │
└─────────────────────┘    └─────────────────────┘
```

Two local files, zero infrastructure. This is the default and requires no extra setup.

### Optional: Single PostgreSQL database (pgvector mode)

```
┌─────────────────────────────────────────┐
│        PostgreSQL + pgvector            │
│  ┌───────────────┐  ┌────────────────┐  │
│  │ memories table │  │ memory_vectors │  │
│  │ memory_stats   │  │ (embeddings)   │  │
│  └───────────────┘  └────────────────┘  │
└─────────────────────────────────────────┘
```

When using `--vector-backend pgvector`, both structured data and vector embeddings are consolidated into a single PostgreSQL database. No SQLite or ChromaDB files are used in this mode.

### Embedding model

| Model | Dimensions | Max Tokens | Params |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` (default) | 384 | 512 | 33M |

Configurable via `MEMORY_EMBEDDING_MODEL` env var. See `memory_mcp/config.py` for all presets.

---

## Running the Memory MCP

### Option 1: Stdio Transport (default -- for LM Studio and Desktop Clients)

```bash
# Default: ChromaDB + SQLite backend
python server.py

# With pgvector backend
python server.py --vector-backend pgvector
```

MCP client config (e.g. LM Studio `mcp.json`, Claude Desktop, Cursor, OpenCode):

```json
{
  "mcpServers": {
    "long_term_memory": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

With pgvector:

```json
{
  "mcpServers": {
    "long_term_memory": {
      "command": "python",
      "args": [
        "/path/to/server.py",
        "--vector-backend", "pgvector",
        "--pg-host", "localhost",
        "--pg-port", "5433"
      ]
    }
  }
}
```

### Option 2: HTTP Transport (for multiple agents and network access)

```bash
# ChromaDB backend over HTTP
python server.py --transport http

# Custom host/port
python server.py --transport http --host 0.0.0.0 --port 3000

# pgvector over HTTP
python server.py --transport http --vector-backend pgvector
```

HTTP client config:

```json
{
  "mcpServers": {
    "long_term_memory": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

See `mcp-config-examples.json` for more configuration examples.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--transport` | `stdio` | Transport protocol: `stdio` or `http` |
| `--host` | `0.0.0.0` | HTTP bind address |
| `--port` | `8000` | HTTP port |
| `--path` | `/mcp/` | HTTP URL path |
| `--vector-backend` | `chromadb` | Vector storage: `chromadb` or `pgvector` |
| `--pg-host` | `PGHOST` or `localhost` | PostgreSQL host |
| `--pg-port` | `PGPORT` or `5433` | PostgreSQL port |
| `--pg-database` | `PGDATABASE` or `memories` | PostgreSQL database name |
| `--pg-user` | `PGUSER` or `memory_user` | PostgreSQL user |
| `--pg-password` | `PGPASSWORD` or `memory_pass` | PostgreSQL password |

All `--pg-*` arguments fall back to the corresponding `PG*` environment variables.

---

## pgvector Backend Setup

### 1. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

This starts a `pgvector/pgvector:pg16` container on **port 5433** (mapped from 5432 inside the container), with database `memories`, user `memory_user`, password `memory_pass`.

### 2. Run the server with pgvector

```bash
python server.py --vector-backend pgvector
```

No extra flags needed -- the defaults match `docker-compose.yml`.

### 3. Auto-migration from ChromaDB

On the first run with `--vector-backend pgvector`, if the pgvector table is empty and a local ChromaDB database exists, the server **automatically migrates** all vectors by re-embedding from the active database backend. This is a one-time operation.

### Custom Postgres connection

```bash
python server.py --vector-backend pgvector \
    --pg-host myserver.example.com \
    --pg-port 5433 \
    --pg-database memories \
    --pg-user memory_user \
    --pg-password secret
```

Or via environment variables:

```bash
export PGHOST=myserver.example.com
export PGPORT=5433
export PGDATABASE=memories
export PGUSER=memory_user
export PGPASSWORD=secret
python server.py --vector-backend pgvector
```

---

## Desktop GUI

`memory_manager_gui.py` is a tkinter-based desktop application for managing your memory database.

```bash
python memory_manager_gui.py
```

### Features

- **Memory Browser**: browse, search, edit, and delete memories
- **Dashboard**: statistics and type breakdown
- **ChromaDB Viewer**: inspect raw vector embeddings (works with both ChromaDB and pgvector)
- **Data source selector**: switch between SQLite/ChromaDB and pgvector/Postgres at runtime
  - Dropdown in the header bar to select the active data source
  - Postgres connection settings bar (host, port, database, user, password) with a Connect button
  - All tabs refresh when switching sources
- **Migration tool**: bidirectional migration between backends
  - SQLite/ChromaDB -> pgvector/Postgres
  - pgvector/Postgres -> SQLite/ChromaDB
  - SQLite -> SQLite (legacy import from another DB file)
  - Migrates both structured data and vectors with progress bars
  - Preview and skip-duplicates options
- **Compare/Diff window**: side-by-side database comparison
  - Shows memories only in SQLite (green), only in Postgres (blue), modified (yellow), identical (grey)
  - Filterable by category
  - Includes vector count comparison
- **Backup and export** to JSON

---

## How Memory Works

- **Cross-Chats**: start a new chat -- memories are still there
- **Cross-Models**: switch models -- the same memory remains available
- **Cross-Machines**: copy the database folder (`memory_db/` and `memory_backups/`) and your system prompt, point to the path, and everything carries over

### Think of it as your AI's diary: chats are conversations, the database is the journal.

## Environment variable for custom data dir

**Windows PowerShell:**
```
$env:AI_COMPANION_DATA_DIR="D:\a.i. apps\long_term_memory_mcp\data"
```

**Linux/macOS:**
```
export AI_COMPANION_DATA_DIR="/home/username/ai_companion_data"
```

---

## Backups

Backups are created automatically:
- Every 24 hours
- Or after 100 new memories (configurable)
- Stored in `memory_backups/` with timestamped folders
- Only the last 10 backups are kept

Each backup includes:
- SQLite DB copy (SQLite mode) or JSON-only export (pgvector mode)
- ChromaDB copy (ChromaDB mode only)
- JSON export of all memories (always, portable and future-proof)

---

## Recommended System Prompt

> You are an AI companion with long-term memory. Store facts naturally ("Got it, I'll remember that."). Recall them when asked in natural language. Never expose internal tool usage to the user. Use memory tools to remember, recall, and update information invisibly.

---

## MCP Tools Overview

Your `RobustMemory` MCP exposes tools that allow your AI companion to interact with its long-term memory. These tools are designed to be called internally by the AI model based on its system prompt.

#### 1. `remember`
Store a new memory (fact, conversation snippet, preference, event).
- `title` (string, required), `content` (string, required)
- `tags` (string, optional), `importance` (integer 1-10, default 5), `memory_type` (string, default "conversation")

#### 2. `search_memories`
Semantic search based on a natural language query.
- `query` (string, required), `limit` (integer, default 10)

#### 3. `search_by_type`
Retrieve memories by `memory_type` (e.g. "fact", "preference").
- `memory_type` (string, required), `limit` (integer, default 20)

#### 4. `search_by_tags`
Find memories by one or more tags.
- `tags` (string, required), `limit` (integer, default 20)

#### 5. `get_recent_memories`
Fetch the most recently stored memories.
- `limit` (integer, default 20)

#### 6. `update_memory`
Modify an existing memory by its ID.
- `memory_id` (string, required), `title`, `content`, `tags`, `importance` (all optional)

#### 7. `delete_memory`
Permanently remove a memory by its ID.
- `memory_id` (string, required)

#### 8. `get_memory_stats`
Retrieve statistics about the memory system (counts, sizes, backend info).

#### 9. `create_backup`
Manually trigger a full backup of the memory system.

#### 10. `search_by_date_range`
Search for memories within a date range.
- `date_from` (string, required, ISO format), `date_to` (string, optional), `limit` (integer, default 50)

---

## Tool Selection Logic

Your AI companion chooses memory tools automatically based on the conversation. The tools are never shown to the user.

| Trigger | Tool |
|---|---|
| User shares a new fact | `remember` |
| Free-form recall question | `search_memories` |
| Category request ("all preferences") | `search_by_type` |
| Tag-based request | `search_by_tags` |
| "What did we talk about recently?" | `get_recent_memories` |
| "Update my favorite color" | `update_memory` |
| "Forget my old phone number" | `delete_memory` |
| "Between Sept 10-15" | `search_by_date_range` |
| "How many memories?" | `get_memory_stats` |
| "Back everything up" | `create_backup` |

---

## What's New

**Pluggable Backend Architecture (Latest)**
- pgvector support: use PostgreSQL + pgvector as an alternative to ChromaDB
- Database backend abstraction: `DatabaseBackend` ABC with SQLite and PostgreSQL implementations
- Vector backend abstraction: `VectorBackend` ABC with ChromaDB and pgvector implementations
- Single-database mode: when using pgvector, structured data and vectors share one Postgres database
- Docker Compose for easy pgvector setup (`docker compose up -d`)
- Auto-migration from ChromaDB to pgvector on first run
- CLI arguments for backend selection and Postgres connection settings
- Optional `[pgvector]` install extra in `pyproject.toml`

**Desktop GUI Enhancements**
- Data source selector: switch between SQLite/ChromaDB and pgvector/Postgres
- Bidirectional migration tool (SQLite <-> Postgres) with progress bars
- Database comparison/diff window (side-by-side view)
- Vector viewer supports both ChromaDB and pgvector backends

**Modular Architecture Refactoring**
- Refactored monolithic file into clean modular `memory_mcp/` package
- Separate `config.py`, `models.py`, `memory_system.py`, `mcp_tools.py`
- New `server.py` entry point
- Legacy `long_term_memory_mcp.py` remains fully functional

**Semantic Search Improvements**
- Distance-to-similarity fix: relevance = 1.0 - distance
- Adaptive threshold: follows top match (clamped) to reduce noise
- Top-1 fallback: if nothing passes threshold, return the strongest candidate

**Human-like Memory Dynamics**
- Lazy Decay: exponential half-life per memory_type, floor protection, rate-limited writes
- Reinforcement: each retrieval accumulates +0.1; at +0.5 accumulation, writes back a +0.5 importance bump (capped at 10)

---

## Contributing

Pull requests welcome!

- Found a bug? Open an issue.
- Want to add features? Let's collaborate.

## License

MIT
