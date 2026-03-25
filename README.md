# Robust Long-Term Memory MCP

A persistent, human-like memory system for AI companions, powered by pluggable backends for both structured storage and vector search. Designed for decades-long use, seamless recall across sessions, and automatic backups -- making your AI companion feel like a continuous, living persona. Now with biological behavior: time-based lazy decay and reinforcement by use.

Detached fork from [Rotoslider](https://github.com/Rotoslider/long-term-memory-mcp)
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
- **3D Vector Visualizer** (`vector_visualizer.py`): interactive Plotly 3D scatter plot of memory embeddings with hover labels, PCA/t-SNE/UMAP
- **System Tray App** (`tray_app.py`): dock/taskbar icon to start, stop, and monitor the MCP server (macOS, Windows, Linux)
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
├── tray_app.py                      # System tray app (macOS/Windows/Linux)
├── vector_visualizer.py             # 3D vector space visualizer (Plotly/Dash)
├── tensorboard_visualizer.py        # TensorBoard Embedding Projector exporter
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
├── benchmark_embeddings.py          # Embedding model quality comparison
├── benchmark_token_usage.py         # Token cost analysis (file vs memory)
├── mcp-config-examples.json         # 15+ MCP client config examples
├── docker-compose.yml               # pgvector Docker service
├── pyproject.toml                   # Package config with optional extras
├── requirements.txt                 # Dependencies
├── opencode/
│   ├── plugin/
│   │   └── long-term-memory.ts      # OpenCode enforcement plugin
│   └── AGENTS.md                    # OpenCode global system prompt
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

### Sub-Agent Access (OpenCode)

OpenCode uses specialized sub-agents (explore, general, etc.) launched via the Task tool. By default, sub-agents cannot call MCP tools. To grant them access to the memory server, add an `agent` block to your `opencode.json`:

```json
{
  "mcp": {
    "long-term-memory": {
      "url": "http://localhost:8000/mcp/",
      "type": "remote",
      "enabled": true
    }
  },
  "agent": {
    "build": {
      "tools": {
        "long-term-memory_*": true
      }
    },
    "plan": {
      "tools": {
        "long-term-memory_*": true
      }
    },
    "general": {
      "tools": {
        "long-term-memory_*": true
      }
    },
    "explore": {
      "tools": {
        "long-term-memory_*": true
      }
    }
  }
}
```

The wildcard `long-term-memory_*` grants each sub-agent access to all tools exposed by the memory MCP server (`remember`, `search_memories`, `search_by_tags`, etc.). Without this, only the primary agent can call memory tools -- sub-agents launched via Task will not have access.

### Enforcement Plugin (OpenCode)

The repo ships with an OpenCode plugin (`opencode/plugin/long-term-memory.ts`) that actively enforces memory usage on every turn by hooking into OpenCode's lifecycle.

**What it enforces:**

| Hook | Behaviour |
|---|---|
| System prompt injection | Injects mandatory memory rules into every LLM call |
| Universal tool gate | Blocks ALL tools (bash, read, edit, write, etc.) until `get_recent_memories` + `search_by_tags` are called this turn |
| End-of-turn store gate | If files were edited but `remember` was not called, ALL tools are blocked at the start of the next turn until `remember` is called |
| Idle warning | Logs a warning when a turn ends with edits but no store |
| Compaction hook | Rewrites the compaction prompt so enforcement state and memory rules survive context compaction |

**Install:**

```bash
# macOS / Linux
cp opencode/plugin/long-term-memory.ts ~/.config/opencode/plugins/

# Windows
copy opencode\plugin\long-term-memory.ts %APPDATA%\opencode\plugins\
```

Install the OpenCode plugin SDK if not already installed:

```bash
cd ~/.config/opencode   # or %APPDATA%\opencode on Windows
bun add @opencode-ai/plugin
```

**How the gate works:**

The recall gate re-arms every turn. The model must call both of these in parallel as its first action before any other tool:

```
long-term-memory_get_recent_memories(limit=5, current_project="<project>")
long-term-memory_search_by_tags(tags="preference,project")
```

If any other tool is called first, the plugin throws an error and blocks until recall is done. Memory tools themselves are always allowed through. After a turn where files were edited without a `remember` call, all tools are blocked at the start of the next turn until `remember` is called.

### AGENTS.md (OpenCode)

`opencode/AGENTS.md` is a condensed system prompt that tells the model what to store, how to tag, and how to write sub-agent prompts. Copy it to your OpenCode global config directory:

```bash
# macOS / Linux
cp opencode/AGENTS.md ~/.config/opencode/AGENTS.md

# Windows
copy opencode\AGENTS.md %APPDATA%\opencode\AGENTS.md
```

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

## 3D Vector Visualizer

![alt text](image.png)

`vector_visualizer.py` renders your memory embeddings as an interactive 3D scatter plot using Plotly + Dash. High-dimensional vectors (384/768D) are projected to 3D via PCA, t-SNE, or UMAP. Launches a local web app (default `http://127.0.0.1:8050`) with hover-over labels showing each memory's title, type, importance, and ID.

### Install

```bash
pip install '.[visualizer]'
# or manually:
pip install plotly dash scikit-learn
# optional for UMAP reduction:
pip install umap-learn
```

### Usage

```bash
# Default: PCA reduction, coloured by memory_type
python vector_visualizer.py

# t-SNE (slower, best cluster separation)
python vector_visualizer.py --method tsne

# UMAP (fast, good local + global structure)
python vector_visualizer.py --method umap

# Colour by importance (red=low, green=high)
python vector_visualizer.py --colour-by importance

# Custom port
python vector_visualizer.py --port 3000

# pgvector backend
python vector_visualizer.py --vector-backend pgvector
```

### Features

- **Live data**: every page refresh re-fetches from the database -- always shows current state
- **Semantic search**: type a query and press Enter to embed it, project to 3D, and highlight matching memories with similarity lines
- **Configurable result limit**: number input (default 10, range 1-100) controls how many search matches to show
- **Click-to-expand word vectors**: click any memory point to extract its words, embed each one, and display them as diamond markers radiating from the parent -- click again to collapse
- **Query Words button**: batch-expand word vectors for all memories matching the current search query
- **Word Paths**: toggle thick starburst lines through words shared by multiple expanded memories, visually bridging shared vocabulary
- **Configurable min-shared threshold**: "Min" input next to Word Paths sets how many memories must share a word before paths are drawn (default 2)
- **Lines toggle**: show/hide all connection lines in one click
- **Camera preservation**: zoom, pan, and rotation are preserved across all updates (Plotly `uirevision` + `Patch()`)
- **Hover labels**: mouse over any point to see its memory title, type, importance, and ID
- **Dark theme**: dark background with subtle grid lines, dark-themed dropdowns and controls
- **Searchable tag dropdown**: type to search tags, select one or more to draw connection lines between vectors sharing those tags
- **Point size** scaled by importance (higher importance = larger point)
- **Colour legend** in the sidebar for memory_type mode; click to isolate/toggle types

### Colour Legend

**By memory_type** (default): blue=conversation, green=fact, amber=preference, red=event, purple=task, grey=ephemeral

**By importance**: red (1-3) -> yellow (4-6) -> green (7-10)

---

## System Tray App

`tray_app.py` adds a persistent dock/taskbar icon for managing the MCP server without a terminal window. Works on macOS (menu bar), Windows (system tray), and Linux (app indicator).

### Install

```bash
pip install '.[tray]'
# or manually:
pip install pystray Pillow
```

### Usage

```bash
# Launch tray icon (server must be started from the menu)
python tray_app.py

# Auto-start the server on launch
python tray_app.py --auto-start

# With custom transport/port
python tray_app.py --transport http --port 3000

# With pgvector backend
python tray_app.py --vector-backend pgvector
```

### Features

- **Status indicator**: green (running), red (stopped), amber (starting)
- **Start / Stop / Restart** the MCP server from the menu
- **Launch** the Memory Manager GUI, Vector Visualizer, or TensorBoard Projector
- **View logs**: separate menu items for server log and visualizer log
- **Backend forwarding**: `--vector-backend` and `--pg-*` args are forwarded to server and visualizer subprocesses
- **PID file management**: reattaches to orphaned server processes from prior tray crashes
- **Auto-start**: `--auto-start` flag starts the server immediately on tray launch
- Server runs as a detached subprocess (survives tray restarts)
- Graceful shutdown with WAL checkpoint on stop

### Menu

```
Status: Running (pid 12345)
──────────────────────────
Start Server
Stop Server
Restart Server
──────────────────────────
Open Memory Manager
Open Vector Visualizer
Open TensorBoard Projector
──────────────────────────
View Server Log
View Visualizer Log
──────────────────────────
Quit
```

---

## TensorBoard Embedding Projector

`tensorboard_visualizer.py` exports memory embeddings to TensorBoard's Embedding Projector format. TensorBoard provides interactive PCA, t-SNE, UMAP, and custom linear projections with nearest-neighbour search and metadata filtering -- all in-browser. No TensorFlow required.

### Install

```bash
pip install '.[tensorboard]'
# or manually:
pip install tensorboard
```

### Usage

```bash
# Export and launch TensorBoard (opens browser automatically)
python tensorboard_visualizer.py

# Export only (no server, no browser)
python tensorboard_visualizer.py --export-only

# pgvector backend
python tensorboard_visualizer.py --vector-backend pgvector

# Custom port / log directory
python tensorboard_visualizer.py --port 6007 --logdir ./my_logs
```

### Features

- Exports `tensors.tsv`, `metadata.tsv`, and `projector_config.pbtxt`
- Metadata columns: title, memory_type, importance, tags, timestamp, memory ID
- Interactive dimensionality reduction (PCA, t-SNE, UMAP) in the browser
- Nearest-neighbour search by point or query
- Colour and filter by any metadata column
- Supports both ChromaDB and pgvector backends

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

#### 11. `rebuild_vectors`
One-time repair: rebuild the vector index from structured storage. Use if semantic search stops working but structured queries are fine.

#### 12. `list_source_memories`
Preview memories from another database file before migrating.
- `source_db_path` (string, required), `limit` (integer, default 100)

#### 13. `migrate_memories`
Migrate memories (structured data + vectors) from a source database to the active database.
- `source_db_path` (string, required), `source_chroma_path` (string, optional -- auto-detected), `memory_ids` (string, optional -- comma-separated for selective migration), `skip_duplicates` (boolean, default true)

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
| Semantic search broken | `rebuild_vectors` |
| "Show me what's in the old database" | `list_source_memories` |
| "Import memories from my other DB" | `migrate_memories` |

---

## What's New

**OpenCode Enforcement Plugin (Latest)**
- `opencode/plugin/long-term-memory.ts`: OpenCode plugin enforcing memory recall before every tool call and memory storage after every file edit
  - Universal tool gate, end-of-turn store gate, system prompt injection, compaction hook, idle warning
- `opencode/AGENTS.md`: condensed system prompt covering what to store, tagging conventions, and sub-agent memory instructions

**3D Vector Visualizer, TensorBoard Projector & System Tray App**
- `vector_visualizer.py`: interactive Plotly + Dash 3D scatter plot of memory embeddings
  - PCA, t-SNE, and UMAP dimensionality reduction (384D -> 3D)
  - Semantic search with adaptive thresholding (same algorithm as MCP `search_memories`)
  - Click-to-expand word vectors with diamond markers and connector lines
  - Word Paths: thick starburst lines through words shared by multiple memories, with configurable min-shared threshold
  - Query Words batch-expand, Lines toggle, dark theme, camera preservation
  - Searchable tag dropdown, hover labels, colour by memory_type or importance
  - Works with both ChromaDB and pgvector backends
- `tensorboard_visualizer.py`: export embeddings to TensorBoard Embedding Projector
  - Interactive PCA, t-SNE, UMAP with nearest-neighbour search in-browser
  - Metadata columns for filtering and colouring (title, type, importance, tags)
  - Export-only mode for offline analysis
- `tray_app.py`: cross-platform system tray / dock icon (pystray + Pillow)
  - Start / stop / restart the MCP server from macOS menu bar or Windows taskbar
  - Launch GUI, Vector Visualizer, or TensorBoard Projector from the menu
  - Status indicator (green/red/amber), PID file recovery, log viewers
  - Backend args forwarded to all subprocesses
- New optional install extras: `pip install '.[visualizer]'`, `pip install '.[tray]'`, `pip install '.[tensorboard]'`
- MCP tools: `rebuild_vectors`, `list_source_memories`, `migrate_memories`

**Pluggable Backend Architecture**
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
