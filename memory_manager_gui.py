"""
Memory Manager - GUI Application
A beautiful, easy-to-use interface for managing AI companion memories

Features:
- Search by all fields (title, content, tags, type, date, importance)
- View, edit, and delete memories
- Backup and restore functionality
- Statistics dashboard
- Export/import capabilities
- Corruption-safe database operations
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from pathlib import Path
import sqlite3
import json
import shutil
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import os

# Token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Configuration — import from shared config to stay in sync with the MCP server
try:
    from memory_mcp.config import (
        DATA_FOLDER,
        EMBEDDING_MODEL,
        EMBEDDING_MODEL_CONFIG,
        EMBEDDING_MODEL_PRESETS,
    )
except ImportError as _cfg_import_err:
    # Fallback if memory_mcp package isn't on the path
    import warnings

    warnings.warn(
        f"Could not import memory_mcp.config: {_cfg_import_err} — GUI running "
        "with degraded embedding info. Install the memory_mcp package or ensure "
        "it is on sys.path for full functionality.",
        stacklevel=2,
    )
    DATA_FOLDER = Path(
        os.environ.get(
            "AI_COMPANION_DATA_DIR",
            str(Path.home() / "Documents" / "ai_companion_memory"),
        )
    )
    EMBEDDING_MODEL = "unknown"
    EMBEDDING_MODEL_CONFIG = {
        "model_name": "unknown",
        "dimensions": None,
        "max_tokens": None,
        "query_prefix": "",
        "description": "Could not load memory_mcp.config",
    }
    EMBEDDING_MODEL_PRESETS = {}

DB_PATH = DATA_FOLDER / "memory_db" / "memories.db"
CHROMA_DB_PATH = DATA_FOLDER / "memory_db" / "chroma_db" / "chroma.sqlite3"
BACKUP_FOLDER = DATA_FOLDER / "memory_backups"


class MemoryManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Memory Manager")
        self.root.geometry("1920x1080")

        # Set modern theme
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Custom dark theme colors
        self.bg_color = "#1e1e1e"  # Main background (darker)
        self.fg_color = "#e0e0e0"  # Main text color
        self.accent_color = "#4a9eff"  # Accent blue
        self.secondary_bg = "#2d2d2d"  # Secondary backgrounds
        self.input_bg = "#252525"  # Input fields background
        self.hover_bg = "#3a3a3a"  # Hover state
        self.border_color = "#3e3e3e"  # Border color
        self.selected_bg = "#264f78"  # Selected item background
        self.button_bg = "#0e639c"  # Button background

        # Apply dark theme to root window
        self.root.configure(bg=self.bg_color)

        # Configure styles
        self.configure_styles()

        # Database connection
        self.db_conn = None
        self.connect_database()

        # ChromaDB connection
        self.chroma_conn = None
        self.connect_chromadb()

        # pgvector/Postgres connection (used when data source is "pgvector")
        self.pg_conn = None
        self.data_source = "sqlite"  # "sqlite" or "pgvector"

        # Postgres connection settings (populated from env vars / GUI fields)
        self._pg_host = os.environ.get("PGHOST", "localhost")
        self._pg_port = os.environ.get("PGPORT", "5433")
        self._pg_database = os.environ.get("PGDATABASE", "memories")
        self._pg_user = os.environ.get("PGUSER", "memory_user")
        self._pg_password = os.environ.get("PGPASSWORD", "")

        # Initialize RobustMemorySystem for write operations (keeps SQLite + ChromaDB in sync)
        self.memory_system = None
        self.init_memory_system()

        # Initialize tokenizer
        self.tokenizer = None
        self.init_tokenizer()

        # Current selection
        self.selected_memory_id = None

        # Sorting state
        self.sort_column = None
        self.sort_reverse = False

        # Build UI
        self.create_widgets()

        # Load initial data
        self.refresh_memories()
        self.update_statistics()

    def configure_styles(self):
        """Configure comprehensive dark theme for all widgets"""
        # Frame styles
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure(
            "TLabelframe",
            background=self.bg_color,
            bordercolor=self.border_color,
            foreground=self.fg_color,
        )
        self.style.configure(
            "TLabelframe.Label",
            background=self.bg_color,
            foreground=self.accent_color,
            font=("Segoe UI", 10, "bold"),
        )

        # Label styles
        self.style.configure(
            "Title.TLabel",
            font=("Segoe UI", 24, "bold"),
            foreground=self.accent_color,
            background=self.bg_color,
        )
        self.style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 12),
            foreground="#999999",
            background=self.bg_color,
        )
        self.style.configure(
            "Header.TLabel",
            font=("Segoe UI", 11, "bold"),
            foreground=self.fg_color,
            background=self.bg_color,
        )
        self.style.configure(
            "Normal.TLabel",
            font=("Segoe UI", 10),
            foreground=self.fg_color,
            background=self.bg_color,
        )

        # Button styles
        self.style.configure(
            "TButton",
            background=self.secondary_bg,
            foreground=self.fg_color,
            bordercolor=self.border_color,
            darkcolor=self.secondary_bg,
            lightcolor=self.hover_bg,
            font=("Segoe UI", 10),
        )
        self.style.map(
            "TButton",
            background=[("active", self.hover_bg), ("pressed", self.button_bg)],
            foreground=[("active", self.fg_color)],
        )

        self.style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.button_bg,
            foreground="#ffffff",
        )
        self.style.map(
            "Accent.TButton", background=[("active", "#1177bb"), ("pressed", "#0d5a8f")]
        )

        # Entry styles
        self.style.configure(
            "TEntry",
            fieldbackground=self.input_bg,
            background=self.input_bg,
            foreground=self.fg_color,
            bordercolor=self.border_color,
            insertcolor=self.fg_color,
        )

        # Combobox styles
        self.style.configure(
            "TCombobox",
            fieldbackground=self.input_bg,
            background=self.secondary_bg,
            foreground=self.fg_color,
            bordercolor=self.border_color,
            arrowcolor=self.fg_color,
            selectbackground=self.selected_bg,
            selectforeground=self.fg_color,
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.input_bg)],
            selectbackground=[("readonly", self.selected_bg)],
        )

        # Treeview styles
        self.style.configure(
            "Treeview",
            background=self.secondary_bg,
            foreground=self.fg_color,
            fieldbackground=self.secondary_bg,
            bordercolor=self.border_color,
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Treeview.Heading",
            background=self.input_bg,
            foreground=self.fg_color,
            bordercolor=self.border_color,
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map(
            "Treeview",
            background=[("selected", self.selected_bg)],
            foreground=[("selected", "#ffffff")],
        )
        self.style.map("Treeview.Heading", background=[("active", self.hover_bg)])

        # Scrollbar styles
        self.style.configure(
            "Vertical.TScrollbar",
            background=self.secondary_bg,
            troughcolor=self.bg_color,
            bordercolor=self.border_color,
            arrowcolor=self.fg_color,
        )
        self.style.configure(
            "Horizontal.TScrollbar",
            background=self.secondary_bg,
            troughcolor=self.bg_color,
            bordercolor=self.border_color,
            arrowcolor=self.fg_color,
        )

    def connect_database(self):
        """Connect to the SQLite database with corruption protection"""
        try:
            if not DB_PATH.exists():
                messagebox.showerror(
                    "Error",
                    f"Database not found at:\n{DB_PATH}\n\nPlease ensure the memory system is initialized.",
                )
                self.root.quit()
                return

            self.db_conn = sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False,
                timeout=30.0,
                isolation_level="DEFERRED",
            )
            self.db_conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency
            self.db_conn.execute("PRAGMA journal_mode=WAL")
            self.db_conn.execute("PRAGMA synchronous=FULL")

            # Verify database integrity
            cursor = self.db_conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            if result and result[0] != "ok":
                messagebox.showwarning(
                    "Warning",
                    "Database integrity check failed. Some data may be corrupted.",
                )

        except Exception as e:
            messagebox.showerror(
                "Database Error", f"Failed to connect to database:\n{str(e)}"
            )
            self.root.quit()

    def connect_chromadb(self):
        """Connect to ChromaDB SQLite database"""
        try:
            if not CHROMA_DB_PATH.exists():
                # ChromaDB is optional - warn but don't fail
                print(f"ChromaDB not found at: {CHROMA_DB_PATH}")
                print("Vector visualization will not be available.")
                self.chroma_conn = None
                return

            self.chroma_conn = sqlite3.connect(
                str(CHROMA_DB_PATH),
                check_same_thread=False,
                timeout=30.0,
                isolation_level="DEFERRED",
            )
            self.chroma_conn.row_factory = sqlite3.Row

        except Exception as e:
            print(f"Failed to connect to ChromaDB: {e}")
            print("Vector visualization will not be available.")
            self.chroma_conn = None

    def init_tokenizer(self):
        """Initialize tiktoken tokenizer for token counting"""
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(f"Failed to initialize tiktoken: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

    def init_memory_system(self):
        """Initialize RobustMemorySystem for write operations (keeps SQLite + ChromaDB in sync)"""
        try:
            from memory_mcp.memory_system import RobustMemorySystem
            from memory_mcp.config import DATA_FOLDER as MCP_DATA_FOLDER

            self.memory_system = RobustMemorySystem(MCP_DATA_FOLDER)
            print("RobustMemorySystem initialized for GUI write operations")
        except Exception as e:
            print(f"Failed to initialize RobustMemorySystem: {e}")
            print(
                "Write operations will fall back to raw SQLite (ChromaDB may get out of sync)"
            )
            self.memory_system = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

    # ── Data source abstraction ────────────────────────────────

    def _get_pg_gui_connkw(self) -> dict:
        """Build psycopg keyword-arg dict from the GUI's Postgres settings."""
        port_str = self._pg_port
        try:
            port = int(port_str)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid port number: '{port_str}'. Enter a numeric port (e.g. 5433)."
            )
        return dict(
            host=self._pg_host,
            port=port,
            dbname=self._pg_database,
            user=self._pg_user,
            password=self._pg_password,
        )

    def _connect_pgvector(self) -> bool:
        """Establish a Postgres connection for GUI reads. Returns True on success."""
        try:
            import psycopg
            from psycopg.rows import dict_row

            if self.pg_conn is not None:
                try:
                    self.pg_conn.close()
                except Exception:
                    pass

            self.pg_conn = psycopg.connect(
                **self._get_pg_gui_connkw(),
                autocommit=True,
                row_factory=dict_row,
            )
            return True
        except ImportError:
            messagebox.showerror(
                "Missing Dependency",
                "psycopg not installed.\nRun: pip install 'psycopg[binary]'",
            )
            return False
        except Exception as e:
            messagebox.showerror(
                "Connection Error", f"Failed to connect to Postgres:\n{e}"
            )
            return False

    def _disconnect_pgvector(self):
        """Close the Postgres GUI connection."""
        if self.pg_conn is not None:
            try:
                self.pg_conn.close()
            except Exception:
                pass
            self.pg_conn = None

    def _execute_read(self, sql: str, params: tuple = ()):
        """Execute a read query against the active data source.

        Handles placeholder translation (? -> %s) for Postgres.
        Returns a list of dict-like row objects.
        """
        if self.data_source == "pgvector" and self.pg_conn is not None:
            pg_sql = sql.replace("?", "%s")
            cur = self.pg_conn.execute(pg_sql, params)
            return cur.fetchall()
        else:
            cursor = self.db_conn.execute(sql, params)
            return cursor.fetchall()

    def _execute_read_one(self, sql: str, params: tuple = ()):
        """Execute a read query and return the first row, or None."""
        if self.data_source == "pgvector" and self.pg_conn is not None:
            pg_sql = sql.replace("?", "%s")
            cur = self.pg_conn.execute(pg_sql, params)
            return cur.fetchone()
        else:
            cursor = self.db_conn.execute(sql, params)
            return cursor.fetchone()

    def _switch_data_source(self, new_source: str):
        """Switch the active data source and refresh all views."""
        if new_source == self.data_source:
            return

        if new_source == "pgvector":
            # Read Postgres settings from GUI fields if they exist
            if hasattr(self, "_pg_host_var"):
                self._pg_host = self._pg_host_var.get().strip()
                self._pg_port = self._pg_port_var.get().strip()
                self._pg_database = self._pg_db_var.get().strip()
                self._pg_user = self._pg_user_var.get().strip()
                self._pg_password = self._pg_pass_var.get()

            if not self._connect_pgvector():
                # Connection failed — revert dropdown
                if hasattr(self, "_source_var"):
                    self._source_var.set("SQLite / ChromaDB")
                return

            self.data_source = "pgvector"
            if hasattr(self, "_pg_settings_frame"):
                self._pg_settings_frame.grid()
        else:
            self._disconnect_pgvector()
            self.data_source = "sqlite"
            if hasattr(self, "_pg_settings_frame"):
                self._pg_settings_frame.grid_remove()

        # Refresh all views
        self.refresh_memories()
        self.update_statistics()

    def _on_source_changed(self, event=None):
        """Handle data source combobox selection change.

        When 'pgvector / Postgres' is selected, shows the connection settings bar
        but does NOT connect immediately — the user must click "Connect".
        When 'SQLite / ChromaDB' is selected, disconnects pgvector and switches back.
        """
        selected = self._source_var.get()
        if selected == "pgvector / Postgres":
            # Show PG settings bar; user clicks "Connect" to actually switch
            if hasattr(self, "_pg_settings_frame"):
                self._pg_settings_frame.grid()
        else:
            # Switching back to SQLite
            if hasattr(self, "_pg_settings_frame"):
                self._pg_settings_frame.grid_remove()
            if self.data_source != "sqlite":
                self._switch_data_source("sqlite")

    def show_compare_window(self):
        """Show the database comparison/diff window between SQLite and Postgres."""
        # Ensure we have settings from the GUI fields
        if hasattr(self, "_pg_host_var"):
            self._pg_host = self._pg_host_var.get().strip()
            self._pg_port = self._pg_port_var.get().strip()
            self._pg_database = self._pg_db_var.get().strip()
            self._pg_user = self._pg_user_var.get().strip()
            self._pg_password = self._pg_pass_var.get()

        self._build_compare_window()

    def _build_compare_window(self):
        """Build and display the database comparison/diff window.

        Shows a git-diff-style comparison between SQLite/ChromaDB (Side A)
        and pgvector/Postgres (Side B):
          - Only in A (green, left side)   — memories in SQLite but not Postgres
          - Only in B (green, right side)  — memories in Postgres but not SQLite
          - Modified (yellow)              — same ID but different content_hash
          - Identical (grey, collapsible)  — same ID and same content_hash
        """
        # ── Connect to both databases ──────────────────────────
        if not DB_PATH.exists():
            messagebox.showerror(
                "Error", f"Local SQLite database not found:\n{DB_PATH}"
            )
            return

        pg_conn = None
        try:
            import psycopg
            from psycopg.rows import dict_row

            pg_conn = psycopg.connect(
                **self._get_pg_gui_connkw(),
                autocommit=True,
                row_factory=dict_row,
            )
        except ImportError:
            messagebox.showerror(
                "Missing Dependency",
                "psycopg not installed.\nRun: pip install 'psycopg[binary]'",
            )
            return
        except Exception as e:
            messagebox.showerror(
                "Connection Error",
                f"Cannot connect to Postgres:\n{e}\n\n"
                "Configure connection settings in the header bar.",
            )
            return

        # ── Read data from both sides ──────────────────────────
        try:
            # Side A: SQLite
            src_a = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)
            src_a.row_factory = sqlite3.Row
            a_rows = src_a.execute(
                "SELECT id, title, memory_type, importance, content_hash FROM memories ORDER BY timestamp"
            ).fetchall()
            src_a.close()
            a_map = {row["id"]: dict(row) for row in a_rows}

            # Side B: Postgres
            cur = pg_conn.execute(
                "SELECT id, title, memory_type, importance, content_hash FROM memories ORDER BY timestamp"
            )
            b_rows = cur.fetchall()
            b_map = {row["id"]: dict(row) for row in b_rows}

            # Also get vector counts
            a_vec_count = 0
            b_vec_count = 0
            try:
                chroma_dir = DB_PATH.parent / "chroma_db"
                if chroma_dir.exists():
                    from memory_mcp.vector_backends.chroma import ChromaBackend

                    chroma = ChromaBackend(db_folder=DB_PATH.parent)
                    chroma.initialize()
                    a_vec_count = chroma.count()
                    chroma.close()
            except Exception:
                pass
            try:
                row = pg_conn.execute(
                    "SELECT COUNT(*) as cnt FROM memory_vectors"
                ).fetchone()
                b_vec_count = row["cnt"] if row else 0
            except Exception:
                pass

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read databases:\n{e}")
            return
        finally:
            if pg_conn:
                try:
                    pg_conn.close()
                except Exception:
                    pass

        # ── Classify memories ──────────────────────────────────
        all_ids = set(a_map.keys()) | set(b_map.keys())
        only_a = []  # in SQLite only
        only_b = []  # in Postgres only
        modified = []  # same ID, different content_hash
        identical = []  # same ID, same content_hash

        for mid in sorted(all_ids):
            in_a = mid in a_map
            in_b = mid in b_map
            if in_a and not in_b:
                only_a.append(a_map[mid])
            elif in_b and not in_a:
                only_b.append(b_map[mid])
            else:
                # Both sides have it — compare content_hash
                hash_a = a_map[mid].get("content_hash") or ""
                hash_b = b_map[mid].get("content_hash") or ""
                if hash_a and hash_b and hash_a == hash_b:
                    identical.append(a_map[mid])
                else:
                    # Different hashes, or one/both missing — treat as modified
                    modified.append((a_map[mid], b_map[mid]))

        # ── Build the comparison window ────────────────────────
        cmp_win = tk.Toplevel(self.root)
        cmp_win.title("Database Comparison — SQLite vs Postgres")
        cmp_win.geometry("1200x800")
        cmp_win.configure(bg=self.bg_color)
        cmp_win.transient(self.root)

        main_frame = ttk.Frame(cmp_win, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        cmp_win.columnconfigure(0, weight=1)
        cmp_win.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # ── Summary bar ────────────────────────────────────────
        summary_frame = ttk.Frame(main_frame)
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(
            summary_frame,
            text=f"SQLite: {len(a_map)} memories, {a_vec_count} vectors",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 30))

        ttk.Label(
            summary_frame,
            text=f"Postgres: {len(b_map)} memories, {b_vec_count} vectors",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=1, sticky=tk.W, padx=(0, 30))

        ttk.Label(
            summary_frame,
            text=(
                f"Only in SQLite: {len(only_a)}  |  "
                f"Only in Postgres: {len(only_b)}  |  "
                f"Modified: {len(modified)}  |  "
                f"Identical: {len(identical)}"
            ),
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        # ── Filter checkboxes ──────────────────────────────────
        filter_frame = ttk.Frame(main_frame)
        filter_frame.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

        show_only_a = tk.BooleanVar(value=True)
        show_only_b = tk.BooleanVar(value=True)
        show_modified = tk.BooleanVar(value=True)
        show_identical = tk.BooleanVar(value=False)

        def _refresh_tree():
            for item in tree.get_children():
                tree.delete(item)
            _populate_tree()

        ttk.Checkbutton(
            filter_frame,
            text=f"Only in SQLite ({len(only_a)})",
            variable=show_only_a,
            command=_refresh_tree,
        ).grid(row=0, column=0, padx=(0, 15))
        ttk.Checkbutton(
            filter_frame,
            text=f"Only in Postgres ({len(only_b)})",
            variable=show_only_b,
            command=_refresh_tree,
        ).grid(row=0, column=1, padx=(0, 15))
        ttk.Checkbutton(
            filter_frame,
            text=f"Modified ({len(modified)})",
            variable=show_modified,
            command=_refresh_tree,
        ).grid(row=0, column=2, padx=(0, 15))
        ttk.Checkbutton(
            filter_frame,
            text=f"Identical ({len(identical)})",
            variable=show_identical,
            command=_refresh_tree,
        ).grid(row=0, column=3, padx=(0, 15))

        # ── Diff treeview ──────────────────────────────────────
        tree_frame = ttk.Frame(main_frame)
        tree_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        tree = ttk.Treeview(
            tree_frame,
            columns=("Status", "ID", "Title", "Type", "Importance", "Hash_A", "Hash_B"),
            show="headings",
            height=25,
        )
        for col, w in [
            ("Status", 130),
            ("ID", 120),
            ("Title", 350),
            ("Type", 100),
            ("Importance", 80),
            ("Hash_A", 120),
            ("Hash_B", 120),
        ]:
            tree.heading(col, text=col)
            tree.column(col, width=w)
        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        tree.configure(yscrollcommand=tree_scroll.set)

        # Tag colors for diff categories
        tree.tag_configure("only_a", foreground="#66bb6a")  # green
        tree.tag_configure("only_b", foreground="#42a5f5")  # blue
        tree.tag_configure("modified", foreground="#ffa726")  # yellow/orange
        tree.tag_configure("identical", foreground="#9e9e9e")  # grey

        def _populate_tree():
            if show_only_a.get():
                for r in only_a:
                    tree.insert(
                        "",
                        tk.END,
                        values=(
                            "Only in SQLite",
                            r["id"][:15],
                            str(r.get("title", ""))[:50],
                            r.get("memory_type", ""),
                            r.get("importance", ""),
                            str(r.get("content_hash", ""))[:12],
                            "",
                        ),
                        tags=("only_a",),
                    )

            if show_only_b.get():
                for r in only_b:
                    tree.insert(
                        "",
                        tk.END,
                        values=(
                            "Only in Postgres",
                            r["id"][:15],
                            str(r.get("title", ""))[:50],
                            r.get("memory_type", ""),
                            r.get("importance", ""),
                            "",
                            str(r.get("content_hash", ""))[:12],
                        ),
                        tags=("only_b",),
                    )

            if show_modified.get():
                for ra, rb in modified:
                    tree.insert(
                        "",
                        tk.END,
                        values=(
                            "Modified",
                            ra["id"][:15],
                            str(ra.get("title", ""))[:50],
                            ra.get("memory_type", ""),
                            ra.get("importance", ""),
                            str(ra.get("content_hash", ""))[:12],
                            str(rb.get("content_hash", ""))[:12],
                        ),
                        tags=("modified",),
                    )

            if show_identical.get():
                for r in identical:
                    tree.insert(
                        "",
                        tk.END,
                        values=(
                            "Identical",
                            r["id"][:15],
                            str(r.get("title", ""))[:50],
                            r.get("memory_type", ""),
                            r.get("importance", ""),
                            str(r.get("content_hash", ""))[:12],
                            str(r.get("content_hash", ""))[:12],
                        ),
                        tags=("identical",),
                    )

        _populate_tree()

        # ── Close button ───────────────────────────────────────
        ttk.Button(main_frame, text="Close", command=cmp_win.destroy).grid(
            row=3, column=0, sticky=tk.E, pady=(10, 0)
        )

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ===== HEADER =====
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5)
        )
        header_frame.columnconfigure(1, weight=1)

        title_label = ttk.Label(
            header_frame, text="Memory Manager", style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, sticky=tk.W)

        subtitle_label = ttk.Label(
            header_frame,
            text="View and manage AI companion memories",
            style="Subtitle.TLabel",
        )
        subtitle_label.grid(row=1, column=0, sticky=tk.W)

        # ── Data source selector (right side of header) ────────
        source_frame = ttk.Frame(header_frame)
        source_frame.grid(row=0, column=1, rowspan=2, sticky=tk.E, padx=(20, 0))

        ttk.Label(source_frame, text="Data Source:", style="Normal.TLabel").grid(
            row=0, column=0, sticky=tk.E, padx=(0, 5)
        )

        self._source_var = tk.StringVar(value="SQLite / ChromaDB")
        source_combo = ttk.Combobox(
            source_frame,
            textvariable=self._source_var,
            state="readonly",
            values=["SQLite / ChromaDB", "pgvector / Postgres"],
            width=22,
            font=("Segoe UI", 10),
        )
        source_combo.grid(row=0, column=1, padx=(0, 5))
        source_combo.bind("<<ComboboxSelected>>", self._on_source_changed)

        # Compare button
        ttk.Button(source_frame, text="Compare", command=self.show_compare_window).grid(
            row=0, column=2, padx=(5, 0)
        )

        # ── Postgres connection settings bar (hidden by default) ──
        self._pg_settings_frame = ttk.LabelFrame(
            main_frame, text="PostgreSQL Connection", padding="5"
        )
        # Will be shown/hidden by _on_source_changed
        self._pg_settings_frame.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5)
        )
        self._pg_settings_frame.grid_remove()  # hidden initially

        self._pg_host_var = tk.StringVar(value=self._pg_host)
        self._pg_port_var = tk.StringVar(value=self._pg_port)
        self._pg_db_var = tk.StringVar(value=self._pg_database)
        self._pg_user_var = tk.StringVar(value=self._pg_user)
        self._pg_pass_var = tk.StringVar(value=self._pg_password)

        ttk.Label(self._pg_settings_frame, text="Host:").grid(row=0, column=0, padx=2)
        ttk.Entry(
            self._pg_settings_frame, textvariable=self._pg_host_var, width=15
        ).grid(row=0, column=1, padx=2)

        ttk.Label(self._pg_settings_frame, text="Port:").grid(row=0, column=2, padx=2)
        ttk.Entry(
            self._pg_settings_frame, textvariable=self._pg_port_var, width=6
        ).grid(row=0, column=3, padx=2)

        ttk.Label(self._pg_settings_frame, text="Database:").grid(
            row=0, column=4, padx=2
        )
        ttk.Entry(self._pg_settings_frame, textvariable=self._pg_db_var, width=12).grid(
            row=0, column=5, padx=2
        )

        ttk.Label(self._pg_settings_frame, text="User:").grid(row=0, column=6, padx=2)
        ttk.Entry(
            self._pg_settings_frame, textvariable=self._pg_user_var, width=12
        ).grid(row=0, column=7, padx=2)

        ttk.Label(self._pg_settings_frame, text="Password:").grid(
            row=0, column=8, padx=2
        )
        ttk.Entry(
            self._pg_settings_frame, textvariable=self._pg_pass_var, show="*", width=12
        ).grid(row=0, column=9, padx=2)

        self._pg_status_var = tk.StringVar(value="")
        ttk.Button(
            self._pg_settings_frame,
            text="Connect",
            command=lambda: self._switch_data_source("pgvector"),
        ).grid(row=0, column=10, padx=(10, 2))
        ttk.Label(
            self._pg_settings_frame,
            textvariable=self._pg_status_var,
            font=("Segoe UI", 8),
        ).grid(row=0, column=11, padx=5)

        # Adjust main content to row=2 (was row=1) to make room for PG settings
        main_frame.rowconfigure(2, weight=1)

        # ===== LEFT PANEL - Search and List =====
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(2, weight=1)

        # Search section
        search_frame = ttk.LabelFrame(left_panel, text="Search Memories", padding="10")
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)

        # Search by text
        ttk.Label(search_frame, text="Search:", style="Normal.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self.on_search_changed())
        search_entry = ttk.Entry(
            search_frame, textvariable=self.search_var, font=("Segoe UI", 10)
        )
        search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))

        # Filter by type
        ttk.Label(search_frame, text="Type:", style="Normal.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.type_var = tk.StringVar(value="All")
        type_combo = ttk.Combobox(
            search_frame,
            textvariable=self.type_var,
            state="readonly",
            font=("Segoe UI", 10),
        )
        type_combo["values"] = [
            "All",
            "conversation",
            "fact",
            "preference",
            "event",
            "task",
            "ephemeral",
        ]
        type_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        type_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_memories())

        # Filter by importance
        ttk.Label(search_frame, text="Min Importance:", style="Normal.TLabel").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.importance_var = tk.StringVar(value="1")
        importance_spin = ttk.Spinbox(
            search_frame,
            from_=1,
            to=10,
            textvariable=self.importance_var,
            width=10,
            font=("Segoe UI", 10),
        )
        importance_spin.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        importance_spin.bind("<Return>", lambda e: self.refresh_memories())

        # Filter by tags
        ttk.Label(search_frame, text="Tags:", style="Normal.TLabel").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.tags_filter_var = tk.StringVar()
        tags_entry = ttk.Entry(
            search_frame, textvariable=self.tags_filter_var, font=("Segoe UI", 10)
        )
        tags_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        tags_entry.bind("<Return>", lambda e: self.refresh_memories())

        # Search button
        search_btn = ttk.Button(
            search_frame,
            text="Search",
            command=self.refresh_memories,
            style="Accent.TButton",
        )
        search_btn.grid(row=4, column=0, columnspan=2, pady=(10, 0))

        # Statistics section
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.stats_label = ttk.Label(
            stats_frame, text="Loading...", style="Normal.TLabel", justify=tk.LEFT
        )
        self.stats_label.grid(row=0, column=0, sticky=tk.W)

        # Memory list
        list_frame = ttk.LabelFrame(left_panel, text="Memories", padding="5")
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Treeview for memories
        columns = ("ID", "Title", "Type", "Importance", "Shared", "Date")
        self.tree = ttk.Treeview(
            list_frame, columns=columns, show="tree headings", selectmode="browse"
        )

        # Configure columns
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("ID", width=0, stretch=tk.NO)
        self.tree.column("Title", width=300)
        self.tree.column("Type", width=100)
        self.tree.column("Importance", width=80, anchor=tk.CENTER)
        self.tree.column("Shared", width=60, anchor=tk.CENTER)
        self.tree.column("Date", width=150)

        # Configure headings
        self.tree.heading("Title", text="Title")
        self.tree.heading(
            "Type", text="Type", command=lambda: self.sort_by_column("Type")
        )
        self.tree.heading("Importance", text="Importance")
        self.tree.heading("Shared", text="Shared")
        self.tree.heading(
            "Date", text="Date", command=lambda: self.sort_by_column("Date")
        )

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure Treeview row colors for dark theme
        self.tree.tag_configure(
            "row", background=self.secondary_bg, foreground=self.fg_color
        )

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_memory_selected)

        # ===== RIGHT PANEL - Details and Actions =====
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

        # Action buttons
        action_frame = ttk.Frame(right_panel)
        action_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(action_frame, text="New Memory", command=self.new_memory).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(action_frame, text="Save Changes", command=self.save_memory).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(action_frame, text="Delete", command=self.delete_memory).grid(
            row=0, column=2, padx=2
        )
        ttk.Button(action_frame, text="Refresh", command=self.refresh_memories).grid(
            row=0, column=3, padx=2
        )
        ttk.Button(action_frame, text="Backup", command=self.create_backup).grid(
            row=0, column=4, padx=2
        )
        ttk.Button(action_frame, text="Export", command=self.export_memories).grid(
            row=0, column=5, padx=2
        )
        ttk.Button(action_frame, text="Vectors", command=self.show_vector_window).grid(
            row=0, column=6, padx=2
        )
        ttk.Button(
            action_frame, text="Migrate", command=self.show_migration_window
        ).grid(row=0, column=7, padx=2)

        # Details section
        details_frame = ttk.LabelFrame(right_panel, text="Memory Details", padding="10")
        details_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(1, weight=1)
        details_frame.rowconfigure(6, weight=1)

        # ID (hidden, for reference)
        self.id_var = tk.StringVar()

        # Title
        ttk.Label(details_frame, text="Title:", style="Header.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.title_var = tk.StringVar()
        title_entry = ttk.Entry(
            details_frame, textvariable=self.title_var, font=("Segoe UI", 11)
        )
        title_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))

        # Type
        ttk.Label(details_frame, text="Type:", style="Header.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.detail_type_var = tk.StringVar()
        type_detail_combo = ttk.Combobox(
            details_frame,
            textvariable=self.detail_type_var,
            state="readonly",
            font=("Segoe UI", 10),
        )
        type_detail_combo["values"] = [
            "conversation",
            "fact",
            "preference",
            "event",
            "task",
            "ephemeral",
        ]
        type_detail_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))

        # Importance
        ttk.Label(details_frame, text="Importance:", style="Header.TLabel").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.detail_importance_var = tk.StringVar()
        importance_detail_spin = ttk.Spinbox(
            details_frame,
            from_=1,
            to=10,
            textvariable=self.detail_importance_var,
            width=10,
            font=("Segoe UI", 10),
        )
        importance_detail_spin.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(10, 0))

        # Tags
        ttk.Label(details_frame, text="Tags:", style="Header.TLabel").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.tags_var = tk.StringVar()
        tags_detail_entry = ttk.Entry(
            details_frame, textvariable=self.tags_var, font=("Segoe UI", 10)
        )
        tags_detail_entry.grid(
            row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0)
        )
        ttk.Label(
            details_frame,
            text="(comma-separated)",
            font=("Segoe UI", 8),
            foreground="#888888",
        ).grid(row=4, column=1, sticky=tk.W, padx=(10, 0))

        # Shared
        ttk.Label(details_frame, text="Shared:", style="Header.TLabel").grid(
            row=5, column=0, sticky=tk.W, pady=5
        )
        self.shared_var = tk.BooleanVar(value=False)
        shared_check = ttk.Checkbutton(
            details_frame,
            variable=self.shared_var,
            text="Broadcast to LAN peers",
        )
        shared_check.grid(row=5, column=1, sticky=tk.W, pady=5, padx=(10, 0))

        # Content
        ttk.Label(details_frame, text="Content:", style="Header.TLabel").grid(
            row=6, column=0, sticky=(tk.W, tk.N), pady=5
        )
        self.content_text = scrolledtext.ScrolledText(
            details_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            height=15,
            bg=self.input_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            selectbackground=self.selected_bg,
            selectforeground="#ffffff",
            borderwidth=1,
            relief=tk.SOLID,
        )
        self.content_text.grid(
            row=6, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(10, 0)
        )

        # Metadata display
        metadata_frame = ttk.LabelFrame(right_panel, text="Metadata", padding="10")
        metadata_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        metadata_frame.columnconfigure(0, weight=1)

        self.metadata_label = ttk.Label(
            metadata_frame,
            text="Select a memory to view details",
            style="Normal.TLabel",
            justify=tk.LEFT,
        )
        self.metadata_label.grid(row=0, column=0, sticky=tk.W)

    def on_search_changed(self):
        """Handle search text changes with debouncing"""
        # Cancel previous scheduled search
        if hasattr(self, "_search_after_id"):
            self.root.after_cancel(self._search_after_id)

        # Schedule new search after 500ms
        self._search_after_id = self.root.after(500, self.refresh_memories)

    def sort_by_column(self, column: str):
        """Toggle sorting for a column"""
        # If clicking the same column, toggle direction
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False

        # Update column headers to show sort direction
        self.update_column_headers()

        # Refresh with new sort order
        self.refresh_memories()

    def update_column_headers(self):
        """Update column headers to show sort indicators"""
        # Reset all headers
        self.tree.heading("Type", text="Type")
        self.tree.heading("Date", text="Date")

        # Add sort indicator to active column
        if self.sort_column:
            indicator = " ▼" if self.sort_reverse else " ▲"
            self.tree.heading(self.sort_column, text=f"{self.sort_column}{indicator}")

    def refresh_memories(self):
        """Refresh the memory list based on current filters"""
        try:
            # Clear current items
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Build query
            conditions = []
            params = []

            # Search text (searches in title and content)
            search_text = self.search_var.get().strip()
            if search_text:
                conditions.append("(title LIKE ? OR content LIKE ?)")
                params.extend([f"%{search_text}%", f"%{search_text}%"])

            # Type filter
            if self.type_var.get() != "All":
                conditions.append("memory_type = ?")
                params.append(self.type_var.get())

            # Importance filter
            try:
                min_importance = int(self.importance_var.get())
                conditions.append("importance >= ?")
                params.append(min_importance)
            except:
                pass

            # Tags filter
            tags_filter = self.tags_filter_var.get().strip()
            if tags_filter:
                tag_list = [t.strip() for t in tags_filter.split(",") if t.strip()]
                if tag_list:
                    tag_conditions = []
                    for tag in tag_list:
                        tag_conditions.append("tags LIKE ?")
                        params.append(f'%"{tag}"%')
                    conditions.append(f"({' OR '.join(tag_conditions)})")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Build ORDER BY clause based on sort column
            order_by = "ORDER BY "
            if self.sort_column == "Type":
                order_by += f"memory_type {'DESC' if self.sort_reverse else 'ASC'}, importance DESC, timestamp DESC"
            elif self.sort_column == "Date":
                order_by += f"timestamp {'DESC' if self.sort_reverse else 'ASC'}, importance DESC"
            else:
                # Default sort: importance DESC, then timestamp DESC
                order_by += "importance DESC, timestamp DESC"

            query = f"""
                SELECT id, title, memory_type, importance, timestamp, tags, content, metadata, created_at, updated_at, last_accessed
                FROM memories
                WHERE {where_clause}
                {order_by}
                LIMIT 1000
            """

            rows = self._execute_read(query, tuple(params))

            # Populate tree
            for row in rows:
                # Format date
                try:
                    dt = datetime.fromisoformat(row["timestamp"])
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = row["timestamp"][:16] if row["timestamp"] else "Unknown"

                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        row["id"],
                        row["title"][:50] + ("..." if len(row["title"]) > 50 else ""),
                        row["memory_type"],
                        row["importance"],
                        "✓" if (row["shared"] if "shared" in row.keys() else 0) else "",
                        date_str,
                    ),
                    tags=("row",),
                )

            # Update status
            self.update_statistics()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh memories:\n{str(e)}")

    def on_memory_selected(self, event):
        """Handle memory selection in the tree"""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        memory_id = item["values"][0]

        try:
            row = self._execute_read_one(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )

            if row:
                self.selected_memory_id = row["id"]
                self.id_var.set(row["id"])
                self.title_var.set(row["title"])
                self.detail_type_var.set(row["memory_type"])
                self.detail_importance_var.set(row["importance"])

                # Parse and display tags
                try:
                    tags = json.loads(row["tags"])
                    self.tags_var.set(", ".join(tags))
                except:
                    self.tags_var.set("")

                # Display content
                self.content_text.delete("1.0", tk.END)
                self.content_text.insert("1.0", row["content"])

                # Shared flag
                try:
                    shared_val = row["shared"] if "shared" in row.keys() else 0
                    self.shared_var.set(bool(shared_val))
                except Exception:
                    self.shared_var.set(False)

                # Display metadata
                try:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                except:
                    metadata = {}

                # Calculate token count for current content
                try:
                    token_count = (
                        row["token_count"]
                        if row["token_count"]
                        else self.count_tokens(row["content"])
                    )
                except (KeyError, TypeError):
                    token_count = self.count_tokens(row["content"])

                metadata_text = f"ID: {row['id']}\n"
                metadata_text += f"Token Count: {token_count:,}\n"
                metadata_text += f"Created: {row['created_at']}\n"
                metadata_text += f"Updated: {row['updated_at']}\n"
                metadata_text += f"Last Accessed: {row['last_accessed']}\n"

                if metadata:
                    metadata_text += f"\nCustom Metadata:\n"
                    for key, value in metadata.items():
                        metadata_text += f"  {key}: {value}\n"

                self.metadata_label.config(text=metadata_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load memory details:\n{str(e)}")

    def new_memory(self):
        """Create a new memory"""
        self.selected_memory_id = None
        self.id_var.set("")
        self.title_var.set("")
        self.detail_type_var.set("conversation")
        self.detail_importance_var.set("5")
        self.tags_var.set("")
        self.shared_var.set(False)
        self.content_text.delete("1.0", tk.END)
        self.metadata_label.config(text="New memory - fill in details and click Save")

    def save_memory(self):
        """Save the current memory (create or update), keeping SQLite and ChromaDB in sync"""
        try:
            title = self.title_var.get().strip()
            content = self.content_text.get("1.0", tk.END).strip()

            if not title or not content:
                messagebox.showwarning(
                    "Validation Error", "Title and content are required."
                )
                return

            memory_type = self.detail_type_var.get()
            importance = int(self.detail_importance_var.get())

            # Parse tags
            tags_str = self.tags_var.get().strip()
            tags = (
                [t.strip() for t in tags_str.split(",") if t.strip()]
                if tags_str
                else []
            )

            shared = self.shared_var.get()

            if self.data_source == "pgvector" and self.pg_conn is not None:
                # Write directly to Postgres — memory_system targets SQLite and
                # would silently persist to the wrong backend when pgvector mode
                # is active in the GUI.
                now_iso = datetime.now(timezone.utc).isoformat()
                try:
                    if self.selected_memory_id:
                        self.pg_conn.execute(
                            """
                            UPDATE memories
                            SET title = %s, content = %s, memory_type = %s,
                                importance = %s, tags = %s, updated_at = %s,
                                token_count = %s, shared = %s
                            WHERE id = %s
                            """,
                            (
                                title,
                                content,
                                memory_type,
                                importance,
                                json.dumps(tags),
                                now_iso,
                                self.count_tokens(content),
                                1 if shared else 0,
                                self.selected_memory_id,
                            ),
                        )
                        self.pg_conn.commit()
                        messagebox.showinfo("Success", "Memory updated in Postgres!")
                    else:
                        import hashlib

                        content_hash = hashlib.sha256(content.encode()).hexdigest()
                        time_hash = hashlib.sha256(now_iso.encode()).hexdigest()[:8]
                        memory_id = f"mem_{time_hash}_{content_hash[:16]}"
                        token_count = self.count_tokens(content)
                        self.pg_conn.execute(
                            """
                            INSERT INTO memories
                                (id, title, content, timestamp, tags, importance,
                                 memory_type, metadata, content_hash, created_at,
                                 updated_at, last_accessed, token_count, shared)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                memory_id,
                                title,
                                content,
                                now_iso,
                                json.dumps(tags),
                                importance,
                                memory_type,
                                "{}",
                                content_hash,
                                now_iso,
                                now_iso,
                                now_iso,
                                token_count,
                                1 if shared else 0,
                            ),
                        )
                        self.pg_conn.commit()
                        self.selected_memory_id = memory_id
                        messagebox.showinfo(
                            "Success",
                            "Memory created in Postgres (vectors not embedded — "
                            "run Rebuild Vectors to add to vector index).",
                        )
                except Exception as pg_err:
                    try:
                        self.pg_conn.rollback()
                    except Exception:
                        pass
                    messagebox.showerror("Error", f"Postgres save failed:\n{pg_err}")
                    return

            elif self.memory_system:
                # Use RobustMemorySystem to keep SQLite + ChromaDB in sync
                if self.selected_memory_id:
                    # Update existing memory
                    result = self.memory_system.update_memory(
                        memory_id=self.selected_memory_id,
                        title=title,
                        content=content,
                        tags=tags,
                        importance=importance,
                        memory_type=memory_type,
                        shared=shared,
                    )
                    if result.success:
                        messagebox.showinfo("Success", "Memory updated successfully!")
                    else:
                        messagebox.showerror(
                            "Error", f"Failed to update memory:\n{result.reason}"
                        )
                        return
                else:
                    # Create new memory
                    result = self.memory_system.remember(
                        title=title,
                        content=content,
                        tags=tags,
                        importance=importance,
                        memory_type=memory_type,
                        shared=shared,
                    )
                    if result.success:
                        self.selected_memory_id = result.data[0]["id"]
                        messagebox.showinfo("Success", "Memory created successfully!")
                    else:
                        messagebox.showerror(
                            "Error", f"Failed to create memory:\n{result.reason}"
                        )
                        return
            else:
                # Fallback: raw SQLite (no ChromaDB sync)
                token_count = self.count_tokens(content)
                now_iso = datetime.now(timezone.utc).isoformat()

                if self.selected_memory_id:
                    self.db_conn.execute(
                        """
                        UPDATE memories
                        SET title = ?, content = ?, memory_type = ?, importance = ?, tags = ?, updated_at = ?, token_count = ?, shared = ?
                        WHERE id = ?
                    """,
                        (
                            title,
                            content,
                            memory_type,
                            importance,
                            json.dumps(tags),
                            now_iso,
                            token_count,
                            1 if shared else 0,
                            self.selected_memory_id,
                        ),
                    )
                    self.db_conn.commit()
                    messagebox.showinfo(
                        "Success",
                        "Memory updated (SQLite only - RobustMemorySystem unavailable).",
                    )
                else:
                    import hashlib

                    content_hash = hashlib.sha256(content.encode()).hexdigest()
                    time_hash = hashlib.sha256(now_iso.encode()).hexdigest()[:8]
                    memory_id = f"mem_{time_hash}_{content_hash[:16]}"

                    self.db_conn.execute(
                        """
                        INSERT INTO memories (id, title, content, timestamp, tags, importance, memory_type, metadata, content_hash, created_at, updated_at, last_accessed, token_count, shared)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            memory_id,
                            title,
                            content,
                            now_iso,
                            json.dumps(tags),
                            importance,
                            memory_type,
                            "{}",
                            content_hash,
                            now_iso,
                            now_iso,
                            now_iso,
                            token_count,
                            1 if shared else 0,
                        ),
                    )
                    self.db_conn.commit()
                    self.selected_memory_id = memory_id
                    messagebox.showinfo(
                        "Success",
                        "Memory created (SQLite only - run rebuild_vectors to sync ChromaDB).",
                    )

            self.refresh_memories()

        except Exception as e:
            if self.db_conn:
                try:
                    self.db_conn.rollback()
                except Exception:
                    pass
            messagebox.showerror("Error", f"Failed to save memory:\n{str(e)}")

    def delete_memory(self):
        """Delete the selected memory from the active data source."""
        if not self.selected_memory_id:
            messagebox.showwarning("No Selection", "Please select a memory to delete.")
            return

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete this memory?\n\nTitle: {self.title_var.get()}\n\nThis action cannot be undone.",
        )

        if confirm:
            try:
                if self.data_source == "pgvector" and self.pg_conn is not None:
                    # Delete directly from Postgres (memories + vectors)
                    self.pg_conn.execute(
                        "DELETE FROM memory_vectors WHERE id = %s",
                        (self.selected_memory_id,),
                    )
                    cur = self.pg_conn.execute(
                        "DELETE FROM memories WHERE id = %s RETURNING id",
                        (self.selected_memory_id,),
                    )
                    deleted = cur.fetchone()
                    if deleted:
                        messagebox.showinfo("Success", "Memory deleted from Postgres.")
                    else:
                        messagebox.showerror("Error", "Memory not found in Postgres.")
                        return
                elif self.memory_system:
                    # Use RobustMemorySystem to delete from both SQLite and ChromaDB
                    result = self.memory_system.delete_memory(self.selected_memory_id)
                    if result.success:
                        messagebox.showinfo("Success", "Memory deleted successfully!")
                    else:
                        messagebox.showerror(
                            "Error", f"Failed to delete memory:\n{result.reason}"
                        )
                        return
                else:
                    # Fallback: raw SQLite only
                    self.db_conn.execute(
                        "DELETE FROM memories WHERE id = ?", (self.selected_memory_id,)
                    )
                    self.db_conn.commit()
                    messagebox.showinfo(
                        "Success",
                        "Memory deleted (SQLite only - run rebuild_vectors to clean up ChromaDB).",
                    )

                self.new_memory()
                self.refresh_memories()

            except Exception as e:
                if self.data_source != "pgvector" and self.db_conn:
                    try:
                        self.db_conn.rollback()
                    except Exception:
                        pass
                messagebox.showerror("Error", f"Failed to delete memory:\n{str(e)}")

    def create_backup(self):
        """Create a backup of the database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"memory_backup_{timestamp}"
            backup_path = BACKUP_FOLDER / backup_name
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup SQLite database
            sqlite_backup = backup_path / "memories.db"
            shutil.copy2(DB_PATH, sqlite_backup)

            # Export to JSON
            rows = self._execute_read("SELECT * FROM memories ORDER BY timestamp")
            memories = []
            for row in rows:
                memory_dict = dict(row)
                memory_dict["tags"] = json.loads(memory_dict["tags"])
                memory_dict["metadata"] = (
                    json.loads(memory_dict["metadata"])
                    if memory_dict["metadata"]
                    else {}
                )
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

            messagebox.showinfo(
                "Success",
                f"Backup created successfully!\n\nLocation: {backup_path}\n\nFiles:\n- memories.db\n- memories_export.json",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create backup:\n{str(e)}")

    def export_memories(self):
        """Export memories to a JSON file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"lissa_memories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            if not file_path:
                return

            rows = self._execute_read("SELECT * FROM memories ORDER BY timestamp")
            memories = []
            for row in rows:
                memory_dict = dict(row)
                memory_dict["tags"] = json.loads(memory_dict["tags"])
                memory_dict["metadata"] = (
                    json.loads(memory_dict["metadata"])
                    if memory_dict["metadata"]
                    else {}
                )
                memories.append(memory_dict)

            with open(file_path, "w", encoding="utf-8") as f:
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

            messagebox.showinfo(
                "Success", f"Exported {len(memories)} memories to:\n{file_path}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export memories:\n{str(e)}")

    def update_statistics(self):
        """Update the statistics display"""
        try:
            stats = self._execute_read_one("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT memory_type) as types,
                    AVG(importance) as avg_importance,
                    SUM(token_count) as total_tokens,
                    AVG(token_count) as avg_tokens
                FROM memories
            """)

            type_breakdown = self._execute_read("""
                SELECT memory_type, COUNT(*) as count, SUM(token_count) as tokens
                FROM memories
                GROUP BY memory_type
                ORDER BY count DESC
            """)

            stats_text = f"Total Memories: {stats['total']}\n"
            stats_text += f"Memory Types: {stats['types']}\n"
            stats_text += f"Avg Importance: {stats['avg_importance']:.1f}\n"

            # Add token statistics
            total_tokens = stats["total_tokens"] or 0
            avg_tokens = stats["avg_tokens"] or 0
            stats_text += f"Total Tokens: {total_tokens:,}\n"
            stats_text += f"Avg Tokens: {avg_tokens:,.0f}\n\n"

            # Embedding model info
            stats_text += f"Embedding: {EMBEDDING_MODEL}\n"
            dims = EMBEDDING_MODEL_CONFIG["dimensions"]
            max_tok = EMBEDDING_MODEL_CONFIG["max_tokens"]
            stats_text += f"  Dims: {dims if dims is not None else '?'}, "
            stats_text += f"Max Tokens: {max_tok if max_tok is not None else '?'}\n\n"

            stats_text += "Breakdown:\n"
            for row in type_breakdown:
                tokens = row["tokens"] or 0
                stats_text += (
                    f"  {row['memory_type']}: {row['count']} ({tokens:,} tokens)\n"
                )

            self.stats_label.config(text=stats_text)

        except Exception as e:
            self.stats_label.config(text=f"Error loading stats:\n{str(e)}")

    def _query_vector_chromadb(self, memory_id: str, results_text):
        """Query a vector from ChromaDB's internal SQLite tables."""
        cursor = self.chroma_conn.execute(
            """
            SELECT eq.seq_id, eq.created_at, eq.id, eq.vector, eq.encoding,
                   e.segment_id, e.id as embedding_row_id
            FROM embeddings_queue eq
            LEFT JOIN embeddings e ON eq.seq_id = e.seq_id
            WHERE eq.id = ?
            """,
            (memory_id,),
        )
        embedding_row = cursor.fetchone()

        if not embedding_row:
            results_text.insert("1.0", f"No vector found for memory_id: {memory_id}")
            return

        # Get metadata
        metadata_cursor = self.chroma_conn.execute(
            """
            SELECT key, string_value, int_value, float_value
            FROM embedding_metadata
            WHERE id = ?
            ORDER BY key
            """,
            (embedding_row["embedding_row_id"],),
        )
        metadata_rows = metadata_cursor.fetchall()

        metadata = {}
        for row in metadata_rows:
            key = row["key"]
            if row["string_value"] is not None:
                metadata[key] = row["string_value"]
            elif row["int_value"] is not None:
                metadata[key] = row["int_value"]
            elif row["float_value"] is not None:
                metadata[key] = row["float_value"]

        output = "=" * 80 + "\n"
        output += "VECTOR EMBEDDING FOUND (ChromaDB)\n"
        output += "=" * 80 + "\n\n"
        output += f"Memory ID: {embedding_row['id']}\n"
        output += f"Seq ID: {embedding_row['seq_id']}\n"
        output += f"Segment ID: {embedding_row['segment_id']}\n"
        output += f"Created At: {embedding_row['created_at']}\n"
        output += f"Encoding: {embedding_row['encoding']}\n\n"
        output += "METADATA:\n"
        output += "-" * 80 + "\n"
        for key, value in metadata.items():
            output += f"  {key}: {value}\n"

        vector_blob = embedding_row["vector"]
        if vector_blob is None:
            results_text.insert(
                "1.0", f"No vector data found for memory_id: {memory_id}"
            )
            return

        import struct

        vector_size = len(vector_blob) // 4
        vector = struct.unpack(f"{vector_size}f", vector_blob)

        output += self._format_vector_stats(vector)
        results_text.insert("1.0", output)

    def _query_vector_pgvector(self, memory_id: str, results_text):
        """Query a vector from the Postgres memory_vectors table."""
        if self.pg_conn is None:
            results_text.insert("1.0", "Not connected to Postgres.")
            return
        row = self.pg_conn.execute(
            "SELECT id, embedding, document, metadata FROM memory_vectors WHERE id = %s",
            (memory_id,),
        ).fetchone()

        if not row:
            results_text.insert("1.0", f"No vector found for memory_id: {memory_id}")
            return

        output = "=" * 80 + "\n"
        output += "VECTOR EMBEDDING FOUND (pgvector/Postgres)\n"
        output += "=" * 80 + "\n\n"
        output += f"Memory ID: {row['id']}\n"
        output += f"Document (preview): {str(row.get('document', ''))[:200]}\n\n"

        meta = row.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        output += "METADATA:\n"
        output += "-" * 80 + "\n"
        if isinstance(meta, dict):
            for key, value in meta.items():
                output += f"  {key}: {value}\n"
        else:
            output += f"  (raw): {meta}\n"

        # Parse the embedding
        emb = row.get("embedding")
        if emb is not None:
            # pgvector returns embedding as a list or numpy array
            vector = (
                emb.tolist() if hasattr(emb, "tolist") else list(emb) if emb else []
            )
            output += self._format_vector_stats(vector)
        else:
            output += "\n(No embedding data)\n"

        results_text.insert("1.0", output)

    @staticmethod
    def _format_vector_stats(vector) -> str:
        """Format vector dimensions and statistics for display."""
        if not vector:
            return "\n(Empty vector)\n"

        output = "\n" + "=" * 80 + "\n"
        output += f"VECTOR DATA ({len(vector)} dimensions)\n"
        output += "=" * 80 + "\n\n"
        output += "First 20 dimensions:\n"
        output += "-" * 80 + "\n"
        for i in range(min(20, len(vector))):
            output += f"  [{i:3d}] = {vector[i]:10.6f}\n"
        if len(vector) > 20:
            output += f"\n... ({len(vector) - 20} more dimensions)\n"

        output += "\n" + "=" * 80 + "\n"
        output += "VECTOR STATISTICS\n"
        output += "=" * 80 + "\n"
        output += f"  Dimensions: {len(vector)}\n"
        output += f"  Min value: {min(vector):.6f}\n"
        output += f"  Max value: {max(vector):.6f}\n"
        output += f"  Mean value: {sum(vector) / len(vector):.6f}\n"
        l2_norm = sum(x * x for x in vector) ** 0.5
        output += f"  L2 Norm: {l2_norm:.6f}\n"
        return output

    def show_vector_window(self):
        """Show the vector visualization window.

        When data source is 'sqlite', queries ChromaDB's internal SQLite tables.
        When data source is 'pgvector', queries the Postgres memory_vectors table.
        """
        if self.data_source == "sqlite" and not self.chroma_conn:
            messagebox.showwarning(
                "ChromaDB Not Available",
                "ChromaDB connection is not available.\n\nVector visualization requires ChromaDB.",
            )
            return

        if self.data_source == "pgvector" and not self.pg_conn:
            messagebox.showwarning(
                "Postgres Not Connected",
                "Connect to Postgres first via the data source selector.",
            )
            return

        # Get selected memory info if available
        selected_memory_title = None
        if self.selected_memory_id:
            try:
                row = self._execute_read_one(
                    "SELECT title FROM memories WHERE id = ?",
                    (self.selected_memory_id,),
                )
                if row:
                    selected_memory_title = row["title"]
            except:
                pass

        # Create new window
        vector_window = tk.Toplevel(self.root)
        vector_window.configure(bg=self.bg_color)  # Dark background for window
        if selected_memory_title:
            vector_window.title(f"Vector Visualization - {selected_memory_title}")
        else:
            vector_window.title("Vector Database Visualization")
        vector_window.geometry("900x700")

        # Main container
        main_frame = ttk.Frame(vector_window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vector_window.columnconfigure(0, weight=1)
        vector_window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Title
        backend_label = (
            "pgvector/Postgres" if self.data_source == "pgvector" else "ChromaDB"
        )
        title_label = ttk.Label(
            main_frame, text=f"{backend_label} Vector Database", style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Statistics Section
        stats_frame = ttk.LabelFrame(main_frame, text="Vector Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.columnconfigure(0, weight=1)

        stats_text = self.get_vector_statistics()
        stats_label = ttk.Label(
            stats_frame, text=stats_text, style="Normal.TLabel", justify=tk.LEFT
        )
        stats_label.grid(row=0, column=0, sticky=tk.W)

        # Query Section
        query_frame = ttk.LabelFrame(
            main_frame, text="Query Vector by Memory ID", padding="10"
        )
        query_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        query_frame.columnconfigure(1, weight=1)

        ttk.Label(query_frame, text="Memory ID:", style="Normal.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        memory_id_var = tk.StringVar()
        # Auto-populate with selected memory ID if available
        if self.selected_memory_id:
            memory_id_var.set(self.selected_memory_id)
        memory_id_entry = ttk.Entry(
            query_frame, textvariable=memory_id_var, font=("Segoe UI", 10), width=40
        )
        memory_id_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))

        # Results text area
        results_frame = ttk.LabelFrame(main_frame, text="Query Results", padding="10")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=("Courier New", 9),
            height=20,
            bg=self.input_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            selectbackground=self.selected_bg,
            selectforeground="#ffffff",
            borderwidth=1,
            relief=tk.SOLID,
        )
        results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        def query_vector():
            """Query vector by memory ID — dispatches to ChromaDB or pgvector."""
            memory_id = memory_id_var.get().strip()
            if not memory_id:
                messagebox.showwarning(
                    "Input Required", "Please enter a memory ID to query."
                )
                return

            try:
                results_text.delete("1.0", tk.END)

                if self.data_source == "pgvector" and self.pg_conn is not None:
                    self._query_vector_pgvector(memory_id, results_text)
                else:
                    self._query_vector_chromadb(memory_id, results_text)

            except Exception as e:
                results_text.insert("1.0", f"Error querying vector:\n{str(e)}")

        # Query button
        query_btn = ttk.Button(
            query_frame,
            text="Query Vector",
            command=query_vector,
            style="Accent.TButton",
        )
        query_btn.grid(row=0, column=2, padx=(5, 0))

        # Clear button to reset view
        def clear_query():
            """Clear the query and results"""
            memory_id_var.set("")
            results_text.delete("1.0", tk.END)
            # Update window title to generic
            vector_window.title("Vector Database Visualization")

        clear_btn = ttk.Button(
            query_frame,
            text="Clear",
            command=clear_query,
        )
        clear_btn.grid(row=0, column=3, padx=(5, 0))

        # Show all button
        def show_all_vectors():
            """Show all vectors in the database"""
            try:
                results_text.delete("1.0", tk.END)

                cursor = self.chroma_conn.execute(
                    """
                    SELECT eq.id, eq.created_at, e.id as embedding_row_id,
                           GROUP_CONCAT(CASE WHEN m.key = 'title' THEN m.string_value END) as title,
                           GROUP_CONCAT(CASE WHEN m.key = 'memory_type' THEN m.string_value END) as memory_type,
                           GROUP_CONCAT(CASE WHEN m.key = 'importance' THEN m.int_value END) as importance
                    FROM embeddings_queue eq
                    LEFT JOIN embeddings e ON eq.seq_id = e.seq_id
                    LEFT JOIN embedding_metadata m ON e.id = m.id
                    GROUP BY eq.id, eq.created_at, e.id
                    ORDER BY eq.created_at DESC
                    LIMIT 100
                    """
                )
                rows = cursor.fetchall()

                output = "=" * 80 + "\n"
                output += f"ALL VECTOR EMBEDDINGS (showing up to 100)\n"
                output += "=" * 80 + "\n\n"

                for row in rows:
                    output += f"Memory ID: {row['id']}\n"
                    output += f"  Title: {row['title'] or 'N/A'}\n"
                    output += f"  Type: {row['memory_type'] or 'N/A'}\n"
                    output += f"  Importance: {row['importance'] or 'N/A'}\n"
                    output += f"  Created: {row['created_at']}\n"
                    output += "-" * 80 + "\n"

                results_text.insert("1.0", output)

            except Exception as e:
                results_text.insert("1.0", f"Error listing vectors:\n{str(e)}")

        show_all_btn = ttk.Button(
            query_frame, text="Show All Vectors", command=show_all_vectors
        )
        show_all_btn.grid(row=0, column=4, padx=(5, 0))

        # Auto-trigger query if a memory is selected
        if self.selected_memory_id:
            # Use after() to trigger query after window is fully constructed
            vector_window.after(100, query_vector)

    def get_vector_statistics(self) -> str:
        """Get statistics about the vector database (ChromaDB or pgvector)."""
        try:
            stats = f"Embedding Model: {EMBEDDING_MODEL}\n"
            stats += f"  HuggingFace: {EMBEDDING_MODEL_CONFIG['model_name']}\n"
            cfg_dims = EMBEDDING_MODEL_CONFIG["dimensions"]
            cfg_max = EMBEDDING_MODEL_CONFIG["max_tokens"]
            stats += f"  Configured Dims: {cfg_dims if cfg_dims is not None else '?'}\n"
            stats += f"  Max Tokens: {cfg_max if cfg_max is not None else '?'}\n"
            if EMBEDDING_MODEL_CONFIG.get("query_prefix"):
                stats += f"  Query Prefix: yes\n"
            stats += "\n"

            if self.data_source == "pgvector" and self.pg_conn is not None:
                return stats + self._get_pgvector_stats()
            else:
                return stats + self._get_chromadb_stats()

        except Exception as e:
            return f"Error loading statistics:\n{str(e)}"

    def _get_chromadb_stats(self) -> str:
        """Get vector statistics from ChromaDB's internal SQLite tables."""
        if not self.chroma_conn:
            return "ChromaDB not connected."

        cursor = self.chroma_conn.execute(
            "SELECT COUNT(*) as count FROM embeddings_queue"
        )
        total_embeddings = cursor.fetchone()["count"]

        cursor = self.chroma_conn.execute(
            """
            SELECT e.segment_id, COUNT(*) as count
            FROM embeddings_queue eq
            LEFT JOIN embeddings e ON eq.seq_id = e.seq_id
            GROUP BY e.segment_id
            """
        )
        segments = cursor.fetchall()

        cursor = self.chroma_conn.execute(
            "SELECT vector FROM embeddings_queue WHERE vector IS NOT NULL LIMIT 1"
        )
        sample = cursor.fetchone()

        vector_dimensions = 0
        if sample and sample["vector"]:
            import struct

            vector_blob = sample["vector"]
            vector_dimensions = len(vector_blob) // 4

        cfg_dims = EMBEDDING_MODEL_CONFIG["dimensions"]
        result = f"Backend: ChromaDB\n"
        result += f"Total Embeddings: {total_embeddings}\n"
        result += f"Vector Dimensions: {vector_dimensions}\n"

        if (
            vector_dimensions is not None
            and cfg_dims is not None
            and vector_dimensions != cfg_dims
        ):
            result += (
                f"  WARNING: stored vectors ({vector_dimensions}d) don't match "
                f"configured model ({cfg_dims}d)\n"
                f"  Run rebuild_vectors to re-embed with current model\n"
            )

        result += f"Segments: {len(segments)}\n"
        for seg in segments:
            result += f"  - {seg['segment_id']}: {seg['count']} embeddings\n"

        return result

    def _get_pgvector_stats(self) -> str:
        """Get vector statistics from the Postgres memory_vectors table."""
        if self.pg_conn is None:
            return "Not connected to Postgres."
        row = self.pg_conn.execute(
            "SELECT COUNT(*) as cnt FROM memory_vectors"
        ).fetchone()
        total = row["cnt"] if row else 0

        # Get dimension from a sample vector
        vector_dimensions = 0
        sample = self.pg_conn.execute(
            "SELECT embedding FROM memory_vectors LIMIT 1"
        ).fetchone()
        if sample and sample.get("embedding") is not None:
            emb = sample["embedding"]
            vector_dimensions = (
                len(emb.tolist()) if hasattr(emb, "tolist") else len(emb) if emb else 0
            )

        cfg_dims = EMBEDDING_MODEL_CONFIG["dimensions"]
        result = f"Backend: pgvector/Postgres\n"
        result += f"Total Embeddings: {total}\n"
        result += f"Vector Dimensions: {vector_dimensions}\n"

        if vector_dimensions and cfg_dims is not None and vector_dimensions != cfg_dims:
            result += (
                f"  WARNING: stored vectors ({vector_dimensions}d) don't match "
                f"configured model ({cfg_dims}d)\n"
            )

        return result

    def show_migration_window(self):
        """Show bidirectional database migration window.

        Supports:
          - SQLite/ChromaDB -> pgvector/Postgres
          - pgvector/Postgres -> SQLite/ChromaDB
          - SQLite -> SQLite (legacy, from another DB file)

        Migrates BOTH structured data (memories table) AND vectors.
        """
        migration_win = tk.Toplevel(self.root)
        migration_win.title("Database Migration Tool")
        migration_win.geometry("1100x800")
        migration_win.configure(bg=self.bg_color)
        migration_win.transient(self.root)
        migration_win.grab_set()

        main_frame = ttk.Frame(migration_win, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        migration_win.columnconfigure(0, weight=1)
        migration_win.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)  # preview gets the stretch

        # ── Title ──────────────────────────────────────────────
        ttk.Label(
            main_frame,
            text="Migrate Memories Between Databases",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        ttk.Label(
            main_frame,
            text="Transfer memories and vectors between SQLite/ChromaDB and pgvector/Postgres, or import from another SQLite file.",
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, sticky=tk.W, pady=(0, 10))

        # ── Direction selector ─────────────────────────────────
        direction_frame = ttk.LabelFrame(
            main_frame, text="Migration Direction", padding="10"
        )
        direction_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        direction_frame.columnconfigure(1, weight=1)

        direction_var = tk.StringVar(value="sqlite_to_pg")

        directions = [
            ("sqlite_to_pg", "SQLite/ChromaDB  -->  pgvector/Postgres"),
            ("pg_to_sqlite", "pgvector/Postgres  -->  SQLite/ChromaDB"),
            ("sqlite_to_sqlite", "SQLite file  -->  Active SQLite (legacy import)"),
        ]

        for i, (val, label) in enumerate(directions):
            ttk.Radiobutton(
                direction_frame,
                text=label,
                variable=direction_var,
                value=val,
                command=lambda: _on_direction_changed(),
            ).grid(row=0, column=i, sticky=tk.W, padx=(0, 20), pady=5)

        # ── Connection settings ────────────────────────────────
        conn_frame = ttk.LabelFrame(
            main_frame, text="PostgreSQL Connection", padding="10"
        )
        conn_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        conn_frame.columnconfigure(1, weight=1)
        conn_frame.columnconfigure(3, weight=1)

        import os as _os

        pg_host_var = tk.StringVar(value=_os.environ.get("PGHOST", "localhost"))
        pg_port_var = tk.StringVar(value=_os.environ.get("PGPORT", "5433"))
        pg_db_var = tk.StringVar(value=_os.environ.get("PGDATABASE", "memories"))
        pg_user_var = tk.StringVar(value=_os.environ.get("PGUSER", "memory_user"))
        pg_pass_var = tk.StringVar(value=_os.environ.get("PGPASSWORD", ""))

        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Entry(conn_frame, textvariable=pg_host_var, width=20).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=3
        )
        ttk.Label(conn_frame, text="Port:").grid(
            row=0, column=2, sticky=tk.W, padx=(15, 0), pady=3
        )
        ttk.Entry(conn_frame, textvariable=pg_port_var, width=8).grid(
            row=0, column=3, sticky=tk.W, padx=5, pady=3
        )

        ttk.Label(conn_frame, text="Database:").grid(
            row=1, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(conn_frame, textvariable=pg_db_var, width=20).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=3
        )
        ttk.Label(conn_frame, text="User:").grid(
            row=1, column=2, sticky=tk.W, padx=(15, 0), pady=3
        )
        ttk.Entry(conn_frame, textvariable=pg_user_var, width=20).grid(
            row=1, column=3, sticky=tk.W, padx=5, pady=3
        )

        ttk.Label(conn_frame, text="Password:").grid(
            row=2, column=0, sticky=tk.W, pady=3
        )
        pw_entry = ttk.Entry(conn_frame, textvariable=pg_pass_var, show="*", width=20)
        pw_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)

        # Test connection button
        conn_status_var = tk.StringVar(value="")

        def test_pg_connection():
            conn = None
            try:
                import psycopg

                conn = psycopg.connect(**_get_pg_connkw(), autocommit=True)
                cur = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name IN ('memories', 'memory_vectors')"
                )
                tables = cur.fetchone()[0]
                conn.execute("SELECT 1")  # basic health
                # Count memories and vectors
                mem_count = 0
                vec_count = 0
                if tables > 0:
                    try:
                        mem_count = conn.execute(
                            "SELECT COUNT(*) FROM memories"
                        ).fetchone()[0]
                    except Exception:
                        pass
                    try:
                        vec_count = conn.execute(
                            "SELECT COUNT(*) FROM memory_vectors"
                        ).fetchone()[0]
                    except Exception:
                        pass
                conn_status_var.set(
                    f"Connected OK  |  Tables: {tables}/2  |  "
                    f"Memories: {mem_count}  |  Vectors: {vec_count}"
                )
            except ImportError:
                conn_status_var.set(
                    "psycopg not installed. Run: pip install 'psycopg[binary]' pgvector"
                )
            except Exception as e:
                conn_status_var.set(f"Connection failed: {e}")
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        ttk.Button(conn_frame, text="Test Connection", command=test_pg_connection).grid(
            row=2, column=2, padx=(15, 5), pady=3
        )
        conn_status_label = ttk.Label(
            conn_frame, textvariable=conn_status_var, font=("Segoe UI", 8)
        )
        conn_status_label.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))

        # SQLite source file (for sqlite_to_sqlite mode)
        sqlite_source_frame = ttk.LabelFrame(
            main_frame, text="SQLite Source File (legacy import only)", padding="10"
        )
        sqlite_source_frame.columnconfigure(1, weight=1)

        source_db_var = tk.StringVar()
        ttk.Label(sqlite_source_frame, text="SQLite DB:").grid(
            row=0, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(sqlite_source_frame, textvariable=source_db_var).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=3
        )

        def browse_source_db():
            default_path = (
                Path.home() / "Documents" / "ai_companion_memory" / "memory_db"
            )
            filename = filedialog.askopenfilename(
                title="Select Source Database",
                initialdir=default_path if default_path.exists() else Path.home(),
                filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")],
            )
            if filename:
                source_db_var.set(filename)

        ttk.Button(sqlite_source_frame, text="Browse", command=browse_source_db).grid(
            row=0, column=2, padx=5, pady=3
        )

        def use_default_location():
            default_db = (
                Path.home()
                / "Documents"
                / "ai_companion_memory"
                / "memory_db"
                / "memories.db"
            )
            if default_db.exists():
                source_db_var.set(str(default_db))
            else:
                messagebox.showwarning(
                    "Not Found", f"Default database not found at:\n{default_db}"
                )

        ttk.Button(
            sqlite_source_frame, text="Use Default", command=use_default_location
        ).grid(row=0, column=3, padx=5, pady=3)

        def _on_direction_changed():
            d = direction_var.get()
            if d == "sqlite_to_sqlite":
                sqlite_source_frame.grid(
                    row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10)
                )
                conn_frame.grid_remove()
            else:
                conn_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
                sqlite_source_frame.grid_remove()

        # Initial state: show postgres settings, hide sqlite source
        sqlite_source_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        sqlite_source_frame.grid_remove()

        # ── Preview ────────────────────────────────────────────
        preview_frame = ttk.LabelFrame(
            main_frame, text="Preview Source Memories", padding="10"
        )
        preview_frame.grid(
            row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)
        )
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        preview_btn_frame = ttk.Frame(preview_frame)
        preview_btn_frame.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        count_label = ttk.Label(preview_btn_frame, text="", font=("Segoe UI", 9))
        count_label.grid(row=0, column=1, padx=10)

        preview_tree = ttk.Treeview(
            preview_frame,
            columns=("ID", "Title", "Type", "Importance", "Tags"),
            show="headings",
            height=10,
        )
        for col, w in [
            ("ID", 120),
            ("Title", 350),
            ("Type", 100),
            ("Importance", 80),
            ("Tags", 200),
        ]:
            preview_tree.heading(col, text=col)
            preview_tree.column(col, width=w)
        preview_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        preview_scroll = ttk.Scrollbar(
            preview_frame, orient="vertical", command=preview_tree.yview
        )
        preview_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        preview_tree.configure(yscrollcommand=preview_scroll.set)

        def _clear_preview():
            for item in preview_tree.get_children():
                preview_tree.delete(item)
            count_label.config(text="")

        def _populate_preview(rows_dicts):
            """rows_dicts: list of dicts with id, title, memory_type, importance, tags"""
            _clear_preview()
            for r in rows_dicts:
                tags_raw = r.get("tags", "")
                if isinstance(tags_raw, str):
                    try:
                        tags_raw = json.loads(tags_raw) if tags_raw else []
                    except (json.JSONDecodeError, TypeError):
                        tags_raw = [tags_raw] if tags_raw else []
                tags_str = ", ".join(str(t) for t in tags_raw)[:30] if tags_raw else ""
                mid = str(r.get("id", ""))
                preview_tree.insert(
                    "",
                    "end",
                    values=(
                        mid[:15] + ("..." if len(mid) > 15 else ""),
                        str(r.get("title", ""))[:50],
                        r.get("memory_type", ""),
                        r.get("importance", ""),
                        tags_str,
                    ),
                )
            count_label.config(text=f"Found {len(rows_dicts)} memories (up to 200)")

        def preview_source():
            direction = direction_var.get()
            try:
                if direction == "sqlite_to_sqlite":
                    spath = source_db_var.get()
                    if not spath or not Path(spath).exists():
                        messagebox.showwarning(
                            "Missing Input", "Select a valid source SQLite file."
                        )
                        return
                    src = sqlite3.connect(spath)
                    src.row_factory = sqlite3.Row
                    rows = src.execute(
                        "SELECT id, title, memory_type, importance, tags FROM memories ORDER BY timestamp DESC LIMIT 200"
                    ).fetchall()
                    src.close()
                    _populate_preview([dict(r) for r in rows])

                elif direction == "sqlite_to_pg":
                    # Source = local SQLite
                    if not DB_PATH.exists():
                        messagebox.showwarning(
                            "Not Found", f"Local SQLite DB not found:\n{DB_PATH}"
                        )
                        return
                    src = sqlite3.connect(str(DB_PATH))
                    src.row_factory = sqlite3.Row
                    rows = src.execute(
                        "SELECT id, title, memory_type, importance, tags FROM memories ORDER BY timestamp DESC LIMIT 200"
                    ).fetchall()
                    src.close()
                    _populate_preview([dict(r) for r in rows])

                elif direction == "pg_to_sqlite":
                    # Source = Postgres
                    import psycopg

                    conn = psycopg.connect(**_get_pg_connkw(), autocommit=True)
                    try:
                        cur = conn.execute(
                            "SELECT id, title, memory_type, importance, tags FROM memories ORDER BY timestamp DESC LIMIT 200"
                        )
                        cols = [d[0] for d in cur.description]
                        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                    finally:
                        conn.close()
                    _populate_preview(rows)

            except ImportError:
                messagebox.showerror(
                    "Missing Dependency",
                    "psycopg not installed.\nRun: pip install 'psycopg[binary]'",
                )
            except Exception as e:
                messagebox.showerror("Preview Error", f"Failed to preview:\n{e}")

        ttk.Button(
            preview_btn_frame, text="Preview Source", command=preview_source
        ).grid(row=0, column=0, padx=2)

        # ── Options ────────────────────────────────────────────
        options_frame = ttk.LabelFrame(
            main_frame, text="Migration Options", padding="10"
        )
        options_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        skip_dup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Skip duplicate memories (by content hash)",
            variable=skip_dup_var,
        ).grid(row=0, column=0, sticky=tk.W, pady=3)

        migrate_vectors_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Migrate vectors (embeddings) alongside structured data",
            variable=migrate_vectors_var,
        ).grid(row=1, column=0, sticky=tk.W, pady=3)

        # ── Status / progress ──────────────────────────────────
        status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            main_frame, textvariable=status_var, font=("Segoe UI", 9)
        )
        status_label.grid(row=6, column=0, sticky=tk.W, pady=(0, 5))

        # ── Action buttons ─────────────────────────────────────
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=7, column=0, sticky=(tk.W, tk.E))

        def _get_pg_connkw():
            """Build psycopg keyword-arg dict (no conninfo string — avoids injection)."""
            port_str = pg_port_var.get().strip()
            try:
                port = int(port_str)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid port number: '{port_str}'. Enter a numeric port (e.g. 5433)."
                )
            return dict(
                host=pg_host_var.get().strip(),
                port=port,
                dbname=pg_db_var.get().strip(),
                user=pg_user_var.get().strip(),
                password=pg_pass_var.get(),
            )

        def perform_migration():
            direction = direction_var.get()

            if direction == "sqlite_to_sqlite":
                _migrate_sqlite_to_sqlite()
            elif direction == "sqlite_to_pg":
                _migrate_sqlite_to_pg()
            elif direction == "pg_to_sqlite":
                _migrate_pg_to_sqlite()

        def _migrate_sqlite_to_sqlite():
            """Legacy: import from another SQLite file into the active DB."""
            spath = source_db_var.get()
            if not spath or not Path(spath).exists():
                messagebox.showwarning(
                    "Missing Input", "Select a valid source SQLite file."
                )
                return
            if not messagebox.askyesno(
                "Confirm", "Import memories from the selected SQLite file?"
            ):
                return
            try:
                if self.memory_system:
                    ms = self.memory_system
                else:
                    from memory_mcp.memory_system import RobustMemorySystem

                    ms = RobustMemorySystem(DATA_FOLDER)

                chroma_path = Path(spath).parent / "chroma_db"
                result = ms.migrate_memories(
                    source_db_path=spath,
                    source_chroma_path=str(chroma_path)
                    if chroma_path.exists()
                    else None,
                    memory_ids=None,
                    skip_duplicates=skip_dup_var.get(),
                )
                if result.success:
                    s = result.data[0]
                    messagebox.showinfo(
                        "Migration Complete",
                        f"Migrated: {s['migrated']}  |  Skipped: {s['skipped_duplicates']}  |  Errors: {s['errors']}",
                    )
                    self.refresh_memories()
                    self.update_statistics()
                    migration_win.destroy()
                else:
                    messagebox.showerror("Failed", result.reason)
            except Exception as e:
                messagebox.showerror("Error", f"Migration failed:\n{e}")

        def _migrate_sqlite_to_pg():
            """Migrate structured data + vectors from SQLite/ChromaDB to pgvector/Postgres."""
            if not DB_PATH.exists():
                messagebox.showerror("Error", f"Local SQLite not found:\n{DB_PATH}")
                return
            if not messagebox.askyesno(
                "Confirm",
                "Migrate all memories from SQLite/ChromaDB to pgvector/Postgres?\n\n"
                "This will INSERT into Postgres (existing records with same ID are updated).",
            ):
                return

            status_var.set("Migrating SQLite -> Postgres ...")
            migration_win.update_idletasks()

            pg_conn = None
            src = None
            chroma = None
            try:
                import psycopg

                pg_conn = psycopg.connect(**_get_pg_connkw(), autocommit=False)

                # Ensure tables exist
                pg_conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
                        timestamp TEXT NOT NULL, tags TEXT, importance INTEGER DEFAULT 5,
                        memory_type TEXT DEFAULT 'conversation', metadata TEXT,
                        content_hash TEXT, created_at TEXT DEFAULT (NOW()::text),
                        updated_at TEXT DEFAULT (NOW()::text),
                        last_accessed TEXT DEFAULT (NOW()::text),
                        token_count INTEGER DEFAULT 0
                    )
                """)
                pg_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                pg_conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_vectors (
                        id TEXT PRIMARY KEY, embedding vector(384),
                        document TEXT, metadata JSONB DEFAULT '{}'::jsonb
                    )
                """)
                pg_conn.commit()

                # Read all SQLite memories
                src = sqlite3.connect(str(DB_PATH))
                src.row_factory = sqlite3.Row
                rows = src.execute(
                    "SELECT * FROM memories ORDER BY timestamp"
                ).fetchall()

                migrated = 0
                skipped = 0
                errors = 0

                for row in rows:
                    rd = dict(row)
                    try:
                        if skip_dup_var.get() and rd.get("content_hash"):
                            dup = pg_conn.execute(
                                "SELECT id FROM memories WHERE content_hash = %s",
                                (rd["content_hash"],),
                            ).fetchone()
                            if dup:
                                skipped += 1
                                continue

                        # Use savepoint so a single row failure doesn't abort
                        # the entire transaction
                        pg_conn.execute("SAVEPOINT row_sp")
                        pg_conn.execute(
                            """
                            INSERT INTO memories
                                (id, title, content, timestamp, tags, importance,
                                 memory_type, metadata, content_hash, created_at,
                                 updated_at, last_accessed, token_count)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (id) DO UPDATE SET
                                title=EXCLUDED.title, content=EXCLUDED.content,
                                timestamp=EXCLUDED.timestamp, tags=EXCLUDED.tags,
                                importance=EXCLUDED.importance, memory_type=EXCLUDED.memory_type,
                                metadata=EXCLUDED.metadata, content_hash=EXCLUDED.content_hash,
                                updated_at=EXCLUDED.updated_at, last_accessed=EXCLUDED.last_accessed,
                                token_count=EXCLUDED.token_count
                            """,
                            (
                                rd["id"],
                                rd["title"],
                                rd["content"],
                                rd["timestamp"],
                                rd["tags"],
                                rd["importance"],
                                rd["memory_type"],
                                rd["metadata"],
                                rd.get("content_hash"),
                                rd.get("created_at"),
                                rd.get("updated_at"),
                                rd.get("last_accessed"),
                                rd.get("token_count", 0),
                            ),
                        )
                        pg_conn.execute("RELEASE SAVEPOINT row_sp")
                        migrated += 1
                    except Exception as e:
                        errors += 1
                        print(f"Error migrating {rd.get('id')}: {e}")
                        try:
                            pg_conn.execute("ROLLBACK TO SAVEPOINT row_sp")
                        except Exception:
                            pass

                pg_conn.commit()

                # Migrate vectors from ChromaDB if requested
                vec_migrated = 0
                if migrate_vectors_var.get():
                    status_var.set("Migrating vectors ...")
                    migration_win.update_idletasks()
                    try:
                        from pgvector.psycopg import register_vector

                        register_vector(pg_conn)

                        from memory_mcp.vector_backends.chroma import ChromaBackend

                        chroma = ChromaBackend(db_folder=DB_PATH.parent)
                        chroma.initialize()

                        # Get all vectors with embeddings
                        all_vecs = chroma.get(include_embeddings=True)
                        ids = all_vecs.get("ids", [])
                        embeddings = all_vecs.get("embeddings", [])
                        documents = all_vecs.get("documents", [])
                        metadatas = all_vecs.get("metadatas", [])

                        for i, vid in enumerate(ids):
                            try:
                                emb = embeddings[i] if i < len(embeddings) else None
                                doc = documents[i] if i < len(documents) else ""
                                meta = metadatas[i] if i < len(metadatas) else {}
                                if emb is not None:
                                    # Ensure embedding is a flat list of floats
                                    emb_list = (
                                        emb.tolist()
                                        if hasattr(emb, "tolist")
                                        else list(emb)
                                    )
                                    pg_conn.execute("SAVEPOINT vec_sp")
                                    pg_conn.execute(
                                        """
                                        INSERT INTO memory_vectors (id, embedding, document, metadata)
                                        VALUES (%s, %s::vector, %s, %s)
                                        ON CONFLICT (id) DO UPDATE SET
                                            embedding=EXCLUDED.embedding,
                                            document=EXCLUDED.document,
                                            metadata=EXCLUDED.metadata
                                        """,
                                        (
                                            vid,
                                            str(emb_list),
                                            doc,
                                            json.dumps(meta or {}),
                                        ),
                                    )
                                    pg_conn.execute("RELEASE SAVEPOINT vec_sp")
                                    vec_migrated += 1
                            except Exception as ve:
                                print(f"Vector migrate error {vid}: {ve}")
                                try:
                                    pg_conn.execute("ROLLBACK TO SAVEPOINT vec_sp")
                                except Exception:
                                    pass

                        pg_conn.commit()
                    except Exception as ve:
                        vec_error = str(ve)
                        print(f"Vector migration error: {ve}")
                    finally:
                        if chroma is not None:
                            try:
                                chroma.close()
                            except Exception:
                                pass
                            chroma = None

                status_var.set("Done!")
                vec_note = ""
                if migrate_vectors_var.get() and vec_migrated == 0:
                    vec_note = "\n\nWARNING: No vectors were migrated."
                    if "vec_error" in dir():
                        vec_note += f"\nVector error: {vec_error}"
                messagebox.showinfo(
                    "Migration Complete",
                    f"SQLite -> Postgres migration complete!\n\n"
                    f"Memories migrated: {migrated}\n"
                    f"Skipped (duplicates): {skipped}\n"
                    f"Errors: {errors}\n"
                    f"Vectors migrated: {vec_migrated}{vec_note}",
                )
                self.refresh_memories()
                self.update_statistics()

            except ImportError:
                status_var.set("Failed")
                messagebox.showerror(
                    "Missing Dependency",
                    "Install: pip install 'psycopg[binary]' pgvector",
                )
            except Exception as e:
                status_var.set("Failed")
                if pg_conn is not None:
                    try:
                        pg_conn.rollback()
                    except Exception:
                        pass
                messagebox.showerror("Error", f"Migration failed:\n{e}")
            finally:
                if src is not None:
                    try:
                        src.close()
                    except Exception:
                        pass
                if pg_conn is not None:
                    try:
                        pg_conn.close()
                    except Exception:
                        pass

        def _migrate_pg_to_sqlite():
            """Migrate structured data + vectors from pgvector/Postgres to SQLite/ChromaDB."""
            if not messagebox.askyesno(
                "Confirm",
                "Migrate all memories from pgvector/Postgres to local SQLite/ChromaDB?\n\n"
                "This will INSERT into SQLite (existing records with same ID are updated).",
            ):
                return

            status_var.set("Migrating Postgres -> SQLite ...")
            migration_win.update_idletasks()

            pg_conn = None
            dest = None
            chroma = None
            try:
                import psycopg

                pg_conn = psycopg.connect(**_get_pg_connkw(), autocommit=True)

                # Read Postgres memories
                cur = pg_conn.execute("SELECT * FROM memories ORDER BY timestamp")
                cols = [d[0] for d in cur.description]
                pg_rows = [dict(zip(cols, row)) for row in cur.fetchall()]

                # Write to SQLite
                dest = sqlite3.connect(
                    str(DB_PATH), check_same_thread=False, timeout=30.0
                )
                dest.row_factory = sqlite3.Row
                dest.execute("PRAGMA journal_mode=WAL")
                # Ensure schema exists (needed for fresh installs)
                dest.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
                        timestamp TEXT NOT NULL, tags TEXT, importance INTEGER DEFAULT 5,
                        memory_type TEXT DEFAULT 'conversation', metadata TEXT,
                        content_hash TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TEXT DEFAULT CURRENT_TIMESTAMP,
                        token_count INTEGER DEFAULT 0
                    )
                """)

                migrated = 0
                skipped = 0
                errors = 0

                for rd in pg_rows:
                    try:
                        if skip_dup_var.get() and rd.get("content_hash"):
                            dup = dest.execute(
                                "SELECT id FROM memories WHERE content_hash = ?",
                                (rd["content_hash"],),
                            ).fetchone()
                            if dup:
                                skipped += 1
                                continue

                        dest.execute(
                            """
                            INSERT OR REPLACE INTO memories
                                (id, title, content, timestamp, tags, importance,
                                 memory_type, metadata, content_hash, created_at,
                                 updated_at, last_accessed, token_count)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                rd["id"],
                                rd["title"],
                                rd["content"],
                                rd["timestamp"],
                                rd.get("tags"),
                                rd.get("importance", 5),
                                rd.get("memory_type", "conversation"),
                                rd.get("metadata"),
                                rd.get("content_hash"),
                                rd.get("created_at"),
                                rd.get("updated_at"),
                                rd.get("last_accessed"),
                                rd.get("token_count", 0),
                            ),
                        )
                        migrated += 1
                    except Exception as e:
                        errors += 1
                        print(f"Error migrating {rd.get('id')}: {e}")

                dest.commit()

                # Migrate vectors from Postgres to ChromaDB
                vec_migrated = 0
                if migrate_vectors_var.get():
                    status_var.set("Migrating vectors ...")
                    migration_win.update_idletasks()
                    try:
                        from pgvector.psycopg import register_vector

                        register_vector(pg_conn)

                        from memory_mcp.vector_backends.chroma import ChromaBackend

                        chroma = ChromaBackend(db_folder=DB_PATH.parent)
                        chroma.initialize()

                        # Get all vectors from Postgres
                        cur = pg_conn.execute(
                            "SELECT id, embedding, document, metadata FROM memory_vectors"
                        )
                        batch_ids, batch_embs, batch_docs, batch_metas = [], [], [], []

                        for row in cur.fetchall():
                            vid, emb, doc, meta = row[0], row[1], row[2], row[3]
                            if emb is not None:
                                # Ensure embedding is a proper list of floats
                                emb_list = (
                                    emb.tolist()
                                    if hasattr(emb, "tolist")
                                    else list(emb)
                                )
                                if isinstance(meta, str):
                                    meta = json.loads(meta)
                                batch_ids.append(vid)
                                batch_embs.append(emb_list)
                                batch_docs.append(doc or "")
                                batch_metas.append(meta or {})

                                if len(batch_ids) >= 100:
                                    chroma.add(
                                        ids=batch_ids,
                                        embeddings=batch_embs,
                                        documents=batch_docs,
                                        metadatas=batch_metas,
                                    )
                                    vec_migrated += len(batch_ids)
                                    batch_ids, batch_embs, batch_docs, batch_metas = (
                                        [],
                                        [],
                                        [],
                                        [],
                                    )

                        if batch_ids:
                            chroma.add(
                                ids=batch_ids,
                                embeddings=batch_embs,
                                documents=batch_docs,
                                metadatas=batch_metas,
                            )
                            vec_migrated += len(batch_ids)

                    except Exception as ve:
                        vec_error_pg = str(ve)
                        print(f"Vector migration error: {ve}")
                    finally:
                        if chroma is not None:
                            try:
                                chroma.close()
                            except Exception:
                                pass
                            chroma = None

                status_var.set("Done!")
                vec_note = ""
                if migrate_vectors_var.get() and vec_migrated == 0:
                    vec_note = "\n\nWARNING: No vectors were migrated."
                    if "vec_error_pg" in dir():
                        vec_note += f"\nVector error: {vec_error_pg}"
                messagebox.showinfo(
                    "Migration Complete",
                    f"Postgres -> SQLite migration complete!\n\n"
                    f"Memories migrated: {migrated}\n"
                    f"Skipped (duplicates): {skipped}\n"
                    f"Errors: {errors}\n"
                    f"Vectors migrated: {vec_migrated}{vec_note}",
                )
                # Refresh main window (which reads from SQLite)
                self.refresh_memories()
                self.update_statistics()

            except ImportError:
                status_var.set("Failed")
                messagebox.showerror(
                    "Missing Dependency",
                    "Install: pip install 'psycopg[binary]' pgvector",
                )
            except Exception as e:
                status_var.set("Failed")
                messagebox.showerror("Error", f"Migration failed:\n{e}")
            finally:
                if dest is not None:
                    try:
                        dest.close()
                    except Exception:
                        pass
                if pg_conn is not None:
                    try:
                        pg_conn.close()
                    except Exception:
                        pass

        ttk.Button(
            action_frame, text="Start Migration", command=perform_migration
        ).grid(row=0, column=0, padx=5)
        ttk.Button(action_frame, text="Close", command=migration_win.destroy).grid(
            row=0, column=1, padx=5
        )

    def on_closing(self):
        """Handle window closing with WAL checkpoint for data safety"""
        # Close RobustMemorySystem first (handles its own WAL checkpoint)
        if self.memory_system:
            try:
                self.memory_system.close()
            except Exception:
                pass

        # Close GUI's own SQLite connection with WAL checkpoint
        if self.db_conn:
            try:
                self.db_conn.commit()
            except Exception:
                pass
            try:
                self.db_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            try:
                self.db_conn.close()
            except Exception:
                pass

        # Close ChromaDB read-only connection
        if self.chroma_conn:
            try:
                self.chroma_conn.close()
            except Exception:
                pass

        # Close pgvector/Postgres GUI connection
        self._disconnect_pgvector()

        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = MemoryManagerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
