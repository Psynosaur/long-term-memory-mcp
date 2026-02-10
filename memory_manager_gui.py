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

# Configuration
DATA_FOLDER = Path(
    os.environ.get(
        "AI_COMPANION_DATA_DIR", str(Path.home() / "Documents" / "ai_companion_memory")
    )
)
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

        # Custom colors
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#4a9eff"
        self.secondary_bg = "#3a3a3a"

        # Configure styles
        self.configure_styles()

        # Database connection
        self.db_conn = None
        self.connect_database()

        # ChromaDB connection
        self.chroma_conn = None
        self.connect_chromadb()

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
        """Configure custom styles for widgets"""
        self.style.configure(
            "Title.TLabel", font=("Segoe UI", 24, "bold"), foreground=self.accent_color
        )
        self.style.configure(
            "Subtitle.TLabel", font=("Segoe UI", 12), foreground="#cccccc"
        )
        self.style.configure(
            "Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=self.fg_color
        )
        self.style.configure(
            "Normal.TLabel", font=("Segoe UI", 10), foreground=self.fg_color
        )
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

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

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

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
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )

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

        # ===== LEFT PANEL - Search and List =====
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
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
        columns = ("ID", "Title", "Type", "Importance", "Date")
        self.tree = ttk.Treeview(
            list_frame, columns=columns, show="tree headings", selectmode="browse"
        )

        # Configure columns
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("ID", width=0, stretch=tk.NO)
        self.tree.column("Title", width=300)
        self.tree.column("Type", width=100)
        self.tree.column("Importance", width=80, anchor=tk.CENTER)
        self.tree.column("Date", width=150)

        # Configure headings
        self.tree.heading("Title", text="Title")
        self.tree.heading(
            "Type", text="Type", command=lambda: self.sort_by_column("Type")
        )
        self.tree.heading("Importance", text="Importance")
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

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_memory_selected)

        # ===== RIGHT PANEL - Details and Actions =====
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
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

        # Details section
        details_frame = ttk.LabelFrame(right_panel, text="Memory Details", padding="10")
        details_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(1, weight=1)
        details_frame.rowconfigure(5, weight=1)

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

        # Content
        ttk.Label(details_frame, text="Content:", style="Header.TLabel").grid(
            row=5, column=0, sticky=(tk.W, tk.N), pady=5
        )
        self.content_text = scrolledtext.ScrolledText(
            details_frame, wrap=tk.WORD, font=("Segoe UI", 10), height=15
        )
        self.content_text.grid(
            row=5, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(10, 0)
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

            cursor = self.db_conn.execute(query, params)
            rows = cursor.fetchall()

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
                        date_str,
                    ),
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
            cursor = self.db_conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()

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
        self.content_text.delete("1.0", tk.END)
        self.metadata_label.config(text="New memory - fill in details and click Save")

    def save_memory(self):
        """Save the current memory (create or update)"""
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

            # Calculate token count
            token_count = self.count_tokens(content)

            now_iso = datetime.now(timezone.utc).isoformat()

            if self.selected_memory_id:
                # Update existing memory
                self.db_conn.execute(
                    """
                    UPDATE memories
                    SET title = ?, content = ?, memory_type = ?, importance = ?, tags = ?, updated_at = ?, token_count = ?
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
                        self.selected_memory_id,
                    ),
                )

                messagebox.showinfo("Success", "Memory updated successfully!")
            else:
                # Create new memory
                import hashlib

                content_hash = hashlib.sha256(content.encode()).hexdigest()
                time_hash = hashlib.sha256(now_iso.encode()).hexdigest()[:8]
                memory_id = f"mem_{time_hash}_{content_hash[:16]}"

                self.db_conn.execute(
                    """
                    INSERT INTO memories (id, title, content, timestamp, tags, importance, memory_type, metadata, content_hash, created_at, updated_at, last_accessed, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )

                self.selected_memory_id = memory_id
                messagebox.showinfo(
                    "Success",
                    "Memory created successfully!\n\nNote: Vector embeddings will be updated when the MCP server is restarted.",
                )

            self.db_conn.commit()
            self.refresh_memories()

        except Exception as e:
            self.db_conn.rollback()
            messagebox.showerror("Error", f"Failed to save memory:\n{str(e)}")

    def delete_memory(self):
        """Delete the selected memory"""
        if not self.selected_memory_id:
            messagebox.showwarning("No Selection", "Please select a memory to delete.")
            return

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete this memory?\n\nTitle: {self.title_var.get()}\n\nThis action cannot be undone.",
        )

        if result:
            try:
                self.db_conn.execute(
                    "DELETE FROM memories WHERE id = ?", (self.selected_memory_id,)
                )
                self.db_conn.commit()

                messagebox.showinfo(
                    "Success",
                    "Memory deleted successfully!\n\nNote: Vector embeddings will be cleaned up when the MCP server is restarted.",
                )

                self.new_memory()
                self.refresh_memories()

            except Exception as e:
                self.db_conn.rollback()
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
            cursor = self.db_conn.execute("SELECT * FROM memories ORDER BY timestamp")
            memories = []
            for row in cursor.fetchall():
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

            cursor = self.db_conn.execute("SELECT * FROM memories ORDER BY timestamp")
            memories = []
            for row in cursor.fetchall():
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
            cursor = self.db_conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT memory_type) as types,
                    AVG(importance) as avg_importance,
                    SUM(token_count) as total_tokens,
                    AVG(token_count) as avg_tokens
                FROM memories
            """)
            stats = cursor.fetchone()

            cursor = self.db_conn.execute("""
                SELECT memory_type, COUNT(*) as count, SUM(token_count) as tokens
                FROM memories
                GROUP BY memory_type
                ORDER BY count DESC
            """)
            type_breakdown = cursor.fetchall()

            stats_text = f"Total Memories: {stats['total']}\n"
            stats_text += f"Memory Types: {stats['types']}\n"
            stats_text += f"Avg Importance: {stats['avg_importance']:.1f}\n"

            # Add token statistics
            total_tokens = stats["total_tokens"] or 0
            avg_tokens = stats["avg_tokens"] or 0
            stats_text += f"Total Tokens: {total_tokens:,}\n"
            stats_text += f"Avg Tokens: {avg_tokens:,.0f}\n\n"

            stats_text += "Breakdown:\n"
            for row in type_breakdown:
                tokens = row["tokens"] or 0
                stats_text += (
                    f"  {row['memory_type']}: {row['count']} ({tokens:,} tokens)\n"
                )

            self.stats_label.config(text=stats_text)

        except Exception as e:
            self.stats_label.config(text=f"Error loading stats:\n{str(e)}")

    def show_vector_window(self):
        """Show the vector visualization window"""
        if not self.chroma_conn:
            messagebox.showwarning(
                "ChromaDB Not Available",
                "ChromaDB connection is not available.\n\nVector visualization requires ChromaDB.",
            )
            return

        # Get selected memory info if available
        selected_memory_title = None
        if self.selected_memory_id:
            try:
                cursor = self.db_conn.execute(
                    "SELECT title FROM memories WHERE id = ?",
                    (self.selected_memory_id,),
                )
                row = cursor.fetchone()
                if row:
                    selected_memory_title = row["title"]
            except:
                pass

        # Create new window
        vector_window = tk.Toplevel(self.root)
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
        title_label = ttk.Label(
            main_frame, text="ChromaDB Vector Database", style="Title.TLabel"
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
            results_frame, wrap=tk.WORD, font=("Courier New", 9), height=20
        )
        results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        def query_vector():
            """Query vector by memory ID"""
            memory_id = memory_id_var.get().strip()
            if not memory_id:
                messagebox.showwarning(
                    "Input Required", "Please enter a memory ID to query."
                )
                return

            try:
                # Clear previous results
                results_text.delete("1.0", tk.END)

                # Query embeddings_queue table for the actual vector data
                # The vector BLOB is stored in embeddings_queue, not embeddings
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
                    results_text.insert(
                        "1.0", f"No vector found for memory_id: {memory_id}"
                    )
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

                # Build metadata dict
                metadata = {}
                for row in metadata_rows:
                    key = row["key"]
                    if row["string_value"] is not None:
                        metadata[key] = row["string_value"]
                    elif row["int_value"] is not None:
                        metadata[key] = row["int_value"]
                    elif row["float_value"] is not None:
                        metadata[key] = row["float_value"]

                # Format output
                output = "=" * 80 + "\n"
                output += f"VECTOR EMBEDDING FOUND\n"
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

                # Get vector from embeddings_queue
                vector_blob = embedding_row["vector"]

                # Check if vector_blob is None or not a BLOB
                if vector_blob is None:
                    results_text.insert(
                        "1.0", f"No vector data found for memory_id: {memory_id}"
                    )
                    return

                # ChromaDB stores vectors as float32 arrays
                import struct

                # The vector BLOB contains the vector as float32 values
                vector_size = len(vector_blob) // 4  # 4 bytes per float32
                vector = struct.unpack(f"{vector_size}f", vector_blob)

                output += "\n" + "=" * 80 + "\n"
                output += f"VECTOR DATA ({vector_size} dimensions)\n"
                output += "=" * 80 + "\n\n"

                # Show first 20 dimensions
                output += "First 20 dimensions:\n"
                output += "-" * 80 + "\n"
                for i in range(min(20, len(vector))):
                    output += f"  [{i:3d}] = {vector[i]:10.6f}\n"

                if len(vector) > 20:
                    output += f"\n... ({len(vector) - 20} more dimensions)\n"

                # Vector statistics
                output += "\n" + "=" * 80 + "\n"
                output += "VECTOR STATISTICS\n"
                output += "=" * 80 + "\n"
                output += f"  Dimensions: {len(vector)}\n"
                output += f"  Min value: {min(vector):.6f}\n"
                output += f"  Max value: {max(vector):.6f}\n"
                output += f"  Mean value: {sum(vector) / len(vector):.6f}\n"

                # L2 norm
                l2_norm = sum(x * x for x in vector) ** 0.5
                output += f"  L2 Norm: {l2_norm:.6f}\n"

                results_text.insert("1.0", output)

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
        """Get statistics about the vector database"""
        try:
            # Count total embeddings
            cursor = self.chroma_conn.execute(
                "SELECT COUNT(*) as count FROM embeddings_queue"
            )
            total_embeddings = cursor.fetchone()["count"]

            # Get segment info (join with embeddings table)
            cursor = self.chroma_conn.execute(
                """
                SELECT e.segment_id, COUNT(*) as count
                FROM embeddings_queue eq
                LEFT JOIN embeddings e ON eq.seq_id = e.seq_id
                GROUP BY e.segment_id
                """
            )
            segments = cursor.fetchall()

            # Get a sample vector to determine dimensions
            cursor = self.chroma_conn.execute(
                "SELECT vector FROM embeddings_queue WHERE vector IS NOT NULL LIMIT 1"
            )
            sample = cursor.fetchone()

            vector_dimensions = 0
            if sample and sample["vector"]:
                import struct

                vector_blob = sample["vector"]
                vector_dimensions = len(vector_blob) // 4

            stats = f"Total Embeddings: {total_embeddings}\n"
            stats += f"Vector Dimensions: {vector_dimensions}\n"
            stats += f"Segments: {len(segments)}\n"

            for seg in segments:
                stats += f"  - {seg['segment_id']}: {seg['count']} embeddings\n"

            return stats

        except Exception as e:
            return f"Error loading statistics:\n{str(e)}"

    def on_closing(self):
        """Handle window closing"""
        if self.db_conn:
            try:
                self.db_conn.commit()
                self.db_conn.close()
            except:
                pass
        if self.chroma_conn:
            try:
                self.chroma_conn.close()
            except:
                pass
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = MemoryManagerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
