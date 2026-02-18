"""
ChromaDB Query Tool - GUI Application
A specialized tool for querying and visualizing ChromaDB vector databases

Features:
- Select any ChromaDB database path via file dialog
- Query vectors by memory ID
- View vector statistics and embeddings
- Browse all vectors in the database
- Visualize vector data and metadata
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from pathlib import Path
import sqlite3
from datetime import datetime
from typing import Optional
import struct


class ChromaQueryTool:
    def __init__(self, root):
        self.root = root
        self.root.title("ChromaDB Query Tool")
        self.root.geometry("1000x800")

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

        # ChromaDB connection
        self.chroma_conn = None
        self.chroma_path = None

        # Build UI
        self.create_widgets()

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

    def connect_chromadb(self, path: str) -> bool:
        """Connect to ChromaDB SQLite database"""
        try:
            # Close existing connection if any
            if self.chroma_conn:
                try:
                    self.chroma_conn.close()
                except:
                    pass
                self.chroma_conn = None

            chroma_path = Path(path)
            if not chroma_path.exists():
                messagebox.showerror(
                    "Error",
                    f"ChromaDB file not found:\n{path}\n\nPlease select a valid chroma.sqlite3 file.",
                )
                return False

            self.chroma_conn = sqlite3.connect(
                str(chroma_path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level="DEFERRED",
            )
            self.chroma_conn.row_factory = sqlite3.Row
            self.chroma_path = path

            # Update connection status
            self.connection_status_label.config(
                text=f"Connected: {chroma_path.name}", foreground="#00ff00"
            )

            # Update statistics
            self.update_statistics()

            return True

        except Exception as e:
            messagebox.showerror(
                "Connection Error", f"Failed to connect to ChromaDB:\n{str(e)}"
            )
            return False

    def select_chromadb(self):
        """Open file dialog to select ChromaDB file"""
        file_path = filedialog.askopenfilename(
            title="Select ChromaDB SQLite File",
            filetypes=[
                ("SQLite files", "*.sqlite3"),
                ("SQLite files", "*.sqlite"),
                ("Database files", "*.db"),
                ("All files", "*.*"),
            ],
            initialfile="chroma.sqlite3",
        )

        if file_path:
            if self.connect_chromadb(file_path):
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert(
                    "1.0",
                    f"Successfully connected to ChromaDB!\n\n"
                    f"Path: {file_path}\n\n"
                    f"You can now:\n"
                    f"  • Query vectors by memory ID\n"
                    f"  • Show all vectors\n"
                    f"  • View statistics\n",
                )

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # ===== HEADER =====
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(0, weight=1)

        title_label = ttk.Label(
            header_frame, text="ChromaDB Query Tool", style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, sticky=tk.W)

        subtitle_label = ttk.Label(
            header_frame,
            text="Query and visualize ChromaDB vector databases",
            style="Subtitle.TLabel",
        )
        subtitle_label.grid(row=1, column=0, sticky=tk.W)

        # ===== CONNECTION SECTION =====
        connection_frame = ttk.LabelFrame(
            main_frame, text="Database Connection", padding="10"
        )
        connection_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        connection_frame.columnconfigure(1, weight=1)

        ttk.Button(
            connection_frame,
            text="Select ChromaDB...",
            command=self.select_chromadb,
            style="Accent.TButton",
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        self.connection_status_label = ttk.Label(
            connection_frame,
            text="No database connected",
            style="Normal.TLabel",
            foreground="#ff0000",
        )
        self.connection_status_label.grid(row=0, column=1, sticky=tk.W)

        # ===== STATISTICS SECTION =====
        stats_frame = ttk.LabelFrame(main_frame, text="Vector Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.columnconfigure(0, weight=1)

        self.stats_label = ttk.Label(
            stats_frame,
            text="Connect to a database to view statistics",
            style="Normal.TLabel",
            justify=tk.LEFT,
        )
        self.stats_label.grid(row=0, column=0, sticky=tk.W)

        # ===== QUERY SECTION =====
        query_frame = ttk.LabelFrame(
            main_frame, text="Query Vector by Memory ID", padding="10"
        )
        query_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        query_frame.columnconfigure(1, weight=1)

        ttk.Label(query_frame, text="Memory ID:", style="Normal.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.memory_id_var = tk.StringVar()
        memory_id_entry = ttk.Entry(
            query_frame, textvariable=self.memory_id_var, font=("Segoe UI", 10)
        )
        memory_id_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))

        ttk.Button(
            query_frame,
            text="Query Vector",
            command=self.query_vector,
            style="Accent.TButton",
        ).grid(row=0, column=2, padx=(5, 0))

        ttk.Button(
            query_frame, text="Show All Vectors", command=self.show_all_vectors
        ).grid(row=0, column=3, padx=(5, 0))

        ttk.Button(query_frame, text="Clear", command=self.clear_results).grid(
            row=0, column=4, padx=(5, 0)
        )

        # ===== RESULTS SECTION =====
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, font=("Courier New", 9), height=25
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initial message
        self.results_text.insert(
            "1.0",
            "Welcome to ChromaDB Query Tool!\n\n"
            "Click 'Select ChromaDB...' to choose a ChromaDB database file.\n\n"
            "Typical locations:\n"
            "  • ~/Documents/ai_companion_memory/memory_db/chroma_db/chroma.sqlite3\n"
            "  • Any project with ChromaDB vector storage\n",
        )

    def update_statistics(self):
        """Update the statistics display"""
        if not self.chroma_conn:
            self.stats_label.config(text="Connect to a database to view statistics")
            return

        try:
            # Count total embeddings
            cursor = self.chroma_conn.execute(
                "SELECT COUNT(*) as count FROM embeddings_queue"
            )
            total_embeddings = cursor.fetchone()["count"]

            # Get segment info
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
                vector_blob = sample["vector"]
                vector_dimensions = len(vector_blob) // 4

            stats = f"Total Embeddings: {total_embeddings}\n"
            stats += f"Vector Dimensions: {vector_dimensions}\n"
            stats += f"Segments: {len(segments)}\n"

            for seg in segments:
                stats += f"  - {seg['segment_id']}: {seg['count']} embeddings\n"

            self.stats_label.config(text=stats)

        except Exception as e:
            self.stats_label.config(text=f"Error loading statistics:\n{str(e)}")

    def query_vector(self):
        """Query vector by memory ID"""
        if not self.chroma_conn:
            messagebox.showwarning(
                "No Connection", "Please connect to a ChromaDB database first."
            )
            return

        memory_id = self.memory_id_var.get().strip()
        if not memory_id:
            messagebox.showwarning(
                "Input Required", "Please enter a memory ID to query."
            )
            return

        try:
            # Clear previous results
            self.results_text.delete("1.0", tk.END)

            # Query embeddings_queue table for the actual vector data
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
                self.results_text.insert(
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
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                output += f"  {key}: {value_str}\n"

            # Get vector from embeddings_queue
            vector_blob = embedding_row["vector"]

            if vector_blob is None:
                self.results_text.insert(
                    "1.0", f"No vector data found for memory_id: {memory_id}"
                )
                return

            # ChromaDB stores vectors as float32 arrays
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

            self.results_text.insert("1.0", output)

        except Exception as e:
            self.results_text.insert("1.0", f"Error querying vector:\n{str(e)}")

    def show_all_vectors(self):
        """Show all vectors in the database"""
        if not self.chroma_conn:
            messagebox.showwarning(
                "No Connection", "Please connect to a ChromaDB database first."
            )
            return

        try:
            self.results_text.delete("1.0", tk.END)

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

            if not rows:
                output += "No vectors found in the database.\n"
            else:
                for row in rows:
                    output += f"Memory ID: {row['id']}\n"

                    # Truncate title if too long
                    title = row["title"] or "N/A"
                    if len(title) > 60:
                        title = title[:60] + "..."
                    output += f"  Title: {title}\n"

                    output += f"  Type: {row['memory_type'] or 'N/A'}\n"
                    output += f"  Importance: {row['importance'] or 'N/A'}\n"
                    output += f"  Created: {row['created_at']}\n"
                    output += "-" * 80 + "\n"

            self.results_text.insert("1.0", output)

        except Exception as e:
            self.results_text.insert("1.0", f"Error listing vectors:\n{str(e)}")

    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete("1.0", tk.END)
        self.memory_id_var.set("")

    def on_closing(self):
        """Handle window closing"""
        if self.chroma_conn:
            try:
                self.chroma_conn.close()
            except:
                pass
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ChromaQueryTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
