#!/usr/bin/env python3
"""
System Tray Application for Long-Term Memory MCP Server.

Provides a persistent dock/taskbar icon (macOS menu bar, Windows system tray)
to start, stop, and monitor the MCP server without a terminal.

Features:
    - Start / Stop the MCP server (HTTP or stdio transport)
    - Live status indicator (green=running, red=stopped, yellow=starting)
    - Quick-launch: Memory Manager GUI, Vector Visualizer
    - View server logs
    - Backend selection (ChromaDB / pgvector)

Usage:
    python tray_app.py
    python tray_app.py --transport http --port 8000
    python tray_app.py --vector-backend pgvector

Dependencies:
    pip install pystray Pillow
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Missing required package: Pillow")
    print("Install with: pip install Pillow")
    sys.exit(1)

try:
    import pystray
    from pystray import MenuItem as Item, Menu
except ImportError:
    print("Missing required package: pystray")
    print("Install with: pip install pystray")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "data" / "logs"
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    import tempfile

    LOG_DIR = Path(tempfile.gettempdir()) / "ltm-mcp-logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "tray_app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("tray")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_SCRIPT = Path(__file__).parent / "server.py"
GUI_SCRIPT = Path(__file__).parent / "memory_manager_gui.py"
VISUALIZER_SCRIPT = Path(__file__).parent / "vector_visualizer.py"
TENSORBOARD_SCRIPT = Path(__file__).parent / "tensorboard_visualizer.py"

ICON_SIZE = 64  # px — pystray renders at the OS-appropriate resolution

# Status values
STATUS_STOPPED = "stopped"
STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_ERROR = "error"

# Cross-platform subprocess detach kwargs.
# On Windows, use CREATE_NO_WINDOW to run child processes headlessly without
# spawning a visible console window. Also add CREATE_NEW_PROCESS_GROUP so the
# child is isolated from the parent's Ctrl+C/Ctrl+Break signal group.
# NOTE: start_new_session and creationflags are mutually exclusive on Windows.
if sys.platform == "win32":
    _DETACH_KWARGS: dict = {
        "creationflags": subprocess.CREATE_NO_WINDOW
        | subprocess.CREATE_NEW_PROCESS_GROUP,
    }
else:
    _DETACH_KWARGS = {"start_new_session": True}


# ---------------------------------------------------------------------------
# Icon generation  (Pillow — no external assets needed)
# ---------------------------------------------------------------------------


def _make_icon(status: str) -> Image.Image:
    """
    Generate a tray icon programmatically.

    A filled circle with a status colour, plus a small "M" for Memory.

    Colours:
        stopped  -> red
        starting -> amber
        running  -> green
        error    -> red outline
    """
    colour_map = {
        STATUS_STOPPED: "#E05050",
        STATUS_STARTING: "#F0B030",
        STATUS_RUNNING: "#40B060",
        STATUS_ERROR: "#E05050",
    }
    fill = colour_map.get(status, "#808080")

    img = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle
    margin = 4
    draw.ellipse(
        [margin, margin, ICON_SIZE - margin, ICON_SIZE - margin],
        fill=fill,
        outline="white",
        width=2,
    )

    # "M" letter centred — try platform-appropriate fonts
    font = None
    for font_name in [
        "arial.ttf",
        "Arial",
        "DejaVuSans.ttf",
        "DejaVuSans",
        "FreeSans",
        "LiberationSans-Regular.ttf",
        "LiberationSans",
    ]:
        try:
            font = ImageFont.truetype(font_name, size=ICON_SIZE // 2)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), "M", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (ICON_SIZE - tw) / 2 - bbox[0]
    ty = (ICON_SIZE - th) / 2 - bbox[1]
    draw.text((tx, ty), "M", fill="white", font=font)

    return img


# ---------------------------------------------------------------------------
# Server Process Manager
# ---------------------------------------------------------------------------


class ServerManager:
    """Manages the MCP server subprocess lifecycle."""

    PID_FILE = LOG_DIR / "server.pid"

    def __init__(self, server_args: list[str], server_env: Optional[dict] = None):
        self._server_args = server_args
        self._server_env = server_env  # extra env vars (e.g. PGPASSWORD)
        self._process: Optional[subprocess.Popen] = None
        self._orphan_pid: Optional[int] = None
        self._status = STATUS_STOPPED
        self._lock = threading.Lock()
        self._log_thread: Optional[threading.Thread] = None
        self._server_log_file = LOG_DIR / "server_output.log"

        # On startup, try to reattach to an orphaned server from a prior crash
        self._reattach_orphan()

    def update_args(self, server_args: list[str]) -> None:
        """Update the server args used for the next start/restart."""
        self._server_args = server_args

    # ── PID file management ──────────────────────────────────────

    def _write_pid(self, pid: int) -> None:
        """Persist the server PID so we can reattach after a tray crash."""
        try:
            self.PID_FILE.write_text(str(pid), encoding="utf-8")
        except OSError as e:
            log.warning("Could not write PID file: %s", e)

    def _clear_pid(self) -> None:
        """Remove the PID file on clean shutdown."""
        try:
            self.PID_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    def _reattach_orphan(self) -> None:
        """If a PID file exists and that process is still alive, reattach to it."""
        if not self.PID_FILE.exists():
            return
        try:
            old_pid = int(self.PID_FILE.read_text().strip())
        except (ValueError, OSError):
            self._clear_pid()
            return

        # Check if the process is still running
        try:
            os.kill(old_pid, 0)  # signal 0 = existence check, no actual signal
        except (ProcessLookupError, PermissionError, OSError):
            # Process is gone — stale PID file
            # OSError covers WinError 87 (invalid parameter) on Windows for
            # processes that are in a zombie/exiting state.
            log.info("Stale PID file (pid %d is dead), cleaning up", old_pid)
            self._clear_pid()
            return

        log.info("Reattaching to existing server (pid %d)", old_pid)
        # We can't get a Popen handle for a process we didn't start, but we
        # can track its PID for stop() via os.kill.
        self._status = STATUS_RUNNING
        # Store the PID in a lightweight wrapper so stop() can terminate it
        self._orphan_pid = old_pid

    # ── Properties ──────────────────────────────────────────────

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def pid(self) -> Optional[int]:
        with self._lock:
            if self._process:
                return self._process.pid
            return self._orphan_pid

    def start(self) -> bool:
        """Start the server subprocess. Returns True on success."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                log.info("Server already running (pid %d)", self._process.pid)
                return True
            if self._orphan_pid is not None:
                log.info("Server already running as orphan (pid %d)", self._orphan_pid)
                return True
            if self._status == STATUS_STARTING:
                log.info("Server already starting, ignoring duplicate start()")
                return True

            self._status = STATUS_STARTING

        cmd = [sys.executable, str(SERVER_SCRIPT)] + self._server_args
        log.info("Starting server: %s", " ".join(cmd))

        try:
            # Build subprocess environment — inherit current env + any extra vars
            env = None
            if self._server_env:
                env = {**os.environ, **self._server_env}

            # Open log file, pass to Popen, then close in parent.
            # The child inherits its own fd and keeps it open independently.
            with open(self._server_log_file, "a", encoding="utf-8") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    # Detach from tray's process group so killing tray doesn't kill server
                    **_DETACH_KWARGS,
                )
        except Exception as e:
            log.error("Failed to start server: %s", e)
            with self._lock:
                self._status = STATUS_ERROR
            return False

        with self._lock:
            self._process = proc
            self._status = STATUS_RUNNING
            self._orphan_pid = None  # clear any orphan — we own this process now

        self._write_pid(proc.pid)
        log.info("Server started (pid %d)", proc.pid)

        # Start a background thread to watch for unexpected exits
        self._log_thread = threading.Thread(
            target=self._watch_process, daemon=True, name="server-watcher"
        )
        self._log_thread.start()

        return True

    def stop(self) -> bool:
        """Stop the server subprocess gracefully. Returns True on success."""
        with self._lock:
            orphan_pid = self._orphan_pid
            if self._process is None and orphan_pid is None:
                self._status = STATUS_STOPPED
                self._clear_pid()
                return True
            if self._process is None and orphan_pid is not None:
                # Orphan from a previous tray session — kill by PID
                log.info("Stopping orphan server (pid %d)...", orphan_pid)
                try:
                    import psutil as _ps

                    _ps.Process(orphan_pid).terminate()
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                except Exception:
                    pass
                self._orphan_pid = None
                self._status = STATUS_STOPPED
                self._clear_pid()
                log.info("Orphan server stopped")
                return True
            if self._process is not None and self._process.poll() is not None:
                self._process = None
                self._status = STATUS_STOPPED
                self._clear_pid()
                return True
            proc = self._process
            assert proc is not None  # guaranteed by the checks above

        log.info("Stopping server (pid %d)...", proc.pid)

        # Send SIGTERM for graceful shutdown (WAL checkpoint, etc.)
        try:
            proc.terminate()
        except OSError:
            pass

        # Wait up to 10s for graceful exit
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("Server did not stop in 10s, sending SIGKILL")
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception as e:
                log.error("Failed to kill server process: %s", e)
                with self._lock:
                    self._status = STATUS_ERROR
                return False

        with self._lock:
            self._process = None
            self._orphan_pid = None
            self._status = STATUS_STOPPED

        self._clear_pid()
        log.info("Server stopped")
        return True

    def restart(self) -> bool:
        """Stop then start the server."""
        if not self.stop():
            log.error("Restart aborted: could not stop server")
            return False
        time.sleep(0.5)
        return self.start()

    def _watch_process(self):
        """Background thread that updates status when process exits."""
        with self._lock:
            proc = self._process
        if proc is None:
            return
        proc.wait()
        with self._lock:
            if self._process is proc:  # still the same process
                rc = proc.returncode
                self._status = STATUS_ERROR if rc != 0 else STATUS_STOPPED
                self._process = None
                log.info("Server exited (code %d)", rc)


# ---------------------------------------------------------------------------
# Activity Monitor (web-based — avoids macOS tkinter main-thread crash)
# ---------------------------------------------------------------------------

_ACTIVITY_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Activity Monitor</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: #1e1e2e; color: #d0d0d8;
    font-family: 'SF Mono','Consolas','Courier New',monospace;
    font-size: 13px; padding: 20px 28px; line-height: 1.7;
  }
  h1 { font-size: 16px; color: #e0e0f0; margin-bottom: 4px;
       border-bottom: 1px solid #3a3a4a; padding-bottom: 8px; }
  .header-row { display: flex; justify-content: space-between;
                align-items: baseline; margin-bottom: 16px; }
  .last-updated { color: #404055; font-size: 11px; }
  .section { margin-bottom: 18px; }
  .section-title {
    color: #6a8aba; font-weight: 700; font-size: 13px; margin-bottom: 4px;
  }
  .row { display: flex; gap: 12px; padding: 2px 0 2px 12px; }
  .label { color: #808090; min-width: 130px; }
  .val { color: #d0d0d8; }
  .val.good { color: #38BE70; }
  .val.warn { color: #F4B042; }
  .val.bad  { color: #DC4E4E; }
  .val.dim  { color: #606070; }
  .process-block { margin-bottom: 10px; padding-left: 12px; }
  .proc-header { color: #c0c0d0; font-weight: 600; }
  .sub { padding-left: 24px; color: #a0a0b0; }
  hr { border: none; border-top: 1px solid #2a2a3a; margin: 12px 0; }
</style>
</head>
<body>
<div class="header-row">
  <h1>Activity Monitor</h1>
  <span class="last-updated" id="last-updated">Loading...</span>
</div>
<div id="content">Loading...</div>
<hr>
<script>
  async function refresh() {
    try {
      const res = await fetch('/data');
      if (!res.ok) return;
      const d = await res.json();

      function row(label, value, css) {
        const cls = css ? `val ${css}` : 'val';
        return `<div class="row"><span class="label">${label}</span>`
             + `<span class="${cls}">${value}</span></div>`;
      }

      let html = '';

      if (d.system) {
        const s = d.system;
        html += '<div class="section"><div class="section-title">System</div>';
        html += row('CPU', `${s.cpu_percent}% (${s.cpu_count} cores)`);
        html += row('Memory', `${s.mem_used} / ${s.mem_total} (${s.mem_percent}%)`);
        html += '</div>';
      }

      if (d.processes && d.processes.length) {
        html += '<div class="section"><div class="section-title">Processes</div>';
        for (const p of d.processes) {
          const statusCss = ['running','sleeping','idle'].includes(p.status) ? 'good'
                          : p.status === 'dead' ? 'bad'
                          : p.status === 'stopped' ? 'dim' : 'warn';
          html += '<div class="process-block">';
          html += `<div class="proc-header">${p.name}</div>`;
          html += `<div class="sub">PID: ${p.pid || '\u2014'} &nbsp; `
               +  `Status: <span class="val ${statusCss}">${p.status}</span></div>`;
          if (p.pid && p.rss) {
            html += `<div class="sub">CPU: ${p.cpu} &nbsp; RSS: ${p.rss}`;
            if (p.children) html += ` (total w/${p.n_children} children: ${p.total_rss})`;
            html += '</div>';
            if (p.uptime) html += `<div class="sub">Uptime: ${p.uptime}</div>`;
          }
          html += '</div>';
        }
        html += '</div>';
      }

      if (d.gpu) {
        const g = d.gpu;
        html += '<div class="section"><div class="section-title">GPU</div>';
        html += row('Device', g.name);
        if (g.total) html += row('VRAM', `${g.allocated} allocated / ${g.total} total`);
        else if (g.allocated) html += row('Allocated', g.allocated);
        html += '</div>';
      }

      if (d.vectors) {
        const v = d.vectors;
        html += '<div class="section"><div class="section-title">Vector Storage</div>';
        html += row('Backend', v.backend || '?');
        html += row('Model', v.model || '?');
        html += row('Dimensions', v.dimensions || '?');
        html += row('Vectors', String(v.count ?? '?'));
        if (v.table_size) html += row('Table Size', v.table_size);
        if (v.raw_embed) html += row('Raw Embeddings', v.raw_embed);
        html += '</div>';
      }

      document.getElementById('content').innerHTML = html;
      const now = new Date();
      document.getElementById('last-updated').textContent =
        'Updated ' + now.toLocaleTimeString();
    } catch (e) {
      document.getElementById('last-updated').textContent = 'Error: ' + e.message;
    }
  }

  refresh();
  setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class ActivityMonitor:
    """Lightweight web-based activity monitor.

    Runs a tiny HTTP server on a background thread, serves a
    self-refreshing HTML page with process and vector storage stats.
    Opens the page in the default browser.  Avoids tkinter entirely
    (tkinter crashes on macOS when called from a non-main thread).
    """

    PORT = 8051  # one above the visualizer default

    def __init__(
        self,
        tray_app: "TrayApp",
        pg_cfg: Optional[dict] = None,
    ):
        self._tray = tray_app
        self._pg_cfg = pg_cfg or {}
        self._server = None
        self._thread = None

        try:
            import psutil as _ps

            self._psutil = _ps
        except ImportError:
            self._psutil = None
            log.warning("psutil not installed — process stats unavailable")

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _human_bytes(n) -> str:
        n = float(n)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    @staticmethod
    def _human_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"

    def _proc_info(self, pid: Optional[int], name: str) -> dict:
        info: dict = {"name": name, "pid": pid, "status": "stopped"}
        if pid is None or self._psutil is None:
            return info
        try:
            p = self._psutil.Process(pid)
            with p.oneshot():
                info["status"] = p.status()
                info["cpu"] = p.cpu_percent(interval=None)
                mem = p.memory_info()
                info["rss"] = mem.rss
                info["vms"] = mem.vms
                info["create_time"] = p.create_time()
                info["uptime"] = time.time() - p.create_time()
                children = p.children(recursive=True)
                info["n_children"] = len(children)
                child_rss = sum(c.memory_info().rss for c in children if c.is_running())
                info["total_rss"] = mem.rss + child_rss
        except Exception:
            info["status"] = "dead"
        return info

    def _find_postgres_pid(self) -> Optional[int]:
        if self._psutil is None:
            return None
        port = self._pg_cfg.get("pg_port") or 5433
        return self._find_pid_by_port(port)

    def _find_pid_by_port(self, port: int) -> Optional[int]:
        """Find a process listening on the given TCP port."""
        # Try psutil first (works on Linux, may fail on macOS without root)
        if self._psutil:
            try:
                for conn in self._psutil.net_connections(kind="inet"):
                    if (
                        conn.laddr
                        and conn.laddr.port == port
                        and conn.status == "LISTEN"
                    ):
                        return conn.pid
            except (self._psutil.AccessDenied, OSError):
                if sys.platform == "win32":
                    log.debug(
                        "psutil.net_connections() requires admin on Windows; "
                        "falling back to netstat"
                    )
                pass
        # Fallback: netstat on Windows, lsof on macOS/Linux
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["netstat", "-ano", "-p", "TCP"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if f":{port}" in line and "LISTENING" in line:
                            parts = line.split()
                            if parts:
                                try:
                                    return int(parts[-1])
                                except ValueError:
                                    pass
            else:
                result = subprocess.run(
                    ["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-t"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return int(result.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return None

    def _find_pid_by_script(self, script_name: str) -> Optional[int]:
        """Find a running Python process whose cmdline contains *script_name*."""
        if self._psutil is None:
            return None
        try:
            for proc in self._psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline") or []
                    if any(script_name in arg for arg in cmdline):
                        return proc.info["pid"]
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return None

    def _get_gpu_info(self) -> Optional[dict]:
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "name": torch.cuda.get_device_name(0),
                    "allocated": torch.cuda.memory_allocated(0),
                    "reserved": torch.cuda.memory_reserved(0),
                    "total": torch.cuda.get_device_properties(0).total_mem,
                }
        except Exception:
            pass
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                alloc = 0
                if hasattr(torch.mps, "current_allocated_memory"):
                    alloc = torch.mps.current_allocated_memory()
                return {
                    "name": "Apple Silicon (MPS)",
                    "allocated": alloc,
                    "reserved": 0,
                    "total": 0,
                }
        except Exception:
            pass
        return None

    def _get_vector_stats(self) -> dict:
        # Cache vector stats for 30 seconds to avoid hammering the DB
        now = time.time()
        if (
            hasattr(self, "_vstats_cache")
            and self._vstats_cache
            and now - self._vstats_cache_time < 30
        ):
            return self._vstats_cache
        stats = self._fetch_vector_stats()
        self._vstats_cache = stats
        self._vstats_cache_time = now
        return stats

    def _fetch_vector_stats(self) -> dict:
        stats: dict = {}
        try:
            from memory_mcp.config import EMBEDDING_MODEL_CONFIG

            stats["dimensions"] = EMBEDDING_MODEL_CONFIG.get("dimensions", "?")
            stats["model"] = EMBEDDING_MODEL_CONFIG.get("model_name", "?")
        except Exception:
            stats["dimensions"] = "?"
            stats["model"] = "?"

        pg_cfg = self._pg_cfg
        if pg_cfg.get("backend_type") == "pgvector":
            try:
                from memory_mcp.vector_backends.pgvector_backend import PgvectorBackend
                from memory_mcp.config import EMBEDDING_MODEL_CONFIG

                backend = PgvectorBackend(
                    host=pg_cfg.get("pg_host"),
                    port=pg_cfg.get("pg_port"),
                    database=pg_cfg.get("pg_database"),
                    user=pg_cfg.get("pg_user"),
                    password=pg_cfg.get("pg_password"),
                    dimensions=EMBEDDING_MODEL_CONFIG.get("dimensions", 384),
                )
                backend.initialize()
                stats["count"] = backend.count()
                stats["storage_bytes"] = backend.storage_size_bytes()
                stats["backend"] = "pgvector"
                backend.close()
            except Exception as exc:
                stats["count"] = f"error: {exc}"
                stats["backend"] = "pgvector (error)"
        else:
            try:
                from memory_mcp.vector_backends.chroma import ChromaBackend
                from memory_mcp.config import DATA_FOLDER

                backend = ChromaBackend(db_folder=Path(DATA_FOLDER) / "memory_db")
                backend.initialize()
                stats["count"] = backend.count()
                stats["backend"] = "chromadb"
                chroma_dir = Path(DATA_FOLDER) / "memory_db" / "chroma_db"
                if chroma_dir.exists():
                    total = sum(
                        f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file()
                    )
                    stats["storage_bytes"] = total
                backend.close()
            except Exception as exc:
                stats["count"] = f"error: {exc}"
                stats["backend"] = "chromadb (error)"
        return stats

    # ── Data generation ─────────────────────────────────────────

    def _build_data(self) -> dict:
        """Build a JSON-serialisable dict of current stats for the /data endpoint."""
        data: dict = {}

        # ── System ──────────────────────────────────────────────
        if self._psutil:
            try:
                vm = self._psutil.virtual_memory()
                cpu = self._psutil.cpu_percent(interval=None)
                data["system"] = {
                    "cpu_percent": cpu,
                    "cpu_count": self._psutil.cpu_count(),
                    "mem_used": self._human_bytes(vm.used),
                    "mem_total": self._human_bytes(vm.total),
                    "mem_percent": vm.percent,
                }
            except Exception:
                pass

        # ── Processes ───────────────────────────────────────────
        processes_out = []

        server_pid = self._tray._server.pid
        procs_to_gather = [
            (server_pid, "MCP Server"),
        ]

        gui_pid = None
        with self._tray._subprocess_lock:
            if self._tray._gui_proc and self._tray._gui_proc.poll() is None:
                gui_pid = self._tray._gui_proc.pid
        if gui_pid is None:
            gui_pid = self._find_pid_by_script("memory_manager_gui.py")
        procs_to_gather.append((gui_pid, "Memory Manager"))

        viz_pid = None
        with self._tray._subprocess_lock:
            if (
                self._tray._visualizer_proc
                and self._tray._visualizer_proc.poll() is None
            ):
                viz_pid = self._tray._visualizer_proc.pid
        if viz_pid is None:
            viz_pid = self._find_pid_by_port(self._tray.VISUALIZER_PORT)
        procs_to_gather.append((viz_pid, "Visualizer"))

        tb_pid = None
        with self._tray._subprocess_lock:
            if (
                self._tray._tensorboard_proc
                and self._tray._tensorboard_proc.poll() is None
            ):
                tb_pid = self._tray._tensorboard_proc.pid
        if tb_pid is None:
            tb_pid = self._find_pid_by_port(self._tray.TENSORBOARD_PORT)
        procs_to_gather.append((tb_pid, "TensorBoard"))

        pg_pid = self._find_postgres_pid()
        if pg_pid:
            procs_to_gather.append((pg_pid, "PostgreSQL"))

        for pid, name in procs_to_gather:
            info = self._proc_info(pid, name)
            entry: dict = {
                "name": info["name"],
                "pid": info["pid"],
                "status": info.get("status", "stopped"),
            }
            if info["pid"] and info.get("rss"):
                entry["cpu"] = f"{info.get('cpu', 0):.1f}%"
                entry["rss"] = self._human_bytes(info["rss"])
                entry["n_children"] = info.get("n_children", 0)
                if info.get("n_children", 0) > 0:
                    entry["children"] = True
                    entry["total_rss"] = self._human_bytes(info.get("total_rss", 0))
                if "uptime" in info:
                    entry["uptime"] = self._human_duration(info["uptime"])
            processes_out.append(entry)

        data["processes"] = processes_out

        # ── GPU ─────────────────────────────────────────────────
        gpu = self._get_gpu_info()
        if gpu:
            gpu_out: dict = {"name": gpu.get("name", "?")}
            if gpu.get("total"):
                gpu_out["allocated"] = self._human_bytes(gpu["allocated"])
                gpu_out["total"] = self._human_bytes(gpu["total"])
            elif gpu.get("allocated"):
                gpu_out["allocated"] = self._human_bytes(gpu["allocated"])
            data["gpu"] = gpu_out

        # ── Vector Storage ──────────────────────────────────────
        try:
            vstats = self._get_vector_stats()
            vec_out: dict = {
                "backend": str(vstats.get("backend", "?")),
                "model": str(vstats.get("model", "?")),
                "dimensions": str(vstats.get("dimensions", "?")),
                "count": vstats.get("count", "?"),
            }
            if "storage_bytes" in vstats:
                vec_out["table_size"] = self._human_bytes(vstats["storage_bytes"])
            count = vstats.get("count")
            dims = vstats.get("dimensions")
            if isinstance(count, int) and dims:
                try:
                    raw_bytes = count * int(dims) * 4
                    vec_out["raw_embed"] = (
                        f"{self._human_bytes(raw_bytes)} ({count} x {dims} x 4B)"
                    )
                except (ValueError, TypeError):
                    pass
            data["vectors"] = vec_out
        except Exception as exc:
            data["vectors"] = {"backend": f"error: {exc}"}

        return data

    # ── Server lifecycle ────────────────────────────────────────

    def show(self):
        """Start the HTTP server (if not running) and open the browser."""
        import webbrowser
        from http.server import HTTPServer, BaseHTTPRequestHandler

        if self._server is not None:
            webbrowser.open(f"http://127.0.0.1:{self.PORT}")
            return

        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/data":
                    import json as _json

                    payload = _json.dumps(monitor._build_data()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    page = _ACTIVITY_HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)

            def log_message(self, format, *args):
                pass  # suppress request logging

        try:
            self._server = HTTPServer(("127.0.0.1", self.PORT), Handler)
        except OSError as exc:
            log.error("Activity Monitor: port %d busy: %s", self.PORT, exc)
            webbrowser.open(f"http://127.0.0.1:{self.PORT}")
            return

        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="activity-monitor"
        )
        self._thread.start()
        log.info("Activity Monitor started on http://127.0.0.1:%d", self.PORT)
        webbrowser.open(f"http://127.0.0.1:{self.PORT}")

    def stop(self):
        """Shutdown the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
            log.info("Activity Monitor stopped")


# ---------------------------------------------------------------------------
# Autostart Manager
# ---------------------------------------------------------------------------

_AUTOSTART_APP_NAME = "LongTermMemoryMCP"
_AUTOSTART_TRAY_SCRIPT = Path(__file__).resolve()

# macOS
_MACOS_PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
_MACOS_PLIST_NAME = f"com.{_AUTOSTART_APP_NAME.lower()}.tray"
_MACOS_PLIST_PATH = _MACOS_PLIST_DIR / f"{_MACOS_PLIST_NAME}.plist"

# Linux
_LINUX_SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
_LINUX_SERVICE_NAME = f"{_AUTOSTART_APP_NAME.lower()}-tray.service"
_LINUX_SERVICE_PATH = _LINUX_SYSTEMD_DIR / _LINUX_SERVICE_NAME


class AutostartManager:
    """Install / uninstall / query the tray app as a login autostart item."""

    def is_enabled(self) -> bool:
        """Return True if autostart is currently registered."""
        try:
            if sys.platform == "win32":
                return self._windows_is_enabled()
            elif sys.platform == "darwin":
                return _MACOS_PLIST_PATH.exists()
            else:
                return _LINUX_SERVICE_PATH.exists()
        except Exception as e:
            log.warning("AutostartManager.is_enabled error: %s", e)
            return False

    def enable(self, extra_args: Optional[list] = None) -> bool:
        """Register autostart. Returns True on success."""
        extra_args = extra_args or []
        try:
            if sys.platform == "win32":
                self._windows_enable(extra_args)
            elif sys.platform == "darwin":
                self._macos_enable(extra_args)
            else:
                self._linux_enable(extra_args)
            log.info("Autostart enabled")
            return True
        except Exception as e:
            log.error("Failed to enable autostart: %s", e)
            return False

    def disable(self) -> bool:
        """Unregister autostart. Returns True on success."""
        try:
            if sys.platform == "win32":
                self._windows_disable()
            elif sys.platform == "darwin":
                self._macos_disable()
            else:
                self._linux_disable()
            log.info("Autostart disabled")
            return True
        except Exception as e:
            log.error("Failed to disable autostart: %s", e)
            return False

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _cmd(extra_args: list) -> list:
        return [sys.executable, str(_AUTOSTART_TRAY_SCRIPT)] + extra_args

    # Windows ──────────────────────────────────────────────────

    @staticmethod
    def _reg_key():
        import winreg

        return winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_READ | winreg.KEY_SET_VALUE,
        )

    def _windows_is_enabled(self) -> bool:
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_READ,
            ) as key:
                winreg.QueryValueEx(key, _AUTOSTART_APP_NAME)
            return True
        except FileNotFoundError:
            return False

    def _windows_enable(self, extra_args: list) -> None:
        import winreg

        cmd_str = subprocess.list2cmdline(self._cmd(extra_args))
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, _AUTOSTART_APP_NAME, 0, winreg.REG_SZ, cmd_str)

    def _windows_disable(self) -> None:
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                winreg.DeleteValue(key, _AUTOSTART_APP_NAME)
        except FileNotFoundError:
            pass

    # macOS ────────────────────────────────────────────────────

    def _macos_enable(self, extra_args: list) -> None:
        cmd = self._cmd(extra_args)
        log_dir = Path.home() / "Library" / "Logs" / _AUTOSTART_APP_NAME
        log_dir.mkdir(parents=True, exist_ok=True)
        _MACOS_PLIST_DIR.mkdir(parents=True, exist_ok=True)
        program_args = "\n        ".join(f"<string>{a}</string>" for a in cmd)

        # Resolve certifi's CA bundle path from the same Python that runs the tray.
        # If certifi isn't installed this stays empty and the key is omitted.
        try:
            import certifi as _certifi

            _ca_bundle = _certifi.where()
        except ImportError:
            _ca_bundle = None

        # Build the EnvironmentVariables block.
        # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: tell huggingface_hub to load
        # the embedding model straight from the local cache — no network needed.
        # Without this the launchd environment lacks a valid CA bundle, so TLS
        # to huggingface.co fails and the server crashes on every autostart.
        env_entries = (
            "    <key>EnvironmentVariables</key>\n"
            "    <dict>\n"
            "        <key>HF_HUB_OFFLINE</key><string>1</string>\n"
            "        <key>TRANSFORMERS_OFFLINE</key><string>1</string>\n"
        )
        if _ca_bundle:
            env_entries += (
                f"        <key>SSL_CERT_FILE</key><string>{_ca_bundle}</string>\n"
                f"        <key>REQUESTS_CA_BUNDLE</key><string>{_ca_bundle}</string>\n"
            )
        env_entries += "    </dict>"

        plist = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{_MACOS_PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        {program_args}
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><false/>
{env_entries}
    <key>StandardOutPath</key><string>{log_dir}/tray_stdout.log</string>
    <key>StandardErrorPath</key><string>{log_dir}/tray_stderr.log</string>
</dict>
</plist>
"""
        _MACOS_PLIST_PATH.write_text(plist, encoding="utf-8")
        subprocess.run(
            ["launchctl", "unload", str(_MACOS_PLIST_PATH)], capture_output=True
        )
        subprocess.run(
            ["launchctl", "load", str(_MACOS_PLIST_PATH)], capture_output=True
        )

    def _macos_disable(self) -> None:
        if _MACOS_PLIST_PATH.exists():
            subprocess.run(
                ["launchctl", "unload", str(_MACOS_PLIST_PATH)], capture_output=True
            )
            _MACOS_PLIST_PATH.unlink()

    # Linux ────────────────────────────────────────────────────

    def _linux_enable(self, extra_args: list) -> None:
        cmd = self._cmd(extra_args)
        exec_start = " ".join(f'"{a}"' if " " in a else a for a in cmd)
        _LINUX_SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
        _LINUX_SERVICE_PATH.write_text(
            f"[Unit]\nDescription=Long-Term Memory MCP Tray App\n"
            f"After=graphical-session.target\n\n"
            f"[Service]\nType=simple\nExecStart={exec_start}\n"
            f"Restart=on-failure\nRestartSec=5\n\n"
            f"[Install]\nWantedBy=default.target\n",
            encoding="utf-8",
        )
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", _LINUX_SERVICE_NAME],
            capture_output=True,
        )

    def _linux_disable(self) -> None:
        if _LINUX_SERVICE_PATH.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", _LINUX_SERVICE_NAME],
                capture_output=True,
            )
            _LINUX_SERVICE_PATH.unlink()
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"], capture_output=True
            )


# ---------------------------------------------------------------------------
# Tray Application
# ---------------------------------------------------------------------------


class TrayApp:
    """System tray application wrapping pystray + ServerManager."""

    VISUALIZER_PORT = 8050
    TENSORBOARD_PORT = 6006

    def __init__(
        self,
        server_args: list[str],
        auto_start: bool = False,
        server_env: Optional[dict] = None,
        visualizer_args: Optional[list[str]] = None,
        visualizer_env: Optional[dict] = None,
        pg_cfg: Optional[dict] = None,
    ):
        self._server = ServerManager(server_args, server_env=server_env)
        self._auto_start = auto_start
        self._icon: Optional[pystray.Icon] = None
        self._refresh_thread: Optional[threading.Thread] = None
        self._running = True
        self._visualizer_args = visualizer_args or []
        self._visualizer_env = visualizer_env  # extra env vars (e.g. PGPASSWORD)
        self._visualizer_proc: Optional[subprocess.Popen] = None
        self._tensorboard_proc: Optional[subprocess.Popen] = None
        self._gui_proc: Optional[subprocess.Popen] = None
        self._subprocess_lock = (
            threading.Lock()
        )  # protects _visualizer_proc/_tensorboard_proc/_gui_proc
        self._activity_monitor = ActivityMonitor(self, pg_cfg=pg_cfg)
        self._autostart = AutostartManager()
        # Remember the args used to launch this tray so autostart can replay them
        self._server_args = server_args

    def _build_subprocess_env(self) -> Optional[dict]:
        """Build subprocess environment with PGPASSWORD if configured."""
        if self._visualizer_env:
            return {**os.environ, **self._visualizer_env}
        return None

    def _find_pid_by_port(self, port: int) -> Optional[int]:
        """Find a process listening on the given TCP port."""
        # Try psutil first (works on Linux, may fail on macOS/Windows without root/admin)
        try:
            import psutil as _psutil2

            for conn in _psutil2.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN":
                    return conn.pid
        except Exception as _e:
            if sys.platform == "win32" and "access" in str(_e).lower():
                log.debug(
                    "psutil.net_connections() requires admin on Windows; "
                    "falling back to netstat"
                )
        # Fallback: netstat on Windows, lsof on macOS/Linux
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["netstat", "-ano", "-p", "TCP"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if f":{port}" in line and "LISTENING" in line:
                            parts = line.split()
                            if parts:
                                try:
                                    return int(parts[-1])
                                except ValueError:
                                    pass
            else:
                result = subprocess.run(
                    ["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-t"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return int(result.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return None

    def _kill_by_port(self, port: int, name: str) -> bool:
        """Find and terminate a process listening on a port. Returns True if killed."""
        pid = self._find_pid_by_port(port)
        if pid is None:
            return False
        log.info("Killing orphan %s (pid %d) on port %d", name, pid, port)
        try:
            import psutil as _psutil

            p = _psutil.Process(pid)
            p.terminate()  # SIGTERM on Unix, TerminateProcess on Windows
            try:
                p.wait(timeout=5)
            except Exception:
                # Force kill if still alive after grace period
                try:
                    p.kill()  # SIGKILL on Unix, TerminateProcess on Windows
                    p.wait(timeout=3)
                except Exception:
                    pass
        except (ProcessLookupError, PermissionError, OSError):
            pass  # already dead or no permission
        except Exception as e:
            log.warning("Failed to kill pid %d: %s", pid, e)
            return False
        log.info("Orphan %s (pid %d) stopped", name, pid)
        return True

    # ── Menu builders ───────────────────────────────────────────

    def _build_menu(self) -> Menu:
        status = self._server.status
        pid = self._server.pid

        status_label = {
            STATUS_STOPPED: "Status: Stopped",
            STATUS_STARTING: "Status: Starting...",
            STATUS_RUNNING: f"Status: Running (pid {pid})",
            STATUS_ERROR: "Status: Error",
        }.get(status, "Status: Unknown")

        is_running = status == STATUS_RUNNING
        viz_running = self._is_visualizer_running()
        tb_running = self._is_tensorboard_running()

        # Build GUI sub-menu: context-aware based on running state
        gui_running = self._is_gui_running()
        if gui_running:
            gui_items = Menu(
                Item("Stop Memory Manager", self._on_stop_gui_menu),
            )
        else:
            gui_items = None

        # Build visualizer sub-menu: context-aware based on running state
        if viz_running:
            viz_items = Menu(
                Item("Restart Visualizer", self._on_restart_visualizer),
                Item(
                    f"Stop Visualizer (port {self.VISUALIZER_PORT})",
                    self._on_stop_visualizer,
                ),
            )
        else:
            viz_items = None

        # Build TensorBoard sub-menu: context-aware based on running state
        if tb_running:
            tb_items = Menu(
                Item("Restart TensorBoard", self._on_restart_tensorboard),
                Item(
                    f"Stop TensorBoard (port {self.TENSORBOARD_PORT})",
                    self._on_stop_tensorboard,
                ),
            )
        else:
            tb_items = None

        return Menu(
            Item(status_label, None, enabled=False),
            Menu.SEPARATOR,
            Item("Start Server", self._on_start, enabled=not is_running),
            Item("Stop Server", self._on_stop, enabled=is_running),
            Item("Restart Server", self._on_restart, enabled=is_running),
            Menu.SEPARATOR,
            Item("Open Memory Manager", self._on_open_gui)
            if not gui_running
            else Item("Memory Manager", gui_items),
            Item("Open Vector Visualizer", self._on_open_visualizer)
            if not viz_running
            else Item("Vector Visualizer", viz_items),
            Item("Open TensorBoard Projector", self._on_open_tensorboard)
            if not tb_running
            else Item("TensorBoard Projector", tb_items),
            Menu.SEPARATOR,
            Item("Activity Monitor", self._on_activity_monitor),
            Item("View Server Log", self._on_view_log),
            Item("View Visualizer Log", self._on_view_visualizer_log),
            Menu.SEPARATOR,
            Item(
                "Start at Login",
                self._on_toggle_autostart,
                checked=lambda item: self._autostart.is_enabled(),
            ),
            Item(
                "Network Sharing",
                self._on_toggle_network_sharing,
                checked=lambda item: self._network_sharing_enabled(),
            ),
            Menu.SEPARATOR,
            Item("Quit", self._on_quit),
        )

    # ── Menu actions ────────────────────────────────────────────

    def _on_toggle_autostart(self, icon, item):
        """Toggle 'Start at Login' on/off."""
        try:
            if self._autostart.is_enabled():
                self._autostart.disable()
            else:
                autostart_args = ["--auto-start"] + self._server_args
                self._autostart.enable(extra_args=autostart_args)
            self._refresh_icon()
        except Exception as e:
            log.error("Autostart toggle failed: %s", e, exc_info=True)

    # ── Network sharing toggle ───────────────────────────────────

    def _network_sharing_enabled(self) -> bool:
        """Return True if --network-sharing is present in the current server args."""
        return "--network-sharing" in self._server_args

    def _on_toggle_network_sharing(self, icon, item):
        """Toggle LAN memory sharing on/off.

        Updates _server_args, restarts the server if it was running, and
        re-registers autostart so the plist reflects the new state.

        Wrapped in a broad try/except because pystray silently kills the
        icon if any unhandled exception escapes a menu callback.
        """
        try:
            if self._network_sharing_enabled():
                # Remove --network-sharing (and adjacent --sharing-poll-interval if present)
                new_args = []
                skip_next = False
                for arg in self._server_args:
                    if skip_next:
                        skip_next = False
                        continue
                    if arg == "--network-sharing":
                        continue
                    if arg == "--sharing-poll-interval":
                        skip_next = True
                        continue
                    new_args.append(arg)
                self._server_args = new_args
            else:
                self._server_args = self._server_args + ["--network-sharing"]

            # Propagate the new args to the ServerManager
            self._server.update_args(self._server_args)

            # Restart the server if it's currently running so the change takes effect
            was_running = self._server.status == STATUS_RUNNING
            if was_running:
                threading.Thread(target=self._do_restart, daemon=True).start()

            # Keep autostart in sync if it's enabled
            if self._autostart.is_enabled():
                try:
                    self._autostart.disable()
                    autostart_args = ["--auto-start"] + self._server_args
                    self._autostart.enable(extra_args=autostart_args)
                except Exception as ae:
                    log.error(
                        "Failed to update autostart after network sharing toggle: %s",
                        ae,
                    )

            self._refresh_icon()

        except Exception as e:
            log.error("Network sharing toggle failed: %s", e, exc_info=True)

    def _on_start(self, icon, item):
        threading.Thread(target=self._do_start, daemon=True).start()

    def _do_start(self):
        self._update_icon(STATUS_STARTING)
        ok = self._server.start()
        if ok:
            # Brief delay to let server bind its port
            time.sleep(1.5)
        self._refresh_icon()

    def _on_stop(self, icon, item):
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _do_stop(self):
        self._server.stop()
        self._refresh_icon()

    def _on_restart(self, icon, item):
        threading.Thread(target=self._do_restart, daemon=True).start()

    def _do_restart(self):
        self._update_icon(STATUS_STARTING)
        self._server.restart()
        time.sleep(1.5)
        self._refresh_icon()

    def _on_open_gui(self, icon, item):
        """Launch the Memory Manager GUI (single instance)."""
        if not GUI_SCRIPT.exists():
            log.warning("GUI script not found: %s", GUI_SCRIPT)
            return
        if self._is_gui_running():
            log.info("Memory Manager already running")
            return
        log.info("Launching Memory Manager GUI")
        proc = subprocess.Popen(
            [sys.executable, str(GUI_SCRIPT)],
            **_DETACH_KWARGS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._subprocess_lock:
            self._gui_proc = proc

    def _is_gui_running(self) -> bool:
        """Check if the Memory Manager GUI is alive (proc or cmdline scan)."""
        with self._subprocess_lock:
            if self._gui_proc is not None:
                if self._gui_proc.poll() is None:
                    return True
                self._gui_proc = None
        # Fallback: scan for python process running the script
        pid = self._activity_monitor._find_pid_by_script("memory_manager_gui.py")
        return pid is not None

    def _stop_gui(self) -> None:
        """Terminate the Memory Manager GUI."""
        with self._subprocess_lock:
            proc = self._gui_proc
            if proc is not None and proc.poll() is not None:
                self._gui_proc = None
                proc = None
        if proc is None:
            # Try finding by cmdline
            pid = self._activity_monitor._find_pid_by_script("memory_manager_gui.py")
            if pid:
                log.info("Killing orphan Memory Manager (pid %d)", pid)
                try:
                    import psutil as _ps

                    _ps.Process(pid).terminate()
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                except Exception:
                    pass
            return
        log.info("Stopping Memory Manager (pid %d)", proc.pid)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception as e:
                log.error("Failed to kill Memory Manager: %s", e)
        except Exception as e:
            log.error("Error stopping Memory Manager: %s", e)
        with self._subprocess_lock:
            self._gui_proc = None
        log.info("Memory Manager stopped")

    def _on_stop_gui_menu(self, icon, item):
        """Menu handler to stop the Memory Manager."""
        threading.Thread(target=self._stop_gui, daemon=True).start()

    def _on_open_visualizer(self, icon, item):
        """Launch the Vector Visualizer Dash app and open the browser."""
        if not VISUALIZER_SCRIPT.exists():
            log.warning("Visualizer script not found: %s", VISUALIZER_SCRIPT)
            return
        if self._is_visualizer_running():
            log.info("Visualizer already running, opening browser")
            self._open_visualizer_browser()
            return

        cmd = (
            [sys.executable, str(VISUALIZER_SCRIPT)]
            + ["--port", str(self.VISUALIZER_PORT)]
            + self._visualizer_args
        )
        log.info("Launching Vector Visualizer: %s", " ".join(cmd))
        try:
            vis_log = LOG_DIR / "visualizer_output.log"
            self._vis_log_fh = open(vis_log, "a", encoding="utf-8")
            self._visualizer_proc = subprocess.Popen(
                cmd,
                **_DETACH_KWARGS,
                stdout=self._vis_log_fh,
                stderr=subprocess.STDOUT,
                env=self._build_subprocess_env(),
            )
        except Exception as e:
            log.error("Failed to start visualizer: %s", e)
            return

        # Give Dash a moment to bind, then open the browser
        def _open_after_delay():
            import time as _time

            _time.sleep(2.5)
            if self._is_visualizer_running():
                self._open_visualizer_browser()
            self._refresh_icon()

        threading.Thread(target=_open_after_delay, daemon=True).start()

    def _on_stop_visualizer(self, icon, item):
        """Stop the running Visualizer process."""
        self._stop_visualizer()
        self._refresh_icon()

    def _on_restart_visualizer(self, icon, item):
        """Restart the Visualizer (stop then start with fresh code)."""
        threading.Thread(target=self._do_restart_visualizer, daemon=True).start()

    def _do_restart_visualizer(self):
        self._stop_visualizer()
        import time as _time

        _time.sleep(0.5)
        # Re-use the open handler which launches and opens browser
        self._on_open_visualizer(None, None)

    def _is_visualizer_running(self) -> bool:
        """Check if the visualizer subprocess is still alive (proc or port)."""
        with self._subprocess_lock:
            if self._visualizer_proc is not None:
                if self._visualizer_proc.poll() is None:
                    return True
                self._visualizer_proc = None
        # Fallback: scan for anything listening on the visualizer port
        return self._find_pid_by_port(self.VISUALIZER_PORT) is not None

    def _stop_visualizer(self) -> None:
        """Terminate the visualizer process (proc handle or port fallback)."""
        with self._subprocess_lock:
            proc = self._visualizer_proc
            if proc is not None and proc.poll() is not None:
                self._visualizer_proc = None
                proc = None
        if proc is None:
            # No tracked proc — try killing by port (orphan from prior session)
            if self._kill_by_port(self.VISUALIZER_PORT, "visualizer"):
                self._close_vis_log()
            return
        log.info("Stopping visualizer (pid %d)", proc.pid)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("Visualizer did not stop, sending SIGKILL")
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception as e:
                log.error("Failed to kill visualizer: %s", e)
        except Exception as e:
            log.error("Error stopping visualizer: %s", e)
        with self._subprocess_lock:
            self._visualizer_proc = None
        self._close_vis_log()
        log.info("Visualizer stopped")

    def _close_vis_log(self) -> None:
        """Close the visualizer log file handle if open."""
        fh = getattr(self, "_vis_log_fh", None)
        if fh and not fh.closed:
            try:
                fh.close()
            except Exception:
                pass
        self._vis_log_fh = None

    def _open_visualizer_browser(self) -> None:
        """Open the default browser to the Dash app URL."""
        import webbrowser

        url = f"http://127.0.0.1:{self.VISUALIZER_PORT}/"
        log.info("Opening browser to %s", url)
        webbrowser.open(url)

    # ── TensorBoard Projector ───────────────────────────────────

    def _build_tensorboard_cmd(self) -> list[str]:
        """Build the command to launch tensorboard_visualizer.py."""
        cmd = [
            sys.executable,
            str(TENSORBOARD_SCRIPT),
            "--port",
            str(self.TENSORBOARD_PORT),
            "--no-browser",
        ] + self._visualizer_args  # reuse same --vector-backend / --pg-* args
        return cmd

    def _on_open_tensorboard(self, icon, item):
        """Launch the TensorBoard Embedding Projector."""
        if not TENSORBOARD_SCRIPT.exists():
            log.warning("TensorBoard script not found: %s", TENSORBOARD_SCRIPT)
            return
        if self._is_tensorboard_running():
            log.info("TensorBoard already running, opening browser")
            self._open_tensorboard_browser()
            return

        cmd = self._build_tensorboard_cmd()
        log.info("Launching TensorBoard Projector: %s", " ".join(cmd))
        try:
            self._tensorboard_proc = subprocess.Popen(
                cmd,
                **_DETACH_KWARGS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._build_subprocess_env(),
            )
        except Exception as e:
            log.error("Failed to start TensorBoard: %s", e)
            return

        # Give TensorBoard time to export + bind, then open browser
        def _open_after_delay():
            import time as _time

            _time.sleep(5)  # TensorBoard needs longer than Dash (export + startup)
            if self._is_tensorboard_running():
                self._open_tensorboard_browser()
            self._refresh_icon()

        threading.Thread(target=_open_after_delay, daemon=True).start()

    def _on_stop_tensorboard(self, icon, item):
        """Stop the running TensorBoard process."""
        self._stop_tensorboard()
        self._refresh_icon()

    def _on_restart_tensorboard(self, icon, item):
        """Restart TensorBoard (stop then start with fresh data)."""
        threading.Thread(target=self._do_restart_tensorboard, daemon=True).start()

    def _do_restart_tensorboard(self):
        self._stop_tensorboard()
        import time as _time

        _time.sleep(0.5)
        self._on_open_tensorboard(None, None)

    def _is_tensorboard_running(self) -> bool:
        """Check if the TensorBoard subprocess is still alive (proc or port)."""
        with self._subprocess_lock:
            if self._tensorboard_proc is not None:
                if self._tensorboard_proc.poll() is None:
                    return True
                self._tensorboard_proc = None
        return self._find_pid_by_port(self.TENSORBOARD_PORT) is not None

    def _stop_tensorboard(self) -> None:
        """Terminate the TensorBoard process (proc handle or port fallback)."""
        with self._subprocess_lock:
            proc = self._tensorboard_proc
            if proc is not None and proc.poll() is not None:
                self._tensorboard_proc = None
                proc = None
        if proc is None:
            self._kill_by_port(self.TENSORBOARD_PORT, "TensorBoard")
            return
        log.info("Stopping TensorBoard (pid %d)", proc.pid)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("TensorBoard did not stop, sending SIGKILL")
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception as e:
                log.error("Failed to kill TensorBoard: %s", e)
        except Exception as e:
            log.error("Error stopping TensorBoard: %s", e)
        with self._subprocess_lock:
            self._tensorboard_proc = None
        log.info("TensorBoard stopped")

    def _open_tensorboard_browser(self) -> None:
        """Open the default browser to the TensorBoard Projector URL."""
        import webbrowser

        url = f"http://127.0.0.1:{self.TENSORBOARD_PORT}/#projector"
        log.info("Opening browser to %s", url)
        webbrowser.open(url)

    def _on_activity_monitor(self, icon, item):
        """Open the Activity Monitor window."""
        log.info("Opening Activity Monitor")
        threading.Thread(target=self._activity_monitor.show, daemon=True).start()

    def _on_view_log(self, icon, item):
        """Open the server log in the system default viewer."""
        self._open_log_file(LOG_DIR / "server_output.log")

    def _on_view_visualizer_log(self, icon, item):
        """Open the visualizer log in the system default viewer."""
        self._open_log_file(LOG_DIR / "visualizer.log")

    def _open_log_file(self, log_path: Path) -> None:
        """Open a log file in the system default viewer."""
        if not log_path.exists():
            log_path.touch()

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", str(log_path)])
            elif system == "Windows":
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as e:
            log.error("Failed to open log: %s", e)

    def _on_quit(self, icon, item):
        """Stop server, visualizer, TensorBoard, and exit."""
        log.info("Quit requested")
        self._running = False
        self._activity_monitor.stop()
        self._stop_gui()
        self._stop_visualizer()
        self._stop_tensorboard()
        self._server.stop()
        if self._icon:
            self._icon.stop()

    # ── Icon management ─────────────────────────────────────────

    def _update_icon(self, status: str):
        """Update the tray icon image to reflect current status."""
        if self._icon:
            self._icon.icon = _make_icon(status)
            self._icon.menu = self._build_menu()

    def _refresh_icon(self):
        """Re-read the actual server status and update the icon."""
        self._update_icon(self._server.status)

    def _status_poll_loop(self):
        """Background thread: poll server status every 3s and update icon."""
        while self._running:
            time.sleep(3)
            if self._icon:
                self._refresh_icon()

    # ── Run ─────────────────────────────────────────────────────

    def run(self):
        """Start the tray icon (blocks the main thread)."""
        log.info("Starting tray application")

        self._icon = pystray.Icon(
            name="LongTermMemoryMCP",
            icon=_make_icon(STATUS_STOPPED),
            title="Long-Term Memory MCP",
            menu=self._build_menu(),
        )

        # Start background status poller
        self._refresh_thread = threading.Thread(
            target=self._status_poll_loop, daemon=True, name="status-poller"
        )
        self._refresh_thread.start()

        # Auto-start server if requested
        if self._auto_start:
            threading.Thread(target=self._do_start, daemon=True).start()

        # This blocks until icon.stop() is called
        self._icon.run()

        log.info("Tray application exited")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _acquire_single_instance_lock():
    """Ensure only one tray process runs at a time.

    Uses an exclusive flock on a well-known lock file (same directory as the
    PID file for the MCP server).  The lock is held until the process exits —
    no explicit release needed.

    On Windows, a named-mutex approach is used instead (fcntl is unavailable).

    Returns the open file handle (Unix) or None (Windows / already locked).
    Exits with code 0 if another instance is already running.
    """
    lock_path = LOG_DIR / "tray_app.lock"

    if sys.platform == "win32":
        try:
            import ctypes

            mutex = ctypes.windll.kernel32.CreateMutexW(
                None, True, "LongTermMemoryMCPTray"
            )
            if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
                log.info("Another tray instance is already running — exiting")
                sys.exit(0)
            return mutex
        except Exception as e:
            log.warning("Single-instance mutex failed: %s", e)
            return None
    else:
        import fcntl

        try:
            fh = open(lock_path, "w")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.write(str(os.getpid()))
            fh.flush()
            return fh  # caller must keep reference alive for the lock to hold
        except BlockingIOError:
            # Lock is held — check if the owning PID is actually alive.
            # If not (stale lock from a crash), remove the file and retry once.
            try:
                stale_pid = int(lock_path.read_text().strip())
                try:
                    os.kill(stale_pid, 0)  # signal 0 = existence check only
                    # Process is alive — another real instance is running
                    log.info(
                        "Another tray instance is already running (pid %d) — exiting",
                        stale_pid,
                    )
                    sys.exit(0)
                except ProcessLookupError:
                    # PID is dead — stale lock from a crash, remove and retry
                    log.warning(
                        "Stale lock file (pid %d dead) — removing and retrying",
                        stale_pid,
                    )
                    lock_path.unlink(missing_ok=True)
                    try:
                        fh2 = open(lock_path, "w")
                        fcntl.flock(fh2, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fh2.write(str(os.getpid()))
                        fh2.flush()
                        return fh2
                    except Exception as retry_err:
                        log.warning("Lock retry failed: %s", retry_err)
                        return None
                except PermissionError:
                    # Can't check the PID (different user) — treat as live
                    log.info("Another tray instance is already running — exiting")
                    sys.exit(0)
            except Exception:
                log.info("Another tray instance is already running — exiting")
                sys.exit(0)
        except Exception as e:
            log.warning("Single-instance lock failed: %s", e)
            return None


def main():
    parser = argparse.ArgumentParser(
        description="System Tray for Long-Term Memory MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Acquire single-instance lock before doing anything else.
    # The handle must stay in scope for the lifetime of the process.
    _instance_lock = _acquire_single_instance_lock()  # noqa: F841

    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Automatically start the MCP server on launch",
    )

    # These are forwarded to server.py
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="http",
        help="Server transport (default: http for tray usage)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", type=str, default="/mcp/")
    parser.add_argument(
        "--vector-backend",
        choices=["chromadb", "pgvector"],
        default="chromadb",
    )
    parser.add_argument("--pg-host", type=str, default=None)
    parser.add_argument("--pg-port", type=int, default=None)
    parser.add_argument("--pg-database", type=str, default=None)
    parser.add_argument("--pg-user", type=str, default=None)
    parser.add_argument("--pg-password", type=str, default=None)
    parser.add_argument(
        "--network-sharing",
        action="store_true",
        default=False,
        help="Enable LAN memory sharing via mDNS on startup",
    )
    parser.add_argument(
        "--sharing-poll-interval",
        type=int,
        default=300,
        help="Seconds between peer polls when network sharing is enabled (default: 300)",
    )

    args = parser.parse_args()

    # Build the server.py argument list from tray args
    server_args = [
        "--transport",
        args.transport,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--path",
        args.path,
        "--vector-backend",
        args.vector_backend,
    ]
    if args.pg_host is not None:
        server_args += ["--pg-host", args.pg_host]
    if args.pg_port is not None:
        server_args += ["--pg-port", str(args.pg_port)]
    if args.pg_database is not None:
        server_args += ["--pg-database", args.pg_database]
    if args.pg_user is not None:
        server_args += ["--pg-user", args.pg_user]
    if args.network_sharing:
        server_args += ["--network-sharing"]
    if args.sharing_poll_interval != 300:
        server_args += ["--sharing-poll-interval", str(args.sharing_poll_interval)]

    # Pass pg-password via environment variable (not CLI arg) to avoid
    # exposing it in the process list (visible via `ps aux` on Unix /
    # `tasklist /v` or `Get-Process` on Windows).
    server_env = None
    if args.pg_password is not None:
        server_env = {"PGPASSWORD": args.pg_password}

    # Build args to forward to vector_visualizer.py and tensorboard_visualizer.py
    # NOTE: pg-password is NOT passed as CLI arg (visible in `ps aux` on Unix /
    # `tasklist /v` on Windows). Both visualizers read PGPASSWORD from the environment instead.
    visualizer_args = ["--vector-backend", args.vector_backend]
    if args.pg_host is not None:
        visualizer_args += ["--pg-host", args.pg_host]
    if args.pg_port is not None:
        visualizer_args += ["--pg-port", str(args.pg_port)]
    if args.pg_database is not None:
        visualizer_args += ["--pg-database", args.pg_database]
    if args.pg_user is not None:
        visualizer_args += ["--pg-user", args.pg_user]

    # Pass pg-password to visualizer subprocesses via PGPASSWORD env var
    visualizer_env = None
    if args.pg_password is not None:
        visualizer_env = {"PGPASSWORD": args.pg_password}

    # Build pg config for Activity Monitor's vector stats queries
    pg_cfg = {
        "backend_type": args.vector_backend,
        "pg_host": args.pg_host,
        "pg_port": args.pg_port,
        "pg_database": args.pg_database,
        "pg_user": args.pg_user,
        "pg_password": args.pg_password,
    }

    app = TrayApp(
        server_args=server_args,
        auto_start=args.auto_start,
        server_env=server_env,
        visualizer_args=visualizer_args,
        visualizer_env=visualizer_env,
        pg_cfg=pg_cfg,
    )
    app.run()


if __name__ == "__main__":
    main()
