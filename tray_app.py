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
    for font_name in ["Arial", "DejaVuSans", "FreeSans", "LiberationSans"]:
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
        except (ProcessLookupError, PermissionError):
            # Process is gone — stale PID file
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
                    start_new_session=True,
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
                    os.kill(orphan_pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
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
        self._subprocess_lock = (
            threading.Lock()
        )  # protects _visualizer_proc/_tensorboard_proc

    def _build_subprocess_env(self) -> Optional[dict]:
        """Build subprocess environment with PGPASSWORD if configured."""
        if self._visualizer_env:
            return {**os.environ, **self._visualizer_env}
        return None

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
            Item("Open Memory Manager", self._on_open_gui),
            Item("Open Vector Visualizer", self._on_open_visualizer)
            if not viz_running
            else Item("Vector Visualizer", viz_items),
            Item("Open TensorBoard Projector", self._on_open_tensorboard)
            if not tb_running
            else Item("TensorBoard Projector", tb_items),
            Menu.SEPARATOR,
            Item("View Server Log", self._on_view_log),
            Item("View Visualizer Log", self._on_view_visualizer_log),
            Menu.SEPARATOR,
            Item("Quit", self._on_quit),
        )

    # ── Menu actions ────────────────────────────────────────────

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
        """Launch the Memory Manager GUI in a detached process."""
        if not GUI_SCRIPT.exists():
            log.warning("GUI script not found: %s", GUI_SCRIPT)
            return
        log.info("Launching Memory Manager GUI")
        subprocess.Popen(
            [sys.executable, str(GUI_SCRIPT)],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

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
                start_new_session=True,
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
        """Check if the visualizer subprocess is still alive."""
        with self._subprocess_lock:
            if self._visualizer_proc is None:
                return False
            if self._visualizer_proc.poll() is not None:
                self._visualizer_proc = None
                return False
            return True

    def _stop_visualizer(self) -> None:
        """Terminate the visualizer process."""
        with self._subprocess_lock:
            proc = self._visualizer_proc
            if proc is None:
                return
            if proc.poll() is not None:
                self._visualizer_proc = None
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
                start_new_session=True,
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
        """Check if the TensorBoard subprocess is still alive."""
        with self._subprocess_lock:
            if self._tensorboard_proc is None:
                return False
            if self._tensorboard_proc.poll() is not None:
                self._tensorboard_proc = None
                return False
            return True

    def _stop_tensorboard(self) -> None:
        """Terminate the TensorBoard process."""
        with self._subprocess_lock:
            proc = self._tensorboard_proc
            if proc is None:
                return
            if proc.poll() is not None:
                self._tensorboard_proc = None
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


def main():
    parser = argparse.ArgumentParser(
        description="System Tray for Long-Term Memory MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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

    # Pass pg-password via environment variable (not CLI arg) to avoid
    # exposing it in the process list visible via `ps aux`.
    server_env = None
    if args.pg_password is not None:
        server_env = {"PGPASSWORD": args.pg_password}

    # Build args to forward to vector_visualizer.py and tensorboard_visualizer.py
    # NOTE: pg-password is NOT passed as CLI arg (visible in `ps aux`).
    # Both visualizers read PGPASSWORD from the environment instead.
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

    app = TrayApp(
        server_args=server_args,
        auto_start=args.auto_start,
        server_env=server_env,
        visualizer_args=visualizer_args,
        visualizer_env=visualizer_env,
    )
    app.run()


if __name__ == "__main__":
    main()
