#!/usr/bin/env python3
"""
Cross-platform autostart installer for the Long-Term Memory MCP tray app.

Registers tray_app.py to run automatically at user login on:
  - Windows  : HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run (registry)
  - macOS    : ~/Library/LaunchAgents/  (launchd plist)
  - Linux    : ~/.config/systemd/user/  (systemd user service)

Usage:
    python install_autostart.py install   [-- tray_app args...]
    python install_autostart.py uninstall
    python install_autostart.py status

Examples:
    python install_autostart.py install -- --auto-start
    python install_autostart.py install -- --transport http --port 8000 --auto-start
    python install_autostart.py uninstall
    python install_autostart.py status
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "LongTermMemoryMCP"
TRAY_SCRIPT = Path(__file__).parent.resolve() / "tray_app.py"

# macOS launchd plist
MACOS_PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
MACOS_PLIST_NAME = f"com.{APP_NAME.lower()}.tray"
MACOS_PLIST_PATH = MACOS_PLIST_DIR / f"{MACOS_PLIST_NAME}.plist"

# Linux systemd user service
LINUX_SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
LINUX_SERVICE_NAME = f"{APP_NAME.lower()}-tray.service"
LINUX_SERVICE_PATH = LINUX_SYSTEMD_DIR / LINUX_SERVICE_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_exe() -> str:
    """Return the current Python interpreter path (absolute)."""
    return sys.executable


def _build_cmd(extra_args: list[str]) -> list[str]:
    """Build the full command: [python, tray_app.py, ...extra_args]."""
    return [_python_exe(), str(TRAY_SCRIPT)] + extra_args


def _print(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _windows_install(extra_args: list[str]) -> None:
    import winreg  # noqa: PLC0415

    cmd = subprocess.list2cmdline(_build_cmd(extra_args))
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
        _print(f"[OK] Registered '{APP_NAME}' in Windows startup registry.")
        _print(f"     Command: {cmd}")
    except OSError as e:
        _print(f"[ERROR] Failed to write registry key: {e}")
        sys.exit(1)


def _windows_uninstall() -> None:
    import winreg  # noqa: PLC0415

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, APP_NAME)
        _print(f"[OK] Removed '{APP_NAME}' from Windows startup registry.")
    except FileNotFoundError:
        _print(f"[INFO] '{APP_NAME}' was not registered — nothing to remove.")
    except OSError as e:
        _print(f"[ERROR] Failed to remove registry key: {e}")
        sys.exit(1)


def _windows_status() -> None:
    import winreg  # noqa: PLC0415

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ
        ) as key:
            value, _ = winreg.QueryValueEx(key, APP_NAME)
        _print(f"[INSTALLED] '{APP_NAME}' is registered for startup.")
        _print(f"            Command: {value}")
    except FileNotFoundError:
        _print(f"[NOT INSTALLED] '{APP_NAME}' is not registered for startup.")
    except OSError as e:
        _print(f"[ERROR] Could not read registry: {e}")


# ---------------------------------------------------------------------------
# macOS (launchd)
# ---------------------------------------------------------------------------


def _macos_plist(cmd: list[str]) -> str:
    """Generate a launchd plist XML string."""
    program_args = "\n        ".join(f"<string>{arg}</string>" for arg in cmd)
    log_dir = Path.home() / "Library" / "Logs" / APP_NAME
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{MACOS_PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        {program_args}
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <false/>

    <key>StandardOutPath</key>
    <string>{log_dir}/tray_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{log_dir}/tray_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
"""


def _macos_install(extra_args: list[str]) -> None:
    cmd = _build_cmd(extra_args)
    log_dir = Path.home() / "Library" / "Logs" / APP_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    MACOS_PLIST_DIR.mkdir(parents=True, exist_ok=True)

    plist_content = _macos_plist(cmd)
    MACOS_PLIST_PATH.write_text(plist_content, encoding="utf-8")
    _print(f"[OK] Written plist: {MACOS_PLIST_PATH}")

    # Unload first in case it was already loaded, suppress errors
    subprocess.run(
        ["launchctl", "unload", str(MACOS_PLIST_PATH)],
        capture_output=True,
    )
    result = subprocess.run(
        ["launchctl", "load", str(MACOS_PLIST_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _print(f"[WARN] launchctl load returned non-zero: {result.stderr.strip()}")
        _print("       The plist is written; it will take effect on next login.")
    else:
        _print(f"[OK] launchd agent loaded — tray will start at login.")
    _print(f"     Command: {subprocess.list2cmdline(cmd)}")


def _macos_uninstall() -> None:
    if MACOS_PLIST_PATH.exists():
        subprocess.run(
            ["launchctl", "unload", str(MACOS_PLIST_PATH)],
            capture_output=True,
        )
        MACOS_PLIST_PATH.unlink()
        _print(f"[OK] Removed plist and unloaded launchd agent.")
    else:
        _print(f"[INFO] Plist not found — nothing to remove.")


def _macos_status() -> None:
    if not MACOS_PLIST_PATH.exists():
        _print(f"[NOT INSTALLED] Plist not found: {MACOS_PLIST_PATH}")
        return
    result = subprocess.run(
        ["launchctl", "list", MACOS_PLIST_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        _print(f"[INSTALLED + LOADED] {MACOS_PLIST_PATH}")
        _print(result.stdout.strip())
    else:
        _print(f"[INSTALLED, NOT LOADED] Plist exists but agent is not loaded:")
        _print(f"  {MACOS_PLIST_PATH}")


# ---------------------------------------------------------------------------
# Linux (systemd user service)
# ---------------------------------------------------------------------------


def _linux_service(cmd: list[str]) -> str:
    """Generate a systemd user service unit file."""
    exec_start = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd)
    return f"""\
[Unit]
Description=Long-Term Memory MCP Tray App
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/%U/bus

[Install]
WantedBy=default.target
"""


def _linux_install(extra_args: list[str]) -> None:
    cmd = _build_cmd(extra_args)
    LINUX_SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    LINUX_SERVICE_PATH.write_text(_linux_service(cmd), encoding="utf-8")
    _print(f"[OK] Written service: {LINUX_SERVICE_PATH}")

    # Reload systemd user daemon
    reload = subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
        text=True,
    )
    if reload.returncode != 0:
        _print(f"[WARN] daemon-reload failed: {reload.stderr.strip()}")

    enable = subprocess.run(
        ["systemctl", "--user", "enable", "--now", LINUX_SERVICE_NAME],
        capture_output=True,
        text=True,
    )
    if enable.returncode != 0:
        _print(f"[WARN] systemctl enable failed: {enable.stderr.strip()}")
        _print("       The service file is written; enable it manually with:")
        _print(f"       systemctl --user enable --now {LINUX_SERVICE_NAME}")
    else:
        _print(f"[OK] systemd user service enabled — tray will start at login.")
    _print(f"     Command: {subprocess.list2cmdline(cmd)}")


def _linux_uninstall() -> None:
    if not LINUX_SERVICE_PATH.exists():
        _print("[INFO] Service file not found — nothing to remove.")
        return
    subprocess.run(
        ["systemctl", "--user", "disable", "--now", LINUX_SERVICE_NAME],
        capture_output=True,
    )
    LINUX_SERVICE_PATH.unlink()
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
    )
    _print(f"[OK] Disabled and removed service: {LINUX_SERVICE_PATH}")


def _linux_status() -> None:
    if not LINUX_SERVICE_PATH.exists():
        _print(f"[NOT INSTALLED] Service file not found: {LINUX_SERVICE_PATH}")
        return
    result = subprocess.run(
        ["systemctl", "--user", "status", LINUX_SERVICE_NAME],
        capture_output=True,
        text=True,
    )
    _print(result.stdout.strip() or result.stderr.strip())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install/uninstall tray_app.py as a login autostart item.",
        epilog=(
            "Pass extra tray_app.py arguments after '--', e.g.:\n"
            "  python install_autostart.py install -- --auto-start\n"
            "  python install_autostart.py install -- --transport http --port 8000 --auto-start"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall", "status"],
        help="Action to perform",
    )

    # Split on '--' to capture forwarded tray args
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        own_argv = argv[:split]
        extra_args = argv[split + 1 :]
    else:
        own_argv = argv
        extra_args = []

    args = parser.parse_args(own_argv)

    if not TRAY_SCRIPT.exists():
        _print(f"[ERROR] tray_app.py not found at: {TRAY_SCRIPT}")
        sys.exit(1)

    platform = sys.platform

    if args.action == "install":
        _print(f"Installing autostart for '{APP_NAME}' on {platform}...")
        if platform == "win32":
            _windows_install(extra_args)
        elif platform == "darwin":
            _macos_install(extra_args)
        else:
            _linux_install(extra_args)

    elif args.action == "uninstall":
        _print(f"Removing autostart for '{APP_NAME}' on {platform}...")
        if platform == "win32":
            _windows_uninstall()
        elif platform == "darwin":
            _macos_uninstall()
        else:
            _linux_uninstall()

    elif args.action == "status":
        if platform == "win32":
            _windows_status()
        elif platform == "darwin":
            _macos_status()
        else:
            _linux_status()


if __name__ == "__main__":
    main()
