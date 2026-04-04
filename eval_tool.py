#!/usr/bin/env python3
"""
eval_tool.py — MCP Tool Evaluator

Provides three interfaces to every long-term-memory MCP tool:

  1. **TUI** — full-screen curses terminal UI
       python eval_tool.py tui

  2. **Programmatic** — import and call from scripts / pytest
       from eval_tool import ToolRunner
       runner = ToolRunner()
       result = runner.run("search_memories", query="camping", limit=5)

  3. **CLI one-shot** — run a single tool call from the command line
       python eval_tool.py run search_memories --query "camping" --limit 5

  4. **Replay** — re-run a recorded audit call
       python eval_tool.py replay --file data/audit/audit_2026-04-01.jsonl --index 3
       python eval_tool.py list-calls --file data/audit/audit_2026-04-01.jsonl

Audit
-----
Every tool call (via any interface) is written as a single JSON line to
  data/audit/audit_YYYY-MM-DD.jsonl

Each line contains:
  {
    "call_id":   "<uuid4>",
    "timestamp": "<ISO-8601>",
    "tool":      "<tool_name>",
    "args":      { ... raw string args as passed ... },
    "success":   true/false,
    "duration_ms": 42,
    "result":    { ... full result dict ... },
    "error":     null | "<traceback string>"
  }

Audit files can be loaded and replayed with the `replay` / `list-calls` commands.
"""

from __future__ import annotations

import argparse
import curses
import json
import os
import sys
import textwrap
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap: ensure memory_mcp is importable when running from repo root
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from memory_mcp import RobustMemorySystem
from memory_mcp.mcp_tools import jsonify_result
from memory_mcp.audit import AuditLogger


# ---------------------------------------------------------------------------
# Tool registry — mirrors mcp_tools.py parameter signatures
# ---------------------------------------------------------------------------

# Each entry:
#   name          — tool name (matches MCP tool name)
#   description   — one-line description for the TUI menu
#   params        — ordered list of (name, type_str, default, description)
#                   type_str: "str" | "int" | "bool" | "str?"  (? = optional)

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "remember",
        "description": "Store a new memory",
        "params": [
            ("title", "str", _REQUIRED := object(), "Short title"),
            ("content", "str", _REQUIRED, "Full text to store"),
            ("tags", "str", "", "Comma-separated tags"),
            ("importance", "int", 5, "1-10 (default 5)"),
            (
                "memory_type",
                "str",
                "conversation",
                "conversation/fact/preference/event",
            ),
            ("shared_with", "str", "", "Comma-separated UUIDs or *"),
            ("file_paths", "str", "", "Comma-separated absolute paths"),
        ],
    },
    {
        "name": "search_memories",
        "description": "Semantic or structured search",
        "params": [
            ("query", "str", _REQUIRED, "Natural language query"),
            ("search_type", "str", "semantic", "semantic or structured"),
            ("limit", "int", 10, "Max results"),
        ],
    },
    {
        "name": "search_by_type",
        "description": "Retrieve memories by type",
        "params": [
            ("memory_type", "str", _REQUIRED, "conversation/fact/preference/event"),
            ("limit", "int", 20, "Max results"),
        ],
    },
    {
        "name": "search_by_tags",
        "description": "Find memories by tags",
        "params": [
            ("tags", "str", _REQUIRED, "Comma-separated tags"),
            ("limit", "int", 20, "Max results"),
        ],
    },
    {
        "name": "get_recent_memories",
        "description": "Most recently stored memories",
        "params": [
            ("limit", "int", 20, "Max results"),
            ("current_project", "str?", None, "Project tag filter (or empty)"),
        ],
    },
    {
        "name": "update_memory",
        "description": "Update an existing memory by ID",
        "params": [
            ("memory_id", "str", _REQUIRED, "Memory ID to update"),
            ("title", "str?", None, "New title (blank = no change)"),
            ("content", "str?", None, "New content (blank = no change)"),
            ("tags", "str?", None, "New comma-separated tags"),
            ("importance", "int?", None, "New importance 1-10"),
            ("memory_type", "str?", None, "New type"),
            ("shared_with", "str?", None, "New sharing (blank = private)"),
        ],
    },
    {
        "name": "delete_memory",
        "description": "Permanently delete a memory",
        "params": [
            ("memory_id", "str", _REQUIRED, "Memory ID to delete"),
        ],
    },
    {
        "name": "get_memory_stats",
        "description": "Memory system statistics",
        "params": [],
    },
    {
        "name": "create_backup",
        "description": "Create a full backup now",
        "params": [],
    },
    {
        "name": "search_by_date_range",
        "description": "Find memories within a date range",
        "params": [
            ("date_from", "str", _REQUIRED, "ISO date, e.g. 2025-09-01"),
            ("date_to", "str?", None, "ISO date (default: now)"),
            ("limit", "int", 50, "Max results"),
        ],
    },
    {
        "name": "rebuild_vectors",
        "description": "Rebuild vector index from SQLite",
        "params": [],
    },
    {
        "name": "list_source_memories",
        "description": "Preview memories in another DB file",
        "params": [
            ("source_db_path", "str", _REQUIRED, "Path to source .db file"),
            ("limit", "int", 100, "Max results"),
        ],
    },
    {
        "name": "migrate_memories",
        "description": "Migrate memories from another DB",
        "params": [
            ("source_db_path", "str", _REQUIRED, "Path to source .db file"),
            ("source_chroma_path", "str?", None, "Path to source chroma dir"),
            ("memory_ids", "str?", None, "Comma-separated IDs to migrate"),
            ("skip_duplicates", "bool", True, "Skip duplicate content"),
        ],
    },
]

# Sentinel used above — re-exposed as module constant so it is not
# the same object after module re-import during testing.
REQUIRED = _REQUIRED


# ---------------------------------------------------------------------------
# ToolRunner — programmatic API
# ---------------------------------------------------------------------------


class ToolRunner:
    """
    Direct programmatic interface to every MCP tool.

    Usage::

        runner = ToolRunner()

        # Positional-style
        result = runner.run("search_memories", query="camping trip", limit=5)

        # Via a pre-built args dict (e.g. from an audit record)
        result = runner.run_from_audit(audit_record)

        runner.close()

    All calls are audited automatically.  Pass ``audit=False`` to ``run()``
    to suppress writing an audit record (useful for replay comparisons).
    """

    def __init__(
        self,
        memory_system: Optional[RobustMemorySystem] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self._own_ms = memory_system is None
        self.ms = memory_system or RobustMemorySystem()
        self.logger = audit_logger or AuditLogger()

    def close(self) -> None:
        if self._own_ms:
            self.ms.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── core dispatcher ─────────────────────────────────────────

    def run(self, tool_name: str, audit: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Call *tool_name* with the given keyword arguments.

        Returns the jsonified result dict (same shape as the MCP tool response).
        Raises ``ValueError`` for unknown tool names.
        Raises ``TypeError`` for missing required parameters.
        """
        if tool_name not in {t["name"] for t in TOOLS}:
            raise ValueError(f"Unknown tool: {tool_name!r}")

        # Validate required params
        tool_def = next(t for t in TOOLS if t["name"] == tool_name)
        for pname, ptype, pdefault, _ in tool_def["params"]:
            if pdefault is REQUIRED and pname not in kwargs:
                raise TypeError(f"Tool {tool_name!r} requires parameter {pname!r}")

        call_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        error_str = None
        result: Dict[str, Any] = {}

        try:
            result = self._dispatch(tool_name, kwargs)
        except Exception:
            error_str = traceback.format_exc()
            result = {"success": False, "reason": error_str.splitlines()[-1]}

        duration_ms = round((time.perf_counter() - t0) * 1000)

        record: Dict[str, Any] = {
            "call_id": call_id,
            "timestamp": ts,
            "tool": tool_name,
            "args": kwargs,
            "success": result.get("success", False)
            if isinstance(result, dict)
            else bool(result),
            "duration_ms": duration_ms,
            "result": result,
            "error": error_str,
        }

        if audit:
            try:
                self.logger.write(record)
            except Exception as _write_exc:  # noqa: BLE001
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "Audit write failed (non-fatal): %s", _write_exc
                )

        return result

    def run_from_audit(
        self, record: Dict[str, Any], audit: bool = True
    ) -> Dict[str, Any]:
        """
        Re-run a tool call from a previously recorded audit record.

        Returns a new result dict.  The replay is itself audited as a new record
        unless *audit=False*.
        """
        tool_name = record["tool"]
        args = dict(record.get("args") or {})
        return self.run(tool_name, audit=audit, **args)

    # ── private dispatch ─────────────────────────────────────────

    def _dispatch(self, tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Call the underlying memory_system method, mirroring mcp_tools.py logic."""
        ms = self.ms

        def _str(k, default=""):
            return str(kwargs.get(k, default)) if kwargs.get(k) is not None else default

        def _int(k, default=None):
            v = kwargs.get(k, default)
            return int(v) if v is not None else default

        def _bool(k, default=True):
            v = kwargs.get(k, default)
            if isinstance(v, str):
                return v.lower() not in ("false", "0", "no", "")
            return bool(v)

        def _split(k, default=""):
            raw = kwargs.get(k) or default
            if not raw:
                return []
            return [x.strip() for x in str(raw).split(",") if x.strip()]

        if tool_name == "remember":
            res = ms.remember(
                title=_str("title"),
                content=_str("content"),
                tags=_split("tags"),
                importance=_int("importance", 5),
                memory_type=_str("memory_type", "conversation"),
                shared_with=_split("shared_with"),
                file_paths=_split("file_paths"),
            )

        elif tool_name == "search_memories":
            search_type = _str("search_type", "semantic")
            limit = _int("limit", 10)
            if search_type == "semantic":
                res = ms.search_semantic(_str("query"), limit)
            else:
                res = ms.search_structured(limit=limit)

        elif tool_name == "search_by_type":
            res = ms.search_structured(
                memory_type=_str("memory_type"),
                limit=_int("limit", 20),
            )

        elif tool_name == "search_by_tags":
            res = ms.search_structured(
                tags=_split("tags"),
                limit=_int("limit", 20),
            )

        elif tool_name == "get_recent_memories":
            current_project = kwargs.get("current_project") or None
            if isinstance(current_project, str) and not current_project.strip():
                current_project = None
            res = ms.get_recent(_int("limit", 20), current_project=current_project)

        elif tool_name == "update_memory":
            tags_raw = kwargs.get("tags")
            tag_list = (
                [t.strip() for t in str(tags_raw).split(",") if t.strip()]
                if tags_raw is not None
                else None
            )
            sw_raw = kwargs.get("shared_with")
            sw_list = (
                [s.strip() for s in str(sw_raw).split(",") if s.strip()]
                if sw_raw is not None
                else None
            )
            res = ms.update_memory(
                memory_id=_str("memory_id"),
                title=kwargs.get("title"),
                content=kwargs.get("content"),
                tags=tag_list,
                importance=_int("importance"),
                memory_type=kwargs.get("memory_type"),
                shared_with=sw_list,
            )

        elif tool_name == "delete_memory":
            res = ms.delete_memory(_str("memory_id"))

        elif tool_name == "get_memory_stats":
            res = ms.get_statistics()

        elif tool_name == "create_backup":
            res = ms.create_backup()

        elif tool_name == "search_by_date_range":
            date_to = kwargs.get("date_to") or datetime.now(timezone.utc).isoformat()
            res = ms.search_structured(
                date_from=_str("date_from"),
                date_to=str(date_to),
                limit=_int("limit", 50),
            )

        elif tool_name == "rebuild_vectors":
            res = ms.rebuild_vector_index()

        elif tool_name == "list_source_memories":
            res = ms.list_source_memories(_str("source_db_path"), _int("limit", 100))

        elif tool_name == "migrate_memories":
            memory_id_list = _split("memory_ids") or None
            res = ms.migrate_memories(
                source_db_path=_str("source_db_path"),
                source_chroma_path=kwargs.get("source_chroma_path"),
                memory_ids=memory_id_list,
                skip_duplicates=_bool("skip_duplicates", True),
            )

        else:
            raise ValueError(f"No dispatch handler for {tool_name!r}")

        return jsonify_result(res)


# ---------------------------------------------------------------------------
# Helpers shared by TUI and CLI
# ---------------------------------------------------------------------------


def _coerce_arg(value: str, type_str: str) -> Any:
    """Convert a raw string *value* to the type indicated by *type_str*."""
    base = type_str.rstrip("?")
    if not value.strip() and "?" in type_str:
        return None
    if base == "int":
        return int(value)
    if base == "bool":
        return value.lower() not in ("false", "0", "no", "")
    return value  # str


def _format_result(result: Dict[str, Any], width: int = 80) -> str:
    """Pretty-print a result dict as human-readable text."""
    lines = []
    success = result.get("success", False)
    lines.append(f"Success: {'YES' if success else 'NO'}")
    if "reason" in result:
        lines.append(f"Reason:  {result['reason']}")
    data = result.get("data")
    if data is None:
        pass
    elif isinstance(data, list):
        lines.append(f"Records: {len(data)}")
        for idx, item in enumerate(data):
            lines.append(f"\n  [{idx + 1}] " + "─" * (width - 6))
            if isinstance(item, dict):
                for k, v in item.items():
                    val_str = str(v)
                    if len(val_str) > 200:
                        val_str = val_str[:197] + "..."
                    for i, chunk in enumerate(
                        textwrap.wrap(val_str, width - 14) or [""]
                    ):
                        prefix = f"      {k:<14}" if i == 0 else " " * 20
                        lines.append(f"{prefix}{chunk}")
            else:
                lines.append(f"      {item}")
    elif isinstance(data, dict):
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TUI — curses full-screen terminal interface
# ---------------------------------------------------------------------------

_COLOUR_TITLE = 1
_COLOUR_SEL = 2
_COLOUR_STATUS = 3
_COLOUR_ERR = 4
_COLOUR_OK = 5
_COLOUR_DIM = 6


class _TuiApp:
    """
    Curses TUI for the eval tool.

    Layout
    ------
    ┌─ TITLE BAR ─────────────────────────────────────────────────┐
    │ Tool list (left pane)  │  Arg form / Result (right pane)    │
    │                        │                                     │
    │                        │                                     │
    └─ STATUS BAR ────────────────────────────────────────────────┘

    Key bindings
    ↑/↓       navigate tool list / form fields
    Enter     select tool / submit form / confirm
    Tab       switch between tool list and result pane
    Esc / q   go back / quit
    r         open replay menu
    a         list today's audit file
    """

    def __init__(self, runner: ToolRunner) -> None:
        self.runner = runner
        self.tools = TOOLS
        self.tool_idx = 0  # currently highlighted tool
        self.state = "menu"  # "menu" | "form" | "result" | "audit"
        self.selected_tool: Optional[Dict] = None
        self.form_fields: List[Dict] = []  # {name, type, default, desc, value, cursor}
        self.form_idx = 0
        self.result_text = ""
        self.audit_records: List[Dict] = []
        self.audit_idx = 0
        self.status_msg = ""
        self.status_ok = True

    # ── main entry ───────────────────────────────────────────────

    def run(self, stdscr) -> None:
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(_COLOUR_TITLE, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(_COLOUR_SEL, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(_COLOUR_STATUS, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(_COLOUR_ERR, curses.COLOR_WHITE, curses.COLOR_RED)
        curses.init_pair(_COLOUR_OK, curses.COLOR_GREEN, -1)
        curses.init_pair(_COLOUR_DIM, curses.COLOR_WHITE, -1)

        self.stdscr = stdscr
        self.stdscr.keypad(True)

        while True:
            self._draw()
            key = self.stdscr.getch()
            done = self._handle(key)
            if done:
                break

    # ── drawing helpers ──────────────────────────────────────────

    def _safe_addstr(
        self,
        y: int,
        x: int,
        text: str,
        attr: int = 0,
        *,
        fill_to: int = 0,
    ) -> None:
        """Write *text* at (y, x), clipping to terminal width and swallowing
        curses.error.

        ``fill_to``: if > 0, left-justify *text* to fill up to column
        ``fill_to - 1`` (never reaching the last column, which avoids the
        bottom-right corner ERR on most terminals).
        """
        scr = self.stdscr
        h, w = scr.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        # Maximum characters we can safely write from column x
        max_chars = w - x - 1  # leave the very last cell untouched
        if max_chars <= 0:
            return
        if fill_to > 0:
            text = text.ljust(min(fill_to, max_chars + x) - x)
        text = text[:max_chars]
        if not text:
            return
        try:
            if attr:
                scr.addstr(y, x, text, attr)
            else:
                scr.addstr(y, x, text)
        except curses.error:
            pass

    def _fill_row(self, y: int, text: str, attr: int = 0) -> None:
        """Write *text* left-justified across the full row, leaving the very
        last cell blank to avoid the bottom-right ERR."""
        scr = self.stdscr
        _, w = scr.getmaxyx()
        self._safe_addstr(y, 0, text, attr, fill_to=w)

    # ── drawing ──────────────────────────────────────────────────

    def _draw(self) -> None:
        scr = self.stdscr
        h, w = scr.getmaxyx()
        scr.erase()

        # Title bar — fill entire row but stop one cell short of the corner
        title = " MCP Eval Tool  |  ↑↓ navigate  Enter select  Esc/q back  a audit  r replay "
        self._fill_row(0, title, curses.color_pair(_COLOUR_TITLE))

        # Status bar — same treatment for last row
        status = self.status_msg or "Ready"
        pair = _COLOUR_ERR if not self.status_ok else _COLOUR_STATUS
        self._fill_row(h - 1, " " + status, pair)

        body_top = 1
        body_bot = h - 1

        if self.state == "menu":
            self._draw_menu(body_top, body_bot, w)
        elif self.state == "form":
            self._draw_form(body_top, body_bot, w)
        elif self.state == "result":
            self._draw_result(body_top, body_bot, w)
        elif self.state == "audit":
            self._draw_audit(body_top, body_bot, w)

        scr.refresh()

    def _draw_menu(self, top: int, bot: int, w: int) -> None:
        height = bot - top
        visible = min(len(self.tools), height - 2)
        start = max(0, self.tool_idx - visible + 1)

        self._safe_addstr(top, 2, "Tools", curses.A_BOLD)
        for row, i in enumerate(range(start, start + visible), top + 1):
            if i >= len(self.tools):
                break
            tool = self.tools[i]
            label = f"  {tool['name']:<28}  {tool['description']}"
            if i == self.tool_idx:
                self._fill_row(row, label, curses.color_pair(_COLOUR_SEL))
            else:
                self._safe_addstr(row, 0, label)

        hint = "  Enter to open tool form"
        self._safe_addstr(
            min(top + visible + 2, bot - 2), 2, hint, curses.color_pair(_COLOUR_DIM)
        )

    def _draw_form(self, top: int, bot: int, w: int) -> None:
        tool = self.selected_tool
        self._safe_addstr(top, 2, f"Tool: {tool['name']}", curses.A_BOLD)
        self._safe_addstr(top + 1, 2, "─" * max(0, w - 4))

        row = top + 2
        for fi, field in enumerate(self.form_fields):
            if row >= bot - 3:
                break
            req = "" if field["default"] is not REQUIRED else " *"
            label = f"  {field['name']}{req} ({field['type']}):"
            label = label[: w // 2]
            self._safe_addstr(
                row, 0, label, curses.A_BOLD if fi == self.form_idx else 0
            )

            # Value input box
            val = field["value"]
            box_x = max(len(label) + 2, w // 2)
            box_w = max(1, w - box_x - 2)
            display_val = (val or "")[:box_w]
            if fi == self.form_idx:
                self._safe_addstr(
                    row,
                    box_x,
                    display_val,
                    curses.color_pair(_COLOUR_SEL),
                    fill_to=box_x + box_w,
                )
                # Show cursor
                cursor_pos = min(len(display_val), box_w - 1)
                try:
                    self.stdscr.move(row, box_x + cursor_pos)
                    curses.curs_set(1)
                except curses.error:
                    pass
            else:
                curses.curs_set(0)
                self._safe_addstr(row, box_x, display_val)

            # Description hint on next line
            if fi == self.form_idx:
                self._safe_addstr(
                    row + 1,
                    0,
                    f"    {field['desc']}",
                    curses.color_pair(_COLOUR_DIM),
                )

            row += 2

        self._safe_addstr(
            min(row + 1, bot - 2),
            2,
            "↑↓ navigate fields  Enter run  Esc cancel",
            curses.color_pair(_COLOUR_DIM),
        )

    def _draw_result(self, top: int, bot: int, w: int) -> None:
        lines = self.result_text.splitlines()
        max_visible = bot - top - 2

        self._safe_addstr(top, 2, "Result", curses.A_BOLD)
        self._safe_addstr(top + 1, 0, "─" * max(0, w - 1))

        for i, line in enumerate(lines[:max_visible]):
            self._safe_addstr(top + 2 + i, 0, line)

        self._safe_addstr(
            bot - 1, 2, "Enter / Esc → back to menu", curses.color_pair(_COLOUR_DIM)
        )

    def _draw_audit(self, top: int, bot: int, w: int) -> None:
        self._safe_addstr(top, 2, "Audit Records (today)", curses.A_BOLD)
        self._safe_addstr(top + 1, 0, "─" * max(0, w - 1))

        if not self.audit_records:
            self._safe_addstr(top + 2, 2, "No audit records found for today.")
            self._safe_addstr(top + 4, 2, "Esc → back", curses.color_pair(_COLOUR_DIM))
            return

        visible = max(1, bot - top - 5)
        start = max(0, self.audit_idx - visible + 1)

        for row_off, i in enumerate(range(start, start + visible)):
            if i >= len(self.audit_records):
                break
            rec = self.audit_records[i]
            ts = rec.get("timestamp", "")[:19].replace("T", " ")
            ok = "✓" if rec.get("success") else "✗"
            dur = rec.get("duration_ms", "?")
            label = f"  {ok} {ts}  {rec['tool']:<28}  {dur}ms"
            screen_row = top + 2 + row_off
            if i == self.audit_idx:
                self._fill_row(screen_row, label, curses.color_pair(_COLOUR_SEL))
            else:
                pair = _COLOUR_OK if rec.get("success") else _COLOUR_ERR
                self._safe_addstr(screen_row, 0, label, curses.color_pair(pair))

        self._safe_addstr(
            bot - 2,
            2,
            "↑↓ navigate  Enter replay selected  Esc back",
            curses.color_pair(_COLOUR_DIM),
        )

    # ── input handling ───────────────────────────────────────────

    def _handle(self, key: int) -> bool:
        """Return True to quit."""
        if self.state == "menu":
            return self._handle_menu(key)
        elif self.state == "form":
            self._handle_form(key)
        elif self.state == "result":
            if key in (ord("\n"), curses.KEY_ENTER, 27, ord("q")):
                self.state = "menu"
        elif self.state == "audit":
            self._handle_audit(key)
        return False

    def _handle_menu(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return True
        if key == curses.KEY_UP:
            self.tool_idx = max(0, self.tool_idx - 1)
        elif key == curses.KEY_DOWN:
            self.tool_idx = min(len(self.tools) - 1, self.tool_idx + 1)
        elif key in (ord("\n"), curses.KEY_ENTER):
            self._open_form(self.tools[self.tool_idx])
        elif key in (ord("a"),):
            self._load_audit()
            self.state = "audit"
        elif key in (ord("r"),):
            self._load_audit()
            self.state = "audit"
        return False

    def _open_form(self, tool: Dict) -> None:
        self.selected_tool = tool
        self.form_fields = []
        for pname, ptype, pdefault, pdesc in tool["params"]:
            default_str = (
                ""
                if pdefault is REQUIRED
                else ("" if pdefault is None else str(pdefault))
            )
            self.form_fields.append(
                {
                    "name": pname,
                    "type": ptype,
                    "default": pdefault,
                    "desc": pdesc,
                    "value": default_str,
                }
            )
        self.form_idx = 0
        self.state = "form"
        self.status_msg = f"Editing {tool['name']} — fill in fields, Enter to run"
        self.status_ok = True

    def _handle_form(self, key: int) -> None:
        if key == 27:  # Esc
            self.state = "menu"
            return

        if not self.form_fields:
            # No params — run immediately
            if key in (ord("\n"), curses.KEY_ENTER):
                self._run_form()
            return

        if key == curses.KEY_UP:
            self.form_idx = max(0, self.form_idx - 1)
        elif key == curses.KEY_DOWN:
            self.form_idx = min(len(self.form_fields) - 1, self.form_idx + 1)
        elif key in (ord("\n"), curses.KEY_ENTER):
            if self.form_idx == len(self.form_fields) - 1:
                self._run_form()
            else:
                self.form_idx += 1
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            field = self.form_fields[self.form_idx]
            field["value"] = field["value"][:-1]
        elif 32 <= key < 256:
            field = self.form_fields[self.form_idx]
            field["value"] += chr(key)

    def _run_form(self) -> None:
        tool = self.selected_tool
        kwargs: Dict[str, Any] = {}
        try:
            for field in self.form_fields:
                raw = field["value"].strip()
                if raw == "" and field["default"] is REQUIRED:
                    self.status_msg = f"Field '{field['name']}' is required"
                    self.status_ok = False
                    return
                if raw == "":
                    # Use default
                    d = field["default"]
                    if d is not REQUIRED and d is not None:
                        kwargs[field["name"]] = d
                    # else omit — will use Python default
                else:
                    kwargs[field["name"]] = _coerce_arg(raw, field["type"])
        except (ValueError, TypeError) as exc:
            self.status_msg = f"Arg error: {exc}"
            self.status_ok = False
            return

        self.status_msg = f"Running {tool['name']}..."
        self.status_ok = True
        self._draw()

        result = self.runner.run(tool["name"], **kwargs)
        self.result_text = _format_result(result)
        self.state = "result"
        self.status_ok = result.get("success", False)
        self.status_msg = (
            "Success" if self.status_ok else result.get("reason", "Failed")
        )

    def _load_audit(self) -> None:
        logger = self.runner.logger
        path = logger._today_path()
        if path.exists():
            try:
                self.audit_records = AuditLogger.load(path)
                self.status_msg = f"Loaded {len(self.audit_records)} audit records"
                self.status_ok = True
            except Exception as exc:
                self.audit_records = []
                self.status_msg = f"Error loading audit: {exc}"
                self.status_ok = False
        else:
            self.audit_records = []
            self.status_msg = "No audit file for today yet"
            self.status_ok = True
        self.audit_idx = max(0, len(self.audit_records) - 1)

    def _handle_audit(self, key: int) -> None:
        if key in (27, ord("q")):
            self.state = "menu"
            return
        if key == curses.KEY_UP:
            self.audit_idx = max(0, self.audit_idx - 1)
        elif key == curses.KEY_DOWN:
            self.audit_idx = min(len(self.audit_records) - 1, self.audit_idx + 1)
        elif key in (ord("\n"), curses.KEY_ENTER):
            if self.audit_records:
                rec = self.audit_records[self.audit_idx]
                self.status_msg = f"Replaying {rec['tool']}..."
                self.status_ok = True
                self._draw()
                result = self.runner.run_from_audit(rec)
                self.result_text = (
                    f"REPLAY of call_id={rec['call_id']}\n"
                    f"Original: {rec['timestamp']}\n"
                    + "─" * 60
                    + "\n"
                    + _format_result(result)
                )
                self.state = "result"
                self.status_ok = result.get("success", False)
                self.status_msg = (
                    "Replay: Success"
                    if self.status_ok
                    else "Replay: " + result.get("reason", "Failed")
                )


def run_tui(runner: ToolRunner) -> None:
    """Launch the curses TUI."""
    curses.wrapper(lambda scr: _TuiApp(runner).run(scr))


# ---------------------------------------------------------------------------
# CLI — `run` subcommand
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace, runner: ToolRunner) -> int:
    """Execute a single tool call from CLI arguments."""
    tool_name = args.tool
    if tool_name not in {t["name"] for t in TOOLS}:
        print(f"Error: unknown tool '{tool_name}'", file=sys.stderr)
        print(f"Available: {', '.join(t['name'] for t in TOOLS)}", file=sys.stderr)
        return 1

    # Parse --key value pairs from remaining args (already parsed by argparse)
    tool_def = next(t for t in TOOLS if t["name"] == tool_name)
    kwargs: Dict[str, Any] = {}
    for pname, ptype, pdefault, _ in tool_def["params"]:
        val = getattr(args, pname, None)
        if val is not None:
            try:
                kwargs[pname] = _coerce_arg(str(val), ptype)
            except (ValueError, TypeError) as exc:
                print(f"Error: bad value for --{pname}: {exc}", file=sys.stderr)
                return 1

    try:
        result = runner.run(tool_name, **kwargs)
    except TypeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_format_result(result))

    return 0 if result.get("success") else 1


# ---------------------------------------------------------------------------
# CLI — `replay` subcommand
# ---------------------------------------------------------------------------


def cmd_replay(args: argparse.Namespace, runner: ToolRunner) -> int:
    """Replay a single audit record."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    records = AuditLogger.load(path)
    idx = args.index
    if idx < 0 or idx >= len(records):
        print(
            f"Error: index {idx} out of range (file has {len(records)} records)",
            file=sys.stderr,
        )
        return 1

    rec = records[idx]
    print(f"Replaying: {rec['tool']}  (original: {rec['timestamp']})")
    print(f"Args: {json.dumps(rec.get('args', {}), indent=2)}")
    print("─" * 60)

    result = runner.run_from_audit(rec, audit=not args.no_audit)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_format_result(result))

    return 0 if result.get("success") else 1


# ---------------------------------------------------------------------------
# CLI — `list-calls` subcommand
# ---------------------------------------------------------------------------


def cmd_list_calls(args: argparse.Namespace) -> int:
    """List all calls in an audit file."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    records = AuditLogger.load(path)
    if not records:
        print("(empty file)")
        return 0

    print(f"{'#':<5} {'OK':<3} {'Timestamp':<22} {'Tool':<30} {'ms':<8} {'call_id'}")
    print("─" * 100)
    for idx, rec in enumerate(records):
        ts = rec.get("timestamp", "")[:19].replace("T", " ")
        ok = "✓" if rec.get("success") else "✗"
        dur = rec.get("duration_ms", "?")
        cid = rec.get("call_id", "")[:16]
        print(f"{idx:<5} {ok:<3} {ts:<22} {rec['tool']:<30} {dur!s:<8} {cid}")

    print(f"\nTotal: {len(records)} calls")
    return 0


# ---------------------------------------------------------------------------
# argparse setup
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MCP Tool Evaluator — TUI, CLI, and audit replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python eval_tool.py tui
          python eval_tool.py run search_memories --query "camping" --limit 5
          python eval_tool.py run remember --title "Test" --content "Hello world" --tags "test"
          python eval_tool.py list-calls --file data/audit/audit_2026-04-01.jsonl
          python eval_tool.py replay --file data/audit/audit_2026-04-01.jsonl --index 3
        """),
    )

    # Global backend options
    parser.add_argument(
        "--vector-backend",
        choices=["chromadb", "pgvector"],
        default="chromadb",
        help="Vector backend (default: chromadb)",
    )
    parser.add_argument("--pg-host", default=None)
    parser.add_argument("--pg-port", type=int, default=None)
    parser.add_argument("--pg-database", default=None)
    parser.add_argument("--pg-user", default=None)
    parser.add_argument("--pg-password", default=None)
    parser.add_argument(
        "--audit-dir",
        default=None,
        help="Directory for audit JSONL files (default: data/audit/)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── tui ──────────────────────────────────────────────────────
    subparsers.add_parser("tui", help="Launch full-screen terminal UI")

    # ── run ──────────────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Run a single tool call")
    run_p.add_argument("tool", metavar="TOOL", help="Tool name")
    run_p.add_argument("--json", action="store_true", help="Output raw JSON")

    # Add one optional arg per tool param (all tools, all params)
    seen: set = set()
    for tool in TOOLS:
        for pname, ptype, _, pdesc in tool["params"]:
            if pname in seen:
                continue
            seen.add(pname)
            run_p.add_argument(
                f"--{pname.replace('_', '-')}",
                dest=pname,
                default=None,
                help=pdesc,
                metavar=ptype.rstrip("?").upper(),
            )

    # ── list-calls ───────────────────────────────────────────────
    lc_p = subparsers.add_parser("list-calls", help="List calls in an audit file")
    lc_p.add_argument(
        "--file", required=True, metavar="PATH", help="Path to .jsonl audit file"
    )

    # ── replay ───────────────────────────────────────────────────
    rep_p = subparsers.add_parser("replay", help="Replay an audit record")
    rep_p.add_argument(
        "--file", required=True, metavar="PATH", help="Path to .jsonl audit file"
    )
    rep_p.add_argument(
        "--index",
        type=int,
        default=0,
        metavar="N",
        help="0-based index of the record to replay (default: 0)",
    )
    rep_p.add_argument("--json", action="store_true", help="Output raw JSON")
    rep_p.add_argument(
        "--no-audit",
        action="store_true",
        help="Do not write a new audit record for this replay",
    )

    return parser


# ---------------------------------------------------------------------------
# Build MemorySystem from args
# ---------------------------------------------------------------------------


def _build_memory_system(args: argparse.Namespace) -> RobustMemorySystem:
    """Instantiate RobustMemorySystem according to backend CLI args."""
    vector_backend = None
    database_backend = None

    if getattr(args, "vector_backend", "chromadb") == "pgvector":
        from memory_mcp.vector_backends.pgvector_backend import PgvectorBackend
        from memory_mcp.database_backends.postgres import PostgresDatabase
        from memory_mcp.config import EMBEDDING_MODEL_CONFIG

        pg_kw = dict(
            host=args.pg_host,
            port=args.pg_port,
            database=args.pg_database,
            user=args.pg_user,
            password=args.pg_password or os.environ.get("PGPASSWORD"),
        )
        vector_backend = PgvectorBackend(
            **pg_kw,
            dimensions=EMBEDDING_MODEL_CONFIG["dimensions"],
        )
        database_backend = PostgresDatabase(**pg_kw)

    return RobustMemorySystem(
        vector_backend=vector_backend,
        database_backend=database_backend,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    audit_dir = Path(args.audit_dir) if args.audit_dir else None

    # list-calls doesn't need a MemorySystem
    if args.command == "list-calls":
        return cmd_list_calls(args)

    ms = _build_memory_system(args)
    logger = AuditLogger(audit_dir)
    runner = ToolRunner(memory_system=ms, audit_logger=logger)

    try:
        if args.command == "tui":
            run_tui(runner)
            return 0
        elif args.command == "run":
            return cmd_run(args, runner)
        elif args.command == "replay":
            return cmd_replay(args, runner)
        else:
            parser.print_help()
            return 0
    finally:
        runner.close()


if __name__ == "__main__":
    sys.exit(main())
