"""
Audit logger for memory tool calls.

Appends one JSON line per tool invocation to a daily-rotating JSONL file.
Used by both the live MCP server (server.py → register_tools) and the
eval/replay CLI (eval_tool.py → ToolRunner).

File layout::

    <audit_dir>/audit_YYYY-MM-DD.jsonl

Each line is a JSON object with the schema::

    {
        "call_id":    "<uuid4>",
        "timestamp":  "<ISO-8601 UTC>",   # captured BEFORE dispatch
        "tool":       "<tool_name>",
        "token_count": <int> | null,      # total tokens across all result items
        "args":       { ... },            # raw kwargs
        "success":    true | false,
        "duration_ms": <int>,
        "result":     { ... },            # full jsonified result
        "error":      null | "<traceback>"
    }
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _default_audit_dir() -> Path:
    """Return the default audit directory (respects AI_COMPANION_DATA_DIR)."""
    # Import lazily to avoid circular imports at module load time.
    from .config import DATA_FOLDER  # type: ignore[import]

    return DATA_FOLDER / "audit"


class AuditLogger:
    """Appends one JSON line per tool call to a daily-rotating JSONL file.

    Thread-safe: a per-instance lock serializes all writes so that concurrent
    calls from multiple threads cannot produce interleaved JSONL lines.
    """

    def __init__(self, audit_dir: Optional[Path] = None) -> None:
        if audit_dir is None:
            audit_dir = _default_audit_dir()
        self.audit_dir = Path(audit_dir)
        self._lock = threading.Lock()
        try:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Cannot create audit directory {self.audit_dir!r}: {exc}. "
                "Override with --audit-dir or set AI_COMPANION_DATA_DIR."
            ) from exc

    def _today_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.audit_dir / f"audit_{date_str}.jsonl"

    def write(self, record: Dict[str, Any]) -> None:
        """Append *record* as a JSON line.

        Failures are logged at WARNING level but never re-raised, so that
        an audit I/O problem never interferes with the actual tool call.
        """
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            with self._lock:
                with open(self._today_path(), "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("Audit write failed (non-fatal): %s", exc)

    @staticmethod
    def load(path: Path) -> List[Dict[str, Any]]:
        """Load all records from a JSONL audit file."""
        records = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {lineno} in {path}: {exc}"
                    ) from exc
        return records
