"""Node identity for the Long-Term Memory MCP server.

Each server instance has a stable identity stored in data/identity.json:
  {
    "username": "Ohan",
    "node_uuid": "019532ab-...",   # UUIDv7 — time-ordered, globally unique
    "created_at": "2026-03-25T..."
  }

The identity is created once on first run and never regenerated.  It is used
for targeted memory sharing: memories store a `shared_with` JSON list of peer
UUIDs (or ["*"] for broadcast), and the /shared/memories endpoint filters by
the requesting peer's UUID.
"""

import json
import logging
import os
import struct
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_IDENTITY_FILENAME = "identity.json"
_DEFAULT_USERNAME_ENV = "LTM_USERNAME"
_MAX_USERNAME_LEN = 256


# ── UUIDv7 generation ──────────────────────────────────────────────────────────


def _sanitize_username(name: str) -> str:
    """Strip control characters and enforce max length."""
    sanitized = "".join(c for c in name if c >= " " or c == "\t")
    sanitized = sanitized.strip()[:_MAX_USERNAME_LEN]
    return sanitized or "unknown"


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via a temp file + rename."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX; near-atomic on Windows (Python 3.3+)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _uuid7() -> str:
    """Generate a UUIDv7 (time-ordered UUID).

    Format (RFC 9562 §5.7):
      48 bits  unix_ts_ms   — 48-bit millisecond timestamp per spec
       4 bits  version=0x7
      12 bits  rand_a       — top 12 bits of random material
       2 bits  variant=0b10
      62 bits  rand_b       — bottom 62 bits of random material

    We generate 80 bits of randomness (10 bytes), use the top 12 for
    rand_a (bits 79-68) and the bottom 62 for rand_b (bits 61-0),
    discarding 6 bits (62-67). No overlap between the two fields.
    """
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    rand = int.from_bytes(os.urandom(10), "big")  # 80 bits of randomness

    rand_a = (rand >> 68) & 0xFFF  # bits 79-68
    rand_b = rand & 0x3FFFFFFFFFFFFFFF  # bits 61-0 (62 bits)

    hi = (ts_ms << 16) | (0x7 << 12) | rand_a
    lo = (0b10 << 62) | rand_b

    b = struct.pack(">QQ", hi, lo)
    hex_str = b.hex()
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"


# ── NodeIdentity dataclass ─────────────────────────────────────────────────────


@dataclass
class NodeIdentity:
    """Stable identity for this server instance."""

    username: str
    node_uuid: str
    created_at: str

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NodeIdentity":
        return cls(
            username=data["username"],
            node_uuid=data["node_uuid"],
            created_at=data.get("created_at") or datetime.now(timezone.utc).isoformat(),
        )

    # ── Factory ────────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, data_folder: Path, username: str = "") -> "NodeIdentity":
        """Load identity from data_folder/identity.json, creating it if absent.

        The username is resolved in this order:
          1. `username` argument (if non-empty)
          2. LTM_USERNAME environment variable
          3. System hostname (always available, no prompt needed)
        """
        path = data_folder / _IDENTITY_FILENAME

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                identity = cls.from_dict(data)
                resolved_name = _sanitize_username(
                    username
                    or os.environ.get(_DEFAULT_USERNAME_ENV, "").strip()
                    or identity.username
                )
                if resolved_name and resolved_name != identity.username:
                    identity.username = resolved_name
                    try:
                        _atomic_write(path, json.dumps(identity.to_dict(), indent=2))
                        logger.info(
                            "Identity username updated to '%s'", identity.username
                        )
                    except OSError as exc:
                        logger.warning("Could not persist updated username: %s", exc)
                return identity
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("Could not parse identity.json (%s) — recreating", exc)
            except OSError as exc:
                raise  # storage error — propagate, don't silently recreate

        # Create a new identity
        resolved_name = _sanitize_username(
            username
            or os.environ.get(_DEFAULT_USERNAME_ENV, "").strip()
            or _system_username()
        )
        identity = cls(
            username=resolved_name,
            node_uuid=_uuid7(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        data_folder.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(identity.to_dict(), indent=2))
        logger.info(
            "Created new identity: username='%s' uuid=%s",
            identity.username,
            identity.node_uuid,
        )
        return identity


def _system_username() -> str:
    """Return the OS login name as a fallback identity username."""
    import socket as _socket

    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return _socket.gethostname().split(".")[0]
