"""
Option A — LAN Memory Sharing via mDNS + HTTP pull.

Architecture
------------
Each running instance:
  1. Advertises itself on the LAN as a _ltm-mcp._tcp mDNS service so peers
     can discover it without any manual IP configuration.
  2. Exposes GET /shared/memories on the existing HTTP server — returns all
     memories where shared=1 as JSON.
  3. Runs a background poller that queries every discovered peer's
     /shared/memories endpoint and ingests new memories into the local store.

Ingest rules
------------
- Peer memories are re-embedded locally (not trusted blindly — model versions
  may differ).
- Deduplication uses the existing content_hash check in remember().
- Ingested memories carry a "peer:<node_id>" tag so their origin is traceable
  and they can be filtered out of local searches if desired.
- Memories that originated from another peer (have a peer: tag) are NOT
  re-broadcast — we only share our own memories, not re-relay peer ones.

Usage
-----
    from memory_mcp.network_sharing import NetworkSharingManager

    mgr = NetworkSharingManager(
        memory_system=memory_system,
        http_host="127.0.0.1",   # host the MCP server is bound to
        http_port=8000,
        poll_interval=300,        # seconds between peer polls (default 5 min)
    )
    mgr.start()          # call once after the HTTP server starts
    ...
    mgr.stop()           # call on shutdown

The shared HTTP route is registered separately via FastMCP's custom_route
decorator in server.py — NetworkSharingManager only handles discovery and polling.
"""

import hashlib
import json
import logging
import socket
import threading
import time
import uuid
from typing import TYPE_CHECKING

import requests

try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

    _ZEROCONF_AVAILABLE = True
except ImportError:
    _ZEROCONF_AVAILABLE = False

if TYPE_CHECKING:
    from .memory_system import RobustMemorySystem

logger = logging.getLogger(__name__)

# mDNS service type — _ltm-mcp._tcp.local.
_SERVICE_TYPE = "_ltm-mcp._tcp.local."

# Prefix added to tags on ingested peer memories
_PEER_TAG_PREFIX = "peer:"


def _node_id() -> str:
    """Stable per-machine node ID derived from hostname + MAC address."""
    raw = f"{socket.gethostname()}-{uuid.getnode()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _is_peer_memory(tags: list) -> bool:
    """Return True if any tag marks this memory as sourced from a peer."""
    return any(t.startswith(_PEER_TAG_PREFIX) for t in tags)


class _MDNSListener:
    """Zeroconf service listener — maintains a live dict of peer addresses."""

    def __init__(self, own_node_id: str):
        self._own = own_node_id
        self._peers: dict[str, tuple[str, int]] = {}  # node_id -> (host, port)
        self._lock = threading.Lock()

    # ── Zeroconf listener protocol ──────────────────────────────

    def add_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if not info:
            return
        node_id = (
            (info.properties or {})
            .get(b"node_id", b"")
            .decode("utf-8", errors="ignore")
        )
        if not node_id or node_id == self._own:
            return  # ignore ourselves
        host = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
        if not host:
            return
        port = info.port
        with self._lock:
            self._peers[node_id] = (host, port)
        logger.info("Peer discovered: %s at %s:%d", node_id, host, port)

    def update_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
        self.add_service(zc, type_, name)

    def remove_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
        # Extract node_id from name (format: "<node_id>._ltm-mcp._tcp.local.")
        node_id = name.split(".")[0]
        with self._lock:
            removed = self._peers.pop(node_id, None)
        if removed:
            logger.info("Peer removed: %s", node_id)

    # ── Public API ───────────────────────────────────────────────

    def peers(self) -> dict[str, tuple[str, int]]:
        with self._lock:
            return dict(self._peers)


class NetworkSharingManager:
    """
    Manages LAN memory sharing for a single server instance.

    Responsibilities:
    - Advertise this node via mDNS
    - Discover peers via mDNS
    - Poll peers for shared memories on a background thread
    - Ingest new peer memories into the local memory system
    """

    def __init__(
        self,
        memory_system: "RobustMemorySystem",
        http_host: str = "127.0.0.1",
        http_port: int = 8000,
        poll_interval: int = 300,
    ):
        self._ms = memory_system
        self._host = http_host
        self._port = http_port
        self._poll_interval = poll_interval
        self._node_id = _node_id()

        self._zeroconf: "Zeroconf | None" = None
        self._service_info: "ServiceInfo | None" = None
        self._browser: "ServiceBrowser | None" = None
        self._listener: "_MDNSListener | None" = None

        self._poll_thread: threading.Thread | None = None
        self._running = False

        # Track which (peer_node_id, peer_memory_id) pairs we've already seen
        # so we don't repeatedly attempt to ingest the same memory.
        self._seen: set[str] = set()
        self._seen_lock = threading.Lock()

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """Start mDNS advertisement, discovery, and background polling."""
        if not _ZEROCONF_AVAILABLE:
            logger.warning(
                "zeroconf package not installed — LAN memory sharing disabled. "
                "Install with: pip install zeroconf"
            )
            return

        self._running = True
        self._start_mdns()

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="ltm-peer-poller",
        )
        self._poll_thread.start()
        logger.info(
            "Network sharing started (node_id=%s, port=%d, poll_interval=%ds)",
            self._node_id,
            self._port,
            self._poll_interval,
        )

    def stop(self) -> None:
        """Stop the poller and unregister from mDNS."""
        self._running = False
        self._stop_mdns()
        logger.info("Network sharing stopped")

    @property
    def node_id(self) -> str:
        return self._node_id

    # ── mDNS ─────────────────────────────────────────────────────

    def _start_mdns(self) -> None:
        try:
            # Resolve the actual LAN IP — needed for mDNS advertisement
            # so peers on other machines can reach us.
            lan_ip = self._resolve_lan_ip()
            packed_ip = socket.inet_aton(lan_ip)

            self._zeroconf = Zeroconf()

            self._service_info = ServiceInfo(
                type_=_SERVICE_TYPE,
                name=f"{self._node_id}.{_SERVICE_TYPE}",
                addresses=[packed_ip],
                port=self._port,
                properties={
                    "node_id": self._node_id.encode(),
                    "version": b"1",
                },
                server=f"{socket.gethostname()}.local.",
            )
            self._zeroconf.register_service(self._service_info)

            self._listener = _MDNSListener(own_node_id=self._node_id)
            self._browser = ServiceBrowser(
                self._zeroconf, _SERVICE_TYPE, self._listener
            )
            logger.info(
                "mDNS service registered as %s at %s:%d",
                self._node_id,
                lan_ip,
                self._port,
            )

        except Exception as exc:
            logger.error("Failed to start mDNS: %s", exc)

    def _stop_mdns(self) -> None:
        try:
            if self._zeroconf and self._service_info:
                self._zeroconf.unregister_service(self._service_info)
            if self._zeroconf:
                self._zeroconf.close()
        except Exception as exc:
            logger.warning("Error stopping mDNS: %s", exc)
        self._zeroconf = None
        self._service_info = None
        self._browser = None

    @staticmethod
    def _resolve_lan_ip() -> str:
        """Return this machine's LAN IP by connecting a UDP socket (no packet sent)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    # ── Polling ──────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread: poll all known peers every poll_interval seconds."""
        # Initial short delay so mDNS has time to discover peers after startup
        time.sleep(10)
        while self._running:
            if self._listener:
                for node_id, (host, port) in self._listener.peers().items():
                    try:
                        self._poll_peer(node_id, host, port)
                    except Exception as exc:
                        logger.debug("Poll failed for peer %s: %s", node_id, exc)
            # Sleep in small increments so stop() responds quickly
            for _ in range(self._poll_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _poll_peer(self, node_id: str, host: str, port: int) -> None:
        """Fetch shared memories from one peer and ingest any new ones."""
        url = f"http://{host}:{port}/shared/memories"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Could not reach peer %s at %s: %s", node_id, url, exc)
            return

        try:
            memories = resp.json().get("memories", [])
        except (ValueError, AttributeError):
            logger.warning("Invalid JSON from peer %s", node_id)
            return

        ingested = 0
        for mem in memories:
            peer_mem_id = mem.get("id", "")
            key = f"{node_id}:{peer_mem_id}"
            with self._seen_lock:
                if key in self._seen:
                    continue
                self._seen.add(key)

            ingested += self._ingest(mem, node_id)

        if ingested:
            logger.info("Ingested %d new memories from peer %s", ingested, node_id)

    def _ingest(self, mem: dict, peer_node_id: str) -> int:
        """
        Ingest a single peer memory into the local store.

        Returns 1 if ingested, 0 if skipped (duplicate or invalid).
        """
        try:
            title = mem.get("title", "").strip()
            content = mem.get("content", "").strip()
            if not title or not content:
                return 0

            # Don't re-relay peer memories — only ingest originals
            existing_tags = mem.get("tags", [])
            if isinstance(existing_tags, str):
                try:
                    existing_tags = json.loads(existing_tags)
                except ValueError:
                    existing_tags = [
                        t.strip() for t in existing_tags.split(",") if t.strip()
                    ]
            if _is_peer_memory(existing_tags):
                return 0

            # Build tag list: keep original tags + add peer attribution
            peer_tag = f"{_PEER_TAG_PREFIX}{peer_node_id}"
            tags = [t for t in existing_tags if t] + [peer_tag]

            importance = int(mem.get("importance", 5))
            memory_type = mem.get("memory_type", "conversation")

            # remember() will reject duplicates via content_hash — safe to call blindly.
            # We re-embed locally (discard peer's vector) for a consistent embedding space.
            result = self._ms.remember(
                title=title,
                content=content,
                tags=tags,
                importance=importance,
                memory_type=memory_type,
                shared=False,  # ingested peer memories are private by default
            )

            if result.success:
                return 1
            # "Duplicate content detected" is expected and not an error
            if "Duplicate" in (result.reason or ""):
                return 0
            logger.debug(
                "Ingest skipped for peer memory '%s': %s", title, result.reason
            )
            return 0

        except Exception as exc:
            logger.warning("Failed to ingest peer memory: %s", exc)
            return 0

    # ── Shared memories query ────────────────────────────────────

    def get_shared_memories(self) -> list[dict]:
        """
        Return all local memories where shared=1, formatted for the HTTP endpoint.

        Excludes peer-sourced memories (tagged peer:*) — we only share our own.
        """
        try:
            cursor = self._ms.db.execute(
                "SELECT id, title, content, timestamp, tags, importance, memory_type "
                "FROM memories WHERE shared = 1"
            )
            rows = cursor.fetchall()
            result = []
            for row in rows:
                try:
                    tags = json.loads(row["tags"]) if row["tags"] else []
                except (ValueError, TypeError):
                    tags = []

                # Don't re-broadcast memories that came from a peer
                if _is_peer_memory(tags):
                    continue

                result.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "tags": tags,
                        "importance": row["importance"],
                        "memory_type": row["memory_type"],
                        "source_node": self._node_id,
                    }
                )
            return result
        except Exception as exc:
            logger.error("Failed to query shared memories: %s", exc)
            return []
