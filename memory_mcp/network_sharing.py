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

from .identity import NodeIdentity

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
        # node_id -> (host, port, node_uuid, username)
        self._peers: dict[str, tuple[str, int, str, str]] = {}
        self._lock = threading.Lock()

    # ── Zeroconf listener protocol ──────────────────────────────

    def add_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if not info:
            return
        props = info.properties or {}
        node_id = props.get(b"node_id", b"").decode("utf-8", errors="ignore")
        if not node_id or node_id == self._own:
            return
        host = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
        if not host:
            return
        port = info.port
        node_uuid = props.get(b"node_uuid", b"").decode("utf-8", errors="ignore")
        username = (
            props.get(b"username", b"").decode("utf-8", errors="ignore") or node_id
        )
        with self._lock:
            self._peers[node_id] = (host, port, node_uuid, username)
        logger.info(
            "[network-sharing] Peer discovered: %s (%s) at %s:%d",
            username,
            node_id,
            host,
            port,
        )

    def update_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
        self.add_service(zc, type_, name)

    def remove_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
        node_id = name.split(".")[0]
        with self._lock:
            removed = self._peers.pop(node_id, None)
        if removed:
            logger.info("[network-sharing] Peer removed: %s (%s)", removed[3], node_id)

    # ── Public API ───────────────────────────────────────────────

    def peers(self) -> dict[str, tuple[str, int, str, str]]:
        """Return {node_id: (host, port, node_uuid, username)}."""
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
        identity: "NodeIdentity",
        http_host: str = "127.0.0.1",
        http_port: int = 8000,
        poll_interval: int = 300,
    ):
        self._ms = memory_system
        self._identity = identity
        self._host = http_host
        self._port = http_port
        self._poll_interval = poll_interval
        self._node_id = _node_id()  # legacy short hash, kept for mDNS name uniqueness

        self._zeroconf: "Zeroconf | None" = None
        self._service_info: "ServiceInfo | None" = None
        self._browser: "ServiceBrowser | None" = None
        self._listener: "_MDNSListener | None" = None

        self._poll_thread: threading.Thread | None = None
        self._running = False

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
            "[network-sharing] Started — node_id=%s port=%d poll_interval=%ds",
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

    @property
    def node_uuid(self) -> str:
        return self._identity.node_uuid

    @property
    def username(self) -> str:
        return self._identity.username

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
                    "node_uuid": self._identity.node_uuid.encode(),
                    "username": self._identity.username.encode(),
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
        """Return the best LAN IP for mDNS advertisement.

        Uses psutil.net_if_addrs() to enumerate all interfaces and pick the
        first IPv4 address that looks like a real LAN IP, skipping:
        - 127.x.x.x   (loopback)
        - 169.254.x.x (link-local / APIPA)
        - 100.64.x.x  (CGNAT / Tailscale / ZeroTier)

        Falls back to UDP-connect trick only if nothing better is found.
        """
        import ipaddress as _ip

        _SKIP = [
            _ip.ip_network("127.0.0.0/8"),
            _ip.ip_network("169.254.0.0/16"),
            _ip.ip_network("100.64.0.0/10"),  # CGNAT / Tailscale / ZeroTier
        ]

        def _is_lan(addr: str) -> bool:
            try:
                a = _ip.ip_address(addr)
                return a.version == 4 and not any(a in n for n in _SKIP)
            except ValueError:
                return False

        try:
            import psutil

            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    # AF_INET == 2
                    if addr.family == 2 and _is_lan(addr.address):
                        logger.info(
                            "[network-sharing] Resolved LAN IP via %s: %s",
                            iface,
                            addr.address,
                        )
                        return addr.address
        except Exception:
            pass

        # Last resort: UDP connect — may return VPN IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            logger.warning(
                "[network-sharing] Could not find clean LAN IP — using UDP fallback: %s "
                "(may be a VPN address)",
                ip,
            )
            return ip
        except Exception:
            return "127.0.0.1"

    # ── Polling ──────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread: poll all known peers every poll_interval seconds."""
        time.sleep(3)
        while self._running:
            peers = self._listener.peers() if self._listener else {}
            if peers:
                logger.info(
                    "[network-sharing] Polling %d peer(s): %s",
                    len(peers),
                    ", ".join(
                        f"{username}@{host}:{port}"
                        for _, (host, port, _, username) in peers.items()
                    ),
                )
                for node_id, (host, port, peer_uuid, peer_username) in peers.items():
                    try:
                        self._poll_peer(node_id, host, port, peer_uuid, peer_username)
                    except Exception as exc:
                        logger.warning(
                            "[network-sharing] Poll failed for %s@%s:%d: %s",
                            peer_username,
                            host,
                            port,
                            exc,
                        )
                for _ in range(self._poll_interval):
                    if not self._running:
                        break
                    time.sleep(1)
            else:
                logger.info(
                    "[network-sharing] No peers discovered yet, retrying in 15s"
                )
                for _ in range(15):
                    if not self._running:
                        break
                    time.sleep(1)

    def _poll_peer(
        self,
        node_id: str,
        host: str,
        port: int,
        peer_uuid: str,
        peer_username: str,
    ) -> None:
        """Fetch shared memories from one peer, passing our UUID so the peer
        can filter to memories shared with us specifically."""
        own_uuid = self._identity.node_uuid
        url = f"http://{host}:{port}/shared/memories?for={own_uuid}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "[network-sharing] Could not reach peer %s at %s: %s",
                peer_username,
                url,
                exc,
            )
            return

        try:
            data = resp.json()
            memories = data.get("memories", [])
        except (ValueError, AttributeError):
            logger.warning("[network-sharing] Invalid JSON from peer %s", peer_username)
            return

        logger.info(
            "[network-sharing] Peer %s returned %d shared memory/memories",
            peer_username,
            len(memories),
        )

        ingested = 0
        skipped = 0
        for mem in memories:
            peer_mem_id = mem.get("id", "")
            key = f"{node_id}:{peer_mem_id}"
            with self._seen_lock:
                if key in self._seen:
                    skipped += 1
                    continue
                self._seen.add(key)
            ingested += self._ingest(mem, node_id, peer_username)

        logger.info(
            "[network-sharing] Peer %s: ingested=%d skipped_seen=%d",
            peer_username,
            ingested,
            skipped,
        )

    def _ingest(self, mem: dict, peer_node_id: str, peer_username: str = "") -> int:
        """Ingest a single peer memory. Returns 1 if ingested, 0 if skipped."""
        try:
            title = mem.get("title", "").strip()
            content = mem.get("content", "").strip()
            if not title or not content:
                return 0

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

            peer_tag = f"{_PEER_TAG_PREFIX}{peer_node_id}"
            tags = [t for t in existing_tags if t] + [peer_tag]

            result = self._ms.remember(
                title=title,
                content=content,
                tags=tags,
                importance=int(mem.get("importance", 5)),
                memory_type=mem.get("memory_type", "conversation"),
                shared_with=[],  # ingested peer memories are private by default
            )

            if result.success:
                logger.info(
                    "[network-sharing] Ingested '%s' from %s",
                    title,
                    peer_username or peer_node_id,
                )
                return 1
            if "Duplicate" in (result.reason or ""):
                return 0
            logger.warning(
                "[network-sharing] Failed to ingest '%s': %s", title, result.reason
            )
            return 0

        except Exception as exc:
            logger.warning(
                "[network-sharing] Exception ingesting '%s': %s",
                mem.get("title", "?"),
                exc,
                exc_info=True,
            )
            return 0

    # ── Shared memories query ────────────────────────────────────

    def get_shared_memories(self, for_uuid: str = "") -> list[dict]:
        """Return memories visible to the requesting peer.

        Args:
            for_uuid: The UUID of the requesting peer.
                      Returns memories where shared_with contains for_uuid or "*".
                      If empty, returns all shared memories (for_uuid="*" semantics).
        """
        try:
            cursor = self._ms.db.execute(
                "SELECT id, title, content, timestamp, tags, importance, "
                "memory_type, shared_with "
                "FROM memories WHERE shared_with IS NOT NULL AND shared_with != '[]'"
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

                try:
                    shared_with = (
                        json.loads(row["shared_with"]) if row["shared_with"] else []
                    )
                except (ValueError, TypeError):
                    shared_with = []

                # Filter by requesting peer's UUID
                if for_uuid:
                    if "*" not in shared_with and for_uuid not in shared_with:
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
                        "source_username": self._identity.username,
                        "source_uuid": self._identity.node_uuid,
                    }
                )
            return result
        except Exception as exc:
            logger.error("[network-sharing] Failed to query shared memories: %s", exc)
            return []

    def get_known_peers(self) -> list[dict]:
        """Return all currently discovered peers as {node_id, node_uuid, username, host, port}."""
        if not self._listener:
            return []
        return [
            {
                "node_id": node_id,
                "node_uuid": node_uuid,
                "username": username,
                "host": host,
                "port": port,
            }
            for node_id, (
                host,
                port,
                node_uuid,
                username,
            ) in self._listener.peers().items()
        ]
