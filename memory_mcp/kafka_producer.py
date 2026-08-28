"""
Kafka producer for sharing and governing memories across team LTM nodes.

Produces memory events to a Kafka topic (default: ltm-project-memories)
so that team members can consume, ingest, update, and — critically —
remove shared memories on instruction from a senior/admin.

Event types
-----------
  remember  — new memory created; key = memory_id
  update    — memory content changed; key = content_hash (dedup: consumer
              checks the hash against its vector store and skips re-embedding
              when the hash already exists)
  delete    — RPC instruction to remove a memory from every consuming node;
              key = memory_id.  The message includes both memory_id and
              content_hash so consumers can locate the memory by either
              identifier and purge it from SQLite + the vector store.

Architecture (modeled on cbt-chat NPR Kafka patterns):
  - SCRAM-SHA-512 authentication (same as MSK clusters)
  - Access control via ALLOWED_KAFKA_USERS in .env (username:node_uuid pairs)
  - Only listed users may produce; consuming nodes honour delete events
    only from sources that appear in their own ALLOWED_KAFKA_USERS list
    (i.e. the senior who issued the delete must be a known trusted user
    on each consumer).

Usage:
    from memory_mcp.kafka_producer import KafkaMemoryProducer

    producer = KafkaMemoryProducer(identity=identity)
    producer.start()
    producer.produce_memory(memory_dict, event="remember")
    producer.produce_memory(memory_dict, event="update")
    producer.produce_delete(memory_id="mem_...", content_hash="ab12...")
    producer.stop()
"""

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .identity import NodeIdentity

logger = logging.getLogger(__name__)

# ── Valid event types ─────────────────────────────────────────────────────────

VALID_EVENTS = frozenset({"remember", "update", "delete"})

# ── Kafka library import (optional dependency) ────────────────────────────────

try:
    from confluent_kafka import Producer as ConfluentProducer

    _CONFLUENT_AVAILABLE = True
except ImportError:
    _CONFLUENT_AVAILABLE = False

try:
    from kafka import KafkaProducer as PythonKafkaProducer

    _KAFKA_PYTHON_AVAILABLE = True
except ImportError:
    _KAFKA_PYTHON_AVAILABLE = False


# ── .env loading ─────────────────────────────────────────────────────────────

def _load_env_file(env_path: str = ".env") -> dict[str, str]:
    """Load key=value pairs from a .env file (no shell expansion)."""
    result: dict[str, str] = {}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                result[key] = value
    except FileNotFoundError:
        pass
    return result


def _get_config(env_path: str = "") -> dict[str, str]:
    """Resolve Kafka config from .env file, falling back to os.environ."""
    if not env_path:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, ".env")

    file_vars = _load_env_file(env_path)

    def _get(key: str, default: str = "") -> str:
        return file_vars.get(key, os.environ.get(key, default))

    return {
        "brokers": _get("KAFKA_BROKERS"),
        "client_id": _get("KAFKA_CLIENT_ID", "ltm-mcp"),
        "mechanism": _get("KAFKA_MECHANISM", "scram-sha-512"),
        "username": _get("KAFKA_USERNAME"),
        "password": _get("KAFKA_PASSWORD"),
        "topic": _get("KAFKA_TOPIC", "ltm-project-memories"),
        "allowed_users": _get("ALLOWED_KAFKA_USERS"),
    }


def _parse_allowed_users(raw: str) -> list[tuple[str, str]]:
    """Parse ALLOWED_KAFKA_USERS into [(username, node_uuid), ...].

    Format: "Ohan:019532ab-...,Alice:019532ab-..."
    """
    if not raw:
        return []
    result = []
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        username, _, node_uuid = entry.partition(":")
        username = username.strip()
        node_uuid = node_uuid.strip()
        if username and node_uuid:
            result.append((username, node_uuid))
    return result


class KafkaMemoryProducer:
    """Kafka producer that publishes memory lifecycle events to a shared topic.

    Access control:
        Only users listed in ALLOWED_KAFKA_USERS can produce.
        Each entry is "username:node_uuid" matching the local identity.json.

    Key strategy:
        - remember: key = memory_id  (e.g. mem_41ac62b2_6f3f5354a2961e21)
        - update:   key = content_hash  (consumer dedup — skip re-embedding)
        - delete:   key = memory_id  (RPC: consumers remove by id + hash)

    Message value always includes content_hash so consumers can:
        1. Check if the content already exists in their vector store (skip)
        2. Locate a memory by content_hash when the id differs (ingested peer
           memories get new local ids)
    """

    def __init__(
        self,
        identity: "NodeIdentity",
        env_path: str = "",
    ):
        self._identity = identity
        self._config = _get_config(env_path)
        self._topic = self._config["topic"]
        self._producer: Optional[object] = None
        self._backend: Optional[str] = None  # "confluent" or "kafka-python"
        self._lock = threading.Lock()
        self._started = False
        self._allowed_users = _parse_allowed_users(self._config["allowed_users"])

    # ── Access control ───────────────────────────────────────────────────────

    def is_user_allowed(self, username: str = "", node_uuid: str = "") -> bool:
        """Check if a user is in the allowed list.

        With no args, checks the current node identity.
        With args, checks an arbitrary user (used by consumers to validate
        the source of a delete event).
        """
        if not self._allowed_users:
            logger.warning(
                "[kafka-producer] ALLOWED_KAFKA_USERS is empty — "
                "no one is authorized to produce"
            )
            return False

        check_username = username or self._identity.username
        check_uuid = node_uuid or self._identity.node_uuid

        for allowed_user, allowed_uuid in self._allowed_users:
            if check_username == allowed_user and check_uuid == allowed_uuid:
                return True

        if not username:
            logger.warning(
                "[kafka-producer] User %s (%s) is NOT in ALLOWED_KAFKA_USERS",
                check_username,
                check_uuid,
            )
        return False

    @property
    def is_configured(self) -> bool:
        """True if Kafka brokers and credentials are present."""
        return bool(
            self._config["brokers"]
            and self._config["username"]
            and self._config["password"]
        )

    @property
    def is_ready(self) -> bool:
        """True if producer is started and the local user is allowed."""
        return self._started and self.is_user_allowed()

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def allowed_users(self) -> list[tuple[str, str]]:
        return list(self._allowed_users)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Initialize the Kafka producer. Returns True on success."""
        if not self.is_configured:
            logger.info(
                "[kafka-producer] Kafka not configured (missing brokers/credentials) "
                "— memory sharing via Kafka disabled"
            )
            return False

        if not self.is_user_allowed():
            logger.info(
                "[kafka-producer] Current user not in ALLOWED_KAFKA_USERS — "
                "Kafka producing disabled"
            )
            return False

        brokers = self._config["brokers"].replace("|", ",")

        try:
            if _CONFLUENT_AVAILABLE:
                self._producer = ConfluentProducer(
                    {
                        "bootstrap.servers": brokers,
                        "client.id": self._config["client_id"],
                        "security.protocol": "SASL_SSL",
                        "sasl.mechanism": self._config["mechanism"].upper(),
                        "sasl.username": self._config["username"],
                        "sasl.password": self._config["password"],
                        "acks": "all",
                        "retries": 3,
                        "retry.backoff.ms": 1000,
                    }
                )
                self._backend = "confluent"
            elif _KAFKA_PYTHON_AVAILABLE:
                self._producer = PythonKafkaProducer(
                    bootstrap_servers=brokers.split(","),
                    client_id=self._config["client_id"],
                    security_protocol="SASL_SSL",
                    sasl_mechanism=self._config["mechanism"].upper(),
                    sasl_plain_username=self._config["username"],
                    sasl_plain_password=self._config["password"],
                    acks="all",
                    retries=3,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                )
                self._backend = "kafka-python"
            else:
                logger.warning(
                    "[kafka-producer] No Kafka library installed. "
                    "Install with: pip install confluent-kafka  OR  pip install kafka-python"
                )
                return False

            self._started = True
            logger.info(
                "[kafka-producer] Started (backend=%s, topic=%s, user=%s)",
                self._backend,
                self._topic,
                self._identity.username,
            )
            return True

        except Exception as exc:
            logger.error("[kafka-producer] Failed to start: %s", exc)
            return False

    def stop(self) -> None:
        """Flush and close the producer."""
        if not self._started:
            return
        try:
            if self._backend == "confluent" and self._producer:
                self._producer.flush(timeout=5)
            elif self._backend == "kafka-python" and self._producer:
                self._producer.flush(timeout=5)
                self._producer.close(timeout=5)
        except Exception as exc:
            logger.warning("[kafka-producer] Error stopping: %s", exc)
        self._started = False
        self._producer = None
        logger.info("[kafka-producer] Stopped")

    def reload_allowed_users(self, env_path: str = "") -> None:
        """Re-read ALLOWED_KAFKA_USERS from .env (hot reload without restart)."""
        self._config = _get_config(env_path)
        self._allowed_users = _parse_allowed_users(self._config["allowed_users"])
        logger.info(
            "[kafka-producer] Reloaded allowed users: %d entries",
            len(self._allowed_users),
        )

    # ── Produce: memory events ───────────────────────────────────────────────

    def produce_memory(
        self,
        memory: dict,
        event: str = "remember",
        content_hash: str = "",
    ) -> bool:
        """Produce a remember or update event for a memory.

        Args:
            memory: Memory dict (must have 'id', 'title', 'content', etc.)
            event: "remember" or "update"
            content_hash: For updates, the hash of the updated content.
                          Consumers use this to skip re-embedding when
                          the hash already exists in their vector store.

        Key strategy:
            - remember: key = memory_id (e.g. mem_41ac62b2_6f3f5354a2961e21)
            - update:   key = content_hash (dedup on consumer side)
        """
        if event not in ("remember", "update"):
            logger.error("[kafka-producer] Invalid event type: %s", event)
            return False

        if not self._started or not self._producer:
            return False
        if not self.is_user_allowed():
            return False

        computed_hash = content_hash or self._compute_hash(memory)

        if event == "remember":
            msg_key = memory.get("id", "")
        else:
            msg_key = computed_hash

        message = {
            "event": event,
            "memory": _serialize_memory(memory),
            "content_hash": computed_hash,
            "source": self._identity.username,
            "produced_at": datetime.now(timezone.utc).isoformat(),
        }

        return self._send(msg_key, message, event, memory.get("id", "?"))

    # ── Produce: delete RPC ──────────────────────────────────────────────────

    def produce_delete(
        self,
        memory_id: str,
        content_hash: str = "",
        title: str = "",
        reason: str = "",
    ) -> bool:
        """Produce a delete event — an RPC instruction to all consuming nodes.

        This tells every LTM node on the topic to remove the specified memory
        from their local SQLite + vector store.  Consumers validate that the
        source user (username:node_uuid) is in their own ALLOWED_KAFKA_USERS
        before honouring the delete.

        Args:
            memory_id: The original memory id (e.g. mem_41ac62b2_6f3f5354a2961e21).
            content_hash: SHA-256 of the memory content.  Consumers that ingested
                          the memory under a different local id can still locate
                          it by hash.
            title: Optional — included for human-readable audit logs.
            reason: Optional — why the senior is removing this memory.

        Key: memory_id (same as remember — ensures ordering per memory).
        """
        if not self._started or not self._producer:
            return False
        if not self.is_user_allowed():
            return False

        message = {
            "event": "delete",
            "memory_id": memory_id,
            "content_hash": content_hash,
            "title": title,
            "reason": reason,
            "source": self._identity.username,
            "produced_at": datetime.now(timezone.utc).isoformat(),
        }

        return self._send(memory_id, message, "delete", memory_id)

    # ── Batch produce ────────────────────────────────────────────────────────

    def produce_batch(
        self,
        memories: list[dict],
        event: str = "remember",
    ) -> dict:
        """Produce multiple memories. Returns {produced: int, failed: int}."""
        produced = 0
        failed = 0
        for mem in memories:
            content_hash = self._compute_hash(mem)
            if self.produce_memory(mem, event=event, content_hash=content_hash):
                produced += 1
            else:
                failed += 1
        self._flush()
        return {"produced": produced, "failed": failed}

    def produce_delete_batch(
        self,
        items: list[dict],
        reason: str = "",
    ) -> dict:
        """Produce delete events for multiple memories.

        Args:
            items: List of dicts, each with at least 'id' and optionally
                   'content_hash' and 'title'.
            reason: Shared reason for all deletes.

        Returns: {produced: int, failed: int}
        """
        produced = 0
        failed = 0
        for item in items:
            if self.produce_delete(
                memory_id=item.get("id", ""),
                content_hash=item.get("content_hash", ""),
                title=item.get("title", ""),
                reason=reason,
            ):
                produced += 1
            else:
                failed += 1
        self._flush()
        return {"produced": produced, "failed": failed}

    # ── Internal send / flush ────────────────────────────────────────────────

    def _send(self, key: str, message: dict, event: str, label: str) -> bool:
        """Send a single message to the topic."""
        try:
            if self._backend == "confluent":
                self._producer.produce(
                    topic=self._topic,
                    key=key.encode("utf-8") if key else None,
                    value=json.dumps(message).encode("utf-8"),
                    callback=self._delivery_callback,
                )
                self._producer.poll(0)
            elif self._backend == "kafka-python":
                self._producer.send(
                    self._topic,
                    key=key,
                    value=message,
                )

            logger.info(
                "[kafka-producer] Produced %s event for %s (key=%s)",
                event,
                label,
                key[:24] if key else "none",
            )
            return True

        except Exception as exc:
            logger.error(
                "[kafka-producer] Failed to produce %s for %s: %s",
                event,
                label,
                exc,
            )
            return False

    def _flush(self) -> None:
        """Flush the producer (call after batches)."""
        if not self._started or not self._producer:
            return
        try:
            if self._backend == "confluent":
                self._producer.flush(timeout=10)
            elif self._backend == "kafka-python":
                self._producer.flush(timeout=10)
        except Exception as exc:
            logger.warning("[kafka-producer] Flush error: %s", exc)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(memory: dict) -> str:
        """Compute SHA-256 content hash for dedup keying."""
        content = memory.get("content", "")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _delivery_callback(err, msg):
        """Confluent Kafka delivery report callback."""
        if err:
            logger.error(
                "[kafka-producer] Delivery failed for %s: %s",
                msg.key(),
                err,
            )
        else:
            logger.debug(
                "[kafka-producer] Delivered to %s [%d] @ %d",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )


def _serialize_memory(memory: dict) -> dict:
    """Ensure all values are JSON-serializable."""
    out = {}
    for k, v in memory.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, (list, dict, str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out
