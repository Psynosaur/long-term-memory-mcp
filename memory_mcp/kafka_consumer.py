"""
Kafka consumer for ingesting shared memories from other team LTM nodes.

Subscribes to the shared topic (default: ltm-project-memories) and processes
three event types produced by KafkaMemoryProducer on other nodes:

  remember — A senior shared a new memory.
      • Check content_hash against local DB — skip if already exists (no re-embed).
      • Otherwise ingest via memory_system.remember() with a "kafka:<source_uuid>"
        tag so the origin is traceable.

  update — A senior updated a shared memory.
      • Locate the local copy by content_hash (the OLD hash is the message key,
        the NEW full record is in the value).
      • If found locally, update title/content/tags/importance in place.
      • If not found, ingest as a new memory (the sender may have shared it
        before this node joined the topic).
      • The new content_hash in the message value is checked first — if it
        already exists locally, skip entirely (already up-to-date).

  delete — RPC instruction from a senior to remove a memory from every node.
      • Validate that the source user (username:node_uuid) appears in the
        local ALLOWED_KAFKA_USERS list — refuse the delete if not.
      • Locate the memory by original memory_id first, then by content_hash
        (ingested peer memories get new local IDs).
      • Delete from both SQLite and the vector store.

Architecture:
  - Runs on a background daemon thread (non-blocking).
  - Own messages are skipped (source.node_uuid == local identity.node_uuid).
  - Uses the same KAFKA_* and ALLOWED_KAFKA_USERS config from .env.
  - Consumer group: "{KAFKA_CLIENT_ID}-consumer" (each LTM instance in its
    own group so every node sees every message).

Usage:
    from memory_mcp.kafka_consumer import KafkaMemoryConsumer

    consumer = KafkaMemoryConsumer(
        memory_system=memory_system,
        identity=identity,
        kafka_producer=kafka_producer,  # for allowed-user validation
    )
    consumer.start()
    ...
    consumer.stop()
"""

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .identity import NodeIdentity
    from .kafka_producer import KafkaMemoryProducer
    from .memory_system import RobustMemorySystem

logger = logging.getLogger(__name__)

# Tag prefix for memories ingested via Kafka (distinct from mDNS peer: tag)
_KAFKA_TAG_PREFIX = "kafka:"

# ── Kafka library imports ────────────────────────────────────────────────────

try:
    from confluent_kafka import Consumer as ConfluentConsumer, KafkaError

    _CONFLUENT_AVAILABLE = True
except ImportError:
    _CONFLUENT_AVAILABLE = False

try:
    from kafka import KafkaConsumer as PythonKafkaConsumer

    _KAFKA_PYTHON_AVAILABLE = True
except ImportError:
    _KAFKA_PYTHON_AVAILABLE = False


def _load_env_file(env_path: str = ".env") -> dict[str, str]:
    """Load key=value pairs from a .env file."""
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
                result[key.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return result


def _get_config(env_path: str = "") -> dict[str, str]:
    """Resolve Kafka config from .env, falling back to os.environ."""
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


def _is_kafka_memory(tags: list) -> bool:
    """Return True if any tag marks this memory as ingested via Kafka."""
    return any(
        isinstance(t, str) and t.startswith(_KAFKA_TAG_PREFIX) for t in tags
    )


class KafkaMemoryConsumer:
    """Background Kafka consumer that ingests shared memories from peers.

    Each LTM instance runs in its own consumer group so every node
    receives every message (fan-out, not competing consumers).
    """

    def __init__(
        self,
        memory_system: "RobustMemorySystem",
        identity: "NodeIdentity",
        kafka_producer: "Optional[KafkaMemoryProducer]" = None,
        env_path: str = "",
    ):
        self._ms = memory_system
        self._identity = identity
        self._kafka_producer = kafka_producer  # for allowed-user validation
        self._config = _get_config(env_path)
        self._topic = self._config["topic"]
        self._consumer: Optional[object] = None
        self._backend: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Stats
        self._stats = {
            "ingested": 0,
            "updated": 0,
            "deleted": 0,
            "skipped_own": 0,
            "skipped_duplicate": 0,
            "skipped_unauthorized_delete": 0,
            "errors": 0,
        }
        self._stats_lock = threading.Lock()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return bool(
            self._config["brokers"]
            and self._config["username"]
            and self._config["password"]
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def _inc(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + 1

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the background consumer thread. Returns True on success."""
        if not self.is_configured:
            logger.info(
                "[kafka-consumer] Kafka not configured — consumer disabled"
            )
            return False

        brokers = self._config["brokers"].replace("|", ",")
        # Each node gets its own consumer group so all nodes see all messages
        group_id = f"{self._config['client_id']}-{self._identity.node_uuid[:8]}"

        try:
            if _CONFLUENT_AVAILABLE:
                self._consumer = ConfluentConsumer(
                    {
                        "bootstrap.servers": brokers,
                        "group.id": group_id,
                        "client.id": f"{self._config['client_id']}-consumer",
                        "security.protocol": "SASL_SSL",
                        "sasl.mechanism": self._config["mechanism"].upper(),
                        "sasl.username": self._config["username"],
                        "sasl.password": self._config["password"],
                        "auto.offset.reset": "earliest",
                        "enable.auto.commit": True,
                        "auto.commit.interval.ms": 5000,
                        "session.timeout.ms": 30000,
                    }
                )
                self._consumer.subscribe([self._topic])
                self._backend = "confluent"

            elif _KAFKA_PYTHON_AVAILABLE:
                self._consumer = PythonKafkaConsumer(
                    self._topic,
                    bootstrap_servers=brokers.split(","),
                    group_id=group_id,
                    client_id=f"{self._config['client_id']}-consumer",
                    security_protocol="SASL_SSL",
                    sasl_mechanism=self._config["mechanism"].upper(),
                    sasl_plain_username=self._config["username"],
                    sasl_plain_password=self._config["password"],
                    auto_offset_reset="earliest",
                    enable_auto_commit=True,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    key_deserializer=lambda k: k.decode("utf-8") if k else None,
                    consumer_timeout_ms=1000,  # poll returns after 1s if no msgs
                )
                self._backend = "kafka-python"
            else:
                logger.warning(
                    "[kafka-consumer] No Kafka library installed — consumer disabled"
                )
                return False

            self._running = True
            self._thread = threading.Thread(
                target=self._consume_loop,
                daemon=True,
                name="ltm-kafka-consumer",
            )
            self._thread.start()
            logger.info(
                "[kafka-consumer] Started (backend=%s, topic=%s, group=%s)",
                self._backend,
                self._topic,
                group_id,
            )
            return True

        except Exception as exc:
            logger.error("[kafka-consumer] Failed to start: %s", exc)
            return False

    def stop(self) -> None:
        """Stop the consumer thread and close the connection."""
        self._running = False
        # Give the thread a moment to exit its poll loop
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        try:
            if self._backend == "confluent" and self._consumer:
                self._consumer.close()
            elif self._backend == "kafka-python" and self._consumer:
                self._consumer.close()
        except Exception as exc:
            logger.warning("[kafka-consumer] Error stopping: %s", exc)
        self._consumer = None
        logger.info("[kafka-consumer] Stopped — stats: %s", self.stats)

    # ── Consume loop ─────────────────────────────────────────────────────────

    def _consume_loop(self) -> None:
        """Background thread: poll for messages and dispatch to handlers."""
        # Short delay so the server is fully up
        time.sleep(3)

        while self._running:
            try:
                if self._backend == "confluent":
                    self._poll_confluent()
                elif self._backend == "kafka-python":
                    self._poll_kafka_python()
            except Exception as exc:
                logger.error(
                    "[kafka-consumer] Error in consume loop: %s", exc, exc_info=True
                )
                self._inc("errors")
                time.sleep(5)  # back off on error

    def _poll_confluent(self) -> None:
        """Poll using confluent-kafka."""
        msg = self._consumer.poll(timeout=1.0)
        if msg is None:
            return
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                return  # end of partition, not an error
            logger.error("[kafka-consumer] Consumer error: %s", msg.error())
            self._inc("errors")
            return

        try:
            value = json.loads(msg.value().decode("utf-8"))
            key = msg.key().decode("utf-8") if msg.key() else None
            self._handle_message(key, value)
        except Exception as exc:
            logger.error(
                "[kafka-consumer] Failed to process message: %s", exc, exc_info=True
            )
            self._inc("errors")

    def _poll_kafka_python(self) -> None:
        """Poll using kafka-python."""
        try:
            # kafka-python returns messages in batches when iterated
            # consumer_timeout_ms=1000 makes it return after 1s if no messages
            for msg in self._consumer:
                if not self._running:
                    break
                try:
                    self._handle_message(msg.key, msg.value)
                except Exception as exc:
                    logger.error(
                        "[kafka-consumer] Failed to process message: %s",
                        exc,
                        exc_info=True,
                    )
                    self._inc("errors")
        except StopIteration:
            pass  # consumer_timeout_ms expired, no messages

    # ── Message dispatch ─────────────────────────────────────────────────────

    def _handle_message(self, key: Optional[str], value: dict) -> None:
        """Route a message to the appropriate handler."""
        event = value.get("event")
        source = value.get("source", {})
        source_uuid = source.get("node_uuid", "")
        source_username = source.get("username", "")

        # Skip own messages — don't re-ingest what we produced
        if source_uuid == self._identity.node_uuid:
            self._inc("skipped_own")
            return

        logger.info(
            "[kafka-consumer] Received %s event from %s (%s) key=%s",
            event,
            source_username,
            source_uuid[:8] if source_uuid else "?",
            (key or "")[:24],
        )

        if event == "remember":
            self._handle_remember(value, source_username, source_uuid)
        elif event == "update":
            self._handle_update(value, source_username, source_uuid)
        elif event == "delete":
            self._handle_delete(value, source_username, source_uuid)
        else:
            logger.warning(
                "[kafka-consumer] Unknown event type: %s", event
            )

    # ── Event handlers ───────────────────────────────────────────────────────

    def _handle_remember(
        self, value: dict, source_username: str, source_uuid: str
    ) -> None:
        """Ingest a new shared memory, skipping if content_hash already exists."""
        memory = value.get("memory", {})
        content_hash = value.get("content_hash", "")

        # Check if this content already exists locally (by hash) — skip re-embed
        if content_hash and self._hash_exists(content_hash):
            logger.debug(
                "[kafka-consumer] Skipping remember — content_hash %s already exists",
                content_hash[:12],
            )
            self._inc("skipped_duplicate")
            return

        self._ingest_memory(memory, source_username, source_uuid)

    def _handle_update(
        self, value: dict, source_username: str, source_uuid: str
    ) -> None:
        """Update an existing local copy, or ingest if not found.

        The new content_hash is checked first — if it already exists
        locally, the memory is already up-to-date and we skip.
        """
        memory = value.get("memory", {})
        content_hash = value.get("content_hash", "")

        # If the NEW content already exists locally, we're up-to-date
        if content_hash and self._hash_exists(content_hash):
            logger.debug(
                "[kafka-consumer] Skipping update — new content_hash %s already exists",
                content_hash[:12],
            )
            self._inc("skipped_duplicate")
            return

        # Try to find the local copy:
        # 1. By original memory_id (if we ingested it with same id)
        # 2. By any kafka tag matching the source
        # 3. By title + source tag (best effort)
        original_id = memory.get("id", "")
        local_id = self._find_local_memory(original_id, source_uuid, memory)

        if local_id:
            # Update in place
            try:
                title = memory.get("title")
                content = memory.get("content")
                tags_raw = memory.get("tags", [])
                tags = self._parse_tags(tags_raw)
                importance = memory.get("importance")
                memory_type = memory.get("memory_type")

                self._ms.update_memory(
                    memory_id=local_id,
                    title=title,
                    content=content,
                    tags=tags if tags else None,
                    importance=int(importance) if importance is not None else None,
                    memory_type=memory_type,
                )
                logger.info(
                    "[kafka-consumer] Updated local memory %s from %s",
                    local_id,
                    source_username,
                )
                self._inc("updated")
            except Exception as exc:
                logger.error(
                    "[kafka-consumer] Failed to update %s: %s", local_id, exc
                )
                self._inc("errors")
        else:
            # Not found locally — ingest as new
            self._ingest_memory(memory, source_username, source_uuid)

    def _handle_delete(
        self, value: dict, source_username: str, source_uuid: str
    ) -> None:
        """Delete a memory from local store on instruction from a senior.

        The source must be in ALLOWED_KAFKA_USERS on this node.
        """
        # Validate the source is an allowed/trusted user
        if not self._is_source_allowed(source_username, source_uuid):
            logger.warning(
                "[kafka-consumer] Refusing delete from untrusted source: %s (%s)",
                source_username,
                source_uuid[:8] if source_uuid else "?",
            )
            self._inc("skipped_unauthorized_delete")
            return

        memory_id = value.get("memory_id", "")
        content_hash = value.get("content_hash", "")
        title = value.get("title", "")
        reason = value.get("reason", "")

        # Find the local memory: by original id, then by content_hash
        local_ids = self._find_deletable_memories(memory_id, content_hash)

        if not local_ids:
            logger.info(
                "[kafka-consumer] Delete target not found locally: id=%s hash=%s",
                memory_id,
                content_hash[:12] if content_hash else "?",
            )
            return

        for lid in local_ids:
            try:
                result = self._ms.delete_memory(lid)
                if result.success:
                    logger.info(
                        "[kafka-consumer] Deleted local memory %s "
                        "(original=%s, title='%s', reason='%s', by=%s)",
                        lid,
                        memory_id,
                        title,
                        reason,
                        source_username,
                    )
                    self._inc("deleted")
                else:
                    logger.warning(
                        "[kafka-consumer] Delete failed for %s: %s",
                        lid,
                        result.reason,
                    )
            except Exception as exc:
                logger.error(
                    "[kafka-consumer] Exception deleting %s: %s", lid, exc
                )
                self._inc("errors")

    # ── Ingest helper ────────────────────────────────────────────────────────

    def _ingest_memory(
        self, memory: dict, source_username: str, source_uuid: str
    ) -> None:
        """Ingest a single memory from a Kafka peer."""
        title = (memory.get("title") or "").strip()
        content = (memory.get("content") or "").strip()
        if not title or not content:
            return

        # Parse tags, skip if it was already a kafka-ingested memory
        tags_raw = memory.get("tags", [])
        existing_tags = self._parse_tags(tags_raw)
        if _is_kafka_memory(existing_tags):
            return  # don't re-relay

        # Add kafka source tag
        kafka_tag = f"{_KAFKA_TAG_PREFIX}{source_uuid}"
        tags = [t for t in existing_tags if t] + [kafka_tag]

        try:
            result = self._ms.remember(
                title=title,
                content=content,
                tags=tags,
                importance=int(memory.get("importance", 5)),
                memory_type=memory.get("memory_type", "conversation"),
                shared_with=[],  # ingested memories are private locally
            )

            if result.success:
                logger.info(
                    "[kafka-consumer] Ingested '%s' from %s",
                    title,
                    source_username,
                )
                self._inc("ingested")
            elif "Duplicate" in (result.reason or ""):
                self._inc("skipped_duplicate")
            else:
                logger.warning(
                    "[kafka-consumer] Failed to ingest '%s': %s",
                    title,
                    result.reason,
                )
                self._inc("errors")
        except Exception as exc:
            logger.error(
                "[kafka-consumer] Exception ingesting '%s': %s",
                title,
                exc,
                exc_info=True,
            )
            self._inc("errors")

    # ── Lookup helpers ───────────────────────────────────────────────────────

    def _hash_exists(self, content_hash: str) -> bool:
        """Check if a content_hash already exists in the local DB."""
        try:
            cursor = self._ms.db.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (content_hash,)
            )
            return cursor.fetchone() is not None
        except Exception:
            return False

    def _find_local_memory(
        self, original_id: str, source_uuid: str, memory: dict
    ) -> Optional[str]:
        """Try to find the local copy of a remote memory.

        Search order:
        1. By original memory_id (if this node ingested with the same id)
        2. By kafka:<source_uuid> tag + title match
        """
        # 1. Direct id match
        if original_id:
            try:
                cursor = self._ms.db.execute(
                    "SELECT id FROM memories WHERE id = ?", (original_id,)
                )
                row = cursor.fetchone()
                if row:
                    return row["id"]
            except Exception:
                pass

        # 2. Tag + title match
        kafka_tag = f"{_KAFKA_TAG_PREFIX}{source_uuid}"
        title = memory.get("title", "")
        if title:
            try:
                cursor = self._ms.db.execute(
                    "SELECT id, tags FROM memories WHERE title = ?", (title,)
                )
                for row in cursor.fetchall():
                    tags = row["tags"] or "[]"
                    try:
                        tag_list = json.loads(tags) if isinstance(tags, str) else tags
                    except (ValueError, TypeError):
                        tag_list = []
                    if kafka_tag in tag_list:
                        return row["id"]
            except Exception:
                pass

        return None

    def _find_deletable_memories(
        self, memory_id: str, content_hash: str
    ) -> list[str]:
        """Find local memories matching either memory_id or content_hash."""
        ids = set()

        if memory_id:
            try:
                cursor = self._ms.db.execute(
                    "SELECT id FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                if row:
                    ids.add(row["id"])
            except Exception:
                pass

        if content_hash:
            try:
                cursor = self._ms.db.execute(
                    "SELECT id FROM memories WHERE content_hash = ?",
                    (content_hash,),
                )
                for row in cursor.fetchall():
                    ids.add(row["id"])
            except Exception:
                pass

        return list(ids)

    def _is_source_allowed(self, username: str, node_uuid: str) -> bool:
        """Check if the source of a delete event is a trusted user.

        Uses the kafka_producer's allowed-user list (same .env source).
        """
        if self._kafka_producer:
            return self._kafka_producer.is_user_allowed(username, node_uuid)

        # Fallback: parse allowed users directly from config
        from .kafka_producer import _parse_allowed_users

        allowed = _parse_allowed_users(self._config.get("allowed_users", ""))
        return any(u == username and n == node_uuid for u, n in allowed)

    @staticmethod
    def _parse_tags(tags_raw) -> list[str]:
        """Parse tags from various formats (list, JSON string, comma-separated)."""
        if isinstance(tags_raw, list):
            return [str(t) for t in tags_raw if t]
        if isinstance(tags_raw, str):
            try:
                parsed = json.loads(tags_raw)
                if isinstance(parsed, list):
                    return [str(t) for t in parsed if t]
            except (ValueError, TypeError):
                pass
            return [t.strip() for t in tags_raw.split(",") if t.strip()]
        return []
