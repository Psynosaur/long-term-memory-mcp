"""
knowledge_graph.py — Temporal Entity-Relationship Graph

A local SQLite-backed knowledge graph that stores structured facts as
entity-relationship triples with temporal validity windows.

Key capabilities:
  - Entity nodes (people, projects, tools, concepts, places, etc.)
  - Typed relationship edges (works_at, knows, uses, has_skill, etc.)
  - Temporal validity — every fact carries valid_from / valid_to dates
  - Time-travel queries: "what was true about X on date Y?"
  - Soft invalidation: facts can be expired without deletion
  - Closet reference: each triple links back to the source memory ID

This competes with Neo4j/Zep's temporal graph at zero cost — pure SQLite.

Usage:
    from memory_mcp.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=Path("/data/knowledge_graph.db"))
    kg.add_triple("Alice", "works_at", "Acme Corp", valid_from="2023-01-01",
                  source_memory_id="mem_abc123")
    kg.add_triple("Alice", "uses", "Python", valid_from="2023-01-01")

    # All facts about Alice (current only)
    kg.query_entity("Alice")

    # What was true about Alice in 2024?
    kg.query_entity("Alice", as_of="2024-06-01")

    # Alice changed jobs — mark the old triple expired
    kg.invalidate("Alice", "works_at", "Acme Corp", ended="2025-03-01")
    kg.add_triple("Alice", "works_at", "NewCo", valid_from="2025-03-01")

    # Chronological story of Alice
    kg.timeline("Alice")
"""

import hashlib
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kg_entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    properties  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kg_triples (
    id               TEXT PRIMARY KEY,
    subject          TEXT NOT NULL,
    predicate        TEXT NOT NULL,
    object           TEXT NOT NULL,
    valid_from       TEXT,
    valid_to         TEXT,
    confidence       REAL NOT NULL DEFAULT 1.0,
    source_memory_id TEXT,
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject) REFERENCES kg_entities(id),
    FOREIGN KEY (object)  REFERENCES kg_entities(id)
);

CREATE INDEX IF NOT EXISTS kg_idx_triples_subject   ON kg_triples(subject);
CREATE INDEX IF NOT EXISTS kg_idx_triples_object    ON kg_triples(object);
CREATE INDEX IF NOT EXISTS kg_idx_triples_predicate ON kg_triples(predicate);
CREATE INDEX IF NOT EXISTS kg_idx_triples_valid     ON kg_triples(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS kg_idx_entities_name     ON kg_entities(name);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _entity_id(name: str) -> str:
    """Stable, deterministic entity ID from a display name."""
    return name.lower().strip().replace(" ", "_").replace("'", "").replace('"', "")


def _norm_predicate(predicate: str) -> str:
    """Normalise predicate to snake_case."""
    return predicate.lower().strip().replace(" ", "_").replace("-", "_")


def _today() -> str:
    return date.today().isoformat()


# ── KnowledgeGraph class ───────────────────────────────────────────────────────


class KnowledgeGraph:
    """
    Temporal entity-relationship knowledge graph backed by SQLite.

    Thread-safety: each public method opens, uses, and closes its own
    connection using WAL mode — safe for concurrent readers/writers from
    the same process (FastMCP + WebUI daemon thread).

    Args:
        db_path: Path to the SQLite database file for the KG.
                 Defaults to ``memory_db/knowledge_graph.db`` relative to
                 the RobustMemorySystem db_folder.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        conn = self._conn()
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _ensure_entity(self, conn: sqlite3.Connection, name: str):
        """Insert entity if it doesn't already exist (idempotent)."""
        eid = _entity_id(name)
        conn.execute(
            "INSERT OR IGNORE INTO kg_entities (id, name) VALUES (?, ?)",
            (eid, name),
        )
        return eid

    # ── Write operations ──────────────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add or update an entity node.

        Returns the entity's stable ID.
        """
        eid = _entity_id(name)
        props_json = json.dumps(properties or {})
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO kg_entities (id, name, entity_type, properties)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    properties  = excluded.properties
                """,
                (eid, name, entity_type, props_json),
            )
            conn.commit()
        finally:
            conn.close()
        return eid

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        confidence: float = 1.0,
        source_memory_id: Optional[str] = None,
    ) -> str:
        """
        Add a fact triple: subject → predicate → object.

        Automatically creates entity nodes for subject and object if they
        don't exist.  If an identical (subject, predicate, object) triple
        with no valid_to already exists, returns its ID without inserting
        a duplicate.

        Args:
            subject: The entity the fact is about (e.g. "Alice").
            predicate: The relationship type (e.g. "works_at", "knows").
            obj: The target entity or value (e.g. "Acme Corp").
            valid_from: ISO date when this fact became true (e.g. "2023-01-01").
            valid_to: ISO date when this fact stopped being true (None = still current).
            confidence: 0.0–1.0 confidence score (default 1.0).
            source_memory_id: ID of the memory this fact was extracted from.

        Returns:
            The triple's ID string.

        Examples:
            kg.add_triple("Alice", "works_at", "Acme Corp", valid_from="2023-01-01")
            kg.add_triple("Bob", "knows", "Alice")
            kg.add_triple("Alice", "has_skill", "Python", valid_from="2020-01-01")
        """
        sub_id = _entity_id(subject)
        obj_id = _entity_id(obj)
        pred = _norm_predicate(predicate)

        conn = self._conn()
        try:
            # Auto-create entity nodes
            self._ensure_entity(conn, subject)
            self._ensure_entity(conn, obj)

            # Idempotency: don't insert if an open triple already exists
            existing = conn.execute(
                """SELECT id FROM kg_triples
                   WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL""",
                (sub_id, pred, obj_id),
            ).fetchone()
            if existing:
                return existing["id"]

            # Generate a stable but unique ID
            entropy = (
                f"{sub_id}|{pred}|{obj_id}|{valid_from}|{datetime.utcnow().isoformat()}"
            )
            triple_id = "kt_" + hashlib.sha256(entropy.encode()).hexdigest()[:16]

            conn.execute(
                """INSERT INTO kg_triples
                   (id, subject, predicate, object, valid_from, valid_to, confidence, source_memory_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    triple_id,
                    sub_id,
                    pred,
                    obj_id,
                    valid_from,
                    valid_to,
                    confidence,
                    source_memory_id,
                ),
            )
            conn.commit()
            return triple_id
        finally:
            conn.close()

    def invalidate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        ended: Optional[str] = None,
    ):
        """
        Mark an open triple as no longer valid by setting its valid_to date.

        Only affects triples where valid_to IS NULL (i.e. currently active).
        Historical triples remain unchanged — time-travel queries still work.

        Args:
            subject: Entity subject.
            predicate: Relationship type.
            obj: Object entity or value.
            ended: ISO date the fact ended (defaults to today).

        Example:
            kg.invalidate("Alice", "works_at", "Acme Corp", ended="2025-03-01")
        """
        sub_id = _entity_id(subject)
        obj_id = _entity_id(obj)
        pred = _norm_predicate(predicate)
        ended = ended or _today()

        conn = self._conn()
        try:
            conn.execute(
                """UPDATE kg_triples
                   SET valid_to = ?
                   WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL""",
                (ended, sub_id, pred, obj_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Query operations ──────────────────────────────────────────────────────

    def query_entity(
        self,
        name: str,
        as_of: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity.

        Args:
            name: Entity name to look up.
            as_of: ISO date for time-travel query. If provided, only facts
                   valid on that date are returned.  If None, returns all
                   currently-open facts (valid_to IS NULL).
            direction: "outgoing" (entity is subject), "incoming" (entity
                       is object), or "both".

        Returns:
            List of fact dicts, each with:
                subject, predicate, object, valid_from, valid_to,
                confidence, source_memory_id, current (bool).

        Examples:
            kg.query_entity("Alice")                        # all current facts
            kg.query_entity("Alice", as_of="2024-01-01")   # facts valid then
            kg.query_entity("Alice", direction="both")      # in + out edges
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(
                f"direction must be 'outgoing', 'incoming', or 'both'; got {direction!r}"
            )

        eid = _entity_id(name)
        conn = self._conn()
        results: List[Dict[str, Any]] = []

        try:

            def _time_clause(as_of: Optional[str]) -> str:
                if as_of:
                    return (
                        " AND (t.valid_from IS NULL OR t.valid_from <= :as_of)"
                        " AND (t.valid_to   IS NULL OR t.valid_to   >= :as_of)"
                    )
                return " AND t.valid_to IS NULL"

            if direction in ("outgoing", "both"):
                sql = (
                    "SELECT t.*, e_o.name AS obj_name"
                    " FROM kg_triples t"
                    " JOIN kg_entities e_o ON t.object = e_o.id"
                    " WHERE t.subject = :eid"
                    + _time_clause(as_of)
                    + " ORDER BY t.valid_from ASC"
                )
                params = {"eid": eid, "as_of": as_of}
                for row in conn.execute(sql, params).fetchall():
                    results.append(
                        {
                            "direction": "outgoing",
                            "subject": name,
                            "predicate": row["predicate"],
                            "object": row["obj_name"],
                            "valid_from": row["valid_from"],
                            "valid_to": row["valid_to"],
                            "confidence": row["confidence"],
                            "source_memory_id": row["source_memory_id"],
                            "current": row["valid_to"] is None,
                        }
                    )

            if direction in ("incoming", "both"):
                sql = (
                    "SELECT t.*, e_s.name AS sub_name"
                    " FROM kg_triples t"
                    " JOIN kg_entities e_s ON t.subject = e_s.id"
                    " WHERE t.object = :eid"
                    + _time_clause(as_of)
                    + " ORDER BY t.valid_from ASC"
                )
                params = {"eid": eid, "as_of": as_of}
                for row in conn.execute(sql, params).fetchall():
                    results.append(
                        {
                            "direction": "incoming",
                            "subject": row["sub_name"],
                            "predicate": row["predicate"],
                            "object": name,
                            "valid_from": row["valid_from"],
                            "valid_to": row["valid_to"],
                            "confidence": row["confidence"],
                            "source_memory_id": row["source_memory_id"],
                            "current": row["valid_to"] is None,
                        }
                    )

        finally:
            conn.close()

        return results

    def timeline(
        self, entity_name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get facts in chronological order, optionally filtered by entity.

        Returns all facts (current and expired), sorted by valid_from.
        Useful for understanding the history of an entity over time.

        Args:
            entity_name: If provided, only triples involving this entity.
                         If None, returns the most recent global timeline.
            limit: Maximum number of facts to return (default 100).

        Returns:
            List of fact dicts with subject, predicate, object, valid_from,
            valid_to, current.
        """
        conn = self._conn()
        try:
            if entity_name:
                eid = _entity_id(entity_name)
                rows = conn.execute(
                    """
                    SELECT t.predicate, t.valid_from, t.valid_to, t.confidence,
                           t.source_memory_id,
                           e_s.name AS sub_name, e_o.name AS obj_name
                    FROM kg_triples t
                    JOIN kg_entities e_s ON t.subject = e_s.id
                    JOIN kg_entities e_o ON t.object  = e_o.id
                    WHERE t.subject = ? OR t.object = ?
                    ORDER BY t.valid_from ASC NULLS LAST, t.created_at ASC
                    LIMIT ?
                    """,
                    (eid, eid, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT t.predicate, t.valid_from, t.valid_to, t.confidence,
                           t.source_memory_id,
                           e_s.name AS sub_name, e_o.name AS obj_name
                    FROM kg_triples t
                    JOIN kg_entities e_s ON t.subject = e_s.id
                    JOIN kg_entities e_o ON t.object  = e_o.id
                    ORDER BY t.valid_from ASC NULLS LAST, t.created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [
                {
                    "subject": r["sub_name"],
                    "predicate": r["predicate"],
                    "object": r["obj_name"],
                    "valid_from": r["valid_from"],
                    "valid_to": r["valid_to"],
                    "confidence": r["confidence"],
                    "source_memory_id": r["source_memory_id"],
                    "current": r["valid_to"] is None,
                }
                for r in rows
            ]
        finally:
            conn.close()

    def stats(self) -> Dict[str, Any]:
        """
        Return summary statistics for the knowledge graph.

        Returns:
            Dict with: entities (int), triples (int), current_facts (int),
            expired_facts (int), relationship_types (list of str).
        """
        conn = self._conn()
        try:
            n_entities = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
            n_triples = conn.execute("SELECT COUNT(*) FROM kg_triples").fetchone()[0]
            n_current = conn.execute(
                "SELECT COUNT(*) FROM kg_triples WHERE valid_to IS NULL"
            ).fetchone()[0]
            predicates = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT predicate FROM kg_triples ORDER BY predicate"
                ).fetchall()
            ]
            return {
                "entities": n_entities,
                "triples": n_triples,
                "current_facts": n_current,
                "expired_facts": n_triples - n_current,
                "relationship_types": predicates,
            }
        finally:
            conn.close()

    def close(self):
        """No-op — connections are opened/closed per operation."""
        pass
