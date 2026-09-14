"""SQLite evidence store with deterministic hashing and point-in-time reads."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from tradingagents.markets.egypt_sources import (
    EgyptSourceKey,
    validate_evidence_source,
)
from tradingagents.portfolio.models import EvidenceRef


@dataclass(frozen=True)
class EvidenceDocument:
    source_key: EgyptSourceKey
    source_url: str
    published_on: date
    observed_at: datetime
    authority: Literal["official", "company", "market_data", "secondary", "manual"]
    payload: dict
    subjects: tuple[str, ...] = ()
    content_hash: str | None = None


@dataclass(frozen=True)
class StoredEvidence:
    record_id: int
    reference: EvidenceRef
    payload: dict


def canonical_payload(payload: dict) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def payload_hash(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("knowledge timestamps must include a timezone")
    return value.astimezone(timezone.utc)


class EvidenceStore:
    """Small auditable store; every read can be limited to what was known then."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY,
                    source_key TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    published_on TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    authority TEXT NOT NULL,
                    subjects_json TEXT NOT NULL DEFAULT '[]',
                    content_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(source_key, source_url, published_on, content_hash)
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(evidence)").fetchall()
            }
            if "subjects_json" not in columns:
                connection.execute(
                    "ALTER TABLE evidence "
                    "ADD COLUMN subjects_json TEXT NOT NULL DEFAULT '[]'"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evidence_known_at
                ON evidence(observed_at, source_key)
                """
            )

    def add(self, document: EvidenceDocument) -> StoredEvidence:
        payload_json = canonical_payload(document.payload)
        calculated_hash = payload_hash(payload_json)
        if document.content_hash and document.content_hash != calculated_hash:
            raise ValueError("Evidence content hash does not match its payload")

        reference = EvidenceRef(
            source=document.source_key.value,
            url=document.source_url,
            published_on=document.published_on,
            observed_at=document.observed_at,
            authority=document.authority,
            subjects=document.subjects,
            content_hash=calculated_hash,
        )
        issues = validate_evidence_source(reference, expected_source=document.source_key)
        if issues:
            raise ValueError("; ".join(issues))

        values = (
            document.source_key.value,
            reference.url,
            reference.published_on.isoformat(),
            reference.observed_at.isoformat(),
            reference.authority,
            json.dumps(reference.subjects),
            calculated_hash,
            payload_json,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO evidence (
                    source_key, source_url, published_on, observed_at,
                    authority, subjects_json, content_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                """
                SELECT id, source_key, source_url, published_on, observed_at,
                       authority, subjects_json, content_hash, payload_json
                FROM evidence
                WHERE source_key = ? AND source_url = ? AND published_on = ?
                      AND content_hash = ?
                """,
                (
                    document.source_key.value,
                    document.source_url,
                    document.published_on.isoformat(),
                    calculated_hash,
                ),
            ).fetchone()

        if row is None:
            raise RuntimeError("Evidence write did not produce a readable record")
        return self._from_row(row)

    def require(self, reference: EvidenceRef) -> StoredEvidence:
        """Resolve an exact reference and re-check its stored payload integrity."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, source_key, source_url, published_on, observed_at,
                       authority, subjects_json, content_hash, payload_json
                FROM evidence
                WHERE source_key = ? AND source_url = ? AND published_on = ?
                      AND content_hash = ?
                ORDER BY observed_at, id
                """,
                (
                    reference.source,
                    reference.url,
                    reference.published_on.isoformat(),
                    reference.content_hash,
                ),
            ).fetchall()

        for row in rows:
            stored = self._from_row(row)
            if stored.reference == reference:
                return stored
        raise ValueError(
            f"Evidence reference is not present in the verified store: "
            f"{reference.source} {reference.content_hash or '<no hash>'}"
        )

    def known_at(
        self,
        knowledge_time: datetime,
        source_key: EgyptSourceKey | None = None,
    ) -> tuple[StoredEvidence, ...]:
        query = """
            SELECT id, source_key, source_url, published_on, observed_at,
                   authority, subjects_json, content_hash, payload_json
            FROM evidence
            WHERE observed_at <= ?
        """
        parameters: list[str] = [_as_utc(knowledge_time).isoformat()]
        if source_key is not None:
            query += " AND source_key = ?"
            parameters.append(source_key.value)
        query += " ORDER BY published_on, observed_at, id"

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._from_row(row) for row in rows)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> StoredEvidence:
        payload = json.loads(row["payload_json"])
        calculated_hash = payload_hash(canonical_payload(payload))
        if calculated_hash != row["content_hash"]:
            raise ValueError(f"Stored evidence {row['id']} failed its integrity check")

        reference = EvidenceRef(
            source=row["source_key"],
            url=row["source_url"],
            published_on=date.fromisoformat(row["published_on"]),
            observed_at=datetime.fromisoformat(row["observed_at"]),
            authority=row["authority"],
            subjects=tuple(json.loads(row["subjects_json"])),
            content_hash=row["content_hash"],
        )
        return StoredEvidence(
            record_id=row["id"],
            reference=reference,
            payload=payload,
        )
