import sqlite3
from datetime import date, datetime, timezone

import pytest

from tradingagents.markets.egypt_sources import EgyptSourceKey
from tradingagents.portfolio.evidence_store import (
    EvidenceDocument,
    EvidenceStore,
    canonical_payload,
    payload_hash,
)


def document(**overrides):
    values = {
        "source_key": EgyptSourceKey.EGX_DISCLOSURES,
        "source_url": "https://beta.egx.com.eg/ar/media-center",
        "published_on": date(2026, 9, 1),
        "observed_at": datetime(2026, 9, 2, 9, tzinfo=timezone.utc),
        "authority": "official",
        "payload": {"ticker": "COMI.CA", "headline": "اختبار"},
        "subjects": ("COMI",),
    }
    values.update(overrides)
    return EvidenceDocument(**values)


def test_add_is_idempotent_and_preserves_arabic(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    first = store.add(document())
    second = store.add(document())

    assert first.record_id == second.record_id
    assert first.reference.content_hash == second.reference.content_hash
    assert first.payload["headline"] == "اختبار"
    assert first.reference.subjects == ("COMI.CA",)


def test_point_in_time_read_excludes_evidence_not_yet_observed(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    store.add(document())

    before = store.known_at(datetime(2026, 9, 2, 8, 59, tzinfo=timezone.utc))
    after = store.known_at(datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc))

    assert before == ()
    assert len(after) == 1


def test_tampered_supplied_hash_is_rejected(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    with pytest.raises(ValueError, match="does not match"):
        store.add(document(content_hash="0" * 64))


def test_local_import_with_matching_hash_is_accepted(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    payload = {"rate": 27.25}
    digest = payload_hash(canonical_payload(payload))
    stored = store.add(
        document(
            source_key=EgyptSourceKey.CBE_MACRO,
            source_url="file:///imports/cbe-rate.json",
            payload=payload,
            content_hash=digest,
        )
    )
    assert stored.reference.content_hash == digest


def test_spoofed_host_is_rejected_before_storage(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    with pytest.raises(ValueError, match="not allowed"):
        store.add(document(source_url="https://egx.com.eg.attacker.example/data"))



def test_read_detects_database_tampering(tmp_path):
    database = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(database)
    store.add(document())

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence SET payload_json = ?",
            ('{"ticker":"TAMPERED.CA"}',),
        )

    with pytest.raises(ValueError, match="integrity check"):
        store.known_at(datetime(2026, 9, 3, tzinfo=timezone.utc))



def test_read_detects_subject_metadata_tampering(tmp_path):
    database = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(database)
    store.add(document())

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence SET subjects_json = ?",
            ('["SWDY.CA"]',),
        )

    with pytest.raises(ValueError, match="metadata failed"):
        store.known_at(datetime(2026, 9, 3, tzinfo=timezone.utc))


def test_reimport_backfills_subjects_in_legacy_database(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    payload_json = canonical_payload(document().payload)
    digest = payload_hash(payload_json)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE evidence (
                id INTEGER PRIMARY KEY,
                source_key TEXT NOT NULL,
                source_url TEXT NOT NULL,
                published_on TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                authority TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(source_key, source_url, published_on, content_hash)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO evidence (
                source_key, source_url, published_on, observed_at,
                authority, content_hash, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "egx_disclosures",
                "https://beta.egx.com.eg/ar/media-center",
                "2026-09-01",
                "2026-09-02T09:00:00+00:00",
                "official",
                digest,
                payload_json,
            ),
        )

    store = EvidenceStore(database)
    stored = store.add(document())

    assert stored.record_id == 1
    assert stored.reference.subjects == ("COMI.CA",)



def test_reimport_preserves_earliest_knowledge_time(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    first = store.add(document())
    later = store.add(
        document(observed_at=datetime(2026, 9, 3, 9, tzinfo=timezone.utc))
    )

    assert later.record_id == first.record_id
    assert later.reference.observed_at == datetime(
        2026,
        9,
        2,
        9,
        tzinfo=timezone.utc,
    )
