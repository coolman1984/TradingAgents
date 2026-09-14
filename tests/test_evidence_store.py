from datetime import date, datetime

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
        "observed_at": datetime(2026, 9, 2, 9),
        "authority": "official",
        "payload": {"ticker": "COMI.CA", "headline": "اختبار"},
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


def test_point_in_time_read_excludes_evidence_not_yet_observed(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    store.add(document())

    before = store.known_at(datetime(2026, 9, 2, 8, 59))
    after = store.known_at(datetime(2026, 9, 2, 9, 0))

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
