import json
from datetime import date, datetime, timezone

import pytest

from tradingagents.dataflows.egypt_ingestion import (
    EgyptSourceClient,
    FetchPolicy,
    import_evidence_bundle,
)
from tradingagents.markets.egypt_sources import EgyptSourceKey
from tradingagents.portfolio.evidence_store import EvidenceStore


class FakeResponse:
    def __init__(self, body=b'{"value": 42}', status_code=200, content_type="application/json"):
        self.body = body
        self.status_code = status_code
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_fetcher_rejects_spoofed_host_before_network(tmp_path):
    session = FakeSession(FakeResponse())
    client = EgyptSourceClient(session=session)
    store = EvidenceStore(tmp_path / "evidence.sqlite3")

    with pytest.raises(ValueError, match="not allowed"):
        client.fetch_json(
            source_key=EgyptSourceKey.EGX_DISCLOSURES,
            url="https://egx.com.eg.attacker.example/data",
            published_on=date(2026, 9, 1),
            observed_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
            store=store,
        )
    assert session.calls == []


def test_fetcher_disables_redirects_and_stores_json(tmp_path):
    session = FakeSession(FakeResponse())
    client = EgyptSourceClient(session=session)
    store = EvidenceStore(tmp_path / "evidence.sqlite3")

    stored = client.fetch_json(
        source_key=EgyptSourceKey.CBE_MACRO,
        url="https://www.cbe.org.eg/api/rates",
        published_on=date(2026, 9, 1),
        observed_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        store=store,
    )

    assert stored.payload == {"value": 42}
    assert session.calls[0][1]["allow_redirects"] is False
    assert stored.reference.content_hash


def test_fetcher_rejects_redirect_and_wrong_content_type(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    redirect = EgyptSourceClient(session=FakeSession(FakeResponse(status_code=302)))
    with pytest.raises(ValueError, match="redirects are disabled"):
        redirect.fetch_json(
            source_key=EgyptSourceKey.FRA_RULES,
            url="https://fra.gov.eg/api/rules",
            published_on=date(2026, 9, 1),
            observed_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
            store=store,
        )

    html = EgyptSourceClient(
        session=FakeSession(FakeResponse(content_type="text/html"))
    )
    with pytest.raises(ValueError, match="Expected JSON"):
        html.fetch_json(
            source_key=EgyptSourceKey.CAPMAS_INFLATION,
            url="https://capmas.gov.eg/api/inflation",
            published_on=date(2026, 9, 1),
            observed_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
            store=store,
        )


def test_fetcher_enforces_streamed_size_limit(tmp_path):
    response = FakeResponse(body=b'{"large":"0123456789"}')
    response.headers.pop("Content-Length")
    client = EgyptSourceClient(
        session=FakeSession(response),
        policy=FetchPolicy(maximum_bytes=10),
    )

    with pytest.raises(ValueError, match="size limit"):
        client.fetch_json(
            source_key=EgyptSourceKey.EGX_PRICES,
            url="https://www.egx.com.eg/api/prices",
            published_on=date(2026, 9, 1),
            observed_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
            store=EvidenceStore(tmp_path / "evidence.sqlite3"),
        )


def test_offline_bundle_is_strict_and_importable(tmp_path):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_key": "egx_sharia_constituents",
                "source_url": "https://www.egx.com.eg/sharia",
                "published_on": "2026-09-01",
                "observed_at": "2026-09-02T09:00:00+00:00",
                "authority": "official",
                "payload": {"tickers": ["EFID.CA", "SWDY.CA"]},
            }
        ),
        encoding="utf-8",
    )
    store = EvidenceStore(tmp_path / "evidence.sqlite3")

    stored = import_evidence_bundle(
        bundle_path,
        store,
        expected_source=EgyptSourceKey.EGX_SHARIA_CONSTITUENTS,
    )
    assert stored.payload["tickers"] == ["EFID.CA", "SWDY.CA"]


def test_offline_bundle_rejects_unknown_fields(tmp_path):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_key": "cbe_macro",
                "source_url": "https://www.cbe.org.eg/rates",
                "published_on": "2026-09-01",
                "observed_at": "2026-09-02T09:00:00+00:00",
                "authority": "official",
                "payload": {},
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        import_evidence_bundle(
            bundle_path,
            EvidenceStore(tmp_path / "evidence.sqlite3"),
        )
