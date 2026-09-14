"""Controlled ingestion for Egyptian official and market-data sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tradingagents.markets.egypt_sources import (
    EGYPT_SOURCE_REGISTRY,
    EgyptSourceKey,
    validate_evidence_source,
)
from tradingagents.portfolio.evidence_store import (
    EvidenceDocument,
    EvidenceStore,
    StoredEvidence,
)
from tradingagents.portfolio.models import EvidenceRef


class EvidenceBundle(BaseModel):
    """Strict offline-import format used when an official site is unavailable."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    source_key: EgyptSourceKey
    source_url: str
    published_on: date
    observed_at: datetime
    authority: Literal["official", "company", "market_data", "secondary", "manual"]
    payload: dict
    content_hash: str | None = None


@dataclass(frozen=True)
class FetchPolicy:
    connect_timeout_seconds: float = 5
    read_timeout_seconds: float = 20
    maximum_bytes: int = 10_000_000
    retries: int = 3
    backoff_factor: float = 0.5


DEFAULT_FETCH_POLICY = FetchPolicy()


def _new_session(policy: FetchPolicy) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=policy.retries,
        connect=policy.retries,
        read=policy.retries,
        status=policy.retries,
        backoff_factor=policy.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class EgyptSourceClient:
    """HTTPS JSON fetcher with allowlists, limits, retries and no redirects."""

    def __init__(
        self,
        session: requests.Session | None = None,
        policy: FetchPolicy = DEFAULT_FETCH_POLICY,
    ):
        self.policy = policy
        self.session = session or _new_session(policy)

    def fetch_json(
        self,
        *,
        source_key: EgyptSourceKey,
        url: str,
        published_on: date,
        store: EvidenceStore,
        observed_at: datetime | None = None,
    ) -> StoredEvidence:
        observed = observed_at or datetime.now(timezone.utc)
        spec = EGYPT_SOURCE_REGISTRY[source_key]
        preflight = EvidenceRef(
            source=source_key.value,
            url=url,
            published_on=published_on,
            observed_at=observed,
            authority=spec.authority,
        )
        issues = validate_evidence_source(preflight, expected_source=source_key)
        if issues:
            raise ValueError("; ".join(issues))

        response = self.session.get(
            url,
            timeout=(
                self.policy.connect_timeout_seconds,
                self.policy.read_timeout_seconds,
            ),
            allow_redirects=False,
            stream=True,
            headers={
                "Accept": "application/json",
                "User-Agent": "TradingAgents-EGX/1.0",
            },
        )
        if 300 <= response.status_code < 400:
            raise ValueError("Source redirects are disabled; validate the new URL explicitly")
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "json" not in content_type:
            raise ValueError(f"Expected JSON response, got {content_type or 'unknown type'}")

        declared_length = response.headers.get("Content-Length")
        if declared_length is not None:
            try:
                parsed_length = int(declared_length)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header") from exc
            if parsed_length < 0:
                raise ValueError("Invalid Content-Length header")
            if parsed_length > self.policy.maximum_bytes:
                raise ValueError("Source response exceeds the configured size limit")

        body = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            body.extend(chunk)
            if len(body) > self.policy.maximum_bytes:
                raise ValueError("Source response exceeds the configured size limit")

        try:
            payload = json.loads(body.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Source returned invalid UTF-8 JSON") from exc
        if isinstance(payload, list):
            payload = {"records": payload}
        if not isinstance(payload, dict):
            raise ValueError("Source JSON root must be an object or list")

        return store.add(
            EvidenceDocument(
                source_key=source_key,
                source_url=url,
                published_on=published_on,
                observed_at=observed,
                authority=spec.authority,
                payload=payload,
            )
        )


def import_evidence_bundle(
    bundle_path: str | Path,
    store: EvidenceStore,
    *,
    expected_source: EgyptSourceKey | None = None,
    maximum_bytes: int = 10_000_000,
) -> StoredEvidence:
    """Import a strict JSON bundle while preserving its source and content hash."""
    path = Path(bundle_path)
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError("Evidence bundle exceeds the configured size limit")

    bundle = EvidenceBundle.model_validate_json(path.read_text(encoding="utf-8-sig"))
    if expected_source is not None and bundle.source_key is not expected_source:
        raise ValueError(
            f"Expected source {expected_source.value}, got {bundle.source_key.value}"
        )
    return store.add(
        EvidenceDocument(
            source_key=bundle.source_key,
            source_url=bundle.source_url,
            published_on=bundle.published_on,
            observed_at=bundle.observed_at,
            authority=bundle.authority,
            payload=bundle.payload,
            content_hash=bundle.content_hash,
        )
    )
