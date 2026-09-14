"""Allowlisted Egyptian-market data sources and provenance validation.

The registry validates identity, transport and authority.  Adapters can change
without weakening these rules, and local imports remain traceable by hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Iterable, Protocol
from urllib.parse import urlparse


class EgyptSourceKey(str, Enum):
    EGX_PRICES = "egx_prices"
    EGX_DISCLOSURES = "egx_disclosures"
    EGX_FINANCIAL_STATEMENTS = "egx_financial_statements"
    EGX_SHARIA_CONSTITUENTS = "egx_sharia_constituents"
    CBE_MACRO = "cbe_macro"
    CAPMAS_INFLATION = "capmas_inflation"
    FRA_RULES = "fra_rules"
    INDEPENDENT_SHARIA_REVIEW = "independent_sharia_review"
    YAHOO_PRICES = "yahoo_prices"


class EvidenceLike(Protocol):
    source: str
    url: str
    published_on: date
    observed_at: datetime
    authority: str
    content_hash: str | None


@dataclass(frozen=True)
class EgyptSourceSpec:
    key: EgyptSourceKey
    label: str
    authority: str
    allowed_hosts: tuple[str, ...]
    required: bool = True
    allow_file_import: bool = True
    max_age_days: int = 120


EGYPT_SOURCE_REGISTRY: dict[EgyptSourceKey, EgyptSourceSpec] = {
    EgyptSourceKey.EGX_PRICES: EgyptSourceSpec(
        EgyptSourceKey.EGX_PRICES,
        "Egyptian Exchange prices",
        "official",
        ("egx.com.eg",),
        max_age_days=7,
    ),
    EgyptSourceKey.EGX_DISCLOSURES: EgyptSourceSpec(
        EgyptSourceKey.EGX_DISCLOSURES,
        "Egyptian Exchange disclosures",
        "official",
        ("egx.com.eg",),
        max_age_days=30,
    ),
    EgyptSourceKey.EGX_FINANCIAL_STATEMENTS: EgyptSourceSpec(
        EgyptSourceKey.EGX_FINANCIAL_STATEMENTS,
        "Egyptian Exchange financial statements",
        "official",
        ("egx.com.eg",),
        max_age_days=220,
    ),
    EgyptSourceKey.EGX_SHARIA_CONSTITUENTS: EgyptSourceSpec(
        EgyptSourceKey.EGX_SHARIA_CONSTITUENTS,
        "EGX Sharia index constituents",
        "official",
        ("egx.com.eg", "fra.gov.eg"),
        max_age_days=220,
    ),
    EgyptSourceKey.CBE_MACRO: EgyptSourceSpec(
        EgyptSourceKey.CBE_MACRO,
        "Central Bank of Egypt",
        "official",
        ("cbe.org.eg",),
        max_age_days=90,
    ),
    EgyptSourceKey.CAPMAS_INFLATION: EgyptSourceSpec(
        EgyptSourceKey.CAPMAS_INFLATION,
        "CAPMAS inflation",
        "official",
        ("capmas.gov.eg",),
        max_age_days=90,
    ),
    EgyptSourceKey.FRA_RULES: EgyptSourceSpec(
        EgyptSourceKey.FRA_RULES,
        "Financial Regulatory Authority",
        "official",
        ("fra.gov.eg",),
        max_age_days=365,
    ),
    EgyptSourceKey.INDEPENDENT_SHARIA_REVIEW: EgyptSourceSpec(
        EgyptSourceKey.INDEPENDENT_SHARIA_REVIEW,
        "Independently reviewed Sharia document",
        "secondary",
        (),
        required=False,
        allow_file_import=True,
        max_age_days=220,
    ),
    EgyptSourceKey.YAHOO_PRICES: EgyptSourceSpec(
        EgyptSourceKey.YAHOO_PRICES,
        "Yahoo Finance EGX price fallback",
        "market_data",
        ("finance.yahoo.com", "query1.finance.yahoo.com", "query2.finance.yahoo.com"),
        required=False,
        allow_file_import=False,
        max_age_days=7,
    ),
}


def _host_matches(host: str, allowed_host: str) -> bool:
    """Match a host or its subdomain without accepting suffix-spoofed domains."""
    return host == allowed_host or host.endswith(f".{allowed_host}")


def validate_evidence_source(
    evidence: EvidenceLike,
    *,
    expected_source: EgyptSourceKey | None = None,
) -> tuple[str, ...]:
    """Return provenance violations for one evidence reference."""
    issues: list[str] = []
    try:
        source_key = EgyptSourceKey(evidence.source)
    except ValueError:
        return (f"Unknown Egyptian source key: {evidence.source}",)

    if expected_source is not None and source_key is not expected_source:
        issues.append(
            f"Expected source {expected_source.value}, got {source_key.value}"
        )

    spec = EGYPT_SOURCE_REGISTRY[source_key]
    if evidence.authority != spec.authority:
        issues.append(
            f"{source_key.value} requires authority {spec.authority}, "
            f"got {evidence.authority}"
        )

    parsed = urlparse(evidence.url)
    if parsed.scheme == "file":
        if not spec.allow_file_import:
            issues.append(f"{source_key.value} does not allow local file imports")
        if not evidence.content_hash:
            issues.append("Local evidence requires a content hash")
        return tuple(issues)

    if parsed.scheme != "https":
        issues.append("Remote evidence must use HTTPS")
        return tuple(issues)

    host = (parsed.hostname or "").lower().rstrip(".")
    if not any(_host_matches(host, allowed) for allowed in spec.allowed_hosts):
        issues.append(f"Host {host or '<missing>'} is not allowed for {source_key.value}")
    if parsed.username or parsed.password:
        issues.append("Evidence URLs must not contain credentials")
    return tuple(issues)


def validate_evidence_for_date(
    evidence: EvidenceLike,
    analysis_date: date,
    *,
    expected_source: EgyptSourceKey | None = None,
    maximum_age_days: int | None = None,
) -> tuple[str, ...]:
    """Validate provenance, integrity and point-in-time freshness together."""
    issues = list(
        validate_evidence_source(evidence, expected_source=expected_source)
    )
    if not evidence.content_hash:
        issues.append("Evidence content hash is required")

    try:
        source_key = EgyptSourceKey(evidence.source)
    except ValueError:
        return tuple(dict.fromkeys(issues))

    if evidence.published_on > analysis_date:
        issues.append("Publication is from the future")
    if evidence.observed_at.date() > analysis_date:
        issues.append("Evidence was not known on analysis date")

    age_days = (analysis_date - evidence.published_on).days
    source_maximum = EGYPT_SOURCE_REGISTRY[source_key].max_age_days
    effective_maximum = (
        source_maximum
        if maximum_age_days is None
        else min(source_maximum, maximum_age_days)
    )
    if age_days > effective_maximum:
        issues.append(
            f"Evidence is stale ({age_days} days, maximum {effective_maximum})"
        )
    return tuple(dict.fromkeys(issues))


def missing_required_sources(source_keys: set[str]) -> tuple[str, ...]:
    """Return required source keys absent from a completed ingestion run."""
    required = {
        key.value for key, spec in EGYPT_SOURCE_REGISTRY.items() if spec.required
    }
    return tuple(sorted(required - source_keys))


def evaluate_source_coverage(
    evidence_refs: Iterable[EvidenceLike],
    analysis_date: date,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return fresh valid source keys and explain rejected evidence."""
    valid: set[str] = set()
    issues: list[str] = []

    for evidence in evidence_refs:
        evidence_issues = validate_evidence_for_date(evidence, analysis_date)
        if evidence_issues:
            issues.extend(f"{evidence.source}: {issue}" for issue in evidence_issues)
            continue

        valid.add(EgyptSourceKey(evidence.source).value)

    return frozenset(valid), tuple(dict.fromkeys(issues))
