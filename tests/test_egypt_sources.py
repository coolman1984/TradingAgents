from datetime import date, datetime, timezone

from tradingagents.markets.egypt_sources import (
    EgyptSourceKey,
    missing_required_sources,
    validate_evidence_source,
)
from tradingagents.portfolio.models import EvidenceRef


def evidence(source: str, url: str, authority: str = "official", content_hash=None):
    return EvidenceRef(
        source=source,
        url=url,
        published_on=date(2026, 9, 1),
        observed_at=datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
        authority=authority,
        content_hash=content_hash,
    )


def test_official_egx_subdomain_is_allowed():
    item = evidence(
        EgyptSourceKey.EGX_DISCLOSURES.value,
        "https://beta.egx.com.eg/ar/media-center",
    )
    assert validate_evidence_source(item) == ()


def test_suffix_spoofed_domain_is_rejected():
    item = evidence(
        EgyptSourceKey.EGX_DISCLOSURES.value,
        "https://egx.com.eg.attacker.example/disclosures",
    )
    issues = validate_evidence_source(item)
    assert any("not allowed" in issue for issue in issues)


def test_credentials_in_url_are_rejected():
    item = evidence(
        EgyptSourceKey.CBE_MACRO.value,
        "https://user:secret@cbe.org.eg/rates",
    )
    assert "Evidence URLs must not contain credentials" in validate_evidence_source(item)


def test_local_official_copy_requires_hash():
    item = evidence(EgyptSourceKey.FRA_RULES.value, "file:///fra/rules.pdf")
    assert "Local evidence requires a content hash" in validate_evidence_source(item)


def test_unknown_source_is_rejected():
    item = evidence("social_media_tip", "https://egx.com.eg/")
    assert validate_evidence_source(item) == (
        "Unknown Egyptian source key: social_media_tip",
    )


def test_optional_fallback_is_not_required():
    missing = missing_required_sources(set())
    assert EgyptSourceKey.YAHOO_PRICES.value not in missing
    assert EgyptSourceKey.EGX_SHARIA_CONSTITUENTS.value in missing
