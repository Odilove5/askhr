from pathlib import Path

import pytest

from askhr.connectors.viamonde_site import ManifestEntry, parse_index

FIXTURE = Path(__file__).parent / "fixtures" / "viamonde_index.html"
BASE_URL = (
    "https://csviamonde.ca/administration-et-gouvernance/politiques-et-directives-administratives"
)


@pytest.fixture
def entries() -> list[ManifestEntry]:
    return parse_index(FIXTURE.read_text(encoding="utf-8"), BASE_URL)


def test_one_entry_per_pdf(entries: list[ManifestEntry]) -> None:
    # 5,100: policy + directive; 5,202: policy + 4 directives; 5,300: policy only
    assert len(entries) == 8


def test_policy_entry(entries: list[ManifestEntry]) -> None:
    policy = entries[0]
    assert policy.doc_id == "5,100"
    assert policy.doc_type == "politique"
    assert policy.title == "Antécédents criminels"  # no "(144 Ko)" file size
    assert policy.section == "Milieu de travail"
    assert policy.language == "fr"


def test_directive_codes_come_from_link_text(entries: list[ManifestEntry]) -> None:
    ids = [e.doc_id for e in entries if e.doc_type == "directive"]
    assert ids == ["5,100-dir", "5,202a", "5,202b", "5,202c", "5,202d"]


def test_directive_title_without_code(entries: list[ManifestEntry]) -> None:
    directive = next(e for e in entries if e.doc_id == "5,202b")
    assert directive.title == "Gestion de la déclaration des absences"


def test_urls_are_absolute(entries: list[ManifestEntry]) -> None:
    assert all(e.url.startswith("https://csviamonde.ca/") for e in entries)
    assert all(e.url.lower().endswith(".pdf") for e in entries)


def test_empty_page_gives_no_entries() -> None:
    assert parse_index("<html><body></body></html>", BASE_URL) == []
