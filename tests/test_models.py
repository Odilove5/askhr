from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from askhr.connectors.viamonde_site import ManifestEntry
from askhr.models import Chunk, Document, Page

ENTRY = ManifestEntry(
    doc_id="5,202a",
    title="Gestion de la confidentialité des dossiers médicaux",
    doc_type="directive",
    section="Milieu de travail",
    language="fr",
    url="https://csviamonde.ca/fileadmin/5_202a.pdf",
)
PAGES = (Page(1, "Première page."), Page(2, "Deuxième page."))


def make_chunk(**overrides: Any) -> Chunk:
    """A valid chunk; pass keyword arguments to change any field."""
    fields: dict[str, Any] = {
        "doc_id": "5,202a",
        "index": 3,
        "text": "4.2 La demande doit être présentée 60 jours avant le congé.",
        "page_start": 2,
        "page_end": 2,
        "title": ENTRY.title,
        "doc_type": "directive",
        "language": "fr",
        "url": ENTRY.url,
    }
    fields.update(overrides)
    return Chunk(**fields)


def test_from_entry_copies_the_manifest_fields() -> None:
    doc = Document.from_entry(ENTRY, PAGES)
    assert (doc.doc_id, doc.title, doc.doc_type, doc.section, doc.language, doc.url) == (
        ENTRY.doc_id,
        ENTRY.title,
        ENTRY.doc_type,
        ENTRY.section,
        ENTRY.language,
        ENTRY.url,
    )
    assert doc.pages == PAGES


def test_document_text_joins_pages_in_order() -> None:
    assert Document.from_entry(ENTRY, PAGES).text == "Première page.\n\nDeuxième page."


def test_content_hash_changes_only_when_text_changes() -> None:
    doc = Document.from_entry(ENTRY, PAGES)
    same = Document.from_entry(ENTRY, (Page(1, "Première page."), Page(2, "Deuxième page.")))
    edited = Document.from_entry(ENTRY, (Page(1, "Première page!"), Page(2, "Deuxième page.")))
    assert doc.content_hash == same.content_hash
    assert doc.content_hash != edited.content_hash


def test_unknown_doc_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="doc_type"):
        Document.from_entry(replace(ENTRY, doc_type="politque"), PAGES)


def test_page_numbers_start_at_one() -> None:
    with pytest.raises(ValueError):
        Page(0, "texte")


def test_chunk_id_format() -> None:
    assert make_chunk().chunk_id == "5,202a#3"


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": "   "},
        {"index": -1},
        {"page_start": 0, "page_end": 0},
        {"page_start": 3, "page_end": 2},
        {"language": "es"},
    ],
)
def test_invalid_chunks_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make_chunk(**overrides)


def test_replace_adds_context_without_changing_the_original() -> None:
    chunk = make_chunk()
    enriched = replace(chunk, context="Directive 5,202a sur les dossiers médicaux du personnel.")
    assert enriched.context.startswith("Directive 5,202a")
    assert chunk.context == ""
    assert enriched.chunk_id == chunk.chunk_id


def test_chunks_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        make_chunk().text = "autre"  # type: ignore[misc]
