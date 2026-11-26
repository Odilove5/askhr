from pathlib import Path

import pytest
from pypdf import PdfWriter

from askhr.connectors.downloader import load_manifest
from askhr.connectors.viamonde_site import ManifestEntry
from askhr.loader import clean_page, header_marker, load_document

# Made-up text shaped like a real Viamonde page (we never copy their text into the repo).
FAKE_PAGE = """
Directives administratives no 9,999
 
SECTION INVENTÉE – POUR LES TESTS
 
TITRE DE TEST
Page 2 de 3
 
9_999_Titre de test_dir_adm
 
1.   Première   règle
Le texte de la règle.
"""

ENTRY = ManifestEntry(
    doc_id="9,999",
    title="Titre de test",
    doc_type="directive",
    section="Tests",
    language="fr",
    url="https://example.org/9_999.pdf",
)


@pytest.mark.parametrize(
    ("doc_id", "expected"),
    [("5,202a", "5_202a_"), ("5,203-dir", "5_203_"), ("5,400-res3", "5_400_")],
)
def test_header_marker(doc_id: str, expected: str) -> None:
    assert header_marker(doc_id) == expected


def test_clean_page_removes_header_and_tidies_lines() -> None:
    assert clean_page(FAKE_PAGE, "9,999") == "1. Première règle\nLe texte de la règle."


def test_clean_page_keeps_a_page_without_header() -> None:
    text = "Plan de rendement\n\n18 juillet 2024"
    assert clean_page(text, "9,999") == "Plan de rendement\n18 juillet 2024"


def test_clean_page_ignores_the_marker_deep_in_the_body() -> None:
    body = [f"ligne {n}" for n in range(30)] + ["9_999_ cité dans le texte"]
    assert clean_page("\n".join(body), "9,999") == "\n".join(body)


def test_clean_page_of_an_empty_page_is_empty() -> None:
    assert clean_page("  \n \n", "9,999") == ""


def test_clean_page_normalizes_accents_to_nfc() -> None:
    decomposed = "congé"  # "e" followed by a combining accent
    assert clean_page(decomposed, "9,999") == "congé"


def test_load_document_keeps_empty_pages_with_true_numbers(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    writer.write(tmp_path / "9_999.pdf")

    doc = load_document(ENTRY, tmp_path)

    assert [page.number for page in doc.pages] == [1, 2]
    assert all(page.text == "" for page in doc.pages)


def test_load_document_explains_a_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="downloader"):
        load_document(ENTRY, tmp_path)


REAL_PDF = Path("data/raw/5_202a.pdf")


@pytest.mark.skipif(not REAL_PDF.exists(), reason="run the downloader first")
def test_real_viamonde_pdf_loses_its_header() -> None:
    entry = next(e for e in load_manifest(Path("corpus/manifest.yaml")) if e.doc_id == "5,202a")

    doc = load_document(entry, REAL_PDF.parent)

    assert len(doc.pages) == 2
    assert doc.pages[1].text.startswith("2. Divulgation")
    assert "Page 1 de 2" not in doc.text  # header lines are gone
    assert "_dir_adm" not in doc.text  # and so is the file-name line
