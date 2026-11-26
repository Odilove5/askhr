from pathlib import Path

import pytest
from pypdf import PdfWriter

from askhr.connectors.downloader import load_manifest
from askhr.connectors.viamonde_site import ManifestEntry
from askhr.loader import clean_page, is_header_end, load_document

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
    ("line", "expected"),
    [
        ("Page 2 de 3", True),
        (" Page 12 sur 29 ", True),
        ("9_999_Titre de test_dir_adm", True),
        ("9,999_Titre_politique", True),  # comma instead of underscore
        ("9_999 Titre_de_test_politique", True),  # space after the number
        ("Prochaine révision en 2027-2028 Page 1 de 17", True),  # page number at the end
        ("membres syndiqués et non syndiqués_dir_adm", True),  # file name wrapped on 2 lines
        ("Voir la page 2 de ce document.", False),
        ("Un salaire de 2,500 $ par mois", False),  # a number, but no file name
        ("PRÉAMBULE", False),
    ],
)
def test_is_header_end(line: str, expected: bool) -> None:
    assert is_header_end(line) is expected


def test_clean_page_removes_header_and_tidies_lines() -> None:
    assert clean_page(FAKE_PAGE) == "1. Première règle\nLe texte de la règle."


def test_clean_page_handles_a_file_name_first_and_page_number_last() -> None:
    page = "9_999_Titre_politique\nPolitique no 9,999\nTITRE\nApprouvée le 1 juin 2020"
    page += "\nPage 1 de 1\nPRÉAMBULE"
    assert clean_page(page) == "PRÉAMBULE"


def test_clean_page_handles_a_header_without_file_name() -> None:
    page = "Directives administratives no 9,999\nTITRE\nPage 2 sur 5\n• Une règle."
    assert clean_page(page) == "• Une règle."


def test_clean_page_keeps_a_page_without_header() -> None:
    text = "Plan de rendement\n\n18 juillet 2024"
    assert clean_page(text) == "Plan de rendement\n18 juillet 2024"


def test_clean_page_ignores_the_marker_deep_in_the_body() -> None:
    body = [f"ligne {n}" for n in range(30)] + ["Page 1 de 2"]
    assert clean_page("\n".join(body)) == "\n".join(body)


def test_clean_page_of_an_empty_page_is_empty() -> None:
    assert clean_page("  \n \n") == ""


def test_clean_page_normalizes_accents_to_nfc() -> None:
    decomposed = "congé"  # "e" followed by a combining accent
    assert clean_page(decomposed) == "congé"


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
