import pytest

from askhr.chunker import chunk_document, is_heading
from askhr.models import Document, Page

# Made-up text shaped like a Viamonde directive (we never copy their text into the repo).
PAGE_1 = """PRÉAMBULE
Le Conseil fictif reconnaît l'importance des règles de test.
MODALITÉS
1. Première règle
Le texte de la première règle."""

PAGE_2 = """2. Deuxième règle
Le texte de la deuxième règle continue sur la page suivante."""


def make_doc(*page_texts: str) -> Document:
    return Document(
        doc_id="9,999",
        title="Titre de test",
        doc_type="directive",
        section="Tests",
        language="fr",
        url="https://example.org/9_999.pdf",
        pages=tuple(Page(number=n, text=text) for n, text in enumerate(page_texts, start=1)),
    )


def all_lines(doc: Document) -> list[str]:
    return [line for page in doc.pages for line in page.text.splitlines()]


@pytest.mark.parametrize(
    "line",
    [
        "PRÉAMBULE",
        "ÉNONCÉ DE LA POLITIQUE",
        "2. Divulgation",
        "2.3. Confidentialité",
        "1.1.2. Harcèlement basé sur l'orientation",
        "2.1.1. La plainte peut être déposée :",
    ],
)
def test_is_heading_accepts_headings(line: str) -> None:
    assert is_heading(line)


@pytest.mark.parametrize(
    "line",
    [
        "Le Conseil reconnaît l'importance des règles.",
        "1. Accessibilité : L'ensemble des membres",
        "1 2 3 4 s.o.",
        "18 juillet 2024",
        "a) Tout commentaire dégradant",
        "",
        "N.B.",
        "X" * 120,
    ],
)
def test_is_heading_rejects_other_lines(line: str) -> None:
    assert not is_heading(line)


def test_small_sections_are_merged_and_keep_the_first_heading() -> None:
    chunks = chunk_document(make_doc(PAGE_1, PAGE_2))

    # "MODALITÉS" starts its own chunk (a heading belongs to the text after it),
    # and the small numbered sections that follow are merged into it, across the page break.
    assert [chunk.heading for chunk in chunks] == ["PRÉAMBULE", "MODALITÉS"]
    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(1, 1), (1, 2)]
    assert "2. Deuxième règle" in chunks[1].text


def test_sections_stay_apart_when_they_do_not_fit() -> None:
    chunks = chunk_document(make_doc(PAGE_1, PAGE_2), max_chars=120)

    assert [chunk.heading for chunk in chunks] == ["PRÉAMBULE", "MODALITÉS", "2. Deuxième règle"]
    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(1, 1), (1, 1), (2, 2)]


def test_stacked_headings_stay_with_the_text_that_follows() -> None:
    doc = make_doc("PRÉAMBULE\nUn texte d'introduction.\nMODALITÉS\nRÔLES\n1. Règle\nLe texte.")

    chunks = chunk_document(doc)

    assert [chunk.heading for chunk in chunks] == ["PRÉAMBULE", "MODALITÉS"]
    assert chunks[1].text == "MODALITÉS\nRÔLES\n1. Règle\nLe texte."


def test_a_long_section_is_split_at_line_breaks() -> None:
    body = "\n".join(f"Phrase numéro {n} de la règle." for n in range(30))
    doc = make_doc("3. Longue règle\n" + body)

    chunks = chunk_document(doc, max_chars=100)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(chunk.heading == "3. Longue règle" for chunk in chunks)


def test_a_single_line_longer_than_the_budget_stays_whole() -> None:
    long_line = "mot " * 60
    chunks = chunk_document(make_doc(long_line.strip()), max_chars=100)

    assert [chunk.text for chunk in chunks] == [long_line.strip()]


def test_nothing_is_lost() -> None:
    doc = make_doc(PAGE_1, PAGE_2, "", "3. Après une page vide\nFin.")

    for max_chars in (60, 120, 1500):
        chunks = chunk_document(doc, max_chars=max_chars)
        chunk_lines = [line for chunk in chunks for line in chunk.text.splitlines()]
        assert chunk_lines == all_lines(doc)


def test_text_before_the_first_heading_has_no_heading() -> None:
    chunks = chunk_document(make_doc("Introduction sans titre.\nPRÉAMBULE\nTexte."), max_chars=30)

    assert chunks[0].heading == ""
    assert chunks[0].text == "Introduction sans titre."


def test_chunks_carry_document_metadata_and_ids() -> None:
    chunks = chunk_document(make_doc(PAGE_1, PAGE_2), max_chars=120)

    assert [chunk.chunk_id for chunk in chunks] == ["9,999#0", "9,999#1", "9,999#2"]
    assert {chunk.title for chunk in chunks} == {"Titre de test"}
    assert {chunk.url for chunk in chunks} == {"https://example.org/9_999.pdf"}


def test_an_empty_document_has_no_chunks() -> None:
    assert chunk_document(make_doc("", "")) == []
