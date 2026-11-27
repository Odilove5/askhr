from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from askhr.chunker import chunk_document
from askhr.context import (
    CONTEXT_PROMPT,
    PAGE_WORD,
    SYSTEM,
    NoteCutError,
    PromptTooLongError,
    add_context,
    build_context_chain,
    chunk_inputs,
    clean_note,
    estimate_tokens,
    generate_note,
    outline_and_neighbours,
    page_span,
    paged_text,
    prompt_tokens,
)
from askhr.models import Chunk, Document, Page

# Made-up text shaped like a Viamonde directive (we never copy their text into the repo).
PAGE_1 = """PRÉAMBULE
Le Conseil fictif reconnaît l'importance des règles de test, et ce texte est assez long
pour former une section à lui seul, car il dépasse le minimum de caractères demandé par
le découpage. Il parle de congés fictifs, de demandes fictives et de délais fictifs.
Chaque membre du personnel fictif doit lire cette directive fictive avec attention.
1. Demande de congé
La demande est envoyée à la direction fictive au moins dix jours ouvrables avant le
début du congé. La direction répond par écrit dans un délai de cinq jours ouvrables.
Une demande incomplète est retournée à son auteur avec la liste des pièces manquantes."""

PAGE_2 = """2. Retour au travail
Au retour, la personne avise son superviseur fictif et remet le formulaire fictif.
Le service fictif des ressources humaines met le dossier à jour sans tarder, et il
confirme la date de retour par courriel à la personne et à son superviseur fictif.
Toute question sur ce retour est adressée au bureau fictif des ressources humaines."""

BIG = 100_000  # a limit large enough for any test document


def make_doc(*page_texts: str) -> Document:
    return Document(
        doc_id="9,999",
        title="Congés fictifs",
        doc_type="directive",
        section="Tests",
        language="fr",
        url="https://example.org/9_999.pdf",
        pages=tuple(Page(number=n, text=text) for n, text in enumerate(page_texts, start=1)),
    )


@pytest.fixture
def doc() -> Document:
    return make_doc(PAGE_1, PAGE_2)


@pytest.fixture
def chunks(doc: Document) -> list[Chunk]:
    result = chunk_document(doc, max_chars=600)
    assert len(result) >= 3  # the tests below need several chunks
    return result


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcd") == 2


def test_chunk_inputs_uses_the_whole_document_when_it_fits(
    doc: Document, chunks: list[Chunk]
) -> None:
    inputs = chunk_inputs(doc, chunks, 1, BIG)
    assert inputs == {
        "title": "Congés fictifs",
        "document": paged_text(doc),
        "heading": chunks[1].heading,
        "pages": page_span(chunks[1]),
        "chunk": chunks[1].text,
    }


def test_the_document_is_sent_with_page_markers(doc: Document) -> None:
    text = paged_text(doc)
    assert text.startswith("[page 1]\nPRÉAMBULE")
    assert "\n\n[page 2]\n2. Retour au travail" in text


def test_page_span_is_one_number_or_a_range(chunks: list[Chunk]) -> None:
    one_page = next(chunk for chunk in chunks if chunk.page_start == chunk.page_end)
    assert page_span(one_page) == str(one_page.page_start)
    spanning = Chunk(**{**one_page.__dict__, "page_end": one_page.page_start + 1})
    assert page_span(spanning) == f"{one_page.page_start}-{one_page.page_start + 1}"


def test_chunk_without_heading_gets_a_placeholder(doc: Document) -> None:
    chunks = chunk_document(make_doc("Un texte sans aucun titre au-dessus."))
    assert chunks[0].heading == ""
    assert chunk_inputs(doc, chunks, 0, BIG)["heading"] == "(sans titre)"


def test_prompt_puts_the_document_first_and_the_chunk_last(
    doc: Document, chunks: list[Chunk]
) -> None:
    messages = CONTEXT_PROMPT.format_messages(**chunk_inputs(doc, chunks, 1, BIG))
    human = str(messages[-1].content)
    assert human.index(paged_text(doc)) < human.index("<extrait")
    assert f'pages="{page_span(chunks[1])}"' in human
    assert human.rstrip().endswith("Note de contexte :")


def test_braces_in_the_document_do_not_break_the_prompt() -> None:
    doc = make_doc("PRÉAMBULE\nUn champ {nom} et une accolade } seule.")
    chunks = chunk_document(doc)
    messages = CONTEXT_PROMPT.format_messages(**chunk_inputs(doc, chunks, 0, BIG))
    assert "{nom}" in str(messages[-1].content)


def test_long_document_falls_back_to_outline_and_neighbours(
    doc: Document, chunks: list[Chunk]
) -> None:
    whole = prompt_tokens(chunk_inputs(doc, chunks, 1, BIG))
    inputs = chunk_inputs(doc, chunks, 1, whole - 1)  # one token too small for the whole text
    assert inputs["document"] == outline_and_neighbours(chunks, 1)
    assert prompt_tokens(inputs) <= whole - 1
    assert chunks[0].text in inputs["document"]  # the chunk before
    assert chunks[2].text in inputs["document"]  # the chunk after
    assert inputs["chunk"] == chunks[1].text  # the chunk itself is always sent


def test_outline_lists_each_heading_once_and_stays_inside_the_document(
    chunks: list[Chunk],
) -> None:
    outline = outline_and_neighbours(chunks, 0)
    for heading in {chunk.heading for chunk in chunks if chunk.heading}:
        assert outline.count(heading) >= 1
    assert chunks[1].text in outline
    last_outline = outline_and_neighbours(chunks, len(chunks) - 1)
    assert chunks[-2].text in last_outline  # the last chunk has only a neighbour before


def test_prompt_that_cannot_fit_raises_instead_of_being_cut(
    doc: Document, chunks: list[Chunk]
) -> None:
    with pytest.raises(PromptTooLongError, match="ASKHR_OLLAMA_NUM_CTX_LONG"):
        chunk_inputs(doc, chunks, 1, 50)


def test_chain_returns_the_model_text() -> None:
    chain = build_context_chain(FakeListChatModel(responses=["  Note.  "]))
    assert chain.invoke(
        {"title": "t", "document": "d", "heading": "h", "pages": "1", "chunk": "c"}
    ) == ("  Note.  ")


def test_add_context_stores_one_clean_line_per_chunk(doc: Document, chunks: list[Chunk]) -> None:
    replies = [f"  Note\n numéro {i}.  " for i in range(len(chunks))]
    chain = build_context_chain(FakeListChatModel(responses=replies))
    result = add_context(doc, chunks, chain, BIG)
    assert [chunk.context for chunk in result] == [f"Note numéro {i}." for i in range(len(chunks))]


def test_each_note_lands_on_its_own_chunk_even_in_parallel(
    doc: Document, chunks: list[Chunk]
) -> None:
    # A fake chain that writes a note naming its chunk, so a mix-up would show.
    echo: RunnableLambda[dict[str, str], str] = RunnableLambda(
        lambda inputs: f"note: {inputs['chunk'][:30]}"
    )
    result = add_context(doc, chunks, echo, BIG, max_concurrency=4)
    for chunk in result:
        assert chunk.context == " ".join(f"note: {chunk.text[:30]}".split())


def test_add_context_keeps_everything_else_and_leaves_originals_unchanged(
    doc: Document, chunks: list[Chunk]
) -> None:
    chain = build_context_chain(FakeListChatModel(responses=["Note."] * len(chunks)))
    result = add_context(doc, chunks, chain, BIG)
    for before, after in zip(chunks, result, strict=True):
        assert before.context == ""
        assert after == Chunk(**{**before.__dict__, "context": "Note."})
    with pytest.raises(FrozenInstanceError):
        result[0].context = "x"  # type: ignore[misc]


def test_prompt_forbids_page_numbers_in_the_note() -> None:
    assert "N'écris aucun numéro de page" in SYSTEM


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Annexe B (page 23), sur le consentement.", "Annexe B, sur le consentement."),
        ("Section 3.3.4 (pages 8-9) : délais.", "Section 3.3.4 : délais."),
        ("Annexe A (Pages 2 et 3).", "Annexe A."),
        ("Section 2 (p. 4).", "Section 2."),
        ("Annexe B (pages 25, 26 à 28) sur le Code.", "Annexe B sur le Code."),
        ("Section 5 (pages 8–9).", "Section 5."),
        (
            "Les sections (10.5 à 12, pages 16-17) sur les dossiers.",
            "Les sections (10.5 à 12) sur les dossiers.",
        ),
        ("Elle se trouve dans l'Annexe B à la page 21.", "Elle se trouve dans l'Annexe B."),
        ("Cette section à la page 4 énumère les critères.", "Cette section énumère les critères."),
        ("Le formulaire de la page 18 pour le dépôt.", "Le formulaire pour le dépôt."),
        ("Les règles aux pages 8 et 9, puis la suite.", "Les règles, puis la suite."),
        ("Note. Mots-clés:plainte, délai", "Note. Mots-clés : plainte, délai"),
        ("Note. Mots-clés :plainte", "Note. Mots-clés : plainte"),
        (
            "Les documents produits.Mots-clés : enquête",
            "Les documents produits. Mots-clés : enquête",
        ),
        ("Mots-clés : seul", "Mots-clés : seul"),
    ],
)
def test_clean_note_removes_page_references_in_brackets(raw: str, clean: str) -> None:
    assert clean_note(raw) == clean


@pytest.mark.parametrize(
    "note",
    [
        "Délai de dix (10) jours calendaires. Mots-clés : délai, plainte",
        "Article 3 (2) de la Loi (L.R.O. 1990, chap. H.19).",
        "La page web du Conseil (section 5).",
        "La page de garde du formulaire, à la page web.",
        "Les sections 10.5 à 12 (dossiers et affichage).",
    ],
)
def test_clean_note_keeps_other_brackets(note: str) -> None:
    assert clean_note(note) == note


def test_clean_note_is_stable() -> None:
    note = "Annexe D (pages 23 à 26) à la page 23. Mots-clés:Code, consentement"
    assert (
        clean_note(clean_note(note))
        == clean_note(note)
        == "Annexe D. Mots-clés : Code, consentement"
    )


def test_page_word_flags_page_mentions_left_without_brackets() -> None:
    assert PAGE_WORD.search("Elle se trouve à la page 23.")
    assert PAGE_WORD.search("voir p. 4")
    assert not PAGE_WORD.search("La page web du Conseil (section 5).")


def test_add_context_strips_page_references(doc: Document, chunks: list[Chunk]) -> None:
    chain = build_context_chain(FakeListChatModel(responses=["Annexe B (page 23)."] * len(chunks)))
    assert {chunk.context for chunk in add_context(doc, chunks, chain, BIG)} == {"Annexe B."}


def test_empty_list_never_calls_the_model(doc: Document) -> None:
    def fail(_: object) -> str:
        raise AssertionError("the model was called")

    assert add_context(doc, [], RunnableLambda(fail), BIG) == []


def test_empty_note_is_an_error(doc: Document, chunks: list[Chunk]) -> None:
    chain = build_context_chain(FakeListChatModel(responses=["   "] * len(chunks)))
    with pytest.raises(ValueError, match="empty note"):
        add_context(doc, chunks, chain, BIG)


def test_chunks_from_another_document_are_refused(chunks: list[Chunk]) -> None:
    other = make_doc(PAGE_1)
    other = Document(**{**other.__dict__, "doc_id": "1,111"})
    chain = build_context_chain(FakeListChatModel(responses=["Note."]))
    with pytest.raises(ValueError, match="does not belong"):
        add_context(other, chunks, chain, BIG)


def test_too_long_prompt_fails_before_any_model_call(doc: Document, chunks: list[Chunk]) -> None:
    def fail(_: object) -> str:
        raise AssertionError("the model was called")

    with pytest.raises(PromptTooLongError):
        add_context(doc, chunks, RunnableLambda(fail), 50)


# generate_note: one reply from the model, checked before it becomes a note (step 6c, row 1).
NOTE_INPUTS = {
    "title": "Directive fictive",
    "document": "[page 1]\nTexte fictif.",
    "heading": "Section fictive",
    "pages": "1",
    "chunk": "Extrait fictif.",
}


def fake_model(content: Any, done_reason: str | None = "stop") -> GenericFakeChatModel:
    """A fake model that sends one scripted reply; done_reason=None sends no metadata."""
    metadata = {} if done_reason is None else {"done_reason": done_reason}
    reply = AIMessage(content=content, response_metadata=metadata)
    return GenericFakeChatModel(messages=iter([reply]))


def test_generate_note_returns_the_cleaned_note() -> None:
    model = fake_model("Annexe B (page 23).  Mots-clés: congé")
    assert generate_note(model, NOTE_INPUTS) == "Annexe B. Mots-clés : congé"


def test_generate_note_raises_when_the_note_is_cut() -> None:
    with pytest.raises(NoteCutError, match="answer-length cap"):
        generate_note(fake_model("Une note coup", done_reason="length"), NOTE_INPUTS)


def test_generate_note_raises_on_an_empty_note() -> None:
    with pytest.raises(ValueError, match="empty note"):
        generate_note(fake_model("   "), NOTE_INPUTS)


def test_generate_note_refuses_a_reply_that_is_not_text() -> None:
    with pytest.raises(TypeError, match="expected text"):
        generate_note(fake_model(["une", "liste"]), NOTE_INPUTS)


def test_generate_note_works_when_the_model_sends_no_done_reason() -> None:
    assert generate_note(fake_model("Note.", done_reason=None), NOTE_INPUTS) == "Note."
