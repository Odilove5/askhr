"""Contextual notes: a short French note per chunk that places it in its document.

    Document + Chunk ──CONTEXT_PROMPT | model | StrOutputParser──▶ Chunk(context="…")

At M2 we index ``context + text``, so a chunk that only says "elle avise son
superviseur" can still be found by a search about medical leave.

Run with:  uv run python -m askhr.context [doc_id]      (default: 5,202a)
"""

import math
import sys
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from askhr.chunker import chunk_document
from askhr.llm import get_chat_model
from askhr.loader import load_all
from askhr.models import Chunk, Document
from askhr.settings import get_settings

PROMPT_VERSION = "v3"  # Increase whenever the prompt changes: step 6c re-runs stale notes.
CHARS_PER_TOKEN = 3  # French is ~3-4 characters per token; 3 overestimates, which is safe.
ANSWER_TOKENS = 512  # Room kept free in the window for the answer.
NOTE_MAX_TOKENS = 200  # Hard cap on the answer, so a looping model stops.

SYSTEM = """Tu rédiges des notes de contexte pour la recherche dans les documents RH \
d'un conseil scolaire.
On te donne un document, découpé en pages, puis un extrait de ce document avec ses pages.
Écris en français une note de 1 ou 2 phrases (au plus 50 mots) qui dit dans quelle \
section (ou annexe) se trouve l'extrait et de quoi il parle. Sers-toi des numéros de page \
pour situer l'extrait : un même texte peut se répéter à plusieurs endroits.
Termine toujours par « Mots-clés : » suivi de 3 à 5 mots qu'un membre du personnel \
utiliserait pour chercher ce passage (par exemple : congé, plainte, délai, formulaire).
Ne répète pas le titre du document : il est déjà connu. Ne commence pas par \
« Cet extrait ».
N'invente rien : utilise seulement ce qui est écrit dans le document.
Réponds uniquement par la note, sans titre, sans guillemets et sans introduction."""

# The document comes first and the chunk last: every chunk of a document then shares the
# same start of prompt, and Ollama reuses the work it already did on that start.
HUMAN = """<document titre="{title}">
{document}
</document>

<extrait section="{heading}" pages="{pages}">
{chunk}
</extrait>

Note de contexte :"""

CONTEXT_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


class PromptTooLongError(ValueError):
    """Even the short fallback prompt does not fit in the model's context window."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def prompt_tokens(inputs: dict[str, str]) -> int:
    """Estimated size of the full prompt (system + human) for these inputs."""
    messages = CONTEXT_PROMPT.format_messages(**inputs)
    return estimate_tokens("".join(str(message.content) for message in messages))


def outline_and_neighbours(chunks: list[Chunk], index: int) -> str:
    """Fallback for a document too long for the window: its headings + nearby chunks."""
    headings = list(dict.fromkeys(chunk.heading for chunk in chunks if chunk.heading))
    nearby = [chunks[i].text for i in (index - 1, index + 1) if 0 <= i < len(chunks)]
    return "\n".join(
        ["Plan du document :", *headings, "", "Passages voisins de l'extrait :", *nearby]
    )


def paged_text(doc: Document) -> str:
    """The document text with a "[page N]" marker before each page, so the model can tell
    apart passages that repeat (the same privacy notice ends Annexe A and Annexe B)."""
    return "\n\n".join(f"[page {page.number}]\n{page.text}" for page in doc.pages)


def page_span(chunk: Chunk) -> str:
    if chunk.page_start == chunk.page_end:
        return str(chunk.page_start)
    return f"{chunk.page_start}-{chunk.page_end}"


def chunk_inputs(
    doc: Document, chunks: list[Chunk], index: int, max_prompt_tokens: int
) -> dict[str, str]:
    """The prompt variables for chunks[index]: the whole document if it fits, else a fallback."""
    chunk = chunks[index]
    inputs = {
        "title": doc.title,
        "document": paged_text(doc),
        "heading": chunk.heading or "(sans titre)",
        "pages": page_span(chunk),
        "chunk": chunk.text,
    }
    if prompt_tokens(inputs) <= max_prompt_tokens:
        return inputs
    inputs["document"] = outline_and_neighbours(chunks, index)
    needed = prompt_tokens(inputs)
    if needed <= max_prompt_tokens:
        return inputs
    raise PromptTooLongError(
        f"{chunk.chunk_id}: the prompt needs about {needed} tokens, the limit is "
        f"{max_prompt_tokens}. Raise ASKHR_OLLAMA_NUM_CTX_LONG."
    )


def build_context_chain(model: BaseChatModel) -> Runnable[dict[str, str], str]:
    """prompt | model | parser. The caller chooses the model (local, cloud or a fake)."""
    return CONTEXT_PROMPT | model | StrOutputParser()


def add_context(
    doc: Document,
    chunks: list[Chunk],
    chain: Runnable[dict[str, str], str],
    max_prompt_tokens: int,
    max_concurrency: int = 1,
) -> list[Chunk]:
    """Return copies of the chunks with their contextual note filled in."""
    if not chunks:
        return []
    for chunk in chunks:
        if chunk.doc_id != doc.doc_id:
            raise ValueError(f"{chunk.chunk_id} does not belong to {doc.doc_id}")
    # Build every prompt first, so a prompt that can't fit fails before any model call.
    inputs = [chunk_inputs(doc, chunks, i, max_prompt_tokens) for i in range(len(chunks))]
    notes = chain.batch(inputs, config={"max_concurrency": max_concurrency})
    result = []
    for chunk, note in zip(chunks, notes, strict=True):
        note = " ".join(note.split())  # one line, no stray spaces or line breaks
        if not note:
            raise ValueError(f"{chunk.chunk_id}: the model returned an empty note")
        result.append(replace(chunk, context=note))
    return result


def main() -> None:
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "5,202a"
    docs = {doc.doc_id: doc for doc in load_all()}
    if doc_id not in docs:
        raise SystemExit(f"Unknown doc_id {doc_id!r}. Known: {', '.join(sorted(docs))}")
    doc = docs[doc_id]
    chunks = chunk_document(doc)

    settings = get_settings()
    num_ctx = settings.ollama_num_ctx_long
    max_prompt_tokens = num_ctx - ANSWER_TOKENS
    model = get_chat_model("local", settings, num_ctx=num_ctx, num_predict=NOTE_MAX_TOKENS)

    # Compare our size estimate with Ollama's own count on the first chunk.
    first = chunk_inputs(doc, chunks, 0, max_prompt_tokens)
    reply = (CONTEXT_PROMPT | model).invoke(first)
    counted = (
        reply.usage_metadata["input_tokens"]
        if isinstance(reply, AIMessage) and reply.usage_metadata
        else None
    )
    print(f"{doc_id}: {len(doc.text)} chars, {len(chunks)} chunks, window {num_ctx} tokens")
    print(f"prompt of chunk 0: estimated {prompt_tokens(first)} tokens, Ollama counted {counted}")

    for chunk in add_context(doc, chunks, build_context_chain(model), max_prompt_tokens):
        print(f"\n[{chunk.chunk_id}] {chunk.heading or '(sans titre)'}  p.{chunk.page_start}")
        print(f"  → {chunk.context}")


if __name__ == "__main__":
    main()
