"""Ingest: every chunk of every document, with its contextual note, in data/chunks.jsonl.

    manifest ──loader──▶ Documents ──chunker──▶ Chunks ──plan──▶ prompts (all checked first)
        ──cache or model──▶ Chunks with notes ──atomic write──▶ data/chunks.jsonl + report

Run with:  uv run python -m askhr.ingest [--doc 5,202a ...] [--dry-run]
"""

import argparse
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel

from askhr.chunker import chunk_document
from askhr.connectors.downloader import MANIFEST_PATH, RAW_DIR, load_manifest
from askhr.context import (
    ANSWER_TOKENS,
    NOTE_MAX_TOKENS,
    PAGE_WORD,
    PROMPT_VERSION,
    NoteCutError,
    chunk_inputs,
    clean_note,
    generate_note,
)
from askhr.llm import get_chat_model
from askhr.loader import load_document
from askhr.models import Chunk, Document
from askhr.notes_cache import NotesCache, note_key
from askhr.settings import Settings, get_settings

OUTPUT_PATH = Path("data/chunks.jsonl")
CACHE_PATH = Path("data/cache/notes.jsonl")
OLLAMA_CHECK_TIMEOUT_S = 5.0

KEYWORDS_MARK = "Mots-clés : "  # clean_note always writes the keywords this way
# A lowercase letter glued to a capital inside a word ("congéDemande"): usually two words
# that lost their space in the PDF. Listed for a person to check, never fixed silently.
CAMEL_JOIN = re.compile(r"\b\w*[a-zà-ÿ][A-ZÀ-Ý]\w*\b")
# Joins with no capital to give them away. Add each one seen in a real run.
KNOWN_JOINS = ("travailet",)


def chunk_row(chunk: Chunk, prompt_version: str, model: str) -> dict[str, Any]:
    """One output line: the chunk id, every Chunk field, then how its note was made.

    Dicts keep insertion order, and asdict follows the class's field order, so the
    keys always come out in the same order and the file is byte-for-byte stable.
    """
    return {
        "chunk_id": chunk.chunk_id,  # a property, so asdict leaves it out
        **asdict(chunk),
        "prompt_version": prompt_version,
        "model": model,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write all rows or nothing: a crash halfway leaves the old file as it was.

    Rows go to a temporary file in the same folder, which then replaces `path` in one
    atomic rename. Same folder matters: a rename is only atomic within one filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # chunks.jsonl.tmp, not chunks.tmp
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())  # on disk before the rename, not just in a buffer
        os.replace(tmp, path)
    except BaseException:  # BaseException, so Ctrl+C (KeyboardInterrupt) cleans up too
        tmp.unlink(missing_ok=True)
        raise  # bare raise: the same error, with its original traceback


# ── Row 4: plan the run before any model call ────────────────────────────────────────


@dataclass(frozen=True)
class DocPlan:
    """One document's chunks, each with its prompt inputs and its cache key."""

    doc: Document
    chunks: tuple[Chunk, ...]
    inputs: tuple[dict[str, str], ...]
    keys: tuple[str, ...]

    def todo(self, cache: NotesCache) -> int:
        """How many of this document's notes are not in the cache yet."""
        return sum(1 for key in self.keys if cache.get(key) is None)


def plan_run(docs: Sequence[Document], max_prompt_tokens: int, model_name: str) -> list[DocPlan]:
    """Chunk every document and build every prompt. Raises PromptTooLongError early,
    so a prompt that can't fit stops the run in seconds, not after 30 minutes."""
    plans = []
    for doc in docs:
        chunks = chunk_document(doc)
        inputs = tuple(chunk_inputs(doc, chunks, i, max_prompt_tokens) for i in range(len(chunks)))
        keys = tuple(
            note_key(PROMPT_VERSION, model_name, doc.content_hash, chunk.text) for chunk in chunks
        )
        plans.append(DocPlan(doc, tuple(chunks), inputs, keys))
    return plans


# ── Row 5: Ollama pre-flight ─────────────────────────────────────────────────────────


class OllamaNotReadyError(RuntimeError):
    """Ollama is not running, or the model is not pulled."""


def check_ollama(settings: Settings, client: httpx.Client | None = None) -> None:
    """Fail in seconds with a clear message, instead of on the first of 400 model calls."""
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    http = client if client is not None else httpx.Client(timeout=OLLAMA_CHECK_TIMEOUT_S)
    try:
        response = http.get(url)
        response.raise_for_status()
        listed = response.json().get("models", [])
    except (httpx.HTTPError, ValueError) as error:
        raise OllamaNotReadyError(
            f"Ollama does not answer at {settings.ollama_base_url} ({error}). "
            "Start it with: ollama serve"
        ) from error
    finally:
        if client is None:  # close only the client we opened ourselves
            http.close()
    wanted = settings.ollama_model
    names = {item.get("name") for item in listed} | {item.get("model") for item in listed}
    # Ollama adds ":latest" to a model pulled without a tag.
    if wanted not in names and f"{wanted}:latest" not in names:
        raise OllamaNotReadyError(f"Model {wanted!r} is not pulled. Run: ollama pull {wanted}")


# ── Row 6: run ───────────────────────────────────────────────────────────────────────


@dataclass
class RunStats:
    cached: int = 0
    new: int = 0
    retried: int = 0  # notes cut at the cap, then written in full with the bigger cap


def note_with_retry(
    model: BaseChatModel, long_model: BaseChatModel, inputs: dict[str, str], chunk_id: str
) -> tuple[str, bool]:
    """The note and whether it needed the retry. Cut twice: stop, naming the chunk."""
    try:
        return generate_note(model, inputs), False
    except NoteCutError:
        pass
    try:
        return generate_note(long_model, inputs), True
    except NoteCutError as error:
        raise NoteCutError(f"{chunk_id}: note still cut with the doubled cap") from error


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def run(
    plans: Sequence[DocPlan],
    cache: NotesCache,
    model: BaseChatModel,
    long_model: BaseChatModel,
    log: Callable[[str], None] = print,
) -> tuple[list[Chunk], RunStats]:
    """Fill in every note: from the cache, or from the model and then straight into the
    cache. Any error stops the run; the notes written before it stay in the cache."""
    stats = RunStats()
    result: list[Chunk] = []
    for plan in plans:
        start = time.monotonic()
        cached = new = 0
        for chunk, inputs, key in zip(plan.chunks, plan.inputs, plan.keys, strict=True):
            note = cache.get(key)
            if note is None:
                try:
                    note, retried = note_with_retry(model, long_model, inputs, chunk.chunk_id)
                except NoteCutError:
                    raise  # already names the chunk
                except Exception as error:
                    error.add_note(f"while writing the note of {chunk.chunk_id}")
                    raise
                cache.put(key, chunk.chunk_id, note)
                new += 1
                stats.retried += retried
            else:
                # Clean again: a note cached before a clean_note fix gets the fix too.
                note = clean_note(note)
                cached += 1
            result.append(replace(chunk, context=note))
        stats.cached += cached
        stats.new += new
        log(
            f"{plan.doc.doc_id}: {len(plan.chunks)} chunks, {cached} cached, {new} new, "
            f"{format_duration(time.monotonic() - start)}"
        )
    return result, stats


# ── Row 7: check report ──────────────────────────────────────────────────────────────


@dataclass
class NoteChecks:
    """Chunk ids (or words) a person should look at. Empty lists mean all clear."""

    naming_a_page: list[str] = field(default_factory=list)
    without_keywords: list[str] = field(default_factory=list)
    joined_words: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, word)


def check_notes(chunks: Iterable[Chunk]) -> NoteChecks:
    checks = NoteChecks()
    for chunk in chunks:
        note = chunk.context
        if PAGE_WORD.search(note):
            checks.naming_a_page.append(chunk.chunk_id)
        if KEYWORDS_MARK not in note:
            checks.without_keywords.append(chunk.chunk_id)
        words = CAMEL_JOIN.findall(note)
        words += [word for word in KNOWN_JOINS if word in note.lower()]
        checks.joined_words += [(chunk.chunk_id, word) for word in words]
    return checks


def report(
    docs: int, manifest_docs: int, chunks: Sequence[Chunk], stats: RunStats, broken_lines: int
) -> str:
    checks = check_notes(chunks)
    lines = [
        "── check report ──",
        f"documents: {docs} (manifest: {manifest_docs})",
        f"chunks: {len(chunks)}",
        f"notes: {stats.cached} from cache, {stats.new} new, {stats.retried} retried",
        f"broken cache lines skipped: {broken_lines}",
        f"notes naming a page: {len(checks.naming_a_page)} {checks.naming_a_page}",
        f"notes without '{KEYWORDS_MARK.strip()}': "
        f"{len(checks.without_keywords)} {checks.without_keywords}",
        f"suspicious joined words: {len(checks.joined_words)}",
        *(f"  {chunk_id}: {word}" for chunk_id, word in checks.joined_words),
    ]
    return "\n".join(lines)


# ── Row 8: command line ──────────────────────────────────────────────────────────────


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m askhr.ingest", description=__doc__)
    parser.add_argument(
        "--doc",
        action="append",
        default=[],
        metavar="DOC_ID",
        help="ingest only this document (repeatable). Notes are cached; "
        f"{OUTPUT_PATH} is not rewritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build every prompt and count the notes to write; no model calls, no files",
    )
    return parser.parse_args(argv)


def select_docs(
    manifest_path: Path, raw_dir: Path, doc_ids: Sequence[str]
) -> tuple[list[Document], int]:
    """The documents to ingest, in manifest order, and the manifest's size."""
    entries = load_manifest(manifest_path)
    known = [entry.doc_id for entry in entries]
    unknown = [doc_id for doc_id in doc_ids if doc_id not in known]
    if unknown:
        raise SystemExit(f"Unknown doc_id {unknown}. Known: {', '.join(sorted(known))}")
    chosen = [entry for entry in entries if not doc_ids or entry.doc_id in doc_ids]
    # Loading every PDF first means a missing one stops the run before any model call.
    return [load_document(entry, raw_dir) for entry in chosen], len(entries)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    manifest_path: Path = MANIFEST_PATH,
    raw_dir: Path = RAW_DIR,
    cache_path: Path = CACHE_PATH,
    output_path: Path = OUTPUT_PATH,
    models: tuple[BaseChatModel, BaseChatModel] | None = None,  # tests pass fakes here
) -> None:
    args = parse_args(argv)
    settings = settings or get_settings()
    num_ctx = settings.ollama_num_ctx_long
    model_name = settings.ollama_model

    docs, manifest_docs = select_docs(manifest_path, raw_dir, args.doc)
    plans = plan_run(docs, num_ctx - ANSWER_TOKENS, model_name)
    cache = NotesCache(cache_path)

    if args.dry_run:
        total = sum(len(plan.chunks) for plan in plans)
        todo = sum(plan.todo(cache) for plan in plans)
        print(f"documents: {len(docs)} (manifest: {manifest_docs})")
        print(f"chunks: {total}, every prompt fits in {num_ctx} tokens")
        print(f"notes: {total - todo} in cache, {todo} would need the model")
        print(f"broken cache lines skipped: {cache.broken_lines}")
        return

    if models is None:
        check_ollama(settings)
        models = (
            get_chat_model("local", settings, num_ctx=num_ctx, num_predict=NOTE_MAX_TOKENS),
            get_chat_model("local", settings, num_ctx=num_ctx, num_predict=2 * NOTE_MAX_TOKENS),
        )
    chunks, stats = run(plans, cache, *models)

    if args.doc:
        print(f"--doc run: notes cached in {cache_path}; {output_path} left unchanged")
    else:
        write_jsonl(output_path, (chunk_row(c, PROMPT_VERSION, model_name) for c in chunks))
        print(f"wrote {len(chunks)} chunks to {output_path}")
    print(report(len(docs), manifest_docs, chunks, stats, cache.broken_lines))


if __name__ == "__main__":
    main()
