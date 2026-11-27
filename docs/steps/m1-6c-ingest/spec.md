# Spec: M1 step 6c, one-command ingest

## Goal

Add `uv run python -m askhr.ingest`, which loads the 37 documents, chunks them, writes a contextual note for every chunk with the local model, and saves the result to `data/chunks.jsonl`. Notes already written are reused from a cache, so a run can stop at any point and resume, and an unchanged corpus needs no model call at all.

## In scope / out of scope

In scope: the ingest command, the notes cache, the output file, a check report printed at the end, tests.

Out of scope: embeddings and the database (M2), English documents, the cloud model, downloading (the downloader stays a separate step).

## Inputs and outputs

| | Path | Format |
|---|---|---|
| Input | `corpus/manifest.yaml`, `data/raw/*.pdf` | as today |
| Cache | `data/cache/notes.jsonl` | one line per note: `{"key", "chunk_id", "context"}` |
| Output | `data/chunks.jsonl` | one line per chunk: every `Chunk` field plus `chunk_id`, `prompt_version` and `model` |

All files are UTF-8 JSON Lines with French characters kept as they are (`ensure_ascii=False`). Everything stays under `data/`, which is gitignored.

Command options:
- `--doc 5,202a`: ingest only this document (repeatable). Useful for testing.
- `--dry-run`: load and chunk everything, build every prompt, and report how many notes would need the model. No model calls, no files written.

## Behaviour

**Normal path**
1. Load and chunk all documents.
2. Build every prompt for every chunk **before the first model call**, so a prompt that can't fit stops the run in seconds, not after 30 minutes.
3. Check that Ollama answers and the model is available; otherwise stop with a clear message.
4. For each document, in manifest order: reuse cached notes, ask the model for the missing ones, and **append each new note to the cache as soon as it's written**.
5. Write `data/chunks.jsonl` to a temporary file, then rename it, so a crash never leaves a half-written output.
6. Print a progress line per document (`5,203-dir: 55 chunks, 12 cached, 43 new, 3 min 40 s`) and the check report at the end.

**Cache key.** A note depends on the chunk, its document (the prompt contains the whole document), the prompt and the model. So the key is a SHA-256 of: `PROMPT_VERSION`, the model name, the document's `content_hash`, and the chunk text. Change any of them and only the affected notes are redone.

**Edge cases**

| Case | What happens |
|---|---|
| Interrupted run (Ctrl+C, crash, power) | Notes written so far are in the cache. Re-running continues where it stopped. |
| Nothing changed since last run | 0 model calls; `chunks.jsonl` is byte-for-byte identical. |
| One document's text changes | Only that document's notes are redone. |
| `PROMPT_VERSION` or model changes | All notes are redone (old cache lines are simply not used). |
| A document removed from the manifest | It disappears from the output. Its cache lines stay unused. |
| A note reaches the length cap (`done_reason == "length"`) | Retry once with double the cap. If still cut, stop with an error naming the chunk. |
| Empty note after cleaning | Stop with an error naming the chunk (as today). |
| Any model error | Stop. The cache keeps everything written so far. |
| A PDF is missing | Stop before any model call, with the downloader command (as the loader does today). |
| A cache line is broken (crash mid-write) | Skip that line with a warning; the note is simply redone. |

**Check report** (printed at the end, also with `--dry-run` where it applies)
- documents and chunks: counts, compared with the manifest (37 documents);
- notes: from cache, new, retried;
- notes still naming a page (`PAGE_WORD`), notes without `Mots-clés :`;
- suspicious joined words in notes (a lowercase letter directly followed by a capital, or common joins like "travailet"), listed for a person to check.

## Done when

- [x] `uv run python -m askhr.ingest --dry-run` reports 37 documents, about 416 chunks, every prompt fits, 0 model calls.
- [x] A full real run on the Mac writes `data/chunks.jsonl` with one line per chunk, each with a non-empty note.
- [x] Interrupting the full run, then re-running, finishes without redoing the notes already written (the report shows them as cached).
- [x] A second run with nothing changed makes 0 model calls, and `md5` of `chunks.jsonl` is unchanged.
- [x] Check report: 0 notes naming a page, 0 notes without keywords, 0 cut notes; any joined words listed and reviewed.
- [x] Tests with fake models cover: cache hit and miss, key changes, resume after a failure halfway, atomic output, cut-note retry, broken cache line, `--doc` and `--dry-run`.
- [ ] The four CI checks pass. (They pass locally; tick once GitHub Actions shows green.)

## Decisions (agreed 2026-11-27)

1. **Cut notes:** retry once with double the cap, then stop with an error naming the chunk.
2. **On a model error:** stop the run; the cache keeps the work done so far.
3. **Cache location:** `data/cache/notes.jsonl`, gitignored with the rest of `data/`.
