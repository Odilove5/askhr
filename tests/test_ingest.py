import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from askhr.context import NoteCutError, PromptTooLongError
from askhr.ingest import (
    OllamaNotReadyError,
    check_notes,
    check_ollama,
    chunk_row,
    main,
    plan_run,
    run,
    write_jsonl,
)
from askhr.models import Chunk, Document, Page
from askhr.notes_cache import NotesCache
from askhr.settings import Settings

CHUNK = Chunk(
    doc_id="9,999",
    index=3,
    text="La demande de congé fictif est envoyée à la direction.",
    page_start=2,
    page_end=3,
    title="Politique fictive",
    doc_type="politique",
    language="fr",
    url="https://example.org/fictive.pdf",
    heading="4. Procédure",
    context="Section fictive sur les congés. Mots-clés : congé, délai",
)

ROWS: list[dict[str, Any]] = [{"a": 1, "texte": "élève"}, {"b": 2}]


def test_chunk_row_has_every_field_in_a_fixed_order() -> None:
    row = chunk_row(CHUNK, "v4", "gemma3:12b")
    assert list(row) == [
        "chunk_id",
        "doc_id",
        "index",
        "text",
        "page_start",
        "page_end",
        "title",
        "doc_type",
        "language",
        "url",
        "heading",
        "context",
        "prompt_version",
        "model",
    ]
    assert row["chunk_id"] == "9,999#3"
    assert row["context"] == CHUNK.context
    assert (row["prompt_version"], row["model"]) == ("v4", "gemma3:12b")


def test_write_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "chunks.jsonl"  # parent folder does not exist yet
    write_jsonl(path, ROWS)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == ROWS


def test_write_jsonl_keeps_french_characters(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, [chunk_row(CHUNK, "v4", "gemma3:12b")])
    text = path.read_text(encoding="utf-8")
    assert "congé" in text
    assert "\\u00e9" not in text


def test_write_jsonl_same_input_gives_same_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(first, ROWS)
    write_jsonl(second, ROWS)
    assert first.read_bytes() == second.read_bytes()


def test_write_jsonl_failure_leaves_old_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, ROWS)
    before = path.read_bytes()
    # The first row is written, then json.dumps fails on the second one.
    with pytest.raises(TypeError, match="not JSON serializable"):
        write_jsonl(path, [{"ok": 1}, {"bad": object()}])
    assert path.read_bytes() == before
    assert not (tmp_path / "chunks.jsonl.tmp").exists()


def test_write_jsonl_cleans_up_on_ctrl_c(tmp_path: Path) -> None:
    def rows() -> Iterator[dict[str, Any]]:
        yield {"ok": 1}
        raise KeyboardInterrupt

    path = tmp_path / "chunks.jsonl"
    with pytest.raises(KeyboardInterrupt):
        write_jsonl(path, rows())
    assert not path.exists()
    assert not (tmp_path / "chunks.jsonl.tmp").exists()


# ── rows 4 to 8: plan, pre-flight, run, report, command line ─────────────────────────


PAGE = """PRÉAMBULE
Le Conseil fictif reconnaît l'importance des règles de test, et ce texte est assez long
pour former une section à lui seul, car il dépasse le minimum de caractères demandé.
1. Demande de congé
La demande est envoyée à la direction fictive au moins dix jours ouvrables avant le
début du congé. La direction répond par écrit dans un délai de cinq jours ouvrables.
2. Retour au travail
Au retour, la personne avise son superviseur fictif et remet le formulaire fictif."""

NOTE = "Section fictive. Mots-clés : congé, délai"
BIG = 100_000


def make_doc(doc_id: str = "9,999", text: str = PAGE) -> Document:
    return Document(
        doc_id=doc_id,
        title="Congés fictifs",
        doc_type="directive",
        section="Tests",
        language="fr",
        url="https://example.org/fictive.pdf",
        pages=(Page(number=1, text=text),),
    )


class CountingModel(GenericFakeChatModel):
    """A fake model that counts its calls; each reply is (text, done_reason)."""

    calls: int = 0

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return super()._generate(*args, **kwargs)


def model(*replies: tuple[str, str]) -> CountingModel:
    messages = [AIMessage(content=t, response_metadata={"done_reason": r}) for t, r in replies]
    return CountingModel(messages=iter(messages))


def ok(n: int) -> CountingModel:
    return model(*[(NOTE, "stop")] * n)


def quiet(_: str) -> None:
    pass


def test_plan_builds_one_prompt_and_key_per_chunk() -> None:
    [plan] = plan_run([make_doc()], BIG, "m")
    assert len(plan.chunks) == len(plan.inputs) == len(plan.keys) >= 1
    assert len(set(plan.keys)) == len(plan.keys)


def test_plan_fails_early_on_a_prompt_too_long() -> None:
    with pytest.raises(PromptTooLongError):
        plan_run([make_doc()], 10, "m")


def test_key_depends_on_model_name() -> None:
    [a] = plan_run([make_doc()], BIG, "m1")
    [b] = plan_run([make_doc()], BIG, "m2")
    assert set(a.keys).isdisjoint(b.keys)


def test_cached_notes_are_cleaned_again_when_read(tmp_path: Path) -> None:
    # A note cached before a clean_note fix must get that fix too, with no model call.
    plans = plan_run([make_doc()], BIG, "m")
    cache = NotesCache(tmp_path / "notes.jsonl")
    for chunk, key in zip(plans[0].chunks, plans[0].keys, strict=True):
        cache.put(key, chunk.chunk_id, "Annexe B (page 23). Mots-clés : congé")
    unused = ok(0)
    chunks, _ = run(plans, cache, unused, unused, log=quiet)
    assert unused.calls == 0
    assert all(c.context == "Annexe B. Mots-clés : congé" for c in chunks)


def test_run_all_new_then_all_cached(tmp_path: Path) -> None:
    plans = plan_run([make_doc()], BIG, "m")
    n = len(plans[0].chunks)
    cache = NotesCache(tmp_path / "notes.jsonl")
    first = ok(n)
    chunks, stats = run(plans, cache, first, ok(0), log=quiet)
    assert (first.calls, stats.new, stats.cached) == (n, n, 0)
    assert all(c.context == NOTE for c in chunks)

    again = ok(0)
    chunks2, stats2 = run(plans, NotesCache(tmp_path / "notes.jsonl"), again, ok(0), log=quiet)
    assert (again.calls, stats2.new, stats2.cached) == (0, 0, n)
    assert chunks2 == chunks


def test_resume_after_a_failure_halfway(tmp_path: Path) -> None:
    other = make_doc("9,998", PAGE.replace("fictif", "imaginaire"))
    plans = plan_run([make_doc(), other], BIG, "m")
    first_doc = len(plans[0].chunks)
    total = first_doc + len(plans[1].chunks)
    path = tmp_path / "notes.jsonl"
    # The model has replies for the first document only, then fails.
    with pytest.raises(Exception) as caught:
        run(plans, NotesCache(path), ok(first_doc), ok(0), log=quiet)
    assert any("9,998#0" in note for note in getattr(caught.value, "__notes__", []))
    second = ok(total - first_doc)
    _, stats = run(plans, NotesCache(path), second, ok(0), log=quiet)
    assert (stats.cached, stats.new, second.calls) == (
        first_doc,
        total - first_doc,
        total - first_doc,
    )


def test_cut_note_is_retried_with_the_long_model(tmp_path: Path) -> None:
    plans = plan_run([make_doc()], BIG, "m")
    n = len(plans[0].chunks)
    short = model(("Note coup", "length"), *[(NOTE, "stop")] * (n - 1))
    long = ok(1)
    chunks, stats = run(plans, NotesCache(tmp_path / "n.jsonl"), short, long, log=quiet)
    assert (stats.retried, long.calls) == (1, 1)
    assert chunks[0].context == NOTE


def test_cut_twice_stops_naming_the_chunk(tmp_path: Path) -> None:
    plans = plan_run([make_doc()], BIG, "m")
    cut = ("Note coup", "length")
    with pytest.raises(NoteCutError, match="9,999#0"):
        run(plans, NotesCache(tmp_path / "n.jsonl"), model(cut), model(cut), log=quiet)
    assert not (tmp_path / "n.jsonl").exists()  # nothing cut was cached


def settings() -> Settings:
    return Settings(ollama_model="gemma:12b", ollama_base_url="http://ollama.test")


def client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_check_ollama_passes_when_the_model_is_pulled() -> None:
    reply = {"models": [{"name": "gemma:12b", "model": "gemma:12b"}]}
    check_ollama(settings(), client(lambda _: httpx.Response(200, json=reply)))


def test_check_ollama_names_the_pull_command_when_the_model_is_missing() -> None:
    reply = {"models": [{"name": "other:1b", "model": "other:1b"}]}
    with pytest.raises(OllamaNotReadyError, match="ollama pull gemma:12b"):
        check_ollama(settings(), client(lambda _: httpx.Response(200, json=reply)))


def test_check_ollama_explains_when_ollama_is_down() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaNotReadyError, match="ollama serve"):
        check_ollama(settings(), client(down))


@pytest.mark.parametrize(
    ("note", "flag"),
    [
        ("Annexe B, page 23. Mots-clés : congé", "naming_a_page"),
        ("Annexe B, sans mots-clés.", "without_keywords"),
        ("Le congéDemande fictif. Mots-clés : congé", "joined_words"),
        ("Retour au travailet suivi. Mots-clés : retour", "joined_words"),
    ],
)
def test_check_notes_flags_a_bad_note(note: str, flag: str) -> None:
    checks = check_notes([replace_context(note)])
    assert getattr(checks, flag)


def test_check_notes_passes_a_clean_note() -> None:
    checks = check_notes([replace_context(NOTE)])
    assert not (checks.naming_a_page or checks.without_keywords or checks.joined_words)


def replace_context(note: str) -> Chunk:
    return replace(CHUNK, context=note)


MANIFEST = """- doc_id: 9,999
  title: Congés fictifs
  doc_type: directive
  section: Tests
  language: fr
  url: https://example.org/fictive.pdf
"""


def run_main(tmp_path: Path, *argv: str) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(MANIFEST, encoding="utf-8")
    main(
        list(argv),
        settings=settings(),
        manifest_path=manifest,
        raw_dir=tmp_path / "raw",
        cache_path=tmp_path / "cache" / "notes.jsonl",
        output_path=tmp_path / "chunks.jsonl",
        models=(ok(0), ok(0)),
    )


def test_unknown_doc_id_lists_the_known_ones(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"Unknown doc_id \['1,234'\]. Known: 9,999"):
        run_main(tmp_path, "--doc", "1,234")


def test_missing_pdf_stops_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="downloader"):
        run_main(tmp_path, "--dry-run")
    assert list(tmp_path.iterdir()) == [tmp_path / "manifest.yaml"]


# main() end to end, with made-up documents in place of the PDFs.
def main_with_fake_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: tuple[Any, Any], *argv: str
) -> None:
    monkeypatch.setattr(
        "askhr.ingest.select_docs", lambda manifest, raw, doc_ids: ([make_doc()], 1)
    )
    main(
        list(argv),
        settings=settings(),
        cache_path=tmp_path / "cache" / "notes.jsonl",
        output_path=tmp_path / "chunks.jsonl",
        models=models,
    )


def chunk_count() -> int:
    return len(plan_run([make_doc()], BIG, "m")[0].chunks)


def test_full_run_writes_every_chunk_with_its_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    n = chunk_count()
    main_with_fake_docs(tmp_path, monkeypatch, (ok(n), ok(0)))
    rows = [
        json.loads(line) for line in (tmp_path / "chunks.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(rows) == n
    assert all(row["context"] == NOTE for row in rows)
    assert f"notes: 0 from cache, {n} new, 0 retried" in capsys.readouterr().out


def test_second_run_makes_no_model_call_and_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_with_fake_docs(tmp_path, monkeypatch, (ok(chunk_count()), ok(0)))
    before = (tmp_path / "chunks.jsonl").read_bytes()
    again = ok(0)
    main_with_fake_docs(tmp_path, monkeypatch, (again, ok(0)))
    assert again.calls == 0
    assert (tmp_path / "chunks.jsonl").read_bytes() == before


def test_dry_run_calls_no_model_and_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    unused = ok(0)
    main_with_fake_docs(tmp_path, monkeypatch, (unused, unused), "--dry-run")
    assert unused.calls == 0
    assert list(tmp_path.iterdir()) == []
    assert f"0 in cache, {chunk_count()} would need the model" in capsys.readouterr().out


def test_doc_run_caches_notes_but_leaves_the_output_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "chunks.jsonl"
    output.write_text('{"old": true}\n', encoding="utf-8")
    main_with_fake_docs(tmp_path, monkeypatch, (ok(chunk_count()), ok(0)), "--doc", "9,999")
    assert output.read_text(encoding="utf-8") == '{"old": true}\n'
    assert len(NotesCache(tmp_path / "cache" / "notes.jsonl")) > 0
