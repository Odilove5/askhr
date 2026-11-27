import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from askhr.ingest import chunk_row, write_jsonl
from askhr.models import Chunk

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
