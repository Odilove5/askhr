from pathlib import Path

import pytest

from askhr.notes_cache import NotesCache, note_key

PARTS = ("v4", "gemma3:12b", "abc123", "La demande est envoyée à la direction fictive.")


def test_round_trip(tmp_path: Path) -> None:
    cache = NotesCache(tmp_path / "notes.jsonl")
    assert cache.broken_lines == 0
    cache.put("k1", "9,999#0", "Section fictive. Mots-clés : congé, délai")
    assert cache.get("k1") == "Section fictive. Mots-clés : congé, délai"
    assert cache.get("missing") is None
    assert len(cache) == 1


def test_reopening_sees_earlier_writes(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "notes.jsonl"  # parent folder does not exist yet
    NotesCache(path).put("k1", "9,999#0", "note un")
    NotesCache(path).put("k2", "9,999#1", "note deux")
    reopened = NotesCache(path)
    assert reopened.get("k1") == "note un"
    assert reopened.get("k2") == "note deux"


def test_key_is_stable() -> None:
    assert note_key(*PARTS) == note_key(*PARTS)
    assert len(note_key(*PARTS)) == 64


@pytest.mark.parametrize("index", range(4))
def test_each_key_part_changes_the_key(index: int) -> None:
    changed = list(PARTS)
    changed[index] += "x"
    assert note_key(*changed) != note_key(*PARTS)


def test_parts_cannot_run_into_each_other() -> None:
    assert note_key("v4", "ab", "c", "t") != note_key("v4", "a", "bc", "t")
    assert note_key("v4", "m|x", "h", "t") != note_key("v4", "m", "x|h", "t")


def test_broken_lines_are_skipped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "notes.jsonl"
    path.write_text(
        '{"key": "good", "chunk_id": "9,999#0", "context": "bonne note"}\n'
        "not json at all\n"
        '{"chunk_id": "9,999#1"}\n'  # no key
        '{"key": 1, "chunk_id": "9,999#2", "context": "x"}\n'  # key is not text
        "[1, 2]\n",  # not an object
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="broken cache line") as record:
        cache = NotesCache(path)
    assert len(record) == 4
    assert cache.broken_lines == 4
    assert cache.get("good") == "bonne note"
    assert len(cache) == 1


def test_write_after_a_cut_last_line_is_not_lost(tmp_path: Path) -> None:
    path = tmp_path / "notes.jsonl"
    path.write_text(
        '{"key": "a", "chunk_id": "9,999#0", "context": "note a"}\n{"key": "b", "chu',
        encoding="utf-8",
    )
    with pytest.warns(UserWarning):
        NotesCache(path).put("c", "9,999#2", "note c")
    with pytest.warns(UserWarning):  # the cut line is still there, still skipped
        reopened = NotesCache(path)
    assert reopened.get("a") == "note a"
    assert reopened.get("c") == "note c"
    assert reopened.broken_lines == 1  # only the cut line; the note written after it is intact


def test_later_line_wins_for_the_same_key(tmp_path: Path) -> None:
    path = tmp_path / "notes.jsonl"
    cache = NotesCache(path)
    cache.put("k", "9,999#0", "ancienne")
    cache.put("k", "9,999#0", "nouvelle")
    assert NotesCache(path).get("k") == "nouvelle"


def test_french_text_kept_as_is(tmp_path: Path) -> None:
    path = tmp_path / "notes.jsonl"
    note = "Congé « fictif » : délai d'été ; prêt ? Mots-clés : congé, élève"
    NotesCache(path).put("k", "9,999#0", note)
    raw = path.read_text(encoding="utf-8")
    assert "Congé « fictif » : délai" in raw  # not escaped as \u00e9
    assert NotesCache(path).get("k") == note
