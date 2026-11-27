"""Notes cache: contextual notes already written, so ingest never asks the model twice.

    data/cache/notes.jsonl   one line per note: {"key", "chunk_id", "context"}

The file is append-only. Each new note is written and flushed to disk at once, so a run
that stops (Ctrl+C, crash, power) keeps every note written before it stopped.
"""

import hashlib
import json
import os
import warnings
from pathlib import Path


def note_key(prompt_version: str, model: str, doc_hash: str, chunk_text: str) -> str:
    """SHA-256 of everything a note depends on. Change any part and the note is redone.

    The parts are hashed as a JSON list, not joined with a separator: chunk text can
    contain any character, so ("a|b", "c") and ("a", "b|c") would otherwise collide.
    """
    parts = [prompt_version, model, doc_hash, chunk_text]
    encoded = json.dumps(parts, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class NotesCache:
    """Notes by key, loaded from an append-only JSON Lines file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._notes: dict[str, str] = {}
        # A crash mid-write leaves a last line with no newline. The next write must start
        # on a fresh line, or it would be glued to the broken one and lost with it.
        self._needs_newline = False
        self.broken_lines = 0  # lines skipped on load; the run's check report shows this
        if path.exists():
            self._load()

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        self._needs_newline = bool(text) and not text.endswith("\n")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key, context = row["key"], row["context"]
                if not isinstance(key, str) or not isinstance(context, str):
                    raise TypeError("key and context must be text")
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                warnings.warn(
                    f"{self.path}:{number}: broken cache line skipped ({error}); "
                    "its note will be redone",
                    stacklevel=3,
                )
                self.broken_lines += 1
                continue
            self._notes[key] = context  # a later line for the same key wins

    def __len__(self) -> int:
        return len(self._notes)

    def get(self, key: str) -> str | None:
        return self._notes.get(key)

    def put(self, key: str, chunk_id: str, context: str) -> None:
        """Remember a note and write it to disk before returning."""
        line = json.dumps(
            {"key": key, "chunk_id": chunk_id, "context": context}, ensure_ascii=False
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            if self._needs_newline:
                file.write("\n")
            file.write(line + "\n")
            file.flush()  # Python's buffer -> the operating system
            os.fsync(file.fileno())  # the operating system's buffer -> the disk
        self._needs_newline = False
        self._notes[key] = context
