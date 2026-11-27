"""Ingest: every chunk of every document, with its contextual note, in data/chunks.jsonl.

Run with:  uv run python -m askhr.ingest
"""

import json
import os
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from askhr.models import Chunk


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
