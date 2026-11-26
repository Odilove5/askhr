"""Split each Document into Chunks: cut at headings first, then fix the sizes.

Run with:  uv run python -m askhr.chunker
"""

import math
import re
from dataclasses import dataclass, field

from askhr.loader import load_all
from askhr.models import Chunk, Document

MAX_CHARS = 1500  # About 350-400 tokens of French; tuned later with the evals (M2)
MIN_CHARS = 400  # A section shorter than this is merged with its neighbour when it fits
TINY_CHARS = (
    200  # A piece this small is merged even if the chunk then passes MAX_CHARS by this much
)
MAX_HEADING_CHARS = 90  # A longer line is a sentence, not a heading
MAX_NUMBERED_HEADING_WORDS = 12  # "7. Le Conseil veille à ce que la présente…" is a sentence

# "2. Divulgation", "1.1.2. Harcèlement…": numbers ending with a dot, then a letter.
# The letter rules out "1 2 3 4 s.o." and dates like "18 juillet 2024".
NUMBERED = re.compile(r"\d+(?:\.\d+)*\.\s+[^\W\d_]")
# "Accessibilité et équité : L'ensemble…" is a list item: a colon followed by more text.
COLON_THEN_TEXT = re.compile(r":\s*\S")
# A real word of 3 letters or more: rules out table cells such as "X X X X".
WORD = re.compile(r"[^\W\d_]{3,}")


def is_heading(line: str) -> bool:
    """Guess whether a line is a section heading. A heuristic: cheap to get wrong."""
    line = line.strip()
    if not line or len(line) > MAX_HEADING_CHARS:
        return False
    if NUMBERED.match(line):
        is_sentence = (
            COLON_THEN_TEXT.search(line) is not None
            or line.endswith((".", ",", ";"))  # list items end with punctuation, headings don't
            or len(line.split()) > MAX_NUMBERED_HEADING_WORDS
        )
        return not is_sentence
    letters = sum(1 for char in line if char.isalpha())
    # "PLUS HAUT)" is the end of a wrapped line, not a heading: a ")" with no "(".
    unbalanced = line.endswith(")") and "(" not in line
    return (
        line.isupper()
        and letters >= 4
        and WORD.search(line) is not None
        and not unbalanced  # "PRÉAMBULE", "MODALITÉS"
    )


@dataclass
class _Section:
    """Work in progress inside this module: a heading and its (page, line) pairs.

    ``parts`` remembers, after merging, the heading and size of each piece that was merged
    in, so the chunk can be named after the right section (see ``_best_heading``).
    """

    heading: str
    lines: list[tuple[int, str]] = field(default_factory=list)
    parts: list[tuple[str, int, bool]] = field(default_factory=list)  # heading, size, starts it

    def size(self) -> int:
        return len(self.text())

    def text(self) -> str:
        return "\n".join(line for _, line in self.lines)


def split_sections(doc: Document) -> list[_Section]:
    """Start a new section at every heading. Text before the first heading gets heading ""."""
    sections = [_Section(heading="")]
    for page in doc.pages:
        for line in page.text.splitlines():
            if is_heading(line):
                sections.append(_Section(heading=line))
            sections[-1].lines.append((page.number, line))
    return [section for section in sections if section.lines]


def _split_long(section: _Section, max_chars: int) -> list[_Section]:
    """Cut a section that is too long at line breaks; every piece keeps the heading.

    The pieces are balanced (about the same size), so the last one is not a tiny leftover:
    1,600 characters become 2 pieces of ~800, not 1,500 + 100.
    """
    count = max(1, math.ceil(section.size() / max_chars))
    target = section.size() / count
    pieces = [_Section(heading=section.heading)]
    for page_line in section.lines:
        current = pieces[-1]
        too_big = current.size() + 1 + len(page_line[1]) > max_chars
        if current.lines and (too_big or current.size() >= target):
            current = _Section(heading=section.heading)
            pieces.append(current)
        current.lines.append(page_line)
    return pieces


def _starts_with_heading(section: _Section) -> bool:
    return bool(section.lines) and is_heading(section.lines[0][1])


def _best_heading(section: _Section) -> str:
    """Name a chunk after its first heading, unless it starts with the tail of a section.

    A chunk that starts with the last lines of section 8 and then holds all of section 9
    is about section 9: in that case use the heading of its biggest part.
    """
    first_heading, _, first_starts = section.parts[0]
    if first_starts or len(section.parts) == 1:
        return first_heading
    return max(section.parts, key=lambda part: part[1])[0]


def pack(sections: list[_Section], max_chars: int = MAX_CHARS) -> list[_Section]:
    """Split the sections that are too long, then merge small neighbours that fit together.

    A heading with no text yet (like "MODALITÉS" right before "1. ...") is not glued to
    the text above it: a heading belongs to the text that follows it. It can still join
    the previous piece when that piece also ends with a heading (stacked headings).

    A tiny piece (under TINY_CHARS, such as a one-line leftover) is merged even if the
    chunk then passes ``max_chars`` by up to TINY_CHARS: a chunk of 1,650 characters is
    better than a 70-character chunk with no context. Both limits scale with ``max_chars``.
    """
    tiny = TINY_CHARS * max_chars // MAX_CHARS  # 200 for the default budget, scaled otherwise
    pieces = [piece for section in sections for piece in _split_long(section, max_chars)]
    packed: list[_Section] = []
    for piece in pieces:
        part = (piece.heading, piece.size(), _starts_with_heading(piece))
        previous = packed[-1] if packed else None
        if previous is not None:
            heading_only = piece.heading != "" and len(piece.lines) == 1
            after_heading = is_heading(previous.lines[-1][1])
            smallest = min(previous.size(), piece.size())
            limit = max_chars + (tiny if smallest < tiny else 0)
            fits = previous.size() + 1 + piece.size() <= limit
            if smallest < MIN_CHARS and fits and (after_heading or not heading_only):
                previous.lines.extend(piece.lines)
                previous.parts.append(part)
                continue
        packed.append(_Section(heading=piece.heading, lines=list(piece.lines), parts=[part]))
    for section in packed:
        section.heading = _best_heading(section)
    return packed


def chunk_document(doc: Document, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Turn one Document into Chunks that carry its metadata, heading and page span."""
    return [
        Chunk(
            doc_id=doc.doc_id,
            index=index,
            text=section.text(),
            page_start=section.lines[0][0],
            page_end=section.lines[-1][0],
            title=doc.title,
            doc_type=doc.doc_type,
            language=doc.language,
            url=doc.url,
            heading=section.heading,
        )
        for index, section in enumerate(pack(split_sections(doc), max_chars))
    ]


def main() -> None:
    total = 0
    too_long = 0
    for doc in load_all():
        chunks = chunk_document(doc)
        sizes = [len(chunk.text) for chunk in chunks]
        total += len(chunks)
        too_long += sum(1 for size in sizes if size > MAX_CHARS + TINY_CHARS)
        average = sum(sizes) // len(sizes) if sizes else 0
        biggest = max(sizes, default=0)
        print(f"{doc.doc_id:<12} {len(chunks):>4} chunks  avg {average:>5}  max {biggest:>5}")
    print(f"Total: {total} chunks, {too_long} over {MAX_CHARS + TINY_CHARS} characters")


if __name__ == "__main__":
    main()
