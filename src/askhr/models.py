"""The data model shared by the loader, the chunker, contextual notes and retrieval.

    ManifestEntry ──loader──▶ Document (pages) ──chunker──▶ Chunk ──step 6──▶ Chunk + context

All types are frozen: once created they never change. To "change" one, make a copy
with dataclasses.replace(obj, field=new_value).
"""

import hashlib
from dataclasses import dataclass
from typing import Literal, cast, get_args

from askhr.connectors.viamonde_site import ManifestEntry

DocType = Literal["politique", "directive", "ressource"]
Language = Literal["fr", "en"]


def _check_choice(value: str, allowed: object, field_name: str) -> None:
    """Raise ValueError if value is not one of the Literal's allowed values."""
    choices = get_args(allowed)
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {choices}, got {value!r}")


@dataclass(frozen=True)
class Page:
    """The text of one PDF page. Page numbers start at 1, like in a PDF viewer."""

    number: int
    text: str

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"page number must be >= 1, got {self.number}")


@dataclass(frozen=True)
class Document:
    """One source document (a PDF) with its metadata and its text, page by page."""

    doc_id: str
    title: str
    doc_type: DocType
    section: str
    language: Language
    url: str
    pages: tuple[Page, ...]

    def __post_init__(self) -> None:
        # Type hints aren't checked at runtime, so check the values coming from YAML here.
        _check_choice(self.doc_type, DocType, "doc_type")
        _check_choice(self.language, Language, "language")

    @classmethod
    def from_entry(cls, entry: ManifestEntry, pages: tuple[Page, ...]) -> "Document":
        """Build a Document from a manifest entry and the pages read from its PDF."""
        return cls(
            doc_id=entry.doc_id,
            title=entry.title,
            doc_type=cast(DocType, entry.doc_type),  # validated in __post_init__
            section=entry.section,
            language=cast(Language, entry.language),
            url=entry.url,
            pages=pages,
        )

    @property
    def text(self) -> str:
        """All pages joined, with a blank line between pages."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def content_hash(self) -> str:
        """A fingerprint of the text: it changes when the document's text changes."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, small enough to embed and to quote in an answer.

    It carries its document's metadata (denormalized) so a retrieved chunk can be
    cited without looking its document up again.
    """

    doc_id: str
    index: int  # position of the chunk in its document, from 0
    text: str
    page_start: int
    page_end: int
    title: str
    doc_type: DocType
    language: Language
    url: str
    heading: str = ""  # the nearest heading above the chunk, e.g. "4. Procédure"
    context: str = ""  # contextual note, filled in at M1 step 6

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("chunk text is empty")
        if self.index < 0:
            raise ValueError(f"chunk index must be >= 0, got {self.index}")
        if self.page_start < 1:
            raise ValueError(f"page_start must be >= 1, got {self.page_start}")
        if self.page_end < self.page_start:
            raise ValueError(f"page_end {self.page_end} is before page_start {self.page_start}")
        _check_choice(self.doc_type, DocType, "doc_type")
        _check_choice(self.language, Language, "language")

    @property
    def chunk_id(self) -> str:
        """A stable, unique id such as "5,202a#3"."""
        return f"{self.doc_id}#{self.index}"
