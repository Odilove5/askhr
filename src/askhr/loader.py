"""Turn the downloaded Viamonde PDFs into Document objects with clean page text.

Every page of a Viamonde policy or directive starts with the same header block
(document number, section, title, "Page x de y" and the file name). We cut that
header so it doesn't end up in every chunk.

Run with:  uv run python -m askhr.loader
"""

import unicodedata
from pathlib import Path

from pypdf import PdfReader

from askhr.connectors.downloader import MANIFEST_PATH, RAW_DIR, load_manifest, target_path
from askhr.connectors.viamonde_site import ManifestEntry
from askhr.models import Document, Page
from askhr.text import normalize_whitespace

MAX_HEADER_LINES = 20  # The header is always near the top; never cut deeper than this


def header_marker(doc_id: str) -> str:
    """Return how the file-name line of the page header starts.

    The file name uses the policy number without any suffix:
    "5,202a" -> "5_202a_", "5,203-dir" -> "5_203_", "5,400-res3" -> "5_400_".
    """
    number = doc_id.split("-")[0]
    return number.replace(",", "_") + "_"


def clean_page(text: str, doc_id: str) -> str:
    """Remove the repeated page header and tidy the whitespace, keeping line breaks."""
    text = unicodedata.normalize("NFC", text)
    lines = text.splitlines()

    marker = header_marker(doc_id)
    for i, line in enumerate(lines[:MAX_HEADER_LINES]):
        if line.strip().startswith(marker):
            lines = lines[i + 1 :]  # Drop the header and the file-name line itself
            break
    # No marker found: unknown layout, so we keep the whole page rather than lose text.

    tidy = (normalize_whitespace(line) for line in lines)
    return "\n".join(line for line in tidy if line)


def extract_pages(path: Path) -> list[str]:
    """Read the raw text of every page in a PDF (an empty string for a page without text)."""
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def load_document(entry: ManifestEntry, raw_dir: Path) -> Document:
    """Load one downloaded PDF and return it as a Document with cleaned pages."""
    path = target_path(entry, raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run first: uv run python -m askhr.connectors.downloader"
        )
    pages = tuple(
        Page(number=number, text=clean_page(text, entry.doc_id))
        for number, text in enumerate(extract_pages(path), start=1)
    )
    return Document.from_entry(entry, pages)


def load_all(manifest_path: Path = MANIFEST_PATH, raw_dir: Path = RAW_DIR) -> list[Document]:
    """Load every document listed in the manifest."""
    return [load_document(entry, raw_dir) for entry in load_manifest(manifest_path)]


def main() -> None:
    documents = load_all()
    empty_pages = 0
    for doc in documents:
        empty = sum(1 for page in doc.pages if not page.text)
        empty_pages += empty
        note = f"  ({empty} empty)" if empty else ""
        print(f"{doc.doc_id:<12} {len(doc.pages):>3} pages {len(doc.text):>7} chars{note}")
    total_pages = sum(len(doc.pages) for doc in documents)
    print(f"Loaded {len(documents)} documents, {total_pages} pages, {empty_pages} empty pages")


if __name__ == "__main__":
    main()
