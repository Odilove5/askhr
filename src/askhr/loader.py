"""Turn the downloaded Viamonde PDFs into Document objects with clean page text.

Every page of a Viamonde policy or directive starts with the same header block
(document number, section, title, "Page x de y" and the file name). We cut that
header so it doesn't end up in every chunk.

Run with:  uv run python -m askhr.loader
"""

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from askhr.connectors.downloader import MANIFEST_PATH, RAW_DIR, load_manifest, target_path
from askhr.connectors.viamonde_site import ManifestEntry
from askhr.models import Document, Page
from askhr.text import normalize_whitespace

MAX_HEADER_LINES = 20  # The header is always near the top; never cut deeper than this

# The header block ends with one of these lines, in any order depending on the PDF:
#   the page number, alone or at the end of a line: "Page 2 de 3", "… 2027-2028 Page 1 de 17"
PAGE_NUMBER = re.compile(r"(?:^|\s)Page \d+ (?:de|sur) \d+$")
#   the file name: "5_202a_Gestion…", "5,201_Code d'éthique_politique", "5_300 Conditions_de_…"
#   (a policy number, then text containing at least one more underscore)
FILE_NAME_START = re.compile(r"\d[_,]\d{3}[a-z]?\S*[ _].*_")
#   the end of a file name that wrapped onto a second line: "… non syndiqués_dir_adm"
FILE_NAME_END = re.compile(r"(?:_dir_adm|_politique|dir\. ?adm)\.?$")


def is_header_end(line: str) -> bool:
    """True for the lines that close a page header: the page number or the file name."""
    line = line.strip()
    return bool(
        PAGE_NUMBER.search(line) or FILE_NAME_START.match(line) or FILE_NAME_END.search(line)
    )


def clean_page(text: str) -> str:
    """Remove the repeated page header and tidy the whitespace, keeping line breaks."""
    text = unicodedata.normalize("NFC", text)
    lines = text.splitlines()

    # Cut after the LAST header-closing line near the top. Some PDFs put the file name
    # first and "Page x de y" last, others the reverse, and some have no file name at all.
    ends = [i for i, line in enumerate(lines[:MAX_HEADER_LINES]) if is_header_end(line)]
    if ends:
        lines = lines[ends[-1] + 1 :]
    # No header line found: unknown layout, so we keep the whole page rather than lose text.

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
        Page(number=number, text=clean_page(text))
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
