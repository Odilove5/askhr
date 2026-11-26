"""Connector for the Viamonde policies and administrative directives web page.

The page lists every policy in HTML tables grouped under <h2> section headings:

    <h2>Conditions de travail</h2>
    <table class="contenttable">
      <tr>
        <td>5,100</td>
        <td><a href="...politique.pdf">Title</a></td>
        <td><a ...>Directive</a></td>
      </tr>

This module turns that page into a manifest: one entry per PDF.
"""

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin

import httpx
import yaml
from bs4 import BeautifulSoup, Tag

from askhr.text import normalize_whitespace

INDEX_URL = (
    "https://csviamonde.ca/administration-et-gouvernance/politiques-et-directives-administratives"
)
HR_PREFIX = "5,"  # Human resources policies are numbered 5,xxx
MIN_EXPECTED = 30  # Fewer HR documents than this means the page layout probably changed

# "5,202a Gestion des absences" or "5,401 a" -> captures "5,202a" / "5,401 a"
_CODE_IN_TEXT = re.compile(r"^(\d,\d{3}\s?[a-z])\b\s*")


@dataclass(frozen=True)
class ManifestEntry:
    """One document (PDF) listed on the Viamonde policies page."""

    doc_id: str
    title: str
    doc_type: str  # "politique", "directive" or "ressource"
    section: str
    language: str
    url: str


def _clean_text(element: Tag) -> str:
    """Return the visible text of a link, without the file-size and icon spans."""
    for span in element.find_all("span"):
        span.decompose()
    return normalize_whitespace(element.get_text(" "))


def _is_pdf(href: str) -> bool:
    return href.lower().endswith(".pdf")


def _parse_related(
    cell: Tag, number: str, policy_title: str, section: str, base_url: str
) -> list[ManifestEntry]:
    """Parse the 'Ressources associées' cell: directives and other resources."""
    entries: list[ManifestEntry] = []
    for index, link in enumerate(cell.find_all("a", href=True), start=1):
        href = str(link["href"])
        if not _is_pdf(href):
            continue
        # The code can sit inside the link text ("5,202a ...") or just before it ("5,401 a").
        previous = link.previous_sibling
        prefix = normalize_whitespace(previous) if isinstance(previous, str) else ""
        text = _clean_text(link)
        is_directive = "dir" in href.lower() or text.lower().startswith("directive")

        match = _CODE_IN_TEXT.match(text) or _CODE_IN_TEXT.match(prefix)
        if match:
            doc_id = match.group(1).replace(" ", "")
            text = _CODE_IN_TEXT.sub("", text) or text
        elif is_directive:
            doc_id = f"{number}-dir"
        else:
            doc_id = f"{number}-res{index}"

        # "Directives administratives - Personnel d'entretien"
        #   -> "<policy title> – Personnel d'entretien"
        if text.lower().startswith("directives administratives"):
            extra = text[len("directives administratives") :].strip(" -–")
            text = f"{policy_title} – {extra}" if extra else policy_title

        entries.append(
            ManifestEntry(
                doc_id=doc_id,
                title=text,
                doc_type="directive" if is_directive else "ressource",
                section=section,
                language="fr",
                url=urljoin(base_url, href),
            )
        )
    return entries


def parse_index(html: str, base_url: str) -> list[ManifestEntry]:
    """Parse the policies page HTML into manifest entries, one per PDF.

    Args:
        html: The HTML of the policies page.
        base_url: The page URL, used to turn relative links into absolute ones.
    """
    soup = BeautifulSoup(html, "html.parser")
    entries: list[ManifestEntry] = []
    for table in soup.select("table.contenttable"):
        heading = table.find_previous("h2")
        section = normalize_whitespace(heading.get_text(" ")) if heading else ""
        for row in table.select("tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            number = normalize_whitespace(cells[0].get_text(" "))
            policy_link = cells[1].find("a", href=True)
            if policy_link is None or not _is_pdf(str(policy_link["href"])):
                continue
            policy_title = _clean_text(policy_link)
            entries.append(
                ManifestEntry(
                    doc_id=number,
                    title=policy_title,
                    doc_type="politique",
                    section=section,
                    language="fr",
                    url=urljoin(base_url, str(policy_link["href"])),
                )
            )
            if len(cells) > 2:
                entries.extend(_parse_related(cells[2], number, policy_title, section, base_url))
    return entries


def fetch_index(url: str = INDEX_URL) -> str:
    """Download the policies page and return its HTML."""
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    return response.text


def write_manifest(entries: list[ManifestEntry], path: Path) -> None:
    """Write the entries to a YAML file, keeping accents readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump([asdict(e) for e in entries], f, allow_unicode=True, sort_keys=False)


def main() -> None:
    """Fetch the page, keep the HR documents (5,xxx) and write corpus/manifest.yaml."""
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus/manifest.yaml")
    entries = [e for e in parse_index(fetch_index(), INDEX_URL) if e.doc_id.startswith(HR_PREFIX)]
    if len(entries) < MIN_EXPECTED:
        raise SystemExit(
            f"Expected at least {MIN_EXPECTED} HR documents but found {len(entries)}; "
            "the page layout may have changed. Manifest not written."
        )
    write_manifest(entries, output)
    print(f"Wrote {len(entries)} entries to {output}")


if __name__ == "__main__":
    main()
