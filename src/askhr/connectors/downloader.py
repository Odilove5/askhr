"""Download the PDFs listed in corpus/manifest.yaml into data/raw/.

Idempotent: files already on disk are skipped, so running it twice does no extra work.
One broken link is reported and skipped; it never stops the whole run.
"""

import time
from pathlib import Path

import httpx
import yaml

from askhr.connectors.viamonde_site import ManifestEntry

MANIFEST_PATH = Path("corpus/manifest.yaml")
RAW_DIR = Path("data/raw")
PDF_MAGIC = b"%PDF"  # every PDF file starts with these bytes
DELAY_SECONDS = 0.5  # pause between downloads, to be polite to the server


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Read the manifest YAML and turn each row back into a ManifestEntry."""
    with manifest_path.open(encoding="utf-8") as f:
        rows = yaml.safe_load(f) or []  # an empty file loads as None
    return [ManifestEntry(**row) for row in rows]


def target_path(entry: ManifestEntry, raw_dir: Path) -> Path:
    """Where the entry's PDF is stored: "5,202a" -> raw_dir/5_202a.pdf.

    Only the doc_id is used: titles can change on the site and may contain accents,
    which would break the "already downloaded?" check.
    """
    return raw_dir / f"{entry.doc_id.replace(',', '_')}.pdf"


def download(entry: ManifestEntry, raw_dir: Path, client: httpx.Client) -> bool:
    """Download one PDF. Return True if downloaded, False if it was already on disk.

    Raises httpx.HTTPError for network or HTTP errors, ValueError if the response
    is not a PDF. Either way no file is left behind.
    """
    target = target_path(entry, raw_dir)
    if target.exists():
        return False

    response = client.get(entry.url)
    response.raise_for_status()
    if not response.content.startswith(PDF_MAGIC):
        raise ValueError(f"{entry.url} did not return a PDF")

    # Write to a temporary name first, then rename: a crash mid-write never leaves
    # a half-written file that a later run would wrongly skip.
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(response.content)
    partial.replace(target)
    return True


def main() -> None:
    """Download every PDF in the manifest and print a summary."""
    entries = load_manifest(MANIFEST_PATH)
    downloaded = skipped = 0
    failed: list[str] = []

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for entry in entries:
            try:
                if download(entry, RAW_DIR, client):
                    downloaded += 1
                    print(f"Downloaded {entry.doc_id}: {entry.title}")
                    time.sleep(DELAY_SECONDS)
                else:
                    skipped += 1
            except (httpx.HTTPError, ValueError) as err:
                failed.append(entry.doc_id)
                print(f"FAILED {entry.doc_id}: {err}")

    print(f"Downloaded {downloaded}, skipped {skipped}, failed {len(failed)}")
    if failed:
        raise SystemExit(f"Failed documents: {', '.join(failed)}")


if __name__ == "__main__":
    main()
