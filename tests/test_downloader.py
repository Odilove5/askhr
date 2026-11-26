from pathlib import Path

import httpx
import pytest

from askhr.connectors.downloader import download, load_manifest, target_path
from askhr.connectors.viamonde_site import ManifestEntry, write_manifest

ENTRY = ManifestEntry(
    doc_id="5,202a",
    title="Gestion de la confidentialité des dossiers médicaux",
    doc_type="directive",
    section="Milieu de travail",
    language="fr",
    url="https://csviamonde.ca/fileadmin/5_202a.pdf",
)


def fake_client(content: bytes, status: int = 200) -> httpx.Client:
    """An httpx client whose 'server' always answers with the given content."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_target_path_uses_doc_id_without_comma(tmp_path: Path) -> None:
    assert target_path(ENTRY, tmp_path) == tmp_path / "5_202a.pdf"


def test_load_manifest_round_trip(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    write_manifest([ENTRY], manifest)
    assert load_manifest(manifest) == [ENTRY]


def test_download_writes_the_pdf(tmp_path: Path) -> None:
    with fake_client(b"%PDF-1.7 fake") as client:
        assert download(ENTRY, tmp_path, client) is True
    assert (tmp_path / "5_202a.pdf").read_bytes() == b"%PDF-1.7 fake"


def test_download_skips_existing_file_without_calling_the_server(tmp_path: Path) -> None:
    (tmp_path / "5_202a.pdf").write_bytes(b"%PDF already here")

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("the server should not be called for an existing file")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert download(ENTRY, tmp_path, client) is False


def test_download_rejects_html_and_leaves_no_file(tmp_path: Path) -> None:
    with fake_client(b"<html>Page introuvable</html>") as client, pytest.raises(ValueError):
        download(ENTRY, tmp_path, client)
    assert list(tmp_path.iterdir()) == []


def test_download_raises_on_http_error(tmp_path: Path) -> None:
    with fake_client(b"", status=404) as client, pytest.raises(httpx.HTTPStatusError):
        download(ENTRY, tmp_path, client)
    assert list(tmp_path.iterdir()) == []
