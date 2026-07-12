import requests

from src.app import checkpoints


class _FakeResponse:
    def __init__(self, content: bytes):
        self._content = content

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self._content


def test_fetch_checkpoints_skips_files_already_present(tmp_path, monkeypatch):
    for filename in checkpoints.CHECKPOINT_FILES:
        (tmp_path / filename).write_bytes(b"already here")

    def _unexpected_get(*args, **kwargs):
        raise AssertionError("should not hit the network when every file already exists")

    monkeypatch.setattr(checkpoints.requests, "get", _unexpected_get)

    downloaded = checkpoints.fetch_checkpoints(dest_dir=str(tmp_path))

    assert downloaded == []


def test_fetch_checkpoints_downloads_missing_files(tmp_path, monkeypatch):
    requested_urls = []

    def _fake_get(url, stream, timeout):
        requested_urls.append(url)
        return _FakeResponse(b"fake weights")

    monkeypatch.setattr(checkpoints.requests, "get", _fake_get)

    downloaded = checkpoints.fetch_checkpoints(repo="owner/repo", tag="vX", dest_dir=str(tmp_path))

    assert sorted(downloaded) == sorted(checkpoints.CHECKPOINT_FILES)
    for filename in checkpoints.CHECKPOINT_FILES:
        assert (tmp_path / filename).read_bytes() == b"fake weights"
        assert f"https://github.com/owner/repo/releases/download/vX/{filename}" in requested_urls


def test_fetch_checkpoints_only_downloads_missing_ones(tmp_path, monkeypatch):
    present = checkpoints.CHECKPOINT_FILES[0]
    (tmp_path / present).write_bytes(b"already here")

    def _fake_get(url, stream, timeout):
        return _FakeResponse(b"fake weights")

    monkeypatch.setattr(checkpoints.requests, "get", _fake_get)

    downloaded = checkpoints.fetch_checkpoints(dest_dir=str(tmp_path))

    assert present not in downloaded
    assert (tmp_path / present).read_bytes() == b"already here"


def test_fetch_checkpoints_does_not_raise_when_network_fails(tmp_path, monkeypatch):
    """A failed fetch (offline, release not published yet) must not crash
    app startup -- the rest of the pipeline already treats a missing
    checkpoint as a graceful fallback, not a hard error.
    """

    def _failing_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(checkpoints.requests, "get", _failing_get)

    downloaded = checkpoints.fetch_checkpoints(dest_dir=str(tmp_path))

    assert downloaded == []
    assert list(tmp_path.iterdir()) == []  # no partial .part files left behind
