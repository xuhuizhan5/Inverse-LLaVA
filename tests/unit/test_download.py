import hashlib
import json
import shutil
import stat
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from invllava.data.download import acquire_http_candidate, extract_verified_archive


class _ArtifactHandler(BaseHTTPRequestHandler):
    payload = b"candidate-artifact" * 1024
    etag = '"fixture-v1"'

    def do_GET(self) -> None:
        start = 0
        raw_range = self.headers.get("Range")
        if raw_range:
            start = int(raw_range.removeprefix("bytes=").removesuffix("-"))
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
        else:
            self.send_response(200)
        body = self.payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", self.etag)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


def test_candidate_acquisition_records_digest_and_resumes(tmp_path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ArtifactHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        target = tmp_path / "artifact.zip"
        partial = target.with_name(target.name + ".partial")
        transfer = target.with_name(target.name + ".partial.json")
        prefix = _ArtifactHandler.payload[:100]
        partial.write_bytes(prefix)
        transfer.write_text(
            json.dumps(
                {
                    "source_url": f"http://127.0.0.1:{server.server_port}/artifact.zip",
                    "etag": _ArtifactHandler.etag,
                }
            ),
            encoding="utf-8",
        )
        artifact, record_path = acquire_http_candidate(
            f"http://127.0.0.1:{server.server_port}/artifact.zip",
            target,
            minimum_free_bytes_after=0,
            allow_insecure_http=True,
        )
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
    assert artifact.read_bytes() == _ArtifactHandler.payload
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["sha256"] == hashlib.sha256(_ArtifactHandler.payload).hexdigest()
    assert record["size_bytes"] == len(_ArtifactHandler.payload)
    assert not partial.exists()
    assert not transfer.exists()
    same_artifact, same_record = acquire_http_candidate(
        f"http://127.0.0.1:{server.server_port}/artifact.zip",
        target,
        minimum_free_bytes_after=0,
        allow_insecure_http=True,
    )
    assert same_artifact == artifact
    assert same_record == record_path


def test_completed_candidate_is_rejected_when_its_bytes_change(tmp_path) -> None:
    target = tmp_path / "artifact.zip"
    target.write_bytes(b"changed-after-acquisition")
    record = target.with_name(target.name + ".candidate.json")
    record.write_text(
        json.dumps(
            {
                "source_url": "https://example.invalid/artifact.zip",
                "size_bytes": target.stat().st_size,
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        acquire_http_candidate(
            "https://example.invalid/artifact.zip",
            target,
            minimum_free_bytes_after=0,
        )


def test_verified_archive_extracts_atomically(tmp_path) -> None:
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("nested/value.txt", "fixture")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "extracted"
    extract_verified_archive(archive, destination, sha256=digest)
    assert (destination / "nested/value.txt").read_text() == "fixture"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o755


def test_archive_path_traversal_is_rejected_without_exposing_destination(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "unsafe")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="escapes destination"):
        extract_verified_archive(archive, destination, sha256=digest)
    assert not destination.exists()


def test_archive_extraction_preserves_requested_free_space(tmp_path) -> None:
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("value.txt", "fixture")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "extracted"
    impossible_reserve = shutil.disk_usage(tmp_path).free + 1
    with pytest.raises(OSError, match="reserved headroom"):
        extract_verified_archive(
            archive,
            destination,
            sha256=digest,
            minimum_free_bytes_after=impossible_reserve,
        )
    assert not destination.exists()


def test_archive_duplicate_targets_are_rejected(tmp_path) -> None:
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("same.txt", "first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            stream.writestr("same.txt", "second")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="duplicate target"):
        extract_verified_archive(archive, destination, sha256=digest)
    assert not destination.exists()
