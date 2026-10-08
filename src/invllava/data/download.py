from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file, verify_sha256
from invllava.config.schema import DataSource

_CONTENT_RANGE = re.compile(r"bytes (?P<start>[0-9]+)-(?P<end>[0-9]+)/(?P<total>[0-9]+)")


@dataclass(frozen=True)
class CandidateArtifact:
    """Evidence from first acquisition, before a digest is trusted by a config."""

    source_url: str
    resolved_url: str
    path: str
    size_bytes: int
    sha256: str
    etag: str | None
    last_modified: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def acquire_http_candidate(
    url: str,
    destination: str | Path,
    *,
    minimum_free_bytes_after: int = 10 * 1024**3,
    allow_insecure_http: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, Path]:
    """Acquire an unfrozen HTTPS artifact and emit a digest sidecar.

    This is deliberately separate from :func:`download_http`: a candidate is
    evidence used to freeze a source, never an input authorized for scientific
    execution. Interrupted transfers resume only when provider validators are
    available; otherwise the partial file is restarted from byte zero.
    """

    if minimum_free_bytes_after < 0:
        raise ValueError("minimum free bytes must be non-negative")
    scheme = urlparse(url).scheme.lower()
    if scheme != "https" and not (allow_insecure_http and scheme == "http"):
        raise ValueError("candidate acquisition requires HTTPS")
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    transfer_record = target.with_name(target.name + ".partial.json")
    final_record = target.with_name(target.name + ".candidate.json")
    if target.exists():
        if not target.is_file():
            raise FileExistsError(target)
        if not final_record.exists() and transfer_record.is_file() and not partial.exists():
            interrupted = json.loads(transfer_record.read_text(encoding="utf-8"))
            if (
                not isinstance(interrupted, dict)
                or interrupted.get("source_url") != url
                or interrupted.get("expected_size_bytes") != target.stat().st_size
            ):
                raise RuntimeError("interrupted finalization record does not match candidate")
            artifact = CandidateArtifact(
                source_url=url,
                resolved_url=str(interrupted["resolved_url"]),
                path=str(target),
                size_bytes=target.stat().st_size,
                sha256=sha256_file(target),
                etag=interrupted.get("etag"),
                last_modified=interrupted.get("last_modified"),
            )
            atomic_write_json(final_record, {"schema_version": 1, **artifact.to_dict()})
            transfer_record.unlink()
            return target, final_record
        if not final_record.is_file():
            raise RuntimeError("completed candidate is missing its final evidence record")
        record = json.loads(final_record.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or record.get("source_url") != url
            or record.get("size_bytes") != target.stat().st_size
            or record.get("sha256") != sha256_file(target)
        ):
            raise RuntimeError("completed candidate does not match its evidence record")
        return target, final_record
    if final_record.exists():
        raise RuntimeError("candidate evidence record exists without its artifact")

    previous: dict[str, Any] | None = None
    if partial.exists():
        if not partial.is_file() or not transfer_record.is_file():
            raise RuntimeError("partial candidate is missing its transfer record")
        value = json.loads(transfer_record.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("source_url") != url:
            raise RuntimeError("partial candidate belongs to a different source URL")
        previous = value
    elif transfer_record.exists():
        raise RuntimeError("candidate transfer record exists without its partial file")

    offset = partial.stat().st_size if partial.exists() else 0
    headers: dict[str, str] = {}
    if offset:
        validator = (
            None if previous is None else previous.get("etag") or previous.get("last_modified")
        )
        if validator:
            headers = {"Range": f"bytes={offset}-", "If-Range": str(validator)}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = response.status
        resolved_url = response.geturl()
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        append = offset > 0 and bool(headers) and status == 206
        content_length = response.headers.get("Content-Length")
        if content_length is None:
            raise RuntimeError("HTTP source did not declare Content-Length; cannot enforce budget")
        incoming = int(content_length)
        if append:
            content_range = response.headers.get("Content-Range")
            match = _CONTENT_RANGE.fullmatch(content_range or "")
            if match is None or int(match.group("start")) != offset:
                raise RuntimeError("provider returned an invalid Content-Range for resume")
            total_bytes = int(match.group("total"))
            if incoming != total_bytes - offset:
                raise RuntimeError("provider range length does not match Content-Range")
            assert previous is not None
            previous_etag = previous.get("etag")
            previous_modified = previous.get("last_modified")
            if previous_etag and etag != previous_etag:
                raise RuntimeError("provider changed the ETag during a resumed transfer")
            if not previous_etag and previous_modified and last_modified != previous_modified:
                raise RuntimeError("provider changed Last-Modified during a resumed transfer")
            previous_size = previous.get("expected_size_bytes")
            if previous_size is not None and int(previous_size) != total_bytes:
                raise RuntimeError("provider changed the artifact size during a resumed transfer")
        else:
            offset = 0
            total_bytes = incoming
        free = shutil.disk_usage(target.parent).free
        if incoming + minimum_free_bytes_after > free:
            raise OSError(
                f"download needs {incoming} bytes plus {minimum_free_bytes_after} bytes "
                f"reserved headroom, but only {free} bytes are free"
            )
        atomic_write_json(
            transfer_record,
            {
                "schema_version": 1,
                "source_url": url,
                "resolved_url": resolved_url,
                "etag": etag,
                "last_modified": last_modified,
                "expected_size_bytes": total_bytes,
            },
        )
        with partial.open("ab" if append else "wb") as output:
            completed = offset
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                completed += len(chunk)
                if progress is not None:
                    progress(completed, total_bytes)
            output.flush()
            os.fsync(output.fileno())

    if partial.stat().st_size != total_bytes:
        raise RuntimeError(f"candidate has {partial.stat().st_size} bytes, expected {total_bytes}")
    partial.replace(target)
    artifact = CandidateArtifact(
        source_url=url,
        resolved_url=resolved_url,
        path=str(target),
        size_bytes=target.stat().st_size,
        sha256=sha256_file(target),
        etag=etag,
        last_modified=last_modified,
    )
    atomic_write_json(final_record, {"schema_version": 1, **artifact.to_dict()})
    transfer_record.unlink()
    return target, final_record


def download_http(
    url: str,
    destination: str | Path,
    *,
    sha256: str,
    minimum_free_bytes_after: int = 10 * 1024**3,
) -> Path:
    """Download to `.partial`, resume when supported, and require a digest."""

    if minimum_free_bytes_after < 0:
        raise ValueError("minimum free bytes must be non-negative")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(destination)
        verify_sha256(destination, sha256)
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.is_file():
        try:
            verify_sha256(partial, sha256)
        except ValueError:
            pass
        else:
            partial.replace(destination)
            return destination
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and response.status == 206
        content_length = response.headers.get("Content-Length")
        if content_length is None:
            raise RuntimeError("HTTP source did not declare Content-Length; cannot enforce budget")
        incoming = int(content_length)
        free = shutil.disk_usage(destination.parent).free
        if incoming + minimum_free_bytes_after > free:
            raise OSError(
                f"download needs {incoming} bytes plus {minimum_free_bytes_after} bytes "
                f"reserved headroom, but only {free} bytes are free"
            )
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    verify_sha256(partial, sha256)
    partial.replace(destination)
    return destination


def download_huggingface(
    source: DataSource,
    destination: str | Path,
    *,
    minimum_free_bytes_after: int = 10 * 1024**3,
) -> Path:
    if minimum_free_bytes_after < 0:
        raise ValueError("minimum free bytes must be non-negative")
    if source.kind != "huggingface":
        raise ValueError("source is not a Hugging Face source")
    if not source.revision or source.revision == "pending-freeze":
        raise ValueError(f"freeze an immutable revision before downloading {source.id}")
    try:
        repository, filename = source.location.rsplit(":", 1)
    except ValueError as error:
        raise ValueError("Hugging Face location must be 'repository:filename'") from error
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / filename
    if candidate.is_file() and source.sha256:
        verify_sha256(candidate, source.sha256)
        return candidate
    metadata = get_hf_file_metadata(
        hf_hub_url(
            repo_id=repository,
            filename=filename,
            repo_type=source.repo_type,
            revision=source.revision,
        )
    )
    expected_size = metadata.size
    if expected_size is None:
        raise RuntimeError(f"provider did not report a size for {source.id}")
    free = shutil.disk_usage(root).free
    if expected_size + minimum_free_bytes_after > free:
        raise OSError(
            f"{source.id} needs {expected_size} bytes plus {minimum_free_bytes_after} bytes "
            f"reserved headroom, but only {free} bytes are free"
        )
    downloaded = Path(
        hf_hub_download(
            repo_id=repository,
            filename=filename,
            revision=source.revision,
            repo_type=source.repo_type,
            local_dir=root,
        )
    )
    if source.sha256:
        verify_sha256(downloaded, source.sha256)
    return downloaded


def _safe_member(root: Path, member: str) -> Path:
    # ``Path.resolve`` performs a filesystem lookup for every path component.
    # Large image archives contain hundreds of thousands of members, and those
    # lookups are prohibitively expensive on network volumes.  ``abspath``
    # supplies the containment property needed here without touching the
    # filesystem.  Verified extraction always targets a new private directory,
    # and archive symlink entries are rejected below.
    target = Path(os.path.abspath(root / member))
    try:
        contained = os.path.commonpath((root, target)) == str(root)
    except ValueError:
        contained = False
    if target == root or not contained:
        raise ValueError(f"archive path escapes destination: {member}")
    return target


def safe_extract_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    minimum_free_bytes_after: int = 0,
) -> None:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        targets: set[Path] = set()
        for info in source.infolist():
            target = _safe_member(root, info.filename)
            if target in targets:
                raise ValueError(f"archive contains a duplicate target: {info.filename}")
            targets.add(target)
            mode = (info.external_attr >> 16) & 0o170000
            if mode and mode not in {stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError(f"archive special file is not accepted: {info.filename}")
        _require_extraction_capacity(
            root,
            sum(info.file_size for info in source.infolist()),
            minimum_free_bytes_after=minimum_free_bytes_after,
        )
        source.extractall(root)


def safe_extract_tar(
    archive: str | Path,
    destination: str | Path,
    *,
    minimum_free_bytes_after: int = 0,
) -> None:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        targets: set[Path] = set()
        for info in source.getmembers():
            target = _safe_member(root, info.name)
            if target in targets:
                raise ValueError(f"archive contains a duplicate target: {info.name}")
            targets.add(target)
            if info.issym() or info.islnk():
                raise ValueError(f"archive links are not accepted: {info.name}")
            if not (info.isfile() or info.isdir()):
                raise ValueError(f"archive special file is not accepted: {info.name}")
        _require_extraction_capacity(
            root,
            sum(info.size for info in source.getmembers()),
            minimum_free_bytes_after=minimum_free_bytes_after,
        )
        source.extractall(root)


def _require_extraction_capacity(
    root: Path,
    uncompressed_bytes: int,
    *,
    minimum_free_bytes_after: int = 0,
) -> None:
    if minimum_free_bytes_after < 0:
        raise ValueError("minimum free bytes must be non-negative")
    free = shutil.disk_usage(root).free
    if uncompressed_bytes + minimum_free_bytes_after > free:
        raise OSError(
            f"archive needs {uncompressed_bytes} uncompressed bytes plus "
            f"{minimum_free_bytes_after} bytes reserved headroom, but only {free} bytes "
            f"are free under {root}"
        )


def extract_verified_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    sha256: str,
    minimum_free_bytes_after: int = 0,
) -> Path:
    """Verify and atomically expose a ZIP or TAR archive extraction."""

    source = Path(archive).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    verify_sha256(source, sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        if zipfile.is_zipfile(source):
            safe_extract_zip(
                source,
                temporary,
                minimum_free_bytes_after=minimum_free_bytes_after,
            )
        elif tarfile.is_tarfile(source):
            safe_extract_tar(
                source,
                temporary,
                minimum_free_bytes_after=minimum_free_bytes_after,
            )
        else:
            raise ValueError(f"unsupported archive format: {source}")
        # ``mkdtemp`` deliberately creates a private (0700) directory.  That
        # mode must not leak into the published component root: acquisition
        # and training commonly run in containers with different UIDs.  Keep
        # member permissions intact and normalize only the directory we own.
        temporary.chmod(0o755)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
