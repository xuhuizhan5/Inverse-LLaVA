from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageFile

from invllava.data.types import ConversationSample


@dataclass(frozen=True)
class DataAudit:
    samples: int
    images: int
    missing_images: int
    duplicate_ids: int
    turns_by_role: dict[str, int]
    samples_by_source: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnnotationAudit:
    samples: int
    image_references: int
    text_only_samples: int
    duplicate_ids: int
    turns_by_role: dict[str, int]
    samples_by_source: dict[str, int]
    ordered_sample_ids_sha256: str
    ordered_image_paths_sha256: str
    first_sample_ids: tuple[str, ...]
    last_sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_annotation(samples: Iterable[ConversationSample]) -> AnnotationAudit:
    """Stream a complete annotation while retaining only IDs and aggregate counts."""

    seen: set[str] = set()
    duplicate_ids = 0
    sample_count = 0
    image_references = 0
    text_only = 0
    roles: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    id_digest = hashlib.sha256()
    image_digest = hashlib.sha256()
    first_ids: list[str] = []
    last_ids: list[str] = []
    for sample in samples:
        sample.validate()
        sample_count += 1
        if sample.id in seen:
            duplicate_ids += 1
        else:
            seen.add(sample.id)
        if len(first_ids) < 10:
            first_ids.append(sample.id)
        last_ids.append(sample.id)
        if len(last_ids) > 10:
            last_ids.pop(0)
        id_digest.update(sample.id.encode("utf-8"))
        id_digest.update(b"\n")
        image_references += len(sample.images)
        text_only += int(not sample.images)
        for path in sample.images:
            image_digest.update(str(path).encode("utf-8"))
            image_digest.update(b"\n")
        roles.update(turn.role for turn in sample.turns)
        sources[sample.source] += 1
    return AnnotationAudit(
        samples=sample_count,
        image_references=image_references,
        text_only_samples=text_only,
        duplicate_ids=duplicate_ids,
        turns_by_role=dict(sorted(roles.items())),
        samples_by_source=dict(sorted(sources.items())),
        ordered_sample_ids_sha256=id_digest.hexdigest(),
        ordered_image_paths_sha256=image_digest.hexdigest(),
        first_sample_ids=tuple(first_ids),
        last_sample_ids=tuple(last_ids),
    )


def audit_samples(samples: list[ConversationSample], *, check_files: bool = True) -> DataAudit:
    counts = Counter(sample.id for sample in samples)
    images = [path for sample in samples for path in sample.images]
    roles = Counter(turn.role for sample in samples for turn in sample.turns)
    image_counts = Counter(images)
    missing_images = (
        sum(references for path, references in image_counts.items() if not Path(path).is_file())
        if check_files
        else 0
    )
    return DataAudit(
        samples=len(samples),
        images=len(images),
        missing_images=missing_images,
        duplicate_ids=sum(count - 1 for count in counts.values() if count > 1),
        turns_by_role=dict(sorted(roles.items())),
        samples_by_source=dict(sorted(Counter(sample.source for sample in samples).items())),
    )


@dataclass(frozen=True)
class ImageIntegrityAudit:
    references: int
    unique_images: int
    decoded_images: int
    missing_images: int
    corrupt_images: int
    zero_sized_images: int
    encoded_bytes: int
    formats: dict[str, int]
    sources: dict[str, int]
    inventory_sha256: str
    failure_details: tuple[dict[str, str], ...]
    failure_details_truncated: bool
    failures_sha256: str

    @property
    def passed(self) -> bool:
        return not (self.missing_images or self.corrupt_images or self.zero_sized_images)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True)
class ImageReferenceSet:
    """Minimal adapter for auditing images outside conversation datasets."""

    images: tuple[Path, ...]
    source: str


@dataclass(frozen=True)
class ImageInventoryRecord:
    """Stable encoded-byte and canonical-pixel identities for one decoded image."""

    logical_path: str
    source: str
    size_bytes: int
    sha256: str
    pixel_sha256: str
    width: int
    height: int
    format: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_rgb_sha256(image: Image.Image) -> str:
    """Hash decoded RGB pixels and dimensions, independent of image encoding."""

    rgb = image if image.mode == "RGB" else image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}x{rgb.height}\n".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _decode_image(
    item: tuple[str, str, str],
) -> tuple[str, str, str, str | None, int, str, str, int, int, str]:
    path_value, logical_path, source = item
    path = Path(path_value)
    try:
        if path.is_symlink():
            return path_value, logical_path, source, "symlink", 0, "", "", 0, 0, ""
        if not path.is_file():
            return path_value, logical_path, source, "missing", 0, "", "", 0, 0, ""
        stat_before = path.stat()
        size = stat_before.st_size
        if size == 0:
            return path_value, logical_path, source, "zero-sized", 0, "", "", 0, 0, ""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with Image.open(path) as image:
            image_format = image.format or "unknown"
            image.verify()
        # verify() checks structure but does not decode pixels. Reopen and load
        # so truncated/corrupt payloads fail before a paid training worker.
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                return (
                    path_value,
                    logical_path,
                    source,
                    "zero-dimension",
                    size,
                    digest.hexdigest(),
                    "",
                    image.width,
                    image.height,
                    image_format,
                )
            width, height = image.width, image.height
            # Exercise the conversion performed by common CLIP processors too.
            rgb = image.convert("RGB")
            rgb.load()
            pixel_digest = canonical_rgb_sha256(rgb)
        stat_after = path.stat()
        if (
            stat_after.st_size != stat_before.st_size
            or stat_after.st_mtime_ns != stat_before.st_mtime_ns
            or stat_after.st_ino != stat_before.st_ino
        ):
            raise RuntimeError("image changed while it was being audited")
    except Exception as error:
        return (
            path_value,
            logical_path,
            source,
            f"decode-error:{type(error).__name__}:{error}",
            0,
            "",
            "",
            0,
            0,
            "",
        )
    return (
        path_value,
        logical_path,
        source,
        None,
        size,
        digest.hexdigest(),
        pixel_digest,
        width,
        height,
        image_format,
    )


def audit_image_integrity(
    samples: Iterable[ConversationSample | ImageReferenceSet],
    *,
    sources: set[str] | None = None,
    image_root: str | Path | None = None,
    workers: int = 8,
    maximum_failure_details: int = 100,
    progress_every: int = 0,
    progress: Callable[[int, int], None] | None = None,
    inventory: Callable[[ImageInventoryRecord], None] | None = None,
) -> ImageIntegrityAudit:
    """Fully decode each unique referenced image, optionally one source at a time."""

    if workers <= 0 or maximum_failure_details < 0 or progress_every < 0:
        raise ValueError(
            "workers must be positive; failure-detail and progress intervals must be non-negative"
        )
    # Some data libraries enable this process-global Pillow escape hatch. An
    # integrity audit must never inherit that permissive setting: accepting a
    # partially decoded JPEG here would defer the failure to a paid worker.
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    root = Path(image_root).resolve() if image_root is not None else None
    unique: dict[str, tuple[str, str]] = {}
    reference_count = 0
    source_counts: Counter[str] = Counter()
    for sample in samples:
        if sources is not None and sample.source not in sources:
            continue
        for path in sample.images:
            reference_count += 1
            source_counts[sample.source] += 1
            # Dataset materializers and the path-safe archive boundary reject
            # links and traversal before this pass.  Avoid resolving every
            # member through the filesystem here: on a network volume that
            # adds hundreds of thousands of metadata round trips before the
            # actual stat/hash/decode work begins.
            resolved = Path(os.path.abspath(path))
            if root is not None:
                try:
                    logical = resolved.relative_to(root).as_posix()
                except ValueError as error:
                    raise ValueError(f"referenced image escapes image root: {path}") from error
            else:
                logical = os.fspath(path)
            unique.setdefault(os.fspath(resolved), (logical, sample.source))
    failures: list[dict[str, str]] = []
    failure_digest = hashlib.sha256()
    missing = corrupt = zero = decoded = encoded_bytes = 0
    formats: Counter[str] = Counter()
    inventory_digest = hashlib.sha256()
    if root is not None:
        checked_directories: set[Path] = set()
        for path_value in unique:
            for parent in Path(path_value).parents:
                if parent == root:
                    break
                if parent in checked_directories:
                    continue
                if parent.is_symlink():
                    raise ValueError(f"referenced image traverses a linked directory: {parent}")
                checked_directories.add(parent)
    items = [(path, *unique[path]) for path in sorted(unique, key=lambda key: unique[key][0])]
    processed = 0
    next_progress = progress_every
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Executor.map eagerly creates one future per item on supported Python
        # versions. Bound submissions so a 558K-image audit stays memory-light.
        batch_size = max(workers * 16, 1)
        for start in range(0, len(items), batch_size):
            for (
                _path,
                logical_path,
                source,
                error,
                size,
                content_digest,
                pixel_digest,
                width,
                height,
                image_format,
            ) in executor.map(_decode_image, items[start : start + batch_size]):
                if error is None:
                    decoded += 1
                    encoded_bytes += size
                    formats[image_format] += 1
                    record = ImageInventoryRecord(
                        logical_path=logical_path,
                        source=source,
                        size_bytes=size,
                        sha256=content_digest,
                        pixel_sha256=pixel_digest,
                        width=width,
                        height=height,
                        format=image_format,
                    )
                    if inventory is not None:
                        inventory(record)
                    inventory_digest.update(
                        (
                            f"{logical_path}\t{size}\t{content_digest}\t"
                            f"{width}x{height}\t{image_format}\n"
                        ).encode()
                    )
                else:
                    category = (
                        "missing"
                        if error == "missing"
                        else "zero-sized"
                        if error in {"zero-sized", "zero-dimension"}
                        else "corrupt"
                    )
                    missing += int(category == "missing")
                    zero += int(category == "zero-sized")
                    corrupt += int(category == "corrupt")
                    reason = error.split(":", 2)[1] if error.startswith("decode-error:") else error
                    canonical = f"{category}\t{source}\t{logical_path}\t{reason}\n"
                    failure_digest.update(canonical.encode("utf-8"))
                    if len(failures) < maximum_failure_details:
                        failures.append(
                            {
                                "category": category,
                                "source": source,
                                "path": logical_path,
                                "error": error,
                            }
                        )
                processed += 1
                if progress is not None and (
                    processed == len(items) or (progress_every and processed >= next_progress)
                ):
                    progress(processed, len(items))
                    while progress_every and next_progress <= processed:
                        next_progress += progress_every
    failed = missing + corrupt + zero
    return ImageIntegrityAudit(
        references=reference_count,
        unique_images=len(unique),
        decoded_images=decoded,
        missing_images=missing,
        corrupt_images=corrupt,
        zero_sized_images=zero,
        encoded_bytes=encoded_bytes,
        formats=dict(sorted(formats.items())),
        sources=dict(sorted(source_counts.items())),
        inventory_sha256=inventory_digest.hexdigest(),
        failure_details=tuple(failures),
        failure_details_truncated=failed > len(failures),
        failures_sha256=failure_digest.hexdigest(),
    )
