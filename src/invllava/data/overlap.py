from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from invllava.artifacts.hashing import sha256_file


def image_groups_from_audit(
    images_by_id: Mapping[str, Sequence[Path]],
    audit_path: str | Path,
    *,
    annotation_sha256: str,
) -> tuple[dict[str, str], dict[str, object]]:
    """Group identical decoded image inputs using a checksum-bound decode audit.

    The caller supplies paths resolved in the audit's execution environment.
    No predictions, references, or scores participate in group selection.
    """

    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("annotation_sha256") != annotation_sha256:
        raise ValueError("image audit must pass and bind the supplied annotation")
    record = audit.get("inventory") or {}
    if record.get("identity") != "pixel_sha256":
        raise ValueError("grouping requires a decoded RGB-pixel inventory")
    inventory = Path(record["path"])
    if sha256_file(inventory) != record.get("sha256"):
        raise ValueError("image inventory checksum mismatch")
    _, by_hash, identity = _read_inventory(inventory)
    if identity != "pixel_sha256":
        raise ValueError("image inventory contains encoded-byte identities")
    root = Path(audit["image_root"])
    by_path: dict[Path, str] = {}
    for digest, paths in by_hash.items():
        for logical_path in paths:
            path = (root / logical_path).resolve()
            if path in by_path:
                raise ValueError(f"duplicate inventory image path: {path}")
            by_path[path] = digest
    groups = {}
    for sample_id, images in images_by_id.items():
        if not images:
            raise ValueError(f"image grouping requires an image for {sample_id}")
        try:
            identities = [by_path[path.resolve()] for path in images]
        except KeyError as error:
            raise ValueError(f"sample image is absent from audit: {error.args[0]}") from error
        groups[sample_id] = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
    return groups, {
        "identity": "ordered_rgb_pixel_content",
        "group_count": len(set(groups.values())),
        "audit_sha256": sha256_file(audit_path),
        "inventory_sha256": record["sha256"],
    }


@dataclass(frozen=True)
class ExactImageOverlap:
    identity: str
    left_inventory: str
    left_inventory_sha256: str
    right_inventory: str
    right_inventory_sha256: str
    left_records: int
    right_records: int
    left_unique_content: int
    right_unique_content: int
    shared_unique_content: int
    shared_examples: tuple[dict[str, object], ...]
    shared_examples_truncated: bool

    @property
    def disjoint(self) -> bool:
        return self.shared_unique_content == 0

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "disjoint": self.disjoint}


def _rows(path: Path) -> list[object]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and isinstance(payload.get("images"), list):
            return payload["images"]
        raise ValueError(f"JSON inventory manifest has no images list: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_inventory(path: Path) -> tuple[int, dict[str, list[str]], str]:
    records = 0
    content_paths: dict[str, list[str]] = {}
    identity_field: str | None = None
    rows = _rows(path)
    for line_number, value in enumerate(rows, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        available_field = "pixel_sha256" if value.get("pixel_sha256") else "sha256"
        if identity_field is None:
            identity_field = available_field
        elif available_field != identity_field:
            raise ValueError(f"{path} mixes pixel and encoded image identities")
        digest = str(value.get(available_field, "")).lower()
        logical_path = str(value.get("logical_path") or value.get("file") or "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{path}:{line_number} has an invalid {available_field}")
        if not logical_path:
            raise ValueError(f"{path}:{line_number} has no logical image path")
        records += 1
        content_paths.setdefault(digest, []).append(logical_path)
    if records == 0:
        raise ValueError(f"image inventory is empty: {path}")
    return records, content_paths, str(identity_field)


def compare_exact_image_overlap(
    left_inventory: str | Path,
    right_inventory: str | Path,
    *,
    maximum_examples: int = 100,
) -> ExactImageOverlap:
    """Compare canonical RGB pixels, or legacy encoded bytes, independent of file names."""

    if maximum_examples < 0:
        raise ValueError("maximum_examples must be non-negative")
    left = Path(left_inventory).resolve()
    right = Path(right_inventory).resolve()
    if left == right:
        raise ValueError("left and right inventories must be distinct")
    left_records, left_by_hash, left_identity = _read_inventory(left)
    right_records, right_by_hash, right_identity = _read_inventory(right)
    if left_identity != right_identity:
        raise ValueError("inventories use different identities; regenerate both with pixel_sha256")
    shared = sorted(left_by_hash.keys() & right_by_hash.keys())
    examples = tuple(
        {
            "content_sha256": digest,
            "left_paths": tuple(sorted(left_by_hash[digest])),
            "right_paths": tuple(sorted(right_by_hash[digest])),
        }
        for digest in shared[:maximum_examples]
    )
    return ExactImageOverlap(
        identity=left_identity,
        left_inventory=str(left),
        left_inventory_sha256=sha256_file(left),
        right_inventory=str(right),
        right_inventory_sha256=sha256_file(right),
        left_records=left_records,
        right_records=right_records,
        left_unique_content=len(left_by_hash),
        right_unique_content=len(right_by_hash),
        shared_unique_content=len(shared),
        shared_examples=examples,
        shared_examples_truncated=len(shared) > len(examples),
    )
