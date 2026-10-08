from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.data.audit import DataAudit
from invllava.data.types import ConversationSample


@dataclass(frozen=True)
class PreparedDatasetManifest:
    schema_version: int
    data_id: str
    source_revision: str
    source_sha256: str
    normalized_path: str
    normalized_sha256: str
    sample_ids_sha256: str
    audit: dict[str, object]
    image_integrity: dict[str, object] | None = None
    id_normalization: dict[str, object] | None = None
    image_path_policy: str | None = None
    source_filter: tuple[str, ...] | None = None

    def write(self, path: str | Path) -> None:
        atomic_write_json(path, asdict(self))

    @classmethod
    def read(cls, path: str | Path) -> PreparedDatasetManifest:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def verify(
        self,
        normalized_path: str | Path,
        *,
        expected_data_id: str,
        expected_source_revision: str | None = None,
        expected_source_sha256: str | None = None,
        expected_source_filter: tuple[str, ...] = (),
    ) -> None:
        if self.data_id != expected_data_id:
            raise ValueError(f"prepared data is {self.data_id}, expected {expected_data_id}")
        if (
            expected_source_revision is not None
            and self.source_revision != expected_source_revision
        ):
            raise ValueError("prepared annotation revision does not match the experiment config")
        if expected_source_sha256 is not None and self.source_sha256 != expected_source_sha256:
            raise ValueError("prepared annotation digest does not match the experiment config")
        if sha256_file(normalized_path) != self.normalized_sha256:
            raise ValueError("normalized dataset digest no longer matches its manifest")
        if self.schema_version >= 3:
            _validate_id_normalization(self.id_normalization)
        if self.schema_version >= 4 and self.image_path_policy != "relative-to-jsonl-parent-v1":
            raise ValueError("prepared data has an unsupported image-path policy")
        expected_filter = tuple(sorted(expected_source_filter))
        if self.schema_version >= 5:
            actual_filter = tuple(self.source_filter or ())
            if actual_filter != tuple(sorted(set(actual_filter))):
                raise ValueError("prepared data has an invalid source filter")
            if actual_filter != expected_filter:
                raise ValueError("prepared data source filter does not match the experiment config")
            audited_sources = tuple(
                sorted(str(key) for key in self.audit.get("samples_by_source", {}))
            )
            if actual_filter and audited_sources != actual_filter:
                raise ValueError("prepared data source filter does not match its sample audit")
        elif expected_filter:
            raise ValueError("legacy prepared data cannot satisfy a source-filtered experiment")
        image_references = int(self.audit.get("images", 0))
        if image_references:
            _validate_embedded_image_integrity(
                self.image_integrity,
                annotation_sha256=self.source_sha256,
                source_revision=self.source_revision,
                image_references=image_references,
                selected_sources=expected_filter or None,
            )


def _validate_id_normalization(payload: dict[str, object] | None) -> None:
    if payload is None:
        raise ValueError("prepared data manifest lacks its internal ID policy")
    policy = payload.get("policy")
    if policy not in {"preserve-source-id-v1", "row-index-prefix-v1"}:
        raise ValueError(f"prepared data has an unsupported internal ID policy: {policy}")
    duplicate_occurrences = int(payload.get("source_duplicate_occurrences", -1))
    duplicate_groups = int(payload.get("source_duplicate_groups", -1))
    if duplicate_occurrences < 0 or duplicate_groups < 0:
        raise ValueError("prepared data has invalid source-ID reuse counts")
    if policy == "preserve-source-id-v1" and (duplicate_occurrences or duplicate_groups):
        raise ValueError("source IDs cannot be preserved when they are reused")


def load_image_integrity_evidence(
    report_path: str | Path,
    *,
    annotation_path: str | Path,
    source_revision: str,
    image_root: str | Path,
    selected_sources: tuple[str, ...] = (),
) -> dict[str, object]:
    """Validate a complete image audit and bind its digest into prepared data."""

    report = Path(report_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("image integrity report must be a JSON object")
    annotation_sha256 = sha256_file(annotation_path)
    _validate_embedded_image_integrity(
        payload,
        annotation_sha256=annotation_sha256,
        source_revision=source_revision,
        image_references=int(payload.get("references", -1)),
        selected_sources=selected_sources or None,
    )
    if Path(str(payload.get("image_root", ""))).resolve() != Path(image_root).resolve():
        raise ValueError("image integrity report was produced for a different image root")
    return {**payload, "report_sha256": sha256_file(report)}


def _validate_embedded_image_integrity(
    payload: dict[str, object] | None,
    *,
    annotation_sha256: str,
    source_revision: str,
    image_references: int,
    selected_sources: tuple[str, ...] | None = None,
) -> None:
    if payload is None:
        raise ValueError("image-bearing prepared data requires a complete image integrity audit")
    if payload.get("annotation_sha256") != annotation_sha256:
        raise ValueError("image integrity audit does not match the annotation digest")
    if payload.get("source_revision") != source_revision:
        raise ValueError("image integrity audit does not match the annotation revision")
    actual_sources = tuple(sorted(str(value) for value in (payload.get("selected_sources") or ())))
    expected_sources = tuple(sorted(selected_sources or ()))
    if actual_sources != expected_sources:
        raise ValueError("image integrity audit has a different source filter")
    if payload.get("passed") is not True:
        raise ValueError("image integrity audit did not pass")
    if int(payload.get("references", -1)) != image_references:
        raise ValueError("image integrity reference count does not match prepared data")
    unique = int(payload.get("unique_images", -1))
    if unique <= 0 or int(payload.get("decoded_images", -1)) != unique:
        raise ValueError("image integrity audit did not decode every unique image")
    for key in ("missing_images", "corrupt_images", "zero_sized_images"):
        if int(payload.get(key, -1)) != 0:
            raise ValueError(f"image integrity audit records unresolved {key}")
    inventory = str(payload.get("inventory_sha256", ""))
    if len(inventory) != 64 or any(character not in "0123456789abcdef" for character in inventory):
        raise ValueError("image integrity audit has an invalid inventory digest")
    report_digest = payload.get("report_sha256")
    if report_digest is not None:
        report_digest = str(report_digest)
        if len(report_digest) != 64 or any(
            character not in "0123456789abcdef" for character in report_digest
        ):
            raise ValueError("image integrity audit has an invalid report digest")


def build_prepared_manifest(
    *,
    data_id: str,
    source_revision: str,
    source_path: str | Path,
    normalized_path: str | Path,
    samples: list[ConversationSample],
    audit: DataAudit,
    image_integrity: dict[str, object] | None = None,
    id_normalization: dict[str, object] | None = None,
    source_filter: tuple[str, ...] = (),
) -> PreparedDatasetManifest:
    ids = "\n".join(sample.id for sample in samples).encode()
    return PreparedDatasetManifest(
        schema_version=5,
        data_id=data_id,
        source_revision=source_revision,
        source_sha256=sha256_file(source_path),
        normalized_path=str(Path(normalized_path).resolve()),
        normalized_sha256=sha256_file(normalized_path),
        sample_ids_sha256=hashlib.sha256(ids).hexdigest(),
        audit=audit.to_dict(),
        image_integrity=image_integrity,
        id_normalization=id_normalization
        or {
            "policy": "preserve-source-id-v1",
            "source_duplicate_occurrences": 0,
            "source_duplicate_groups": 0,
        },
        image_path_policy="relative-to-jsonl-parent-v1",
        source_filter=tuple(sorted(source_filter)) or None,
    )
