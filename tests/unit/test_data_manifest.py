import hashlib
import json

import pytest
from PIL import Image

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.data.audit import DataAudit, audit_image_integrity
from invllava.data.manifest import (
    build_prepared_manifest,
    load_image_integrity_evidence,
)
from invllava.data.types import ConversationSample, Turn


def test_prepared_manifest_binds_annotation_revision_and_digest(tmp_path) -> None:
    source = tmp_path / "source.json"
    normalized = tmp_path / "normalized.jsonl"
    source.write_text("[]\n")
    normalized.write_text("")
    manifest = build_prepared_manifest(
        data_id="fixture",
        source_revision="commit-1",
        source_path=source,
        normalized_path=normalized,
        samples=[],
        audit=DataAudit(0, 0, 0, 0, {}, {}),
    )
    manifest.verify(
        normalized,
        expected_data_id="fixture",
        expected_source_revision="commit-1",
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    assert manifest.schema_version == 5
    assert manifest.image_path_policy == "relative-to-jsonl-parent-v1"
    assert manifest.id_normalization == {
        "policy": "preserve-source-id-v1",
        "source_duplicate_occurrences": 0,
        "source_duplicate_groups": 0,
    }
    with pytest.raises(ValueError, match="revision"):
        manifest.verify(
            normalized,
            expected_data_id="fixture",
            expected_source_revision="commit-2",
        )


def test_prepared_manifest_requires_matching_complete_image_audit(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    image = root / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    source = tmp_path / "source.json"
    source.write_text("[]\n")
    normalized = tmp_path / "normalized.jsonl"
    normalized.write_text(json.dumps({"id": "one"}) + "\n")
    sample = ConversationSample(
        "one",
        (image,),
        (Turn("user", "<image>"), Turn("assistant", "answer")),
        "fixture",
    )
    image_audit = audit_image_integrity([sample], image_root=root, workers=1)
    report = tmp_path / "image-audit.json"
    atomic_write_json(
        report,
        {
            "schema_version": 1,
            "annotation_sha256": sha256_file(source),
            "source_revision": "commit-1",
            "image_root": str(root.resolve()),
            "selected_sources": None,
            **image_audit.to_dict(),
        },
    )
    evidence = load_image_integrity_evidence(
        report,
        annotation_path=source,
        source_revision="commit-1",
        image_root=root,
    )
    audit = DataAudit(1, 1, 0, 0, {"assistant": 1, "user": 1}, {"fixture": 1})
    manifest = build_prepared_manifest(
        data_id="fixture",
        source_revision="commit-1",
        source_path=source,
        normalized_path=normalized,
        samples=[sample],
        audit=audit,
        image_integrity=evidence,
    )
    manifest.verify(normalized, expected_data_id="fixture")
    assert evidence["report_sha256"] == sha256_file(report)

    unverified = build_prepared_manifest(
        data_id="fixture",
        source_revision="commit-1",
        source_path=source,
        normalized_path=normalized,
        samples=[sample],
        audit=audit,
    )
    with pytest.raises(ValueError, match="requires a complete image integrity audit"):
        unverified.verify(normalized, expected_data_id="fixture")


def test_prepared_manifest_binds_source_filtered_image_audit(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    image = root / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    source = tmp_path / "source.json"
    source.write_text("[]\n")
    normalized = tmp_path / "normalized.jsonl"
    normalized.write_text(json.dumps({"id": "one"}) + "\n")
    sample = ConversationSample(
        "one",
        (image,),
        (Turn("user", "<image>"), Turn("assistant", "answer")),
        "ocr_vqa",
    )
    image_audit = audit_image_integrity([sample], sources={"ocr_vqa"}, image_root=root, workers=1)
    report = tmp_path / "image-audit.json"
    atomic_write_json(
        report,
        {
            "schema_version": 1,
            "annotation_sha256": sha256_file(source),
            "source_revision": "commit-1",
            "image_root": str(root.resolve()),
            "selected_sources": ["ocr_vqa"],
            **image_audit.to_dict(),
        },
    )
    evidence = load_image_integrity_evidence(
        report,
        annotation_path=source,
        source_revision="commit-1",
        image_root=root,
        selected_sources=("ocr_vqa",),
    )
    audit = DataAudit(1, 1, 0, 0, {"assistant": 1, "user": 1}, {"ocr_vqa": 1})
    manifest = build_prepared_manifest(
        data_id="ocr-fixture",
        source_revision="commit-1",
        source_path=source,
        normalized_path=normalized,
        samples=[sample],
        audit=audit,
        image_integrity=evidence,
        source_filter=("ocr_vqa",),
    )
    manifest.verify(
        normalized,
        expected_data_id="ocr-fixture",
        expected_source_filter=("ocr_vqa",),
    )
    with pytest.raises(ValueError, match="source filter"):
        manifest.verify(normalized, expected_data_id="ocr-fixture")
