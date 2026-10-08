import json
from pathlib import Path

import pytest

from invllava.artifacts.hashing import sha256_file
from invllava.data.overlap import compare_exact_image_overlap, image_groups_from_audit


def _write_inventory(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps({"logical_path": logical_path, "pixel_sha256": digest}) + "\n"
            for logical_path, digest in rows
        ),
        encoding="utf-8",
    )


def test_exact_image_overlap_uses_content_identity(tmp_path: Path) -> None:
    shared = "a" * 64
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_inventory(left, [("train/a.jpg", shared), ("train/b.jpg", "b" * 64)])
    _write_inventory(right, [("eval/renamed.png", shared), ("eval/c.png", "c" * 64)])

    report = compare_exact_image_overlap(left, right)

    assert not report.disjoint
    assert report.identity == "pixel_sha256"
    assert report.shared_unique_content == 1
    assert report.shared_examples[0]["left_paths"] == ("train/a.jpg",)
    assert report.shared_examples[0]["right_paths"] == ("eval/renamed.png",)


def test_exact_image_overlap_reports_disjoint_inventories(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_inventory(left, [("train/a.jpg", "a" * 64)])
    _write_inventory(right, [("eval/b.jpg", "b" * 64)])

    report = compare_exact_image_overlap(left, right)

    assert report.disjoint
    assert report.shared_examples == ()


def _grouping_fixture(tmp_path: Path) -> Path:
    inventory = tmp_path / "inventory.jsonl"
    _write_inventory(inventory, [("a.png", "a" * 64), ("renamed.png", "a" * 64)])
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "passed": True,
                "annotation_sha256": "b" * 64,
                "image_root": str(tmp_path),
                "inventory": {
                    "identity": "pixel_sha256",
                    "path": str(inventory),
                    "sha256": sha256_file(inventory),
                },
            }
        )
    )
    return audit


def test_grouping_joins_renamed_identical_images(tmp_path: Path) -> None:
    groups, metadata = image_groups_from_audit(
        {"q1": [tmp_path / "a.png"], "q2": [tmp_path / "renamed.png"]},
        _grouping_fixture(tmp_path),
        annotation_sha256="b" * 64,
    )
    assert groups["q1"] == groups["q2"]
    assert metadata["group_count"] == 1


def test_grouping_rejects_unbound_or_modified_inventory(tmp_path: Path) -> None:
    audit = _grouping_fixture(tmp_path)
    with pytest.raises(ValueError, match="bind"):
        image_groups_from_audit({"q": [tmp_path / "a.png"]}, audit, annotation_sha256="c" * 64)
    (tmp_path / "inventory.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        image_groups_from_audit({"q": [tmp_path / "a.png"]}, audit, annotation_sha256="b" * 64)


def test_grouping_rejects_missing_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absent"):
        image_groups_from_audit(
            {"q": [tmp_path / "missing.png"]},
            _grouping_fixture(tmp_path),
            annotation_sha256="b" * 64,
        )
