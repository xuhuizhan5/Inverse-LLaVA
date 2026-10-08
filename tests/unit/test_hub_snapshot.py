from pathlib import Path
from unittest.mock import Mock

import pytest

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.hub_snapshot import acquire_hub_snapshot, verify_hub_snapshot


def _snapshot(root: Path) -> Path:
    root.mkdir()
    value = root / "config.json"
    value.write_text("{}\n", encoding="utf-8")
    atomic_write_json(
        root / "snapshot-manifest.json",
        {
            "repo_id": "fixture/model",
            "repo_type": "model",
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "allow_patterns": ["*.json"],
            "files": [
                {
                    "path": value.name,
                    "size_bytes": value.stat().st_size,
                    "sha256": sha256_file(value),
                }
            ],
        },
    )
    atomic_write_text(root / "COMPLETE", "complete\n")
    return root


def test_verify_hub_snapshot_checks_bytes(tmp_path: Path) -> None:
    root = _snapshot(tmp_path / "snapshot")
    assert verify_hub_snapshot(root)["files"][0]["path"] == "config.json"
    (root / "config.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        verify_hub_snapshot(root)


def test_verify_hub_snapshot_rejects_symlink(tmp_path: Path) -> None:
    root = _snapshot(tmp_path / "snapshot")
    (root / "linked.json").symlink_to("config.json")
    with pytest.raises(ValueError, match="symlink"):
        verify_hub_snapshot(root)


@pytest.mark.parametrize("revision", ["main", "a" * 40])
def test_existing_snapshot_can_be_reused_by_requested_or_resolved_revision(tmp_path, revision):
    root = _snapshot(tmp_path / "snapshot")
    assert (
        acquire_hub_snapshot(
            repo_id="fixture/model", revision=revision, destination=root, allow_patterns=("*.json",)
        )
        == root
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"repo_id": "different/model"},
        {"revision": "b" * 40},
        {"repo_type": "dataset"},
        {"allow_patterns": ("*.safetensors",)},
    ],
)
def test_existing_snapshot_rejects_changed_requested_identity(tmp_path, changed):
    root = _snapshot(tmp_path / "snapshot")
    request = {
        "repo_id": "fixture/model",
        "revision": "a" * 40,
        "destination": root,
        "allow_patterns": ("*.json",),
    }
    with pytest.raises(ValueError, match="different Hub source or file selection"):
        acquire_hub_snapshot(**dict(request, **changed))


def test_partial_without_source_record_is_preserved_and_rejected(tmp_path):
    root = tmp_path / "snapshot"
    partial = tmp_path / "snapshot.partial"
    partial.mkdir()
    value = partial / "download.incomplete"
    value.write_bytes(b"keep")
    with pytest.raises(ValueError, match="acquisition record"):
        acquire_hub_snapshot(repo_id="fixture/model", revision="a" * 40, destination=root)
    assert value.read_bytes() == b"keep" and not root.exists()


def test_partial_rejects_changed_file_selection_without_downloading(tmp_path, monkeypatch):
    import huggingface_hub

    download = Mock(side_effect=AssertionError("must reject before downloading"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    root = tmp_path / "snapshot"
    (tmp_path / "snapshot.partial").mkdir()
    atomic_write_json(
        tmp_path / "snapshot.acquisition.json",
        {
            "repo_id": "fixture/model",
            "repo_type": "model",
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "allow_patterns": ["*.json"],
        },
    )
    with pytest.raises(ValueError, match="file selection"):
        acquire_hub_snapshot(
            repo_id="fixture/model",
            revision="main",
            destination=root,
            allow_patterns=("*.safetensors",),
        )
    download.assert_not_called()
