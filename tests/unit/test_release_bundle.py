import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invllava.artifacts.hashing import sha256_file
from invllava.model.champion_checkpoint import RELEASE_CONFIG_FILENAME, RELEASE_FORMAT
from invllava.release.bundle import (
    TRAINING_RELEASE_SOURCE_FORMAT,
    _release_model_spec,
    seal_training_checkpoint,
)


def test_release_reader_translates_only_noop_v1_projection_fields():
    from invllava.config.loader import ConfigRepository

    spec = ConfigRepository().resolve("configs/experiment/canonical_5pct_calibration.yaml").model
    original = spec.model_dump(mode="json")
    original["fusion"].update(project_visual_features=False, projection_dim=1024)
    before = json.dumps(original, sort_keys=True)
    assert _release_model_spec(original) == spec
    assert json.dumps(original, sort_keys=True) == before
    assert _release_model_spec(spec.model_dump(mode="json")) == spec


@pytest.mark.parametrize(
    "fields", [{"project_visual_features": True}, {"projection_dim": 512}, {"projection_dim": True}]
)
def test_release_reader_rejects_architecture_changing_projection_fields(fields):
    from invllava.config.loader import ConfigRepository

    model = ConfigRepository().resolve("configs/experiment/canonical_5pct_calibration.yaml").model
    value = model.model_dump(mode="json")
    value["fusion"].update(fields)
    with pytest.raises(ValueError, match="projection"):
        _release_model_spec(value)


def _write_checksums(root: Path) -> None:
    lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_parent(root: Path) -> Path:
    parent = root / "parent"
    parent.mkdir()
    delta = {"layer.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    save_file(delta, parent / "model_delta.safetensors")
    (parent / RELEASE_CONFIG_FILENAME).write_text(
        json.dumps({"format": RELEASE_FORMAT, "weights": "model_delta.safetensors"}),
        encoding="utf-8",
    )
    (parent / "metadata.json").write_text(
        json.dumps(
            {
                "format": RELEASE_FORMAT,
                "source_format": "fixture",
                "model_delta_sha256": sha256_file(parent / "model_delta.safetensors"),
                "upstream_base_projection_verification": {"status": "exact"},
            }
        ),
        encoding="utf-8",
    )
    (parent / "README.md").write_text("# Fixture\n", encoding="utf-8")
    (parent / "COMPLETE").write_text("complete\n", encoding="utf-8")
    _write_checksums(parent)
    return parent


def _write_checkpoint(root: Path, *, shape: tuple[int, ...] = (2, 3)) -> Path:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    save_file(
        {"layer.weight": torch.full(shape, 7, dtype=torch.bfloat16)},
        checkpoint / "model_delta.safetensors",
    )
    (checkpoint / "metadata.json").write_text(
        json.dumps({"experiment_id": "paired-1pct", "total_steps": 44}),
        encoding="utf-8",
    )
    (checkpoint / "state.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "global_step": 44,
                "samples_seen": 5580,
                "tokens_seen": 12345,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint


def test_seal_training_checkpoint_preserves_release_contract(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    checkpoint = _write_checkpoint(tmp_path)

    output = seal_training_checkpoint(
        checkpoint,
        parent_release=parent,
        destination=tmp_path / "release",
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_format"] == TRAINING_RELEASE_SOURCE_FORMAT
    assert metadata["training_summary"]["global_step"] == 44
    assert metadata["training_summary"]["samples_seen"] == 5580
    assert metadata["parent_release"]["model_delta_sha256"] == sha256_file(
        parent / "model_delta.safetensors"
    )
    assert metadata["model_delta_sha256"] == sha256_file(checkpoint / "model_delta.safetensors")
    assert (output / "README.md").read_text(encoding="utf-8") == "# Fixture\n"
    assert (output / "COMPLETE").read_text(encoding="utf-8") == "complete\n"
    assert output.stat().st_mode & 0o777 == 0o755
    declared = {
        line.split("  ", 1)[1]
        for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    }
    assert declared == {
        "COMPLETE",
        "README.md",
        RELEASE_CONFIG_FILENAME,
        "metadata.json",
        "model_delta.safetensors",
    }


def test_seal_training_checkpoint_rejects_incompatible_delta(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    checkpoint = _write_checkpoint(tmp_path, shape=(3, 2))

    with pytest.raises(ValueError, match="tensor inventories disagree"):
        seal_training_checkpoint(
            checkpoint,
            parent_release=parent,
            destination=tmp_path / "release",
        )
