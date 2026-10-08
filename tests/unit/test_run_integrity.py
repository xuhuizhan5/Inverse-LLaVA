import hashlib
import json
from pathlib import Path

import pytest
import torch

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.manifest import ArtifactRecord, RunManifest
from invllava.artifacts.run_integrity import verify_training_run
from invllava.config.schema import OptimizerSpec
from invllava.train.checkpoint import CheckpointManager, checkpoint_inventory
from invllava.train.metrics import JsonlMetricWriter
from invllava.train.optimizer import build_scheduler
from invllava.train.state import TrainState


def _complete_run(root: Path) -> Path:
    run = root / "fixture-run"
    run.mkdir()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, OptimizerSpec(), total_steps=1)
    CheckpointManager(run / "checkpoints").save(
        name="step-0000001",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=TrainState(global_step=1),
        metadata={"fixture": True},
    )
    JsonlMetricWriter(run / "metrics.jsonl").write({"step": 1, "train/loss": 0.5})
    inventory = run / "checkpoint_inventory.json"
    atomic_write_json(inventory, checkpoint_inventory(run / "checkpoints"))
    completion = b"complete\n"
    atomic_write_text(run / "RUN_COMPLETE", completion.decode())
    manifest = RunManifest.create(
        run_id=run.name,
        experiment_id="fixture",
        scientific_id="sci-fixture",
        method_revision="fixture-v1",
        repository_root=root,
        source_configs=[],
        dataset_revisions={},
    )
    for path, role, digest in (
        (inventory, "checkpoint_inventory", sha256_file(inventory)),
        (run / "metrics.jsonl", "training_metrics", sha256_file(run / "metrics.jsonl")),
        (run / "RUN_COMPLETE", "run_complete_marker", hashlib.sha256(completion).hexdigest()),
    ):
        manifest.add_artifact(
            ArtifactRecord(
                role=role,
                relative_path=path.relative_to(run).as_posix(),
                sha256=digest,
                size_bytes=path.stat().st_size,
            )
        )
    manifest.write(run / "manifest.json")
    return run


def test_verify_training_run_checks_metrics_checkpoints_and_manifest(tmp_path: Path) -> None:
    report = verify_training_run(_complete_run(tmp_path))
    assert report.complete
    assert report.checkpoint_count == 1
    assert report.metric_records == 1


def _bind_model_configuration(run: Path) -> None:
    from invllava.config.loader import ConfigRepository

    model = (
        ConfigRepository(Path(__file__).resolve().parents[2] / "configs")
        .resolve("configs/experiment/canonical_7b.yaml")
        .model.model_dump(mode="json")
    )
    path = run / "resolved_config.json"
    atomic_write_json(path, {"model": model})
    manifest = RunManifest.read(run / "manifest.json")
    manifest.checkpoint_inputs = {
        key: f"{model[key]['checkpoint']}@{model[key]['revision']}"
        for key in ("language", "vision")
    }
    manifest.add_artifact(
        ArtifactRecord(
            role="resolved_configuration",
            relative_path=path.name,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
    )
    manifest.write(run / "manifest.json")


def test_export_uses_bound_run_configuration_without_parent(tmp_path: Path) -> None:
    from invllava.config.schema import ModelSpec
    from invllava.model.champion_checkpoint import RELEASE_CONFIG_FILENAME
    from invllava.release.bundle import _validate_upstream_provenance, export_training_checkpoint

    run = _complete_run(tmp_path)
    _bind_model_configuration(run)
    output = export_training_checkpoint(
        run, run / "checkpoints/step-0000001", destination=tmp_path / "release"
    )
    config = json.loads((output / RELEASE_CONFIG_FILENAME).read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    assert config["model"] == json.loads((run / "resolved_config.json").read_text())["model"]
    assert "parent_release" not in metadata
    assert metadata["experiment_id"] == RunManifest.read(run / "manifest.json").experiment_id
    assert "upstream_base_projection_verification" not in metadata
    _validate_upstream_provenance(metadata, ModelSpec.model_validate(config["model"]))
    metadata["training_provenance"]["checkpoint_inputs"]["language"] = "wrong@revision"
    with pytest.raises(ValueError, match="provenance"):
        _validate_upstream_provenance(metadata, ModelSpec.model_validate(config["model"]))


def test_export_rejects_changed_configuration_and_foreign_checkpoint(tmp_path: Path) -> None:
    from invllava.release.bundle import export_training_checkpoint

    run = _complete_run(tmp_path)
    _bind_model_configuration(run)
    with pytest.raises(ValueError, match="belong"):
        export_training_checkpoint(run, tmp_path / "foreign", destination=tmp_path / "out")
    (run / "resolved_config.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        export_training_checkpoint(
            run, run / "checkpoints/step-0000001", destination=tmp_path / "out"
        )


def test_verify_training_run_detects_tampered_inventory(tmp_path: Path) -> None:
    run = _complete_run(tmp_path)
    inventory = json.loads((run / "checkpoint_inventory.json").read_text(encoding="utf-8"))
    inventory["checkpoints"][0]["files"]["state.json"]["sha256"] = "0" * 64
    atomic_write_json(run / "checkpoint_inventory.json", inventory)
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        verify_training_run(run)


def test_verify_training_run_rejects_symlink_artifacts(tmp_path: Path) -> None:
    run = _complete_run(tmp_path)
    target = run / "metrics.jsonl"
    link = run / "metrics-link.jsonl"
    link.symlink_to(target.name)
    manifest = RunManifest.read(run / "manifest.json")
    manifest.add_artifact(
        ArtifactRecord(
            role="invalid_symlink",
            relative_path=link.name,
            sha256=sha256_file(target),
            size_bytes=target.stat().st_size,
        )
    )
    manifest.write(run / "manifest.json")
    with pytest.raises(ValueError, match="must not be a symlink"):
        verify_training_run(run)
