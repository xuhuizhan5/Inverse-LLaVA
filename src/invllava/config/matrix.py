"""Atomic materialization of reviewed experiment matrices."""

from __future__ import annotations

import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file
from invllava.config.identifiers import content_id, scientific_payload
from invllava.config.loader import ConfigRepository
from invllava.config.schema import ExperimentFile, ExperimentMatrixSpec


def _safe_repository_path(root: Path, reference: Path) -> Path:
    target = (root / reference).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"experiment matrix reference escapes repository: {reference}")
    return target


def set_config_path(document: dict[str, Any], dotted: str, value: Any) -> None:
    """Set an existing config path, allowing new keys only below ``overrides``."""

    keys = dotted.split(".")
    if not all(keys):
        raise ValueError("matrix patch paths must contain non-empty keys")
    current = document
    for key in keys[:-1]:
        if key not in current:
            if keys[0] != "overrides":
                raise ValueError(f"matrix patch refers to unknown path: {dotted}")
            current[key] = {}
        existing = current[key]
        if not isinstance(existing, dict):
            raise ValueError(f"cannot descend through non-mapping key in {dotted}")
        current = existing
    leaf = keys[-1]
    if leaf not in current and keys[0] != "overrides":
        raise ValueError(f"matrix patch refers to unknown path: {dotted}")
    current[leaf] = value


def materialize_experiment_matrix(
    matrix_path: str | Path,
    output_dir: str | Path,
    *,
    config_root: str | Path = "configs",
) -> Path:
    """Write a complete validated matrix and evidence manifest atomically."""

    matrix_source = Path(matrix_path).resolve(strict=True)
    config_directory = Path(config_root).resolve(strict=True)
    repository_root = config_directory.parent
    raw_matrix = yaml.safe_load(matrix_source.read_text(encoding="utf-8"))
    matrix = ExperimentMatrixSpec.model_validate(raw_matrix)
    base_path = _safe_repository_path(repository_root, matrix.base_experiment)
    base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base = ExperimentFile.model_validate(base_raw)

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    repository = ConfigRepository(config_directory)
    records: list[dict[str, Any]] = []
    try:
        for variant in matrix.variants:
            document = deepcopy(base_raw)
            document["id"] = f"{base.id}-{variant.id_suffix}"
            document["description"] = variant.description
            document["tags"] = list(
                dict.fromkeys(
                    [
                        *document.get("tags", []),
                        "ablation",
                        matrix.matrix_id,
                        f"factor:{variant.factor}",
                        f"value:{variant.value}",
                    ]
                )
            )
            for dotted, value in variant.patches.items():
                set_config_path(document, dotted, value)
            ExperimentFile.model_validate(document)
            path = temporary / f"{variant.id_suffix}.yaml"
            atomic_write_text(path, yaml.safe_dump(document, sort_keys=False))
            resolved = repository.resolve(path)
            records.append(
                {
                    "variant": variant.id_suffix,
                    "factor": variant.factor,
                    "value": variant.value,
                    "experiment_id": resolved.id,
                    "scientific_id": content_id(scientific_payload(resolved), prefix="sci"),
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
            )
        atomic_write_json(
            temporary / "matrix.manifest.json",
            {
                "schema_version": 1,
                "matrix_id": matrix.matrix_id,
                "matrix_config": str(matrix_source),
                "matrix_config_sha256": sha256_file(matrix_source),
                "base_experiment": str(base_path),
                "base_experiment_sha256": sha256_file(base_path),
                "selection_metric": matrix.selection_metric,
                "seed": matrix.seed,
                "variants": records,
            },
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
