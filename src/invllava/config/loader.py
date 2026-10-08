"""Resolve small referenced YAML files into one validated experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from invllava.config.schema import (
    DataSpec,
    ExperimentFile,
    ModelSpec,
    ResolvedExperiment,
    RuntimeSpec,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _merge(base: dict[str, Any], override: dict[str, Any], path: str = "") -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        here = f"{path}.{key}" if path else key
        if key not in result:
            raise ValueError(f"override refers to unknown key: {here}")
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _merge(result[key], value, here)
        else:
            result[key] = value
    return result


class ConfigRepository:
    def __init__(self, root: str | Path = "configs") -> None:
        self.root = Path(root).resolve()

    def _find(self, group: str, reference: str) -> Path:
        candidate = (self.root / group / reference).resolve()
        if candidate.suffix not in {".yaml", ".yml"}:
            candidate = candidate.with_suffix(".yaml")
        if self.root not in candidate.parents:
            raise ValueError(f"configuration escapes root: {reference}")
        return candidate

    def resolve(
        self,
        experiment_path: str | Path,
        *,
        runtime_ref: str | None = None,
        microbatch_size: int | None = None,
        gradient_checkpointing: bool | None = None,
        maximum_samples: int | None = None,
    ) -> ResolvedExperiment:
        path = Path(experiment_path).resolve()
        raw_experiment = _read_yaml(path)
        experiment = ExperimentFile.model_validate(raw_experiment)
        model_path = self._find("model", experiment.model_ref)
        data_path = self._find("data", experiment.data_ref)
        declared_runtime_path = self._find("runtime", experiment.runtime_ref)
        runtime_path = self._find("runtime", runtime_ref or experiment.runtime_ref)

        training = experiment.training.model_dump(mode="python")
        if microbatch_size is not None and microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if runtime_ref is not None or microbatch_size is not None:
            declared_runtime = RuntimeSpec.model_validate(_read_yaml(declared_runtime_path))
            selected_runtime = RuntimeSpec.model_validate(_read_yaml(runtime_path))
            effective_batch = (
                training["per_device_batch_size"]
                * training["gradient_accumulation_steps"]
                * declared_runtime.num_processes
            )
            selected_per_device = microbatch_size or training["per_device_batch_size"]
            selected_microbatch = selected_per_device * selected_runtime.num_processes
            if effective_batch % selected_microbatch:
                raise ValueError(
                    "execution override cannot preserve the declared effective global batch: "
                    f"{effective_batch} is not divisible by {selected_microbatch}"
                )
            training["per_device_batch_size"] = selected_per_device
            training["gradient_accumulation_steps"] = effective_batch // selected_microbatch
        if gradient_checkpointing is not None:
            training["gradient_checkpointing"] = gradient_checkpointing

        aggregate: dict[str, Any] = {
            "model": ModelSpec.model_validate(_read_yaml(model_path)).model_dump(mode="python"),
            "data": DataSpec.model_validate(_read_yaml(data_path)).model_dump(mode="python"),
            "runtime": RuntimeSpec.model_validate(_read_yaml(runtime_path)).model_dump(
                mode="python"
            ),
            "training": training,
            "generation": experiment.generation.model_dump(mode="python"),
        }
        aggregate = _merge(aggregate, experiment.overrides)
        if maximum_samples is not None:
            if maximum_samples <= 0:
                raise ValueError("maximum_samples must be positive")
            aggregate["data"]["max_samples"] = maximum_samples

        return ResolvedExperiment(
            id=experiment.id,
            description=experiment.description,
            method_revision=experiment.method_revision,
            evidence_class=experiment.evidence_class,
            initial_checkpoint_id=experiment.initial_checkpoint_id,
            model=ModelSpec.model_validate(aggregate["model"]),
            data=DataSpec.model_validate(aggregate["data"]),
            runtime=RuntimeSpec.model_validate(aggregate["runtime"]),
            training=experiment.training.model_validate(aggregate["training"]),
            generation=experiment.generation.model_validate(aggregate["generation"]),
            tags=experiment.tags,
            source_files=(path, model_path, data_path, runtime_path),
        )
