#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml

from invllava.config.loader import ConfigRepository
from invllava.config.schema import (
    BenchmarkSpec,
    DataLayoutSpec,
    DataSpec,
    ExperimentMatrixSpec,
    ModelSpec,
    ReproductionSpec,
    RuntimeSpec,
)
from invllava.eval.language_suite import load_language_suite


def main() -> None:
    repository = ConfigRepository("configs")
    groups = (
        ("model", ModelSpec),
        ("data", DataSpec),
        ("runtime", RuntimeSpec),
    )
    group_counts: dict[str, int] = {}
    for group, schema in groups:
        paths = sorted(Path("configs", group).glob("*.yaml"))
        for path in paths:
            schema.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        group_counts[group] = len(paths)
    experiments = [
        path
        for path in Path("configs/experiment").glob("*.yaml")
        if not path.name.endswith("matrix.yaml") and "_matrix" not in path.name
    ]
    for path in experiments:
        repository.resolve(path)
    for path in Path("configs/benchmark").glob("*.yaml"):
        BenchmarkSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    language_suites = list(Path("configs/evaluation").glob("*.yaml"))
    for path in language_suites:
        load_language_suite(path)
    reproduction_specs = list(Path("configs/reproduction").glob("*.yaml"))
    for path in reproduction_specs:
        ReproductionSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    experiment_matrices = list(Path("configs/experiment").glob("*_matrix.yaml"))
    for path in experiment_matrices:
        ExperimentMatrixSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    data_layouts = list(Path("configs/data_layout").glob("*.yaml"))
    for path in data_layouts:
        DataLayoutSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    print(
        f"validated {group_counts['model']} models, {group_counts['data']} data recipes, "
        f"{group_counts['runtime']} runtimes, {len(experiments)} experiments, "
        f"benchmark catalog, {len(language_suites)} language suite(s), and "
        f"{len(reproduction_specs)} reproduction manifest(s), "
        f"{len(experiment_matrices)} experiment matrix/matrices, and "
        f"{len(data_layouts)} data layout(s)"
    )


if __name__ == "__main__":
    main()
