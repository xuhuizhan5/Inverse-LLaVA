"""Deterministic, self-contained evaluation subsets for development checks."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples, write_examples
from invllava.eval.types import EvaluationExample


def _rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def materialize_evaluation_subset(
    source_examples: str | Path,
    destination: str | Path,
    *,
    maximum: int,
    seed: int,
    stratify_metadata: str | None = None,
    preserve_groups: bool = False,
) -> Path:
    """Create an immutable subset whose image paths survive source cleanup."""

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    source_path = Path(source_examples).resolve()
    target = Path(destination).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target.exists():
        raise FileExistsError(target)
    examples = load_examples(source_path)
    if not examples:
        raise ValueError("source evaluation set is empty")
    units: list[tuple[str, tuple[EvaluationExample, ...]]] = []
    if preserve_groups:
        by_group: dict[str, list[EvaluationExample]] = defaultdict(list)
        for example in examples:
            group_id = str(example.group_id or "").strip()
            if not group_id:
                raise ValueError(f"example {example.id} lacks a group_id")
            by_group[group_id].append(example)
        units = [
            (group_id, tuple(sorted(values, key=lambda item: item.id)))
            for group_id, values in sorted(by_group.items())
        ]
    else:
        units = [(example.id, (example,)) for example in examples]

    unit_strata: dict[str, list[tuple[str, tuple[EvaluationExample, ...]]]] = {}
    if stratify_metadata is None:
        ordered_units = sorted(units, key=lambda item: _rank(item[0], seed))
    else:
        grouped_units: dict[str, list[tuple[str, tuple[EvaluationExample, ...]]]] = defaultdict(
            list
        )
        for unit_id, members in units:
            values = {
                str(example.metadata.get(stratify_metadata) or "").strip() for example in members
            }
            if "" in values:
                raise ValueError(
                    f"group {unit_id} lacks stratification metadata {stratify_metadata}"
                )
            if len(values) != 1:
                raise ValueError(f"group {unit_id} spans multiple {stratify_metadata} strata")
            grouped_units[next(iter(values))].append((unit_id, members))
        unit_strata = {
            key: sorted(values, key=lambda item: _rank(item[0], seed))
            for key, values in sorted(grouped_units.items())
        }
        ordered_units = []
        offset = 0
        while len(ordered_units) < len(units):
            added = False
            for key in sorted(unit_strata):
                if offset < len(unit_strata[key]):
                    ordered_units.append(unit_strata[key][offset])
                    added = True
            if not added:
                break
            offset += 1

    selected = []
    selected_group_ids: list[str] = []
    for unit_id, members in ordered_units:
        if len(selected) + len(members) <= maximum:
            selected.extend(members)
            selected_group_ids.append(unit_id)
    if not selected:
        qualifier = "complete group" if preserve_groups else "example"
        raise ValueError(f"maximum={maximum} cannot accommodate any {qualifier}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        image_root = temporary / "images"
        image_root.mkdir()
        rewritten = []
        image_records: list[dict[str, object]] = []
        copied: dict[Path, Path] = {}
        for example in selected:
            new_images = []
            for image in example.images:
                source_image = image.resolve()
                if not source_image.is_file():
                    raise FileNotFoundError(source_image)
                if source_image not in copied:
                    suffix = source_image.suffix.lower() or ".image"
                    name = hashlib.sha256(str(source_image).encode()).hexdigest() + suffix
                    physical = image_root / name
                    try:
                        os.link(source_image, physical)
                    except OSError:
                        shutil.copy2(source_image, physical)
                    copied[source_image] = physical
                    image_records.append(
                        {
                            "file": f"images/{name}",
                            "sha256": sha256_file(physical),
                            "size_bytes": physical.stat().st_size,
                        }
                    )
                new_images.append(Path("images") / copied[source_image].name)
            rewritten.append(replace(example, images=tuple(new_images)))

        examples_path = temporary / "examples.jsonl"
        write_examples(rewritten, examples_path)
        atomic_write_json(
            temporary / "manifest.json",
            {
                "format": "invllava-eval-subset-v1",
                "development_only": True,
                "source_examples": str(source_path),
                "source_examples_sha256": sha256_file(source_path),
                "seed": seed,
                "maximum": maximum,
                "stratify_metadata": stratify_metadata,
                "preserve_groups": preserve_groups,
                "selected_group_ids": selected_group_ids if preserve_groups else [],
                "selected_group_count": len(selected_group_ids) if preserve_groups else None,
                "source_stratum_counts": {
                    key: sum(len(members) for _, members in values)
                    for key, values in sorted(unit_strata.items())
                },
                "selected_stratum_counts": (
                    {
                        key: sum(
                            str(example.metadata[stratify_metadata]).strip() == key
                            for example in rewritten
                        )
                        for key in sorted(unit_strata)
                    }
                    if stratify_metadata is not None
                    else {}
                ),
                "sample_count": len(rewritten),
                "selected_ids": [example.id for example in rewritten],
                "examples_sha256": sha256_file(examples_path),
                "images": image_records,
            },
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
