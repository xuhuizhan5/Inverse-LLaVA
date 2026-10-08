"""Checksummed materialization of public MMBench circular development splits."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from invllava import __version__
from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import optional_sha256_environment, sha256_file
from invllava.config.schema import BenchmarkSpec
from invllava.data.audit import ImageReferenceSet, audit_image_integrity, canonical_rgb_sha256
from invllava.data.download import download_http
from invllava.eval.datasets import prepare_mmbench, write_examples

_EXPECTED_ROWS = 4_329
_EXPECTED_GROUPS = 1_164


def materialize_mmbench_benchmark(
    spec: BenchmarkSpec,
    destination: str | Path,
    *,
    cache_dir: str | Path,
    config_sha256: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Download, validate, and materialize one public MMBench dev split."""

    if spec.id not in {"mmbench-en", "mmbench-cn"}:
        raise ValueError(f"{spec.id} is not an MMBench benchmark")
    if spec.annotations.kind != "http" or not spec.annotations.sha256:
        raise ValueError("MMBench requires a checksummed HTTP annotation source")
    if spec.images is not None:
        raise ValueError("MMBench images must be embedded in the declared TSV source")
    source_name = Path(spec.annotations.location).name
    source = download_http(
        spec.annotations.location,
        Path(cache_dir) / "benchmarks" / spec.annotations.id / source_name,
        sha256=spec.annotations.sha256,
    )

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    physical_images = temporary / "images"
    try:
        examples = prepare_mmbench(
            source,
            physical_images,
            logical_image_root=Path("images"),
            spec=spec,
        )
        groups = Counter(str(example.group_id) for example in examples)
        if len(examples) != _EXPECTED_ROWS or len(groups) != _EXPECTED_GROUPS:
            raise ValueError(
                "MMBench dev source must contain "
                f"{_EXPECTED_ROWS} rotations and {_EXPECTED_GROUPS} groups"
            )
        example_ids = {example.id for example in examples}
        if len(example_ids) != len(examples):
            raise ValueError("MMBench source contains duplicate row IDs")
        for example in examples:
            if (
                example.group_id is None
                or groups[example.group_id] > len(example.choices)
                or example.group_id not in example_ids
                or example.references[0] not in "ABCD"[: len(example.choices)]
            ):
                raise ValueError(f"MMBench circular contract failed for group {example.group_id}")
        if progress is not None:
            progress("convert", len(examples), len(examples))

        examples_path = temporary / "examples.jsonl"
        write_examples(examples, examples_path)
        audit = audit_image_integrity(
            (
                ImageReferenceSet((physical_images / example.images[0].name,), spec.id)
                for example in examples
            ),
            image_root=physical_images,
            workers=8,
            progress_every=250,
            progress=(
                (lambda completed, total: progress("image-audit", completed, total))
                if progress is not None
                else None
            ),
        )
        audit_payload = {
            "schema_version": 1,
            "benchmark_id": spec.id,
            "protocol_revision": spec.protocol_revision,
            "image_root": "images",
            **audit.to_dict(),
        }
        audit_path = temporary / "image-integrity.json"
        atomic_write_json(audit_path, audit_payload)
        if audit.unique_images != _EXPECTED_GROUPS or not audit.passed:
            raise RuntimeError("materialized MMBench image integrity audit failed")

        image_records = []
        for position, path in enumerate(sorted(physical_images.iterdir()), start=1):
            with Image.open(path) as image:
                image.load()
                pixel_sha256 = canonical_rgb_sha256(image.convert("RGB"))
                image_format = str(image.format or "unknown")
            image_records.append(
                {
                    "file": f"images/{path.name}",
                    "sha256": sha256_file(path),
                    "pixel_sha256": pixel_sha256,
                    "size_bytes": path.stat().st_size,
                    "format": image_format,
                }
            )
            if progress is not None and (position % 250 == 0 or position == _EXPECTED_GROUPS):
                progress("inventory", position, _EXPECTED_GROUPS)

        atomic_write_json(
            temporary / "manifest.json",
            {
                "format": "invllava-eval-dataset-v1",
                "invllava_version": __version__,
                "execution_source_sha256": optional_sha256_environment(
                    "INVLLAVA_EXECUTION_SOURCE_SHA256"
                ),
                "benchmark_id": spec.id,
                "protocol_revision": spec.protocol_revision,
                "protocol_config_sha256": config_sha256,
                "conversation_template": spec.conversation_template,
                "dataset": spec.annotations.location,
                "dataset_revision": spec.annotations.revision,
                "dataset_sha256": spec.annotations.sha256,
                "split": spec.split,
                "sample_count": len(examples),
                "group_count": len(groups),
                "group_size_counts": dict(sorted(Counter(groups.values()).items())),
                "examples_sha256": sha256_file(examples_path),
                "images": image_records,
                "image_integrity": {
                    **audit_payload,
                    "report_sha256": sha256_file(audit_path),
                },
            },
        )
        temporary.chmod(0o755)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
