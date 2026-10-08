#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.config.schema import BenchmarkSpec
from invllava.data.audit import audit_image_integrity, audit_samples
from invllava.data.manifest import build_prepared_manifest
from invllava.data.prepare import write_normalized_jsonl
from invllava.data.types import ConversationSample, Turn
from invllava.eval.datasets import write_examples
from invllava.eval.prompts import render_benchmark_prompt
from invllava.eval.types import EvaluationExample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists():
        raise FileExistsError(root)
    images = root / "images"
    images.mkdir(parents=True)
    samples: list[ConversationSample] = []
    for sample_id, name, color, answer in (
        ("fixture-red", "red.png", (255, 0, 0), "Red."),
        ("fixture-blue", "blue.png", (0, 0, 255), "Blue."),
    ):
        path = images / name
        Image.new("RGB", (16, 16), color).save(path)
        samples.append(
            ConversationSample(
                id=sample_id,
                images=(path.resolve(),),
                source="generated-contract-fixture",
                turns=(
                    Turn("user", "<image> What color is the square?"),
                    Turn("assistant", answer),
                ),
            )
        )
    source = root / "fixture-source.json"
    source.write_text(
        json.dumps(
            {
                "generator": "scripts/create_synthetic_fixture.py",
                "schema_version": 1,
                "samples": [sample.id for sample in samples],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    normalized = root / "conversations.jsonl"
    write_normalized_jsonl(samples, normalized)
    audit = audit_samples(samples)
    image_audit = audit_image_integrity(samples, image_root=images, workers=1)
    image_audit_path = root / "image-integrity.json"
    image_audit_payload = {
        "schema_version": 1,
        "annotation": str(source.resolve()),
        "annotation_sha256": sha256_file(source),
        "source_revision": "repository-generated-v1",
        "image_root": str(images.resolve()),
        "selected_sources": None,
        **image_audit.to_dict(),
    }
    atomic_write_json(image_audit_path, image_audit_payload)
    manifest = build_prepared_manifest(
        data_id="synthetic-contract-fixture",
        source_revision="repository-generated-v1",
        source_path=source,
        normalized_path=normalized,
        samples=samples,
        audit=audit,
        image_integrity={
            **image_audit_payload,
            "report_sha256": sha256_file(image_audit_path),
        },
    )
    manifest_path = root / "conversations.manifest.json"
    manifest.write(manifest_path)
    evaluation_path = root / "evaluation.jsonl"
    benchmark_path = (
        Path(__file__).resolve().parents[1] / "configs/benchmark/synthetic_contract.yaml"
    )
    benchmark = BenchmarkSpec.model_validate(
        yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    )
    write_examples(
        [
            EvaluationExample(
                id=sample.id,
                prompt=render_benchmark_prompt(
                    benchmark, {"question": "What color is the square?"}
                ),
                images=sample.images,
                references=(sample.turns[-1].text.rstrip("."),),
                metadata={"fixture": True},
            )
            for sample in samples
        ],
        evaluation_path,
    )
    print(
        json.dumps(
            {
                "evaluation": str(evaluation_path),
                "image_audit": str(image_audit_path),
                "jsonl": str(normalized),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
