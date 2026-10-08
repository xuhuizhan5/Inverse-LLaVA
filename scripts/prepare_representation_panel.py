"""Freeze a question-diverse representation panel without consulting predictions.

One image and one normalized prompt per selected row. The selection is a
diagnostic design, not an estimate of the source benchmark's task distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples


def normalized_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", prompt)).strip().casefold()


def select_rows(
    rows: list[dict], image_hashes: dict[str, str], *, count: int, seed: int
) -> list[dict]:
    if count < 1 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("positive count and unique sample IDs are required")
    ranked = sorted(
        rows,
        key=lambda row: (hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), row["id"]),
    )
    prompts, images, selected = set(), set(), []
    for row in ranked:
        prompt, image = normalized_prompt(row["prompt"]), image_hashes[row["id"]]
        if not prompt or prompt in prompts or image in images:
            continue
        selected.append(row)
        prompts.add(prompt)
        images.add(image)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"only {len(selected)} distinct prompt/image pairs available; requested {count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    rows = [json.loads(line) for line in args.examples.read_text().splitlines() if line.strip()]
    examples = load_examples(args.examples)
    if any(len(example.images) != 1 for example in examples):
        raise ValueError("This diagnostic requires one image per example")
    hashes = {}
    for example in examples:
        with Image.open(example.images[0]) as image:
            rgb = image.convert("RGB")
            payload = f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()
            hashes[example.id] = hashlib.sha256(payload).hexdigest()
    selected = select_rows(rows, hashes, count=args.count, seed=args.seed)
    args.output.mkdir(parents=True)
    output = args.output / "examples.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))

    def category_counts(items):
        return dict(
            Counter(row.get("metadata", {}).get("question_type", "unspecified") for row in items)
        )

    record = {
        "source_sha256": sha256_file(args.examples),
        "examples_sha256": sha256_file(output),
        "generator_sha256": sha256_file(__file__),
        "seed": args.seed,
        "selection": (
            "ascending SHA256(seed:sample_id), retaining the first unused normalized prompt "
            "and decoded RGB image; no prediction or score input"
        ),
        "source_count": len(rows),
        "source_distinct_prompts": len({normalized_prompt(row["prompt"]) for row in rows}),
        "sample_ids": [row["id"] for row in selected],
        "prompt_sha256": [
            hashlib.sha256(normalized_prompt(row["prompt"]).encode()).hexdigest()
            for row in selected
        ],
        "image_rgb_sha256": [hashes[row["id"]] for row in selected],
        "source_categories": category_counts(rows),
        "selected_categories": category_counts(selected),
        "scope": (
            "Question-diverse diagnostic sample. Prompt deduplication changes the task mix; "
            "this panel does not estimate benchmark accuracy or category prevalence. "
            "Use this exact ordered manifest for every compared capture and derived figure."
        ),
    }
    (args.output / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": len(selected),
                "categories": record["selected_categories"],
                "examples_sha256": record["examples_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
