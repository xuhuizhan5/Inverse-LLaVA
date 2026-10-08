from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from invllava.data.audit import canonical_rgb_sha256
from invllava.eval.types import EvaluationExample


def shuffled_indices(count: int, *, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("image shuffling requires at least two examples")
    rng = np.random.default_rng(seed)
    # Sattolo's algorithm creates one cycle and therefore guarantees a
    # derangement without an unbounded rejection loop.
    values = np.arange(count)
    for index in range(count - 1, 0, -1):
        swap = int(rng.integers(0, index))
        values[index], values[swap] = values[swap], values[index]
    return values


def shuffled_image_examples(
    examples: list[EvaluationExample], *, seed: int
) -> list[EvaluationExample]:
    """Derange distinct RGB images, keeping all questions for one image together."""

    if any(len(example.images) != 1 for example in examples):
        raise ValueError("shuffled-image intervention requires one image per example")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("shuffled-image intervention requires unique example IDs")
    digests: dict[Path, str] = {}
    representatives: dict[str, EvaluationExample] = {}
    for example in examples:
        path = example.images[0].resolve()
        if path not in digests:
            with Image.open(path) as image:
                digests[path] = canonical_rgb_sha256(image.convert("RGB"))
        representatives.setdefault(digests[path], example)
    image_keys = list(representatives)
    if len(image_keys) < 2:
        raise ValueError("image shuffling requires at least two distinct RGB images")
    permutation = shuffled_indices(len(image_keys), seed=seed)
    replacements = {
        key: image_keys[int(permutation[index])] for index, key in enumerate(image_keys)
    }
    results = []
    for example in examples:
        original_digest = digests[example.images[0].resolve()]
        replacement_digest = replacements[original_digest]
        replacement = representatives[replacement_digest]
        results.append(
            replace(
                example,
                images=replacement.images,
                metadata={
                    **example.metadata,
                    "intervention_kind": "deranged_image",
                    "intervention_source_id": replacement.id,
                    "intervention_seed": seed,
                    "intervention_unit": "decoded_rgb_image",
                    "intervention_original_rgb_sha256": original_digest,
                    "intervention_replacement_rgb_sha256": replacement_digest,
                },
            )
        )
    return results


def blank_image_examples(
    examples: list[EvaluationExample],
    destination: str | Path,
    *,
    rgb: tuple[int, int, int] = (127, 127, 127),
) -> list[EvaluationExample]:
    if any(len(example.images) != 1 for example in examples):
        raise ValueError("blank-image intervention requires one image per example")
    if any(not 0 <= channel <= 255 for channel in rgb):
        raise ValueError("blank RGB channels must be integers in [0,255]")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    names: list[str] = []
    try:
        for example in examples:
            name = hashlib.sha256(example.id.encode()).hexdigest()[:24] + ".png"
            names.append(name)
            with Image.open(example.images[0]) as source:
                size = source.size
            Image.new("RGB", size, rgb).save(temporary / name, format="PNG")
        temporary.chmod(0o755)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return [
        replace(
            example,
            images=((destination / names[index]).resolve(),),
            metadata={
                **example.metadata,
                "intervention_kind": "blank_image",
                "intervention_rgb": list(rgb),
            },
        )
        for index, example in enumerate(examples)
    ]


def intervention_effect(correct: np.ndarray, intervened: np.ndarray) -> dict[str, float | int]:
    correct = np.asarray(correct, dtype=float)
    intervened = np.asarray(intervened, dtype=float)
    if correct.shape != intervened.shape:
        raise ValueError("intervention arrays must align")
    if correct.ndim != 1 or correct.size == 0:
        raise ValueError("intervention arrays must be non-empty vectors")
    difference = correct - intervened
    return {
        "mean_drop": float(difference.mean()),
        "fraction_harmed": float((difference > 0).mean()),
        "fraction_improved": float((difference < 0).mean()),
        "fraction_unchanged": float((difference == 0).mean()),
        "correct_to_incorrect": int(((correct == 1) & (intervened == 0)).sum()),
        "incorrect_to_correct": int(((correct == 0) & (intervened == 1)).sum()),
        "both_correct": int(((correct == 1) & (intervened == 1)).sum()),
        "both_incorrect": int(((correct == 0) & (intervened == 0)).sum()),
    }


def intervention_response_effect(
    original: Sequence[str], intervened: Sequence[str]
) -> dict[str, float | int | list[int]]:
    """Summarize output sensitivity without changing benchmark scoring.

    Exact comparison preserves the generated text verbatim. The normalized view
    only collapses whitespace and case, so formatting-only changes can be audited
    separately from substantive response changes.
    """

    if len(original) != len(intervened):
        raise ValueError("intervention responses must align")
    if not original:
        raise ValueError("intervention responses must be non-empty")
    exact_changed = [original[index] != intervened[index] for index in range(len(original))]

    def normalize(value: str) -> str:
        return " ".join(value.split()).casefold()

    normalized_changed = [
        normalize(original[index]) != normalize(intervened[index]) for index in range(len(original))
    ]
    return {
        "exact_response_changed": int(sum(exact_changed)),
        "exact_response_change_fraction": float(np.mean(exact_changed)),
        "normalized_response_changed": int(sum(normalized_changed)),
        "normalized_response_change_fraction": float(np.mean(normalized_changed)),
        "exact_changed_indices": [index for index, changed in enumerate(exact_changed) if changed],
        "normalized_changed_indices": [
            index for index, changed in enumerate(normalized_changed) if changed
        ],
    }
