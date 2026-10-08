from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from PIL import Image

from invllava.analysis.interventions import (
    blank_image_examples,
    intervention_effect,
    intervention_response_effect,
    shuffled_image_examples,
    shuffled_indices,
)
from invllava.eval.types import EvaluationExample


def test_shuffled_indices_are_a_deterministic_derangement() -> None:
    first = shuffled_indices(20, seed=7)
    second = shuffled_indices(20, seed=7)
    assert np.array_equal(first, second)
    assert not np.any(first == np.arange(20))


def test_intervention_examples_preserve_ids_and_generate_blank_images() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        originals = []
        for index in range(3):
            path = root / f"source-{index}.png"
            Image.new("RGB", (4 + index, 5), (index, index, index)).save(path)
            originals.append(EvaluationExample(str(index), "prompt", (path,), ("yes",)))
        shuffled = shuffled_image_examples(originals, seed=9)
        assert [example.id for example in shuffled] == ["0", "1", "2"]
        assert all(
            example.images != originals[index].images for index, example in enumerate(shuffled)
        )

        blank_root = root / "blank"
        blank = blank_image_examples(originals, blank_root, rgb=(10, 20, 30))
        assert [example.id for example in blank] == ["0", "1", "2"]
        assert blank_root.stat().st_mode & 0o777 == 0o755
        with Image.open(blank[1].images[0]) as generated:
            assert generated.size == (5, 5)
            assert generated.getpixel((0, 0)) == (10, 20, 30)


def test_intervention_effect_reports_paired_drop() -> None:
    result = intervention_effect(np.array([1, 1, 0]), np.array([0, 1, 0]))
    assert np.isclose(result["mean_drop"], 1 / 3)
    assert np.isclose(result["fraction_harmed"], 1 / 3)
    assert result["correct_to_incorrect"] == 1
    assert result["both_correct"] == 1
    assert result["both_incorrect"] == 1


def test_intervention_response_effect_separates_formatting_changes() -> None:
    result = intervention_response_effect(
        ["Answer A", "blue car", "same"],
        [" answer   a ", "red car", "same"],
    )
    assert result["exact_response_changed"] == 2
    assert result["normalized_response_changed"] == 1
    assert result["exact_changed_indices"] == [0, 1]
    assert result["normalized_changed_indices"] == [1]


def test_shuffling_groups_duplicate_pixels_and_preserves_question_pairs(tmp_path) -> None:
    originals = []
    for index, color in enumerate([10, 10, 20, 20, 30, 30]):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (4, 5), (color, color, color)).save(path)
        originals.append(
            EvaluationExample(
                str(index), f"question {index}", (path,), ("yes",), group_id=f"pair-{index // 2}"
            )
        )
    shuffled = shuffled_image_examples(originals, seed=2026)
    assert shuffled == shuffled_image_examples(originals, seed=2026)
    for index, example in enumerate(shuffled):
        original = originals[index]
        assert (example.id, example.prompt, example.references, example.group_id) == (
            original.id,
            original.prompt,
            original.references,
            original.group_id,
        )
        assert (
            example.metadata["intervention_original_rgb_sha256"]
            != example.metadata["intervention_replacement_rgb_sha256"]
        )
    for index in (0, 2, 4):
        assert shuffled[index].images == shuffled[index + 1].images
    assert len({example.images for example in shuffled}) == 3


def test_shuffling_rejects_different_paths_with_identical_pixels(tmp_path) -> None:
    examples = []
    for index in range(2):
        path = tmp_path / f"same-{index}.png"
        Image.new("RGB", (3, 3), (4, 5, 6)).save(path)
        examples.append(EvaluationExample(str(index), "question", (path,), ("yes",)))
    with pytest.raises(ValueError, match="distinct RGB"):
        shuffled_image_examples(examples, seed=1)


def test_shuffling_rejects_duplicate_ids_before_loading_images() -> None:
    example = EvaluationExample("same", "question", (Path("unused.png"),), ("yes",))
    with pytest.raises(ValueError, match="unique example IDs"):
        shuffled_image_examples([example, example], seed=1)


def test_distinct_image_shuffling_preserves_the_previous_seeded_assignment(tmp_path) -> None:
    examples = []
    for index in range(5):
        path = tmp_path / f"distinct-{index}.png"
        Image.new("RGB", (3, 3), (index, 0, 0)).save(path)
        examples.append(EvaluationExample(str(index), "question", (path,), ("yes",)))
    old_order = shuffled_indices(len(examples), seed=2026)
    shuffled = shuffled_image_examples(examples, seed=2026)
    assert [example.images for example in shuffled] == [
        examples[index].images for index in old_order
    ]
