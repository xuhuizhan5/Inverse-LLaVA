import json
from pathlib import Path

from PIL import Image

from invllava.eval.datasets import load_examples, write_examples
from invllava.eval.sampling import materialize_evaluation_subset
from invllava.eval.types import EvaluationExample


def test_evaluation_subset_is_repeatable_and_self_contained(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    examples = []
    for index in range(8):
        image = source / f"{index}.png"
        Image.new("RGB", (2, 2), (index, 0, 0)).save(image)
        examples.append(
            EvaluationExample(
                id=str(index),
                prompt=f"question {index}",
                images=(image,),
                references=("A",),
                choices=("x", "y"),
            )
        )
    examples_path = source / "examples.jsonl"
    write_examples(examples, examples_path)

    first = materialize_evaluation_subset(examples_path, tmp_path / "subset-a", maximum=3, seed=17)
    second = materialize_evaluation_subset(examples_path, tmp_path / "subset-b", maximum=3, seed=17)
    first_examples = load_examples(first / "examples.jsonl")
    second_examples = load_examples(second / "examples.jsonl")
    assert [item.id for item in first_examples] == [item.id for item in second_examples]
    assert all(image.is_file() for item in first_examples for image in item.images)
    first_row = json.loads((first / "examples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_row["images"][0].startswith("images/")
    metadata = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert metadata["development_only"] is True
    assert metadata["sample_count"] == 3


def test_evaluation_subset_can_balance_metadata_strata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    examples = []
    for index in range(12):
        image = source / f"{index}.png"
        Image.new("RGB", (2, 2), (index, 0, 0)).save(image)
        examples.append(
            EvaluationExample(
                id=str(index),
                prompt=f"question {index}",
                images=(image,),
                references=("A",),
                choices=("x", "y"),
                metadata={"category": "rare" if index < 3 else "common"},
            )
        )
    examples_path = source / "examples.jsonl"
    write_examples(examples, examples_path)

    subset = materialize_evaluation_subset(
        examples_path,
        tmp_path / "balanced",
        maximum=6,
        seed=17,
        stratify_metadata="category",
    )
    metadata = json.loads((subset / "manifest.json").read_text(encoding="utf-8"))

    assert metadata["source_stratum_counts"] == {"common": 9, "rare": 3}
    assert metadata["selected_stratum_counts"] == {"common": 3, "rare": 3}


def test_evaluation_subset_preserves_complete_groups(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    examples = []
    for group_index in range(6):
        image = source / f"{group_index}.png"
        Image.new("RGB", (2, 2), (group_index, 0, 0)).save(image)
        for pair_index in range(2):
            examples.append(
                EvaluationExample(
                    id=f"{group_index}/{pair_index}",
                    prompt=f"question {group_index}/{pair_index}",
                    images=(image,),
                    references=("Yes",),
                    group_id=str(group_index),
                    metadata={"category": "a" if group_index < 3 else "b"},
                )
            )
    examples_path = source / "examples.jsonl"
    write_examples(examples, examples_path)

    subset = materialize_evaluation_subset(
        examples_path,
        tmp_path / "paired",
        maximum=7,
        seed=17,
        stratify_metadata="category",
        preserve_groups=True,
    )
    selected = load_examples(subset / "examples.jsonl")
    metadata = json.loads((subset / "manifest.json").read_text(encoding="utf-8"))
    group_counts: dict[str, int] = {}
    for example in selected:
        assert example.group_id is not None
        group_counts[example.group_id] = group_counts.get(example.group_id, 0) + 1

    assert len(selected) == 6
    assert set(group_counts.values()) == {2}
    assert metadata["preserve_groups"] is True
    assert metadata["selected_group_count"] == 3
    assert metadata["source_stratum_counts"] == {"a": 6, "b": 6}
    assert sum(metadata["selected_stratum_counts"].values()) == 6
