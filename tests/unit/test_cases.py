from pathlib import Path

from invllava.analysis.cases import materialize_case_images, select_multimodel_cases
from invllava.eval.types import EvaluationExample


def test_multimodel_case_selection_is_disjoint_and_complete() -> None:
    correct = {
        "inverse": {
            "all-right": True,
            "all-wrong": False,
            "inverse-only": True,
            "baselines-only": False,
            "mixed-right": True,
            "mixed-wrong": False,
        },
        "llava-lora": {
            "all-right": True,
            "all-wrong": False,
            "inverse-only": False,
            "baselines-only": True,
            "mixed-right": True,
            "mixed-wrong": True,
        },
        "llava-fft": {
            "all-right": True,
            "all-wrong": False,
            "inverse-only": False,
            "baselines-only": True,
            "mixed-right": False,
            "mixed-wrong": False,
        },
    }

    selected, patterns, counts = select_multimodel_cases(
        correct,
        primary_model="inverse",
        per_group=2,
        seed=2026,
    )

    assert selected == {
        "all_correct": ["all-right"],
        "all_wrong": ["all-wrong"],
        "inverse_only": ["inverse-only"],
        "inverse_with_baseline_disagreement": ["mixed-right"],
        "baseline_disagreement": ["mixed-wrong"],
        "baselines_only": ["baselines-only"],
    }
    assert set(patterns) == set(correct["inverse"])
    assert counts == {key: 1 for key in selected}


def test_multimodel_case_selection_rejects_misaligned_ids() -> None:
    try:
        select_multimodel_cases(
            {"inverse": {"a": True}, "llava": {"b": True}},
            primary_model="inverse",
        )
    except ValueError as error:
        assert "identical" in str(error)
    else:
        raise AssertionError("misaligned model scores must be rejected")


def test_case_images_are_self_contained_and_content_addressed(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image fixture")
    examples = {
        "one": EvaluationExample("one", "", (source,)),
        "two": EvaluationExample("two", "", (source,)),
    }
    destination = tmp_path / "case-images"
    paths, inventory = materialize_case_images(examples, ["one", "two"], destination)

    assert paths["one"] == paths["two"]
    assert Path(paths["one"][0]).is_file()
    assert destination.stat().st_mode & 0o777 == 0o755
    assert len(inventory) == 1
    assert inventory[0]["sample_ids"] == ["one", "two"]
