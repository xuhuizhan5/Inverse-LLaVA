import zipfile
from pathlib import Path

import yaml
from PIL import Image

from invllava.config.schema import BenchmarkSpec
from invllava.eval.gqa_data import _extract_images, build_gqa_examples
from invllava.eval.protocols.gqa import GQAProtocol, normalize_gqa_prediction
from invllava.eval.types import EvaluationExample
from invllava.prompting import format_vicuna_v1_user_prompt


def test_gqa_uses_llava_conversion_then_exact_equality() -> None:
    assert normalize_gqa_prediction("Red.") == "red"
    examples = [EvaluationExample("1", "", (), ("red",))]
    assert GQAProtocol("fixture").score({"1": "Red."}, examples).value == 1.0
    assert GQAProtocol("fixture").score({"1": "the red"}, examples).value == 0.0


def test_gqa_reports_official_question_type_breakdowns() -> None:
    examples = [
        EvaluationExample(
            "1",
            "",
            (),
            ("red",),
            metadata={"structural_type": "query", "semantic_type": "attr"},
        ),
        EvaluationExample(
            "2",
            "",
            (),
            ("no",),
            metadata={"structural_type": "verify", "semantic_type": "attr"},
        ),
    ]
    result = GQAProtocol("fixture").score({"1": "red", "2": "yes"}, examples)
    assert result.details["accuracy_by_type"] == {
        "structural": {"query": 1.0, "verify": 0.0},
        "semantic": {"attr": 0.5},
    }
    assert result.details["answer_type_accuracy"] == {"binary": 0.0, "open": 1.0}
    assert result.details["structural_accuracy"] == {"query": 1.0, "verify": 0.0}
    assert result.details["semantic_accuracy"] == {"attr": 0.5}


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec.model_validate(
        yaml.safe_load(Path("configs/benchmark/gqa.yaml").read_text(encoding="utf-8"))
    )


def test_gqa_join_freezes_official_prompt_answer_and_image() -> None:
    questions = {
        "one": {
            "imageId": "n123",
            "question": "Is it overcast?",
            "answer": "no",
            "types": {"structural": "verify", "semantic": "global", "detailed": "weather"},
        }
    }
    rows = [
        {
            "question_id": "one",
            "image": "n123.jpg",
            "text": ("Is it overcast?\nAnswer the question using a single word or phrase."),
        }
    ]
    examples = build_gqa_examples(questions, rows, spec=_spec())
    assert examples == [
        EvaluationExample(
            id="one",
            prompt=format_vicuna_v1_user_prompt(
                "<image>\nIs it overcast?\nAnswer the question using a single word or phrase."
            ),
            images=(Path("images/n123.jpg"),),
            references=("no",),
            metadata={
                "image_id": "n123",
                "structural_type": "verify",
                "semantic_type": "global",
                "detailed_type": "weather",
            },
        )
    ]


def test_gqa_extracts_only_referenced_images(tmp_path: Path) -> None:
    source = tmp_path / "images.zip"
    wanted = tmp_path / "wanted.jpg"
    unused = tmp_path / "unused.jpg"
    Image.new("RGB", (4, 3), (1, 2, 3)).save(wanted)
    Image.new("RGB", (4, 3), (4, 5, 6)).save(unused)
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.write(wanted, "images/n123.jpg")
        bundle.write(unused, "images/unused.jpg")
    output = tmp_path / "output"
    records = _extract_images(
        source,
        [EvaluationExample("one", "", (Path("images/n123.jpg"),), ("no",))],
        output,
        progress=None,
    )
    assert [path.name for path in output.iterdir()] == ["n123.jpg"]
    assert records[0]["file"] == "images/n123.jpg"
