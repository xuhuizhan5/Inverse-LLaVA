from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from PIL import Image

from invllava.config.schema import BenchmarkSpec
from invllava.eval.fetch import _convert
from invllava.prompting import format_vicuna_v1_user_prompt


def _spec(name: str) -> BenchmarkSpec:
    path = Path("configs/benchmark") / f"{name}.yaml"
    return BenchmarkSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_ai2d_converter_freezes_prompt_target_and_image() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        physical = root / "physical"
        physical.mkdir()
        logical = root / "logical"
        example, record = _convert(
            _spec("ai2d"),
            {
                "id": "diagram-1",
                "image": Image.new("RGB", (2, 2)),
                "question": "Which label?",
                "options": ["one", "two"],
                "answer": 1,
            },
            0,
            physical_image_root=physical,
            logical_image_root=logical,
        )
        assert example is not None and record is not None
        assert example.references == ("B",)
        assert example.prompt == format_vicuna_v1_user_prompt(
            "<image>\nWhich label?\nA. one\nB. two\n"
            "Answer with the option's letter from the given choices directly."
        )
        assert (physical / Path(record["file"]).name).is_file()


def test_converter_uses_the_declared_prompt_template() -> None:
    spec = _spec("ai2d").model_copy(update={"prompt_template": "<image>\nQ={question}\n{choices}"})
    with TemporaryDirectory() as directory:
        root = Path(directory)
        physical = root / "physical"
        physical.mkdir()
        example, _ = _convert(
            spec,
            {
                "id": "diagram-2",
                "image": Image.new("RGB", (2, 2)),
                "question": "Pick",
                "options": ["left", "right"],
                "answer": 0,
            },
            0,
            physical_image_root=physical,
            logical_image_root=root / "logical",
        )
        assert example is not None
        assert example.prompt == format_vicuna_v1_user_prompt("<image>\nQ=Pick\nA. left\nB. right")


def test_scienceqa_converter_matches_official_llava_prompt() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        physical = root / "physical"
        physical.mkdir()
        example, _ = _convert(
            _spec("scienceqa_img"),
            {
                "image": Image.new("RGB", (2, 2)),
                "question": "Which object is blue?",
                "choices": ["the circle", "the square"],
                "answer": 1,
                "hint": "Look at the shapes.",
            },
            0,
            physical_image_root=physical,
            logical_image_root=root / "logical",
            official_record={
                "id": "science-1",
                "image": "science-1/image.png",
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nContext: Look at the shapes.\n"
                            "Which object is blue?\nA. the circle\nB. the square"
                        ),
                    },
                    {"from": "gpt", "value": "B"},
                ],
            },
        )
        assert example is not None
        assert example.references == ("B",)
        assert example.prompt == format_vicuna_v1_user_prompt(
            "<image>\nContext: Look at the shapes.\n"
            "Which object is blue?\nA. the circle\nB. the square\n"
            "Answer with the option's letter from the given choices directly."
        )


def test_mmstar_converter_preserves_official_question_and_categories() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        physical = root / "physical"
        physical.mkdir()
        example, _ = _convert(
            _spec("mmstar"),
            {
                "index": 17,
                "image": Image.new("RGB", (2, 2)),
                "question": "Which item?\nA. circle\nB. square\nC. line\nD. dot",
                "answer": "B",
                "category": "fine-grained perception",
                "l2_category": "recognition",
            },
            0,
            physical_image_root=physical,
            logical_image_root=root / "logical",
        )
        assert example is not None
        assert example.id == "17"
        assert example.references == ("B",)
        assert example.metadata == {
            "category": "fine-grained perception",
            "l2_category": "recognition",
        }
        assert example.prompt == format_vicuna_v1_user_prompt(
            "<image>\nWhich item?\nA. circle\nB. square\nC. line\nD. dot\n"
            "Answer with the option's letter from the given choices directly"
        )


def test_textvqa_converter_matches_paper_prompt_and_official_answers() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        physical = root / "physical"
        physical.mkdir()
        row = {
            "question_id": 34602,
            "image_id": "003a8ae2ef43b901",
            "image": Image.new("RGB", (2, 2)),
            "question": "what is the brand of this camera?",
            "answers": ["dakota"] * 10,
            "ocr_tokens": ["DAKOTA", "DIGITAL"],
        }
        example, _ = _convert(
            _spec("textvqa"),
            row,
            0,
            physical_image_root=physical,
            logical_image_root=root / "logical",
            official_record={
                key: row[key] for key in ("question_id", "image_id", "question", "answers")
            },
        )
        assert example is not None
        assert example.id == "34602"
        assert example.references == ("dakota",) * 10
        assert example.metadata == {
            "image_id": "003a8ae2ef43b901",
            "ocr_token_count": 2,
        }
        assert example.prompt == format_vicuna_v1_user_prompt(
            "<image>\nWhat is the brand of this camera?\n"
            "Reference OCR token: DAKOTA, DIGITAL\n"
            "Answer the question using a single word or phrase."
        )


def test_textvqa_converter_rejects_mirror_drift() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        physical = root / "physical"
        physical.mkdir()
        row = {
            "question_id": 1,
            "image_id": "image-1",
            "image": Image.new("RGB", (2, 2)),
            "question": "what word?",
            "answers": ["open"] * 10,
            "ocr_tokens": ["OPEN"],
        }
        official = {
            "question_id": 1,
            "image_id": "image-1",
            "question": "what word?",
            "answers": ["closed"] * 10,
        }
        try:
            _convert(
                _spec("textvqa"),
                row,
                0,
                physical_image_root=physical,
                logical_image_root=root / "logical",
                official_record=official,
            )
        except ValueError as error:
            assert "answers" in str(error)
        else:
            raise AssertionError("TextVQA mirror drift must fail materialization")
