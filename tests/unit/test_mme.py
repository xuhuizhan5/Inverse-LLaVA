from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import canonicalize_mme_question, prepare_mme
from invllava.eval.protocols.mme import extract_mme_answer
from invllava.prompting import format_vicuna_v1_user_prompt


def _spec() -> BenchmarkSpec:
    path = Path("configs/benchmark/mme_perception.yaml")
    return BenchmarkSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _write_pair(path: Path) -> None:
    path.write_text(
        "Is there a cat? Please answer yes or no.\tYes\n"
        "Is there a dog? Please answer yes or no.\tNo\n",
        encoding="utf-8",
    )


def test_mme_preparation_supports_both_official_layouts() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory) / "MME_Benchmark_release_version" / "MME_Benchmark"
        for category in ("existence", "count", "position", "color", "posters"):
            category_root = root / category
            category_root.mkdir(parents=True)
            (category_root / "sample.jpg").write_bytes(b"image")
            _write_pair(category_root / "sample.txt")
        for category in ("celebrity", "scene", "landmark", "artwork", "OCR"):
            category_root = root / category
            (category_root / "images").mkdir(parents=True)
            (category_root / "questions_answers_YN").mkdir()
            (category_root / "images" / "sample.png").write_bytes(b"image")
            _write_pair(category_root / "questions_answers_YN" / "sample.txt")

        examples = prepare_mme(root.parent, domain="perception", spec=_spec())
        assert len(examples) == 20
        assert all(example.metadata["domain"] == "perception" for example in examples)
        assert examples[0].prompt == format_vicuna_v1_user_prompt(
            "<image>\nIs there a cat?\nAnswer the question using a single word or phrase."
        )


def test_mme_extracts_only_a_leading_binary_answer() -> None:
    assert extract_mme_answer("Yes, because it is visible.") == "yes"
    assert extract_mme_answer("No.") == "no"
    assert extract_mme_answer("y") == "yes"
    assert extract_mme_answer("n") == "no"
    assert extract_mme_answer("I think yes") is None


def test_mme_question_matches_released_llava_prompt() -> None:
    assert (
        canonicalize_mme_question("Is a cat visible?  Please answer yes or no.")
        == "Is a cat visible?\nAnswer the question using a single word or phrase."
    )
