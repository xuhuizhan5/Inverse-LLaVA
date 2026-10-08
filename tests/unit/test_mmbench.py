import base64
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from PIL import Image

from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import load_examples, prepare_mmbench
from invllava.eval.mmbench_data import materialize_mmbench_benchmark
from invllava.eval.protocols.mmbench import MMBenchCircularProtocol, infer_mmbench_choice
from invllava.eval.types import EvaluationExample
from invllava.prompting import format_vicuna_v1_user_prompt


def _spec() -> BenchmarkSpec:
    path = Path("configs/benchmark/mmbench_en.yaml")
    return BenchmarkSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _rotation(sample_id: str, answer: str, *, group: str = "1") -> EvaluationExample:
    return EvaluationExample(
        sample_id,
        "prompt",
        (Path("image.jpg"),),
        (answer,),
        ("red", "blue", "green", "white"),
        group_id=group,
        metadata={"category": "attribute_reasoning"},
    )


def test_mmbench_choice_extraction_is_unambiguous() -> None:
    choices = ("red", "blue", "green", "white")
    assert infer_mmbench_choice("The answer is B.", choices) == "B"
    assert infer_mmbench_choice("blue", choices) == "B"
    assert infer_mmbench_choice("A or B", choices) is None


def test_mmbench_requires_every_rotation() -> None:
    examples = [
        _rotation("1", "A"),
        _rotation("1000001", "D"),
        _rotation("2000001", "C"),
        _rotation("3000001", "B"),
    ]
    score = MMBenchCircularProtocol("fixture").score(
        {"1": "A", "1000001": "D", "2000001": "C", "3000001": "A"},
        examples,
    )
    assert score.value == 0.0
    assert score.count == 1
    assert score.details["per_item"] == {"1": 0}


def test_mmbench_scores_surviving_rotations_in_official_group() -> None:
    score = MMBenchCircularProtocol("fixture").score(
        {"1": "A", "1000001": "D", "2000001": "C"},
        [_rotation("1", "A"), _rotation("1000001", "D"), _rotation("2000001", "C")],
    )
    assert score.value == 1.0


def test_mmbench_rejects_group_without_base_rotation() -> None:
    with pytest.raises(ValueError, match="lacks its base rotation"):
        MMBenchCircularProtocol("fixture").score({"1000001": "D"}, [_rotation("1000001", "D")])


def test_mmbench_preparation_preserves_circular_identity_and_hint() -> None:
    encoded = base64.b64encode(b"fixture-image").decode()
    header = "index\tg_index\timage\thint\tquestion\tA\tB\tanswer\tcategory\n"
    rows = (
        f"1\t1\t{encoded}\tlook closely\tcolor?\tred\tblue\tA\tattribute_reasoning\n"
        f"1000001\t1\t{encoded}\tlook closely\tcolor?\tblue\tred\tB\tattribute_reasoning\n"
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "mmbench.tsv"
        source.write_text(header + rows, encoding="utf-8")
        examples = prepare_mmbench(
            source,
            root / "images",
            spec=_spec(),
        )
        assert [example.group_id for example in examples] == ["1", "1"]
        assert examples[0].images == examples[1].images
        assert examples[0].prompt == format_vicuna_v1_user_prompt(
            "<image>\nHint: look closely\nQuestion: color?\nOptions:\nA. red\nB. blue\n"
            "Please select the correct answer from the options above."
        )


def test_mmbench_preparation_resolves_compact_image_references() -> None:
    encoded = base64.b64encode(b"fixture-image").decode()
    header = "index\tquestion\timage\tA\tB\tanswer\tcategory\n"
    rows = (
        f"1\tcolor?\t{encoded}\tred\tblue\tA\tattribute_reasoning\n"
        "1000001\tcolor?\t1\tblue\tred\tB\tattribute_reasoning\n"
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "mmbench.tsv"
        source.write_text(header + rows, encoding="utf-8")
        examples = prepare_mmbench(
            source,
            root / "physical",
            logical_image_root="images",
            spec=_spec(),
        )

        assert [example.images for example in examples] == [
            (Path("images/1.jpg"),),
            (Path("images/1.jpg"),),
        ]
        assert (root / "physical/1.jpg").read_bytes() == b"fixture-image"


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("hint", ["", "look closely", "nan"])
def test_llava_prompt_variant_preserves_raw_hint_and_language(tmp_path, language, hint):
    encoded = base64.b64encode(b"fixture-image").decode()
    source = tmp_path / "source.tsv"
    source.write_text(
        f"index\timage\thint\tquestion\tA\tB\tanswer\n1\t{encoded}\t{hint}\tcolor?\tred\tblue\tA\n"
    )
    spec = BenchmarkSpec.model_validate(
        yaml.safe_load(Path(f"configs/benchmark/mmbench_{language}_llava.yaml").read_text())
    )
    example = prepare_mmbench(source, tmp_path / "images", spec=spec)[0]
    prefix = "look closely\n" if hint == "look closely" else ""
    suffix = (
        "请直接回答选项字母。"
        if language == "cn"
        else "Answer with the option's letter from the given choices directly."
    )
    assert example.prompt == format_vicuna_v1_user_prompt(
        f"<image>\n{prefix}color?\nA. red\nB. blue\n{suffix}"
    )
    assert spec.generation.max_new_tokens == 1024


def test_mmbench_materializer_writes_audited_portable_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (3, 2), (17, 23, 42)).save(image_buffer, format="JPEG")
    encoded = base64.b64encode(image_buffer.getvalue()).decode()
    source = tmp_path / "MMBench_DEV_EN.tsv"
    source.write_text(
        "index\tquestion\timage\tA\tB\tanswer\tcategory\n"
        f"1\tcolor?\t{encoded}\tred\tblue\tA\tattribute_reasoning\n"
        "1000001\tcolor?\t1\tblue\tred\tB\tattribute_reasoning\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("invllava.eval.mmbench_data.download_http", lambda *_a, **_k: source)
    monkeypatch.setattr("invllava.eval.mmbench_data._EXPECTED_ROWS", 2)
    monkeypatch.setattr("invllava.eval.mmbench_data._EXPECTED_GROUPS", 1)

    destination = materialize_mmbench_benchmark(
        _spec(),
        tmp_path / "artifact",
        cache_dir=tmp_path / "cache",
        config_sha256="a" * 64,
    )
    examples = load_examples(destination / "examples.jsonl")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    assert len(examples) == 2
    assert examples[0].images == (destination / "images/1.jpg",)
    assert manifest["sample_count"] == 2
    assert manifest["group_count"] == 1
    assert manifest["image_integrity"]["passed"] is True
