import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from invllava.artifacts.hashing import sha256_file
from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import load_examples, prepare_vizwiz
from invllava.eval.protocols.vizwiz import VizWizProtocol, normalize_vizwiz_answer
from invllava.eval.types import EvaluationExample
from invllava.eval.vizwiz_data import materialize_vizwiz_benchmark, verify_llava_questions
from invllava.prompting import format_vicuna_v1_user_prompt

_LLAVA_QUESTION = (
    "What?\nWhen the provided information is insufficient, respond with 'Unanswerable'.\n"
    "Answer the question using a single word or phrase."
)


def _spec() -> BenchmarkSpec:
    path = Path("configs/benchmark/vizwiz.yaml")
    return BenchmarkSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_vizwiz_requires_answer_bearing_public_release(tmp_path) -> None:
    annotations = tmp_path / "test.json"
    annotations.write_text(json.dumps([{"image": "test.jpg", "question": "What?"}]))
    with pytest.raises(ValueError, match="April 2026"):
        prepare_vizwiz(
            annotations,
            tmp_path,
            spec=_spec(),
            llava_questions={},
        )


def test_vizwiz_public_release_preserves_ten_answers(tmp_path) -> None:
    annotations = tmp_path / "VQA_test.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "image": "test.jpg",
                    "question": "WHAT?",
                    "answers": [{"answer": "unanswerable"}] * 10,
                    "answerable": 0,
                }
            ]
        )
    )
    examples = prepare_vizwiz(
        annotations,
        tmp_path,
        spec=_spec(),
        llava_questions={"test.jpg": _LLAVA_QUESTION},
    )
    assert examples[0].id == "test.jpg"
    assert examples[0].references == ("unanswerable",) * 10
    assert examples[0].prompt == format_vicuna_v1_user_prompt("<image>\n" + _LLAVA_QUESTION)


def test_vizwiz_normalizer_preserves_upstream_apostrophe_quirks() -> None:
    assert normalize_vizwiz_answer("The TWO, cats!") == "2 cats"
    assert normalize_vizwiz_answer("Dont") == "don\u2019t"
    assert normalize_vizwiz_answer("1,000.5") == "1000.5"


def test_vizwiz_protocol_uses_prediction_only_normalization() -> None:
    example = EvaluationExample(
        id="one",
        prompt="question",
        images=(),
        references=("cat",) * 10,
        metadata={"answer_type": "other"},
    )
    score = VizWizProtocol("test").score({"one": "The cat!"}, [example])

    assert score.value == 1.0
    assert score.details["answer_type_accuracy"] == {"other": 1.0}


def test_vizwiz_http_materializer_is_relocatable_and_audited(tmp_path, monkeypatch) -> None:
    annotations = tmp_path / "source-answers.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "image": "test.jpg",
                    "question": "What?",
                    "answers": [{"answer": "red"}] * 10,
                    "answerable": 1,
                    "answer_type": "other",
                }
            ]
        ),
        encoding="utf-8",
    )
    source_image = tmp_path / "test.jpg"
    Image.new("RGB", (5, 4), (255, 0, 0)).save(source_image)
    archive = tmp_path / "source-images.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(source_image, "test/test.jpg")
    fixture = tmp_path / "llava-eval.zip"
    with zipfile.ZipFile(fixture, "w") as bundle:
        bundle.writestr(
            "vizwiz/llava_test.jsonl",
            json.dumps(
                {
                    "question_id": 0,
                    "image": "test.jpg",
                    "text": _LLAVA_QUESTION,
                }
            )
            + "\n",
        )

    spec = _spec()
    spec = spec.model_copy(
        update={
            "annotations": spec.annotations.model_copy(update={"sha256": sha256_file(annotations)}),
            "images": spec.images.model_copy(update={"sha256": sha256_file(archive)}),
            "protocol_sources": (
                spec.protocol_sources[0].model_copy(update={"sha256": sha256_file(fixture)}),
            ),
        }
    )
    sources = {
        spec.annotations.location: annotations,
        spec.images.location: archive,
        spec.protocol_sources[0].location: fixture,
    }

    def fake_download(url, destination, *, sha256):
        source = sources[url]
        assert sha256_file(source) == sha256
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    monkeypatch.setattr("invllava.eval.vizwiz_data.download_http", fake_download)
    monkeypatch.setattr("invllava.eval.vizwiz_data._EXPECTED_ROWS", 1)
    destination = tmp_path / "materialized"
    materialize_vizwiz_benchmark(
        spec,
        destination,
        cache_dir=tmp_path / "cache",
        config_sha256="a" * 64,
    )

    examples = load_examples(destination / "examples.jsonl")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert examples[0].images[0] == destination / "images/test.jpg"
    assert manifest["sample_count"] == 1
    assert manifest["unique_image_count"] == 1
    assert manifest["image_integrity"]["passed"] is True
    assert manifest["prompt_fixture"]["verified_prompts"] == 1


def test_vizwiz_fixture_rejects_changed_answer_instruction(tmp_path):
    fixture = tmp_path / "eval.zip"
    with zipfile.ZipFile(fixture, "w") as bundle:
        bundle.writestr(
            "vizwiz/llava_test.jsonl",
            json.dumps(
                {
                    "question_id": 0,
                    "image": "one.jpg",
                    "text": "What?\nShort answer.",
                }
            ),
        )
    example = EvaluationExample(
        id="one.jpg",
        prompt=format_vicuna_v1_user_prompt("<image>\nWhat?"),
        images=(Path("one.jpg"),),
        references=("cat",),
    )
    with pytest.raises(ValueError, match="prompt/image"):
        verify_llava_questions([example], fixture)


def test_vizwiz_fixture_rejects_duplicate_question_ids(tmp_path):
    fixture = tmp_path / "eval.zip"
    with zipfile.ZipFile(fixture, "w") as bundle:
        bundle.writestr(
            "vizwiz/llava_test.jsonl",
            "\n".join(
                json.dumps(
                    {
                        "question_id": 0,
                        "image": name,
                        "text": "What?",
                    }
                )
                for name in ("one.jpg", "two.jpg")
            ),
        )
    with pytest.raises(ValueError, match="duplicate"):
        verify_llava_questions([], fixture)
