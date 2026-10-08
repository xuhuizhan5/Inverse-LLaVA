import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from invllava.artifacts.hashing import sha256_file
from invllava.config.identifiers import benchmark_protocol_id
from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import load_examples, prepare_vqav2
from invllava.eval.protocols.vqa import VQAv2Protocol
from invllava.eval.types import EvaluationExample
from invllava.eval.vqav2_data import materialize_vqav2_benchmark
from invllava.prompting import format_vicuna_v1_user_prompt
from scripts.verify_vqav2_golden import _official_evaluator, _rounded_equal


def _spec() -> BenchmarkSpec:
    payload = yaml.safe_load(Path("configs/benchmark/vqav2_val.yaml").read_text())
    payload["annotations"] = {
        "id": "vqav2-val-questions",
        "kind": "http",
        "location": "https://example.test/questions.zip",
        "revision": "v2",
        "sha256": "a" * 64,
        "required": True,
    }
    payload["images"] = {
        "id": "coco-val2014",
        "kind": "http",
        "location": "https://example.test/val2014.zip",
        "revision": "2014",
        "sha256": "b" * 64,
        "required": True,
    }
    payload["protocol_sources"] = [
        {
            "id": "vqav2-val-answers",
            "kind": "http",
            "location": "https://example.test/annotations.zip",
            "revision": "v2",
            "sha256": "c" * 64,
            "required": True,
        },
        {
            "id": "vqav2-official-evaluator",
            "kind": "http",
            "location": "https://example.test/vqaEval.py",
            "revision": "fixture",
            "sha256": "d" * 64,
            "required": True,
        },
    ]
    return BenchmarkSpec.model_validate(payload)


def test_prepare_vqav2_preserves_official_metadata_and_prompt(tmp_path) -> None:
    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    questions.write_text(
        json.dumps(
            {"questions": [{"question_id": 7, "image_id": 42, "question": "What is shown?"}]}
        )
    )
    annotations.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "question_id": 7,
                        "image_id": 42,
                        "answers": [{"answer": "cat"}] * 10,
                        "answer_type": "other",
                        "question_type": "what is this",
                    }
                ]
            }
        )
    )

    examples = prepare_vqav2(
        questions,
        Path("images"),
        coco_split="val2014",
        spec=_spec(),
        annotations_path=annotations,
    )

    assert examples == [
        EvaluationExample(
            id="7",
            prompt=format_vicuna_v1_user_prompt(
                "<image>\nWhat is shown?\nAnswer the question using a single word or phrase."
            ),
            images=(Path("images/COCO_val2014_000000000042.jpg"),),
            references=("cat",) * 10,
            metadata={
                "image_id": 42,
                "coco_split": "val2014",
                "answer_type": "other",
                "question_type": "what is this",
            },
        )
    ]


def test_vqav2_protocol_reports_official_breakdowns() -> None:
    examples = [
        EvaluationExample(
            "one",
            "",
            (),
            ("yes",) * 10,
            metadata={"answer_type": "yes/no", "question_type": "is there"},
        ),
        EvaluationExample(
            "two",
            "",
            (),
            ("two",) * 10,
            metadata={"answer_type": "number", "question_type": "how many"},
        ),
    ]
    score = VQAv2Protocol("fixture").score({"one": "yes", "two": "three"}, examples)

    assert score.value == 0.5
    assert score.details["answer_type_accuracy"] == {"number": 0.0, "yes/no": 1.0}
    assert score.details["question_type_accuracy"] == {"how many": 0.0, "is there": 1.0}


def test_vqav2_golden_loads_upstream_python2_class(tmp_path) -> None:
    source = tmp_path / "vqaEval.py"
    source.write_text(
        'class VQAEval:\n\tdef run(self):\n\t\tprint "computing accuracy"\n\t\treturn 7\n'
    )

    assert _official_evaluator(source)().run() == 7


def test_vqav2_golden_compares_at_upstream_percentage_precision() -> None:
    assert _rounded_equal(0.6669158878504673, 0.6669158878499999)
    assert not _rounded_equal(0.6669158878504673, 0.6669158)


def test_vqav2_http_materializer_is_relocatable_and_audited(tmp_path, monkeypatch) -> None:
    source_image = tmp_path / "COCO_val2014_000000000042.jpg"
    Image.new("RGB", (5, 4), (255, 0, 0)).save(source_image)
    questions_archive = tmp_path / "questions.zip"
    answers_archive = tmp_path / "annotations.zip"
    images_archive = tmp_path / "images.zip"
    evaluator = tmp_path / "vqaEval.py"
    evaluator.write_text("class VQAEval:\n    pass\n")
    with zipfile.ZipFile(questions_archive, "w") as bundle:
        bundle.writestr(
            "nested/v2_OpenEnded_mscoco_val2014_questions.json",
            json.dumps({"questions": [{"question_id": 7, "image_id": 42, "question": "What?"}]}),
        )
    with zipfile.ZipFile(answers_archive, "w") as bundle:
        bundle.writestr(
            "v2_mscoco_val2014_annotations.json",
            json.dumps(
                {
                    "annotations": [
                        {
                            "question_id": 7,
                            "image_id": 42,
                            "answers": [{"answer": "red"}] * 10,
                            "answer_type": "other",
                            "question_type": "what",
                        }
                    ]
                }
            ),
        )
    with zipfile.ZipFile(images_archive, "w") as bundle:
        bundle.write(source_image, f"val2014/{source_image.name}")

    spec = _spec()
    sources = {
        spec.annotations.location: questions_archive,
        spec.images.location: images_archive,
        spec.protocol_sources[0].location: answers_archive,
        spec.protocol_sources[1].location: evaluator,
    }
    spec = spec.model_copy(
        update={
            "annotations": spec.annotations.model_copy(
                update={"sha256": sha256_file(questions_archive)}
            ),
            "images": spec.images.model_copy(update={"sha256": sha256_file(images_archive)}),
            "protocol_sources": (
                spec.protocol_sources[0].model_copy(
                    update={"sha256": sha256_file(answers_archive)}
                ),
                spec.protocol_sources[1].model_copy(update={"sha256": sha256_file(evaluator)}),
            ),
        }
    )

    def fake_download(url, destination, *, sha256):
        source = sources[url]
        assert sha256_file(source) == sha256
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    monkeypatch.setattr("invllava.eval.vqav2_data.download_http", fake_download)
    monkeypatch.setattr("invllava.eval.vqav2_data._EXPECTED_ROWS", 1)
    monkeypatch.setattr("invllava.eval.vqav2_data._EXPECTED_IMAGES", 1)
    destination = tmp_path / "materialized"
    materialize_vqav2_benchmark(
        spec,
        destination,
        cache_dir=tmp_path / "cache",
        config_sha256="d" * 64,
    )

    examples = load_examples(destination / "examples.jsonl")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert examples[0].images[0] == destination / "images/COCO_val2014_000000000042.jpg"
    assert examples[0].metadata["answer_type"] == "other"
    assert manifest["sample_count"] == 1
    assert manifest["protocol_id"] == benchmark_protocol_id(spec)
    assert manifest["unique_image_count"] == 1
    assert manifest["image_integrity"]["passed"] is True
    assert manifest["sources"]["vqav2-val-answers"]["sha256"] == sha256_file(answers_archive)
    assert manifest["source_files"]["vqaEval.py"] == sha256_file(evaluator)


def test_vqav2_rejects_question_annotation_image_mismatch(tmp_path) -> None:
    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    questions.write_text(
        json.dumps({"questions": [{"question_id": 7, "image_id": 42, "question": "?"}]})
    )
    annotations.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "question_id": 7,
                        "image_id": 43,
                        "answers": [{"answer": "red"}] * 10,
                        "answer_type": "other",
                        "question_type": "what",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="image mismatch"):
        prepare_vqav2(
            questions,
            tmp_path,
            coco_split="val2014",
            spec=_spec(),
            annotations_path=annotations,
        )
