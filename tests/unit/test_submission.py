import json

import pytest

from invllava.eval.records import PredictionRecord
from invllava.eval.submission import (
    build_mmvet_submission,
    build_vqav2_submission,
    parse_vqav2_result,
    submission_checkpoint_digest,
    write_submission_manifest,
)


def test_vqav2_server_result_retains_percentage_units():
    metrics = {"yes/no": 92.37, "number": 60.45, "other": 70.68, "overall": 78.45}
    assert parse_vqav2_result([{"test-dev": metrics}]) == metrics
    assert parse_vqav2_result([{"test-standard": metrics}], split="test-standard") == metrics


@pytest.mark.parametrize("value", [True, "78.45", None, float("nan"), float("inf"), -1, 101])
def test_vqav2_server_result_rejects_invalid_metric(value):
    metrics = {"yes/no": 50, "number": 50, "other": 50, "overall": value}
    with pytest.raises(ValueError, match="finite percentages"):
        parse_vqav2_result([{"test-dev": metrics}])


@pytest.mark.parametrize("payload", [[], {}, [{}, {}], [{"test-standard": {}}], [None]])
def test_vqav2_server_result_rejects_ambiguous_or_changed_split(payload):
    with pytest.raises(ValueError, match="specified VQAv2 split"):
        parse_vqav2_result(payload)


def test_vqav2_server_result_requires_all_four_metrics():
    with pytest.raises(ValueError, match="metric names"):
        parse_vqav2_result([{"test-dev": {"overall": 78.45}}])
    with pytest.raises(ValueError, match="unsupported"):
        parse_vqav2_result([], split="validation")


def test_submission_checkpoint_preserves_delta_digest():
    assert submission_checkpoint_digest("a" * 64) == "a" * 64


def test_submission_checkpoint_rejects_unverified_hf_locator():
    with pytest.raises(ValueError, match="checkpoint-snapshot"):
        submission_checkpoint_digest("/workspace/model@revision")


def test_submission_checkpoint_verifies_matching_snapshot(tmp_path, monkeypatch):
    from invllava.artifacts import hub_snapshot

    inventory = {"repo_id": "official/model", "resolved_revision": "revision", "files": []}
    seen = []

    def verified(path):
        seen.append(path)
        return inventory

    monkeypatch.setattr(hub_snapshot, "verify_hub_snapshot", verified)
    result = submission_checkpoint_digest(f"{tmp_path}@revision", tmp_path)
    assert len(result) == 64 and seen == [tmp_path]
    with pytest.raises(ValueError, match="revision"):
        submission_checkpoint_digest(f"{tmp_path}@other", tmp_path)
    with pytest.raises(ValueError, match="path"):
        submission_checkpoint_digest("/wrong/path@revision", tmp_path)


def mmvet_records():
    return [
        PredictionRecord(
            schema_version=1,
            protocol_id="mmvet-test",
            experiment_id="experiment",
            checkpoint_id="checkpoint",
            sample_id=f"v1_{index}",
            prompt="What?",
            prediction="  Two\n",
            references=("two<OR>2",),
        )
        for index in range(218)
    ]


def test_mmvet_submission_preserves_raw_text_and_excludes_references(tmp_path):
    destination = tmp_path / "submission.json"
    records = mmvet_records()
    build_mmvet_submission(records, {record.sample_id for record in records}, destination)
    result = json.loads(destination.read_text())
    assert result == {f"v1_{index}": "  Two\n" for index in range(218)}


def test_mmvet_submission_requires_full_v1_coverage(tmp_path):
    records = mmvet_records()[:217]
    with pytest.raises(ValueError, match="218"):
        build_mmvet_submission(
            records, {record.sample_id for record in records}, tmp_path / "a.json"
        )


def test_vqav2_submission_preserves_text_for_hidden_reference_branch(tmp_path) -> None:
    destination = tmp_path / "submission.json"
    record = PredictionRecord(
        schema_version=1,
        protocol_id="vqav2-test",
        experiment_id="experiment",
        checkpoint_id="checkpoint",
        sample_id="7",
        prompt="What?",
        prediction="  Two\n",
        references=(),
        metadata={},
    )

    build_vqav2_submission([record], {"7"}, destination)

    assert json.loads(destination.read_text()) == [{"question_id": 7, "answer": "Two"}]


def test_submission_manifest_accepts_content_addressed_source(tmp_path) -> None:
    destination = tmp_path / "manifest.json"
    write_submission_manifest(
        destination,
        {
            "checkpoint_sha256": "a" * 64,
            "protocol_id": "protocol",
            "dataset_split": "test-dev",
            "code_commit": None,
            "execution_source_sha256": "b" * 64,
        },
    )
    assert json.loads(destination.read_text())["execution_source_sha256"] == "b" * 64


def test_llava_vqav2_envelope_normalizes_only_present_predictions(tmp_path):
    record = PredictionRecord(1, "p", "e", "c", "7", "What?", "The TWO, cats!")
    output = tmp_path / "upload.json"
    build_vqav2_submission([record], {"7"}, output, full_test_question_ids=[9, 7, 8])
    assert json.loads(output.read_text()) == [
        {"question_id": 9, "answer": ""},
        {"question_id": 7, "answer": "2 cats"},
        {"question_id": 8, "answer": ""},
    ]


@pytest.mark.parametrize("ids", [[8, 9], [7, 7], ["7"], [True]])
def test_llava_vqav2_envelope_rejects_bad_coverage(tmp_path, ids):
    record = PredictionRecord(1, "p", "e", "c", "7", "What?", "two")
    with pytest.raises(ValueError, match="envelope"):
        build_vqav2_submission([record], {"7"}, tmp_path / "a.json", full_test_question_ids=ids)


def test_llava_vqav2_never_pads_missing_testdev_predictions(tmp_path):
    record = PredictionRecord(1, "p", "e", "c", "7", "What?", "two")
    with pytest.raises(ValueError):
        build_vqav2_submission(
            [record], {"7", "8"}, tmp_path / "a.json", full_test_question_ids=[7, 8, 9]
        )


def test_submission_manifest_rejects_missing_source_identity(tmp_path) -> None:
    with pytest.raises(ValueError, match="code commit or execution-source"):
        write_submission_manifest(
            tmp_path / "manifest.json",
            {
                "checkpoint_sha256": "a" * 64,
                "protocol_id": "protocol",
                "dataset_split": "test-dev",
            },
        )


@pytest.mark.parametrize("tamper", [None, "predictions", "examples"])
def test_package_mmvet_cli_binds_evaluated_bytes(tmp_path, monkeypatch, tamper):
    from types import SimpleNamespace

    from invllava.artifacts.hashing import sha256_file
    from invllava.artifacts.manifest import ArtifactRecord, RunManifest
    from invllava.cli import _load_benchmark, build_parser
    from invllava.config.identifiers import benchmark_protocol_id
    from invllava.eval.datasets import write_examples
    from invllava.eval.records import PredictionStore
    from invllava.eval.types import EvaluationExample

    protocol = "configs/benchmark/mmvet_gpt41_hosted.yaml"
    protocol_id = benchmark_protocol_id(_load_benchmark(protocol))
    examples_path, predictions_path = tmp_path / "examples.jsonl", tmp_path / "predictions.jsonl"
    examples = [EvaluationExample(f"v1_{i}", "What?", (), ("two",)) for i in range(218)]
    write_examples(examples, examples_path)
    store = PredictionStore(predictions_path, protocol_id=protocol_id, checkpoint_id="checkpoint")
    for example in examples:
        store.append(
            PredictionRecord(
                1,
                protocol_id,
                "experiment",
                "checkpoint",
                example.id,
                example.prompt,
                "  Two\n",
                references=example.references,
            )
        )
    evaluation_manifest = tmp_path / "evaluation.manifest.json"
    evaluation_manifest.write_text("fixture")
    manifest = SimpleNamespace(
        protocol_ids=[protocol_id],
        code_commit=None,
        code_dirty=None,
        execution_source_sha256="b" * 64,
        checkpoint_inputs={"checkpoint": "a" * 64},
        experiment_id="experiment",
        notes=[f"examples_sha256={sha256_file(examples_path)}"],
        artifacts=[
            ArtifactRecord(
                "immutable_predictions", predictions_path.name, sha256_file(predictions_path)
            )
        ],
    )
    monkeypatch.setattr(RunManifest, "read", lambda path: manifest)
    if tamper == "predictions":
        with predictions_path.open("a") as handle:
            handle.write("\n")
    if tamper == "examples":
        # Semantically identical JSON with different bytes still breaks the
        # exported artifact binding to the original evaluation manifest.
        rows = [json.loads(line) for line in examples_path.read_text().splitlines()]
        examples_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        manifest.notes = ["examples_sha256=" + "c" * 64]
    output = tmp_path / "submission.json"
    args = build_parser().parse_args(
        [
            "package-submission",
            protocol,
            "--predictions",
            str(predictions_path),
            "--examples",
            str(examples_path),
            "--evaluation-manifest",
            str(evaluation_manifest),
            "--output",
            str(output),
            "--output-manifest",
            str(tmp_path / "submission.manifest.json"),
            "--allow-unverified",
        ]
    )
    if tamper:
        with pytest.raises(ValueError, match="differ"):
            args.function(args)
        assert not output.exists()
    else:
        args.function(args)
        assert json.loads(output.read_text()) == {f"v1_{i}": "  Two\n" for i in range(218)}
