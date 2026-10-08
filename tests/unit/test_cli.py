import json
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from invllava.artifacts.hashing import sha256_file
from invllava.cli import (
    _downloads_authorized,
    _execution_run_root,
    _load_benchmark,
    build_parser,
    command_data_normalize,
    command_paired_interval,
    command_prepare_eval,
    command_score,
    command_stratify_scores,
)
from invllava.config.identifiers import benchmark_protocol_id
from invllava.eval.datasets import load_examples, validate_image_paths, write_examples
from invllava.eval.records import PredictionRecord, PredictionStore
from invllava.eval.types import EvaluationExample
from invllava.prompting import format_vicuna_v1_user_prompt


def test_execution_run_root_defaults_to_configured_path() -> None:
    configured = Path("/workspace/runs")
    assert _execution_run_root(configured, None) == configured


def test_default_reproduction_manifest_is_in_the_public_catalog() -> None:
    args = build_parser().parse_args(["reproduction", "audit"])
    assert args.manifest == "configs/reproduction/full.yaml"
    assert Path(args.manifest).is_file()


def test_shuffled_intervention_cli_records_image_group_policy(tmp_path: Path) -> None:
    examples = []
    for index in range(4):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (3, 3), (index // 2, 0, 0)).save(path)
        examples.append(EvaluationExample(str(index), "question", (path,), ("yes",)))
    source = tmp_path / "source.jsonl"
    output = tmp_path / "shuffled.jsonl"
    write_examples(examples, source)
    args = build_parser().parse_args(
        [
            "prepare-interventions",
            "--examples",
            str(source),
            "--mode",
            "shuffled",
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    )
    args.function(args)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["shuffle_unit"] == "decoded_rgb_image"
    assert manifest["image_group_count"] == 2
    assert manifest["examples_sha256"] == sha256_file(output)
    assert manifest["source_examples_sha256"] == sha256_file(source)
    restored = load_examples(output)
    assert restored[0].images == restored[1].images
    assert restored[2].images == restored[3].images
    with pytest.raises(FileExistsError, match="immutable"):
        args.function(args)


def test_execution_run_root_requires_absolute_override() -> None:
    try:
        _execution_run_root(Path("/workspace/runs"), "runs/pod-a")
    except ValueError as error:
        assert "absolute" in str(error)
    else:
        raise AssertionError("relative execution roots must be rejected")


def test_download_authorization_requires_flag_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(allow_download=True)
    monkeypatch.delenv("INVLLAVA_ALLOW_DOWNLOADS", raising=False)
    assert _downloads_authorized(args) is False
    monkeypatch.setenv("INVLLAVA_ALLOW_DOWNLOADS", "1")
    assert _downloads_authorized(args) is True
    args.allow_download = False
    assert _downloads_authorized(args) is False


def test_data_normalize_binds_a_source_filter(tmp_path: Path) -> None:
    annotation = tmp_path / "source.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": "ocr",
                    "source": "ocr_vqa",
                    "conversations": [
                        {"from": "human", "value": "Question?"},
                        {"from": "gpt", "value": "OCR answer"},
                    ],
                },
                {
                    "id": "caption",
                    "source": "coco",
                    "conversations": [
                        {"from": "human", "value": "Question?"},
                        {"from": "gpt", "value": "Caption answer"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"
    manifest = tmp_path / "normalized.manifest.json"
    command_data_normalize(
        Namespace(
            annotation=str(annotation),
            image_root=str(tmp_path),
            output=str(output),
            data_id="ocr-fixture",
            source_revision="fixture-1",
            default_source=None,
            source=["ocr_vqa"],
            manifest=str(manifest),
            image_audit=None,
            allow_unverified_images=False,
        )
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert [row["id"] for row in rows] == ["ocr"]
    assert metadata["source_filter"] == ["ocr_vqa"]
    assert metadata["audit"]["samples_by_source"] == {"ocr_vqa": 1}


def test_optional_runtime_override_is_exposed_only_where_it_is_meaningful() -> None:
    parser = build_parser()
    captured = parser.parse_args(
        [
            "capture-representations",
            "experiment.yaml",
            "--checkpoint",
            "checkpoint",
            "--examples",
            "examples.jsonl",
            "--runtime-ref",
            "cuda_1gpu",
            "--output",
            "representations.npz",
        ]
    )
    assert captured.runtime_ref == "cuda_1gpu"
    audited = parser.parse_args(
        [
            "kernel-audit",
            "configs/runtime/cuda_1gpu_inductor.yaml",
            "--output",
            "audit.json",
        ]
    )
    assert audited.dtype == "bfloat16"


def test_checkpoint_inventory_cli_has_one_immutable_output() -> None:
    parsed = build_parser().parse_args(
        ["checkpoint-inventory", "checkpoint", "--output", "inventory.json"]
    )
    assert parsed.checkpoint == "checkpoint"
    assert parsed.output == "inventory.json"


def test_hf_llava_profiler_requires_pinned_local_inputs() -> None:
    parsed = build_parser().parse_args(
        [
            "profile-hf-llava",
            "--checkpoint",
            "checkpoint",
            "--revision",
            "0123456789012345678901234567890123456789",
            "--examples",
            "examples.jsonl",
            "--output",
            "profile.json",
        ]
    )
    assert parsed.attention_backend == "sdpa"
    assert parsed.batch_sizes == (1, 4)


def test_native_profile_labels_inference_gradient_flags(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from invllava.cli import command_profile_native
    from invllava.config.loader import ConfigRepository

    examples_path = tmp_path / "examples.jsonl"
    write_examples([EvaluationExample("fixture", "Question", ())], examples_path)
    model = torch.nn.Module()
    model.language_model = torch.nn.Linear(2, 3, bias=False)
    model.language_model.weight.requires_grad_(False)
    resolved = ConfigRepository().resolve("configs/experiment/official_llava_lora_eval.yaml")
    runtime = SimpleNamespace(
        resolved=resolved,
        model=model,
        generator=SimpleNamespace(prepare_many=lambda _: None),
        checkpoint_sha256="a" * 64,
        kernel_report=SimpleNamespace(to_dict=lambda: {}),
        numerical_policy={},
    )
    monkeypatch.setattr(
        "invllava.runtime.native.load_native_inference_runtime", lambda **_: runtime
    )
    monkeypatch.setattr(
        "invllava.analysis.profiling.profile_autoregressive",
        lambda *_, **__: SimpleNamespace(to_dict=lambda: {}),
    )
    args = build_parser().parse_args(
        [
            "profile-native",
            "configs/experiment/official_llava_lora_eval.yaml",
            "--checkpoint",
            "unused",
            "--examples",
            str(examples_path),
            "--batch-sizes",
            "1",
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )
    command_profile_native(args)
    payload = json.loads((tmp_path / "profile.json").read_text())
    assert payload["total_parameters"] == 6
    assert payload["runtime_requires_grad_parameters"] == 0
    assert "trainable_parameters" not in payload


def test_language_evaluation_limit_is_explicitly_canary_only() -> None:
    parsed = build_parser().parse_args(
        [
            "language-eval",
            "configs/evaluation/language_retention.yaml",
            "--model-kind",
            "hf-causal",
            "--model",
            "model",
            "--revision",
            "0123456789012345678901234567890123456789",
            "--limit",
            "32",
            "--task",
            "arc_challenge",
            "--output",
            "result",
        ]
    )

    assert parsed.limit == 32
    assert parsed.task == ["arc_challenge"]


def test_language_interval_requires_metric_and_filter() -> None:
    parsed = build_parser().parse_args(
        [
            "language-interval",
            "--left-result",
            "left.json",
            "--right-result",
            "right.json",
            "--metric",
            "exact_match",
            "--filter",
            "flexible-extract",
            "--output",
            "interval.json",
        ]
    )

    assert parsed.metric == "exact_match"
    assert parsed.filter == "flexible-extract"


def test_score_strata_cli_requires_explicit_metadata_contract() -> None:
    parsed = build_parser().parse_args(
        [
            "stratify-scores",
            "--series",
            "Inverse",
            "inverse.json",
            "--series",
            "LLaVA",
            "llava.json",
            "--examples",
            "examples.jsonl",
            "--metadata-key",
            "ocr_token_count",
            "--image-audit",
            "audit.json",
            "--boundaries",
            "0",
            "1",
            "6",
            "11",
            "21",
            "--primary",
            "Inverse",
            "--output",
            "strata.json",
        ]
    )

    assert parsed.boundaries == [0.0, 1.0, 6.0, 11.0, 21.0]
    assert parsed.primary == "Inverse"
    assert parsed.image_audit == "audit.json"


@pytest.mark.parametrize("grouped", [False, True])
def test_score_strata_accepts_content_equivalent_protocol_ids(
    tmp_path: Path, grouped: bool
) -> None:
    examples = tmp_path / "examples.jsonl"
    write_examples(
        [
            EvaluationExample(
                id="sample-1",
                prompt="Read the text.",
                images=(tmp_path / "image.png",),
                metadata={"ocr_token_count": 0},
            )
        ],
        examples,
    )
    common = {
        "benchmark": "textvqa",
        "scorer_id": "textvqa-scorer",
        "examples_sha256": sha256_file(examples),
        "protocol_config_sha256": "2" * 64,
        "details": {"per_item": {"sample-1": 1.0}},
    }
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({**common, "protocol_id": "protocol-old"}), encoding="utf-8")
    right.write_text(json.dumps({**common, "protocol_id": "protocol-current"}), encoding="utf-8")
    output = tmp_path / "strata.json"
    audit = None
    if grouped:
        inventory = tmp_path / "inventory.jsonl"
        inventory.write_text(
            json.dumps({"logical_path": "image.png", "pixel_sha256": "a" * 64}) + "\n"
        )
        audit = tmp_path / "audit.json"
        audit.write_text(
            json.dumps(
                {
                    "passed": True,
                    "annotation_sha256": sha256_file(examples),
                    "image_root": str(tmp_path),
                    "inventory": {
                        "identity": "pixel_sha256",
                        "path": str(inventory),
                        "sha256": sha256_file(inventory),
                    },
                }
            )
        )

    command_stratify_scores(
        Namespace(
            series=[["left", str(left)], ["right", str(right)]],
            examples=str(examples),
            metadata_key="ocr_token_count",
            boundaries=[0.0],
            primary="left",
            resamples=100,
            confidence=0.95,
            seed=3,
            output=str(output),
            image_audit=str(audit) if audit else None,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol_id"] == "protocol-old"
    assert payload["source_protocol_ids"] == ["protocol-current", "protocol-old"]
    assert payload["protocol_equivalence"]["scorer_id"] == "textvqa-scorer"
    assert payload["resampling_unit"] == ("image_group" if grouped else "item")
    if grouped:
        assert payload["grouping"]["group_count"] == 1
        assert payload["grouping"]["audit_sha256"] == sha256_file(audit)


@pytest.mark.parametrize(
    "benchmark", ["mme-cognition", "mme-perception", "mmbench-en", "mmbench-cn", "mmstar"]
)
def test_numeric_strata_rejects_non_item_mean_benchmarks(tmp_path: Path, benchmark: str) -> None:
    score = tmp_path / "score.json"
    score.write_text(json.dumps({"benchmark": benchmark}))
    with pytest.raises(ValueError, match="item-mean"):
        command_stratify_scores(
            Namespace(
                series=[["left", str(score)], ["right", str(score)]],
                output=str(tmp_path / "out.json"),
            )
        )


def fixture_vqav2_protocol(tmp_path: Path, questions: Path) -> str:
    spec = _load_benchmark("configs/benchmark/vqav2_testdev.yaml").model_dump(mode="json")
    spec["annotations"]["sha256"] = sha256_file(questions)
    spec["protocol_revision"] = "unit-test-fixture"
    path = tmp_path / "fixture-protocol.yaml"
    path.write_text(json.dumps(spec))
    return str(path)


def test_prepare_eval_uses_protocol_yaml_and_writes_manifest(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps({"questions": [{"question_id": 7, "image_id": 3, "question": "What color?"}]}),
        encoding="utf-8",
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (4, 3), (255, 0, 0)).save(image_root / "COCO_test2015_000000000003.jpg")
    output = tmp_path / "examples.jsonl"
    manifest = tmp_path / "examples.manifest.json"

    command_prepare_eval(
        Namespace(
            benchmark=fixture_vqav2_protocol(tmp_path, questions),
            annotations=str(questions),
            image_root=str(image_root),
            coco_split=None,
            output=str(output),
            manifest=str(manifest),
            image_audit=None,
            image_audit_workers=1,
            image_audit_progress_every=0,
            allow_unverified_images=False,
        )
    )

    examples = load_examples(output)
    assert examples[0].prompt == format_vicuna_v1_user_prompt(
        "<image>\nWhat color?\nAnswer the question using a single word or phrase."
    )
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["benchmark_id"] == "vqav2-testdev"
    assert metadata["sample_count"] == 1
    assert metadata["examples_sha256"]
    assert metadata["image_integrity"]["passed"] is True
    assert metadata["image_integrity"]["decoded_images"] == 1


def test_prepare_eval_records_corrupt_image_before_inference(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps({"questions": [{"question_id": 7, "image_id": 3, "question": "What color?"}]}),
        encoding="utf-8",
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "COCO_test2015_000000000003.jpg").write_bytes(b"not-a-jpeg")
    output = tmp_path / "examples.jsonl"

    with pytest.raises(RuntimeError, match="image integrity failed"):
        command_prepare_eval(
            Namespace(
                benchmark=fixture_vqav2_protocol(tmp_path, questions),
                annotations=str(questions),
                image_root=str(image_root),
                coco_split=None,
                output=str(output),
                manifest=None,
                image_audit=None,
                image_audit_workers=1,
                image_audit_progress_every=0,
                allow_unverified_images=False,
            )
        )
    report = json.loads(output.with_suffix(".image-integrity.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["corrupt_images"] == 1
    assert not output.exists()


def test_manual_benchmark_rejects_different_question_file(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text('{"questions": []}')
    with pytest.raises(ValueError, match="manual annotations differ"):
        command_prepare_eval(
            Namespace(
                benchmark="configs/benchmark/vqav2_testdev.yaml",
                annotations=str(questions),
                image_root=str(tmp_path),
                output=str(tmp_path / "out.jsonl"),
                manifest=None,
                image_audit=None,
                allow_unverified_images=False,
            )
        )


def test_model_backed_evaluation_preflight_rechecks_images(tmp_path: Path) -> None:
    image = tmp_path / "later-corrupted.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
    example = EvaluationExample(
        id="one",
        prompt="<image>",
        images=(image,),
        references=("answer",),
    )
    validate_image_paths([example])
    payload = image.read_bytes()
    image.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(RuntimeError, match="evaluation image preflight failed"):
        validate_image_paths([example])


def _bind_interval_fixtures(examples: Path, *scores: Path) -> None:
    for path in scores:
        payload = json.loads(path.read_text())
        payload["examples_sha256"] = sha256_file(examples)
        path.write_text(json.dumps(payload))


def test_mme_cognition_interval_resamples_paired_image_groups(tmp_path: Path) -> None:
    ids = ("a/0", "a/1", "b/0", "b/1")
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    examples = tmp_path / "examples.jsonl"
    output = tmp_path / "interval.json"
    left.write_text(
        json.dumps(
            {
                "benchmark": "mme-cognition",
                "details": {"per_item": {sample_id: 1.0 for sample_id in ids}},
            }
        ),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps(
            {
                "benchmark": "mme-cognition",
                "details": {
                    "per_item": {
                        "a/0": 1.0,
                        "a/1": 0.0,
                        "b/0": 1.0,
                        "b/1": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    write_examples(
        [
            EvaluationExample(
                id=sample_id,
                prompt="fixture",
                images=(),
                references=("Yes",),
                group_id=sample_id.split("/", 1)[0],
                metadata={"category": "numerical_calculation"},
            )
            for sample_id in ids
        ],
        examples,
    )

    _bind_interval_fixtures(examples, left, right)
    command_paired_interval(
        Namespace(
            left_score=str(left),
            right_score=str(right),
            examples=str(examples),
            resamples=100,
            confidence=0.95,
            seed=3,
            output=str(output),
        )
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["unit"] == "mme_image_group"
    assert result["estimate"] == 150.0


def test_paired_interval_rejects_different_score_or_example_contracts(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(
        json.dumps(
            {
                "benchmark": "fixture",
                "scorer_id": "scorer-v1",
                "examples_sha256": "a" * 64,
                "details": {"per_item": {"sample": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps(
            {
                "benchmark": "fixture",
                "scorer_id": "scorer-v2",
                "examples_sha256": "a" * 64,
                "details": {"per_item": {"sample": 0.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different scorer_id"):
        command_paired_interval(
            Namespace(
                left_score=str(left),
                right_score=str(right),
                examples=None,
                resamples=100,
                confidence=0.95,
                seed=3,
                output=str(tmp_path / "interval.json"),
            )
        )


def test_mmstar_interval_resamples_within_l2_categories(tmp_path: Path) -> None:
    ids = ("a", "b", "c", "d")
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    examples = tmp_path / "examples.jsonl"
    output = tmp_path / "interval.json"
    left.write_text(
        json.dumps(
            {
                "benchmark": "mmstar",
                "details": {"per_item": {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps({"benchmark": "mmstar", "details": {"per_item": {item: 0.0 for item in ids}}}),
        encoding="utf-8",
    )
    write_examples(
        [
            EvaluationExample(
                id=sample_id,
                prompt="fixture",
                images=(),
                references=("A",),
                metadata={"l2_category": "large" if sample_id != "d" else "small"},
            )
            for sample_id in ids
        ],
        examples,
    )

    _bind_interval_fixtures(examples, left, right)
    command_paired_interval(
        Namespace(
            left_score=str(left),
            right_score=str(right),
            examples=str(examples),
            resamples=100,
            confidence=0.95,
            seed=3,
            output=str(output),
        )
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["estimate"] == 0.5
    assert result["unit"] == "fixed_strata_item"


def test_mmbench_interval_validates_rotations_and_resamples_groups(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    examples = tmp_path / "examples.jsonl"
    output = tmp_path / "interval.json"
    left.write_text(
        json.dumps({"benchmark": "mmbench-en", "details": {"per_item": {"a": 1.0, "b": 0.0}}}),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps({"benchmark": "mmbench-en", "details": {"per_item": {"a": 0.0, "b": 0.0}}}),
        encoding="utf-8",
    )
    write_examples(
        [
            EvaluationExample(
                id=f"{group}/{rotation}",
                prompt="fixture",
                images=(),
                references=("A",),
                choices=("one", "two"),
                group_id=group,
            )
            for group in ("a", "b")
            for rotation in range(2)
        ],
        examples,
    )

    _bind_interval_fixtures(examples, left, right)
    command_paired_interval(
        Namespace(
            left_score=str(left),
            right_score=str(right),
            examples=str(examples),
            resamples=100,
            confidence=0.95,
            seed=3,
            output=str(output),
        )
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["unit"] == "group"
    assert result["estimate"] == 0.5


def test_score_artifact_binds_inputs_and_is_immutable(tmp_path: Path) -> None:
    benchmark = Path("configs/benchmark/synthetic_contract.yaml")
    spec = _load_benchmark(benchmark)
    protocol_id = benchmark_protocol_id(spec)
    checkpoint_id = "fixture-checkpoint"
    prompt = format_vicuna_v1_user_prompt("<image>\nName the color.\nAnswer using a single word.")
    examples = tmp_path / "examples.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "score.json"
    write_examples(
        [EvaluationExample("fixture", prompt, (), ("blue",))],
        examples,
    )
    PredictionStore(
        predictions,
        protocol_id=protocol_id,
        checkpoint_id=checkpoint_id,
    ).append(
        PredictionRecord(
            1,
            protocol_id,
            "fixture-experiment",
            checkpoint_id,
            "fixture",
            prompt,
            "blue",
        )
    )
    arguments = Namespace(
        benchmark=str(benchmark),
        predictions=str(predictions),
        examples=str(examples),
        output=str(output),
        protocol_id=None,
        checkpoint_id=None,
        allow_unverified=True,
    )

    command_score(arguments)

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["checkpoint_id"] == checkpoint_id
    assert result["protocol_config_sha256"]
    assert result["predictions_sha256"]
    assert result["examples_sha256"]
    try:
        command_score(arguments)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing score artifact was overwritten")
