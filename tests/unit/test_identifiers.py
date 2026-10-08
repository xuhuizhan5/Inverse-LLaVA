from invllava.config.identifiers import (
    benchmark_protocol_id,
    benchmark_protocol_payload,
    canonical_json,
    content_id,
    scientific_payload,
)
from invllava.config.schema import BenchmarkSpec


def test_identity_is_order_independent() -> None:
    left = {"b": 2, "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": 2}
    assert canonical_json(left) == canonical_json(right)
    assert content_id(left, prefix="test") == content_id(right, prefix="test")


def test_execution_paths_do_not_change_scientific_payload() -> None:
    left = {
        "model": {"id": "fixture"},
        "runtime": {"id": "a100-observed", "run_root": "/workspace/runs/first"},
    }
    right = {
        "model": {"id": "fixture"},
        "runtime": {"id": "h100-observed", "run_root": "/workspace/runs/main"},
    }
    assert scientific_payload(left) == scientific_payload(right)


def _benchmark(**overrides: object) -> BenchmarkSpec:
    payload = {
        "id": "fixture",
        "display_name": "Fixture",
        "split": "validation",
        "protocol_revision": "fixture-v1",
        "task_type": "vqa",
        "annotations": {
            "id": "answers",
            "kind": "http",
            "location": "https://primary.example/answers.json",
            "revision": "v1",
            "sha256": "a" * 64,
        },
        "images": {
            "id": "images",
            "kind": "http",
            "location": "https://primary.example/images.zip",
            "revision": "v1",
            "sha256": "b" * 64,
        },
        "prompt_template": "<image>\n{question}",
        "answer_extraction": "fixture-normalization",
        "scorer": "fixture-consensus",
        "verification_status": "unverified",
        "notes": "initial note",
    }
    payload.update(overrides)
    return BenchmarkSpec.model_validate(payload)


def test_benchmark_protocol_identity_excludes_descriptive_and_admission_state() -> None:
    left = _benchmark()
    right = _benchmark(
        display_name="Renamed fixture",
        notes="expanded documentation",
        verification_status="golden_verified",
    )

    assert benchmark_protocol_payload(left) == benchmark_protocol_payload(right)
    assert benchmark_protocol_id(left) == benchmark_protocol_id(right)


def test_benchmark_protocol_identity_uses_content_not_checksummed_mirror() -> None:
    left = _benchmark()
    right = _benchmark(
        annotations={
            "id": "renamed-answers",
            "kind": "http",
            "location": "https://mirror.example/answers.json",
            "revision": "mirror-label-v2",
            "sha256": "a" * 64,
        }
    )

    assert benchmark_protocol_id(left) == benchmark_protocol_id(right)


def test_benchmark_protocol_identity_changes_with_scientific_inputs() -> None:
    original = _benchmark()
    changed_prompt = _benchmark(prompt_template="<image>\nQuestion: {question}")
    changed_annotations = _benchmark(
        annotations={
            "id": "answers",
            "kind": "http",
            "location": "https://primary.example/answers.json",
            "revision": "v2",
            "sha256": "c" * 64,
        }
    )

    assert benchmark_protocol_id(original) != benchmark_protocol_id(changed_prompt)
    assert benchmark_protocol_id(original) != benchmark_protocol_id(changed_annotations)


def test_benchmark_protocol_source_order_is_not_scientific_identity() -> None:
    sources = [
        {
            "id": "evaluator",
            "kind": "http",
            "location": "https://example.test/evaluator.py",
            "revision": "v1",
            "sha256": "c" * 64,
        },
        {
            "id": "fixture",
            "kind": "http",
            "location": "https://example.test/fixture.zip",
            "revision": "v1",
            "sha256": "d" * 64,
        },
    ]
    left = _benchmark(protocol_sources=sources)
    right = _benchmark(protocol_sources=list(reversed(sources)))

    assert benchmark_protocol_id(left) == benchmark_protocol_id(right)
