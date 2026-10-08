from pathlib import Path

import yaml

from invllava.config.identifiers import scientific_payload
from invllava.config.loader import ConfigRepository
from invllava.config.schema import BenchmarkSpec
from invllava.eval.language_suite import load_language_suite


def test_all_direct_experiment_configs_resolve() -> None:
    repository = ConfigRepository("configs")
    for path in Path("configs/experiment").glob("*.yaml"):
        if "matrix" not in path.name:
            repository.resolve(path)


def test_runtime_override_preserves_global_batch_and_scientific_identity() -> None:
    repository = ConfigRepository("configs")
    experiment = "configs/experiment/canonical_7b.yaml"
    declared = repository.resolve(experiment)
    two_gpu = repository.resolve(experiment, runtime_ref="cuda_2gpu")
    two_gpu_zero2 = repository.resolve(experiment, runtime_ref="cuda_2gpu_zero2")
    single_gpu = repository.resolve(experiment, runtime_ref="cuda_1gpu")
    optimized = repository.resolve(experiment, runtime_ref="cuda_1gpu_inductor")
    assert declared.runtime.num_processes == 2
    assert declared.runtime.distributed_strategy == "deepspeed_zero2"
    assert declared.training.gradient_accumulation_steps == 2
    assert two_gpu.runtime.num_processes == 2
    assert two_gpu.training.gradient_accumulation_steps == 2
    assert two_gpu_zero2.runtime.distributed_strategy == "deepspeed_zero2"
    assert two_gpu_zero2.training.gradient_accumulation_steps == 2
    assert single_gpu.runtime.num_processes == 1
    assert single_gpu.training.gradient_accumulation_steps == 4
    assert scientific_payload(single_gpu) == scientific_payload(declared)
    assert scientific_payload(two_gpu) == scientific_payload(declared)
    assert scientific_payload(two_gpu_zero2) == scientific_payload(declared)
    assert scientific_payload(optimized) == scientific_payload(declared)
    assert optimized.runtime.kernel_optimization.compile_scope == "fusion"


def test_microbatch_override_preserves_global_batch_and_is_recorded() -> None:
    repository = ConfigRepository("configs")
    experiment = "configs/experiment/paired_continuation_base.yaml"
    declared = repository.resolve(experiment)
    thor = repository.resolve(
        experiment,
        runtime_ref="thor_real_model",
        microbatch_size=1,
    )
    declared_global = (
        declared.training.per_device_batch_size
        * declared.training.gradient_accumulation_steps
        * declared.runtime.num_processes
    )
    thor_global = (
        thor.training.per_device_batch_size
        * thor.training.gradient_accumulation_steps
        * thor.runtime.num_processes
    )
    assert declared_global == thor_global == 128
    assert thor.training.per_device_batch_size == 1
    assert thor.training.gradient_accumulation_steps == 128
    assert scientific_payload(thor) != scientific_payload(declared)


def test_h100_execution_overrides_preserve_batch_and_record_checkpointing() -> None:
    repository = ConfigRepository("configs")
    experiment = "configs/experiment/canonical_7b.yaml"
    strict = repository.resolve(
        experiment,
        runtime_ref="cuda_2gpu_zero2",
        microbatch_size=32,
        gradient_checkpointing=True,
    )
    accelerated = repository.resolve(
        experiment,
        runtime_ref="cuda_2gpu_zero2",
        microbatch_size=64,
        gradient_checkpointing=False,
    )
    assert strict.training.per_device_batch_size == 32
    assert strict.training.gradient_accumulation_steps == 2
    assert strict.training.gradient_checkpointing is True
    assert accelerated.training.per_device_batch_size == 64
    assert accelerated.training.gradient_accumulation_steps == 1
    assert accelerated.training.gradient_checkpointing is False
    for resolved in (strict, accelerated):
        assert (
            resolved.training.per_device_batch_size
            * resolved.training.gradient_accumulation_steps
            * resolved.runtime.num_processes
            == 128
        )
    assert scientific_payload(accelerated) != scientific_payload(strict)


def test_maximum_sample_override_is_recorded() -> None:
    repository = ConfigRepository("configs")
    experiment = "configs/experiment/paired_continuation_base.yaml"
    declared = repository.resolve(experiment)
    bounded = repository.resolve(experiment, maximum_samples=128)
    assert declared.data.max_samples == 5580
    assert bounded.data.max_samples == 128
    assert scientific_payload(bounded) != scientific_payload(declared)


def test_continuation_controls_are_bound_to_the_accepted_parent_and_label_counter() -> None:
    from scripts.materialize_continuation_controls import CONTROL_REVISION

    repository = ConfigRepository("configs")
    resolved = repository.resolve("configs/experiment/paired_continuation_base.yaml")
    assert resolved.initial_checkpoint_id == (
        "sha256:9ed8914aedbbeb55d01607da0aab96af2d55f638e0b827f20ac232d0a4f6741f"
    )
    assert resolved.runtime.num_processes == 2
    for name in (
        "instruction_token_matched_1pct",
        "paired_instruction_mixed_1pct",
        "paired_shuffled_1pct",
    ):
        spec = yaml.safe_load(Path(f"configs/data/{name}.yaml").read_text())
        assert spec["annotation"]["revision"] == CONTROL_REVISION


def test_all_benchmark_configs_are_strict() -> None:
    for path in Path("configs/benchmark").glob("*.yaml"):
        BenchmarkSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_language_retention_suite_is_strict() -> None:
    suite = load_language_suite("configs/evaluation/language_retention.yaml")
    assert [task.id for task in suite.tasks] == [
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "gsm8k",
    ]
