"""Strict scientific and execution configuration contracts.

Unknown fields are errors. This is intentional: a misspelled ablation setting
must fail before paid training rather than silently falling back to a default.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceClass(str, Enum):
    controlled = "controlled_architecture"
    official = "official_reference"
    contextual = "protocol_normalized_context"


class LanguageSpec(StrictModel):
    checkpoint: str
    revision: str = "pending-freeze"
    family: Literal["vicuna", "llama"] = "vicuna"
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    image_token: str = "<image>"
    max_length: int = Field(default=2048, gt=0)
    trust_remote_code: bool = False
    # Existing checkpoints retain their model-dtype frequency policy. A change
    # to FP32 is an explicit numerical intervention with a new run identity.
    rotary_precision: Literal["model", "float32"] = "model"


class VisionSpec(StrictModel):
    checkpoint: str = "openai/clip-vit-large-patch14-336"
    revision: str = "pending-freeze"
    feature_layers: tuple[int, ...] = (-1,)
    feature_select: Literal["patch", "cls_patch"] = "patch"
    feature_dim: int = Field(default=1024, gt=0)
    image_size: int = Field(default=336, gt=0)
    aspect_ratio: Literal["pad", "square"] = "pad"
    freeze: bool = True
    processor_backend: Literal["torchvision", "pil"] = "torchvision"


class FusionSpec(StrictModel):
    layers: tuple[int, ...] = (0,)
    targets: tuple[Literal["q", "k", "v"], ...] = ("q", "k", "v")
    operator: Literal["concat", "add", "gated", "disabled"] = "concat"
    mapper_rank: int | None = Field(default=None, gt=0)
    trainable: bool = True
    mapper_trainable: bool = True
    visual_normalization: Literal["rms", "none"] = "rms"
    visual_norm_eps: float = Field(default=1e-6, gt=0)
    scale_mode: Literal["learned", "fixed"] = "learned"
    initial_scale: float = 1.0
    mapper_init_std: float | None = None
    output_init_std: float = Field(default=1e-4, gt=0)
    initialization_id: str

    @model_validator(mode="after")
    def validate_fusion(self) -> FusionSpec:
        if len(set(self.layers)) != len(self.layers) or any(layer < 0 for layer in self.layers):
            raise ValueError("fusion layers must be unique non-negative indices")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("fusion targets must be unique")
        return self


class AdaptationSpec(StrictModel):
    method: Literal["lora", "full", "frozen"] = "lora"
    rank: int = Field(default=128, ge=0)
    alpha: int = Field(default=256, ge=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    trainable: bool = True
    target_suffixes: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    exclude_fusion_layers: bool = True
    excluded_layer_indices: tuple[int, ...] = ()


class ProjectorSpec(StrictModel):
    """Early vision-to-language projector used only by the controlled baseline."""

    kind: Literal["mlp2x_gelu"] = "mlp2x_gelu"
    input_dim: int = Field(gt=0)
    output_dim: int = Field(gt=0)
    trainable: bool = True
    initial_checkpoint_id: str | None = None
    initialization: Literal["checkpoint", "random"] = "checkpoint"

    @model_validator(mode="after")
    def validate_initialization(self) -> ProjectorSpec:
        if (self.initialization == "checkpoint") != bool(self.initial_checkpoint_id):
            raise ValueError("checkpoint initialization requires an identity; random forbids one")
        return self


class ModelSpec(StrictModel):
    id: str
    architecture: Literal["inverse_llava", "llava_reference"]
    language: LanguageSpec
    vision: VisionSpec
    fusion: FusionSpec
    projector: ProjectorSpec | None = None
    adaptation: AdaptationSpec
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    @model_validator(mode="after")
    def validate_dimensions(self) -> ModelSpec:
        if any(layer >= self.language.num_layers for layer in self.fusion.layers):
            raise ValueError("fusion layer exceeds language-model depth")
        excluded = self.adaptation.excluded_layer_indices
        if len(excluded) != len(set(excluded)) or any(
            index < 0 or index >= self.language.num_layers for index in excluded
        ):
            raise ValueError("LoRA excluded layers must be unique valid language-layer indices")
        if self.architecture == "inverse_llava" and self.projector is not None:
            raise ValueError("Inverse-LLaVA must not contain an early vision-to-language projector")
        if self.architecture == "llava_reference":
            if self.projector is None:
                raise ValueError("native LLaVA references require an explicit projector contract")
            if self.projector.input_dim != self.vision.feature_dim:
                raise ValueError("projector input dimension must match the selected visual feature")
            if self.projector.output_dim != self.language.hidden_size:
                raise ValueError("projector output dimension must match the language hidden size")
            if self.fusion.operator != "disabled" or self.fusion.layers:
                raise ValueError("the early-projector reference must not enable inverse fusion")
        return self


class DataSource(StrictModel):
    id: str
    kind: Literal["huggingface", "http", "manual", "generated"]
    location: str
    repo_type: Literal["dataset", "model"] = "dataset"
    subset: str | None = None
    revision: str | None = None
    sha256: str | None = None
    required: bool = True


class DataSpec(StrictModel):
    id: str
    annotation: DataSource
    image_sources: tuple[DataSource, ...] = ()
    split: str
    include_sources: tuple[str, ...] = ()
    sample_fraction: float = Field(default=1.0, gt=0, le=1)
    sample_seed: int = 17
    max_samples: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_source_filter(self) -> DataSpec:
        if any(not source.strip() for source in self.include_sources):
            raise ValueError("included data sources must be non-empty")
        if tuple(sorted(set(self.include_sources))) != self.include_sources:
            raise ValueError("included data sources must be sorted and unique")
        return self


class DataLayoutEntry(StrictModel):
    source: Path
    destination: Path


class DataLayoutSpec(StrictModel):
    id: str
    entries: tuple[DataLayoutEntry, ...]

    @model_validator(mode="after")
    def validate_layout(self) -> DataLayoutSpec:
        if not self.entries:
            raise ValueError("data layout must contain at least one entry")
        sources = [entry.source.as_posix() for entry in self.entries]
        destinations = [entry.destination.as_posix() for entry in self.entries]
        for label, values in (("source", sources), ("destination", destinations)):
            if len(values) != len(set(values)):
                raise ValueError(f"data layout contains duplicate {label} paths")
            for value in values:
                path = Path(value)
                if path.is_absolute() or ".." in path.parts or value in {"", "."}:
                    raise ValueError(f"data layout contains unsafe {label} path: {value}")
        ordered = sorted(destinations)
        for index, value in enumerate(ordered):
            prefix = value.rstrip("/") + "/"
            if any(other.startswith(prefix) for other in ordered[index + 1 :]):
                raise ValueError(f"data layout destinations overlap: {value}")
        return self


class OptimizerSpec(StrictModel):
    name: Literal["adamw"] = "adamw"
    update_dtype: Literal["float32", "model"] = "float32"
    learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0)
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)


class TrainingSpec(StrictModel):
    seed: int = 42
    epochs: float = Field(default=1.0, gt=0)
    per_device_batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    gradient_checkpointing: bool = True
    group_by_modality_length: bool = True
    max_grad_norm: float = Field(default=1.0, gt=0)
    log_every_steps: int = Field(default=1, gt=0)
    save_every_steps: int = Field(default=250, gt=0)
    checkpoint_milestones: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
    optimizer: OptimizerSpec = OptimizerSpec()

    @model_validator(mode="after")
    def validate_milestones(self) -> TrainingSpec:
        if any(x <= 0 or x > 1 for x in self.checkpoint_milestones):
            raise ValueError("checkpoint milestones must be fractions in (0, 1]")
        if tuple(sorted(set(self.checkpoint_milestones))) != self.checkpoint_milestones:
            raise ValueError("checkpoint milestones must be sorted and unique")
        if not self.checkpoint_milestones or self.checkpoint_milestones[-1] != 1.0:
            raise ValueError("checkpoint milestones must include the final 1.0 checkpoint")
        return self


class KernelOptimizationSpec(StrictModel):
    """Optional execution optimization; never part of scientific identity."""

    compile_scope: Literal["none", "fusion", "language_model"] = "none"
    compile_backend: Literal["inductor"] = "inductor"
    compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ] = "default"
    dynamic_shapes: bool = False
    fullgraph: bool = False
    require_triton: bool = True

    @model_validator(mode="after")
    def validate_compile_contract(self) -> KernelOptimizationSpec:
        if self.compile_scope == "none" and (
            self.compile_mode != "default" or self.dynamic_shapes or self.fullgraph
        ):
            raise ValueError("disabled kernel compilation must retain neutral compile settings")
        return self


class RuntimeSpec(StrictModel):
    id: str
    accelerator: Literal["cpu", "cuda"]
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    num_processes: int = Field(default=1, gt=0)
    dataloader_workers: int = Field(default=4, ge=0)
    dataloader_prefetch_factor: int = Field(default=2, ge=1)
    dataloader_persistent_workers: bool = True
    distributed_strategy: Literal["none", "ddp", "deepspeed_zero2"] = "none"
    attention_backend: Literal["sdpa", "eager"] = "sdpa"
    allow_tf32: bool = True
    deterministic_algorithms: bool = False
    cudnn_benchmark: bool = False
    cuda_allocator_conf: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.,:+-]+$",
    )
    run_root: Path
    cache_root: Path
    tracker: Literal["local", "tensorboard", "wandb"] = "local"
    hardware_telemetry_interval_seconds: float = Field(default=0.0, ge=0)
    keep_last_checkpoints: int = Field(default=2, ge=1)
    kernel_optimization: KernelOptimizationSpec = KernelOptimizationSpec()

    @model_validator(mode="after")
    def validate_topology(self) -> RuntimeSpec:
        if self.distributed_strategy == "none" and self.num_processes != 1:
            raise ValueError("distributed_strategy=none requires num_processes=1")
        if self.distributed_strategy == "ddp" and self.num_processes == 1:
            raise ValueError("distributed_strategy=ddp requires multiple processes")
        if self.accelerator != "cuda" and self.cuda_allocator_conf is not None:
            raise ValueError("cuda_allocator_conf requires accelerator=cuda")
        return self


class GenerationSpec(StrictModel):
    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    num_beams: int = Field(default=1, gt=0)


class ExperimentFile(StrictModel):
    id: str
    description: str
    method_revision: str
    model_ref: str
    data_ref: str
    runtime_ref: str
    evidence_class: EvidenceClass = EvidenceClass.controlled
    initial_checkpoint_id: str | None = None
    training: TrainingSpec = TrainingSpec()
    generation: GenerationSpec = GenerationSpec()
    tags: tuple[str, ...] = ()
    overrides: dict[str, Any] = Field(default_factory=dict)


class ExperimentVariantSpec(StrictModel):
    id_suffix: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    factor: str = Field(min_length=1)
    value: str = Field(min_length=1)
    description: str = Field(min_length=1)
    patches: dict[str, Any]


class ExperimentMatrixSpec(StrictModel):
    matrix_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    base_experiment: Path
    selection_metric: str = Field(min_length=1)
    seed: int
    variants: tuple[ExperimentVariantSpec, ...]

    @model_validator(mode="after")
    def validate_variants(self) -> ExperimentMatrixSpec:
        if not self.variants:
            raise ValueError("experiment matrix must contain at least one variant")
        identifiers = [variant.id_suffix for variant in self.variants]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("experiment matrix variant IDs must be unique")
        return self


class ResolvedExperiment(StrictModel):
    id: str
    description: str
    method_revision: str
    evidence_class: EvidenceClass
    initial_checkpoint_id: str | None = None
    model: ModelSpec
    data: DataSpec
    runtime: RuntimeSpec
    training: TrainingSpec
    generation: GenerationSpec
    tags: tuple[str, ...] = ()
    source_files: tuple[Path, ...]


class BenchmarkSpec(StrictModel):
    id: str
    display_name: str
    split: str
    protocol_revision: str
    task_type: Literal[
        "multiple_choice", "vqa", "exact_match", "mme", "judge", "external_submission"
    ]
    annotations: DataSource
    images: DataSource | None = None
    protocol_sources: tuple[DataSource, ...] = ()
    conversation_template: Literal["vicuna_v1"] = "vicuna_v1"
    prompt_template: str
    answer_extraction: str
    scorer: str
    external_only: bool = False
    generation: GenerationSpec = GenerationSpec()
    verification_status: Literal["unverified", "fixture_verified", "golden_verified"] = "unverified"
    notes: str = ""

    @model_validator(mode="after")
    def validate_visual_prompt(self) -> BenchmarkSpec:
        if self.prompt_template.count("<image>") != 1:
            raise ValueError("visual benchmark prompt_template must contain one <image> token")
        source_ids = [source.id for source in self.protocol_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("benchmark protocol_sources must have unique IDs")
        return self


class ReproductionEnvironmentSpec(StrictModel):
    dependency_lock: Path
    base_image_lock: Path
    publication_image_lock: Path


class ReproductionSpec(StrictModel):
    """The minimum evidence catalog for one named reproduction effort."""

    id: str
    experiments: tuple[Path, ...]
    local_benchmarks: tuple[Path, ...]
    judge_benchmarks: tuple[Path, ...] = ()
    external_benchmarks: tuple[Path, ...] = ()
    optional_benchmarks: tuple[Path, ...] = ()
    environment: ReproductionEnvironmentSpec

    @model_validator(mode="after")
    def validate_unique_references(self) -> ReproductionSpec:
        groups = (
            self.experiments,
            self.local_benchmarks,
            self.judge_benchmarks,
            self.external_benchmarks,
            self.optional_benchmarks,
        )
        flattened = [str(path) for group in groups for path in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("reproduction references must be unique across all groups")
        return self
