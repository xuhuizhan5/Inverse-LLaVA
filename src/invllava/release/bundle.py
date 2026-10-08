"""A small public loading interface for local and Hugging Face artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file, verify_sha256
from invllava.config.schema import ModelSpec
from invllava.eval.runtime import InverseGenerator
from invllava.eval.types import GenerationRequest
from invllava.model.adaptation import merge_lora_for_inference
from invllava.model.champion_checkpoint import RELEASE_CONFIG_FILENAME, RELEASE_FORMAT
from invllava.model.loaders import LoadReport, build_model, load_image_processor, load_tokenizer
from invllava.model.modeling import InverseLLaVAForConditionalGeneration
from invllava.prompting import format_vicuna_v1_user_prompt
from invllava.runtime.cache import configure_runtime_cache
from invllava.runtime.numerics import configure_torch_numerics_policy
from invllava.train.checkpoint import MODEL_DELTA_FILENAME, load_trainable_weights

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
TRAINING_RELEASE_SOURCE_FORMAT = "invllava-training-checkpoint-v1"


def _release_model_spec(value: dict[str, Any]) -> ModelSpec:
    """Read explicit no-op visual-projection fields from early v1 bundles.

    Checksums are verified before this in-memory normalization. A learned
    visual reducer or a different interface width is a different architecture
    and cannot be silently translated to the current model.
    """
    model = dict(value)
    fusion = dict(model.get("fusion", {}))
    if "project_visual_features" in fusion:
        if fusion.pop("project_visual_features") is not False:
            raise ValueError("release requires an unsupported visual-feature projection")
    if "projection_dim" in fusion:
        width = fusion.pop("projection_dim")
        if type(width) is not int or width != model.get("vision", {}).get("feature_dim"):
            raise ValueError("release projection width differs from its visual interface")
    model["fusion"] = fusion
    return ModelSpec.model_validate(model)


def export_training_checkpoint(
    run_dir: str | Path, checkpoint: str | Path, *, destination: str | Path
) -> Path:
    """Export a completed run using its own verified configuration and weights.

    No parent model card, tensor layout, or architecture is inherited. Base
    provenance comes from the run's pinned inputs; this is distinct from the
    tensor-by-tensor copied-projection audit used by imported checkpoints.
    """

    from invllava.artifacts.manifest import RunManifest
    from invllava.artifacts.run_integrity import verify_training_run

    root = Path(run_dir).resolve()
    source = Path(checkpoint).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    if source.parent != root / "checkpoints":
        raise ValueError("checkpoint must belong to the verified run")
    report = verify_training_run(root)
    if (source / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
        raise ValueError("checkpoint is incomplete")
    manifest = RunManifest.read(root / "manifest.json")
    config_path = root / "resolved_config.json"
    record = next(
        (item for item in manifest.artifacts if item.relative_path == config_path.name), None
    )
    if record is None or record.sha256 != sha256_file(config_path):
        raise ValueError("run manifest does not bind the resolved configuration")
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    spec = ModelSpec.model_validate(resolved["model"])
    if spec.architecture != "inverse_llava" or spec.adaptation.method == "full":
        raise ValueError("compact Inverse release requires a frozen base with adapter adaptation")
    if not spec.vision.freeze:
        raise ValueError("compact release requires a frozen vision encoder")
    inputs = {
        "language": f"{spec.language.checkpoint}@{spec.language.revision}",
        "vision": f"{spec.vision.checkpoint}@{spec.vision.revision}",
    }
    if any(manifest.checkpoint_inputs.get(key) != value for key, value in inputs.items()):
        raise ValueError("resolved configuration disagrees with pinned training inputs")
    if any(
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        for revision in (spec.language.revision, spec.vision.revision)
    ):
        raise ValueError("training export requires immutable upstream revisions")
    delta = source / MODEL_DELTA_FILENAME
    inventory = _tensor_inventory(delta)
    if not inventory:
        raise ValueError("checkpoint has no model tensors")
    state = json.loads((source / "state.json").read_text(encoding="utf-8"))
    metadata = {
        "format": RELEASE_FORMAT,
        "source_format": TRAINING_RELEASE_SOURCE_FORMAT,
        "experiment_id": manifest.experiment_id,
        "scientific_id": manifest.scientific_id,
        "method_revision": manifest.method_revision,
        "model_delta_sha256": sha256_file(delta),
        "model_delta_size_bytes": delta.stat().st_size,
        "tensor_count": len(inventory),
        "tensor_names": sorted(inventory),
        "training_provenance": {
            "kind": "pinned-training-inputs",
            "run_id": manifest.run_id,
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "resolved_config_sha256": record.sha256,
            "execution_source_sha256": manifest.execution_source_sha256,
            "checkpoint_inputs": manifest.checkpoint_inputs,
            "integrity": report.to_dict(),
            "checkpoint": source.name,
            "checkpoint_metadata_sha256": sha256_file(source / "metadata.json"),
            "checkpoint_state_sha256": sha256_file(source / "state.json"),
        },
        "training_summary": state,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        shutil.copyfile(delta, temporary / MODEL_DELTA_FILENAME)
        atomic_write_json(
            temporary / RELEASE_CONFIG_FILENAME,
            {
                "format": RELEASE_FORMAT,
                "weights": MODEL_DELTA_FILENAME,
                "model": resolved["model"],
            },
        )
        atomic_write_json(temporary / "metadata.json", metadata)
        atomic_write_text(temporary / "COMPLETE", "complete\n")
        atomic_write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(temporary.iterdir())),
        )
        _verify_checksums(temporary)
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _validate_upstream_provenance(metadata: dict[str, Any], spec: ModelSpec) -> None:
    provenance = metadata.get("training_provenance", {})
    if provenance.get("kind") == "pinned-training-inputs":
        expected = {
            "language": f"{spec.language.checkpoint}@{spec.language.revision}",
            "vision": f"{spec.vision.checkpoint}@{spec.vision.revision}",
        }
        inputs = provenance.get("checkpoint_inputs", {})
        if (
            any(inputs.get(key) != value for key, value in expected.items())
            or spec.adaptation.method == "full"
            or not spec.vision.freeze
            or not provenance.get("integrity", {}).get("complete")
            or not _DIGEST.fullmatch(str(provenance.get("resolved_config_sha256", "")))
        ):
            raise ValueError("release has inconsistent pinned training provenance")
        return
    verification = metadata.get("upstream_base_projection_verification", {})
    if (
        verification.get("status") != "exact"
        or verification.get("base_model") != spec.language.checkpoint
        or verification.get("base_model_revision") != spec.language.revision
        or verification.get("omitted_from_release") is not True
    ):
        raise ValueError("release lacks exact verification of omitted upstream base projections")


@dataclass(frozen=True)
class PretrainedInverseLLaVA:
    """Loaded model plus the provenance needed to identify generated outputs."""

    model: InverseLLaVAForConditionalGeneration
    tokenizer: Any
    image_processor: Any
    generator: InverseGenerator
    model_spec: ModelSpec
    base_load_report: LoadReport
    release_root: Path
    release_revision: str | None
    checkpoint_sha256: str
    metadata: dict[str, Any]
    lora_execution: Literal["unmerged", "merged"]
    numerical_policy: dict[str, Any]

    @property
    def checkpoint_id(self) -> str:
        return f"ckpt-{self.checkpoint_sha256[:16]}"

    def answer(self, image: str | Path, question: str) -> str:
        """Run one Vicuna-v1 image/question request with the canonical prompt."""

        if not question.strip():
            raise ValueError("question must not be empty")
        prompt = format_vicuna_v1_user_prompt(f"<image>\n{question.strip()}")
        return self.generator(
            GenerationRequest(
                id="interactive",
                prompt=prompt,
                images=(Path(image),),
            )
        )


def _resolve_release(
    model_id_or_path: str | Path,
    *,
    revision: str | None,
    cache_root: Path | None,
    local_files_only: bool,
    token: str | bool | None,
) -> tuple[Path, str | None]:
    local = Path(model_id_or_path)
    if local.is_dir():
        return local.resolve(), None
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            str(model_id_or_path),
            revision=revision,
            cache_dir=(cache_root / "release_snapshots" if cache_root is not None else None),
            local_files_only=local_files_only,
            token=token,
            allow_patterns=(
                "COMPLETE",
                "README.md",
                "LICENSE*",
                "NOTICE*",
                "USE_POLICY.md",
                "checksums.sha256",
                "metadata.json",
                RELEASE_CONFIG_FILENAME,
                MODEL_DELTA_FILENAME,
            ),
        )
    ).resolve()
    resolved_revision = snapshot.name if snapshot.parent.name == "snapshots" else revision
    return snapshot, resolved_revision


def _verify_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    declared: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if separator != "  " or not _DIGEST.fullmatch(digest) or Path(filename).name != filename:
            raise ValueError(f"invalid release checksum line: {line!r}")
        if filename in declared:
            raise ValueError(f"duplicate release checksum entry: {filename}")
        declared[filename] = digest
    required = {"COMPLETE", "metadata.json", RELEASE_CONFIG_FILENAME, MODEL_DELTA_FILENAME}
    if not required.issubset(declared):
        raise ValueError(
            f"release checksums omit required files: {sorted(required - declared.keys())}"
        )
    for filename, digest in declared.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        verify_sha256(path, digest)


def _tensor_inventory(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read tensor names, shapes, and dtypes without materializing their payloads."""

    from safetensors import safe_open

    inventory: dict[str, tuple[tuple[int, ...], str]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            view = handle.get_slice(name)
            inventory[name] = (tuple(view.get_shape()), str(view.get_dtype()))
    return inventory


def seal_training_checkpoint(
    checkpoint: str | Path,
    *,
    parent_release: str | Path,
    destination: str | Path,
) -> Path:
    """Seal a complete training checkpoint for the canonical release loader.

    The parent release supplies the immutable architecture and upstream-model
    provenance. Exact tensor inventories prevent an unrelated or partial delta
    from being published under that identity.
    """

    source = Path(checkpoint).resolve()
    parent = Path(parent_release).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    for root, label in ((source, "checkpoint"), (parent, "parent release")):
        if not root.is_dir():
            raise NotADirectoryError(root)
        if (root / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
            raise ValueError(f"{label} is incomplete: {root}")
    _verify_checksums(parent)

    source_files = {
        name: source / name for name in (MODEL_DELTA_FILENAME, "metadata.json", "state.json")
    }
    missing = [name for name, path in source_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training checkpoint is missing files: {missing}")
    parent_config = json.loads((parent / RELEASE_CONFIG_FILENAME).read_text(encoding="utf-8"))
    parent_metadata = json.loads((parent / "metadata.json").read_text(encoding="utf-8"))
    if (
        parent_config.get("format") != RELEASE_FORMAT
        or parent_metadata.get("format") != RELEASE_FORMAT
    ):
        raise ValueError("parent uses an unsupported Inverse-LLaVA release format")
    source_inventory = _tensor_inventory(source_files[MODEL_DELTA_FILENAME])
    parent_inventory = _tensor_inventory(parent / MODEL_DELTA_FILENAME)
    if source_inventory != parent_inventory:
        missing_tensors = sorted(set(parent_inventory) - set(source_inventory))
        extra_tensors = sorted(set(source_inventory) - set(parent_inventory))
        incompatible = sorted(
            name
            for name in set(source_inventory).intersection(parent_inventory)
            if source_inventory[name] != parent_inventory[name]
        )
        raise ValueError(
            "training and parent tensor inventories disagree; "
            f"missing={missing_tensors[:8]}, extra={extra_tensors[:8]}, "
            f"incompatible={incompatible[:8]}"
        )

    state = json.loads(source_files["state.json"].read_text(encoding="utf-8"))
    training_metadata = json.loads(source_files["metadata.json"].read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        shutil.copyfile(source_files[MODEL_DELTA_FILENAME], temporary / MODEL_DELTA_FILENAME)
        shutil.copyfile(parent / RELEASE_CONFIG_FILENAME, temporary / RELEASE_CONFIG_FILENAME)
        if (parent / "README.md").is_file():
            shutil.copyfile(parent / "README.md", temporary / "README.md")
        delta_digest = sha256_file(temporary / MODEL_DELTA_FILENAME)
        metadata = {
            **parent_metadata,
            "source_format": TRAINING_RELEASE_SOURCE_FORMAT,
            "model_delta_sha256": delta_digest,
            "model_delta_size_bytes": (temporary / MODEL_DELTA_FILENAME).stat().st_size,
            "tensor_count": len(source_inventory),
            "tensor_names": sorted(source_inventory),
            "parent_release": {
                "model_delta_sha256": sha256_file(parent / MODEL_DELTA_FILENAME),
                "metadata_sha256": sha256_file(parent / "metadata.json"),
            },
            "training_checkpoint": {
                "model_delta_sha256": sha256_file(source_files[MODEL_DELTA_FILENAME]),
                "metadata_sha256": sha256_file(source_files["metadata.json"]),
                "state_sha256": sha256_file(source_files["state.json"]),
                "metadata": training_metadata,
                "state": state,
            },
            "training_summary": {
                "epoch": state.get("epoch"),
                "global_step": state.get("global_step"),
                "samples_seen": state.get("samples_seen"),
                "tokens_seen": state.get("tokens_seen"),
            },
        }
        atomic_write_json(temporary / "metadata.json", metadata)
        atomic_write_text(temporary / "COMPLETE", "complete\n")
        checksum_lines = [
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "checksums.sha256"
        ]
        atomic_write_text(temporary / "checksums.sha256", "\n".join(checksum_lines) + "\n")
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_pretrained(
    model_id_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    device: str = "cuda",
    dtype: Literal["bfloat16", "float16", "float32"] | None = None,
    attention_backend: str = "sdpa",
    generation_cache: Literal["kv", "full_recompute"] = "kv",
    lora_execution: Literal["unmerged", "merged"] = "unmerged",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> PretrainedInverseLLaVA:
    """Load a sealed release from a directory or Hub repository.

    Published weights are adapter/fusion deltas. The base Vicuna and CLIP
    revisions embedded in the release configuration are resolved separately,
    keeping the distributable artifact compact and its upstream licenses clear.
    """

    if lora_execution not in {"unmerged", "merged"}:
        raise ValueError("lora_execution must be 'unmerged' or 'merged'")
    cache_root = Path(cache_dir).resolve() if cache_dir is not None else None
    if cache_root is not None:
        configure_runtime_cache(cache_root)
    root, resolved_revision = _resolve_release(
        model_id_or_path,
        revision=revision,
        cache_root=cache_root,
        local_files_only=local_files_only,
        token=token,
    )
    if (root / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
        raise ValueError("release is incomplete")
    _verify_checksums(root)
    release_config = json.loads((root / RELEASE_CONFIG_FILENAME).read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if release_config.get("format") != RELEASE_FORMAT or metadata.get("format") != RELEASE_FORMAT:
        raise ValueError("unsupported Inverse-LLaVA release format")
    if release_config.get("weights") != MODEL_DELTA_FILENAME:
        raise ValueError("release configuration references an unsupported weight file")
    model_spec = _release_model_spec(release_config.get("model"))
    if dtype is not None:
        model_spec = ModelSpec.model_validate(
            {**model_spec.model_dump(mode="python"), "torch_dtype": dtype}
        )
    numerical_policy = configure_torch_numerics_policy(
        mixed_precision=("no" if model_spec.torch_dtype == "float32" else model_spec.torch_dtype),
        allow_tf32=True,
        deterministic_algorithms=False,
        cudnn_benchmark=False,
    )
    if model_spec.architecture != "inverse_llava":
        raise ValueError("release does not describe an Inverse-LLaVA model")
    for upstream_revision in (model_spec.language.revision, model_spec.vision.revision):
        if upstream_revision in {"", "main", "pending-freeze"}:
            raise ValueError("release contains a mutable upstream model revision")
    checkpoint_digest = sha256_file(root / MODEL_DELTA_FILENAME)
    if checkpoint_digest != metadata.get("model_delta_sha256"):
        raise ValueError("release metadata and model delta digest disagree")
    _validate_upstream_provenance(metadata, model_spec)

    model, load_report = build_model(
        model_spec,
        attention_backend=attention_backend,
        local_files_only=local_files_only,
        token=token,
    )
    if not isinstance(model, InverseLLaVAForConditionalGeneration):
        raise TypeError("release unexpectedly built a non-inverse architecture")
    load_trainable_weights(root, model)
    model.to(device).eval()
    if lora_execution == "merged":
        merged_lora = merge_lora_for_inference(model)
        if model_spec.adaptation.method == "lora" and not merged_lora:
            raise RuntimeError("release declared LoRA adaptation but no adapters were merged")
    tokenizer = load_tokenizer(
        model_spec,
        local_files_only=local_files_only,
        token=token,
    )
    image_processor = load_image_processor(
        model_spec.vision,
        local_files_only=local_files_only,
        token=token,
    )
    generator = InverseGenerator(
        model,
        tokenizer,
        image_processor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_to_square=model_spec.vision.aspect_ratio == "pad",
        cache_mode=generation_cache,
    )
    return PretrainedInverseLLaVA(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        generator=generator,
        model_spec=model_spec,
        base_load_report=load_report,
        release_root=root,
        release_revision=resolved_revision,
        checkpoint_sha256=checkpoint_digest,
        metadata=metadata,
        lora_execution=lora_execution,
        numerical_policy=numerical_policy,
    )
