#!/usr/bin/env python3
"""Fingerprint native adapter and multimodal initialization before ablations.

Run on a staging host with pinned backbones already cached. Models are loaded
sequentially on CPU. Common tensor hashes expose seed-consumption differences;
they do not establish equivalence of different architectures or forward paths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.source import execution_source_sha256
from invllava.config.loader import ConfigRepository
from invllava.model.loaders import build_model
from invllava.train.checkpoint import (
    MODEL_DELTA_FILENAME,
    checkpoint_parameter_names,
    load_trainable_weights,
)
from invllava.train.state import seed_everything


def _is_lora(name: str) -> bool:
    return any(marker in f".{name}." for marker in (".lora_a.", ".lora_b."))


def _is_interface(name: str) -> bool:
    return ".fusion." in name or name.startswith("multimodal_projector.")


def align_common_initialization(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
    *,
    reference_layer: int | None,
    candidate_layer: int | None,
    allow_feature_width_change: bool = False,
) -> dict[str, object]:
    """Match shared tensors while preserving architecture-specific initializers.

    Fusion tensors follow the block when its insertion depth changes. LoRA
    tensors retain their absolute language-layer identity. Different-shaped
    output projections and candidate-only gates keep their declared initializer.
    A conventional projector uses ``None`` for its fusion layer. Cross-interface
    comparisons copy every LoRA tensor and retain each interface's own weights.
    Validate the complete copy plan before changing any parameter.
    """

    cross_interface = (reference_layer is None) != (candidate_layer is None)
    source_prefix = f"language_model.model.layers.{reference_layer}.self_attn.fusion."
    target_prefix = f"language_model.model.layers.{candidate_layer}.self_attn.fusion."
    copied, retained, used = {}, {}, set()
    names = checkpoint_parameter_names(model)
    candidate_lora = {name for name in names if _is_lora(name)}
    reference_lora = {name for name in reference if _is_lora(name)}
    if not candidate_lora or candidate_lora != reference_lora:
        raise ValueError(
            "LoRA initialization targets differ: "
            f"missing={sorted(reference_lora - candidate_lora)}, "
            f"extra={sorted(candidate_lora - reference_lora)}"
        )
    copies = []
    for name, parameter in model.named_parameters():
        if name not in names:
            continue
        if cross_interface and _is_interface(name):
            retained[name] = "architecture-specific interface initializer"
            continue
        source_name = (
            source_prefix + name.removeprefix(target_prefix)
            if candidate_layer is not None and name.startswith(target_prefix)
            else name
        )
        source = reference.get(source_name)
        if source is None:
            if ".fusion.gate." not in name:
                raise ValueError(f"unexplained candidate-only initialization tensor: {name}")
            retained[name] = "candidate-only tensor"
        elif source.shape != parameter.shape:
            width_specific = allow_feature_width_change and any(
                marker in name for marker in (".fusion.text_to_vision.", ".fusion.visual_norm.")
            )
            if ".fusion.output." not in name and not width_specific:
                raise ValueError(f"unexpected shared initialization shape mismatch: {name}")
            if source.dtype != parameter.dtype:
                raise ValueError(f"shared initialization dtype mismatch: {name}")
            retained[name] = (
                "visual-width-specific initializer"
                if allow_feature_width_change
                else "operator-specific output shape"
            )
        else:
            if source.dtype != parameter.dtype:
                raise ValueError(f"shared initialization dtype mismatch: {name}")
            copied[name] = source_name
            used.add(source_name)
            copies.append((name, parameter, source))
    unused = reference.keys() - used
    if cross_interface and any(not _is_interface(name) for name in unused):
        raise ValueError("unexpected reference-only state in a cross-interface comparison")
    with torch.no_grad():
        for name, parameter, source in copies:
            parameter.copy_(source)
            if not torch.equal(parameter.detach().cpu(), source):
                raise RuntimeError(f"initialization copy failed: {name}")
    if not copied:
        raise ValueError("no compatible tensors shared with the initialization reference")
    return {
        "copied_from_reference": copied,
        "retained_candidate_initializers": retained,
        "unused_reference_tensors": sorted(unused),
    }


def validate_model_contract(reference, candidate, *, allow_feature_width_change=False) -> None:
    """Keep backbone and visual input contracts fixed in a matched audit."""

    if reference.language != candidate.language or reference.torch_dtype != candidate.torch_dtype:
        raise ValueError("shared initialization requires identical language/precision contracts")
    excluded = {"feature_layers"}
    if allow_feature_width_change:
        if (
            reference.architecture != "inverse_llava"
            or candidate.architecture != "inverse_llava"
            or reference.fusion != candidate.fusion
            or reference.adaptation != candidate.adaptation
        ):
            raise ValueError("feature-width audit requires matching Inverse fusion/adaptation")
        widths = [
            (spec.vision.feature_dim, len(spec.vision.feature_layers))
            for spec in (reference, candidate)
        ]
        if (
            any(width % layers for width, layers in widths)
            or len({width // layers for width, layers in widths}) != 1
        ):
            raise ValueError("feature width must match concatenated layers of the same encoder")
        excluded.add("feature_dim")
    if reference.vision.model_dump(exclude=excluded) != candidate.vision.model_dump(
        exclude=excluded
    ):
        raise ValueError("only CLIP feature-layer selection may differ in a matched vision audit")
    for spec in (reference, candidate):
        if spec.architecture == "inverse_llava":
            if len(spec.fusion.layers) != 1:
                raise ValueError("initialization alignment requires one Inverse-LLaVA fusion block")
        elif spec.architecture == "llava_reference":
            if (
                spec.projector is None
                or spec.projector.initialization != "random"
                or spec.projector.initial_checkpoint_id is not None
            ):
                raise ValueError("a one-stage comparison requires a random, untrained projector")
        else:
            raise ValueError(f"unsupported initialization architecture: {spec.architecture}")


def load_initialization_reference(path: Path, model: torch.nn.Module, resolved) -> dict:
    """Use an exact saved untrained anchor when platform RNG implementations differ."""
    metadata = json.loads((path / "metadata.json").read_text())
    if (
        metadata.get("artifact_kind") != "untrained_initialization"
        or metadata.get("trained_updates") != 0
    ):
        raise ValueError("initialization reference must be explicitly untrained")
    if metadata.get("seed") != resolved.training.seed:
        raise ValueError("initialization reference seed differs")
    if metadata.get("resolved_model") != resolved.model.model_dump(mode="json"):
        raise ValueError("initialization reference model contract differs")
    digest = sha256_file(path / MODEL_DELTA_FILENAME)
    if digest != metadata.get("delta_sha256"):
        raise ValueError("initialization reference checksum differs")
    load_trainable_weights(path, model)
    return {
        "path": str(path),
        "delta_sha256": digest,
        "metadata_sha256": sha256_file(path / "metadata.json"),
    }


def save_initialization(
    destination: Path, model: torch.nn.Module, experiment: str, metadata: dict
) -> str:
    """Publish a strict-loadable delta with zero trained updates and no optimizer."""

    from safetensors.torch import save_file

    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        names = checkpoint_parameter_names(model)
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
            if name in names
        }
        save_file(state, temporary / MODEL_DELTA_FILENAME)
        digest = sha256_file(temporary / MODEL_DELTA_FILENAME)
        atomic_write_json(
            temporary / "metadata.json",
            {
                **metadata,
                "artifact_kind": "untrained_initialization",
                "trained_updates": 0,
                "delta_sha256": digest,
            },
        )
        recipe = yaml.safe_load(Path(experiment).read_text(encoding="utf-8"))
        recipe["initial_checkpoint_id"] = f"sha256:{digest}"
        atomic_write_text(temporary / "experiment.yaml", yaml.safe_dump(recipe, sort_keys=False))
        atomic_write_text(temporary / "COMPLETE", "complete\n")
        atomic_write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(temporary.iterdir())),
        )
        # Exercise the same strict reader used by a fresh training invocation.
        load_trainable_weights(temporary, model)
        os.replace(temporary, destination)
        return digest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--require-matched-common", action="store_true")
    parser.add_argument(
        "--allow-feature-width-change",
        action="store_true",
        help="match shared LoRA/scalars while retaining width-specific fusion initializers",
    )
    parser.add_argument("--aligned-output-dir", help="seal untrained deltas with shared tensors")
    parser.add_argument("--reference-audit", help="require the reference to match this prior audit")
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        help="load an exact untrained reference before aligning the other models",
    )
    args = parser.parse_args()
    if len(args.experiment) < 2 or args.threads <= 0:
        raise ValueError("provide at least two experiments and a positive thread count")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    repository = ConfigRepository(args.config_root)
    if len({Path(path).stem for path in args.experiment}) != len(args.experiment):
        raise ValueError("experiment filenames must have distinct stems")
    source, _, _ = execution_source_sha256(Path(__file__).resolve().parents[1])
    reference_state = None
    reference_layer = None
    reference_model = None
    records = []
    for experiment in args.experiment:
        resolved = repository.resolve(experiment)
        if records and resolved.training.seed != records[0]["seed"]:
            raise ValueError("shared-initialization experiments require the same seed")
        if resolved.initial_checkpoint_id:
            raise ValueError("initialization audit requires an untrained experiment")
        if reference_model is None:
            reference_model = resolved.model
        validate_model_contract(
            reference_model,
            resolved.model,
            allow_feature_width_change=args.allow_feature_width_change,
        )
        seed_everything(resolved.training.seed)
        model, _ = build_model(
            resolved.model,
            attention_backend=resolved.runtime.attention_backend,
            local_files_only=True,
        )
        alignment = None
        saved_reference = None
        if not records and args.reference_checkpoint:
            saved_reference = load_initialization_reference(
                args.reference_checkpoint, model, resolved
            )
        if reference_state is not None:
            alignment = align_common_initialization(
                model,
                reference_state,
                reference_layer=reference_layer,
                candidate_layer=(resolved.model.fusion.layers or (None,))[0],
                allow_feature_width_change=args.allow_feature_width_change,
            )
        parameters = {}
        owned_names = checkpoint_parameter_names(model)
        for name, parameter in model.named_parameters():
            if name not in owned_names:
                continue
            payload = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            parameters[name] = {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        if not parameters:
            raise ValueError(f"no adapter or fusion parameters in {experiment}")
        if not records and args.reference_audit:
            prior = json.loads(Path(args.reference_audit).read_text())["records"][0]
            if prior["seed"] != resolved.training.seed or prior["parameters"] != parameters:
                raise RuntimeError("reference differs from the previously audited initialization")
        initialization = None
        if args.aligned_output_dir:
            if reference_state is None:
                names = checkpoint_parameter_names(model)
                reference_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.named_parameters()
                    if name in names
                }
                reference_layer = (resolved.model.fusion.layers or (None,))[0]
            destination = Path(args.aligned_output_dir) / Path(experiment).stem
            digest = save_initialization(
                destination,
                model,
                experiment,
                {
                    "source_sha256": source,
                    "seed": resolved.training.seed,
                    "resolved_model": resolved.model.model_dump(mode="json"),
                    "reference_experiment": args.experiment[0],
                    "alignment": alignment,
                    "initialization_reference": saved_reference,
                },
            )
            repository.resolve(destination / "experiment.yaml")
            initialization = {"path": str(destination), "delta_sha256": digest}
        records.append(
            {
                "experiment": experiment,
                "seed": resolved.training.seed,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "parameters": parameters,
                "initialization": initialization,
                "alignment": alignment,
                "initialization_reference": saved_reference,
            }
        )
        del model, parameter, payload
        gc.collect()
        print(f"Audited {resolved.id}", flush=True)
    reference = records[0]["parameters"]
    comparisons = []
    for record in records[1:]:
        candidate = record["parameters"]
        common = sorted(reference.keys() & candidate.keys())
        changed = [
            name
            for name in common
            if any(
                reference[name][key] != candidate[name][key] for key in ("shape", "dtype", "sha256")
            )
        ]
        comparisons.append(
            {
                "experiment": record["experiment"],
                "common_count": len(common),
                "changed_tensors": changed,
                "reference_only": sorted(reference.keys() - candidate.keys()),
                "candidate_only": sorted(candidate.keys() - reference.keys()),
                "changed_trainability": [
                    name
                    for name in common
                    if reference[name]["requires_grad"] != candidate[name]["requires_grad"]
                ],
            }
        )
    matched = all(row["common_count"] > 0 and not row["changed_tensors"] for row in comparisons)
    atomic_write_json(
        output,
        {
            "schema_version": 1,
            "source_sha256": source,
            "torch_version": torch.__version__,
            "all_common_tensors_match": matched,
            "allow_feature_width_change": args.allow_feature_width_change,
            "reference": records[0]["experiment"],
            "records": records,
            "comparisons": comparisons,
            "scope": "CPU initialization only; identical tensors do not imply identical functions.",
        },
    )
    if args.require_matched_common and not matched:
        raise RuntimeError(f"shared initialization mismatch; see {output}")
    print(output, flush=True)


if __name__ == "__main__":
    main()
