from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from invllava import __version__
from invllava.artifacts.atomic import atomic_write_json
from invllava.config.identifiers import benchmark_protocol_id, content_id, scientific_payload
from invllava.config.loader import ConfigRepository
from invllava.config.schema import BenchmarkSpec, DataSpec
from invllava.runtime.cache import configure_runtime_cache


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=False))


def _execution_run_root(configured: Path, requested: str | None) -> Path:
    if requested is None:
        return configured
    candidate = Path(requested)
    if not candidate.is_absolute():
        raise ValueError("--run-root must be an absolute execution path")
    return candidate.resolve()


def _downloads_authorized(args: argparse.Namespace) -> bool:
    """Require the CLI flag and environment opt-in for network acquisition."""

    return bool(args.allow_download) and os.environ.get("INVLLAVA_ALLOW_DOWNLOADS") == "1"


def command_config_resolve(args: argparse.Namespace) -> None:
    resolved = ConfigRepository(args.config_root).resolve(
        args.experiment, runtime_ref=args.runtime_ref
    )
    payload = resolved.model_dump(mode="json")
    result = {
        "scientific_id": content_id(scientific_payload(resolved), prefix="sci"),
        "resolved": payload,
    }
    if args.output:
        atomic_write_json(args.output, result)
    _json(result)


def command_reproduction_audit(args: argparse.Namespace) -> None:
    from invllava.reproducibility import audit_reproduction

    report = audit_reproduction(
        args.manifest,
        stage=args.stage,
        repository_root=args.repository_root,
    )
    payload = report.to_dict()
    if args.output:
        atomic_write_json(args.output, payload)
    _json(payload)
    if args.strict and not report.passed:
        raise SystemExit(1)


def command_data_plan(args: argparse.Namespace) -> None:
    from invllava.data.registry import plan_sources

    spec = DataSpec.model_validate(
        yaml.safe_load(Path(args.data_config).read_text(encoding="utf-8"))
    )
    _json([item.__dict__ for item in plan_sources(spec, args.destination)])


def command_data_normalize(args: argparse.Namespace) -> None:
    from invllava.data.audit import audit_samples
    from invllava.data.manifest import (
        build_prepared_manifest,
        load_image_integrity_evidence,
    )
    from invllava.data.prepare import (
        iter_llava_json,
        qualify_reused_sample_ids,
        write_normalized_jsonl,
    )

    output_path = Path(args.output)
    manifest_path = Path(args.manifest or str(output_path.with_suffix(".manifest.json")))
    protected_inputs = {Path(args.annotation).resolve()}
    if args.image_audit:
        protected_inputs.add(Path(args.image_audit).resolve())
    if (
        output_path.resolve() == manifest_path.resolve()
        or {
            output_path.resolve(),
            manifest_path.resolve(),
        }
        & protected_inputs
    ):
        raise ValueError("normalized data, manifest, annotation, and image audit must be distinct")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("normalized data and manifest destinations must be new")

    source_samples = list(
        iter_llava_json(
            args.annotation,
            args.image_root,
            default_source=args.default_source,
        )
    )
    samples, id_normalization = qualify_reused_sample_ids(source_samples)
    source_filter = tuple(sorted(set(args.source or ())))
    if source_filter:
        known_sources = {sample.source for sample in samples}
        unknown_sources = set(source_filter) - known_sources
        if unknown_sources:
            raise ValueError(f"unknown --source values: {sorted(unknown_sources)}")
        samples = [sample for sample in samples if sample.source in source_filter]
    source_audit = audit_samples(samples, check_files=not args.allow_unverified_images)
    if source_audit.missing_images and not args.allow_unverified_images:
        raise ValueError(f"data audit failed: {source_audit.to_dict()}")
    audit = audit_samples(samples, check_files=False)
    if audit.duplicate_ids:
        raise RuntimeError("internal sample ID normalization did not produce unique IDs")
    image_integrity = None
    if audit.images:
        if args.image_audit:
            image_integrity = load_image_integrity_evidence(
                args.image_audit,
                annotation_path=args.annotation,
                source_revision=args.source_revision,
                image_root=args.image_root,
                selected_sources=source_filter,
            )
            if int(image_integrity["references"]) != audit.images:
                raise ValueError("image audit reference count does not match normalized data")
        elif not args.allow_unverified_images:
            raise ValueError(
                "image-bearing data requires --image-audit; run 'invllava data audit-images' first"
            )
    write_normalized_jsonl(samples, output_path)
    manifest = build_prepared_manifest(
        data_id=args.data_id,
        source_revision=args.source_revision,
        source_path=args.annotation,
        normalized_path=output_path,
        samples=samples,
        audit=audit,
        image_integrity=image_integrity,
        id_normalization=id_normalization,
        source_filter=source_filter,
    )
    manifest.write(manifest_path)
    _json(
        {
            "audit": audit.to_dict(),
            "id_normalization": id_normalization,
            "manifest": str(manifest_path),
        }
    )


def command_data_audit_images(args: argparse.Namespace) -> None:
    from invllava.artifacts.hashing import sha256_file
    from invllava.data.audit import ImageInventoryRecord, ImageReferenceSet, audit_image_integrity
    from invllava.data.prepare import iter_llava_json
    from invllava.eval.datasets import load_examples

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    def report_progress(completed: int, total: int) -> None:
        print(
            json.dumps(
                {"event": "image-audit-progress", "completed": completed, "total": total},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    inventory_path = Path(args.inventory_output) if args.inventory_output else None
    if inventory_path is not None and inventory_path.resolve() == output.resolve():
        raise ValueError("image audit and inventory outputs must be distinct")
    if inventory_path is not None and inventory_path.exists():
        raise FileExistsError(inventory_path)
    temporary_inventory = None
    inventory_stream = None
    if inventory_path is not None:
        import tempfile

        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{inventory_path.name}.", dir=inventory_path.parent
        )
        temporary_inventory = Path(temporary_name)
        inventory_stream = os.fdopen(descriptor, "w", encoding="utf-8")

    def write_inventory(record: ImageInventoryRecord) -> None:
        if inventory_stream is None:
            return
        inventory_stream.write(
            json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        )

    if args.input_format == "llava":
        samples = iter_llava_json(
            args.annotation,
            args.image_root,
            default_source=args.default_source,
        )
    else:
        if not args.default_source:
            raise ValueError("--default-source is required for eval-jsonl image audits")
        samples = (
            ImageReferenceSet(example.images, args.default_source)
            for example in load_examples(args.annotation)
        )

    try:
        audit = audit_image_integrity(
            samples,
            sources=set(args.source) if args.source else None,
            image_root=args.image_root,
            workers=args.workers,
            maximum_failure_details=args.maximum_failure_details,
            progress_every=args.progress_every,
            progress=report_progress,
            inventory=write_inventory if inventory_stream is not None else None,
        )
        if inventory_stream is not None and temporary_inventory is not None:
            inventory_stream.flush()
            os.fsync(inventory_stream.fileno())
            inventory_stream.close()
            inventory_stream = None
            os.replace(temporary_inventory, inventory_path)
    except BaseException:
        if inventory_stream is not None:
            inventory_stream.close()
        if temporary_inventory is not None:
            temporary_inventory.unlink(missing_ok=True)
        raise
    payload = {
        "schema_version": 2 if inventory_path is not None else 1,
        "annotation": str(Path(args.annotation).resolve()),
        "annotation_sha256": sha256_file(args.annotation),
        "source_revision": args.source_revision,
        "image_root": str(Path(args.image_root).resolve()),
        "selected_sources": sorted(args.source) if args.source else None,
        "inventory": (
            {
                "path": str(inventory_path.resolve()),
                "sha256": sha256_file(inventory_path),
                "identity": "pixel_sha256",
            }
            if inventory_path is not None
            else None
        ),
        **audit.to_dict(),
    }
    atomic_write_json(output, payload)
    _json(payload)
    if audit.unique_images == 0:
        raise RuntimeError(f"image audit selected no images; inspect {output}")
    if not audit.passed:
        raise RuntimeError(f"image integrity audit failed; inspect {output}")


def command_data_compare_image_inventories(args: argparse.Namespace) -> None:
    from invllava.data.overlap import compare_exact_image_overlap

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = compare_exact_image_overlap(
        args.left,
        args.right,
        maximum_examples=args.maximum_examples,
    )
    payload = {"schema_version": 1, **report.to_dict()}
    atomic_write_json(output, payload)
    _json(payload)
    if args.require_disjoint and not report.disjoint:
        raise RuntimeError(f"image inventories overlap; inspect {output}")


def command_data_audit_annotation(args: argparse.Namespace) -> None:
    from invllava.artifacts.hashing import sha256_file
    from invllava.data.audit import audit_annotation
    from invllava.data.prepare import iter_llava_json

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    annotation_digest = sha256_file(args.annotation)
    if args.expected_sha256 and annotation_digest != args.expected_sha256:
        raise ValueError("annotation SHA-256 does not match --expected-sha256")
    audit = audit_annotation(
        iter_llava_json(
            args.annotation,
            args.image_root,
            default_source=args.default_source,
        )
    )
    payload = {
        "schema_version": 1,
        "annotation": str(Path(args.annotation).resolve()),
        "annotation_sha256": annotation_digest,
        "source_revision": args.source_revision,
        "logical_image_root": str(Path(args.image_root).resolve()),
        **audit.to_dict(),
    }
    atomic_write_json(output, payload)
    _json(payload)
    if audit.samples == 0 or (args.require_unique_ids and audit.duplicate_ids):
        raise RuntimeError(f"annotation audit failed; inspect {output}")
    if args.expected_samples is not None and audit.samples != args.expected_samples:
        raise RuntimeError(
            f"annotation has {audit.samples} samples, expected {args.expected_samples}; "
            f"inspect {output}"
        )


def command_data_fetch(args: argparse.Namespace) -> None:
    if not args.allow_download or os.environ.get("INVLLAVA_ALLOW_DOWNLOADS") != "1":
        raise PermissionError(
            "data fetch requires both --allow-download and INVLLAVA_ALLOW_DOWNLOADS=1"
        )
    from invllava.data.download import download_http, download_huggingface
    from invllava.data.registry import plan_sources

    if args.minimum_free_gib < 0:
        raise ValueError("--minimum-free-gib must be non-negative")
    spec = DataSpec.model_validate(
        yaml.safe_load(Path(args.data_config).read_text(encoding="utf-8"))
    )
    plans = {item.id: item for item in plan_sources(spec, args.destination)}
    selected = set(args.source_id or ())
    known = set(plans)
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown --source-id values: {sorted(unknown)}")
    results: list[dict[str, str]] = []
    for source in (spec.annotation, *spec.image_sources):
        if selected and source.id not in selected:
            continue
        plan = plans[source.id]
        if source.kind == "manual":
            results.append(
                {"id": source.id, "status": "manual-required", "location": source.location}
            )
            continue
        if source.kind == "generated":
            path = Path(source.location)
            if not path.exists():
                raise FileNotFoundError(path)
            results.append({"id": source.id, "status": "verified-fixture", "path": str(path)})
            continue
        if source.kind == "huggingface":
            path = download_huggingface(
                source,
                plan.destination,
                minimum_free_bytes_after=args.minimum_free_gib * 1024**3,
            )
        else:
            if not source.sha256:
                raise ValueError(f"HTTP source {source.id} requires sha256")
            path = download_http(
                source.location,
                plan.destination / Path(source.location).name,
                sha256=source.sha256,
                minimum_free_bytes_after=args.minimum_free_gib * 1024**3,
            )
        results.append({"id": source.id, "status": "downloaded", "path": str(path)})
    _json(results)


def command_data_acquire_candidate(args: argparse.Namespace) -> None:
    if not args.allow_download or os.environ.get("INVLLAVA_ALLOW_DOWNLOADS") != "1":
        raise PermissionError(
            "candidate acquisition requires both --allow-download and INVLLAVA_ALLOW_DOWNLOADS=1"
        )
    if args.minimum_free_gib < 0:
        raise ValueError("--minimum-free-gib must be non-negative")
    from invllava.data.download import acquire_http_candidate

    last_reported = 0

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_reported
        interval = 1024**3
        if completed == total or completed - last_reported >= interval:
            last_reported = completed
            print(
                json.dumps(
                    {
                        "event": "candidate-download-progress",
                        "completed_bytes": completed,
                        "total_bytes": total,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    artifact, record = acquire_http_candidate(
        args.url,
        args.destination,
        minimum_free_bytes_after=args.minimum_free_gib * 1024**3,
        allow_insecure_http=args.allow_insecure_http,
        progress=report_progress,
    )
    _json({"artifact": str(artifact), "candidate_record": str(record)})


def command_fetch_hub_snapshot(args: argparse.Namespace) -> None:
    if not args.allow_download or os.environ.get("INVLLAVA_ALLOW_DOWNLOADS") != "1":
        raise PermissionError(
            "Hub acquisition requires both --allow-download and INVLLAVA_ALLOW_DOWNLOADS=1"
        )
    from invllava.artifacts.hub_snapshot import acquire_hub_snapshot

    output = acquire_hub_snapshot(
        repo_id=args.repo_id,
        revision=args.revision,
        repo_type=args.repo_type,
        destination=args.destination,
        allow_patterns=tuple(args.allow_pattern or ()),
        minimum_free_bytes_after=args.minimum_free_gib * 1024**3,
    )
    _json(
        {
            "snapshot": str(output),
            "manifest": str(output / "snapshot-manifest.json"),
        }
    )


def command_data_extract(args: argparse.Namespace) -> None:
    from invllava.data.download import extract_verified_archive

    if args.minimum_free_gib < 0:
        raise ValueError("--minimum-free-gib must be non-negative")
    output = extract_verified_archive(
        args.archive,
        args.destination,
        sha256=args.sha256,
        minimum_free_bytes_after=args.minimum_free_gib * 1024**3,
    )
    _json({"archive": str(Path(args.archive).resolve()), "destination": str(output)})


def command_data_assemble_layout(args: argparse.Namespace) -> None:
    from invllava.artifacts.hashing import sha256_file
    from invllava.config.schema import DataLayoutSpec
    from invllava.data.layout import materialize_hardlink_layout

    layout_path = Path(args.layout)
    spec = DataLayoutSpec.model_validate(yaml.safe_load(layout_path.read_text(encoding="utf-8")))
    output = materialize_hardlink_layout(
        spec,
        component_root=args.component_root,
        destination=args.destination,
        manifest_path=args.manifest,
        layout_config_sha256=sha256_file(layout_path),
    )
    _json({"destination": str(output), "manifest": str(Path(args.manifest).resolve())})


def command_fetch_eval(args: argparse.Namespace) -> None:
    if not args.allow_download or os.environ.get("INVLLAVA_ALLOW_DOWNLOADS") != "1":
        raise PermissionError(
            "evaluation materialization requires both --allow-download and "
            "INVLLAVA_ALLOW_DOWNLOADS=1"
        )
    from invllava.artifacts.hashing import sha256_file

    spec = _load_benchmark(args.benchmark)
    if spec.id in {"mme-perception", "mme-cognition"}:
        from invllava.eval.mme_data import materialize_mme_benchmark

        materialize = materialize_mme_benchmark
    elif spec.id in {"mmbench-en", "mmbench-cn"}:
        from invllava.eval.mmbench_data import materialize_mmbench_benchmark

        materialize = materialize_mmbench_benchmark
    elif spec.id == "gqa":
        from invllava.eval.gqa_data import materialize_gqa_benchmark

        materialize = materialize_gqa_benchmark
    elif spec.id == "vizwiz":
        from invllava.eval.vizwiz_data import materialize_vizwiz_benchmark

        materialize = materialize_vizwiz_benchmark
    elif spec.id == "vqav2-val":
        from invllava.eval.vqav2_data import materialize_vqav2_benchmark

        materialize = materialize_vqav2_benchmark
    else:
        from invllava.eval.fetch import materialize_huggingface_benchmark

        materialize = materialize_huggingface_benchmark

    def report_progress(stage: str, completed: int, total: int) -> None:
        print(
            json.dumps(
                {
                    "event": "evaluation-materialization-progress",
                    "stage": stage,
                    "completed": completed,
                    "total": total,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    output = materialize(
        spec,
        args.destination,
        cache_dir=args.cache_dir,
        config_sha256=sha256_file(args.benchmark),
        progress=report_progress,
    )
    _json(
        {
            "destination": str(output),
            "examples": str(output / "examples.jsonl"),
            "manifest": str(output / "manifest.json"),
        }
    )


def command_sample_eval(args: argparse.Namespace) -> None:
    from invllava.eval.sampling import materialize_evaluation_subset

    output = materialize_evaluation_subset(
        args.examples,
        args.destination,
        maximum=args.maximum,
        seed=args.seed,
        stratify_metadata=args.stratify_metadata,
        preserve_groups=args.preserve_groups,
    )
    _json(
        {
            "destination": str(output),
            "examples": str(output / "examples.jsonl"),
            "manifest": str(output / "manifest.json"),
        }
    )


def _load_benchmark(path: str | Path) -> BenchmarkSpec:
    return BenchmarkSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def command_score(args: argparse.Namespace) -> None:
    from invllava.artifacts.hashing import sha256_file
    from invllava.eval.datasets import load_examples
    from invllava.eval.protocols import build_protocol
    from invllava.eval.records import PredictionStore
    from invllava.eval.validation import (
        validate_coverage,
        validate_frozen_prompts,
        validate_no_reference_leakage,
    )

    if args.output and Path(args.output).exists():
        raise FileExistsError(f"score artifact already exists: {args.output}")
    spec = _load_benchmark(args.benchmark)
    if spec.verification_status != "golden_verified" and not args.allow_unverified:
        raise ValueError(
            f"protocol {spec.id} is {spec.verification_status}; "
            "pass --allow-unverified only for development"
        )
    if getattr(args, "judge_dir", None):
        if spec.id != "mmvet" or spec.scorer != "official_mmvet_hosted_gpt41":
            raise ValueError("--judge-dir currently accepts the official hosted MM-Vet protocol")
        if not args.submission:
            raise ValueError("importing judge grades requires the exact --submission JSON")
        from invllava.eval.protocols.mmvet import HostedMMVetProtocol

        protocol = HostedMMVetProtocol(spec.protocol_revision, args.judge_dir, args.submission)
    else:
        if getattr(args, "submission", None):
            raise ValueError("--submission requires --judge-dir")
        protocol = build_protocol(spec)
    protocol_id = args.protocol_id or benchmark_protocol_id(spec)
    checkpoint_id = args.checkpoint_id
    if checkpoint_id is None:
        with Path(args.predictions).open(encoding="utf-8") as stream:
            first = next((json.loads(line) for line in stream if line.strip()), None)
        if first is None:
            raise ValueError("cannot infer checkpoint identity from an empty prediction file")
        checkpoint_id = str(first["checkpoint_id"])
    records = list(
        PredictionStore(args.predictions, protocol_id=protocol_id, checkpoint_id=checkpoint_id)
    )
    predictions = {record.sample_id: record.prediction for record in records}
    examples = load_examples(args.examples)
    validate_coverage(records, {example.id for example in examples})
    validate_frozen_prompts(records, examples)
    validate_no_reference_leakage(records)
    score = protocol.score(predictions, examples)
    result = {
        "benchmark": spec.id,
        "protocol_id": protocol_id,
        "scorer_id": protocol.id,
        "checkpoint_id": checkpoint_id,
        "protocol_config_sha256": sha256_file(args.benchmark),
        "predictions_sha256": sha256_file(args.predictions),
        "examples_sha256": sha256_file(args.examples),
        **score.__dict__,
    }
    if args.output:
        atomic_write_json(args.output, result)
    _json(result)


def command_predict(args: argparse.Namespace) -> None:
    invocation_started_at = datetime.now(timezone.utc).isoformat()
    invocation_started = time.perf_counter()
    downloads_authorized = _downloads_authorized(args)
    if args.allow_download and not downloads_authorized:
        raise PermissionError("--allow-download also requires INVLLAVA_ALLOW_DOWNLOADS=1")
    if not downloads_authorized:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from invllava.artifacts.hashing import optional_sha256_environment, sha256_file
    from invllava.artifacts.source import checkout_source_sha256

    source_digest = checkout_source_sha256(
        Path(__file__).resolve().parents[2],
        expected=optional_sha256_environment("INVLLAVA_EXECUTION_SOURCE_SHA256"),
    )
    if source_digest is not None:
        os.environ["INVLLAVA_EXECUTION_SOURCE_SHA256"] = source_digest

    import torch

    from invllava.artifacts.manifest import ArtifactRecord, RunManifest
    from invllava.eval.datasets import load_examples, validate_image_paths
    from invllava.eval.interop.hf_llava import HuggingFaceLLaVAGenerator
    from invllava.eval.records import PredictionStore
    from invllava.eval.runtime import run_batched_evaluation
    from invllava.eval.validation import (
        validate_coverage,
        validate_frozen_prompts,
        validate_no_reference_leakage,
    )

    spec = _load_benchmark(args.benchmark)
    if args.backend != "native" and args.runtime_ref is not None:
        raise ValueError("--runtime-ref is only valid with --backend native")
    if args.backend == "native" and args.attention_backend is not None:
        raise ValueError("native prediction takes its attention backend from the runtime config")
    if args.backend != "release" and args.generation_cache is not None:
        raise ValueError("--generation-cache is only valid with --backend release")
    if args.backend != "release" and args.lora_execution is not None:
        raise ValueError("--lora-execution is only valid with --backend release")
    if spec.verification_status != "golden_verified" and not args.allow_unverified:
        raise ValueError(
            f"protocol {spec.id} is {spec.verification_status}; "
            "pass --allow-unverified only for development"
        )
    if spec.generation.num_beams != 1:
        raise ValueError("the clean runtime currently implements num_beams=1 only")
    examples = load_examples(args.examples)
    if args.batch_size <= 0:
        raise ValueError("prediction batch size must be positive")
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation examples contain duplicate IDs")
    validate_image_paths(examples)
    protocol_id = benchmark_protocol_id(spec)
    generation = spec.generation.model_dump(mode="json")
    generation["runtime_batch_size"] = args.batch_size
    source_configs = [str(Path(args.benchmark).resolve())]
    dataset_revisions = {spec.annotations.id: spec.annotations.revision}
    if spec.images is not None:
        dataset_revisions[spec.images.id] = spec.images.revision
    dataset_revisions.update({source.id: source.revision for source in spec.protocol_sources})

    if args.backend == "native":
        if not args.experiment or not args.checkpoint:
            raise ValueError("native prediction requires --experiment and --checkpoint")
        from invllava.runtime.native import load_native_inference_runtime

        runtime = load_native_inference_runtime(
            experiment=args.experiment,
            checkpoint=args.checkpoint,
            config_root=args.config_root,
            runtime_ref=args.runtime_ref,
            device=args.device,
            max_new_tokens=spec.generation.max_new_tokens,
            temperature=spec.generation.temperature,
            top_p=spec.generation.top_p,
            local_files_only=not downloads_authorized,
        )
        resolved = runtime.resolved
        generator = runtime.generator
        kernel_report = runtime.kernel_report
        checkpoint_digest = runtime.checkpoint_sha256
        checkpoint_id = runtime.checkpoint_id
        experiment_id = resolved.id
        scientific_id = content_id(scientific_payload(resolved), prefix="sci")
        method_revision = resolved.method_revision
        source_configs.extend(str(path) for path in resolved.source_files)
        checkpoint_inputs = {checkpoint_id: checkpoint_digest}
        execution_dtype = resolved.model.torch_dtype
        execution_attention = resolved.runtime.attention_backend
        execution_generation_cache = "kv"
        execution_numerical_policy = runtime.numerical_policy
        generation.update(
            {
                "runtime_dtype": execution_dtype,
                "runtime_attention_backend": execution_attention,
                "runtime_generation_cache": execution_generation_cache,
                "runtime_numerical_policy": execution_numerical_policy,
            }
        )
        store = PredictionStore(
            args.output,
            protocol_id=protocol_id,
            checkpoint_id=checkpoint_id,
            experiment_id=experiment_id,
        )
        written = run_batched_evaluation(
            examples,
            generate=generator.generate_many,
            batch_size=args.batch_size,
            store=store,
            experiment_id=experiment_id,
            checkpoint_id=checkpoint_id,
            protocol_id=protocol_id,
            generation=generation,
        )
    elif args.backend == "release":
        if not args.model:
            raise ValueError("release prediction requires --model")
        local_release = Path(args.model).is_dir()
        if not local_release and (not args.revision or args.revision in {"main", "pending-freeze"}):
            raise ValueError("remote release prediction requires an immutable --revision")
        from invllava.release import load_pretrained

        release = load_pretrained(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=args.dtype,
            attention_backend=args.attention_backend or "sdpa",
            generation_cache=args.generation_cache or "kv",
            lora_execution=args.lora_execution or "unmerged",
            max_new_tokens=spec.generation.max_new_tokens,
            temperature=spec.generation.temperature,
            top_p=spec.generation.top_p,
            local_files_only=not downloads_authorized,
        )
        generator = release.generator
        checkpoint_digest = release.checkpoint_sha256
        checkpoint_id = release.checkpoint_id
        experiment_id = str(release.metadata.get("experiment_id", "inverse-llava-champion"))
        execution_attention = args.attention_backend or "sdpa"
        execution_generation_cache = args.generation_cache or "kv"
        execution_lora = release.lora_execution
        scientific_id = content_id(
            {
                "model": release.model_spec.model_dump(mode="json"),
                "attention_backend": execution_attention,
                "generation_cache": execution_generation_cache,
                "lora_execution": execution_lora,
            },
            prefix="sci",
        )
        method_revision = "invllava-hub-release-v1"
        source_configs.append(str(release.release_root / "inverse_llava_config.json"))
        checkpoint_inputs = {checkpoint_id: checkpoint_digest}
        execution_dtype = release.model_spec.torch_dtype
        execution_numerical_policy = release.numerical_policy
        generation.update(
            {
                "runtime_dtype": execution_dtype,
                "runtime_attention_backend": execution_attention,
                "runtime_generation_cache": execution_generation_cache,
                "runtime_lora_execution": execution_lora,
                "runtime_numerical_policy": execution_numerical_policy,
            }
        )
        store = PredictionStore(
            args.output,
            protocol_id=protocol_id,
            checkpoint_id=checkpoint_id,
            experiment_id=experiment_id,
        )
        written = run_batched_evaluation(
            examples,
            generate=generator.generate_many,
            batch_size=args.batch_size,
            store=store,
            experiment_id=experiment_id,
            checkpoint_id=checkpoint_id,
            protocol_id=protocol_id,
            generation=generation,
        )
    else:
        if not args.model or not args.revision:
            raise ValueError("Hugging Face reference prediction requires --model and --revision")
        if args.revision in {"main", "pending-freeze"}:
            raise ValueError("reference model revision must be immutable")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
        from invllava.runtime.numerics import configure_torch_numerics_policy

        execution_numerical_policy = configure_torch_numerics_policy(
            mixed_precision="no" if args.dtype == "float32" else args.dtype,
            allow_tf32=True,
            deterministic_algorithms=False,
            cudnn_benchmark=False,
        )
        if args.backend == "hf-multimodal":
            from invllava.eval.interop.hf_multimodal import HuggingFaceMultimodalGenerator

            if spec.generation.temperature != 0:
                raise ValueError("native multimodal reference requires greedy benchmark decoding")
            generator = HuggingFaceMultimodalGenerator(
                args.model,
                revision=args.revision,
                dtype=dtype,
                device=args.device,
                max_new_tokens=spec.generation.max_new_tokens,
                attention_backend=args.attention_backend or "sdpa",
                local_files_only=not downloads_authorized,
            )
            reference_policy = generator.provenance
        else:
            generator = HuggingFaceLLaVAGenerator(
                args.model,
                revision=args.revision,
                dtype=dtype,
                device=args.device,
                max_new_tokens=spec.generation.max_new_tokens,
                attention_backend=args.attention_backend,
                image_aspect_ratio=args.hf_image_aspect_ratio,
                local_files_only=not downloads_authorized,
            )
            reference_policy = {"image_aspect_ratio": args.hf_image_aspect_ratio}
        reference_attention = args.attention_backend or (
            "sdpa" if args.backend == "hf-multimodal" else "transformers-default"
        )
        identity = {
            "model": args.model,
            "revision": args.revision,
            "dtype": args.dtype,
            "attention_backend": reference_attention,
            **reference_policy,
        }
        checkpoint_id = content_id(identity, prefix="ckpt")
        experiment_id = f"{args.backend}-protocol-reference"
        scientific_id = content_id(identity, prefix="reference")
        method_revision = (
            "huggingface-native-multimodal-reference-v1"
            if args.backend == "hf-multimodal"
            else "huggingface-llava-reference-runtime"
        )
        checkpoint_inputs = {checkpoint_id: f"{args.model}@{args.revision}"}
        execution_dtype = args.dtype
        execution_attention = reference_attention
        execution_generation_cache = "transformers-default"
        generation.update(
            {
                "runtime_dtype": execution_dtype,
                "runtime_attention_backend": execution_attention,
                "runtime_generation_cache": execution_generation_cache,
                "runtime_image_aspect_ratio": (
                    args.hf_image_aspect_ratio if args.backend == "hf-llava" else "native"
                ),
                "runtime_numerical_policy": execution_numerical_policy,
                **{
                    f"runtime_{key}": value
                    for key, value in reference_policy.items()
                    if key != "image_aspect_ratio"
                },
            }
        )
        store = PredictionStore(
            args.output,
            protocol_id=protocol_id,
            checkpoint_id=checkpoint_id,
            experiment_id=experiment_id,
        )
        written = run_batched_evaluation(
            examples,
            generate=generator.generate_many,
            batch_size=args.batch_size,
            store=store,
            experiment_id=experiment_id,
            checkpoint_id=checkpoint_id,
            protocol_id=protocol_id,
            generation=generation,
        )

    records = list(store)
    invocation_elapsed_seconds = time.perf_counter() - invocation_started
    validate_coverage(records, set(ids))
    validate_no_reference_leakage(records)
    validate_frozen_prompts(records, examples)
    manifest_path = (
        Path(args.manifest) if args.manifest else Path(args.output).with_suffix(".manifest.json")
    )
    if manifest_path.exists():
        if written:
            raise FileExistsError("evaluation manifest predates newly appended predictions")
    else:
        manifest = RunManifest.create(
            run_id=f"eval-{experiment_id}-{protocol_id}-{checkpoint_id}",
            experiment_id=experiment_id,
            scientific_id=scientific_id,
            method_revision=method_revision,
            repository_root=Path.cwd(),
            source_configs=source_configs,
            dataset_revisions=dataset_revisions,
            checkpoint_inputs=checkpoint_inputs,
            protocol_ids=[protocol_id],
        )
        prediction_path = Path(args.output)
        manifest.add_artifact(
            ArtifactRecord(
                role="immutable_predictions",
                relative_path=prediction_path.name,
                sha256=sha256_file(prediction_path),
                size_bytes=prediction_path.stat().st_size,
            )
        )
        manifest.notes.append(f"examples_sha256={sha256_file(args.examples)}")
        manifest.notes.append(f"benchmark_verification={spec.verification_status}")
        manifest.notes.append(f"execution_dtype={execution_dtype}")
        manifest.notes.append(f"execution_attention={execution_attention}")
        manifest.notes.append(f"execution_generation_cache={execution_generation_cache}")
        manifest.notes.append(
            f"numerical_policy={json.dumps(execution_numerical_policy, sort_keys=True)}"
        )
        manifest.notes.append(f"prediction_invocation_started_at={invocation_started_at}")
        manifest.notes.append(
            f"prediction_invocation_elapsed_seconds={invocation_elapsed_seconds:.6f}"
        )
        manifest.notes.append(f"prediction_examples_written={written}")
        manifest.notes.append(f"prediction_examples_total={len(records)}")
        manifest.notes.append(f"prediction_batch_size={args.batch_size}")
        if args.backend == "release":
            manifest.notes.append(f"execution_lora={execution_lora}")
        if args.backend == "native":
            manifest.notes.append(
                f"kernel_optimization={json.dumps(kernel_report.to_dict(), sort_keys=True)}"
            )
        manifest.write(manifest_path)
    _json(
        {
            "written": written,
            "total": len(records),
            "protocol_id": protocol_id,
            "checkpoint_id": checkpoint_id,
            "elapsed_seconds": invocation_elapsed_seconds,
            "predictions": args.output,
            "manifest": str(manifest_path),
        }
    )


def command_package_submission(args: argparse.Namespace) -> None:
    from invllava.artifacts.hashing import sha256_file
    from invllava.artifacts.manifest import RunManifest
    from invllava.eval.datasets import load_examples
    from invllava.eval.records import PredictionStore
    from invllava.eval.submission import (
        build_mmvet_submission,
        build_vqav2_submission,
        submission_checkpoint_digest,
        write_submission_manifest,
    )
    from invllava.eval.validation import validate_frozen_prompts

    spec = _load_benchmark(args.benchmark)
    if not (
        (spec.id == "vqav2-testdev" and spec.external_only)
        or (spec.id == "mmvet" and spec.task_type == "judge")
    ):
        raise ValueError("external packaging supports VQAv2 test-dev and MM-Vet v1")
    if spec.verification_status != "golden_verified" and not args.allow_unverified:
        raise ValueError("adapter must be golden_verified before final packaging")
    protocol_id = benchmark_protocol_id(spec)
    evaluation_manifest_path = (
        Path(args.evaluation_manifest)
        if args.evaluation_manifest
        else Path(args.predictions).with_suffix(".manifest.json")
    )
    evaluation_manifest = RunManifest.read(evaluation_manifest_path)
    if protocol_id not in evaluation_manifest.protocol_ids:
        raise ValueError("evaluation manifest does not contain the requested protocol identity")
    clean_commit = (
        evaluation_manifest.code_commit if evaluation_manifest.code_dirty is False else None
    )
    if clean_commit is None and evaluation_manifest.execution_source_sha256 is None:
        raise ValueError(
            "external packaging requires a clean Git commit or execution-source SHA-256"
        )
    if len(evaluation_manifest.checkpoint_inputs) != 1:
        raise ValueError("external packaging requires one unambiguous input checkpoint")
    checkpoint_id, checkpoint_input = next(iter(evaluation_manifest.checkpoint_inputs.items()))
    checkpoint_digest = submission_checkpoint_digest(checkpoint_input, args.checkpoint_snapshot)
    records = list(
        PredictionStore(
            args.predictions,
            protocol_id=protocol_id,
            checkpoint_id=checkpoint_id,
            experiment_id=evaluation_manifest.experiment_id,
        )
    )
    examples = load_examples(args.examples)
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("submission examples contain duplicate IDs")
    if spec.id == "vqav2-testdev" and any(example.references for example in examples):
        raise ValueError("VQAv2 test-dev examples must not contain hidden references")
    validate_frozen_prompts(records, examples)
    example_digest = sha256_file(args.examples)
    if f"examples_sha256={example_digest}" not in evaluation_manifest.notes:
        raise ValueError("submission examples differ from the evaluated input bytes")
    prediction_digest = sha256_file(args.predictions)
    bound_predictions = [
        artifact
        for artifact in evaluation_manifest.artifacts
        if artifact.role == "immutable_predictions"
    ]
    if len(bound_predictions) != 1 or bound_predictions[0].sha256 != prediction_digest:
        raise ValueError("submission predictions differ from the evaluation manifest")
    destination = Path(args.output)
    submission_manifest = Path(args.output_manifest)
    if destination.exists() or submission_manifest.exists():
        raise FileExistsError("submission package destinations must be new")
    envelope_metadata = {}
    if spec.id == "mmvet":
        if getattr(args, "full_test_questions", None):
            raise ValueError("--full-test-questions is specific to VQAv2 test-dev")
        build_mmvet_submission(records, {example.id for example in examples}, destination)
    else:
        if not getattr(args, "full_test_questions", None):
            raise ValueError(
                "VQAv2 test-dev upload requires --full-test-questions for its envelope"
            )
        source = next(
            (row for row in spec.protocol_sources if row.id == "vqav2-full-test-envelope"), None
        )
        if (
            source is None
            or not source.sha256
            or sha256_file(args.full_test_questions) != source.sha256
        ):
            raise ValueError("full-test question bytes differ from the pinned upload envelope")
        full_test = json.loads(Path(args.full_test_questions).read_text())["questions"]
        if len(full_test) != 447793 or len(examples) != 107394:
            raise ValueError("VQAv2 test-dev input or full-test envelope coverage is incomplete")
        full_ids = [row["question_id"] for row in full_test]
        build_vqav2_submission(
            records,
            {example.id for example in examples},
            destination,
            full_test_question_ids=full_ids,
        )
        envelope_metadata = {
            "full_test_questions_sha256": source.sha256,
            "evaluated_questions": len(examples),
            "upload_questions": len(full_ids),
            "outside_testdev_empty_answers": len(full_ids) - len(examples),
            "answer_processing": "LLaVA M4C EvalAIAnswerProcessor",
            "submission_scope": "test-dev only; outside-subset empty answers are upload padding",
        }
    package_metadata = {
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_id": checkpoint_id,
        "evaluation_checkpoint_input": checkpoint_input,
        "checkpoint_digest_kind": (
            "verified_hub_inventory" if args.checkpoint_snapshot else "model_delta"
        ),
        "protocol_id": protocol_id,
        "dataset_split": spec.split,
        "code_commit": clean_commit,
        "execution_source_sha256": evaluation_manifest.execution_source_sha256,
        "predictions_sha256": prediction_digest,
        "examples_sha256": example_digest,
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "submission_sha256": sha256_file(destination),
        **envelope_metadata,
    }
    write_submission_manifest(submission_manifest, package_metadata)
    _json({"submission": str(destination), "manifest": str(submission_manifest)})


def command_prepare_eval(args: argparse.Namespace) -> None:
    from invllava.artifacts.hashing import sha256_file
    from invllava.data.audit import ImageReferenceSet, audit_image_integrity
    from invllava.eval.datasets import (
        prepare_mmvet,
        prepare_vqav2,
        write_examples,
    )

    spec = _load_benchmark(args.benchmark)
    if spec.annotations.kind == "huggingface" or spec.id in {
        "textvqa",
        "mme-perception",
        "mme-cognition",
        "mmbench-en",
        "mmbench-cn",
        "gqa",
        "vqav2-val",
        "vizwiz",
    }:
        raise ValueError(f"{spec.id} must be materialized with fetch-eval")
    if not args.image_root:
        raise ValueError(f"--image-root is required for {spec.id}")
    output = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")
    image_audit_path = (
        Path(args.image_audit) if args.image_audit else output.with_suffix(".image-integrity.json")
    )
    destinations = [output, manifest_path]
    if not args.allow_unverified_images:
        destinations.append(image_audit_path)
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("evaluation examples, audit, and manifest paths must be distinct")
    if any(path.exists() for path in destinations):
        raise FileExistsError("evaluation examples, audit, and manifest destinations must be new")

    if spec.annotations.kind == "manual" and spec.annotations.sha256:
        if sha256_file(args.annotations) != spec.annotations.sha256:
            raise ValueError("manual annotations differ from the pinned benchmark source")
    if spec.id == "mmvet":
        examples = prepare_mmvet(args.annotations, args.image_root, spec=spec)
    elif spec.id == "vqav2-testdev":
        coco_split = args.coco_split or "test2015"
        examples = prepare_vqav2(
            args.annotations,
            args.image_root,
            coco_split=coco_split,
            spec=spec,
        )
    else:
        raise ValueError(f"no manual-file converter is registered for {spec.id}")
    image_integrity = None
    if not args.allow_unverified_images:

        def report_progress(completed: int, total: int) -> None:
            print(
                json.dumps(
                    {
                        "event": "evaluation-image-audit-progress",
                        "completed": completed,
                        "total": total,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

        audit_root = Path(args.image_root)
        audit = audit_image_integrity(
            (ImageReferenceSet(example.images, spec.id) for example in examples),
            image_root=audit_root,
            workers=args.image_audit_workers,
            maximum_failure_details=100,
            progress_every=args.image_audit_progress_every,
            progress=report_progress,
        )
        image_audit_payload = {
            "schema_version": 1,
            "benchmark_id": spec.id,
            "protocol_revision": spec.protocol_revision,
            "image_root": str(audit_root.resolve()),
            **audit.to_dict(),
        }
        atomic_write_json(image_audit_path, image_audit_payload)
        if audit.unique_images == 0 or not audit.passed:
            raise RuntimeError(f"evaluation image integrity failed; inspect {image_audit_path}")
        image_integrity = {
            **image_audit_payload,
            "report_sha256": sha256_file(image_audit_path),
        }
    write_examples(examples, output)
    source_files = (
        {"annotations": sha256_file(args.annotations)} if Path(args.annotations).is_file() else {}
    )
    atomic_write_json(
        manifest_path,
        {
            "format": "invllava-eval-dataset-v1",
            "benchmark_id": spec.id,
            "protocol_revision": spec.protocol_revision,
            "protocol_config_sha256": sha256_file(args.benchmark),
            "conversation_template": spec.conversation_template,
            "split": spec.split,
            "sample_count": len(examples),
            "examples_sha256": sha256_file(output),
            "source_files": source_files,
            "declared_annotation_revision": spec.annotations.revision,
            "declared_annotation_sha256": spec.annotations.sha256,
            "declared_image_revision": spec.images.revision if spec.images else None,
            "declared_image_sha256": spec.images.sha256 if spec.images else None,
            "image_integrity": image_integrity,
        },
    )
    _json({"examples": len(examples), "output": str(output), "manifest": str(manifest_path)})


def command_complexity(args: argparse.Namespace) -> None:
    from invllava.analysis.complexity import fusion_complexity, llava_projector_complexity

    fusion = fusion_complexity(
        hidden_size=args.hidden_size,
        visual_size=args.visual_size,
        visual_input_size=args.visual_input_size,
        sequence_length=args.sequence_length,
        target_output_sizes=(args.hidden_size, args.hidden_size, args.hidden_size),
        fusion_layers=args.fusion_layers,
        operator=args.operator,
    )
    _json(
        {
            "inverse_fusion": fusion.to_dict(),
            "llava_mlp2x_gelu_projector": llava_projector_complexity(
                args.hidden_size,
                args.visual_size,
                args.patches,
                kind="mlp2x_gelu",
            ),
            "labels": {"parameters": "calculated", "macs": "calculated"},
        }
    )


def command_language_eval(args: argparse.Namespace) -> None:
    downloads_authorized = _downloads_authorized(args)
    if args.allow_download and not downloads_authorized:
        raise PermissionError("--allow-download also requires INVLLAVA_ALLOW_DOWNLOADS=1")
    if not downloads_authorized:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.model_kind == "native" and (not args.experiment or not args.checkpoint):
        raise ValueError("native evaluation requires --experiment and --checkpoint")
    if args.model_kind == "release" and not args.model:
        raise ValueError("release evaluation requires --model")
    if args.model_kind == "native":
        resolved = ConfigRepository(args.config_root).resolve(
            args.experiment, runtime_ref=args.runtime_ref
        )
        configure_runtime_cache(resolved.runtime.cache_root)
    from invllava.artifacts.hashing import optional_sha256_environment, sha256_file
    from invllava.eval.interop.lm_eval import (
        NativeTextLM,
        ReleaseTextLM,
        build_reference_text_lm,
        run_language_suite,
    )
    from invllava.eval.language_suite import load_language_suite
    from invllava.train.checkpoint import MODEL_DELTA_FILENAME

    suite = load_language_suite(args.suite)
    if args.max_length is not None and args.max_length != suite.context_length:
        raise ValueError("--max-length must match the frozen suite context_length")
    if args.model_kind == "native":
        trainable = Path(args.checkpoint) / MODEL_DELTA_FILENAME
        model = NativeTextLM(
            experiment=args.experiment,
            checkpoint=args.checkpoint,
            config_root=args.config_root,
            runtime_ref=args.runtime_ref,
            device=args.device,
            batch_size=args.batch_size,
            max_length=suite.context_length,
            add_bos_token=suite.add_bos_token,
            dtype=args.dtype,
            local_files_only=not downloads_authorized,
        )
        metadata = {
            "kind": args.model_kind,
            "experiment": args.experiment,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(trainable),
            "kernel_optimization": model.kernel_optimization,
        }
    elif args.model_kind == "release":
        model = ReleaseTextLM(
            model=args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            max_length=suite.context_length,
            add_bos_token=suite.add_bos_token,
            local_files_only=not downloads_authorized,
        )
        metadata = {
            "kind": args.model_kind,
            "checkpoint": str(model.release_root),
            "checkpoint_sha256": model.checkpoint_sha256,
            "release_revision": model.release_revision,
        }
    else:
        if not args.model or not args.revision:
            raise ValueError("reference evaluation requires --model and --revision")
        model = build_reference_text_lm(
            kind=args.model_kind,
            checkpoint=args.model,
            revision=args.revision,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            max_length=suite.context_length,
            add_bos_token=suite.add_bos_token,
            local_files_only=not downloads_authorized,
        )
        metadata = {
            "kind": args.model_kind,
            "checkpoint": args.model,
            "revision": args.revision,
        }
    metadata["invllava_version"] = __version__
    metadata["text_input_policy"] = {
        "context_length": suite.context_length,
        "add_bos_token": suite.add_bos_token,
        "dtype": args.dtype,
        "adapter_execution": "unmerged" if args.model_kind in {"native", "release"} else None,
    }
    for field, variable in (
        ("execution_source_sha256", "INVLLAVA_EXECUTION_SOURCE_SHA256"),
        ("language_environment_sha256", "INVLLAVA_LANGUAGE_ENV_SHA256"),
    ):
        value = optional_sha256_environment(variable)
        if value:
            metadata[field] = value
    result = run_language_suite(
        model,
        suite,
        args.output,
        metadata=metadata,
        limit=args.limit,
        task_ids=set(args.task) if args.task else None,
    )
    _json(result)


def command_cleanup(args: argparse.Namespace) -> None:
    from invllava.artifacts.cleanup import cleanup_plan, execute_cleanup

    plan = cleanup_plan(args.manifest, args.root)
    _json(
        {
            "targets": [str(item.path) for item in plan],
            "bytes": sum(item.size_bytes for item in plan),
        }
    )
    if args.confirm:
        removed = execute_cleanup(plan, confirmed=True)
        print(f"removed {removed} bytes", file=sys.stderr)


def command_verify_run(args: argparse.Namespace) -> None:
    from invllava.artifacts.run_integrity import verify_training_run

    report = verify_training_run(args.run_dir, require_complete=not args.allow_incomplete)
    payload = report.to_dict()
    if args.output:
        atomic_write_json(args.output, payload)
    _json(payload)


def command_convert_projector(args: argparse.Namespace) -> None:
    from invllava.model.projector_checkpoint import convert_projector_checkpoint

    output = convert_projector_checkpoint(args.source, args.output)
    _json({"projector_checkpoint": str(output)})


def command_convert_official_llava_lora(args: argparse.Namespace) -> None:
    from invllava.model.reference_checkpoint import convert_official_llava_lora

    output = convert_official_llava_lora(
        adapter_model=args.adapter_model,
        adapter_config=args.adapter_config,
        non_lora_trainables=args.non_lora_trainables,
        destination=args.output,
    )
    _json({"reference_checkpoint": str(output)})


def command_convert_champion_checkpoint(args: argparse.Namespace) -> None:
    if not args.allow_download or os.environ.get("INVLLAVA_ALLOW_DOWNLOADS") != "1":
        raise PermissionError(
            "champion conversion verifies the pinned base model; pass --allow-download "
            "and set INVLLAVA_ALLOW_DOWNLOADS=1"
        )
    from invllava.model.champion_checkpoint import convert_champion_checkpoint

    output = convert_champion_checkpoint(
        source=args.source,
        model_config=args.model_config,
        destination=args.output,
        model_card=args.model_card,
    )
    _json(
        {
            "release_bundle": str(output),
            "metadata": str(output / "metadata.json"),
            "checksums": str(output / "checksums.sha256"),
        }
    )


def command_seal_training_checkpoint(args: argparse.Namespace) -> None:
    from invllava.release.bundle import export_training_checkpoint, seal_training_checkpoint

    if args.run_dir:
        output = export_training_checkpoint(args.run_dir, args.checkpoint, destination=args.output)
    else:
        output = seal_training_checkpoint(
            args.checkpoint, parent_release=args.parent_release, destination=args.output
        )
    _json(
        {
            "release_bundle": str(output),
            "metadata": str(output / "metadata.json"),
            "checksums": str(output / "checksums.sha256"),
        }
    )


def command_audit_llava_fft_conversion(args: argparse.Namespace) -> None:
    from invllava.model.reference_audit import audit_llava_fft_conversion

    _json(
        audit_llava_fft_conversion(
            args.official,
            args.converted,
            output=args.output,
        )
    )


def command_checkpoint_inventory(args: argparse.Namespace) -> None:
    from invllava.artifacts.hashing import optional_sha256_environment
    from invllava.model.reference_audit import safetensors_index_inventory

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "execution_source_sha256": optional_sha256_environment("INVLLAVA_EXECUTION_SOURCE_SHA256"),
        **safetensors_index_inventory(args.checkpoint),
    }
    atomic_write_json(output, payload)
    _json(payload)


def command_capture_representations(args: argparse.Namespace) -> None:
    downloads_authorized = _downloads_authorized(args)
    model_is_local = bool(args.model) and Path(args.model).exists()
    if args.backend in {"release", "hf-llava", "hf-causal"} and not (
        model_is_local or downloads_authorized
    ):
        raise PermissionError(
            "remote representation capture requires --allow-download and INVLLAVA_ALLOW_DOWNLOADS=1"
        )
    import torch

    from invllava.analysis.runtime import (
        capture_huggingface_representations,
        capture_representations,
        write_representation_artifact,
    )
    from invllava.artifacts.hashing import sha256_file
    from invllava.eval.datasets import load_examples, validate_image_paths
    from invllava.runtime.native import load_native_inference_runtime

    examples = load_examples(args.examples)
    if args.maximum is not None:
        examples = examples[: args.maximum]
    ids = [example.id for example in examples]
    if not examples or len(ids) != len(set(ids)):
        raise ValueError("representation examples must be non-empty with unique IDs")
    validate_image_paths(examples)
    if args.backend == "native":
        if not args.experiment or not args.checkpoint:
            raise ValueError("native representation capture requires experiment and --checkpoint")
        runtime = load_native_inference_runtime(
            experiment=args.experiment,
            checkpoint=args.checkpoint,
            config_root=args.config_root,
            runtime_ref=args.runtime_ref,
            device=args.device,
            max_new_tokens=1,
            require_portable=True,
            local_files_only=not downloads_authorized,
        )
        resolved = runtime.resolved
        sample_ids, arrays = capture_representations(
            runtime.generator,
            examples,
            batch_size=args.batch_size,
        )
        checkpoint_digest = runtime.checkpoint_sha256
        metadata = {
            "backend": "native",
            "experiment_id": resolved.id,
            "scientific_id": content_id(scientific_payload(resolved), prefix="sci"),
            "checkpoint_id": f"ckpt-{checkpoint_digest[:16]}",
            "checkpoint_sha256": checkpoint_digest,
            "examples_sha256": sha256_file(args.examples),
            "source_configs": [str(path) for path in resolved.source_files],
            "device": str(args.device),
            "torch_version": torch.__version__,
            "batch_size": args.batch_size,
            "input_condition": "image-conditioned benchmark prompt",
        }
        pooling = (
            "hidden.<layer>: mean over valid expanded positions; "
            "hidden.last.<layer>: final valid prompt position; "
            "vision and fusion arrays: documented sample means"
        )
    elif args.backend == "release":
        if args.experiment or args.checkpoint or not args.model:
            raise ValueError("release representation capture requires --model only")
        from invllava.release import load_pretrained

        release = load_pretrained(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            device=args.device,
            dtype=args.dtype,
            attention_backend=args.attention_backend,
            max_new_tokens=1,
            local_files_only=not downloads_authorized,
        )
        sample_ids, arrays = capture_representations(
            release.generator,
            examples,
            batch_size=args.batch_size,
        )
        metadata = {
            "backend": "release",
            "experiment_id": release.metadata.get("experiment_id", "inverse-llava"),
            "checkpoint_id": release.checkpoint_id,
            "checkpoint_sha256": release.checkpoint_sha256,
            "release_revision": release.release_revision,
            "examples_sha256": sha256_file(args.examples),
            "source_configs": [str(release.release_root / "inverse_llava_config.json")],
            "device": str(args.device),
            "torch_version": torch.__version__,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "attention_backend": args.attention_backend,
            "lora_execution": release.lora_execution,
            "input_condition": "image-conditioned benchmark prompt",
        }
        pooling = (
            "hidden.<layer>: mean over valid expanded positions; "
            "hidden.last.<layer>: final valid prompt position; "
            "vision and fusion arrays: documented sample means"
        )
    else:
        if args.experiment or args.checkpoint:
            raise ValueError("Hugging Face representation capture uses --model and --revision only")
        if not args.model or not args.revision:
            raise ValueError("Hugging Face representation capture requires --model and --revision")
        sample_ids, arrays = capture_huggingface_representations(
            kind=args.backend,
            checkpoint=args.model,
            revision=args.revision,
            examples=examples,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            attention_backend=args.attention_backend,
            image_aspect_ratio=args.hf_image_aspect_ratio,
            local_files_only=not downloads_authorized,
        )
        identity = {
            "backend": args.backend,
            "model": args.model,
            "revision": args.revision,
            "dtype": args.dtype,
            "attention_backend": args.attention_backend,
            "image_aspect_ratio": (
                args.hf_image_aspect_ratio if args.backend == "hf-llava" else None
            ),
        }
        metadata = {
            **identity,
            "checkpoint_id": content_id(identity, prefix="ckpt"),
            "examples_sha256": sha256_file(args.examples),
            "device": str(args.device),
            "torch_version": torch.__version__,
            "batch_size": args.batch_size,
            "input_condition": (
                "image-conditioned benchmark prompt"
                if args.backend == "hf-llava"
                else "same benchmark prompt with image placeholders removed"
            ),
        }
        pooling = (
            "hidden.last.<layer>: final valid prompt position; "
            "vision arrays: mean over the projector patch sequence"
        )
    if tuple(ids) != sample_ids:
        raise RuntimeError("representation capture changed the frozen sample order")
    metadata_path = args.metadata or str(Path(args.output).with_suffix(".json"))
    write_representation_artifact(
        args.output,
        metadata_path,
        sample_ids=sample_ids,
        arrays=arrays,
        metadata=metadata,
        pooling=pooling,
    )
    _json({"output": args.output, "metadata": metadata_path, "arrays": sorted(arrays)})


def command_compare_representations(args: argparse.Namespace) -> None:
    from dataclasses import asdict

    import numpy as np

    from invllava.analysis.representations import (
        compare_representations,
        matched_permutation_test,
        matched_shuffled_margin,
        paired_numerical_similarity,
    )

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    with np.load(args.left) as left, np.load(args.right) as right:
        left_ids = left["sample_ids"].astype(str)
        right_ids = right["sample_ids"].astype(str)
        if not np.array_equal(left_ids, right_ids):
            raise ValueError("representation artifacts must have identical ordered sample IDs")
        if args.left_key not in left or args.right_key not in right:
            raise KeyError("requested representation key is absent")
        result = asdict(
            compare_representations(
                left[args.left_key],
                right[args.right_key],
                k=args.k,
            )
        )
        if args.matched_margin:
            result["matched_shuffled_margin"] = matched_shuffled_margin(
                left[args.left_key], right[args.right_key], seed=args.seed
            )
            result["matched_shuffled_seed"] = args.seed
        if args.matched_permutations:
            result["matched_permutation_test"] = matched_permutation_test(
                left[args.left_key],
                right[args.right_key],
                permutations=args.matched_permutations,
                confidence=args.confidence,
                seed=args.seed,
            )
        if args.paired_numerics:
            result["paired_numerical_similarity"] = paired_numerical_similarity(
                left[args.left_key], right[args.right_key]
            )
    payload = {
        "left": str(args.left),
        "left_key": args.left_key,
        "right": str(args.right),
        "right_key": args.right_key,
        "sample_count": len(left_ids),
        **result,
    }
    atomic_write_json(args.output, payload)
    _json(payload)


def command_plot_representation_pca(args: argparse.Namespace) -> None:
    import numpy as np

    from invllava.analysis.figures import save_joint_pca
    from invllava.artifacts.hashing import sha256_file

    output = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output.with_suffix(".json")
    if output.resolve() == metadata_path.resolve():
        raise ValueError("figure and metadata paths must differ")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("representation figures are immutable")
    series: dict[str, np.ndarray] = {}
    source_hashes: dict[str, str] = {}
    expected_ids: np.ndarray | None = None
    for label, artifact, key in args.series:
        if label in series:
            raise ValueError(f"duplicate PCA series label: {label}")
        with np.load(artifact) as data:
            if key not in data:
                raise KeyError(f"{key} is absent from {artifact}")
            ids = data["sample_ids"].astype(str)
            if expected_ids is None:
                expected_ids = ids
            elif not np.array_equal(expected_ids, ids):
                raise ValueError("PCA series must have identical ordered sample IDs")
            series[label] = np.asarray(data[key]).copy()
        source_hashes[label] = sha256_file(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = save_joint_pca(series, output)
    payload = {
        **metadata,
        "series": [
            {"label": label, "artifact": artifact, "key": key}
            for label, artifact, key in args.series
        ],
        "source_sha256": source_hashes,
        "figure_sha256": sha256_file(output),
    }
    atomic_write_json(metadata_path, payload)
    _json({"figure": str(output), "metadata": str(metadata_path)})


def command_plot_representation_cka(args: argparse.Namespace) -> None:
    import re

    import numpy as np

    from invllava.analysis.figures import save_cka_rank_panel
    from invllava.analysis.representations import effective_rank, linear_cka, select_layer_views
    from invllava.artifacts.hashing import sha256_file

    def natural_key(value: str) -> list[Any]:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]

    output = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output.with_suffix(".json")
    if output.resolve() == metadata_path.resolve():
        raise ValueError("figure and metadata paths must differ")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("representation figures are immutable")
    with np.load(args.left) as artifact:
        left_ids = artifact["sample_ids"].astype(str)
        left_keys = sorted(
            (key for key in artifact.files if key.startswith(args.left_prefix)), key=natural_key
        )
        left = {key: np.asarray(artifact[key]).copy() for key in left_keys}
    with np.load(args.right) as artifact:
        right_ids = artifact["sample_ids"].astype(str)
        right_keys = sorted(
            (key for key in artifact.files if key.startswith(args.right_prefix)), key=natural_key
        )
        right = {key: np.asarray(artifact[key]).copy() for key in right_keys}
    if not np.array_equal(left_ids, right_ids):
        raise ValueError("CKA artifacts must have identical ordered sample IDs")
    if not left_keys or not right_keys:
        raise ValueError("CKA prefixes selected no representation arrays")
    left_keys, left_constant = select_layer_views(left, args.left_prefix)
    right_keys, right_constant = select_layer_views(right, args.right_prefix)
    matrix = np.asarray(
        [
            [linear_cka(left[left_key], right[right_key]) for right_key in right_keys]
            for left_key in left_keys
        ]
    )
    left_ranks = [effective_rank(left[key])["participation_ratio"] for key in left_keys]
    right_ranks = [effective_rank(right[key])["participation_ratio"] for key in right_keys]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_cka_rank_panel(
        matrix,
        [key.removeprefix(args.left_prefix) for key in left_keys],
        [key.removeprefix(args.right_prefix) for key in right_keys],
        left_ranks,
        right_ranks,
        output,
    )
    payload = {
        "sample_count": len(left_ids),
        "left": str(args.left),
        "right": str(args.right),
        "left_sha256": sha256_file(args.left),
        "right_sha256": sha256_file(args.right),
        "left_keys": left_keys,
        "right_keys": right_keys,
        "left_pooling_prefix": args.left_prefix,
        "right_pooling_prefix": args.right_prefix,
        "excluded_constant_left": left_constant,
        "excluded_constant_right": right_constant,
        "linear_cka": matrix.tolist(),
        "left_participation_ratio": left_ranks,
        "right_participation_ratio": right_ranks,
        "figure_sha256": sha256_file(output),
    }
    atomic_write_json(metadata_path, payload)
    _json({"figure": str(output), "metadata": str(metadata_path)})


def command_plot_training_curves(args: argparse.Namespace) -> None:
    from invllava.analysis.training_curves import audit_metric_accounting, save_training_curves
    from invllava.artifacts.hashing import sha256_file
    from invllava.train.metrics import read_metric_history

    output = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output.with_suffix(".json")
    if output.resolve() == metadata_path.resolve():
        raise ValueError("figure and metadata paths must differ")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("training-curve outputs are immutable")
    labels = [label for label, _ in args.series]
    if len(labels) != len(set(labels)):
        raise ValueError("training-curve series labels must be unique")
    histories = {label: read_metric_history(path) for label, path in args.series}
    accounting = audit_metric_accounting(dict(args.series), required=args.require_matched_exposure)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = save_training_curves(
        histories,
        output,
        view=args.view,
        require_matched_exposure=args.require_matched_exposure,
    )
    payload = {
        **metadata,
        "metric_accounting_audit": accounting,
        "sources": [
            {"label": label, "metrics": path, "sha256": sha256_file(path)}
            for label, path in args.series
        ],
        "figure_sha256": sha256_file(output),
    }
    atomic_write_json(metadata_path, payload)
    _json({"figure": str(output), "metadata": str(metadata_path)})


def command_plot_score_breakdown(args: argparse.Namespace) -> None:
    from invllava.analysis.figures import save_grouped_score_breakdown
    from invllava.artifacts.hashing import sha256_file

    output = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output.with_suffix(".json")
    if output.resolve() == metadata_path.resolve():
        raise ValueError("figure and metadata paths must differ")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("score-breakdown outputs are immutable")
    labels = [label for label, _ in args.series]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("score breakdown requires at least two unique model labels")

    series: dict[str, dict[str, float]] = {}
    sources = []
    expected_benchmark: str | None = None
    expected_protocol: str | None = None
    for label, score_path in args.series:
        payload = json.loads(Path(score_path).read_text(encoding="utf-8"))
        benchmark = str(payload.get("benchmark", ""))
        protocol = str(payload.get("protocol_id", ""))
        if not benchmark or not protocol:
            raise ValueError(f"score artifact lacks benchmark/protocol identity: {score_path}")
        if expected_benchmark is None:
            expected_benchmark = benchmark
            expected_protocol = protocol
        elif benchmark != expected_benchmark or protocol != expected_protocol:
            raise ValueError("score-breakdown artifacts must share benchmark and protocol")
        values = payload.get("details", {}).get(args.detail_key)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"score artifact has no details.{args.detail_key}: {score_path}")
        series[label] = {str(key): float(value) for key, value in values.items()}
        sources.append(
            {
                "label": label,
                "score": score_path,
                "sha256": sha256_file(score_path),
                "checkpoint_id": payload.get("checkpoint_id"),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = save_grouped_score_breakdown(
        series,
        output,
        title=args.title or str(expected_benchmark),
        ylabel=args.ylabel,
    )
    atomic_write_json(
        metadata_path,
        {
            **metadata,
            "benchmark": expected_benchmark,
            "protocol_id": expected_protocol,
            "detail_key": args.detail_key,
            "sources": sources,
            "figure_sha256": sha256_file(output),
        },
    )
    _json({"figure": str(output), "metadata": str(metadata_path)})


def command_plot_profile_comparison(args: argparse.Namespace) -> None:
    from invllava.analysis.figures import save_profile_comparison
    from invllava.artifacts.hashing import sha256_file

    output = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output.with_suffix(".json")
    if output.resolve() == metadata_path.resolve():
        raise ValueError("figure and metadata paths must differ")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("profile-comparison outputs are immutable")
    labels = [label for label, _ in args.profile]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("profile comparison requires at least two unique labels")
    profiles = {
        label: json.loads(Path(path).read_text(encoding="utf-8")) for label, path in args.profile
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = save_profile_comparison(profiles, output)
    atomic_write_json(
        metadata_path,
        {
            **metadata,
            "sources": [
                {"label": label, "profile": path, "sha256": sha256_file(path)}
                for label, path in args.profile
            ],
            "figure_sha256": sha256_file(output),
        },
    )
    _json({"figure": str(output), "metadata": str(metadata_path)})


def _load_item_scores(path: str | Path) -> dict[str, float]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    per_item = value.get("details", {}).get("per_item")
    if not isinstance(per_item, dict) or not per_item:
        raise ValueError(f"score artifact has no details.per_item: {path}")
    return {str(key): float(item) for key, item in per_item.items()}


def command_paired_interval(args: argparse.Namespace) -> None:
    from dataclasses import asdict

    import numpy as np

    from invllava.analysis.statistics import (
        grouped_paired_bootstrap,
        mme_paired_bootstrap,
        paired_bootstrap,
        stratified_macro_paired_bootstrap,
    )
    from invllava.eval.datasets import load_examples

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    left_payload = json.loads(Path(args.left_score).read_text(encoding="utf-8"))
    right_payload = json.loads(Path(args.right_score).read_text(encoding="utf-8"))
    if left_payload.get("benchmark") != right_payload.get("benchmark"):
        raise ValueError("paired score artifacts belong to different benchmarks")
    for field in ("scorer_id", "examples_sha256"):
        if left_payload.get(field) != right_payload.get(field):
            raise ValueError(f"paired score artifacts have different {field}")
    mme_benchmarks = {"mme-perception", "mme-cognition"}
    mmbench_benchmarks = {"mmbench-en", "mmbench-cn"}
    benchmark = left_payload.get("benchmark")
    image_audit = getattr(args, "image_audit", None)
    if image_audit and (
        not args.examples or benchmark in {*mme_benchmarks, *mmbench_benchmarks, "mmstar"}
    ):
        raise ValueError("--image-audit requires --examples and an item-mean benchmark")
    grouping = None
    if benchmark in {*mme_benchmarks, *mmbench_benchmarks, "mmstar"} and not args.examples:
        raise ValueError(f"{benchmark} intervals require --examples for metric-aware resampling")
    left = _load_item_scores(args.left_score)
    right = _load_item_scores(args.right_score)
    if left.keys() != right.keys():
        raise ValueError("paired score artifacts have different sample IDs")
    ids = sorted(left)
    left_values = np.asarray([left[item] for item in ids])
    right_values = np.asarray([right[item] for item in ids])
    if args.examples:
        from invllava.artifacts.hashing import sha256_file

        if sha256_file(args.examples) != left_payload.get("examples_sha256"):
            raise ValueError("score artifacts do not bind the supplied examples")
        example_list = load_examples(args.examples)
        if benchmark in mmbench_benchmarks:
            if any(example.group_id is None for example in example_list):
                raise ValueError("MMBench interval examples require circular group IDs")
            example_groups = {str(example.group_id) for example in example_list}
            if example_groups != left.keys():
                raise ValueError("MMBench rotation groups do not match score sample IDs")
            interval = grouped_paired_bootstrap(
                left_values,
                right_values,
                np.asarray(ids),
                resamples=args.resamples,
                confidence=args.confidence,
                seed=args.seed,
            )
        else:
            examples = {example.id: example for example in example_list}
            if examples.keys() != left.keys():
                raise ValueError("grouped interval examples do not match score sample IDs")
            groups = np.asarray([examples[item].group_id or item for item in ids])
            if image_audit:
                from invllava.data.overlap import image_groups_from_audit

                content_groups, grouping = image_groups_from_audit(
                    {item: examples[item].images for item in ids},
                    image_audit,
                    annotation_sha256=left_payload["examples_sha256"],
                )
                groups = np.asarray([content_groups[item] for item in ids])
        if benchmark in mme_benchmarks:
            categories = np.asarray(
                [str(examples[item].metadata.get("category", "")) for item in ids]
            )
            if np.any(categories == ""):
                raise ValueError("MME grouped bootstrap requires category metadata")
            interval = mme_paired_bootstrap(
                left_values,
                right_values,
                groups,
                categories,
                resamples=args.resamples,
                confidence=args.confidence,
                seed=args.seed,
            )
        elif benchmark == "mmstar":
            strata = np.asarray(
                [str(examples[item].metadata.get("l2_category", "")) for item in ids]
            )
            if np.any(strata == ""):
                raise ValueError("MMStar bootstrap requires l2_category metadata")
            interval = stratified_macro_paired_bootstrap(
                left_values,
                right_values,
                strata,
                resamples=args.resamples,
                confidence=args.confidence,
                seed=args.seed,
            )
        elif benchmark not in mmbench_benchmarks:
            interval = grouped_paired_bootstrap(
                left_values,
                right_values,
                groups,
                resamples=args.resamples,
                confidence=args.confidence,
                seed=args.seed,
            )
    else:
        interval = paired_bootstrap(
            left_values,
            right_values,
            resamples=args.resamples,
            confidence=args.confidence,
            seed=args.seed,
        )
    payload = {
        "left_score": str(args.left_score),
        "right_score": str(args.right_score),
        "sample_count": len(ids),
        "grouping": grouping,
        **asdict(interval),
    }
    atomic_write_json(args.output, payload)
    _json(payload)


def command_stratify_scores(args: argparse.Namespace) -> None:
    from invllava.analysis.strata import numeric_score_strata
    from invllava.artifacts.hashing import sha256_file
    from invllava.eval.datasets import load_examples

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    labels = [label for label, _ in args.series]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("stratified analysis requires at least two unique labels")

    score_payloads = {
        label: json.loads(Path(path).read_text(encoding="utf-8")) for label, path in args.series
    }
    benchmarks = {payload.get("benchmark") for payload in score_payloads.values()}
    if len(benchmarks) != 1 or None in benchmarks:
        raise ValueError("stratified score artifacts belong to different benchmarks")
    if next(iter(benchmarks)) in {
        "mme-cognition",
        "mme-perception",
        "mmbench-en",
        "mmbench-cn",
        "mmstar",
    }:
        raise ValueError(
            "numeric score strata require an item-mean metric; use metric-aware category analysis"
        )
    protocols = {payload.get("protocol_id") for payload in score_payloads.values()}
    if None in protocols:
        raise ValueError("stratified score artifacts omit protocol identity")
    equivalence_fields = ("scorer_id", "examples_sha256", "protocol_config_sha256")
    equivalence = {}
    for field in equivalence_fields:
        values = {payload.get(field) for payload in score_payloads.values()}
        if len(values) != 1 or None in values:
            raise ValueError(f"stratified score artifacts have different {field}")
        equivalence[field] = next(iter(values))
    examples_digest = sha256_file(args.examples)
    if equivalence["examples_sha256"] != examples_digest:
        raise ValueError("stratified score artifacts do not bind the supplied examples")

    scores = {label: _load_item_scores(path) for label, path in args.series}
    example_list = load_examples(args.examples)
    examples = {example.id: example for example in example_list}
    if len(examples) != len(example_list):
        raise ValueError("stratified analysis examples contain duplicate IDs")
    groups, grouping = None, None
    if getattr(args, "image_audit", None):
        from invllava.data.overlap import image_groups_from_audit

        groups, grouping = image_groups_from_audit(
            {sample_id: example.images for sample_id, example in examples.items()},
            args.image_audit,
            annotation_sha256=examples_digest,
        )
    analysis = numeric_score_strata(
        scores,
        {sample_id: example.metadata for sample_id, example in examples.items()},
        metadata_key=args.metadata_key,
        boundaries=tuple(args.boundaries),
        primary_model=args.primary,
        groups_by_id=groups,
        resamples=args.resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    payload = {
        **analysis,
        "benchmark": next(iter(benchmarks)),
        "protocol_id": score_payloads[args.primary]["protocol_id"],
        "source_protocol_ids": sorted(protocols),
        "protocol_equivalence": equivalence,
        "examples": str(args.examples),
        "examples_sha256": examples_digest,
        "grouping": grouping,
        "sources": [
            {
                "label": label,
                "score": path,
                "sha256": sha256_file(path),
                "checkpoint_id": score_payloads[label].get("checkpoint_id"),
            }
            for label, path in args.series
        ],
    }
    atomic_write_json(output, payload)
    _json(payload)


def command_language_interval(args: argparse.Namespace) -> None:
    from invllava.eval.language_compare import compare_language_results

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    payload = compare_language_results(
        args.left_result,
        args.right_result,
        metric=args.metric,
        filter_name=args.filter,
        resamples=args.resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    atomic_write_json(args.output, payload)
    _json(payload)


def _load_prediction_records(path: str | Path) -> list[Any]:
    from invllava.eval.records import PredictionRecord

    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                value["references"] = tuple(value.get("references", ()))
                value["image_ids"] = tuple(value.get("image_ids", ()))
                records.append(PredictionRecord(**value))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"invalid prediction at {path}:{line_number}") from error
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"prediction file contains duplicate sample IDs: {path}")
    return records


def command_select_cases(args: argparse.Namespace) -> None:
    from invllava.analysis.cases import select_multimodel_cases
    from invllava.artifacts.hashing import sha256_file

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if not 0 <= args.correct_threshold <= 1:
        raise ValueError("correct-threshold must be in [0,1]")
    labels = [label for label, _, _ in args.model]
    if len(labels) != len(set(labels)):
        raise ValueError("qualitative model labels must be unique")
    sources: dict[str, dict[str, str]] = {}
    correct_by_model: dict[str, dict[str, bool]] = {}
    expected_ids: set[str] | None = None
    for label, predictions_path, score_path in args.model:
        predictions = _load_prediction_records(predictions_path)
        prediction_ids = {record.sample_id for record in predictions}
        scores = {
            key: value >= args.correct_threshold
            for key, value in _load_item_scores(score_path).items()
        }
        if scores.keys() != prediction_ids:
            raise ValueError(f"prediction and score IDs differ for model {label}")
        if expected_ids is None:
            expected_ids = prediction_ids
        elif expected_ids != prediction_ids:
            raise ValueError("qualitative models must have identical sample IDs")
        correct_by_model[label] = scores
        sources[label] = {
            "predictions": predictions_path,
            "predictions_sha256": sha256_file(predictions_path),
            "score": score_path,
            "score_sha256": sha256_file(score_path),
        }
    selected, patterns, group_counts = select_multimodel_cases(
        correct_by_model,
        primary_model=args.primary,
        per_group=args.per_group,
        seed=args.seed,
    )
    payload = {
        "seed": args.seed,
        "per_group": args.per_group,
        "correct_threshold": args.correct_threshold,
        "primary_model": args.primary,
        "model_labels": labels,
        "sources": sources,
        "candidate_group_counts": group_counts,
        "selected_group_counts": {key: len(values) for key, values in selected.items()},
        "selection": selected,
        "correctness": patterns,
    }
    atomic_write_json(args.output, payload)
    _json(payload)


def command_export_cases(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from invllava.analysis.cases import materialize_case_images
    from invllava.artifacts.atomic import atomic_write_text
    from invllava.artifacts.hashing import sha256_file
    from invllava.eval.datasets import load_examples, write_examples

    if (
        Path(args.output).exists()
        or (args.markdown and Path(args.markdown).exists())
        or (args.examples_output and Path(args.examples_output).exists())
    ):
        raise FileExistsError("case-study outputs are immutable")
    if args.examples_output and not args.image_root:
        raise ValueError("--examples-output requires --image-root for self-contained cases")
    selection_payload = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selection = selection_payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection artifact has no selection mapping")
    examples = {example.id: example for example in load_examples(args.examples)}
    labels = [label for label, _ in args.prediction]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("case export requires at least two unique prediction labels")
    predictions = {
        label: {record.sample_id: record for record in _load_prediction_records(prediction_path)}
        for label, prediction_path in args.prediction
    }
    declared_labels = selection_payload.get("model_labels")
    if declared_labels != labels:
        raise ValueError("prediction labels must match the case-selection model order")
    selected_ids = [str(sample_id) for values in selection.values() for sample_id in values]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("case selection contains duplicate IDs")
    available = set(examples)
    for records in predictions.values():
        available.intersection_update(records)
    missing = set(selected_ids) - available
    if missing:
        raise ValueError(f"selected case inputs are incomplete: {sorted(missing)}")
    copied_images: dict[str, list[str]] | None = None
    image_inventory: list[dict[str, Any]] | None = None
    if args.image_root:
        copied_images, image_inventory = materialize_case_images(
            examples,
            selected_ids,
            args.image_root,
        )
    selected_examples_path: Path | None = None
    if args.examples_output:
        assert copied_images is not None
        selected_examples_path = Path(args.examples_output).resolve()
        write_examples(
            [
                replace(
                    examples[sample_id],
                    images=tuple(Path(path) for path in copied_images[sample_id]),
                )
                for sample_id in selected_ids
            ],
            selected_examples_path,
        )

    cases: list[dict[str, Any]] = []
    for bucket, values in selection.items():
        for sample_id in values:
            sample_id = str(sample_id)
            example = examples[sample_id]
            cases.append(
                {
                    "bucket": str(bucket),
                    "sample_id": sample_id,
                    "images": (
                        copied_images[sample_id]
                        if copied_images is not None
                        else [str(path) for path in example.images]
                    ),
                    "prompt": example.prompt,
                    "references": list(example.references),
                    "outputs": {
                        label: predictions[label][sample_id].prediction for label in labels
                    },
                    "correctness": selection_payload.get("correctness", {}).get(sample_id),
                    "error_label": None,
                    "author_note": None,
                }
            )
    payload = {
        "schema_version": 1,
        "source_hashes": {
            "selection": sha256_file(args.selection),
            "examples": sha256_file(args.examples),
            **{f"predictions.{label}": sha256_file(path) for label, path in args.prediction},
        },
        "cases": cases,
    }
    if image_inventory is not None:
        payload["image_root"] = str(Path(args.image_root).resolve())
        payload["image_inventory"] = image_inventory
    if selected_examples_path is not None:
        payload["selected_examples"] = str(selected_examples_path)
        payload["selected_examples_sha256"] = sha256_file(selected_examples_path)
    atomic_write_json(args.output, payload)
    if args.markdown:
        lines = [
            "# Auditable qualitative cases",
            "",
            "Outputs below are unedited. Fill error labels and author notes "
            "only in the source JSON.",
            "",
        ]
        for case in cases:
            lines.extend(
                [
                    f"## {case['bucket']}: {case['sample_id']}",
                    "",
                    f"Image paths: {', '.join(case['images'])}",
                    "",
                    "Prompt:",
                    "",
                    "    " + str(case["prompt"]).replace("\n", "\n    "),
                    "",
                    f"Reference: {json.dumps(case['references'], ensure_ascii=False)}",
                    "",
                ]
            )
            for label, output in case["outputs"].items():
                lines.extend(
                    [
                        f"{label} output:",
                        "",
                        "    " + str(output).replace("\n", "\n    "),
                        "",
                    ]
                )
            lines.extend(["Error label: _pending author annotation_", ""])
        atomic_write_text(args.markdown, "\n".join(lines))
    _json({"cases": len(cases), "output": args.output, "markdown": args.markdown})


def command_prepare_interventions(args: argparse.Namespace) -> None:
    from invllava.analysis.interventions import blank_image_examples, shuffled_image_examples
    from invllava.artifacts.hashing import sha256_file
    from invllava.eval.datasets import load_examples, validate_image_paths, write_examples

    output = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("intervention artifacts are immutable")
    examples = load_examples(args.examples)
    validate_image_paths(examples)
    if args.mode == "shuffled":
        if args.image_root:
            raise ValueError("--image-root is only valid for blank-image intervention")
        intervened = shuffled_image_examples(examples, seed=args.seed)
    else:
        if not args.image_root:
            raise ValueError("blank-image intervention requires --image-root")
        intervened = blank_image_examples(examples, args.image_root, rgb=tuple(args.blank_rgb))
    write_examples(intervened, output)
    image_hashes = (
        {example.id: sha256_file(example.images[0]) for example in intervened}
        if args.mode == "blank"
        else {}
    )
    manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "seed": args.seed if args.mode == "shuffled" else None,
        "blank_rgb": args.blank_rgb if args.mode == "blank" else None,
        "sample_count": len(intervened),
        "source_examples_sha256": sha256_file(args.examples),
        "examples_sha256": sha256_file(output),
        "generated_image_sha256": image_hashes,
    }
    if args.mode == "shuffled":
        manifest["shuffle_unit"] = "decoded_rgb_image"
        manifest["image_group_count"] = len(
            {example.metadata["intervention_original_rgb_sha256"] for example in intervened}
        )
    atomic_write_json(manifest_path, manifest)
    _json({"examples": str(output), "manifest": str(manifest_path), "count": len(intervened)})


def command_intervention_summary(args: argparse.Namespace) -> None:
    import numpy as np

    from invllava.analysis.interventions import (
        intervention_effect,
        intervention_response_effect,
    )
    from invllava.artifacts.hashing import sha256_file

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    original = _load_item_scores(args.original_score)
    intervened = _load_item_scores(args.intervened_score)
    if original.keys() != intervened.keys():
        raise ValueError("intervention score artifacts have different item IDs")
    ids = sorted(original)
    payload = {
        "original_score": args.original_score,
        "intervened_score": args.intervened_score,
        "sample_count": len(ids),
        **intervention_effect(
            np.asarray([original[sample_id] for sample_id in ids]),
            np.asarray([intervened[sample_id] for sample_id in ids]),
        ),
    }
    prediction_paths = (args.original_predictions, args.intervened_predictions)
    if any(prediction_paths) and not all(prediction_paths):
        raise ValueError("provide both original and intervened prediction files")
    if all(prediction_paths):
        original_records = {
            record.sample_id: record
            for record in _load_prediction_records(args.original_predictions)
        }
        intervened_records = {
            record.sample_id: record
            for record in _load_prediction_records(args.intervened_predictions)
        }
        if original_records.keys() != intervened_records.keys() or set(ids) != set(
            original_records
        ):
            raise ValueError("intervention predictions and scores have different item IDs")
        response_effect = intervention_response_effect(
            [original_records[sample_id].prediction for sample_id in ids],
            [intervened_records[sample_id].prediction for sample_id in ids],
        )
        for key in ("exact_changed_indices", "normalized_changed_indices"):
            response_effect[key.removesuffix("_indices") + "_ids"] = [
                ids[index] for index in response_effect.pop(key)
            ]
        payload.update(
            {
                "original_predictions": args.original_predictions,
                "intervened_predictions": args.intervened_predictions,
                "original_predictions_sha256": sha256_file(args.original_predictions),
                "intervened_predictions_sha256": sha256_file(args.intervened_predictions),
                **response_effect,
            }
        )
    atomic_write_json(args.output, payload)
    _json(payload)


def command_profile_native(args: argparse.Namespace) -> None:
    downloads_authorized = _downloads_authorized(args)
    import platform

    import torch

    from invllava.analysis.profiling import profile_autoregressive, profile_callable
    from invllava.artifacts.hashing import sha256_file
    from invllava.eval.datasets import load_examples, validate_image_paths
    from invllava.eval.types import generation_request
    from invllava.runtime.native import load_native_inference_runtime

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    examples = load_examples(args.examples)
    validate_image_paths(examples)
    largest_batch = max(args.batch_sizes)
    if largest_batch > len(examples) or min(args.batch_sizes) <= 0:
        raise ValueError("profile batch sizes must be positive and fit the example panel")
    runtime = load_native_inference_runtime(
        experiment=args.experiment,
        checkpoint=args.checkpoint,
        config_root=args.config_root,
        runtime_ref=args.runtime_ref,
        device=args.device,
        max_new_tokens=args.decode_tokens,
        local_files_only=not downloads_authorized,
    )
    resolved = runtime.resolved
    model = runtime.model
    generator = runtime.generator
    fusion_feature_dim = (
        resolved.model.vision.feature_dim
        if resolved.model.architecture == "inverse_llava"
        and resolved.model.fusion.operator != "disabled"
        else None
    )
    profiles: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        requests = [generation_request(example) for example in examples[:batch_size]]
        expanded = generator.prepare_many(requests)
        value: dict[str, Any] = {
            "autoregressive": profile_autoregressive(
                model.language_model,
                expanded,
                fixed_decode_tokens=args.decode_tokens,
                fusion_feature_dim=fusion_feature_dim,
                warmups=args.warmups,
                repetitions=args.repetitions,
            ).to_dict()
        }
        if args.include_preparation:
            value["image_prompt_preparation"] = profile_callable(
                lambda current=requests: generator.prepare_many(current),
                warmups=args.warmups,
                repetitions=args.repetitions,
            ).to_dict()
        profiles[str(batch_size)] = value
    checkpoint_digest = runtime.checkpoint_sha256
    parameters = list(model.parameters())
    payload = {
        "measurement": "measured",
        "experiment_id": resolved.id,
        "scientific_id": content_id(scientific_payload(resolved), prefix="sci"),
        "architecture": resolved.model.architecture,
        "checkpoint_id": f"ckpt-{checkpoint_digest[:16]}",
        "checkpoint_sha256": checkpoint_digest,
        "examples_sha256": sha256_file(args.examples),
        "hardware": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else platform.processor()
        ),
        "torch_version": torch.__version__,
        "dtype": resolved.model.torch_dtype,
        "attention_backend": resolved.runtime.attention_backend,
        "kernel_optimization": runtime.kernel_report.to_dict(),
        "numerical_policy": runtime.numerical_policy,
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "runtime_requires_grad_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_count_scope": (
            "Loaded inference flags; report original training-policy counts separately."
        ),
        "fixed_decode_policy": "greedy tokens with EOS ignored for equal work",
        "profiles": profiles,
    }
    atomic_write_json(args.output, payload)
    _json(payload)


def command_profile_hf_llava(args: argparse.Namespace) -> None:
    """Profile an audited HF LLaVA conversion under the native decode contract."""

    import platform

    import torch

    from invllava.analysis.profiling import profile_autoregressive, profile_callable
    from invllava.artifacts.hashing import sha256_file
    from invllava.eval.datasets import load_examples, validate_image_paths
    from invllava.eval.interop.hf_llava import HuggingFaceLLaVAGenerator
    from invllava.eval.types import generation_request

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(args.checkpoint).resolve()
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            "matched reference profiling requires a local indexed safetensors checkpoint"
        )
    examples = load_examples(args.examples)
    validate_image_paths(examples)
    largest_batch = max(args.batch_sizes)
    if largest_batch > len(examples) or min(args.batch_sizes) <= 0:
        raise ValueError("profile batch sizes must be positive and fit the example panel")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    from invllava.runtime.numerics import configure_torch_numerics_policy

    numerical_policy = configure_torch_numerics_policy(
        mixed_precision="no" if args.dtype == "float32" else args.dtype,
        allow_tf32=True,
        deterministic_algorithms=False,
        cudnn_benchmark=False,
    )
    generator = HuggingFaceLLaVAGenerator(
        str(checkpoint),
        revision=args.revision,
        dtype=dtype,
        device=args.device,
        max_new_tokens=args.decode_tokens,
        attention_backend=args.attention_backend,
        image_aspect_ratio=args.image_aspect_ratio,
        local_files_only=True,
    )
    parity_requests = [generation_request(examples[0])]
    preparation_parity = generator.verify_preparation_parity(parity_requests)
    profiles: dict[str, Any] = {}
    embed_tokens = generator.model.model.language_model.embed_tokens
    for batch_size in args.batch_sizes:
        requests = [generation_request(example) for example in examples[:batch_size]]
        expanded = generator.prepare_many(requests)
        value: dict[str, Any] = {
            "autoregressive": profile_autoregressive(
                generator.model,
                expanded,
                fixed_decode_tokens=args.decode_tokens,
                warmups=args.warmups,
                repetitions=args.repetitions,
                embed_tokens=embed_tokens,
            ).to_dict()
        }
        if args.include_preparation:
            value["image_prompt_preparation"] = profile_callable(
                lambda current=requests: generator.prepare_many(current),
                warmups=args.warmups,
                repetitions=args.repetitions,
            ).to_dict()
        profiles[str(batch_size)] = value
    parameters = list(generator.model.parameters())
    payload = {
        "measurement": "measured",
        "runtime": "huggingface_llava_reference",
        "checkpoint": str(checkpoint),
        "checkpoint_index_sha256": sha256_file(index_path),
        "checkpoint_revision": args.revision,
        "examples_sha256": sha256_file(args.examples),
        "hardware": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else platform.processor()
        ),
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "numerical_policy": numerical_policy,
        "image_aspect_ratio": args.image_aspect_ratio,
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "runtime_requires_grad_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "fixed_decode_policy": "greedy tokens with EOS ignored for equal work",
        "preparation_parity": preparation_parity,
        "profiles": profiles,
    }
    atomic_write_json(output, payload)
    _json(payload)


def command_kernel_audit(args: argparse.Namespace) -> None:
    """Qualify an optional kernel runtime without model or dataset downloads."""

    from invllava.analysis.kernel_audit import run_kernel_audit
    from invllava.artifacts.hashing import sha256_file
    from invllava.config.schema import RuntimeSpec

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    runtime_path = Path(args.runtime_config)
    runtime = RuntimeSpec.model_validate(yaml.safe_load(runtime_path.read_text(encoding="utf-8")))
    configure_runtime_cache(runtime.cache_root)
    payload = run_kernel_audit(
        runtime,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        visual_size=args.visual_size,
        layers=args.layers,
        warmups=args.warmups,
        repetitions=args.repetitions,
        atol=args.atol,
        rtol=args.rtol,
        trace=args.trace,
    )
    payload["runtime_config"] = str(runtime_path.resolve())
    payload["runtime_config_sha256"] = sha256_file(runtime_path)
    atomic_write_json(output, payload)
    _json(payload)
    if not payload["passed"]:
        raise RuntimeError(f"kernel qualification failed; inspect {output}")


def command_train(args: argparse.Namespace) -> None:
    downloads_authorized = _downloads_authorized(args)
    if args.allow_download and not downloads_authorized:
        raise PermissionError("--allow-download also requires INVLLAVA_ALLOW_DOWNLOADS=1")
    if not downloads_authorized:
        # Keep scientific execution independent of Hub availability. These are
        # set before importing Transformers and datasets in this worker.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from torch.utils.data import DataLoader, Subset

    from invllava.artifacts.atomic import atomic_write_text
    from invllava.artifacts.hashing import sha256_file
    from invllava.artifacts.manifest import ArtifactRecord, RunManifest
    from invllava.config.validation import require_frozen_execution
    from invllava.data.collate import (
        MultimodalCollator,
        encode_vicuna_v1,
        expand_to_square,
        image_mean_background,
    )
    from invllava.data.dataset import NormalizedConversationDataset
    from invllava.data.manifest import PreparedDatasetManifest
    from invllava.data.sampling import stratified_nested_indices
    from invllava.model.loaders import build_model, load_image_processor, load_tokenizer
    from invllava.model.projector_checkpoint import (
        PROJECTOR_FILENAME,
        load_projector_weights,
    )
    from invllava.train.checkpoint import (
        MODEL_DELTA_FILENAME,
        checkpoint_inventory,
        load_trainable_weights,
    )
    from invllava.train.engine import TrainingEngine
    from invllava.train.sampler import (
        EpochSeededSampler,
        ModalityLengthGroupedSampler,
        dataloader_generator,
    )
    from invllava.train.state import seed_everything

    repository = ConfigRepository(args.config_root)
    resolved = repository.resolve(
        args.experiment,
        runtime_ref=args.runtime_ref,
        microbatch_size=args.microbatch_size,
        gradient_checkpointing=(
            args.gradient_checkpointing == "on" if args.gradient_checkpointing is not None else None
        ),
        maximum_samples=args.maximum_samples,
    )
    expected_allocator = resolved.runtime.cuda_allocator_conf
    actual_allocator = os.environ.get("PYTORCH_ALLOC_CONF")
    if actual_allocator != expected_allocator:
        raise RuntimeError(
            "CUDA allocator policy differs from the resolved runtime; launch through "
            "scripts/launch_train.py: "
            f"environment={actual_allocator!r}, runtime={expected_allocator!r}"
        )
    require_frozen_execution(resolved)
    # Fusion and LoRA parameters are created below, so the scientific seed must
    # be active before model construction on every process.
    seed_everything(resolved.training.seed)
    configure_runtime_cache(resolved.runtime.cache_root)
    from invllava.runtime.numerics import configure_torch_numerics

    numerical_policy = configure_torch_numerics(resolved.runtime)

    prepared_manifest = PreparedDatasetManifest.read(args.prepared_manifest)
    prepared_manifest.verify(
        args.prepared_jsonl,
        expected_data_id=resolved.data.id,
        expected_source_revision=resolved.data.annotation.revision,
        expected_source_sha256=resolved.data.annotation.sha256,
        expected_source_filter=resolved.data.include_sources,
    )
    scientific_id = content_id(scientific_payload(resolved), prefix="sci")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not args.run_id:
        raise ValueError("multi-process launch requires one explicit shared --run-id")
    run_id = args.run_id or f"{resolved.id}-{scientific_id}-{timestamp}"
    run_root = _execution_run_root(resolved.runtime.run_root, args.run_root)
    run_dir = run_root / run_id
    if args.resume_from and (run_dir / "RUN_COMPLETE").exists():
        raise ValueError("completed runs are immutable and cannot be resumed")
    full_dataset = NormalizedConversationDataset(args.prepared_jsonl)
    manifest_samples = prepared_manifest.audit.get("samples")
    if manifest_samples != len(full_dataset) or not full_dataset:
        raise ValueError(
            "prepared manifest sample count does not match a non-empty normalized dataset"
        )
    if prepared_manifest.audit.get("missing_images") or prepared_manifest.audit.get(
        "duplicate_ids"
    ):
        raise ValueError("prepared dataset manifest records unresolved audit failures")
    selected_indices = stratified_nested_indices(
        full_dataset.ids,
        full_dataset.sources,
        fraction=resolved.data.sample_fraction,
        seed=resolved.data.sample_seed,
        maximum=resolved.data.max_samples,
    )
    if not selected_indices:
        raise ValueError("experiment selected no training samples")
    dataset = Subset(full_dataset, selected_indices)
    selected_ids = [full_dataset.ids[index] for index in selected_indices]
    selected_lengths = [full_dataset.modality_lengths[index] for index in selected_indices]
    selected_index_set = set(selected_indices)
    vicuna_masking_issues = [
        {"row_index": index, "sample_id": sample_id, "issue": issue}
        for index, sample_id, issue in full_dataset.vicuna_v1_masking_issues
        if index in selected_index_set
    ]
    vicuna_masking_report = {
        "schema_version": 1,
        "policy": "llava-v1-mask-entire-sample",
        "selected_samples": len(selected_indices),
        "masked_samples": len(vicuna_masking_issues),
        "issues": vicuna_masking_issues,
    }
    tokenizer = load_tokenizer(
        resolved.model,
        local_files_only=not downloads_authorized,
    )
    # Exercise ordinary and longest selected prompts before allocating the 7B
    # model. This catches tokenizer/template drift while failure is still
    # inexpensive. The training loader remains authoritative for every row.
    audit_positions = list(range(min(128, len(selected_indices))))
    audit_positions.extend(
        sorted(
            range(len(selected_indices)),
            key=lambda position: abs(selected_lengths[position]),
            reverse=True,
        )[:128]
    )
    tokenization_audit_indices = tuple(
        dict.fromkeys(selected_indices[position] for position in audit_positions)
    )
    for index in tokenization_audit_indices:
        encode_vicuna_v1(tokenizer, full_dataset[index].turns)
    model, report = build_model(
        resolved.model,
        attention_backend=resolved.runtime.attention_backend,
        local_files_only=not downloads_authorized,
    )
    if bool(resolved.initial_checkpoint_id) != bool(args.initial_checkpoint):
        raise ValueError(
            "experiment initial_checkpoint_id and --initial-checkpoint must either "
            "both be set or both be absent"
        )
    checkpoint_inputs = {
        "language": (f"{resolved.model.language.checkpoint}@{resolved.model.language.revision}"),
        "vision": f"{resolved.model.vision.checkpoint}@{resolved.model.vision.revision}",
    }
    if resolved.model.projector is not None and resolved.model.projector.initialization == "random":
        if args.projector_checkpoint:
            raise ValueError("random-projector experiment forbids --projector-checkpoint")
        checkpoint_inputs["projector_initialization"] = "random-seeded"
    elif resolved.model.projector is not None:
        if not args.projector_checkpoint:
            raise ValueError(
                "controlled LLaVA training requires a converted --projector-checkpoint; "
                "use an explicitly declared random-projector experiment for a single-stage control"
            )
        load_projector_weights(args.projector_checkpoint, model.multimodal_projector)
        projector_path = Path(args.projector_checkpoint) / PROJECTOR_FILENAME
        projector_digest = sha256_file(projector_path)
        projector_identity = f"sha256:{projector_digest}"
        if resolved.model.projector.initial_checkpoint_id != projector_identity:
            raise ValueError(
                "projector checkpoint digest does not match model.projector.initial_checkpoint_id"
            )
        checkpoint_inputs["alignment_projector"] = projector_identity
    elif args.projector_checkpoint:
        raise ValueError("--projector-checkpoint is only valid for llava_reference training")
    if args.initial_checkpoint:
        load_trainable_weights(args.initial_checkpoint, model)
        trainable_path = Path(args.initial_checkpoint) / MODEL_DELTA_FILENAME
        initial_identity = f"sha256:{sha256_file(trainable_path)}"
        if resolved.initial_checkpoint_id != initial_identity:
            raise ValueError("initial checkpoint digest does not match initial_checkpoint_id")
        checkpoint_inputs["initial_trainable_delta"] = initial_identity
    from invllava.runtime.optimization import configure_model_kernels

    kernel_report = configure_model_kernels(model, resolved.runtime.kernel_optimization)
    processor = load_image_processor(
        resolved.model.vision,
        local_files_only=not downloads_authorized,
    )

    def transform(image: Any) -> torch.Tensor:
        if resolved.model.vision.aspect_ratio == "pad":
            background = image_mean_background(processor.image_mean)
            image = expand_to_square(image, background)
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    collator = MultimodalCollator(tokenizer, transform, resolved.model.language.max_length)
    process_rank = int(os.environ.get("RANK", "0"))
    sampler = (
        ModalityLengthGroupedSampler(
            selected_lengths,
            batch_size=resolved.training.per_device_batch_size,
            group_count=(
                resolved.runtime.num_processes * resolved.training.gradient_accumulation_steps
            ),
            seed=resolved.training.seed,
        )
        if resolved.training.group_by_modality_length
        else EpochSeededSampler(dataset, seed=resolved.training.seed)
    )
    loader_worker_options: dict[str, Any] = {}
    if resolved.runtime.dataloader_workers > 0:
        loader_worker_options = {
            "prefetch_factor": resolved.runtime.dataloader_prefetch_factor,
            "persistent_workers": resolved.runtime.dataloader_persistent_workers,
        }
    loader = DataLoader(
        dataset,
        batch_size=resolved.training.per_device_batch_size,
        sampler=sampler,
        num_workers=resolved.runtime.dataloader_workers,
        collate_fn=collator,
        pin_memory=resolved.runtime.accelerator == "cuda",
        generator=dataloader_generator(resolved.training.seed, rank=process_rank),
        **loader_worker_options,
    )
    engine = TrainingEngine(
        config=resolved,
        model=model,
        dataloader=loader,
        run_dir=run_dir,
        resume=args.resume_from is not None,
    )
    if engine.accelerator.is_main_process:
        selected_path = run_dir / "selected_sample_ids.txt"
        selected_payload = "\n".join(selected_ids) + "\n"
        resolved_path = run_dir / "resolved_config.json"
        prepared_snapshot_path = run_dir / "prepared_dataset_manifest.json"
        masking_report_path = run_dir / "vicuna_v1_masking_report.json"
        prepared_payload = json.loads(Path(args.prepared_manifest).read_text(encoding="utf-8"))
        resolved_payload = resolved.model_dump(mode="json")
        resolved_payload.pop("source_files", None)
        if args.resume_from:
            checkpoint_parent = Path(args.resume_from).resolve().parent
            if checkpoint_parent != (run_dir / "checkpoints").resolve():
                raise ValueError(
                    "exact resume checkpoint must belong to the selected run directory"
                )
            manifest = RunManifest.read(run_dir / "manifest.json")
            if (
                manifest.run_id != run_id
                or manifest.experiment_id != resolved.id
                or manifest.scientific_id != scientific_id
            ):
                raise ValueError("resume manifest identity does not match resolved experiment")
            if selected_path.read_text(encoding="utf-8") != selected_payload:
                raise ValueError("resume sample IDs do not match the original run")
            if json.loads(resolved_path.read_text(encoding="utf-8")) != resolved_payload:
                raise ValueError("resume resolved configuration does not match the original run")
            if json.loads(prepared_snapshot_path.read_text(encoding="utf-8")) != prepared_payload:
                raise ValueError("resume prepared-data evidence does not match the original run")
            current_source = os.environ.get("INVLLAVA_EXECUTION_SOURCE_SHA256")
            if current_source != manifest.execution_source_sha256:
                if not args.resume_source_change_reason:
                    raise ValueError(
                        "resume execution source differs from the original run; "
                        "provide --resume-source-change-reason for an auditable recovery"
                    )
                history_path = run_dir / "resume_history.json"
                history = (
                    json.loads(history_path.read_text(encoding="utf-8"))
                    if history_path.is_file()
                    else {"schema_version": 1, "sessions": []}
                )
                history["sessions"].append(
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "checkpoint": str(Path(args.resume_from).resolve()),
                        "original_execution_source_sha256": manifest.execution_source_sha256,
                        "resume_execution_source_sha256": current_source,
                        "reason": args.resume_source_change_reason,
                    }
                )
                atomic_write_json(history_path, history)
                history_record = ArtifactRecord(
                    role="resume_history",
                    relative_path=history_path.relative_to(run_dir).as_posix(),
                    sha256=sha256_file(history_path),
                    size_bytes=history_path.stat().st_size,
                )
                manifest.artifacts = [
                    item
                    for item in manifest.artifacts
                    if item.relative_path != history_record.relative_path
                ]
                manifest.add_artifact(history_record)
            failure_paths = sorted(run_dir.glob("RUN_FAILED*.json"))
            if failure_paths:
                archive_root = run_dir / "failures" / timestamp
                archive_root.mkdir(parents=True, exist_ok=False)
                for failure_path in failure_paths:
                    archived = archive_root / failure_path.name
                    failure_path.replace(archived)
                    manifest.add_artifact(
                        ArtifactRecord(
                            role="failed_training_attempt",
                            relative_path=archived.relative_to(run_dir).as_posix(),
                            sha256=sha256_file(archived),
                            size_bytes=archived.stat().st_size,
                        )
                    )
        else:
            atomic_write_text(selected_path, selected_payload)
            atomic_write_json(resolved_path, resolved_payload)
            atomic_write_json(prepared_snapshot_path, prepared_payload)
            manifest = RunManifest.create(
                run_id=run_id,
                experiment_id=resolved.id,
                scientific_id=scientific_id,
                method_revision=resolved.method_revision,
                repository_root=Path.cwd(),
                source_configs=[str(path) for path in resolved.source_files],
                dataset_revisions={
                    source.id: source.revision
                    for source in (resolved.data.annotation, *resolved.data.image_sources)
                },
                checkpoint_inputs=checkpoint_inputs,
            )
            manifest.host["torch"] = torch.__version__
            manifest.host["cuda_runtime"] = str(torch.version.cuda)
            manifest.host["world_size"] = os.environ.get("WORLD_SIZE", "1")
            if torch.cuda.is_available():
                manifest.host["visible_cuda_devices"] = json.dumps(
                    [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ]
                )
            manifest.add_artifact(
                ArtifactRecord(
                    role="training_sample_ids",
                    relative_path=selected_path.relative_to(run_dir).as_posix(),
                    sha256=sha256_file(selected_path),
                    size_bytes=selected_path.stat().st_size,
                )
            )
            manifest.add_artifact(
                ArtifactRecord(
                    role="resolved_configuration",
                    relative_path=resolved_path.relative_to(run_dir).as_posix(),
                    sha256=sha256_file(resolved_path),
                    size_bytes=resolved_path.stat().st_size,
                )
            )
            manifest.add_artifact(
                ArtifactRecord(
                    role="prepared_dataset_manifest",
                    relative_path=prepared_snapshot_path.relative_to(run_dir).as_posix(),
                    sha256=sha256_file(prepared_snapshot_path),
                    size_bytes=prepared_snapshot_path.stat().st_size,
                )
            )
            manifest.notes.append(f"base load report: {report}")
            manifest.notes.append(f"execution_runtime={resolved.runtime.id}")
            manifest.notes.append(
                "numerical_policy=" + json.dumps(numerical_policy, sort_keys=True)
            )
            manifest.notes.append(f"network_downloads_authorized={downloads_authorized}")
            manifest.notes.append(
                f"tokenization_preflight_samples={len(tokenization_audit_indices)}"
            )
            manifest.notes.append(
                f"kernel_optimization={json.dumps(kernel_report.to_dict(), sort_keys=True)}"
            )
            effective_batch = (
                resolved.training.per_device_batch_size
                * resolved.training.gradient_accumulation_steps
                * resolved.runtime.num_processes
            )
            manifest.notes.append(f"effective_global_batch={effective_batch}")
            manifest.notes.append(f"execution_run_root={run_root}")
        atomic_write_json(masking_report_path, vicuna_masking_report)
        masking_record = ArtifactRecord(
            role="vicuna_v1_masking_report",
            relative_path=masking_report_path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(masking_report_path),
            size_bytes=masking_report_path.stat().st_size,
        )
        manifest.artifacts = [
            item
            for item in manifest.artifacts
            if item.relative_path != masking_record.relative_path
        ]
        manifest.add_artifact(masking_record)
        masking_note = f"vicuna_v1_fully_masked_samples={len(vicuna_masking_issues)}"
        if masking_note not in manifest.notes:
            manifest.notes.append(masking_note)
        manifest.write(run_dir / "manifest.json")
    if engine.accelerator.num_processes > 1:
        engine.accelerator.wait_for_everyone()
    try:
        state = engine.run(resume_from=args.resume_from)
    except BaseException as error:
        process_index = engine.accelerator.process_index
        failure_path = run_dir / (
            "RUN_FAILED.json"
            if engine.accelerator.is_main_process
            else f"RUN_FAILED.rank-{process_index:05d}.json"
        )
        checkpoint_root = run_dir / "checkpoints"
        failure_checkpoints = [
            path.name
            for path in sorted(checkpoint_root.glob("step-*"))
            if path.is_dir() and (path / "COMPLETE").is_file()
        ]
        atomic_write_json(
            failure_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "experiment_id": resolved.id,
                "scientific_id": scientific_id,
                "run_id": run_id,
                "runtime_id": resolved.runtime.id,
                "process_index": process_index,
                "world_size": engine.accelerator.num_processes,
                "complete_pre_failure_checkpoints": failure_checkpoints,
                "recovery_rule": "resume only from the latest complete pre-failure checkpoint",
            },
        )
        raise
    if engine.accelerator.is_main_process:
        inventory_path = run_dir / "checkpoint_inventory.json"
        atomic_write_json(inventory_path, checkpoint_inventory(run_dir / "checkpoints"))
        completion_path = run_dir / "RUN_COMPLETE"
        completion_payload = "complete\n"
        manifest = RunManifest.read(run_dir / "manifest.json")
        refreshed = {
            "checkpoint_inventory.json": ArtifactRecord(
                role="checkpoint_inventory",
                relative_path="checkpoint_inventory.json",
                sha256=sha256_file(inventory_path),
                size_bytes=inventory_path.stat().st_size,
            ),
            "RUN_COMPLETE": ArtifactRecord(
                role="run_complete_marker",
                relative_path="RUN_COMPLETE",
                sha256=hashlib.sha256(completion_payload.encode()).hexdigest(),
                size_bytes=len(completion_payload.encode()),
            ),
        }
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.is_file():
            refreshed["metrics.jsonl"] = ArtifactRecord(
                role="training_metrics",
                relative_path="metrics.jsonl",
                sha256=sha256_file(metrics_path),
                size_bytes=metrics_path.stat().st_size,
            )
        for diagnostic_name, role in (
            ("tracking.json", "tracking_metadata"),
            ("hardware.jsonl", "hardware_telemetry"),
        ):
            diagnostic_path = run_dir / diagnostic_name
            if diagnostic_path.is_file():
                refreshed[diagnostic_name] = ArtifactRecord(
                    role=role,
                    relative_path=diagnostic_name,
                    sha256=sha256_file(diagnostic_path),
                    size_bytes=diagnostic_path.stat().st_size,
                )
        for orphaned_metrics in sorted(run_dir.glob("metrics.orphaned-*.jsonl")):
            refreshed[orphaned_metrics.name] = ArtifactRecord(
                role="orphaned_training_metrics",
                relative_path=orphaned_metrics.name,
                sha256=sha256_file(orphaned_metrics),
                size_bytes=orphaned_metrics.stat().st_size,
            )
        manifest.artifacts = [
            record for record in manifest.artifacts if record.relative_path not in refreshed
        ] + list(refreshed.values())
        manifest.notes = [note for note in manifest.notes if not note.startswith("final state: ")]
        manifest.notes.append(f"final state: {state.to_dict()}")
        manifest.write(run_dir / "manifest.json")
        atomic_write_text(completion_path, completion_payload)
    engine.accelerator.wait_for_everyone()
    _json(state.to_dict())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invllava",
        description=(
            "Reproducible data, training, evaluation, and analysis workflows for "
            "Inverse-LLaVA. Commands are fail-closed around downloads and mutable artifacts."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    config = subparsers.add_parser("config", help="resolve and identify scientific recipes")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    resolve = config_sub.add_parser("resolve")
    resolve.add_argument("experiment")
    resolve.add_argument("--config-root", default="configs")
    resolve.add_argument("--runtime-ref")
    resolve.add_argument("--output")
    resolve.set_defaults(function=command_config_resolve)

    reproduction = subparsers.add_parser(
        "reproduction", help="audit code, environment, and evidence readiness"
    )
    reproduction_sub = reproduction.add_subparsers(dest="reproduction_command", required=True)
    audit_reproduction = reproduction_sub.add_parser(
        "audit", help="separate code, environment, and scientific readiness"
    )
    audit_reproduction.add_argument("manifest", nargs="?", default="configs/reproduction/full.yaml")
    audit_reproduction.add_argument(
        "--stage", choices=("code", "environment", "scientific"), default="code"
    )
    audit_reproduction.add_argument("--repository-root")
    audit_reproduction.add_argument("--output")
    audit_reproduction.add_argument("--strict", action="store_true")
    audit_reproduction.set_defaults(function=command_reproduction_audit)

    data = subparsers.add_parser("data", help="acquire, audit, normalize, and arrange data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    plan = data_sub.add_parser("plan", help="print a no-download acquisition plan")
    plan.add_argument("data_config")
    plan.add_argument("--destination", required=True)
    plan.set_defaults(function=command_data_plan)
    normalize = data_sub.add_parser(
        "normalize", help="write audited conversations and a bound manifest"
    )
    normalize.add_argument("annotation")
    normalize.add_argument("--image-root", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--data-id", required=True)
    normalize.add_argument("--source-revision", required=True)
    normalize.add_argument("--default-source")
    normalize.add_argument(
        "--source",
        action="append",
        help="include one semantic annotation source; repeat for a controlled source subset",
    )
    normalize.add_argument("--manifest")
    normalize.add_argument("--image-audit")
    normalize.add_argument("--allow-unverified-images", action="store_true")
    normalize.set_defaults(function=command_data_normalize)
    audit_images = data_sub.add_parser(
        "audit-images", help="fully decode and inventory referenced images"
    )
    audit_images.add_argument("annotation")
    audit_images.add_argument("--image-root", required=True)
    audit_images.add_argument("--source-revision", required=True)
    audit_images.add_argument("--default-source")
    audit_images.add_argument(
        "--input-format",
        choices=("llava", "eval-jsonl"),
        default="llava",
    )
    audit_images.add_argument("--source", action="append")
    audit_images.add_argument("--workers", type=int, default=8)
    audit_images.add_argument("--maximum-failure-details", type=int, default=100)
    audit_images.add_argument("--progress-every", type=int, default=10000)
    audit_images.add_argument(
        "--inventory-output",
        help="optional JSONL content inventory written during the decode pass",
    )
    audit_images.add_argument("--output", required=True)
    audit_images.set_defaults(function=command_data_audit_images)
    compare_inventories = data_sub.add_parser(
        "compare-image-inventories",
        help="report exact encoded-image overlap between two audit inventories",
    )
    compare_inventories.add_argument("--left", required=True)
    compare_inventories.add_argument("--right", required=True)
    compare_inventories.add_argument("--maximum-examples", type=int, default=100)
    compare_inventories.add_argument("--require-disjoint", action="store_true")
    compare_inventories.add_argument("--output", required=True)
    compare_inventories.set_defaults(function=command_data_compare_image_inventories)
    audit_annotation = data_sub.add_parser(
        "audit-annotation", help="stream structural and identity checks over annotations"
    )
    audit_annotation.add_argument("annotation")
    audit_annotation.add_argument("--image-root", required=True)
    audit_annotation.add_argument("--source-revision", required=True)
    audit_annotation.add_argument("--default-source")
    audit_annotation.add_argument("--expected-sha256")
    audit_annotation.add_argument("--expected-samples", type=int)
    audit_annotation.add_argument("--require-unique-ids", action="store_true")
    audit_annotation.add_argument("--output", required=True)
    audit_annotation.set_defaults(function=command_data_audit_annotation)
    fetch = data_sub.add_parser("fetch", help="fetch only sources with frozen provenance")
    fetch.add_argument("data_config")
    fetch.add_argument("--destination", required=True)
    fetch.add_argument("--source-id", action="append")
    fetch.add_argument("--minimum-free-gib", type=int, default=10)
    fetch.add_argument("--allow-download", action="store_true")
    fetch.set_defaults(function=command_data_fetch)
    acquire_candidate = data_sub.add_parser(
        "acquire-candidate",
        help="download an unfrozen HTTPS artifact and record its observed digest",
    )
    acquire_candidate.add_argument("url")
    acquire_candidate.add_argument("--destination", required=True)
    acquire_candidate.add_argument("--minimum-free-gib", type=int, default=10)
    acquire_candidate.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="permit a provider's HTTP-only endpoint; freeze and verify the observed digest",
    )
    acquire_candidate.add_argument("--allow-download", action="store_true")
    acquire_candidate.set_defaults(function=command_data_acquire_candidate)
    fetch_hub = subparsers.add_parser(
        "fetch-hub-snapshot", help="resolve, resume, hash, and seal a Hub repository snapshot"
    )
    fetch_hub.add_argument("repo_id")
    fetch_hub.add_argument("--revision", required=True)
    fetch_hub.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    fetch_hub.add_argument("--destination", required=True)
    fetch_hub.add_argument("--allow-pattern", action="append")
    fetch_hub.add_argument("--minimum-free-gib", type=int, default=50)
    fetch_hub.add_argument("--allow-download", action="store_true")
    fetch_hub.set_defaults(function=command_fetch_hub_snapshot)
    extract = data_sub.add_parser("extract", help="verify and atomically extract an archive")
    extract.add_argument("archive")
    extract.add_argument("--destination", required=True)
    extract.add_argument("--sha256", required=True)
    extract.add_argument("--minimum-free-gib", type=int, default=10)
    extract.set_defaults(function=command_data_extract)
    assemble = data_sub.add_parser(
        "assemble-layout",
        help="atomically assemble verified component trees with same-filesystem hardlinks",
    )
    assemble.add_argument("layout")
    assemble.add_argument("--component-root", required=True)
    assemble.add_argument("--destination", required=True)
    assemble.add_argument("--manifest", required=True)
    assemble.set_defaults(function=command_data_assemble_layout)

    fetch_eval = subparsers.add_parser(
        "fetch-eval", help="download and materialize a pinned local evaluation split"
    )
    fetch_eval.add_argument("benchmark")
    fetch_eval.add_argument("--destination", required=True)
    fetch_eval.add_argument("--cache-dir", required=True)
    fetch_eval.add_argument("--allow-download", action="store_true")
    fetch_eval.set_defaults(function=command_fetch_eval)
    sample_eval = subparsers.add_parser(
        "sample-eval", help="materialize a deterministic self-contained development subset"
    )
    sample_eval.add_argument("--examples", required=True)
    sample_eval.add_argument("--destination", required=True)
    sample_eval.add_argument("--maximum", type=int, required=True)
    sample_eval.add_argument("--seed", type=int, default=2026)
    sample_eval.add_argument(
        "--stratify-metadata",
        help="round-robin across values of one required example metadata field",
    )
    sample_eval.add_argument(
        "--preserve-groups",
        action="store_true",
        help="select complete EvaluationExample group_id units without exceeding --maximum",
    )
    sample_eval.set_defaults(function=command_sample_eval)

    train = subparsers.add_parser("train", help="run or exactly resume a resolved experiment")
    train.add_argument("experiment")
    train.add_argument("--prepared-jsonl", required=True)
    train.add_argument("--prepared-manifest", required=True)
    train.add_argument("--config-root", default="configs")
    train.add_argument("--runtime-ref")
    train.add_argument(
        "--microbatch-size",
        type=int,
        help=(
            "execution microbatch per process; accumulation is derived to preserve "
            "the experiment's declared global optimizer-update batch"
        ),
    )
    train.add_argument(
        "--gradient-checkpointing",
        choices=("on", "off"),
        help="recorded execution override used for hardware qualification",
    )
    train.add_argument(
        "--maximum-samples",
        type=int,
        help="bounded execution subset; the resolved value is part of the scientific identity",
    )
    train.add_argument("--run-id")
    train.add_argument("--run-root")
    train.add_argument("--resume-from")
    train.add_argument(
        "--resume-source-change-reason",
        help="required audit note when an exact resume uses a different execution source",
    )
    train.add_argument("--initial-checkpoint")
    train.add_argument("--projector-checkpoint")
    train.add_argument("--allow-download", action="store_true")
    train.set_defaults(function=command_train)

    convert_projector = subparsers.add_parser(
        "convert-projector", help="safely convert the controlled LLaVA projector"
    )
    convert_projector.add_argument("source")
    convert_projector.add_argument("--output", required=True)
    convert_projector.set_defaults(function=command_convert_projector)
    convert_official_lora = subparsers.add_parser(
        "convert-official-llava-lora",
        help="convert the official LLaVA-1.5 LoRA release into a safe native delta",
    )
    convert_official_lora.add_argument("--adapter-model", required=True)
    convert_official_lora.add_argument("--adapter-config", required=True)
    convert_official_lora.add_argument("--non-lora-trainables", required=True)
    convert_official_lora.add_argument("--output", required=True)
    convert_official_lora.set_defaults(function=command_convert_official_llava_lora)
    convert_champion = subparsers.add_parser(
        "convert-champion-checkpoint",
        help="convert and seal the author checkpoint for native and Hub loading",
    )
    convert_champion.add_argument("source")
    convert_champion.add_argument("--model-config", required=True)
    convert_champion.add_argument("--model-card")
    convert_champion.add_argument("--output", required=True)
    convert_champion.add_argument("--allow-download", action="store_true")
    convert_champion.set_defaults(function=command_convert_champion_checkpoint)
    seal_checkpoint = subparsers.add_parser(
        "seal-training-checkpoint",
        help="export a checkpoint using its verified training run or a compatible parent release",
    )
    seal_checkpoint.add_argument("checkpoint")
    seal_context = seal_checkpoint.add_mutually_exclusive_group(required=True)
    seal_context.add_argument("--parent-release")
    seal_context.add_argument("--run-dir")
    seal_checkpoint.add_argument("--output", required=True)
    seal_checkpoint.set_defaults(function=command_seal_training_checkpoint)
    audit_fft = subparsers.add_parser(
        "audit-llava-fft-conversion",
        help="compare official FFT language/projector tensors with an HF conversion",
    )
    audit_fft.add_argument("--official", required=True)
    audit_fft.add_argument("--converted", required=True)
    audit_fft.add_argument("--output", required=True)
    audit_fft.set_defaults(function=command_audit_llava_fft_conversion)
    checkpoint_inventory = subparsers.add_parser(
        "checkpoint-inventory",
        help="count indexed safetensors by component without loading model weights",
    )
    checkpoint_inventory.add_argument("checkpoint")
    checkpoint_inventory.add_argument("--output", required=True)
    checkpoint_inventory.set_defaults(function=command_checkpoint_inventory)

    capture_representations = subparsers.add_parser(
        "capture-representations", help="capture aligned hidden and fusion features"
    )
    capture_representations.add_argument("experiment", nargs="?")
    capture_representations.add_argument(
        "--backend", choices=("native", "release", "hf-llava", "hf-causal"), default="native"
    )
    capture_representations.add_argument("--checkpoint")
    capture_representations.add_argument("--model")
    capture_representations.add_argument("--revision")
    capture_representations.add_argument("--examples", required=True)
    capture_representations.add_argument("--config-root", default="configs")
    capture_representations.add_argument("--cache-dir")
    capture_representations.add_argument("--runtime-ref")
    capture_representations.add_argument("--device", default="cuda")
    capture_representations.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    capture_representations.add_argument(
        "--attention-backend", choices=("sdpa", "eager"), default="sdpa"
    )
    capture_representations.add_argument(
        "--hf-image-aspect-ratio",
        choices=("pad", "square"),
        default="pad",
        help="preprocessing policy for the isolated Hugging Face LLaVA reference",
    )
    capture_representations.add_argument("--batch-size", type=int, default=4)
    capture_representations.add_argument("--maximum", type=int)
    capture_representations.add_argument("--output", required=True)
    capture_representations.add_argument("--metadata")
    capture_representations.add_argument("--allow-download", action="store_true")
    capture_representations.set_defaults(function=command_capture_representations)

    compare_representations = subparsers.add_parser(
        "compare-representations", help="compute CKA, RSA, neighborhood, and rank statistics"
    )
    compare_representations.add_argument("--left", required=True)
    compare_representations.add_argument("--left-key", required=True)
    compare_representations.add_argument("--right", required=True)
    compare_representations.add_argument("--right-key", required=True)
    compare_representations.add_argument("--k", type=int, default=10)
    compare_representations.add_argument("--matched-margin", action="store_true")
    compare_representations.add_argument("--paired-numerics", action="store_true")
    compare_representations.add_argument("--matched-permutations", type=int, default=0)
    compare_representations.add_argument("--confidence", type=float, default=0.95)
    compare_representations.add_argument("--seed", type=int, default=2026)
    compare_representations.add_argument("--output", required=True)
    compare_representations.set_defaults(function=command_compare_representations)

    plot_pca = subparsers.add_parser(
        "plot-representation-pca", help="plot a joint PCA without refitting series separately"
    )
    plot_pca.add_argument(
        "--series",
        nargs=3,
        action="append",
        metavar=("LABEL", "ARTIFACT", "KEY"),
        required=True,
    )
    plot_pca.add_argument("--output", required=True)
    plot_pca.add_argument("--metadata")
    plot_pca.set_defaults(function=command_plot_representation_pca)

    plot_cka = subparsers.add_parser(
        "plot-representation-cka", help="plot layerwise representation similarity"
    )
    plot_cka.add_argument("--left", required=True)
    plot_cka.add_argument("--right", required=True)
    plot_cka.add_argument("--left-prefix", default="hidden.")
    plot_cka.add_argument("--right-prefix", default="hidden.")
    plot_cka.add_argument("--output", required=True)
    plot_cka.add_argument("--metadata")
    plot_cka.set_defaults(function=command_plot_representation_cka)

    plot_training = subparsers.add_parser(
        "plot-training-curves", help="plot unsmoothed immutable metric histories"
    )
    plot_training.add_argument(
        "--series",
        nargs=2,
        action="append",
        metavar=("LABEL", "METRICS_JSONL"),
        required=True,
    )
    plot_training.add_argument("--output", required=True)
    plot_training.add_argument("--metadata")
    plot_training.add_argument("--view", choices=("diagnostics", "loss"), default="diagnostics")
    plot_training.add_argument(
        "--require-matched-exposure",
        action="store_true",
        help="require identical steps, sample/token counters, targets, and learning-rate histories",
    )
    plot_training.set_defaults(function=command_plot_training_curves)

    plot_breakdown = subparsers.add_parser(
        "plot-score-breakdown", help="plot an aligned category breakdown from score artifacts"
    )
    plot_breakdown.add_argument(
        "--series",
        nargs=2,
        action="append",
        metavar=("LABEL", "SCORE_JSON"),
        required=True,
    )
    plot_breakdown.add_argument("--detail-key", default="category_accuracy")
    plot_breakdown.add_argument("--title")
    plot_breakdown.add_argument("--ylabel", default="Accuracy")
    plot_breakdown.add_argument("--output", required=True)
    plot_breakdown.add_argument("--metadata")
    plot_breakdown.set_defaults(function=command_plot_score_breakdown)

    plot_profiles = subparsers.add_parser(
        "plot-profile-comparison", help="plot comparable sealed inference profiles"
    )
    plot_profiles.add_argument(
        "--profile",
        nargs=2,
        action="append",
        metavar=("LABEL", "PROFILE_JSON"),
        required=True,
    )
    plot_profiles.add_argument("--output", required=True)
    plot_profiles.add_argument("--metadata")
    plot_profiles.set_defaults(function=command_plot_profile_comparison)

    paired_interval = subparsers.add_parser(
        "paired-interval", help="bootstrap a paired benchmark-score difference"
    )
    paired_interval.add_argument("--left-score", required=True)
    paired_interval.add_argument("--right-score", required=True)
    paired_interval.add_argument("--examples")
    paired_interval.add_argument(
        "--image-audit", help="checksum-bound RGB-pixel audit for clustering repeated images"
    )
    paired_interval.add_argument("--resamples", type=int, default=10000)
    paired_interval.add_argument("--confidence", type=float, default=0.95)
    paired_interval.add_argument("--seed", type=int, default=2026)
    paired_interval.add_argument("--output", required=True)
    paired_interval.set_defaults(function=command_paired_interval)

    score_strata = subparsers.add_parser(
        "stratify-scores", help="analyze paired scores within numeric metadata strata"
    )
    score_strata.add_argument(
        "--series",
        nargs=2,
        action="append",
        metavar=("LABEL", "SCORE_JSON"),
        required=True,
    )
    score_strata.add_argument("--examples", required=True)
    score_strata.add_argument(
        "--image-audit", help="Use verified decoded-image groups for resampling"
    )
    score_strata.add_argument("--metadata-key", required=True)
    score_strata.add_argument("--boundaries", nargs="+", type=float, required=True)
    score_strata.add_argument("--primary", required=True)
    score_strata.add_argument("--resamples", type=int, default=10000)
    score_strata.add_argument("--confidence", type=float, default=0.95)
    score_strata.add_argument("--seed", type=int, default=2026)
    score_strata.add_argument("--output", required=True)
    score_strata.set_defaults(function=command_stratify_scores)

    language_interval = subparsers.add_parser(
        "language-interval", help="bootstrap paired lm-eval sample differences"
    )
    language_interval.add_argument("--left-result", required=True)
    language_interval.add_argument("--right-result", required=True)
    language_interval.add_argument("--metric", required=True)
    language_interval.add_argument("--filter", required=True)
    language_interval.add_argument("--resamples", type=int, default=10000)
    language_interval.add_argument("--confidence", type=float, default=0.95)
    language_interval.add_argument("--seed", type=int, default=2026)
    language_interval.add_argument("--output", required=True)
    language_interval.set_defaults(function=command_language_interval)

    select_cases = subparsers.add_parser(
        "select-cases", help="select deterministic multi-model successes and failures"
    )
    select_cases.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("LABEL", "PREDICTIONS", "SCORE"),
        required=True,
    )
    select_cases.add_argument("--primary", required=True)
    select_cases.add_argument("--per-group", type=int, default=4)
    select_cases.add_argument("--seed", type=int, default=2026)
    select_cases.add_argument("--correct-threshold", type=float, default=1.0)
    select_cases.add_argument("--output", required=True)
    select_cases.set_defaults(function=command_select_cases)

    export_cases = subparsers.add_parser(
        "export-cases", help="export auditable qualitative cases for author annotation"
    )
    export_cases.add_argument("--selection", required=True)
    export_cases.add_argument("--examples", required=True)
    export_cases.add_argument(
        "--prediction",
        nargs=2,
        action="append",
        metavar=("LABEL", "PREDICTIONS"),
        required=True,
    )
    export_cases.add_argument("--output", required=True)
    export_cases.add_argument("--markdown")
    export_cases.add_argument(
        "--image-root",
        help="copy selected images into a new content-addressed directory",
    )
    export_cases.add_argument(
        "--examples-output",
        help="write selected examples against --image-root for interventions",
    )
    export_cases.set_defaults(function=command_export_cases)

    prepare_interventions = subparsers.add_parser(
        "prepare-interventions", help="build shuffled-image or blank-image controls"
    )
    prepare_interventions.add_argument("--examples", required=True)
    prepare_interventions.add_argument("--mode", choices=("shuffled", "blank"), required=True)
    prepare_interventions.add_argument("--seed", type=int, default=2026)
    prepare_interventions.add_argument("--blank-rgb", type=int, nargs=3, default=(127, 127, 127))
    prepare_interventions.add_argument("--image-root")
    prepare_interventions.add_argument("--output", required=True)
    prepare_interventions.add_argument("--manifest")
    prepare_interventions.set_defaults(function=command_prepare_interventions)

    intervention_summary = subparsers.add_parser(
        "intervention-summary", help="summarize paired intervention effects"
    )
    intervention_summary.add_argument("--original-score", required=True)
    intervention_summary.add_argument("--intervened-score", required=True)
    intervention_summary.add_argument("--original-predictions")
    intervention_summary.add_argument("--intervened-predictions")
    intervention_summary.add_argument("--output", required=True)
    intervention_summary.set_defaults(function=command_intervention_summary)

    profile_native = subparsers.add_parser(
        "profile-native", help="measure preparation and cached native decoding"
    )
    profile_native.add_argument("experiment")
    profile_native.add_argument("--checkpoint", required=True)
    profile_native.add_argument("--examples", required=True)
    profile_native.add_argument("--config-root", default="configs")
    profile_native.add_argument("--runtime-ref")
    profile_native.add_argument("--device", default="cuda")
    profile_native.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 4))
    profile_native.add_argument("--decode-tokens", type=int, default=32)
    profile_native.add_argument("--warmups", type=int, default=10)
    profile_native.add_argument("--repetitions", type=int, default=30)
    profile_native.add_argument("--include-preparation", action="store_true")
    profile_native.add_argument("--output", required=True)
    profile_native.add_argument("--allow-download", action="store_true")
    profile_native.set_defaults(function=command_profile_native)
    profile_hf_llava = subparsers.add_parser(
        "profile-hf-llava",
        help="measure preparation and fixed cached decoding for an audited HF LLaVA reference",
    )
    profile_hf_llava.add_argument("--checkpoint", required=True)
    profile_hf_llava.add_argument("--revision", required=True)
    profile_hf_llava.add_argument("--examples", required=True)
    profile_hf_llava.add_argument("--device", default="cuda")
    profile_hf_llava.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    profile_hf_llava.add_argument("--attention-backend", default="sdpa")
    profile_hf_llava.add_argument("--image-aspect-ratio", choices=("pad", "square"), default="pad")
    profile_hf_llava.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 4))
    profile_hf_llava.add_argument("--decode-tokens", type=int, default=32)
    profile_hf_llava.add_argument("--warmups", type=int, default=10)
    profile_hf_llava.add_argument("--repetitions", type=int, default=30)
    profile_hf_llava.add_argument("--include-preparation", action="store_true")
    profile_hf_llava.add_argument("--output", required=True)
    profile_hf_llava.set_defaults(function=command_profile_hf_llava)

    kernel_audit = subparsers.add_parser(
        "kernel-audit",
        help="no-download numerical-parity and warm-speed qualification",
    )
    kernel_audit.add_argument("runtime_config")
    kernel_audit.add_argument("--device", default="cuda")
    kernel_audit.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    kernel_audit.add_argument("--seed", type=int, default=2026)
    kernel_audit.add_argument("--batch-size", type=int, default=2)
    kernel_audit.add_argument("--sequence-length", type=int, default=32)
    kernel_audit.add_argument("--hidden-size", type=int, default=64)
    kernel_audit.add_argument("--visual-size", type=int, default=32)
    kernel_audit.add_argument("--layers", type=int, default=2)
    kernel_audit.add_argument("--warmups", type=int, default=5)
    kernel_audit.add_argument("--repetitions", type=int, default=15)
    kernel_audit.add_argument("--atol", type=float)
    kernel_audit.add_argument("--rtol", type=float)
    kernel_audit.add_argument("--trace")
    kernel_audit.add_argument("--output", required=True)
    kernel_audit.set_defaults(function=command_kernel_audit)

    predict = subparsers.add_parser(
        "predict", help="write resumable, checkpoint-bound benchmark predictions"
    )
    predict.add_argument("benchmark")
    predict.add_argument("--examples", required=True)
    predict.add_argument(
        "--backend", choices=("native", "release", "hf-llava", "hf-multimodal"), required=True
    )
    predict.add_argument("--experiment")
    predict.add_argument("--checkpoint")
    predict.add_argument("--model")
    predict.add_argument("--revision")
    predict.add_argument("--config-root", default="configs")
    predict.add_argument("--cache-dir")
    predict.add_argument("--runtime-ref")
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    predict.add_argument("--attention-backend", choices=("sdpa", "eager"))
    predict.add_argument(
        "--hf-image-aspect-ratio",
        choices=("pad", "square"),
        default="pad",
        help="preprocessing policy for the isolated Hugging Face LLaVA reference",
    )
    predict.add_argument("--generation-cache", choices=("kv", "full_recompute"))
    predict.add_argument(
        "--lora-execution",
        choices=("unmerged", "merged"),
        help="release-only adapter execution; defaults to numerically faithful unmerged LoRA",
    )
    predict.add_argument("--batch-size", type=int, default=1)
    predict.add_argument("--output", required=True)
    predict.add_argument("--manifest")
    predict.add_argument("--allow-download", action="store_true")
    predict.add_argument("--allow-unverified", action="store_true")
    predict.set_defaults(function=command_predict)

    score = subparsers.add_parser("score", help="score complete predictions locally")
    score.add_argument("benchmark")
    score.add_argument("--predictions", required=True)
    score.add_argument("--examples", required=True)
    score.add_argument("--protocol-id")
    score.add_argument("--checkpoint-id")
    score.add_argument("--output")
    score.add_argument("--judge-dir", help="import an official hosted MM-Vet grading receipt")
    score.add_argument("--submission", help="exact answer JSON uploaded for these judge grades")
    score.add_argument("--allow-unverified", action="store_true")
    score.set_defaults(function=command_score)

    package = subparsers.add_parser(
        "package-submission",
        aliases=["package-vqav2"],
        help="package verified VQAv2 or MM-Vet predictions for external scoring",
    )
    package.add_argument("benchmark")
    package.add_argument("--predictions", required=True)
    package.add_argument("--examples", required=True)
    package.add_argument("--evaluation-manifest")
    package.add_argument("--full-test-questions", help="official VQAv2 full-test upload envelope")
    package.add_argument(
        "--checkpoint-snapshot", help="HF reference snapshot at the recorded evaluation path"
    )
    package.add_argument("--output", required=True)
    package.add_argument("--output-manifest", required=True)
    package.add_argument("--allow-unverified", action="store_true")
    package.set_defaults(function=command_package_submission)

    prepare_eval = subparsers.add_parser(
        "prepare-eval", help="convert and audit a manual evaluation release"
    )
    prepare_eval.add_argument("benchmark", help="frozen configs/benchmark/*.yaml protocol")
    prepare_eval.add_argument("--annotations", required=True)
    prepare_eval.add_argument("--image-root")
    prepare_eval.add_argument("--coco-split")
    prepare_eval.add_argument("--output", required=True)
    prepare_eval.add_argument("--manifest")
    prepare_eval.add_argument("--image-audit")
    prepare_eval.add_argument("--image-audit-workers", type=int, default=8)
    prepare_eval.add_argument("--image-audit-progress-every", type=int, default=10000)
    prepare_eval.add_argument("--allow-unverified-images", action="store_true")
    prepare_eval.set_defaults(function=command_prepare_eval)

    complexity = subparsers.add_parser(
        "complexity", help="calculate connector parameter and MAC counts"
    )
    complexity.add_argument("--hidden-size", type=int, default=4096)
    complexity.add_argument("--visual-size", type=int, default=1024)
    complexity.add_argument(
        "--visual-input-size",
        type=int,
        help="native encoder width; defaults to --visual-size (no learned reducer)",
    )
    complexity.add_argument("--sequence-length", type=int, default=2048)
    complexity.add_argument("--patches", type=int, default=576)
    complexity.add_argument("--fusion-layers", type=int, default=1)
    complexity.add_argument("--operator", choices=("concat", "add", "gated"), default="concat")
    complexity.set_defaults(function=command_complexity)

    language_eval = subparsers.add_parser(
        "language-eval", help="run the frozen text-only retention suite"
    )
    language_eval.add_argument("suite")
    language_eval.add_argument(
        "--model-kind",
        choices=("native", "release", "hf-causal", "hf-llava"),
        required=True,
    )
    language_eval.add_argument("--experiment")
    language_eval.add_argument("--checkpoint")
    language_eval.add_argument("--model")
    language_eval.add_argument("--revision")
    language_eval.add_argument("--cache-dir")
    language_eval.add_argument("--config-root", default="configs")
    language_eval.add_argument("--runtime-ref")
    language_eval.add_argument("--device", default="cuda")
    language_eval.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    language_eval.add_argument("--batch-size", type=int, default=4)
    language_eval.add_argument("--max-length", type=int)
    language_eval.add_argument(
        "--limit",
        type=int,
        help="canary-only maximum documents per task; omit for a full result",
    )
    language_eval.add_argument(
        "--task",
        action="append",
        help="configured task to run; repeat to select more than one",
    )
    language_eval.add_argument("--output", required=True)
    language_eval.add_argument("--allow-download", action="store_true")
    language_eval.set_defaults(function=command_language_eval)

    cleanup = subparsers.add_parser(
        "cleanup", help="review or delete only manifest-owned artifact files"
    )
    cleanup.add_argument("--manifest", required=True)
    cleanup.add_argument("--root", required=True)
    cleanup.add_argument("--confirm", action="store_true")
    cleanup.set_defaults(function=command_cleanup)
    verify_run = subparsers.add_parser(
        "verify-run", help="verify a training run before or after durable transfer"
    )
    verify_run.add_argument("run_dir")
    verify_run.add_argument("--output")
    verify_run.add_argument("--allow-incomplete", action="store_true")
    verify_run.set_defaults(function=command_verify_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
