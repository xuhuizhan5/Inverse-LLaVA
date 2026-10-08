"""Distinguish code validity, frozen environment, and scientific authorization."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

import yaml

from invllava.config.loader import ConfigRepository
from invllava.config.schema import BenchmarkSpec, DataSource, ReproductionSpec
from invllava.config.validation import require_frozen_execution

_SHA256 = re.compile(r"[0-9a-f]{64}")
_HUGGINGFACE_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMAGE_DIGEST = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")


class ReadinessStage(IntEnum):
    code = 1
    environment = 2
    scientific = 3


@dataclass(frozen=True)
class ReadinessCheck:
    id: str
    area: str
    status: Literal["ready", "blocked", "optional-pending"]
    detail: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessReport:
    reproduction_id: str
    requested_stage: str
    passed: bool
    checks: tuple[ReadinessCheck, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "reproduction_id": self.reproduction_id,
            "requested_stage": self.requested_stage,
            "passed": self.passed,
            "summary": {
                status: sum(check.status == status for check in self.checks)
                for status in ("ready", "blocked", "optional-pending")
            },
            "checks": [check.to_dict() for check in self.checks],
            "limitations": list(self.limitations),
        }


def _load_spec(path: Path) -> ReproductionSpec:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ReproductionSpec.model_validate(value)


def _resolve_reference(root: Path, reference: Path) -> Path:
    target = (root / reference).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"reproduction reference escapes repository: {reference}")
    return target


def _discover_repository_root(manifest_path: Path) -> Path:
    for candidate in manifest_path.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise ValueError(
        "could not discover repository root from the reproduction manifest; "
        "pass repository_root explicitly"
    )


def _source_issues(source: DataSource) -> list[str]:
    if not source.required or source.kind == "generated":
        return []
    issues: list[str] = []
    if source.kind == "huggingface":
        if not source.revision or _HUGGINGFACE_COMMIT.fullmatch(source.revision) is None:
            issues.append("immutable Hugging Face commit")
        if source.sha256 is not None and _SHA256.fullmatch(source.sha256) is None:
            issues.append("valid optional SHA-256")
    elif not source.sha256 or _SHA256.fullmatch(source.sha256) is None:
        issues.append("SHA-256")
    return issues


def _image_lock_check(identifier: str, path: Path) -> ReadinessCheck:
    if not path.is_file():
        return ReadinessCheck(
            identifier, "environment", "blocked", "lock file is absent", str(path)
        )
    value = path.read_text(encoding="utf-8").strip()
    if _IMAGE_DIGEST.fullmatch(value) is None:
        return ReadinessCheck(
            identifier,
            "environment",
            "blocked",
            "expected one immutable image reference: name@sha256:<64 lowercase hex>",
            str(path),
        )
    return ReadinessCheck(identifier, "environment", "ready", value, str(path))


def audit_reproduction(
    manifest: str | Path,
    *,
    stage: str = "code",
    repository_root: str | Path | None = None,
) -> ReadinessReport:
    """Audit source-controlled prerequisites; runtime artifacts remain separate evidence."""

    requested = ReadinessStage[stage]
    manifest_path = Path(manifest).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else _discover_repository_root(manifest_path)
    )
    spec = _load_spec(manifest_path)
    checks: list[ReadinessCheck] = []
    repository = ConfigRepository(root / "configs")

    for reference in spec.experiments:
        path = _resolve_reference(root, reference)
        identifier = f"experiment:{path.stem}"
        try:
            resolved = repository.resolve(path)
        except Exception as error:
            checks.append(
                ReadinessCheck(identifier, "configuration", "blocked", str(error), str(path))
            )
            continue
        checks.append(
            ReadinessCheck(
                identifier, "configuration", "ready", "schema and references resolve", str(path)
            )
        )
        if requested >= ReadinessStage.scientific:
            try:
                require_frozen_execution(resolved)
            except ValueError as error:
                checks.append(
                    ReadinessCheck(
                        identifier + ":provenance",
                        "scientific-provenance",
                        "blocked",
                        str(error),
                        str(path),
                    )
                )
            else:
                checks.append(
                    ReadinessCheck(
                        identifier + ":provenance",
                        "scientific-provenance",
                        "ready",
                        "all required model and data identities are frozen",
                        str(path),
                    )
                )

    benchmark_groups = (
        (spec.local_benchmarks, False, "local"),
        (spec.judge_benchmarks, False, "judge"),
        (spec.external_benchmarks, False, "external"),
        (spec.optional_benchmarks, True, "optional"),
    )
    for references, optional, mode in benchmark_groups:
        for reference in references:
            path = _resolve_reference(root, reference)
            identifier = f"benchmark:{path.stem}"
            try:
                benchmark = BenchmarkSpec.model_validate(
                    yaml.safe_load(path.read_text(encoding="utf-8"))
                )
            except Exception as error:
                status = "optional-pending" if optional else "blocked"
                checks.append(
                    ReadinessCheck(identifier, "benchmark", status, str(error), str(path))
                )
                continue
            checks.append(
                ReadinessCheck(
                    identifier,
                    "benchmark",
                    "ready",
                    f"{mode} protocol schema resolves",
                    str(path),
                )
            )
            if requested < ReadinessStage.scientific:
                continue
            issues = [
                f"{source.id}: {', '.join(source_issues)}"
                for source in (
                    benchmark.annotations,
                    benchmark.images,
                    *benchmark.protocol_sources,
                )
                if source is not None and (source_issues := _source_issues(source))
            ]
            if benchmark.verification_status != "golden_verified":
                issues.append(
                    "official-scorer status is "
                    f"{benchmark.verification_status}, not golden_verified"
                )
            status = "optional-pending" if optional and issues else "blocked" if issues else "ready"
            checks.append(
                ReadinessCheck(
                    identifier + ":provenance",
                    "benchmark-provenance",
                    status,
                    "; ".join(issues) if issues else "source and official-scorer golden are frozen",
                    str(path),
                )
            )

    if requested >= ReadinessStage.environment:
        dependency_lock = _resolve_reference(root, spec.environment.dependency_lock)
        checks.append(
            ReadinessCheck(
                "dependency-lock",
                "environment",
                "ready" if dependency_lock.is_file() else "blocked",
                "frozen dependency lock exists"
                if dependency_lock.is_file()
                else "uv.lock is absent",
                str(dependency_lock),
            )
        )
        checks.append(
            _image_lock_check(
                "base-image-lock",
                _resolve_reference(root, spec.environment.base_image_lock),
            )
        )
    if requested >= ReadinessStage.scientific:
        checks.append(
            _image_lock_check(
                "publication-image-lock",
                _resolve_reference(root, spec.environment.publication_image_lock),
            )
        )

    passed = not any(check.status == "blocked" for check in checks)
    return ReadinessReport(
        reproduction_id=spec.id,
        requested_stage=requested.name,
        passed=passed,
        checks=tuple(checks),
        limitations=(
            "This report validates source-controlled declarations only.",
            "Data image audits, checkpoint reloads, benchmark predictions, and profiler traces "
            "must be verified from their runtime manifests.",
        ),
    )
