from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from invllava.config.schema import DataSource, DataSpec


@dataclass(frozen=True)
class SourcePlan:
    id: str
    action: str
    source: str
    destination: Path
    revision: str | None
    sha256: str | None


def plan_sources(spec: DataSpec, destination_root: str | Path) -> tuple[SourcePlan, ...]:
    root = Path(destination_root)

    def one(source: DataSource) -> SourcePlan:
        action = {
            "huggingface": "download_huggingface",
            "http": "download_http",
            "manual": "verify_manual_presence",
            "generated": "verify_repository_fixture",
        }[source.kind]
        return SourcePlan(
            id=source.id,
            action=action,
            source=source.location,
            destination=root / spec.id / "sources" / source.id,
            revision=source.revision,
            sha256=source.sha256,
        )

    return tuple(one(source) for source in (spec.annotation, *spec.image_sources))
