from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimEvidence:
    claim_id: str
    claim: str
    required_artifacts: tuple[str, ...]
    status: str
    fallback_wording: str


def unresolved_claims(items: list[ClaimEvidence]) -> list[ClaimEvidence]:
    return [item for item in items if item.status != "supported"]
