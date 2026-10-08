from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultCell:
    value: float
    low: float | None = None
    high: float | None = None
    unit: str = "score"


@dataclass(frozen=True)
class ResultRow:
    model: str
    evidence_class: str
    checkpoint_id: str
    protocol_id: str
    cells: dict[str, ResultCell]
    provenance: str


def validate_comparison(rows: list[ResultRow]) -> None:
    controlled = [row for row in rows if row.evidence_class == "controlled_architecture"]
    if controlled:
        protocols = {row.protocol_id for row in controlled}
        if len(protocols) != 1:
            raise ValueError("controlled table mixes evaluation protocols")
    identities = {(row.model, row.checkpoint_id, row.protocol_id) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("duplicate result rows")
