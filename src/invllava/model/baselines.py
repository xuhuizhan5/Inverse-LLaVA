from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineIdentity:
    id: str
    tuning: str
    provenance: str
    controlled: bool


OFFICIAL_LLAVA15_FFT = BaselineIdentity(
    id="llava-1.5-7b-fft-official",
    tuning="full fine-tuning",
    provenance="published LLaVA-1.5 reference (e.g. MME 1510.7)",
    controlled=False,
)

LOCAL_LLAVA15_LORA = BaselineIdentity(
    id="llava-1.5-7b-lora-local",
    tuning="LoRA",
    provenance="locally re-evaluated controlled checkpoint (e.g. MME perception 1477)",
    controlled=True,
)


def assert_distinct_baselines(left: BaselineIdentity, right: BaselineIdentity) -> None:
    if left.id == right.id:
        raise ValueError("official FFT and controlled LoRA baselines must never share an identity")
