from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLaVADataIdentity:
    id: str
    annotation_name: str
    purpose: str


LLAVA_MIX665K = LLaVADataIdentity(
    id="llava-v1.5-mix665k",
    annotation_name="llava_v1_5_mix665k.json",
    purpose="canonical multimodal instruction training",
)

LLAVA_PRETRAIN558K = LLaVADataIdentity(
    id="llava-pretrain-558k",
    annotation_name="blip_laion_cc_sbu_558k.json",
    purpose="dose-controlled paired-supervision continuation",
)
