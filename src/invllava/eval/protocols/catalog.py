from __future__ import annotations

from invllava.config.schema import BenchmarkSpec
from invllava.eval.protocols.ai2d import AI2DProtocol
from invllava.eval.protocols.gqa import GQAProtocol
from invllava.eval.protocols.mmbench import MMBenchCircularProtocol
from invllava.eval.protocols.mme import MMEProtocol
from invllava.eval.protocols.mmstar import MMStarProtocol
from invllava.eval.protocols.multiple_choice import MultipleChoiceProtocol
from invllava.eval.protocols.ocrbench import OCRBenchProtocol
from invllava.eval.protocols.scienceqa import ScienceQAProtocol
from invllava.eval.protocols.vizwiz import VizWizProtocol
from invllava.eval.protocols.vqa import ExactMatchProtocol, TextVQAProtocol, VQAv2Protocol
from invllava.eval.types import ScoringProtocol


def build_protocol(spec: BenchmarkSpec) -> ScoringProtocol:
    if "circular" in spec.answer_extraction:
        if spec.id not in {"mmbench-en", "mmbench-cn"}:
            raise ValueError(f"no audited circular evaluator is registered for {spec.id}")
        return MMBenchCircularProtocol(spec.protocol_revision)
    if spec.scorer == "official_ocrbench":
        return OCRBenchProtocol()
    if spec.scorer == "official_mmstar_macro_l2":
        return MMStarProtocol(spec.protocol_revision)
    if spec.scorer == "official_ai2d_exact_match":
        return AI2DProtocol(spec.protocol_revision)
    if spec.scorer == "official_mathvista":
        raise ValueError(
            f"{spec.id} requires a golden-verified official extraction adapter before local scoring"
        )
    if spec.scorer == "gqa_llava_official_exact":
        return GQAProtocol(spec.protocol_revision)
    if spec.scorer == "llava_scienceqa_img_accuracy":
        return ScienceQAProtocol(spec.protocol_revision)
    if spec.task_type == "multiple_choice":
        return MultipleChoiceProtocol(spec.protocol_revision)
    if spec.task_type == "mme":
        if spec.scorer == "mme_perception_accuracy_plus":
            return MMEProtocol(spec.protocol_revision, domain="perception")
        if spec.scorer == "mme_cognition_accuracy_plus":
            return MMEProtocol(spec.protocol_revision, domain="cognition")
        raise ValueError(f"MME benchmark {spec.id} has unsupported scorer {spec.scorer}")
    if spec.scorer == "vqav2_consensus":
        return VQAv2Protocol(spec.protocol_revision)
    if spec.scorer == "textvqa_consensus":
        return TextVQAProtocol(spec.protocol_revision)
    if spec.scorer == "vizwiz_consensus":
        return VizWizProtocol(spec.protocol_revision)
    if spec.task_type in {"vqa", "exact_match"}:
        return ExactMatchProtocol(spec.protocol_revision)
    raise ValueError(
        f"protocol {spec.id} requires an official judge or external submission "
        "adapter, not local scoring"
    )
