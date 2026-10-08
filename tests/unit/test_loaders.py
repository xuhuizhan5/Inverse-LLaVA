import json
from pathlib import Path

import pytest
import torch

from invllava.model.loaders import (
    _checkpoint_keys,
    _compare_base_keys,
    _set_fusion_trainability,
    _validate_rope_contract,
)


@pytest.mark.parametrize(
    "trainable,mapper_trainable,scale_mode",
    [
        (True, True, "learned"),
        (True, True, "fixed"),
        (True, False, "learned"),
        (False, True, "learned"),
    ],
)
def test_fusion_trainability_preserves_component_exclusions(
    trainable, mapper_trainable, scale_mode
) -> None:
    from invllava.config.schema import FusionSpec
    from invllava.model.fusion import FusionBlock

    spec = FusionSpec(
        initialization_id="fixture",
        trainable=trainable,
        mapper_trainable=mapper_trainable,
        scale_mode=scale_mode,
    )
    fusion = FusionBlock(hidden_size=4, visual_size=2, target_sizes={"q": 4}, targets=("q",))
    _set_fusion_trainability(fusion, spec)
    assert fusion.scales["q"].requires_grad == (trainable and scale_mode == "learned")
    assert fusion.text_to_vision["q"].weight.requires_grad == (trainable and mapper_trainable)
    assert fusion.output["q"].weight.requires_grad == trainable
    if trainable:
        before = fusion.scales["q"].detach().clone()
        optimizer = torch.optim.SGD([p for p in fusion.parameters() if p.requires_grad], lr=0.1)
        sum(p.sum() for p in fusion.parameters() if p.requires_grad).backward()
        optimizer.step()
        assert torch.equal(fusion.scales["q"], before) == (scale_mode == "fixed")


def test_base_key_audit_is_strict_and_handles_tied_alias() -> None:
    expected = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
    missing, unexpected = _compare_base_keys(
        expected,
        {"model.embed_tokens.weight", "model.norm.weight"},
        tied_word_embeddings=True,
    )
    assert missing == ()
    assert unexpected == ()
    missing, unexpected = _compare_base_keys(
        expected,
        {"model.embed_tokens.weight", "foreign.weight"},
        tied_word_embeddings=False,
    )
    assert missing == ("lm_head.weight", "model.norm.weight")
    assert unexpected == ("foreign.weight",)


def test_official_sharded_pytorch_index_is_audited_without_unpickling(
    tmp_path: Path,
) -> None:
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "pytorch_model-00001-of-00002.bin",
                    "lm_head.weight": "pytorch_model-00002-of-00002.bin",
                }
            }
        ),
        encoding="utf-8",
    )
    keys, checkpoint_format = _checkpoint_keys(tmp_path)
    assert keys == {"model.embed_tokens.weight", "lm_head.weight"}
    assert checkpoint_format == "pytorch-bin-sharded-weights-only"


def test_rope_contract_accepts_only_unscaled_library_normalization() -> None:
    _validate_rope_contract(None, rope_theta=10000.0)
    _validate_rope_contract({"rope_type": "default", "rope_theta": 10000.0}, rope_theta=10000.0)
    for value in (
        {"rope_type": "linear", "rope_theta": 10000.0},
        {"rope_type": "default", "rope_theta": 10000.0, "factor": 2.0},
    ):
        try:
            _validate_rope_contract(value, rope_theta=10000.0)
        except ValueError as error:
            assert "scaled RoPE" in str(error)
        else:
            raise AssertionError("scaled RoPE metadata should be rejected")


@pytest.mark.parametrize("backend", ["torchvision", "pil"])
def test_shared_image_loader_passes_declared_backend(monkeypatch, backend):
    from types import SimpleNamespace

    import transformers

    from invllava.config.schema import VisionSpec
    from invllava.model.loaders import load_image_processor

    calls = []
    sentinel = SimpleNamespace(backend=backend)

    def load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return sentinel

    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", load)
    spec = VisionSpec(processor_backend=backend, revision="fixed")
    assert load_image_processor(spec, local_files_only=True, token=False) is sentinel
    assert calls == [
        (
            spec.checkpoint,
            {
                "revision": "fixed",
                "backend": backend,
                "local_files_only": True,
                "token": False,
            },
        )
    ]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("precision", ["model", "float32"])
def test_real_base_loader_respects_rotary_precision(tmp_path, monkeypatch, dtype, precision):
    import huggingface_hub
    import transformers

    from invllava.config.schema import (
        AdaptationSpec,
        FusionSpec,
        LanguageSpec,
        ModelSpec,
        VisionSpec,
    )
    from invllava.model.llama.configuration import LlamaArchitecture
    from invllava.model.llama.modeling import InverseLlamaForCausalLM
    from invllava.model.loaders import build_language_model

    config = transformers.LlamaConfig(
        vocab_size=67,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        pad_token_id=0,
    )
    reference = InverseLlamaForCausalLM(LlamaArchitecture.from_hf(config))
    torch.save(reference.state_dict(), tmp_path / "pytorch_model.bin")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(tmp_path))
    spec = ModelSpec(
        id="precision-fixture",
        architecture="inverse_llava",
        torch_dtype=dtype,
        language=LanguageSpec(
            checkpoint="fixture", hidden_size=128, num_layers=1, rotary_precision=precision
        ),
        vision=VisionSpec(),
        fusion=FusionSpec(initialization_id="none", layers=(), operator="disabled"),
        adaptation=AdaptationSpec(method="frozen"),
    )
    actual, report = build_language_model(spec, local_files_only=True)
    rotary = actual.model.layers[0].self_attn.rotary_emb
    expected_dtype = torch.float32 if precision == "float32" else getattr(torch, dtype)
    expected = reference.model.layers[0].self_attn.rotary_emb.inv_freq.to(expected_dtype)
    assert rotary.inv_freq.dtype == expected_dtype
    assert report.rotary_frequency_dtype == str(expected_dtype)
    assert torch.equal(rotary.inv_freq, expected)
    assert actual.model.layers[0].self_attn.q_proj.weight.dtype == getattr(torch, dtype)
    actual.to(dtype=getattr(torch, dtype))
    assert torch.equal(rotary.inv_freq, expected)


def test_image_loader_rejects_an_undeclared_fallback(monkeypatch):
    from types import SimpleNamespace

    import transformers

    from invllava.config.schema import VisionSpec
    from invllava.model.loaders import load_image_processor

    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(backend="pil"),
    )
    with pytest.raises(RuntimeError, match="declared backend"):
        load_image_processor(VisionSpec(processor_backend="torchvision"), local_files_only=True)
