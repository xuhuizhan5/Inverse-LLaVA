import torch
from torch import nn

from invllava.model.adaptation import (
    LoRALinear,
    apply_lora,
    merge_lora_for_inference,
    set_lora_trainable,
)


def test_lora_inherits_wrapped_linear_dtype_and_device() -> None:
    base = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    adapter = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
    assert adapter.lora_a.weight.dtype == base.weight.dtype
    assert adapter.lora_b.weight.dtype == base.weight.dtype
    assert adapter.lora_a.weight.device == base.weight.device
    output = adapter(torch.randn(2, 4, dtype=torch.float64))
    assert output.dtype == torch.float64


def test_lora_policy_excludes_only_named_fusion_layer() -> None:
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)

    model = nn.Module()
    model.layers = nn.ModuleList([Layer(), Layer()])
    selected = apply_lora(
        model,
        target_suffixes=("q_proj",),
        rank=2,
        alpha=4,
        dropout=0.0,
        excluded_layer_indices=(0,),
    )
    assert selected == ("layers.1.q_proj",)
    assert isinstance(model.layers[0].q_proj, nn.Linear)
    assert isinstance(model.layers[1].q_proj, LoRALinear)


def test_merge_lora_for_inference_preserves_eval_output() -> None:
    torch.manual_seed(9)
    model = nn.Sequential(LoRALinear(nn.Linear(4, 3), rank=2, alpha=4, dropout=0.25)).eval()
    model[0].lora_b.weight.data.normal_()
    value = torch.randn(5, 4)
    expected = model(value)
    assert merge_lora_for_inference(model) == ("0",)
    assert isinstance(model[0], nn.Linear)
    torch.testing.assert_close(model(value), expected)


def test_lora_weights_can_be_loaded_and_frozen_for_mechanism_control() -> None:
    adapter = LoRALinear(nn.Linear(4, 3), rank=2, alpha=4, dropout=0.0)
    model = nn.Sequential(adapter)

    assert set_lora_trainable(model, False) == ("0",)
    assert not adapter.base.weight.requires_grad
    assert not adapter.lora_a.weight.requires_grad
    assert not adapter.lora_b.weight.requires_grad
