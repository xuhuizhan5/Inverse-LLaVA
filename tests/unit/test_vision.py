from types import SimpleNamespace

import pytest
import torch
from torch import nn

from invllava.model.vision import VisionEncoder, _validate_clip_loading_info


@pytest.mark.parametrize("layer", [-1, -2])
def test_feature_layer_selects_patch_hidden_states_without_changing_width(layer):
    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, pixel_values, output_hidden_states):
            assert output_hidden_states
            states = tuple(torch.full((2, 577, 1024), float(i)) for i in range(3))
            return SimpleNamespace(hidden_states=states)

    actual = VisionEncoder(Tower(), feature_layers=(layer,))(torch.zeros(2, 3, 336, 336))
    assert actual.shape == (2, 576, 1024)
    assert torch.all(actual == (2 if layer == -1 else 1))


def test_frozen_vision_tower_stays_in_eval_mode() -> None:
    tower = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    encoder = VisionEncoder(tower, freeze=True)
    encoder.train()
    assert not encoder.tower.training
    assert not any(parameter.requires_grad for parameter in encoder.tower.parameters())


def test_trainable_vision_tower_follows_parent_mode() -> None:
    tower = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    encoder = VisionEncoder(tower, freeze=False)
    encoder.train()
    assert encoder.tower.training


def test_clip_loading_info_accepts_only_expected_text_tower_keys() -> None:
    _validate_clip_loading_info(
        {
            "missing_keys": [],
            "unexpected_keys": [
                "text_model.encoder.layers.0.self_attn.q_proj.weight",
                "text_projection.weight",
                "visual_projection.weight",
                "logit_scale",
            ],
            "mismatched_keys": [],
            "error_msgs": [],
        }
    )
    with pytest.raises(RuntimeError, match=r"rogue\.weight"):
        _validate_clip_loading_info(
            {
                "missing_keys": [],
                "unexpected_keys": ["rogue.weight"],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        )
