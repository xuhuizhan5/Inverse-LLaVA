from __future__ import annotations

import torch
from torch import nn

from invllava.train.engine import _encode_batch_images


class _ReferenceImagePath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.projector = nn.Linear(4, 8, bias=False, dtype=torch.float32)

    def encode_images(self, values: torch.Tensor) -> torch.Tensor:
        return self.projector(self.vision(values.to(torch.bfloat16)))


def test_reference_image_encoding_supports_fp32_update_parameters() -> None:
    model = _ReferenceImagePath()
    images = [[torch.randn(3, 4)], [torch.randn(3, 4)]]

    with torch.autocast("cpu", dtype=torch.bfloat16):
        features = _encode_batch_images(model, images, torch.device("cpu"))

    assert [len(row) for row in features] == [1, 1]
    assert all(item.dtype == torch.bfloat16 for row in features for item in row)
