from __future__ import annotations

from torch import Tensor, nn


class MLP2xGELUProjector(nn.Module):
    """The two-layer GELU projector used by LLaVA-1.5.

    Experiments record whether these weights come from a verified alignment
    checkpoint or an explicit random initialization for a one-stage control.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, visual_features: Tensor) -> Tensor:
        return self.linear_2(self.activation(self.linear_1(visual_features)))
