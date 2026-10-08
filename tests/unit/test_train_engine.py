import torch

from invllava.train.engine import _all_processes_finite, _encode_batch_images


class RejectingEncoder:
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise AssertionError("text-only batches must not call the vision encoder")


class TwoRankReducer:
    device = torch.device("cpu")
    num_processes = 2

    def __init__(self, *, peer_is_finite: bool) -> None:
        self.peer_is_finite = peer_is_finite

    def reduce(self, value: torch.Tensor, *, reduction: str) -> torch.Tensor:
        assert reduction == "sum"
        return value + int(self.peer_is_finite)


def test_text_only_batch_bypasses_vision_encoder() -> None:
    result = _encode_batch_images(RejectingEncoder(), [[], []], torch.device("cpu"))
    assert result == [[], []]


def test_finite_gate_requires_every_rank() -> None:
    all_finite, count = _all_processes_finite(
        TwoRankReducer(peer_is_finite=True),
        torch.tensor(1.0),
    )
    assert all_finite is True
    assert count == 2

    peer_failed, count = _all_processes_finite(
        TwoRankReducer(peer_is_finite=False),
        torch.tensor(1.0),
    )
    assert peer_failed is False
    assert count == 1

    local_failed, count = _all_processes_finite(
        TwoRankReducer(peer_is_finite=True),
        torch.tensor(float("nan")),
    )
    assert local_failed is False
    assert count == 1
