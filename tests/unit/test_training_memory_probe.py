"""No-download CPU checks for the full-length operational workload."""

import copy

import pytest
import torch

from invllava.model.adaptation import apply_lora
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.modeling import LLaVAReferenceForConditionalGeneration
from invllava.model.projector import MLP2xGELUProjector
from invllava.model.vision import VisionEncoder
from scripts.smoke_model_contract import TinyVisionTower, build_tiny_model
from scripts.verify_training_memory import FullLengthWorkload, verify_history


def tiny_projector_model():
    architecture = LlamaArchitecture(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    language = InverseLlamaForCausalLM(architecture)
    language.requires_grad_(False)
    apply_lora(language, target_suffixes=("q_proj",), rank=2, alpha=4, dropout=0)
    return LLaVAReferenceForConditionalGeneration(
        language,
        VisionEncoder(TinyVisionTower(8), feature_layers=(-1,), freeze=True),
        MLP2xGELUProjector(8, 16),
        hidden_size=16,
        max_length=32,
    )


@pytest.mark.parametrize("patches", [0, 4])
@pytest.mark.parametrize("padded", [False, True])
def test_projector_workload_preserves_counts_and_trainable_dependencies(patches, padded):
    workload = FullLengthWorkload(
        rows=4, length=32, patches=patches, image_size=2, dense_rows=2 if padded else None
    )
    batch = workload.collate([2, 3])
    model = tiny_projector_model()
    image_rows = batch.pop("pixel_values")
    batch.pop("sample_ids")
    features = [[model.encode_images(image.unsqueeze(0))[0] for image in row] for row in image_rows]
    output = model(**batch, image_features=features)
    assert output.expanded_token_count.item() == 64 - int(padded)
    assert output.supervised_token_count.item() == 2 * (31 - patches) - int(padded)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if patches:
        assert any(p.grad.abs().sum() > 0 for p in model.multimodal_projector.parameters())


@pytest.mark.parametrize("patches", [0, 4])
def test_workload_exercises_exact_expanded_length(patches):
    workload = FullLengthWorkload(rows=2, length=32, patches=patches, image_size=2)
    batch = workload.collate([workload[0], workload[1]])
    model = build_tiny_model(torch.device("cpu"))
    image_rows = batch.pop("pixel_values")
    batch.pop("sample_ids")
    features = [[model.encode_images(image.unsqueeze(0))[0] for image in row] for row in image_rows]
    output = model(**batch, image_features=features)
    assert output.expanded_token_count.item() == 64
    assert output.supervised_token_count.item() == 2 * (31 - patches)
    assert torch.isfinite(output.loss)
    output.loss.backward()


@pytest.mark.parametrize("values", [(0, 32, 4, 2), (2, 4, 4, 2), (2, 32, -1, 2), (2, 32, 4, 0)])
def test_invalid_workload_rejected(values):
    with pytest.raises(ValueError):
        FullLengthWorkload(
            rows=values[0], length=values[1], patches=values[2], image_size=values[3]
        )


def test_out_of_range_row_rejected():
    workload = FullLengthWorkload(rows=2, length=32, patches=4, image_size=2)
    with pytest.raises(IndexError):
        workload[2]


def test_fixed_ids_and_image_slots():
    workload = FullLengthWorkload(rows=2, length=32, patches=4, image_size=2)
    batch = workload.collate([0, 1])
    assert batch["sample_ids"] == ["synthetic-memory-0", "synthetic-memory-1"]
    assert batch["input_ids"][:, 1].tolist() == [-200, -200]
    assert batch["labels"][:, 1].tolist() == [-100, -100]
    assert all(len(row) == 1 for row in batch["pixel_values"])


@pytest.fixture
def history():
    return [
        {
            "step": step,
            "train/samples_seen": 128 * step,
            "train/tokens_seen": 128 * 2048 * step,
            "train/loss": 2.0,
            "train/gradient_norm": 1.0,
        }
        for step in (1, 2)
    ]


def test_complete_history_passes(history):
    verify_history(history, updates=2, global_batch=128, length=2048)


@pytest.mark.parametrize("patches", [0, 4])
def test_padded_workload_exercises_explicit_attention_mask(patches):
    workload = FullLengthWorkload(rows=4, length=32, patches=patches, image_size=2, dense_rows=2)
    assert workload.collate([0, 1])["attention_mask"].all()
    batch = workload.collate([2, 3])
    assert not batch["attention_mask"][-1, -1] and batch["labels"][-1, -1] == -100
    model = build_tiny_model(torch.device("cpu"))
    image_rows = batch.pop("pixel_values")
    batch.pop("sample_ids")
    features = [[model.encode_images(image.unsqueeze(0))[0] for image in row] for row in image_rows]
    output = model(**batch, image_features=features)
    assert output.expanded_token_count.item() == 63
    assert output.supervised_token_count.item() == 2 * (31 - patches) - 1
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_padded_history_counts_real_tokens_and_preserves_first_dense_update(history):
    history[-1]["train/tokens_seen"] -= 64
    verify_history(history, updates=2, global_batch=128, length=2048, padded_tokens_per_update=64)


def test_missing_padding_rejected(history):
    with pytest.raises(ValueError, match="expanded length"):
        verify_history(
            history, updates=2, global_batch=128, length=2048, padded_tokens_per_update=64
        )


@pytest.mark.parametrize("fault", ["step", "samples", "tokens", "loss", "gradient"])
def test_incomplete_or_nonfinite_history_rejected(history, fault):
    history = copy.deepcopy(history)
    keys = {
        "step": "step",
        "samples": "train/samples_seen",
        "tokens": "train/tokens_seen",
        "loss": "train/loss",
        "gradient": "train/gradient_norm",
    }
    history[-1][keys[fault]] = float("nan") if fault in ("loss", "gradient") else 1
    with pytest.raises(ValueError):
        verify_history(history, updates=2, global_batch=128, length=2048)
