from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from invllava.analysis.profiling import profile_autoregressive
from invllava.analysis.runtime import (
    _last_valid,
    _masked_mean,
    _text_only_prompt,
    write_representation_artifact,
)
from invllava.eval.runtime import _left_pad_token_rows
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.types import ExpandedSequence, FusionState


def test_masked_mean_excludes_padding() -> None:
    value = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    torch.testing.assert_close(_masked_mean(value, mask), torch.tensor([[2.0]]))


def test_last_valid_supports_left_and_right_padding() -> None:
    value = torch.tensor(
        [
            [[10.0], [1.0], [2.0]],
            [[3.0], [4.0], [20.0]],
        ]
    )
    mask = torch.tensor(
        [
            [False, True, True],
            [True, True, False],
        ]
    )

    torch.testing.assert_close(_last_valid(value, mask), torch.tensor([[2.0], [4.0]]))


def test_text_only_prompt_removes_only_image_placeholders() -> None:
    prompt = "USER: <image>\nWhich option is correct? ASSISTANT:"

    assert _text_only_prompt(prompt) == "USER: Which option is correct? ASSISTANT:"


def test_evaluation_token_rows_are_left_padded() -> None:
    input_ids, attention_mask = _left_pad_token_rows(
        ([1, 7], [1, 8, 9]),
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert torch.equal(input_ids, torch.tensor([[0, 1, 7], [1, 8, 9]]))
    assert torch.equal(
        attention_mask,
        torch.tensor([[False, True, True], [True, True, True]]),
    )


def test_representation_artifact_is_aligned_and_immutable() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "features.npz"
        metadata = root / "features.json"
        write_representation_artifact(
            artifact,
            metadata,
            sample_ids=("a", "b"),
            arrays={"hidden.0": np.ones((2, 3))},
            metadata={"checkpoint_id": "fixture"},
        )
        loaded = np.load(artifact)
        assert loaded["sample_ids"].tolist() == ["a", "b"]
        try:
            write_representation_artifact(
                artifact,
                metadata,
                sample_ids=("a", "b"),
                arrays={"hidden.0": np.ones((2, 3))},
                metadata={},
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("representation artifact must be immutable")


def test_autoregressive_profile_executes_fixed_cached_decode() -> None:
    architecture = LlamaArchitecture(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    language = InverseLlamaForCausalLM(architecture).eval()
    ids = torch.tensor([[1, 4, 5]])
    embeddings = language.model.embed_tokens(ids)
    mask = torch.ones_like(ids, dtype=torch.bool)
    expanded = ExpandedSequence(
        inputs_embeds=embeddings,
        attention_mask=mask,
        position_ids=torch.arange(3).unsqueeze(0),
        labels=None,
        fusion_state=FusionState(
            visual_features=torch.zeros((1, 3, 4)),
            text_mask=mask,
            vision_mask=torch.zeros_like(mask),
        ),
    )
    profile = profile_autoregressive(
        language,
        expanded,
        fixed_decode_tokens=2,
        warmups=0,
        repetitions=3,
    )
    assert profile.batch_size == 1
    assert profile.fixed_decode_tokens == 2
    assert profile.median_decode_tokens_per_second > 0
