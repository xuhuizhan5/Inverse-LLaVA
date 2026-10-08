from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from invllava.cli import build_parser
from invllava.eval.interop.hf_multimodal import (
    HuggingFaceMultimodalGenerator,
    native_messages,
    user_content,
)
from invllava.eval.types import GenerationRequest
from invllava.prompting import format_vicuna_v1_user_prompt


def test_native_reference_preserves_question_and_image_position_without_metadata():
    content = "Inspect this <image>\nA. cat\nB. dog\nAnswer with the option's letter."
    request = GenerationRequest(
        "1", format_vicuna_v1_user_prompt(content), (Path("1.png"),), {"answer": "SECRET"}
    )
    image = Image.new("RGB", (2, 2))
    result = native_messages(request, image)
    assert user_content(request.prompt) == content
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this "},
                {"type": "image", "image": image},
                {"type": "text", "text": "\nA. cat\nB. dog\nAnswer with the option's letter."},
            ],
        }
    ]
    image.close()


@pytest.mark.parametrize(
    "content", ["<image>", "<image><image> Question", "<image> Q ASSISTANT: A"]
)
def test_native_reference_rejects_ambiguous_content(content):
    with pytest.raises(ValueError):
        user_content(format_vicuna_v1_user_prompt(content))


def test_native_reference_rejects_unknown_wrapper():
    with pytest.raises(ValueError, match="single-turn"):
        user_content("Question with <image>")


def test_native_reference_uses_processor_and_decodes_only_generated_suffix(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (7, 3)).save(image_path)
    requests = [
        GenerationRequest(str(i), format_vicuna_v1_user_prompt("<image>\nQuestion"), (image_path,))
        for i in range(2)
    ]

    class Processor:
        def apply_chat_template(self, conversations, **kwargs):
            assert len(conversations) == 2
            assert conversations[0][0]["content"][0]["image"].size == (7, 3)
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
                "processor_kwargs": {"padding": True},
            }
            return {"input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]])}

        def batch_decode(self, suffix, **kwargs):
            assert suffix.tolist() == [[9], [9]]
            assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
            return ["A", "B"]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def generate(self, input_ids, **kwargs):
            assert kwargs == {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 7,
                "use_cache": True,
            }
            return torch.cat([input_ids, torch.full((2, 1), 9)], dim=1)

    generator = HuggingFaceMultimodalGenerator.__new__(HuggingFaceMultimodalGenerator)
    generator.processor, generator.model, generator.max_new_tokens = Processor(), Model(), 7
    assert generator.generate_many(requests) == ["A", "B"]
    with pytest.raises(ValueError):
        generator.generate_many([])


def test_native_reference_requires_immutable_revision_before_download():
    with pytest.raises(ValueError, match="commit SHA"):
        HuggingFaceMultimodalGenerator("unused/model", revision="main", local_files_only=True)


def test_predict_cli_exposes_optional_native_multimodal_backend():
    args = build_parser().parse_args(
        [
            "predict",
            "configs/benchmark/ocrbench.yaml",
            "--backend",
            "hf-multimodal",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--revision",
            "a" * 40,
            "--examples",
            "examples.jsonl",
            "--output",
            "predictions.jsonl",
        ]
    )
    assert args.backend == "hf-multimodal"
