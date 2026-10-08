from types import SimpleNamespace

import torch
from PIL import Image

from invllava.eval.interop.hf_llava import HuggingFaceLLaVAGenerator
from invllava.eval.types import GenerationRequest


def _generator(aspect_ratio: str) -> HuggingFaceLLaVAGenerator:
    generator = object.__new__(HuggingFaceLLaVAGenerator)
    generator.image_aspect_ratio = aspect_ratio
    generator.processor = SimpleNamespace(
        image_processor=SimpleNamespace(image_mean=(0.5, 0.25, 0.0))
    )
    return generator


def test_hf_llava_reference_pads_rectangular_images_with_clip_mean() -> None:
    image = Image.new("RGB", (4, 2), (1, 2, 3))

    prepared = _generator("pad")._prepare_image(image)

    assert prepared.size == (4, 4)
    assert prepared.getpixel((0, 0)) == (127, 63, 0)
    assert prepared.getpixel((0, 1)) == (1, 2, 3)


def test_hf_llava_reference_keeps_square_processor_policy_explicit() -> None:
    image = Image.new("RGB", (4, 2), (1, 2, 3))

    prepared = _generator("square")._prepare_image(image)

    assert prepared.size == (4, 2)


def test_hf_llava_reference_batches_one_image_requests(tmp_path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2), (index, 0, 0)).save(path)
        paths.append(path)

    class Processor:
        image_processor = SimpleNamespace(image_mean=(0.5, 0.5, 0.5))

        def __call__(self, *, text, images, padding, return_tensors):
            assert text == ["prompt 0", "prompt 1"]
            assert len(images) == 2
            assert padding is True
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[1, 2, 3], [0, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1], [0, 1, 1]]),
            }

        def batch_decode(self, rows, *, skip_special_tokens):
            assert skip_special_tokens is True
            return [" first ", "second"]

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def generate(self, *, input_ids, attention_mask, do_sample, max_new_tokens):
            assert attention_mask.shape == input_ids.shape
            assert do_sample is False
            assert max_new_tokens == 4
            suffix = torch.tensor([[4, 5], [6, 7]], device=input_ids.device)
            return torch.cat((input_ids, suffix), dim=1)

    generator = object.__new__(HuggingFaceLLaVAGenerator)
    generator.image_aspect_ratio = "pad"
    generator.processor = Processor()
    generator.model = Model()
    generator.max_new_tokens = 4
    requests = [
        GenerationRequest(str(index), f"prompt {index}", (path,))
        for index, path in enumerate(paths)
    ]

    assert generator.generate_many(requests) == ["first", "second"]
