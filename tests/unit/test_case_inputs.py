import numpy as np
import pytest
import torch
from PIL import Image

from scripts.audit_case_inputs import pixel_preview, processed_input


class Processor:
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.25, 0.25, 0.25)

    def __call__(self, images, return_tensors):
        assert images.mode == "RGB" and return_tensors == "pt"
        pixels = torch.from_numpy(np.array(images)).permute(2, 0, 1).float() / 255
        return {"pixel_values": ((pixels - 0.5) / 0.25).unsqueeze(0)}


def test_case_input_follows_integer_mean_padding_and_preserves_original():
    processor = Processor()
    with Image.new("L", (2, 4), 255) as original:
        pixels = processed_input(original, processor, pad=True)
        assert original.size == (2, 4) and original.getpixel((0, 0)) == 255
        assert pixels.shape == (3, 4, 4)
        with pixel_preview(pixels, processor) as preview:
            assert preview.getpixel((0, 0)) == (127, 127, 127)
            assert preview.getpixel((1, 0)) == (255, 255, 255)


def test_case_input_unpadded_and_nonfinite_rejection():
    with Image.new("RGB", (2, 4)) as original:
        assert processed_input(original, Processor(), pad=False).shape == (3, 4, 2)

        def invalid(**kwargs):
            return {"pixel_values": torch.full((1, 3, 2, 2), float("nan"))}

        with pytest.raises(ValueError, match="finite"):
            processed_input(original, invalid, pad=False)
