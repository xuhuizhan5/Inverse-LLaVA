"""Render the actual CLIP input for a frozen casebook without loading the LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.config.loader import ConfigRepository
from invllava.data.collate import expand_to_square, image_mean_background


def processed_input(image: Image.Image, processor, *, pad: bool) -> torch.Tensor:
    """Follow InverseGenerator's RGB, padding, and processor sequence exactly."""
    rgb = image.convert("RGB")
    prepared = expand_to_square(rgb, image_mean_background(processor.image_mean)) if pad else rgb
    try:
        pixels = processor(images=prepared, return_tensors="pt")["pixel_values"][0]
        if pixels.ndim != 3 or pixels.shape[0] != 3 or not torch.isfinite(pixels).all():
            raise ValueError("processor must produce a finite three-channel input")
        return pixels.detach().cpu().contiguous()
    finally:
        if prepared is not rgb:
            prepared.close()
        rgb.close()


def pixel_preview(pixels: torch.Tensor, processor) -> Image.Image:
    """Undo normalization for display only; retain the exact tensor hash separately."""
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    rgb = ((pixels.float() * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)
    return Image.fromarray(rgb.permute(1, 2, 0).numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    resolved = ConfigRepository(args.config_root).resolve(args.experiment)
    vision = resolved.model.vision
    if vision.aspect_ratio not in {"pad", "square"}:
        raise ValueError("only the declared single-image pad/square pipeline is supported")
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(
        vision.checkpoint, revision=vision.revision, local_files_only=True
    )
    casebook = json.loads(args.casebook.read_text())
    cases = casebook["cases"]
    ids = [str(case["sample_id"]) for case in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("casebook must contain unique, nonempty sample IDs")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    records = []
    for case in cases:
        if len(case["images"]) != 1:
            raise ValueError("case-input audit requires exactly one image per case")
        path = Path(case["images"][0])
        with Image.open(path) as original:
            size = original.size
            pixels = processed_input(original, processor, pad=vision.aspect_ratio == "pad")
        filename = hashlib.sha256(str(case["sample_id"]).encode()).hexdigest()[:16] + ".png"
        preview = args.output_directory / filename
        with pixel_preview(pixels, processor) as display:
            display.save(preview)
        records.append(
            {
                "sample_id": str(case["sample_id"]),
                "original_path": str(path),
                "original_sha256": sha256_file(path),
                "original_width_height": size,
                "processed_chw": list(pixels.shape),
                "processed_dtype": str(pixels.dtype),
                "processed_tensor_sha256": hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
                "preview": filename,
                "preview_sha256": sha256_file(preview),
            }
        )
    atomic_write_json(
        args.output_directory / "audit.json",
        {
            "status": "passed",
            "casebook_sha256": sha256_file(args.casebook),
            "vision": vision.model_dump(mode="json"),
            "processor_class": type(processor).__name__,
            "processor_configuration": processor.to_dict(),
            "cases": records,
            "scope": (
                "Input inspection only. Readability, error causes, and recovery "
                "require separate evidence."
            ),
        },
    )
    print(
        json.dumps(
            {"status": "passed", "cases": len(records), "output": str(args.output_directory)}
        )
    )


if __name__ == "__main__":
    main()
