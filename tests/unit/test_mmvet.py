import hashlib
import json
from pathlib import Path

import pytest
import yaml

from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import prepare_mmvet


def fixture(tmp_path, *, image="v1_0.png"):
    path = tmp_path / "mm-vet.json"
    path.write_text(
        json.dumps(
            {
                "v1_0": {
                    "imagename": image,
                    "question": "What is x?",
                    "answer": "-1<AND>-5",
                    "capability": ["ocr", "math"],
                }
            }
        )
    )
    values = yaml.safe_load(Path("configs/benchmark/mmvet.yaml").read_text())
    values["annotations"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, BenchmarkSpec.model_validate(values)


def test_mmvet_preserves_ids_prompt_and_compositional_reference(tmp_path):
    path, spec = fixture(tmp_path)
    rows = prepare_mmvet(path, tmp_path / "images", spec=spec)
    assert rows[0].id == "v1_0"
    assert rows[0].references == ("-1<AND>-5",)
    assert rows[0].group_id == "v1_0.png"
    assert "<image>\nWhat is x? ASSISTANT:" in rows[0].prompt
    assert "-1<AND>-5" not in rows[0].prompt


@pytest.mark.parametrize("name", ["../image.png", "/image.png", "folder/image.png", ".."])
def test_mmvet_rejects_escaping_image_names(tmp_path, name):
    path, spec = fixture(tmp_path, image=name)
    with pytest.raises(ValueError, match="unsafe"):
        prepare_mmvet(path, tmp_path / "images", spec=spec)


def test_mmvet_rejects_changed_annotation_bytes(tmp_path):
    path, spec = fixture(tmp_path)
    path.write_text("{}")
    with pytest.raises(ValueError):
        prepare_mmvet(path, tmp_path / "images", spec=spec)
