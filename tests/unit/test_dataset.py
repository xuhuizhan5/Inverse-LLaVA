import json
import shutil
import tempfile
from pathlib import Path

import pytest

from invllava.data.dataset import NormalizedConversationDataset
from invllava.data.prepare import write_normalized_jsonl
from invllava.data.types import ConversationSample, Turn


def test_dataset_indexes_metadata_and_reuses_a_process_local_stream() -> None:
    records = [
        {
            "id": "visual",
            "images": ["image.png"],
            "turns": [
                {"role": "user", "text": "<image> what is shown"},
                {"role": "assistant", "text": "a fixture"},
            ],
            "source": "fixture",
        },
        {
            "id": "text",
            "images": [],
            "turns": [
                {"role": "user", "text": "say hello"},
                {"role": "assistant", "text": "hello"},
            ],
            "source": "fixture",
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        dataset = NormalizedConversationDataset(path)
        assert dataset.ids == ["visual", "text"]
        assert dataset.vicuna_v1_masking_issues == []
        assert dataset.modality_lengths[0] > 0
        assert dataset.modality_lengths[1] < 0
        assert dataset[0].id == "visual"
        stream = dataset._stream
        assert dataset[1].id == "text"
        assert dataset._stream is stream


def test_dataset_indexes_legacy_masked_vicuna_samples() -> None:
    record = {
        "id": "masked",
        "images": [],
        "turns": [
            {"role": "user", "text": "ASSISTANT: embedded upstream role prefix"},
            {"role": "assistant", "text": "answer"},
        ],
        "source": "fixture",
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        dataset = NormalizedConversationDataset(path)

        assert dataset.vicuna_v1_masking_issues == [
            (0, "masked", "round 0 contains 2 serialized assistant delimiters")
        ]


def test_normalized_package_is_relocatable(tmp_path: Path) -> None:
    original = tmp_path / "original"
    images = original / "images"
    images.mkdir(parents=True)
    image = images / "fixture.bin"
    image.write_bytes(b"fixture")
    normalized = original / "train.jsonl"
    write_normalized_jsonl(
        [
            ConversationSample(
                id="visual",
                images=(image,),
                turns=(Turn("user", "<image> describe"), Turn("assistant", "fixture")),
                source="fixture",
            )
        ],
        normalized,
    )
    record = json.loads(normalized.read_text(encoding="utf-8"))
    assert record["images"] == ["images/fixture.bin"]

    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    dataset = NormalizedConversationDataset(relocated / "train.jsonl")
    assert dataset[0].images == ((relocated / "images/fixture.bin").resolve(),)


def test_normalized_package_rejects_external_images(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="inside the normalized dataset package root"):
        write_normalized_jsonl(
            [
                ConversationSample(
                    id="visual",
                    images=(outside,),
                    turns=(Turn("user", "<image> describe"), Turn("assistant", "fixture")),
                    source="fixture",
                )
            ],
            package / "train.jsonl",
        )
