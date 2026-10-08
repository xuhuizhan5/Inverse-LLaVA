from pathlib import Path

import pytest
from PIL import Image

from invllava.data.control_packages import materialize_control_package
from invllava.data.dataset import NormalizedConversationDataset
from invllava.data.manifest import PreparedDatasetManifest
from invllava.data.types import ConversationSample, Turn


def _sample(path: Path) -> ConversationSample:
    return ConversationSample(
        "sample-a",
        (path,),
        (Turn("user", "<image> question"), Turn("assistant", "answer")),
        "textvqa",
    )


def test_control_package_is_relocatable_verified_and_hard_linked(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 6), (10, 20, 30)).save(source)
    report = materialize_control_package(
        tmp_path / "controls",
        name="fixture",
        data_id="fixture",
        revision="test-v1",
        samples=[_sample(source)],
        parents={"input": "a" * 64},
        token_counts={"sample-a": 3},
        workers=1,
    )
    package = Path(report["path"])
    moved = tmp_path / "relocated"
    package.rename(moved)
    manifest = PreparedDatasetManifest.read(moved / "train.manifest.json")
    manifest.verify(
        moved / "train.jsonl", expected_data_id="fixture", expected_source_revision="test-v1"
    )
    sample = NormalizedConversationDataset(moved / "train.jsonl")[0]
    assert sample.images[0].stat().st_ino == source.stat().st_ino
    assert sample.images[0].is_relative_to(moved)
    assert manifest.image_integrity["passed"] is True
    assert report["supervised_tokens"] == 3


@pytest.mark.parametrize("name", ["../escape", "..", ".", "/absolute"])
def test_control_package_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        materialize_control_package(
            tmp_path,
            name=name,
            data_id="fixture",
            revision="test-v1",
            samples=[_sample(tmp_path / "unused.png")],
            parents={},
            token_counts={"sample-a": 3},
            workers=1,
        )


def test_control_package_rejects_corruption_without_publishing(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.png"
    source.write_bytes(b"not an image")
    root = tmp_path / "controls"
    with pytest.raises(ValueError, match="image integrity"):
        materialize_control_package(
            root,
            name="fixture",
            data_id="fixture",
            revision="test-v1",
            samples=[_sample(source)],
            parents={},
            token_counts={"sample-a": 3},
            workers=1,
        )
    assert not list(root.iterdir())
