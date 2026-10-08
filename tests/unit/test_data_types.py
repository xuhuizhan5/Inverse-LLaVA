import json
from pathlib import Path

from PIL import Image, ImageFile

from invllava.data.audit import (
    ImageInventoryRecord,
    audit_image_integrity,
    audit_samples,
    canonical_rgb_sha256,
)
from invllava.data.collate import image_mean_background
from invllava.data.prepare import (
    iter_json_array,
    iter_llava_json,
    qualify_reused_sample_ids,
)
from invllava.data.types import ConversationSample, Turn


def test_clip_padding_background_matches_llava_integer_conversion() -> None:
    assert image_mean_background((0.48145466, 0.4578275, 0.40821073)) == (122, 116, 104)


def test_canonical_rgb_hash_ignores_container_encoding(tmp_path: Path) -> None:
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    png = tmp_path / "image.png"
    bmp = tmp_path / "image.bmp"
    image.save(png)
    image.save(bmp)
    with Image.open(png) as png_image, Image.open(bmp) as bmp_image:
        assert canonical_rgb_sha256(png_image) == canonical_rgb_sha256(bmp_image)


def test_conversation_validation_accepts_alternating_complete_dialogue() -> None:
    sample = ConversationSample(
        "one",
        (Path("image.png"),),
        (Turn("user", "<image> question"), Turn("assistant", "answer")),
        "fixture",
    )
    sample.validate()


def test_conversation_validation_rejects_non_alternating_roles() -> None:
    sample = ConversationSample(
        "one",
        (),
        (Turn("user", "first"), Turn("user", "second")),
        "fixture",
    )
    try:
        sample.validate()
    except ValueError as error:
        assert "turn 1" in str(error)
    else:
        raise AssertionError("non-alternating dialogue should fail")


def test_conversation_validation_rejects_missing_assistant_target() -> None:
    sample = ConversationSample("one", (), (Turn("user", "question"),), "fixture")
    try:
        sample.validate()
    except ValueError as error:
        assert "end with an assistant" in str(error)
    else:
        raise AssertionError("dialogue without a target should fail")


def test_streaming_json_array_parser_crosses_small_chunk_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "records.json"
    records = [{"id": index, "text": "value" * 7} for index in range(5)]
    source.write_text(json.dumps(records), encoding="utf-8")
    assert list(iter_json_array(source, chunk_size=11)) == records


def test_streaming_json_array_parser_handles_multibyte_text(tmp_path: Path) -> None:
    source = tmp_path / "records.json"
    records = [{"id": "视觉-1", "text": "café 图像"}, {"id": "视觉-2", "text": "✓"}]
    source.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    assert list(iter_json_array(source, chunk_size=5)) == records


def test_llava_parser_rejects_image_paths_outside_root(tmp_path: Path) -> None:
    annotation = tmp_path / "unsafe.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": "unsafe",
                    "image": "../outside.png",
                    "conversations": [
                        {"from": "human", "value": "<image>"},
                        {"from": "gpt", "value": "answer"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    try:
        list(iter_llava_json(annotation, tmp_path / "images"))
    except ValueError as error:
        assert "unsafe image path" in str(error)
    else:
        raise AssertionError("path traversal should fail")


def test_single_corpus_source_overrides_numeric_image_shards(tmp_path: Path) -> None:
    annotation = tmp_path / "single-corpus.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "image": "00453/image.jpg",
                    "conversations": [
                        {"from": "human", "value": "<image>"},
                        {"from": "gpt", "value": "answer"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    sample = next(
        iter_llava_json(
            annotation,
            tmp_path / "images",
            default_source="llava-pretrain-558k-paired",
        )
    )
    assert sample.source == "llava-pretrain-558k-paired"


def test_reused_source_ids_receive_stable_unique_internal_ids() -> None:
    first = ConversationSample(
        "shared",
        (),
        (Turn("user", "first"), Turn("assistant", "one")),
        "a",
    )
    second = ConversationSample(
        "shared",
        (),
        (Turn("user", "second"), Turn("assistant", "two")),
        "b",
    )
    qualified, evidence = qualify_reused_sample_ids([first, second])
    assert [sample.id for sample in qualified] == [
        "row-000000000:shared",
        "row-000000001:shared",
    ]
    assert [sample.turns for sample in qualified] == [first.turns, second.turns]
    assert [sample.images for sample in qualified] == [first.images, second.images]
    assert [sample.source for sample in qualified] == [first.source, second.source]
    assert evidence == {
        "policy": "row-index-prefix-v1",
        "source_duplicate_occurrences": 1,
        "source_duplicate_groups": 1,
    }


def test_unique_source_ids_are_preserved() -> None:
    samples = [
        ConversationSample(
            str(index),
            (),
            (Turn("user", "question"), Turn("assistant", "answer")),
            "fixture",
        )
        for index in range(2)
    ]
    qualified, evidence = qualify_reused_sample_ids(samples)
    assert qualified is samples
    assert evidence["policy"] == "preserve-source-id-v1"


def test_sample_audit_stats_each_unique_image_once_but_counts_references(
    monkeypatch,
) -> None:
    image = Path("shared-missing.png")
    samples = [
        ConversationSample(
            str(index),
            (image,),
            (Turn("user", "<image>"), Turn("assistant", "answer")),
            "fixture",
        )
        for index in range(2)
    ]
    calls = 0

    def missing(_path: Path) -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(Path, "is_file", missing)
    audit = audit_samples(samples)
    assert calls == 1
    assert audit.images == 2
    assert audit.missing_images == 2


def test_image_integrity_audit_fully_decodes_and_classifies_failures(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    (image_root / "coco").mkdir(parents=True)
    Image.new("RGB", (3, 2), (10, 20, 30)).save(image_root / "coco" / "valid.png")
    (image_root / "coco" / "corrupt.png").write_bytes(b"not an image")
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": "valid",
                    "image": "coco/valid.png",
                    "conversations": [
                        {"from": "human", "value": "<image>\nDescribe."},
                        {"from": "gpt", "value": "valid"},
                    ],
                },
                {
                    "id": "corrupt",
                    "image": "coco/corrupt.png",
                    "conversations": [
                        {"from": "human", "value": "<image>\nDescribe."},
                        {"from": "gpt", "value": "corrupt"},
                    ],
                },
                {
                    "id": "missing",
                    "image": "coco/missing.png",
                    "conversations": [
                        {"from": "human", "value": "<image>\nDescribe."},
                        {"from": "gpt", "value": "missing"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    audit = audit_image_integrity(
        iter_llava_json(annotation, image_root), sources={"coco"}, workers=2
    )
    assert audit.unique_images == 3
    assert audit.decoded_images == 1
    assert audit.missing_images == 1
    assert audit.corrupt_images == 1
    assert not audit.passed


def test_image_integrity_audit_fingerprints_decoded_inventory(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "valid.png"
    Image.new("RGB", (4, 5), (10, 20, 30)).save(image)
    sample = ConversationSample(
        "valid",
        (image,),
        (Turn("user", "<image>"), Turn("assistant", "answer")),
        "fixture",
    )
    records: list[ImageInventoryRecord] = []
    first = audit_image_integrity(
        [sample], image_root=image_root, workers=1, inventory=records.append
    )
    assert first.passed
    assert first.encoded_bytes == image.stat().st_size
    assert first.formats == {"PNG": 1}
    assert len(first.inventory_sha256) == 64
    assert records[0].logical_path == "valid.png"
    assert len(records[0].pixel_sha256) == 64
    Image.new("RGB", (4, 5), (30, 20, 10)).save(image)
    second = audit_image_integrity([sample], image_root=image_root, workers=1)
    assert second.inventory_sha256 != first.inventory_sha256


def test_image_integrity_audit_rejects_linked_files_and_directories(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "valid.png"
    Image.new("RGB", (4, 5), (10, 20, 30)).save(target)
    turns = (Turn("user", "<image>"), Turn("assistant", "answer"))

    linked_file = image_root / "linked.png"
    linked_file.symlink_to(target)
    file_audit = audit_image_integrity(
        [ConversationSample("file", (linked_file,), turns, "fixture")],
        image_root=image_root,
        workers=1,
    )
    assert file_audit.corrupt_images == 1
    assert file_audit.failure_details[0]["error"] == "symlink"

    linked_directory = image_root / "linked-directory"
    linked_directory.symlink_to(outside, target_is_directory=True)
    try:
        audit_image_integrity(
            [
                ConversationSample(
                    "directory",
                    (linked_directory / target.name,),
                    turns,
                    "fixture",
                )
            ],
            image_root=image_root,
            workers=1,
        )
    except ValueError as error:
        assert "linked directory" in str(error)
    else:
        raise AssertionError("linked image directories must be rejected")


def test_image_integrity_audit_overrides_permissive_truncated_image_mode(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "truncated.jpg"
    Image.new("RGB", (32, 32), (20, 40, 60)).save(image)
    payload = image.read_bytes()
    image.write_bytes(payload[: len(payload) // 2])
    sample = ConversationSample(
        "truncated",
        (image,),
        (Turn("user", "<image>"), Turn("assistant", "answer")),
        "fixture",
    )
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    audit = audit_image_integrity([sample], image_root=image_root, workers=1)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False
    assert audit.corrupt_images == 1
    assert not audit.passed
