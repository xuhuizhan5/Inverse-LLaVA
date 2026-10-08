from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path, PurePosixPath

from invllava.data.types import ConversationSample, Turn

ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def iter_json_array(path: str | Path, *, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    """Incrementally parse a top-level JSON array without loading a GB-scale file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as stream:
        buffer = ""
        cursor = 0
        eof = False

        def read_more() -> None:
            nonlocal buffer, cursor, eof
            chunk = stream.read(chunk_size)
            buffer = buffer[cursor:] + chunk
            cursor = 0
            # TextIOWrapper.tell() is an opaque cookie, not a byte offset that
            # can be compared safely with fstat() for non-ASCII annotations.
            eof = chunk == ""

        def skip_whitespace() -> None:
            nonlocal cursor
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer) or eof:
                    return
                read_more()

        read_more()
        skip_whitespace()
        if cursor >= len(buffer) or buffer[cursor] != "[":
            raise ValueError("LLaVA annotation must be a JSON array")
        cursor += 1
        first = True
        while True:
            skip_whitespace()
            if cursor >= len(buffer):
                raise ValueError("unterminated JSON array")
            if buffer[cursor] == "]":
                cursor += 1
                skip_whitespace()
                if cursor != len(buffer):
                    raise ValueError("unexpected content after JSON array")
                return
            if not first:
                if buffer[cursor] != ",":
                    raise ValueError("JSON array values must be comma-separated")
                cursor += 1
                skip_whitespace()
                if cursor < len(buffer) and buffer[cursor] == "]":
                    raise ValueError("JSON array must not have a trailing comma")
            first = False
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError(f"invalid JSON array near offset {error.pos}") from error
                    read_more()
                else:
                    cursor = end
                    yield value
                    break


def normalize_llava_record(
    record: dict[str, object],
    image_root: str | Path,
    *,
    default_source: str | None = None,
) -> ConversationSample:
    sample_id = str(record["id"])
    image_value = record.get("image")
    if image_value is None:
        images = ()
    else:
        logical_image = PurePosixPath(str(image_value))
        if logical_image.is_absolute() or ".." in logical_image.parts or "\\" in str(image_value):
            raise ValueError(f"sample {sample_id}: unsafe image path {image_value!r}")
        images = (Path(image_root).joinpath(*logical_image.parts),)
    raw_turns = record.get("conversations")
    if not isinstance(raw_turns, list):
        raise ValueError(f"sample {sample_id}: conversations must be a list")
    turns: list[Turn] = []
    for raw in raw_turns:
        if not isinstance(raw, dict) or "from" not in raw or "value" not in raw:
            raise ValueError(f"sample {sample_id}: malformed conversation turn")
        role = ROLE_MAP.get(str(raw["from"]))
        if role is None:
            raise ValueError(f"sample {sample_id}: unsupported role {raw['from']}")
        turns.append(Turn(role, str(raw["value"])))
    if "source" in record:
        source = str(record["source"])
    elif default_source is not None:
        source = default_source
    elif image_value is not None and "/" in str(image_value):
        # This is an image-storage family, not a supervision-task label.
        # LLaVA's textvqa/ images include TextCaps caption conversations.
        source = str(image_value).split("/", 1)[0]
    else:
        source = "llava"
    sample = ConversationSample(sample_id, images, tuple(turns), source)
    sample.validate()
    return sample


def iter_llava_json(
    path: str | Path,
    image_root: str | Path,
    *,
    default_source: str | None = None,
) -> Iterator[ConversationSample]:
    for record in iter_json_array(path):
        if not isinstance(record, dict):
            raise ValueError("each LLaVA record must be an object")
        yield normalize_llava_record(record, image_root, default_source=default_source)


def qualify_reused_sample_ids(
    samples: list[ConversationSample],
) -> tuple[list[ConversationSample], dict[str, object]]:
    """Create stable internal IDs when an upstream mixture reuses source IDs.

    Official LLaVA mixtures contain valid rows from several corpora whose local
    ID namespaces overlap. Nothing is removed or reordered. If any ID is reused,
    every row receives a deterministic row prefix so checkpoint and sampling
    identities remain unambiguous without changing training content.
    """

    counts = Counter(sample.id for sample in samples)
    duplicate_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_groups = sum(count > 1 for count in counts.values())
    if duplicate_occurrences == 0:
        return samples, {
            "policy": "preserve-source-id-v1",
            "source_duplicate_occurrences": 0,
            "source_duplicate_groups": 0,
        }
    width = max(9, len(str(max(len(samples) - 1, 0))))
    qualified = [
        replace(sample, id=f"row-{index:0{width}d}:{sample.id}")
        for index, sample in enumerate(samples)
    ]
    return qualified, {
        "policy": "row-index-prefix-v1",
        "source_duplicate_occurrences": duplicate_occurrences,
        "source_duplicate_groups": duplicate_groups,
    }


def write_normalized_jsonl(samples: Iterable[ConversationSample], destination: str | Path) -> None:
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for sample in samples:
                sample.validate()
                if sample.id in seen:
                    raise ValueError(f"duplicate sample id: {sample.id}")
                seen.add(sample.id)
                image_paths: list[str] = []
                for path in sample.images:
                    # Inputs reached this point through the validated
                    # annotation and image-audit boundaries.  Lexical
                    # normalization avoids a remote filesystem lookup for
                    # every training row while retaining package containment.
                    resolved = Path(os.path.abspath(path))
                    try:
                        relative = resolved.relative_to(target.parent)
                    except ValueError as error:
                        raise ValueError(
                            f"sample {sample.id}: image must be inside the normalized "
                            f"dataset package root {target.parent}: {resolved}"
                        ) from error
                    image_paths.append(relative.as_posix())
                stream.write(
                    json.dumps(
                        {
                            "id": sample.id,
                            "images": image_paths,
                            "turns": [
                                {"role": turn.role, "text": turn.text} for turn in sample.turns
                            ],
                            "source": sample.source,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
