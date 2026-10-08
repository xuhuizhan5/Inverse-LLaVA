from __future__ import annotations

import json
import os
from pathlib import Path
from typing import BinaryIO

from torch.utils.data import Dataset

from invllava.data.types import ConversationSample, Turn
from invllava.prompting import vicuna_v1_masking_issue


class NormalizedConversationDataset(Dataset[ConversationSample]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.package_root = self.path.parent
        self._offsets: list[int] = []
        self.ids: list[str] = []
        self.sources: list[str] = []
        self.modality_lengths: list[int] = []
        self.vicuna_v1_masking_issues: list[tuple[int, str, str]] = []
        self._stream: BinaryIO | None = None
        self._stream_pid: int | None = None
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    row_index = len(self._offsets)
                    self._offsets.append(offset)
                    value = json.loads(line)
                    self.ids.append(str(value["id"]))
                    self.sources.append(str(value["source"]))
                    turns = tuple(
                        Turn(str(turn["role"]), str(turn["text"])) for turn in value["turns"]
                    )
                    masking_issue = vicuna_v1_masking_issue(turns)
                    if masking_issue is not None:
                        self.vicuna_v1_masking_issues.append(
                            (row_index, str(value["id"]), masking_issue)
                        )
                    word_count = max(
                        1,
                        sum(len(turn.text.split()) for turn in turns),
                    )
                    # The sign separates visual and text-only examples while the
                    # magnitude provides a cheap, tokenizer-independent padding
                    # proxy. Exact tokenization still happens in the collator.
                    self.modality_lengths.append(word_count if value.get("images") else -word_count)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> ConversationSample:
        stream = self._process_stream()
        stream.seek(self._offsets[index])
        value = json.loads(stream.readline())
        images: list[Path] = []
        for value_path in value["images"]:
            path = Path(value_path)
            if path.is_absolute():
                images.append(path)
                continue
            resolved = (self.package_root / path).resolve()
            if resolved == self.package_root or self.package_root not in resolved.parents:
                raise ValueError(f"sample {value['id']}: image escapes dataset package root")
            images.append(resolved)
        sample = ConversationSample(
            id=str(value["id"]),
            images=tuple(images),
            turns=tuple(Turn(str(turn["role"]), str(turn["text"])) for turn in value["turns"]),
            source=str(value["source"]),
        )
        sample.validate()
        return sample

    def _process_stream(self) -> BinaryIO:
        """Reuse one descriptor per DataLoader worker, never across processes."""

        pid = os.getpid()
        if self._stream is None or self._stream.closed or self._stream_pid != pid:
            if self._stream is not None and not self._stream.closed:
                self._stream.close()
            self._stream = self.path.open("rb")
            self._stream_pid = pid
        return self._stream

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_stream"] = None
        state["_stream_pid"] = None
        return state

    def __del__(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None and not stream.closed:
            stream.close()
