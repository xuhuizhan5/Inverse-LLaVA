from __future__ import annotations

import pytest

from scripts.launch_train import (
    _apply_cuda_allocator_conf,
    _apply_execution_source_identity,
)


def test_allocator_policy_is_applied_before_worker_launch() -> None:
    environment = {"PATH": "/bin"}

    result = _apply_cuda_allocator_conf(
        environment,
        "garbage_collection_threshold:0.75,expandable_segments:True",
    )

    assert result["PYTORCH_ALLOC_CONF"] == (
        "garbage_collection_threshold:0.75,expandable_segments:True"
    )


def test_allocator_policy_rejects_an_unrecorded_override() -> None:
    with pytest.raises(ValueError, match="conflicts with the resolved runtime"):
        _apply_cuda_allocator_conf(
            {"PYTORCH_ALLOC_CONF": "expandable_segments:False"},
            "expandable_segments:True",
        )


def test_execution_source_identity_is_exported_to_workers() -> None:
    digest = "a" * 64

    result = _apply_execution_source_identity({"PATH": "/bin"}, digest)

    assert result["INVLLAVA_EXECUTION_SOURCE_SHA256"] == digest


def test_execution_source_identity_rejects_a_stale_override() -> None:
    with pytest.raises(ValueError, match="conflicts with the launcher source"):
        _apply_execution_source_identity(
            {"INVLLAVA_EXECUTION_SOURCE_SHA256": "b" * 64},
            "a" * 64,
        )
