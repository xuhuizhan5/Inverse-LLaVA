import json

import numpy as np
import pytest

from invllava.analysis.runtime import write_representation_artifact
from scripts.derive_representation_change import derive_change


@pytest.mark.parametrize("wrong_checkpoint", [False, True])
def test_change_checks_identity_and_subtracts_aligned_activations(tmp_path, wrong_checkpoint):
    original, control, output = [
        tmp_path / f"{name}.npz" for name in ("original", "control", "change")
    ]
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    for path, value, condition, checkpoint in (
        (original, values, "original", "fixture"),
        (control, values - 2, "control", "other" if wrong_checkpoint else "fixture"),
    ):
        write_representation_artifact(
            path,
            path.with_suffix(".json"),
            sample_ids=["a", "b", "c", "d"],
            arrays={"hidden.last.1": value},
            metadata={
                "checkpoint_id": checkpoint,
                "backend": "fixture",
                "batch_size": 1,
                "torch_version": "fixture",
                "examples_sha256": condition,
            },
        )
    manifest = tmp_path / "intervention.json"
    manifest.write_text(
        json.dumps(
            {"mode": "blank", "source_examples_sha256": "original", "examples_sha256": "control"}
        )
    )
    if wrong_checkpoint:
        with pytest.raises(ValueError, match="checkpoint_id"):
            derive_change(original, control, manifest, output)
        assert not output.exists()
    else:
        derive_change(original, control, manifest, output)
        with np.load(output) as artifact:
            assert np.array_equal(artifact["hidden.last.1"], np.full((4, 6), 2))
        metadata = json.loads(output.with_suffix(".json").read_text())
        assert metadata["input_condition"] == "activation change: original-minus-blank"
