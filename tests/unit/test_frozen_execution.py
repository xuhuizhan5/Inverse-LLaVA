import pytest

from invllava.config.loader import ConfigRepository
from invllava.config.schema import ProjectorSpec
from invllava.config.validation import require_frozen_execution


def configuration(projector=None):
    resolved = ConfigRepository("configs").resolve("configs/experiment/canonical_7b.yaml")
    model = resolved.model.model_copy(update={"projector": projector})
    return resolved.model_copy(update={"model": model})


def test_frozen_inverse_configuration_passes():
    require_frozen_execution(configuration())


def test_random_projector_needs_no_checkpoint_identity():
    projector = ProjectorSpec(input_dim=1024, output_dim=4096, initialization="random")
    require_frozen_execution(configuration(projector))


@pytest.mark.parametrize("identity", ["sha256:" + "a" * 64, "embedded-in-official-checkpoint"])
def test_checkpoint_projector_accepts_frozen_identity(identity):
    projector = ProjectorSpec(input_dim=1024, output_dim=4096, initial_checkpoint_id=identity)
    require_frozen_execution(configuration(projector))


@pytest.mark.parametrize("identity", ["pending-freeze", "sha256:abc", "main"])
def test_checkpoint_projector_rejects_unfrozen_identity(identity):
    projector = ProjectorSpec(input_dim=1024, output_dim=4096, initial_checkpoint_id=identity)
    with pytest.raises(ValueError, match=r"model\.projector\.initial_checkpoint_id"):
        require_frozen_execution(configuration(projector))


def test_random_projector_does_not_bypass_other_provenance_checks():
    projector = ProjectorSpec(input_dim=1024, output_dim=4096, initialization="random")
    value = configuration(projector)
    with pytest.raises(ValueError, match="initial_checkpoint_id"):
        require_frozen_execution(value.model_copy(update={"initial_checkpoint_id": "unfrozen"}))
