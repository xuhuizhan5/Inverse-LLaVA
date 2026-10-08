import stat

from invllava.artifacts.atomic import atomic_write_bytes, atomic_write_json


def test_atomic_artifacts_are_readable_outside_the_creating_container(tmp_path) -> None:
    binary = tmp_path / "artifact.bin"
    metadata = tmp_path / "artifact.json"
    atomic_write_bytes(binary, b"evidence")
    atomic_write_json(metadata, {"passed": True})
    assert stat.S_IMODE(binary.stat().st_mode) == 0o644
    assert stat.S_IMODE(metadata.stat().st_mode) == 0o644
