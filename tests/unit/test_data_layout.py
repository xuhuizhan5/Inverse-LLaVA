import hashlib
import json
import os

import pytest

import invllava.data.layout as data_layout
from invllava.config.schema import DataLayoutSpec
from invllava.data.layout import materialize_hardlink_layout


def test_layout_is_atomic_and_uses_hardlinks(tmp_path) -> None:
    components = tmp_path / "components"
    source = components / "corpus" / "images"
    source.mkdir(parents=True)
    original = source / "one.jpg"
    original.write_bytes(b"fixture-image")
    spec = DataLayoutSpec.model_validate(
        {
            "id": "fixture-layout",
            "entries": [{"source": "corpus/images", "destination": "corpus/images"}],
        }
    )
    output = tmp_path / "raw"
    manifest = tmp_path / "layout.json"
    materialize_hardlink_layout(
        spec,
        component_root=components,
        destination=output,
        manifest_path=manifest,
        layout_config_sha256=hashlib.sha256(b"fixture-config").hexdigest(),
    )
    assembled = output / "corpus" / "images" / "one.jpg"
    assert assembled.read_bytes() == b"fixture-image"
    assert os.stat(assembled).st_ino == os.stat(original).st_ino
    assert json.loads(manifest.read_text())["files"] == 1


def test_layout_rejects_links_in_a_component(tmp_path) -> None:
    components = tmp_path / "components"
    source = components / "corpus"
    source.mkdir(parents=True)
    target = tmp_path / "outside.jpg"
    target.write_bytes(b"outside")
    (source / "link.jpg").symlink_to(target)
    spec = DataLayoutSpec.model_validate(
        {"id": "fixture-layout", "entries": [{"source": "corpus", "destination": "corpus"}]}
    )
    with pytest.raises(ValueError, match="link or special"):
        materialize_hardlink_layout(
            spec,
            component_root=components,
            destination=tmp_path / "raw",
            manifest_path=tmp_path / "layout.json",
            layout_config_sha256="0" * 64,
        )
    assert not (tmp_path / "raw").exists()


def test_layout_rejects_destination_nested_under_components(tmp_path) -> None:
    components = tmp_path / "components"
    (components / "corpus").mkdir(parents=True)
    spec = DataLayoutSpec.model_validate(
        {"id": "fixture-layout", "entries": [{"source": "corpus", "destination": "corpus"}]}
    )

    with pytest.raises(ValueError, match="separate sibling trees"):
        materialize_hardlink_layout(
            spec,
            component_root=components,
            destination=components / "assembled",
            manifest_path=tmp_path / "layout.json",
            layout_config_sha256="0" * 64,
        )


def test_layout_rejects_manifest_inside_output(tmp_path) -> None:
    components = tmp_path / "components"
    (components / "corpus").mkdir(parents=True)
    spec = DataLayoutSpec.model_validate(
        {"id": "fixture-layout", "entries": [{"source": "corpus", "destination": "corpus"}]}
    )
    output = tmp_path / "raw"

    with pytest.raises(ValueError, match="manifest"):
        materialize_hardlink_layout(
            spec,
            component_root=components,
            destination=output,
            manifest_path=output / "layout.json",
            layout_config_sha256="0" * 64,
        )


def test_layout_reports_hardlink_ownership_failure_without_partial_output(
    tmp_path, monkeypatch
) -> None:
    components = tmp_path / "components"
    source = components / "corpus"
    source.mkdir(parents=True)
    (source / "one.jpg").write_bytes(b"fixture-image")
    spec = DataLayoutSpec.model_validate(
        {"id": "fixture-layout", "entries": [{"source": "corpus", "destination": "corpus"}]}
    )

    def reject_link(_source, _target) -> None:
        raise PermissionError(1, "operation not permitted")

    monkeypatch.setattr(data_layout.os, "link", reject_link)
    output = tmp_path / "raw"
    with pytest.raises(RuntimeError, match=r"source UID=.*effective UID=.*same UID"):
        materialize_hardlink_layout(
            spec,
            component_root=components,
            destination=output,
            manifest_path=tmp_path / "layout.json",
            layout_config_sha256="0" * 64,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".raw.*"))
