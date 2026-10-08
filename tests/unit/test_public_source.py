"""The distributable research tree must remain usable without author files."""

import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).parents[2]


def exporter():
    spec = importlib.util.spec_from_file_location(
        "public_export", ROOT / "tools/artifacts/export_public_source.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_allowlist_retains_scientific_code_and_excludes_author_material():
    paths = {str(p.relative_to(ROOT)) for p in exporter().selected(ROOT)}
    assert {
        "src/invllava/cli.py",
        "configs/experiment/canonical_7b.yaml",
        "requirements.txt",
        ".dockerignore",
        "assets/overview.json",
        "docs/training.md",
        "LICENSE",
        "NOTICE",
        "third_party/licenses/Pythia-BSD.txt",
        ".gitattributes",
        ".pre-commit-config.yaml",
    } <= paths
    assert all(
        not p.startswith(
            (
                ".git/",
                ".runpod/",
                "Inverse-LLaVA-Tex/",
                "docs/revision/",
                "tools/runpod/",
                "tools/thor/",
            )
        )
        for p in paths
    )
    assert ".env" not in paths
    assert "scripts/run_code_only_tests.py" not in paths
    assert not any(Path(p).name.startswith("test_runpod_") for p in paths)


def test_container_preserves_dependency_records_for_source_identity():
    docker = (ROOT / "containers/Dockerfile").read_text()
    ignore = (ROOT / ".dockerignore").read_text()
    assert "COPY pyproject.toml uv.lock requirements.txt README.md ./" in docker
    assert "COPY requirements ./requirements" in docker
    assert "COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md ./" in docker
    assert "COPY third_party/licenses ./third_party/licenses" in docker
    assert "!LICENSE" in ignore and "!third_party/licenses/**" in ignore
    assert "!requirements.txt" in ignore and "!requirements/**" in ignore


def test_public_link_checker_reports_missing_local_targets(tmp_path):
    (tmp_path / "README.md").write_text("[Paper](https://example.org) [Guide](guide.md)")
    assert exporter().local_links(tmp_path) == ["README.md -> guide.md"]
    (tmp_path / "guide.md").write_text("# Guide\n")
    assert exporter().local_links(tmp_path) == []


def test_public_ignore_removes_manuscript_exceptions_and_is_idempotent():
    module = exporter()
    result = module.public_gitignore((ROOT / ".gitignore").read_text())
    assert "\n/Inverse-LLaVA-Tex/\n" in result
    assert "!Inverse-LLaVA-Tex/" not in result
    assert "\n.env\n" in result and "\n.aws/\n" in result
    assert module.public_gitignore(result) == result


def test_export_is_self_contained_and_checksums_output(tmp_path):
    import hashlib

    module = exporter()
    output = tmp_path / "public"
    record = module.export(ROOT, output)
    assert record["missing_document_links"] == []
    for name, expected in record["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert set(record["files"]) == {
        str(path.relative_to(output)) for path in module.selected(output)
    }


def test_identity_assets_are_standalone_vectors_with_the_same_emblem():
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assets = [
        ET.parse(ROOT / "assets" / name).getroot()
        for name in ("inverse-llava-logo.svg", "inverse-llava-mark.svg")
    ]
    for root in assets:
        assert root.get("role") == "img"
        assert root.find("svg:title", ns).text.startswith("Inverse-LLaVA")
        assert root.find("svg:desc", ns).text
        assert "prefers-color-scheme: dark" in root.find("svg:style", ns).text
        for node in root.iter():
            assert node.tag.rsplit("}", 1)[-1] not in ("image", "script", "text", "foreignObject")
            assert not any(key.rsplit("}", 1)[-1] == "href" for key in node.attrib)
    paths = [[path.attrib for path in root.find("svg:g[@id='llama']", ns)] for root in assets]
    assert paths[0] == paths[1]
