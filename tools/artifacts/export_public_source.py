"""Export the research code without author correspondence or cloud credentials.

An explicit allowlist keeps this operation independent of private Git history.
It never deletes or modifies the source workspace. No model dependencies are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

DIRECTORIES = (
    "src",
    "configs",
    "scripts",
    "tests",
    "requirements",
    "containers",
    "third_party",
    "assets",
    "tools/artifacts",
    "tools/profiling",
    "docs/method",
    "docs/extension",
)
FILES = (
    "README.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
    "NOTICE",
    ".gitattributes",
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "tools/__init__.py",
    ".gitignore",
    ".dockerignore",
    ".env.example",
    "docs/training.md",
    "docs/evaluation.md",
    "docs/analysis.md",
    "docs/checkpoints.md",
    "docs/benchmarks/INDEX.md",
    "docs/guides/DATA_PREPARATION.md",
    "docs/guides/EVALUATION.md",
    "docs/guides/PROFILING.md",
    "docs/guides/REPRESENTATIONS_AND_CASES.md",
    "docs/guides/LMMS_EVAL.md",
    "docs/guides/KERNEL_OPTIMIZATION.md",
)
TOKEN = re.compile(
    rb"rpa_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|"
    rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|"
    rb"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}"
)
EXCLUDED_FILES = {
    "scripts/run_code_only_tests.py",
    "tests/unit/test_code_only_runner.py",
    "configs/experiment/champion_continuation_canary.yaml",
}
PRIVATE_IGNORE_MARKER = "# Private author workspace (excluded from public source)"


def public_gitignore(text):
    """Keep runtime exclusions and remove private manuscript tracking exceptions."""
    for marker in ("# Generated manuscript files", PRIVATE_IGNORE_MARKER):
        text = text.split(marker, 1)[0]
    return (
        text.rstrip()
        + "\n\n"
        + PRIVATE_IGNORE_MARKER
        + "\n"
        + "\n".join(
            (
                "/Inverse-LLaVA-Tex/",
                "/docs/revision/",
                "/docs/results/",
                "/docs/reproducibility/",
                "/tools/runpod/",
                "/tools/thor/",
                "/revision_addressable.md",
                "/SOURCE_MANIFEST.json",
                "",
            )
        )
    )


def selected(root):
    paths = [root / name for name in FILES]
    for name in DIRECTORIES:
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    for path in sorted(set(paths)):
        if path.relative_to(root).as_posix() in EXCLUDED_FILES:
            continue
        if any(p in ("__pycache__", ".pytest_cache", ".ruff_cache") for p in path.parts):
            continue
        if (
            path.name.startswith(("._", "test_runpod_"))
            or path.name == ".DS_Store"
            or path.suffix in (".pyc", ".pyo")
        ):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked source: {path.relative_to(root)}")
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError(f"Unexpected large public file: {path.relative_to(root)}")
        if TOKEN.search(path.read_bytes()):
            raise ValueError(f"Credential-shaped content: {path.relative_to(root)}")
        yield path


def local_links(root):
    missing = []
    for path in root.rglob("*.md"):
        text = path.read_text()
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        targets = re.findall(r"\]\(([^)\s]+)\)", text)
        targets += re.findall(r'(?:src|href)="([^"]+)"', text)
        for target in targets:
            if target.startswith(("http:", "https:", "mailto:", "#", "data:")):
                continue
            target = target.split("#")[0].strip("<>")
            if not (path.parent / target).exists():
                missing.append(f"{path.relative_to(root)} -> {target}")
    return missing


def export(root, output):
    root, output = root.resolve(), output.absolute()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; existing exports are retained.")
    paths = list(selected(root))
    output.mkdir(parents=True)
    for path in paths:
        relative = path.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if relative.as_posix() == ".gitignore":
            target.write_text(public_gitignore(path.read_text()))
        target.chmod(path.stat().st_mode & 0o777)
    missing = local_links(output)
    manifest = {
        "files": {
            str(p.relative_to(root)): hashlib.sha256(
                (output / p.relative_to(root)).read_bytes()
            ).hexdigest()
            for p in paths
        },
        "missing_document_links": missing,
        "excluded": (
            "Private author workspace, manuscript sources, correspondence, "
            "dated experiment records, cloud operations, machine adapters, "
            "caches and Git history."
        ),
        "license": "Apache-2.0 for project source; see THIRD_PARTY_NOTICES.md for separate terms.",
    }
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if missing:
        raise ValueError("Unresolved documentation links:\n" + "\n".join(missing))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = export(args.root, args.output)
    print(json.dumps({"files": len(record["files"]), "output": str(args.output)}))
