"""Check saved linear CKA against a normalized unbiased-HSIC sensitivity estimate."""

from __future__ import annotations

import argparse
import ast
import itertools
import json
from pathlib import Path

import numpy as np

from invllava.analysis.representations import linear_cka
from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.source import execution_source_sha256

REFERENCE_REVISION = "89e3921863e276cdbe49bd25077905f75e981f4e"
REFERENCE_SHA256 = "de82b8f18d8dac00fb92da0ed6bee43cff177203610ad4d5d681aaeb20b475d9"
REFERENCE_URL = (
    "https://github.com/google-research/google-research/blob/"
    + REFERENCE_REVISION
    + "/representation_similarity/Demo.ipynb"
)


def official_cka(path: Path):
    """Load only three definitions from the hash-pinned reference notebook."""
    if sha256_file(path) != REFERENCE_SHA256:
        raise ValueError("official CKA reference checksum differs")
    required = {"gram_linear", "center_gram", "cka"}
    nodes = []
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            nodes.extend(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in required
            )
    if len(nodes) != 3 or {node.name for node in nodes} != required:
        raise ValueError("official CKA definitions differ")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["cka"]


def compare_estimators(left, right, reference):
    """Preserve signed estimates and verify the independent official Gram path."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    ordinary = linear_cka(left, right)
    corrected = linear_cka(left, right, debiased=True)
    # Centering also improves numerical conditioning when states share a large mean.
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    expected = reference(left @ left.T, right @ right.T, debiased=True)
    error = abs(corrected - expected)
    if not np.isfinite(expected) or error > 1e-9:
        raise ValueError("unbiased-HSIC CKA differs from the official reference")
    return {
        "linear_cka": ordinary,
        "unbiased_hsic_cka": corrected,
        "official_reference_absolute_error": error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--official-notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.audit) != args.expected_audit_sha256:
        raise ValueError("accepted geometry audit changed")
    audit = json.loads(args.audit.read_text())
    reference = official_cka(args.official_notebook)
    layers = audit["layers"]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("layers must be nonempty and unique")
    captures, ids, condition = {}, None, None
    for name, record in audit["captures"].items():
        path = Path(record["path"])
        if sha256_file(path) != record["artifact_sha256"]:
            raise ValueError("capture checksum differs")
        if sha256_file(path.with_suffix(".json")) != record["metadata_sha256"]:
            raise ValueError("capture metadata changed")
        if record["examples_sha256"] != audit["examples_sha256"] or record["batch_size"] != 1:
            raise ValueError("capture input or batch policy differs")
        if ids is None:
            ids, condition = record["sample_ids"], record["input_condition"]
        if record["sample_ids"] != ids or record["input_condition"] != condition:
            raise ValueError("sample correspondence or input condition differs")
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["sample_ids"].astype(str).tolist() != ids:
                raise ValueError("capture tensor rows differ")
            captures[name] = {layer: arrays[f"hidden.last.{layer}"] for layer in layers}
    if len(captures) < 2 or ids is None or len(ids) != len(set(ids)):
        raise ValueError("two or more aligned captures with unique sample IDs are required")
    expected_keys = {
        (frozenset((a, b)), layer)
        for a, b in itertools.combinations(captures, 2)
        for layer in layers
    }
    prior = {(r["left"], r["right"], r["layer"]): r for r in audit["results"]}
    actual_keys = {(frozenset((a, b)), layer) for a, b, layer in prior}
    if len(actual_keys) != len(audit["results"]) or actual_keys != expected_keys:
        raise ValueError("pair/layer coverage differs")
    results = []
    for (left, right, layer), row in prior.items():
        result = compare_estimators(captures[left][layer], captures[right][layer], reference)
        if not np.isclose(result["linear_cka"], row["cka"], rtol=0, atol=1e-12):
            raise ValueError("original CKA no longer reproduces")
        results.append({"left": left, "right": right, "layer": layer, **result})
    atomic_write_json(
        args.output,
        {
            "status": "passed",
            "audit_sha256": sha256_file(args.audit),
            "official_reference_url": REFERENCE_URL,
            "official_reference_sha256": REFERENCE_SHA256,
            "script_sha256": sha256_file(__file__),
            "source_sha256": execution_source_sha256(Path(__file__).resolve().parents[1])[0],
            "sample_count": len(ids),
            "input_condition": condition,
            "results": results,
            "scope": "Estimator sensitivity on the same fitted-model captures. Normalizing "
            "unbiased HSIC does not make CKA unbiased. No new uncertainty intervals, model "
            "evaluations, or evidence of grounding or architectural superiority.",
        },
    )
    print(json.dumps({"status": "passed", "comparisons": len(results), "results": results}))


if __name__ == "__main__":
    main()
