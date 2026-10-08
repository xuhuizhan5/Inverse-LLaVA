"""Compare every prepared MMBench prompt with the pinned LLaVA evaluator body.

The upstream AST supplies its actual prompt-building statements. Model loading,
image decoding, tokenization, and inference are excluded from this CPU oracle;
their conformance requires separate checks. Original source hashes are retained.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import load_examples


def load_prompt_oracle(source: Path, conversation: Path):
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "eval_model"
    )
    outer = next(node for node in function.body if isinstance(node, ast.For))
    inner = next(node for node in outer.body if isinstance(node, ast.For))
    body = []
    for node in inner.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "input_ids" for target in node.targets
        ):
            break
        body.append(node)
    if not body or not any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "prompt" for target in node.targets)
        for node in body
    ):
        raise ValueError("unexpected upstream evaluator AST")
    wrapper = ast.parse("def oracle(row, options, args, model, conv_templates):\n    pass\n").body[
        0
    ]
    wrapper.body = [*body, ast.Return(value=ast.Name(id="prompt", ctx=ast.Load()))]
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"is_none", "get_options"}
    ]
    module = ast.fix_missing_locations(ast.Module(body=[*definitions, wrapper], type_ignores=[]))
    namespace = {
        "math": math,
        "all_options": list("ABCD"),
        "DEFAULT_IMAGE_TOKEN": "<image>",
        "DEFAULT_IM_START_TOKEN": "<im_start>",
        "DEFAULT_IM_END_TOKEN": "<im_end>",
        "load_image_from_base64": lambda _: None,
    }
    exec(compile(module, str(source), "exec"), namespace)
    spec = importlib.util.spec_from_file_location("pinned_llava_conversation", conversation)
    if spec is None or spec.loader is None:
        raise ImportError(conversation)
    loaded = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return namespace, loaded.conv_templates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--official-conversation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.official_evaluator) != (
        "e4c9234a948f8ac9cca95a55a5a2c5d0d0f04aa611c21d62736a5266599be242"
    ) or sha256_file(args.official_conversation) != (
        "5afd26fa231082cb6fdc100c3887d663f72d8c99a93dcdd285dac303ca72e45c"
    ):
        raise ValueError("upstream source differs from pinned LLaVA c121f043")
    benchmark = BenchmarkSpec.model_validate(yaml.safe_load(args.benchmark.read_text()))
    if sha256_file(args.tsv) != benchmark.annotations.sha256:
        raise ValueError("TSV hash differs from the declared release")
    namespace, templates = load_prompt_oracle(args.official_evaluator, args.official_conversation)
    rows = pd.read_table(args.tsv)
    examples = load_examples(args.examples)
    indexed = {row.id: row for row in examples}
    if len(rows) != len(indexed) or len(indexed) != 4329:
        raise ValueError("expected all 4,329 unique dev rotations")
    model = SimpleNamespace(config=SimpleNamespace(mm_use_im_start_end=False))
    settings = SimpleNamespace(
        single_pred_prompt=True,
        conv_mode="vicuna_v1",
        lang="cn" if benchmark.id == "mmbench-cn" else "en",
    )
    for _, row in rows.iterrows():
        options = namespace["get_options"](row, list("ABCD"))
        expected = namespace["oracle"](row, options, settings, model, templates)
        example = indexed[str(row["index"])]
        if expected != example.prompt:
            raise ValueError(f"prompt differs from actual upstream statements: {example.id}")
        if example.references != (row["answer"],) or tuple(options) != example.choices:
            raise ValueError(f"reference/option disagreement: {example.id}")
    atomic_write_json(
        args.output,
        {
            "status": "passed",
            "prompts_checked": len(rows),
            "benchmark": benchmark.id,
            "protocol_revision": benchmark.protocol_revision,
            "upstream_revision": "c121f0432da27facab705978f83c4ada465e46fd",
            "sources": {
                key: sha256_file(value) for key, value in vars(args).items() if key != "output"
            },
            "scope": "Actual upstream AST prompt construction and Vicuna conversation, all rows. "
            "No model forward, image-processor, or hidden-label judge equivalence claim.",
        },
    )
    print(f"Passed: {len(rows)} {benchmark.id} prompts, options, and references")


if __name__ == "__main__":
    main()
