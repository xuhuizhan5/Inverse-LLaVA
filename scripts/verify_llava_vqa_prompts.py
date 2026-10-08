"""Verify prepared MM-Vet prompts against LLaVA's actual VQA evaluator body."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--official-conversation", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        sha256_file(args.official_evaluator)
        != "51e892a0d3e2b23572b951fb6d99f8a1b2aec976fc9e299a4493a7d60dd9c94b"
    ):
        raise ValueError("unexpected LLaVA c121f043 VQA source")
    if (
        sha256_file(args.official_conversation)
        != "5afd26fa231082cb6fdc100c3887d663f72d8c99a93dcdd285dac303ca72e45c"
    ):
        raise ValueError("unexpected LLaVA conversation source")
    tree = ast.parse(args.official_evaluator.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "eval_model"
    )
    loop = next(node for node in function.body if isinstance(node, ast.For))
    body = []
    for node in loop.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "input_ids" for t in node.targets
        ):
            break
        body.append(node)
    if not any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets)
        for node in body
    ):
        raise ValueError("unexpected upstream prompt construction")
    statements = compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
        str(args.official_evaluator),
        "exec",
    )
    module_spec = importlib.util.spec_from_file_location(
        "pinned_llava_vqa_conversation", args.official_conversation
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    questions = [json.loads(line) for line in args.questions.read_text().splitlines()]
    references = json.loads(args.annotations.read_text())
    prepared = load_examples(args.examples)
    indexed = {row.id: row for row in prepared}
    expected_ids = [f"v1_{row['question_id']}" for row in questions]
    if (
        len(questions) != 218
        or len(indexed) != len(prepared)
        or set(expected_ids) != indexed.keys()
        or indexed.keys() != references.keys()
    ):
        raise ValueError("MM-Vet full-ID coverage mismatch")
    for question in questions:
        sample_id = f"v1_{question['question_id']}"
        example, reference = indexed[sample_id], references[sample_id]
        namespace = dict(
            line=question,
            conv_templates=module.conv_templates,
            args=SimpleNamespace(conv_mode="vicuna_v1"),
            model=SimpleNamespace(config=SimpleNamespace(mm_use_im_start_end=False)),
            DEFAULT_IMAGE_TOKEN="<image>",
            DEFAULT_IM_START_TOKEN="<im_start>",
            DEFAULT_IM_END_TOKEN="<im_end>",
        )
        exec(statements, namespace)
        if example.prompt != namespace["prompt"] or question["text"] != reference["question"]:
            raise ValueError(f"prompt mismatch: {sample_id}")
        if (
            example.references != (reference["answer"],)
            or example.images[0].name != question["image"]
            or question["image"] != reference["imagename"]
        ):
            raise ValueError(f"image/reference mismatch: {sample_id}")
    atomic_write_json(
        args.output,
        {
            "status": "passed",
            "questions": 218,
            "image_count": len({row.images[0] for row in prepared}),
            "sources": {
                key: sha256_file(value) for key, value in vars(args).items() if key != "output"
            },
            "scope": (
                "Actual pinned LLaVA prompt statements, all IDs, image names, and "
                "compositional references. No judge or model-forward equivalence claim."
            ),
        },
    )
    print("Passed: all 218 MM-Vet prompts, IDs, references, and image bindings")


if __name__ == "__main__":
    main()
