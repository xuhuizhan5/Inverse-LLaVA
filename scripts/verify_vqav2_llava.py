"""Check all test-dev prompts and the full upload envelope against pinned LLaVA code.

The released 13B answers are a converter fixture, not a new model evaluation.
This check establishes input/export equivalence; it does not establish model
forward equivalence or server acceptance.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.records import PredictionRecord
from invllava.eval.submission import build_vqav2_submission


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def statements(body, filename):
    return compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(filename), "exec"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "loader",
        "conversation",
        "normalizer",
        "converter",
        "archive",
        "full-test-questions",
        "examples",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pins = {
        "loader": "c91a9ea3b5cb4cc27799ba9585a0d2a1f33c37ed822ae21ba4b5b20444f90681",
        "conversation": "5afd26fa231082cb6fdc100c3887d663f72d8c99a93dcdd285dac303ca72e45c",
        "normalizer": "8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774",
        "converter": "7d932fcb5b91dc2fa54efc173a39a27c53c5aeb0eaccb3c71fd3559309e6f47c",
        "archive": "2df36a33e3d3947d3e351e627449ee7cc7725d69ecc8dcbb2103d90172fbe17a",
        "full_test_questions": "69169f086e0fa9f878a7425aabe6573e999af79029726b64bbf144482000960a",
    }
    for key, digest in pins.items():
        if sha256_file(getattr(args, key)) != digest:
            raise ValueError(f"pinned upstream input changed: {key}")
    conversation = load_module(args.conversation, "vqav2_pinned_conversation")
    normalizer = load_module(args.normalizer, "vqav2_pinned_normalizer")
    tree = ast.parse(args.loader.read_text())
    dataset = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CustomDataset"
    )
    getter = next(
        node
        for node in dataset.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getitem__"
    )
    body = []
    for node in getter.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "image" for target in node.targets
        ):
            break
        body.append(node)
    prompt_code = statements(body, args.loader)
    with zipfile.ZipFile(args.archive) as bundle:

        def read_member(name):
            return [json.loads(line) for line in bundle.read(name).splitlines()]

        questions = read_member("vqav2/llava_vqav2_mscoco_test-dev2015.jsonl")
        full = read_member("vqav2/llava_vqav2_mscoco_test2015.jsonl")
        answers = read_member(
            "vqav2/answers/llava_vqav2_mscoco_test-dev2015/llava-v1.5-13b/merge.jsonl"
        )
    examples = load_examples(args.examples)
    indexed = {row.id: row for row in examples}
    expected = {str(row["question_id"]) for row in questions}
    if len(questions) != 107394 or len(indexed) != len(examples) or indexed.keys() != expected:
        raise ValueError("incomplete or duplicate test-dev IDs")
    for question in questions:
        namespace = dict(
            self=SimpleNamespace(
                questions=[question], model_config=SimpleNamespace(mm_use_im_start_end=False)
            ),
            index=0,
            args=SimpleNamespace(conv_mode="vicuna_v1"),
            conv_templates=conversation.conv_templates,
            DEFAULT_IMAGE_TOKEN="<image>",
            DEFAULT_IM_START_TOKEN="<im_start>",
            DEFAULT_IM_END_TOKEN="<im_end>",
        )
        exec(prompt_code, namespace)
        example = indexed[str(question["question_id"])]
        if (
            example.prompt != namespace["prompt"]
            or example.references
            or len(example.images) != 1
            or example.images[0].name != question["image"]
        ):
            raise ValueError(f"prompt/image/reference mismatch: {example.id}")
    official = json.loads(args.full_test_questions.read_text())["questions"]
    full_ids = [row["question_id"] for row in official]
    if len(full_ids) != len(set(full_ids)) or len(full_ids) != 447793:
        raise ValueError("incomplete full-test envelope")
    if full_ids != [row["question_id"] for row in full]:
        raise ValueError("official question order differs from the LLaVA upload fixture")
    for annotation, fixture in zip(official, full, strict=True):
        if (
            fixture["image"] != f"COCO_test2015_{annotation['image_id']:012d}.jpg"
            or fixture["text"]
            != annotation["question"] + "\nAnswer the question using a single word or phrase."
        ):
            raise ValueError("full-test fixture has changed question/image bindings")
    records = [
        PredictionRecord(
            1,
            "released-export-fixture",
            "fixture",
            "published-13b-fixture",
            str(row["question_id"]),
            "Fixture only",
            row["text"],
        )
        for row in answers
    ]
    tree = ast.parse(args.converter.read_text())
    main_body = next(node.body for node in tree.body if isinstance(node, ast.If))
    loop = next(
        node
        for node in main_body
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "test_split"
    )
    namespace = dict(
        test_split=full,
        results={int(row.sample_id): row.prediction for row in records},
        all_answers=[],
        answer_processor=normalizer.EvalAIAnswerProcessor(),
    )
    exec(statements([loop], args.converter), namespace)
    with tempfile.TemporaryDirectory(
        prefix="vqav2-export-golden-", dir=args.output.parent
    ) as temporary:
        exported = Path(temporary) / "submission.json"
        build_vqav2_submission(records, expected, exported, full_test_question_ids=full_ids)
        if json.loads(exported.read_text()) != namespace["all_answers"]:
            raise ValueError("submission differs from the actual LLaVA converter")
        digest = sha256_file(exported)
    atomic_write_json(
        args.output,
        {
            "status": "passed",
            "testdev_questions": len(examples),
            "upload_questions": len(full_ids),
            "outside_testdev_empty_answers": len(full_ids) - len(examples),
            "images": len({row.images[0] for row in examples}),
            "released_fixture_answers": len(records),
            "fixture_export_sha256": digest,
            "sources": {
                key: sha256_file(value) for key, value in vars(args).items() if key != "output"
            },
            "scope": "Actual upstream prompt and converter statements; all test-dev inputs and "
            "released-answer normalization. Model-forward and server acceptance are separate.",
        },
    )
    print(
        json.dumps({"status": "passed", "questions": len(examples), "upload_rows": len(full_ids)})
    )


if __name__ == "__main__":
    main()
