# ruff: noqa: RUF001
"""Exact answer-accuracy path from the public VizWiz VQA evaluator."""

from __future__ import annotations

import re
from collections import defaultdict

from invllava.eval.protocols.evalai import ARTICLES, NUMBER_MAP, PUNCTUATION, clean_vqa_text
from invllava.eval.types import EvaluationExample, Score

# The April 2026 answer release points to the January 2023 VizWiz API. Its
# contraction table contains deliberate straight/curly-apostrophe asymmetries;
# preserving them is required for exact local scoring.
VIZWIZ_CONTRACTIONS = {
    "Id’ve": "I’d’ve",
    "Im": "I’m",
    "Ive": "I’ve",
    "I’dve": "I’d’ve",
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldn'tve": "couldn’t’ve",
    "couldnt": "couldn't",
    "couldnt’ve": "couldn’t’ve",
    "couldve": "could've",
    "didnt": "didn’t",
    "doesnt": "doesn’t",
    "dont": "don’t",
    "hadn'tve": "hadn’t’ve",
    "hadnt": "hadn’t",
    "hadnt’ve": "hadn’t’ve",
    "hasnt": "hasn’t",
    "havent": "haven’t",
    "hed": "he’d",
    "hed’ve": "he’d’ve",
    "hes": "he’s",
    "he’dve": "he’d’ve",
    "howd": "how’d",
    "howll": "how’ll",
    "hows": "how’s",
    "isnt": "isn’t",
    "itd": "it’d",
    "itd’ve": "it’d’ve",
    "itll": "it’ll",
    "it’dve": "it’d’ve",
    "let’s": "let’s",
    "maam": "ma’am",
    "mightnt": "mightn’t",
    "mightnt’ve": "mightn’t’ve",
    "mightn’tve": "mightn’t’ve",
    "mightve": "might’ve",
    "mustnt": "mustn’t",
    "mustve": "must’ve",
    "neednt": "needn’t",
    "notve": "not’ve",
    "oclock": "o’clock",
    "oughtnt": "oughtn’t",
    "ow’s’at": "’ow’s’at",
    "shant": "shan’t",
    "shed’ve": "she’d’ve",
    "she’dve": "she’d’ve",
    "she’s": "she’s",
    "shouldnt": "shouldn’t",
    "shouldnt’ve": "shouldn’t’ve",
    "shouldn’tve": "shouldn’t’ve",
    "shouldve": "should’ve",
    "somebodyd’ve": "somebody’d’ve",
    "somebodyll": "somebody’ll",
    "somebodys": "somebody’s",
    "somebody’d": "somebodyd",
    "somebody’dve": "somebody’d’ve",
    "someoned": "someone’d",
    "someoned’ve": "someone’d’ve",
    "someonell": "someone’ll",
    "someones": "someone’s",
    "someone’dve": "someone’d’ve",
    "somethingd": "something’d",
    "somethingd’ve": "something’d’ve",
    "somethingll": "something’ll",
    "something’dve": "something’d’ve",
    "thats": "that’s",
    "thered": "there’d",
    "thered’ve": "there’d’ve",
    "therere": "there’re",
    "theres": "there’s",
    "there’dve": "there’d’ve",
    "theyd": "they’d",
    "theyd’ve": "they’d’ve",
    "theyll": "they’ll",
    "theyre": "they’re",
    "theyve": "they’ve",
    "they’dve": "they’d’ve",
    "twas": "’twas",
    "wasnt": "wasn’t",
    "wed’ve": "we’d’ve",
    "werent": "weren’t",
    "weve": "we've",
    "we’dve": "we’d’ve",
    "whatll": "what’ll",
    "whatre": "what’re",
    "whats": "what’s",
    "whatve": "what’ve",
    "whens": "when’s",
    "whered": "where’d",
    "wheres": "where's",
    "whereve": "where’ve",
    "whod": "who’d",
    "whod’ve": "who’d’ve",
    "wholl": "who’ll",
    "whos": "who’s",
    "whove": "who've",
    "who’dve": "who’d’ve",
    "whyll": "why’ll",
    "whyre": "why’re",
    "whys": "why’s",
    "wont": "won’t",
    "wouldnt": "wouldn’t",
    "wouldnt’ve": "wouldn’t’ve",
    "wouldn’tve": "wouldn’t’ve",
    "wouldve": "would’ve",
    "yall": "y’all",
    "yall’d’ve": "y’all’d’ve",
    "yall’ll": "y’all’ll",
    "youd": "you’d",
    "youd’ve": "you’d’ve",
    "youll": "you’ll",
    "youre": "you’re",
    "youve": "you’ve",
    "you’dve": "you’d’ve",
    "y’alld’ve": "y’all’d’ve",
    "y’allll": "y’all’ll",
    "y’all’dve": "y’all’d’ve",
    "’ows’at": "’ow’s’at",
    "’ow’sat": "’ow’s’at",
}

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize_vizwiz_answer(value: str) -> str:
    """Apply the public VizWiz API's prediction-only normalization exactly."""

    value = clean_vqa_text(value)
    normalized = value
    for punctuation in PUNCTUATION:
        if (f"{punctuation} " in value or f" {punctuation}" in value) or _COMMA_STRIP.search(value):
            normalized = normalized.replace(punctuation, "")
        else:
            normalized = normalized.replace(punctuation, " ")
    normalized = _PERIOD_STRIP.sub("", normalized)
    words = []
    for word in normalized.lower().split():
        word = NUMBER_MAP.get(word, word)
        if word not in ARTICLES:
            words.append(VIZWIZ_CONTRACTIONS.get(word, word))
    return " ".join(words)


def _consensus(prediction: str, answers: list[str]) -> float:
    values = []
    for index in range(len(answers)):
        matches = sum(
            prediction == answer for other, answer in enumerate(answers) if other != index
        )
        values.append(min(1.0, matches / 3.0))
    return sum(values) / len(values)


class VizWizProtocol:
    """VizWiz's prediction-normalized, leave-one-annotator-out accuracy."""

    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        per_item: dict[str, float] = {}
        by_answer_type: dict[str, list[float]] = defaultdict(list)
        for example in examples:
            answers = [str(answer) for answer in example.references]
            if len(answers) != 10:
                raise ValueError(
                    f"VizWiz example {example.id} has {len(answers)} references; expected 10"
                )
            prediction = normalize_vizwiz_answer(predictions[example.id])
            accuracy = _consensus(prediction, answers)
            per_item[example.id] = accuracy
            by_answer_type[str(example.metadata.get("answer_type", "unknown"))].append(accuracy)
        return Score(
            sum(per_item.values()) / len(per_item) if per_item else 0.0,
            len(per_item),
            {
                "per_item": per_item,
                "answer_type_accuracy": {
                    key: sum(values) / len(values) for key, values in sorted(by_answer_type.items())
                },
            },
        )
