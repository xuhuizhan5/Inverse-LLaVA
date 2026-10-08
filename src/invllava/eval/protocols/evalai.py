# Normalization tables derive from Pythia/MMF through LLaVA's m4c evaluator.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# See third_party/licenses/Pythia-BSD.txt and NOTICE for attribution and changes.
"""VQA EvalAI answer normalization.

This is a dependency-free reimplementation of the normalization specified by
the official VQA evaluator. Keep it golden-tested against that evaluator and
the TextVQA reference path before marking either benchmark verified.
"""

from __future__ import annotations

import re

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id've": "i'd've",
    "i'dve": "i'd've",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = {"a", "an", "the"}
UNREACHABLE_CAPITALIZED_CONTRACTIONS = {"id've", "i'dve", "im", "ive"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")
PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)


def clean_vqa_text(value: str) -> str:
    return value.replace("\n", " ").replace("\t", " ").strip()


def normalize_evalai_answer(value: str) -> str:
    """Apply official EvalAI punctuation, number, article, and contraction rules."""
    value = clean_vqa_text(value).lower()
    normalized = value
    for punctuation in PUNCTUATION:
        if (f"{punctuation} " in value or f" {punctuation}" in value) or COMMA_STRIP.search(value):
            normalized = normalized.replace(punctuation, "")
        else:
            normalized = normalized.replace(punctuation, " ")
    normalized = PERIOD_STRIP.sub("", normalized)
    words = []
    for word in normalized.split():
        word = NUMBER_MAP.get(word, word)
        if word not in ARTICLES:
            words.append(
                word
                if word in UNREACHABLE_CAPITALIZED_CONTRACTIONS
                else CONTRACTIONS.get(word, word)
            )
    return " ".join(words)


def normalize_textvqa_answer(value: str) -> str:
    """Match the paper-era LLaVA/MMF TextVQA answer processor exactly.

    TextVQA's released evaluator removes commas and question marks before its
    general punctuation pass. It also contains four capitalized contraction
    keys that are unreachable after lower-casing; preserving that behavior is
    part of reproducing the reported metric.
    """

    normalized = value.lower().replace(",", "").replace("?", "").replace("'s", " 's")
    normalized = clean_vqa_text(normalized)
    for punctuation in PUNCTUATION:
        if (
            f"{punctuation} " in normalized or f" {punctuation}" in normalized
        ) or COMMA_STRIP.search(normalized):
            normalized = normalized.replace(punctuation, "")
        else:
            normalized = normalized.replace(punctuation, " ")
    normalized = PERIOD_STRIP.sub("", normalized)
    words = []
    for word in normalized.lower().split():
        word = NUMBER_MAP.get(word, word)
        if word not in ARTICLES:
            words.append(
                word
                if word in UNREACHABLE_CAPITALIZED_CONTRACTIONS
                else CONTRACTIONS.get(word, word)
            )
    return " ".join(words)
