import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")

from invllava.eval.interop import lm_eval as lm_eval_interop
from invllava.eval.interop.lm_eval import NativeTextLM, _json_default, run_language_suite
from invllava.eval.language_suite import LanguageRetentionSuite


def _fixture_filter() -> None:
    pass


def test_lm_eval_metadata_serializes_torch_runtime_values() -> None:
    payload = {
        "dtype": torch.bfloat16,
        "device": torch.device("cpu"),
        "filter": _fixture_filter,
    }

    encoded = json.dumps(payload, default=_json_default)

    restored = json.loads(encoded)
    assert restored["dtype"] == "torch.bfloat16"
    assert restored["device"] == "cpu"
    assert restored["filter"].endswith("._fixture_filter")


class _TokenizerFixture:
    pad_token_id = 0

    def decode(self, token):
        return "<s>" if token == 1 else "word"

    def encode(self, text, add_special_tokens=False):
        inline = [1] if text.startswith("<s>") else []
        return ([1] if add_special_tokens else []) + inline + [2]


def test_native_language_tokenization_matches_harness_bos_and_truncation() -> None:
    stub = SimpleNamespace(
        tokenizer=_TokenizerFixture(),
        add_bos_token=True,
        prefix_token_id=1,
    )
    assert NativeTextLM.tok_encode(stub, "word") == [1, 2]
    assert NativeTextLM.tok_encode(stub, "<s>word") == [1, 2]
    assert NativeTextLM.tok_encode(stub, "word", add_special_tokens=False) == [2]
    assert NativeTextLM.tok_encode(stub, "word", left_truncate_len=1) == [2]


def test_language_loglikelihood_rejects_nonfinite_logits(monkeypatch) -> None:
    stub = SimpleNamespace(
        tokenizer=_TokenizerFixture(),
        batch_size=1,
        max_length=8,
        device=torch.device("cpu"),
        feature_dim=2,
        model=object(),
    )
    monkeypatch.setattr(
        lm_eval_interop,
        "forward_text_only",
        lambda *args, **kwargs: SimpleNamespace(logits=torch.full((1, 1, 8), float("nan"))),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        NativeTextLM._loglikelihood_tokens(stub, [(None, [1], [2])])


def test_native_likelihood_matches_hflm_padding_and_left_truncation(monkeypatch) -> None:
    from lm_eval.models.huggingface import HFLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"<pad>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5, "d": 6, "<unk>": 7},
                unk_token="<unk>",
            )
        ),
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    reference = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device="cpu",
        batch_size=2,
        max_length=8,
        add_bos_token=True,
    )
    stub = SimpleNamespace(
        tokenizer=tokenizer,
        batch_size=2,
        max_length=8,
        device=torch.device("cpu"),
        feature_dim=2,
        model=model,
    )
    monkeypatch.setattr(
        lm_eval_interop,
        "forward_text_only",
        lambda model, ids, **kwargs: model(input_ids=ids, attention_mask=kwargs["attention_mask"]),
    )
    requests = [
        (None, [1, 3, 4], [5, 6]),
        (None, [1, 4, 3, 5], [3]),
        (None, [1, 3, 4, 5] * 3, [3, 4]),
    ]
    actual = NativeTextLM._loglikelihood_tokens(stub, requests)
    expected = reference._loglikelihood_tokens(requests)
    for (actual_ll, actual_greedy), (expected_ll, expected_greedy) in zip(
        actual,
        expected,
        strict=True,
    ):
        assert actual_ll == pytest.approx(expected_ll, abs=1e-5)
        assert actual_greedy == expected_greedy


def test_language_suite_records_per_task_and_total_timing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = LanguageRetentionSuite.model_validate(
        {
            "id": "timing-fixture",
            "harness": {
                "package": "lm-eval",
                "version": "0.4.12",
                "source_commit": "a" * 40,
            },
            "tasks": [
                {
                    "id": "fixture_task",
                    "num_fewshot": 0,
                    "primary_metric": "acc",
                    "filter": "none",
                }
            ],
        }
    )
    monkeypatch.setattr(
        lm_eval_interop,
        "simple_evaluate",
        lambda **_: {"results": {"fixture_task": {"acc,none": 0.5}}},
    )
    clock = iter([10.0, 11.0, 14.0, 16.0])
    monkeypatch.setattr(lm_eval_interop.time, "perf_counter", lambda: next(clock))

    summary = run_language_suite(
        object(), suite, tmp_path / "result", metadata={"kind": "fixture"}, limit=2
    )

    assert summary["timing"]["tasks"]["fixture_task"]["elapsed_seconds"] == 3.0
    assert summary["timing"]["elapsed_seconds"] == 6.0
    assert summary["primary_results"]["fixture_task"] == {
        "metric": "acc",
        "filter": "none",
        "value": 0.5,
    }
    persisted = json.loads((tmp_path / "result" / "summary.json").read_text())
    assert persisted["timing"] == summary["timing"]
