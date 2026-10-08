"""Pinned lm-evaluation-harness boundary for text-only retention tests.

This optional module is imported only by ``invllava language-eval``. Keeping the
harness outside the core dependency set prevents its task registry and transitive
packages from changing training or multimodal evaluation environments.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from lm_eval import simple_evaluate, utils
from lm_eval.api.model import TemplateLM
from lm_eval.models.huggingface import HFLM

from invllava.config.loader import ConfigRepository
from invllava.config.validation import require_frozen_execution
from invllava.eval.language_suite import LanguageRetentionSuite
from invllava.model.loaders import build_language_model, load_tokenizer
from invllava.model.text import forward_text_only, generate_text_only
from invllava.runtime.optimization import configure_model_kernels
from invllava.train.checkpoint import load_component_weights


class NativeTextLM(TemplateLM):
    """Expose a native Inverse or controlled-LLaVA no-image language path."""

    backend = "causal"

    def __init__(
        self,
        *,
        experiment: str,
        checkpoint: str,
        config_root: str = "configs",
        runtime_ref: str | None = None,
        device: str = "cuda",
        batch_size: int = 4,
        max_length: int | None = None,
        max_gen_toks: int = 256,
        local_files_only: bool = False,
        add_bos_token: bool = True,
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        resolved = ConfigRepository(config_root).resolve(experiment, runtime_ref=runtime_ref)
        require_frozen_execution(resolved)
        model, _ = build_language_model(
            resolved.model.model_copy(update={"torch_dtype": dtype}),
            attention_backend=resolved.runtime.attention_backend,
            local_files_only=local_files_only,
        )
        load_component_weights(checkpoint, model, prefix="language_model")
        self._device = torch.device(device)
        self.model = model.to(self._device)
        report = configure_model_kernels(model, resolved.runtime.kernel_optimization)
        self.model.eval()
        self.kernel_optimization = report.to_dict()
        self.tokenizer = load_tokenizer(
            resolved.model,
            local_files_only=local_files_only,
        )
        self.batch_size = batch_size
        self.add_bos_token = add_bos_token
        self.max_length = max_length or resolved.model.language.max_length
        self.max_gen_toks = max_gen_toks
        self.feature_dim = resolved.model.vision.feature_dim
        self._tokenizer_name = (
            f"{resolved.model.language.checkpoint}@{resolved.model.language.revision}"
        )

    @property
    def eot_token_id(self) -> int:
        value = self.tokenizer.eos_token_id
        if value is None:
            raise ValueError("tokenizer has no EOS token")
        return int(value)

    @property
    def prefix_token_id(self) -> int:
        value = self.tokenizer.bos_token_id
        return self.eot_token_id if value is None else int(value)

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        # Reuse the pinned harness boundary, including duplicate-BOS handling.
        return HFLM.tok_encode(
            self,
            string,
            add_special_tokens=add_special_tokens,
            left_truncate_len=left_truncate_len,
            **kwargs,
        )

    def tok_decode(self, tokens: int | list[int], **kwargs: Any) -> str:
        return str(self.tokenizer.decode(tokens, **kwargs))

    @torch.inference_mode()
    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        **_: Any,
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        pad_token = int(self.tokenizer.pad_token_id)
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start : start + self.batch_size]
            rows: list[list[int]] = []
            continuations: list[list[int]] = []
            for _, context, continuation in chunk:
                if not context or not continuation:
                    raise ValueError("lm-eval loglikelihood requires context and continuation")
                if len(continuation) > self.max_length:
                    raise ValueError("continuation exceeds the configured context length")
                combined = (context + continuation)[-(self.max_length + 1) :]
                rows.append(combined[:-1])
                continuations.append(continuation)
            width = max(map(len, rows))
            input_ids = torch.full(
                (len(rows), width), pad_token, dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for row_index, row in enumerate(rows):
                input_ids[row_index, : len(row)] = torch.tensor(row, device=self.device)
                attention_mask[row_index, : len(row)] = True
            output = forward_text_only(
                self.model,
                input_ids,
                feature_dim=self.feature_dim,
                attention_mask=attention_mask,
            )
            for row_index, (row, continuation) in enumerate(zip(rows, continuations, strict=True)):
                length = len(continuation)
                logits = output.logits[row_index, len(row) - length : len(row)].float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("non-finite language-evaluation logits")
                target = torch.tensor(continuation, dtype=torch.long, device=self.device)
                log_prob = functional.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1))
                result = (
                    float(log_prob.sum()),
                    bool(torch.equal(logits.argmax(dim=-1), target)),
                )
                results.append(result)
                request_key = chunk[row_index][0]
                if request_key is not None:
                    self.cache_hook.add_partial("loglikelihood", request_key, result)
        return results

    def loglikelihood_rolling(self, requests: list[Any], disable_tqdm: bool = False) -> list[float]:
        del disable_tqdm
        totals: list[float] = []
        for request in requests:
            (text,) = request.args
            windows = [
                (None, context, continuation)
                for context, continuation in map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(text),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            ]
            total = sum(value for value, _ in self._loglikelihood_tokens(windows))
            totals.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), total)
        return totals

    @torch.inference_mode()
    def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
        del disable_tqdm
        results: list[str] = []
        supported = {"until", "max_gen_toks", "do_sample", "temperature", "top_p"}
        for request in requests:
            context, generation = request.args
            unknown = set(generation).difference(supported)
            if unknown:
                raise ValueError(f"unsupported generation arguments: {sorted(unknown)}")
            if generation.get("do_sample", False):
                raise ValueError("language-retention evaluation must be deterministic")
            max_new_tokens = int(generation.get("max_gen_toks", self.max_gen_toks))
            if max_new_tokens >= self.max_length:
                raise ValueError("max_gen_toks must be smaller than max_length")
            tokens = self.tok_encode(context)[-(self.max_length - max_new_tokens) :]
            if not tokens:
                tokens = [self.prefix_token_id]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            generated = generate_text_only(
                self.model,
                input_ids,
                feature_dim=self.feature_dim,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            text = self.tok_decode(generated[0].tolist(), skip_special_tokens=True)
            stops = generation.get("until", [])
            stops = [stops] if isinstance(stops, str) else list(stops)
            stop_positions = [position for stop in stops if (position := text.find(stop)) >= 0]
            if stop_positions:
                text = text[: min(stop_positions)]
            results.append(text)
            self.cache_hook.add_partial("generate_until", (context, generation), text)
        return results


class ReleaseTextLM(NativeTextLM):
    """Expose the language pathway from a checksum-verified public release."""

    def __init__(
        self,
        *,
        model: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 4,
        max_length: int | None = None,
        max_gen_toks: int = 256,
        local_files_only: bool = False,
        add_bos_token: bool = True,
    ) -> None:
        TemplateLM.__init__(self)
        from invllava.release import load_pretrained

        release = load_pretrained(
            model,
            revision=revision,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            attention_backend="sdpa",
            generation_cache="kv",
            lora_execution="unmerged",
            max_new_tokens=max_gen_toks,
            local_files_only=local_files_only,
        )
        self._device = torch.device(device)
        self.model = release.model.language_model
        self.tokenizer = release.tokenizer
        self.batch_size = batch_size
        self.add_bos_token = add_bos_token
        self.max_length = max_length or release.model_spec.language.max_length
        self.max_gen_toks = max_gen_toks
        self.feature_dim = release.model_spec.vision.feature_dim
        self._tokenizer_name = (
            f"{release.model_spec.language.checkpoint}@{release.model_spec.language.revision}"
        )
        self.checkpoint_sha256 = release.checkpoint_sha256
        self.release_root = release.release_root
        self.release_revision = release.release_revision


def build_reference_text_lm(
    *,
    kind: str,
    checkpoint: str,
    revision: str,
    device: str,
    dtype: str,
    batch_size: int,
    max_length: int | None,
    local_files_only: bool = False,
    add_bos_token: bool = True,
) -> HFLM:
    if revision in {"", "main", "pending-freeze"}:
        raise ValueError("reference evaluation requires an immutable checkpoint revision")
    if kind == "hf-causal":
        return HFLM(
            pretrained=checkpoint,
            revision=revision,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_length=max_length,
            add_bos_token=add_bos_token,
            backend="causal",
            trust_remote_code=False,
            use_fast_tokenizer=False,
            local_files_only=local_files_only,
        )
    if kind != "hf-llava":
        raise ValueError(f"unknown reference kind: {kind}")
    from transformers import AutoTokenizer, LlavaForConditionalGeneration

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    full_model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        revision=revision,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=revision,
        use_fast=False,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token
    return HFLM(
        pretrained=full_model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=batch_size,
        max_length=max_length,
        add_bos_token=add_bos_token,
        revision=revision,
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, (set, tuple, Path)):
        return list(value) if not isinstance(value, Path) else str(value)
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    # lm-eval's pinned serializer uses str() as its final fallback. Retaining
    # that boundary keeps task metadata writable as the harness adds types.
    return str(value)


def run_language_suite(
    model: TemplateLM,
    suite: LanguageRetentionSuite,
    output_dir: str | Path,
    *,
    metadata: dict[str, Any],
    limit: int | None = None,
    task_ids: set[str] | None = None,
) -> dict[str, Any]:
    suite_started = time.perf_counter()
    if limit is not None and limit <= 0:
        raise ValueError("language-evaluation limit must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    configured = {task.id for task in suite.tasks}
    selected = configured if task_ids is None else set(task_ids)
    unknown = selected.difference(configured)
    if unknown:
        raise ValueError(f"language suite has no configured tasks: {sorted(unknown)}")
    if not selected:
        raise ValueError("language evaluation requires at least one selected task")
    summary: dict[str, Any] = {
        "suite": suite.model_dump(mode="json"),
        "model": metadata,
        "scope": "full" if limit is None else "canary",
        "limit_per_task": limit,
        "selected_tasks": [task.id for task in suite.tasks if task.id in selected],
    }
    (output / "run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    task_results: dict[str, Any] = {}
    primary_results: dict[str, dict[str, Any]] = {}
    task_timings: dict[str, dict[str, float]] = {}
    for task in suite.tasks:
        if task.id not in selected:
            continue
        task_started = time.perf_counter()
        result = simple_evaluate(
            model=model,
            tasks=[task.id],
            num_fewshot=task.num_fewshot,
            apply_chat_template=suite.apply_chat_template,
            fewshot_as_multiturn=suite.fewshot_as_multiturn,
            log_samples=True,
            random_seed=suite.seed,
            numpy_random_seed=suite.seed,
            torch_random_seed=suite.seed,
            fewshot_random_seed=suite.seed,
            limit=limit,
        )
        if result is None:
            raise RuntimeError(f"lm-eval returned no result for {task.id}")
        task_timings[task.id] = {
            "elapsed_seconds": time.perf_counter() - task_started,
        }
        (output / f"{task.id}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        metrics = result.get("results", {}).get(task.id)
        if metrics is None and task.id == "mmlu":
            metrics = {
                key: value for key, value in result.get("groups", {}).items() if key == "mmlu"
            }
        if not isinstance(metrics, dict):
            raise RuntimeError(f"lm-eval returned invalid metrics for {task.id}")
        primary_key = f"{task.primary_metric},{task.filter}"
        if primary_key not in metrics:
            raise RuntimeError(
                f"lm-eval result for {task.id} has no frozen primary metric {primary_key}"
            )
        task_results[task.id] = metrics
        primary_results[task.id] = {
            "metric": task.primary_metric,
            "filter": task.filter,
            "value": metrics[primary_key],
        }
    summary["results"] = task_results
    summary["primary_results"] = primary_results
    summary["timing"] = {
        "elapsed_seconds": time.perf_counter() - suite_started,
        "tasks": task_timings,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return summary
