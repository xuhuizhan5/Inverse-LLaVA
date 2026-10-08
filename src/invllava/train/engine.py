from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from invllava.config.schema import ResolvedExperiment
from invllava.model.modeling import (
    InverseLLaVAForConditionalGeneration,
    LLaVAReferenceForConditionalGeneration,
)
from invllava.runtime.telemetry import NvidiaSmiMonitor
from invllava.train.checkpoint import CheckpointManager, load_trainable_weights
from invllava.train.metrics import reconcile_metric_history_for_resume
from invllava.train.optimizer import (
    build_optimizer,
    build_scheduler,
    prepare_trainable_parameters,
)
from invllava.train.state import TrainState, capture_rng_state, seed_everything
from invllava.train.tracking import NullTracker, make_tracker


def _encode_batch_images(
    model: InverseLLaVAForConditionalGeneration | LLaVAReferenceForConditionalGeneration,
    image_rows: list[list[Tensor]],
    device: torch.device,
) -> list[list[Tensor]]:
    counts = [len(row) for row in image_rows]
    flat = [image for row in image_rows for image in row]
    if not flat:
        # The official 665K mixture contains text-only conversations. They are
        # valid language-retention examples and must not be silently dropped.
        return [[] for _ in image_rows]
    features = model.encode_images(torch.stack(flat).to(device=device, non_blocking=True))
    result: list[list[Tensor]] = []
    offset = 0
    for count in counts:
        result.append([features[index] for index in range(offset, offset + count)])
        offset += count
    return result


def _gather_rng_states() -> list[dict[str, Any]]:
    local = capture_rng_state()
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("failed to gather every process RNG state")
    return [item for item in gathered if item is not None]


def _all_processes_finite(accelerator: Any, value: Tensor) -> tuple[bool, int]:
    """Return one fail-fast finite decision shared by every training rank."""

    local_finite = (
        torch.isfinite(value.detach())
        .all()
        .to(
            device=accelerator.device,
            dtype=torch.int32,
        )
    )
    finite_count = accelerator.reduce(local_finite, reduction="sum")
    count = int(finite_count.item())
    return count == accelerator.num_processes, count


def _restore_deepspeed_checkpoint(
    *,
    accelerator: Any,
    model: nn.Module,
    source: Path,
) -> TrainState:
    """Restore a compact ZeRO-2 checkpoint over its immutable base models."""

    if not (source / "COMPLETE").is_file():
        raise ValueError(f"incomplete checkpoint: {source}")
    # DeepSpeed's runtime checkpoint contains only the learned partition
    # because save_state excludes immutable Vicuna/CLIP tensors. Validate that
    # partition against the reconstructed model before restoring optimizer,
    # scheduler, and per-rank RNG state. Non-strict module loading leaves the
    # freshly loaded frozen base weights untouched.
    load_trainable_weights(source, accelerator.unwrap_model(model))
    accelerator.load_state(
        str(source / "runtime"),
        load_module_strict=False,
    )
    return CheckpointManager.load_distributed_state(source)


class TrainingEngine:
    def __init__(
        self,
        *,
        config: ResolvedExperiment,
        model: InverseLLaVAForConditionalGeneration | LLaVAReferenceForConditionalGeneration,
        dataloader: DataLoader[Any],
        run_dir: str | Path,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.run_dir = Path(run_dir)
        # PyTorch 2.13 keeps this collective as a compatibility alias while
        # DeepSpeed migrates to ``all_gather_single``. Suppress this one known
        # upstream warning so every optimizer update does not repeat it.
        warnings.filterwarnings(
            "ignore",
            message=(
                r"`torch\.distributed\.all_gather_into_tensor` is deprecated\. "
                r"Please use `torch\.distributed\.all_gather_single` instead\."
            ),
            category=FutureWarning,
            module=r"torch\.distributed\.c10d_logger",
        )
        from accelerate import Accelerator
        from accelerate.utils import DeepSpeedPlugin

        deepspeed_plugin = None
        if self.config.runtime.distributed_strategy == "deepspeed_zero2":
            train_batch_size = (
                self.config.training.per_device_batch_size
                * self.config.training.gradient_accumulation_steps
                * self.config.runtime.num_processes
            )
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config={
                    "bf16": {"enabled": self.config.runtime.mixed_precision == "bf16"},
                    "fp16": {"enabled": self.config.runtime.mixed_precision == "fp16"},
                    "train_micro_batch_size_per_gpu": (self.config.training.per_device_batch_size),
                    "gradient_accumulation_steps": (
                        self.config.training.gradient_accumulation_steps
                    ),
                    "train_batch_size": train_batch_size,
                    "gradient_clipping": self.config.training.max_grad_norm,
                    "zero_optimization": {
                        "stage": 2,
                        "overlap_comm": True,
                        "contiguous_gradients": True,
                        "reduce_scatter": True,
                    },
                    "steps_per_print": 2_000_000,
                }
            )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision=self.config.runtime.mixed_precision,
            deepspeed_plugin=deepspeed_plugin,
            step_scheduler_with_optimizer=False,
        )
        if self.accelerator.num_processes != self.config.runtime.num_processes:
            raise ValueError(
                f"runtime expects {self.config.runtime.num_processes} processes, "
                f"launcher provided {self.accelerator.num_processes}"
            )
        if self.accelerator.is_main_process:
            if resume:
                if not self.run_dir.is_dir():
                    raise FileNotFoundError(f"resume run directory is missing: {self.run_dir}")
            else:
                self.run_dir.mkdir(parents=True, exist_ok=False)
        # A one-process DeepSpeed rehearsal initializes its process group inside
        # ``accelerator.prepare``. There is nothing to synchronize before that
        # point; multi-process launchers already initialize the group here.
        if self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()

    def run(self, *, resume_from: str | Path | None = None) -> TrainState:
        seed_everything(self.config.training.seed)
        accelerator = self.accelerator
        self.model.language_model.model.set_gradient_checkpointing(
            self.config.training.gradient_checkpointing
        )
        optimizer_precision = prepare_trainable_parameters(
            self.model,
            self.config.training.optimizer,
            external_fp32_master=(self.config.runtime.distributed_strategy == "deepspeed_zero2"),
        )
        optimizer = build_optimizer(self.model.parameters(), self.config.training.optimizer)
        projected_batches = math.ceil(len(self.dataloader) / accelerator.num_processes)
        projected_steps_per_epoch = max(
            1,
            math.ceil(projected_batches / self.config.training.gradient_accumulation_steps),
        )
        total_steps = max(1, math.ceil(projected_steps_per_epoch * self.config.training.epochs))
        scheduler = build_scheduler(
            optimizer, self.config.training.optimizer, total_steps=total_steps
        )
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            self.model, optimizer, self.dataloader, scheduler
        )
        actual_steps_per_epoch = max(
            1,
            math.ceil(len(dataloader) / self.config.training.gradient_accumulation_steps),
        )
        actual_total_steps = max(1, math.ceil(actual_steps_per_epoch * self.config.training.epochs))
        if actual_total_steps != total_steps:
            raise RuntimeError(
                "prepared dataloader changed the optimizer-update budget: "
                f"projected={total_steps}, actual={actual_total_steps}"
            )
        state = TrainState()
        checkpoints = CheckpointManager(self.run_dir / "checkpoints")
        if resume_from is not None:
            if self.config.runtime.distributed_strategy == "deepspeed_zero2":
                source = Path(resume_from)
                state = _restore_deepspeed_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    source=source,
                )
            else:
                state = checkpoints.load(
                    resume_from,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    rank=accelerator.process_index,
                )
            if accelerator.is_main_process:
                reconcile_metric_history_for_resume(
                    self.run_dir / "metrics.jsonl",
                    checkpoint_step=state.global_step,
                )
            accelerator.wait_for_everyone()
        session_start_step = state.global_step
        session_start_samples = state.samples_seen
        session_start_tokens = state.tokens_seen
        tracker = (
            make_tracker(
                self.config.runtime.tracker,
                self.run_dir,
                run_id=self.run_dir.name,
                config=self.config.model_dump(mode="json"),
            )
            if accelerator.is_main_process
            else NullTracker()
        )
        monitor: NvidiaSmiMonitor | None = None
        if (
            accelerator.is_main_process
            and self.config.runtime.accelerator == "cuda"
            and self.config.runtime.hardware_telemetry_interval_seconds > 0
        ):
            monitor = NvidiaSmiMonitor(
                self.run_dir / "hardware.jsonl",
                interval_seconds=self.config.runtime.hardware_telemetry_interval_seconds,
            )
            monitor.start()
        milestones = {
            max(1, round(total_steps * value))
            for value in self.config.training.checkpoint_milestones
        }
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        clock = time.perf_counter()
        pending_samples = 0
        pending_tokens = 0
        pending_supervised_tokens = 0
        pending_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        for epoch in range(state.epoch, int(self.config.training.epochs + 0.999999)):
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            elif hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(dataloader):
                if epoch == state.epoch and batch_index < state.batch_in_epoch:
                    continue
                if state.global_step >= total_steps:
                    break
                with accelerator.accumulate(model):
                    with torch.profiler.record_function("invllava::vision_encode"):
                        # The portable optimizer keeps trainable parameters in
                        # FP32.  Controlled LLaVA therefore has an FP32 projector
                        # behind a BF16 vision encoder.  This direct helper call
                        # bypasses Accelerate's wrapped model.forward, so enter
                        # the same autocast policy explicitly.
                        with accelerator.autocast():
                            features = _encode_batch_images(
                                accelerator.unwrap_model(model),
                                batch.pop("pixel_values"),
                                accelerator.device,
                            )
                    sample_ids = batch.pop("sample_ids")
                    with torch.profiler.record_function("invllava::multimodal_forward"):
                        output = model(image_features=features, **batch)
                    if output.loss is None:
                        raise RuntimeError("training forward returned no loss")
                    if output.supervised_token_count is None or output.expanded_token_count is None:
                        raise RuntimeError(
                            "training forward omitted expanded-sequence token counts"
                        )
                    supervised_tokens = int(output.supervised_token_count.item())
                    expanded_tokens = int(output.expanded_token_count.item())
                    loss_is_finite, finite_loss_ranks = _all_processes_finite(
                        accelerator,
                        output.loss,
                    )
                    if not loss_is_finite:
                        raise FloatingPointError(
                            "non-finite training loss; refusing to update or checkpoint; "
                            f"epoch={epoch}, batch={batch_index}, global_step={state.global_step}, "
                            f"finite_ranks={finite_loss_ranks}/{accelerator.num_processes}, "
                            f"sample_ids={sample_ids[:8]}"
                        )
                    pending_samples += len(sample_ids)
                    pending_tokens += expanded_tokens
                    pending_supervised_tokens += supervised_tokens
                    pending_loss += output.loss.detach().float() * supervised_tokens
                    with torch.profiler.record_function("invllava::backward"):
                        accelerator.backward(output.loss)
                    gradient_norm: float | None = None
                    if accelerator.sync_gradients:
                        norm = accelerator.clip_grad_norm_(
                            model.parameters(), self.config.training.max_grad_norm
                        )
                        gradient_norm = (
                            float(norm.detach().float())
                            if isinstance(norm, Tensor)
                            else float(norm)
                        )
                        gradient_is_finite, finite_gradient_ranks = _all_processes_finite(
                            accelerator,
                            torch.as_tensor(gradient_norm, device=accelerator.device),
                        )
                        if not gradient_is_finite:
                            raise FloatingPointError(
                                "non-finite gradient norm; refusing to update or checkpoint; "
                                f"epoch={epoch}, batch={batch_index}, "
                                f"global_step={state.global_step}, "
                                "finite_ranks="
                                f"{finite_gradient_ranks}/{accelerator.num_processes}, "
                                f"sample_ids={sample_ids[:8]}"
                            )
                    optimizer.step()
                    if accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients:
                    continue
                state.global_step += 1
                state.batch_in_epoch = batch_index + 1
                counts = accelerator.reduce(
                    torch.tensor(
                        [pending_samples, pending_tokens, pending_supervised_tokens],
                        device=accelerator.device,
                        dtype=torch.long,
                    ),
                    reduction="sum",
                )
                loss_total = accelerator.reduce(pending_loss, reduction="sum")
                state.samples_seen += int(counts[0].item())
                state.tokens_seen += int(counts[1].item())
                mean_loss = loss_total / counts[2].clamp_min(1)
                pending_samples = 0
                pending_tokens = 0
                pending_supervised_tokens = 0
                pending_loss.zero_()
                elapsed = time.perf_counter() - clock
                if state.global_step % self.config.training.log_every_steps == 0:
                    scale_metrics = {
                        f"fusion_scale/{name}": float(parameter.detach().float())
                        for name, parameter in accelerator.unwrap_model(model).named_parameters()
                        if ".fusion.scales." in name
                    }
                    memory_metrics: dict[str, int] = {}
                    if torch.cuda.is_available():
                        local_memory = torch.tensor(
                            [
                                torch.cuda.max_memory_allocated(),
                                torch.cuda.max_memory_reserved(),
                            ],
                            dtype=torch.long,
                            device=accelerator.device,
                        ).unsqueeze(0)
                        gathered_memory = accelerator.gather(local_memory).reshape(-1, 2)
                        maximum_memory = gathered_memory.max(dim=0).values
                        memory_metrics = {
                            "train/peak_allocated_bytes": int(maximum_memory[0].item()),
                            "train/peak_reserved_bytes": int(maximum_memory[1].item()),
                        }
                    tracker.log(
                        {
                            "train/loss": float(mean_loss),
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/gradient_norm": gradient_norm,
                            "train/samples_seen": state.samples_seen,
                            "train/tokens_seen": state.tokens_seen,
                            "train/supervised_tokens_in_update": int(counts[2].item()),
                            "train/session_wall_seconds": elapsed,
                            "train/steps_per_second": (state.global_step - session_start_step)
                            / max(elapsed, 1e-9),
                            "train/examples_per_second": (
                                state.samples_seen - session_start_samples
                            )
                            / max(elapsed, 1e-9),
                            "train/expanded_tokens_per_second": (
                                state.tokens_seen - session_start_tokens
                            )
                            / max(elapsed, 1e-9),
                            **scale_metrics,
                            **memory_metrics,
                        },
                        state.global_step,
                    )
                should_save = (
                    state.global_step in milestones
                    or state.global_step % self.config.training.save_every_steps == 0
                )
                if should_save:
                    accelerator.wait_for_everyone()
                    checkpoint_name = f"step-{state.global_step:07d}"
                    metadata = {
                        "experiment_id": self.config.id,
                        "total_steps": total_steps,
                        "optimizer_precision": optimizer_precision,
                        "metric_token_accounting": "expanded-causal-v2",
                        "runtime_state_format": (
                            "accelerate-deepspeed-zero2"
                            if self.config.runtime.distributed_strategy == "deepspeed_zero2"
                            else "portable-pytorch"
                        ),
                    }
                    if self.config.runtime.distributed_strategy == "deepspeed_zero2":
                        checkpoint_path = checkpoints.root / checkpoint_name
                        if accelerator.is_main_process:
                            checkpoints.begin_distributed(name=checkpoint_name)
                        accelerator.wait_for_everyone()
                        accelerator.save_state(
                            str(checkpoint_path / "runtime"),
                            safe_serialization=True,
                            exclude_frozen_parameters=True,
                        )
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            checkpoints.finalize_distributed(
                                path=checkpoint_path,
                                model=accelerator.unwrap_model(model),
                                model_state=accelerator.get_state_dict(model, unwrap=False),
                                state=state,
                                metadata=metadata,
                            )
                    elif accelerator.is_main_process:
                        checkpoints.save(
                            name=checkpoint_name,
                            model=accelerator.unwrap_model(model),
                            optimizer=optimizer,
                            scheduler=scheduler,
                            state=state,
                            metadata=metadata,
                            rng_states=_gather_rng_states(),
                        )
                    elif torch.distributed.is_available() and torch.distributed.is_initialized():
                        _gather_rng_states()
                    if accelerator.is_main_process:
                        checkpoints.prune_unprotected(
                            keep_last=self.config.runtime.keep_last_checkpoints,
                            protected={f"step-{step:07d}" for step in milestones},
                        )
                    accelerator.wait_for_everyone()
            state.epoch = epoch + 1
            state.batch_in_epoch = 0
            if state.global_step >= total_steps:
                break
        if monitor is not None:
            monitor.close()
        tracker.close()
        accelerator.wait_for_everyone()
        return state
