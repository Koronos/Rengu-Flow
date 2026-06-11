"""CUDA OOM catch-and-continue around training steps (ai-toolkit style)."""

from __future__ import annotations

import torch

from rengu_flow.utils.common import empty_cuda_cache, is_main_process


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)


class OomSkipState:
    def __init__(self, max_consecutive: int = 3) -> None:
        self.max_consecutive = max_consecutive
        self.consecutive = 0
        self.total_skips = 0

    def record_success(self) -> None:
        self.consecutive = 0

    def record_skip(self) -> None:
        self.consecutive += 1
        self.total_skips += 1
        if self.consecutive > self.max_consecutive:
            raise RuntimeError(
                f"OOM during training step {self.max_consecutive} times in a row, aborting training"
            )


def reset_engine_timers(model_engine) -> None:
    """Reset DeepSpeed engine timers after train_batch aborted mid-step.

    An OOM inside train_batch leaves TRAIN_BATCH_TIMER (and, with
    wall_clock_breakdown, the micro-step timers) started; the retrying
    train_batch would then die on "timer has already been started".
    """
    group = getattr(model_engine, "timers", None)
    timers = getattr(group, "timers", None)
    if isinstance(timers, dict):
        for timer in timers.values():
            timer.reset()


def handle_oom_skip(
    state: OomSkipState,
    model_engine,
    *,
    clear_cache: bool = True,
    step: int | None = None,
    tb_writer=None,
) -> None:
    """Zero gradients and optionally clear CUDA cache after an OOM skip."""
    if hasattr(model_engine, "zero_grad"):
        model_engine.zero_grad()
    elif getattr(model_engine, "optimizer", None) is not None:
        model_engine.optimizer.zero_grad(set_to_none=True)
    reset_engine_timers(model_engine)

    if clear_cache and torch.cuda.is_available():
        empty_cuda_cache()
        torch.cuda.ipc_collect()

    if is_main_process():
        banner = (
            f"# OOM during training step, skipping batch {state.consecutive}/"
            f"{state.max_consecutive} #"
        )
        print(banner)
        if step is not None:
            print(f"step={step} (skipped, no loss logged)")

    if tb_writer is not None:
        tb_writer.add_scalar("train/oom_skip", state.total_skips, step or 0)
        tb_writer.add_scalar("train/consecutive_oom", state.consecutive, step or 0)
