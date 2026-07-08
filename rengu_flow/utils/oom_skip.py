"""CUDA OOM catch-and-continue around training steps (ai-toolkit style)."""

from __future__ import annotations

import torch

from rengu_flow.utils.common import empty_cuda_cache, is_main_process


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)


_OOM_WINDOW = 10  # fixed: act once max_in_window OOMs land within this many training steps


class OomSkipState:
    """Track OOM-skipped steps over a sliding window. Consecutive counting misses the real
    failure mode: OOMs interleaved with good steps (a success used to reset the streak, so it
    never tripped while the run bled skipped steps). Instead, act once ``max_in_window`` OOMs
    fall inside the last ``_OOM_WINDOW`` steps."""

    def __init__(self, max_in_window: int = 3) -> None:
        self.max_in_window = max_in_window
        self.window = _OOM_WINDOW
        self.oom_steps: list[int] = []
        self.total_skips = 0

    def record_skip(self, step: int) -> None:
        """Count this OOM at ``step``. Call BEFORE the banner so it reads N/max for THIS skip."""
        self.oom_steps.append(step)
        self.total_skips += 1

    def reset_window(self) -> None:
        """Drop the window after a swap bump — pre-bump OOMs must not count against the new level."""
        self.oom_steps.clear()

    def recent(self, step: int) -> int:
        """OOMs within the last ``window`` steps (prunes aged-out entries as a side effect)."""
        lo = step - self.window + 1
        self.oom_steps = [s for s in self.oom_steps if s >= lo]
        return len(self.oom_steps)

    def at_limit(self, step: int) -> bool:
        """True once ``max_in_window`` OOMs have hit within the window — bump swap or abort."""
        return self.recent(step) >= self.max_in_window


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
    sink=None,
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

    in_window = state.recent(step) if step is not None else state.total_skips
    if is_main_process():
        print(
            f"# OOM during training step, skipping batch "
            f"{in_window}/{state.max_in_window} in last {state.window} steps #"
        )
        if step is not None:
            print(f"step={step} (skipped, no loss logged)")

    if sink is not None:
        sink.scalar("train/oom_skip", state.total_skips, step or 0)
        sink.scalar("train/oom_in_window", in_window, step or 0)
