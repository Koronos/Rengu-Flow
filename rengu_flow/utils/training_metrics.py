"""Training-step metrics, routed through the tracking sink (aligned with diffusion-pipe train.py)."""

from __future__ import annotations

from typing import Any

import torch


def get_prodigy_d(optimizer) -> float:
    """Average Prodigy adaptive scale `d` across param groups."""
    total = 0.0
    for group in optimizer.param_groups:
        total += group["d"]
    return total / len(optimizer.param_groups)


def get_automagic_lrs(optimizer) -> tuple[torch.Tensor, float]:
    """Per-parameter LRs for Automagic / GenericOptim; returns (tensor, mean)."""
    lrs = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    stacked = torch.stack(lrs)
    return stacked, stacked.mean().item()


def _resolve_lr(optimizer) -> float | None:
    """Applied learning rate of the first param group (the scheduler updates it in place)."""
    try:
        groups = optimizer.param_groups
    except AttributeError:
        return None
    if not groups:
        return None
    lr = groups[0].get("lr")
    return float(lr) if isinstance(lr, (int, float)) else None


def _resolve_grad_norm(model_engine, optimizer) -> float | None:
    """Global grad norm, optimizer-agnostic.

    DeepSpeed computes it whenever ``gradient_clipping`` is set (the repo default is 1.0), so it
    works for any optimizer. Fall back to ``GenericOptim._grad_norm`` for the gradient-release path
    (clipping 0, no engine norm). Returns None when neither is available / meaningful.
    """
    if model_engine is not None:
        try:
            gn = model_engine.get_global_grad_norm()
        except Exception:
            gn = None
        if isinstance(gn, (int, float)) and gn > 0:
            return float(gn)
    inner = getattr(optimizer, "_grad_norm", None)
    return float(inner) if isinstance(inner, (int, float)) else None


def log_training_step(
    *,
    sink: Any,
    optimizer: Any,
    loss: float,
    x_axis: int,
    step: int,
    logging_steps: int,
    is_main: bool,
    model_engine: Any = None,
) -> None:
    """Log scalars (and automagic histogram) on logging_steps boundaries via the tracking sink."""
    if not is_main or step % logging_steps != 0:
        return

    sink.scalar("train/loss", loss, x_axis)

    lr = _resolve_lr(optimizer)
    if lr is not None:
        sink.scalar("train/lr", lr, x_axis)

    grad_norm = _resolve_grad_norm(model_engine, optimizer)
    if grad_norm is not None:
        sink.scalar("train/grad_norm", grad_norm, x_axis)

    opt_name = type(optimizer).__name__
    if opt_name == "Prodigy":
        sink.scalar("train/prodigy_d", get_prodigy_d(optimizer), x_axis)

    if opt_name in ("Automagic", "GenericOptim") and hasattr(optimizer, "_get_lr"):
        lrs, avg_lr = get_automagic_lrs(optimizer)
        if avg_lr > 0:
            sink.histogram("train/automagic_lrs", lrs, x_axis)
            sink.scalar("train/automagic_avg_lr", avg_lr, x_axis)
