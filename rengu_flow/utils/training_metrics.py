"""Training-step metrics for TensorBoard and WandB (aligned with diffusion-pipe train.py)."""

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


def log_training_step(
    *,
    tb_writer: Any,
    wandb_enable: bool,
    optimizer: Any,
    loss: float,
    x_axis: int,
    step: int,
    logging_steps: int,
    is_main: bool,
) -> None:
    """Log scalars (and automagic histogram) on logging_steps boundaries."""
    if not is_main or step % logging_steps != 0:
        return

    if tb_writer is not None:
        tb_writer.add_scalar("train/loss", loss, x_axis)
    _wandb_log(wandb_enable, {"train/loss": loss, "step": x_axis})

    if hasattr(optimizer, "_grad_norm"):
        if tb_writer is not None:
            tb_writer.add_scalar("train/grad_norm", optimizer._grad_norm, x_axis)
        _wandb_log(wandb_enable, {"train/grad_norm": optimizer._grad_norm, "step": x_axis})

    opt_name = type(optimizer).__name__
    if opt_name == "Prodigy":
        prodigy_d = get_prodigy_d(optimizer)
        if tb_writer is not None:
            tb_writer.add_scalar("train/prodigy_d", prodigy_d, x_axis)
        _wandb_log(wandb_enable, {"train/prodigy_d": prodigy_d, "step": x_axis})

    if opt_name in ("Automagic", "GenericOptim") and hasattr(optimizer, "_get_lr"):
        lrs, avg_lr = get_automagic_lrs(optimizer)
        if avg_lr > 0:
            if tb_writer is not None:
                tb_writer.add_histogram("train/automagic_lrs", lrs, x_axis)
                tb_writer.add_scalar("train/automagic_avg_lr", avg_lr, x_axis)
            _wandb_log(
                wandb_enable,
                {"train/automagic_avg_lr": avg_lr, "step": x_axis},
            )


def _wandb_log(wandb_enable: bool, metrics: dict[str, Any]) -> None:
    if not wandb_enable:
        return
    try:
        import wandb

        wandb.log(metrics)
    except ImportError:
        pass
