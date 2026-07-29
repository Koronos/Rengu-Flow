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

    Whenever ``gradient_clipping`` is set (repo default 1.0), DeepSpeed computes the norm while
    clipping; ``install_grad_norm_capture`` keeps it on the engine, so this works for ANY optimizer
    on that path (AdamW, Prodigy, GenericOptim, custom). The ``optimizer._grad_norm`` fallback only
    catches an optimizer that sets it directly with clipping off. The gradient-release path has no
    grad norm: clipping is 0 (no engine norm) and grads are freed per-param before a global pass,
    so it returns None there. Returns None when nothing is available.
    """
    if model_engine is not None:
        try:
            gn = model_engine.get_global_grad_norm()
        except Exception:
            gn = None
        if torch.is_tensor(gn):
            if gn.numel() == 1:
                value = float(gn.detach().item())
                return value if value > 0 else None
        elif isinstance(gn, (int, float)) and gn > 0:
            return float(gn)
    inner = getattr(optimizer, "_grad_norm", None)
    return float(inner) if isinstance(inner, (int, float)) else None


def install_grad_norm_capture(model_engine, _clip_fn=None) -> None:
    """Keep the global grad norm DeepSpeed computes while clipping, so train/grad_norm can log.

    On the bf16/fp32 optimizer path DeepSpeed's ``clip_fp32_gradients()`` clips in place and throws
    away the norm, leaving ``get_global_grad_norm()`` at None for AdamW/Prodigy — the UI chart stays
    empty. Re-wrap that clip to store the value it already computes; no extra passes over the grads.
    No-op when clipping is off (gradient-release path); ``GenericOptim._grad_norm`` covers that.

    ``_clip_fn`` is a test seam: defaults to DeepSpeed's ``clip_grad_norm_``, imported lazily so the
    ~17s deepspeed import stays out of tests.
    """
    if model_engine is None or not hasattr(model_engine, "clip_fp32_gradients"):
        return
    if model_engine.gradient_clipping() <= 0:
        return

    def _clip_and_capture():
        clip = _clip_fn
        if clip is None:
            from deepspeed.runtime.utils import clip_grad_norm_ as clip
        total_norm = clip(
            parameters=model_engine.module.parameters(),
            max_norm=model_engine.gradient_clipping(),
            mpu=model_engine.mpu,
        )
        # Preserve device scalars so clipping does not introduce a cudaStreamSynchronize before
        # optimizer.step(). The logging boundary resolves it to a float after the loss sync.
        model_engine._global_grad_norm = (
            total_norm.detach() if torch.is_tensor(total_norm) else float(total_norm)
        )

    model_engine.clip_fp32_gradients = _clip_and_capture


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

    # Human-readable per-step line to stdout (every logging_steps). Not a @@RFPROG@@ marker, so
    # the web UI does not strip it — this is what shows in the live log panel (the throttled
    # marker only drives the progress bar/metrics). Raise logging_steps to log less often.
    parts = [f"step {step}", f"loss {loss:.4f}"]
    if lr is not None:
        parts.append(f"lr {lr:.2e}")
    if grad_norm is not None:
        parts.append(f"grad_norm {grad_norm:.3f}")
    print(" | ".join(parts), flush=True)

    opt_name = type(optimizer).__name__
    if opt_name == "Prodigy":
        sink.scalar("train/prodigy_d", get_prodigy_d(optimizer), x_axis)

    if opt_name in ("Automagic", "GenericOptim") and hasattr(optimizer, "_get_lr"):
        lrs, avg_lr = get_automagic_lrs(optimizer)
        if avg_lr > 0:
            sink.histogram("train/automagic_lrs", lrs, x_axis)
            sink.scalar("train/automagic_avg_lr", avg_lr, x_axis)
