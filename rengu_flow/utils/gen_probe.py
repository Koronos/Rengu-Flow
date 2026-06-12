"""Deterministic generalization probe: held-out val loss, a matched train probe, and the
train-val GAP — the fast overfitting signal (rising gap = overfitting).

Why this exists: train loss alone is misleading for diffusion fine-tuning — a model can drive
train loss down while memorizing and producing worse samples. The recognized cheap signal
(EveryDream2 / kohya / OneTrainer) is a held-out validation loss, and especially the
TRAIN-VAL GAP. This probe is forward-only (no sampling/generation) and deterministic so the
curves are smooth and comparable across steps.

Determinism: we reuse the existing per-quantile eval path (``evaluate_single``), which fixes
the timestep per pass via ``set_eval_quantile`` and averages over a fixed spread of quantiles
(``TIMESTEP_QUANTILES_FOR_EVAL``) — i.e. ``t`` is spread across the schedule, frozen per eval.
The per-item noise ``eps`` (drawn inside ``model.prepare_inputs``) is frozen by reseeding the
RNG to a fixed seed under ``isolate_rng()`` before every probe, so the same items get the same
noise on every eval. The loss is the SAME loss the trainer uses (``model.get_loss_fn`` wired
into the pipeline engine) — eps-pred / v-pred / flow-matching is whatever the model defines; we
never reinvent it.
"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch

from rengu_flow.utils.common import empty_cuda_cache, get_rank, is_main_process
from rengu_flow.utils.eval import TIMESTEP_QUANTILES_FOR_EVAL, evaluate_single
from rengu_flow.utils.isolate_rng import isolate_rng


def _deterministic_loss(
    model_engine,
    dataloader,
    eval_gradient_accumulation_steps: int,
    max_batches: int | None,
    seed: int,
) -> float:
    """Mean forward-only loss over a fixed timestep spread with frozen per-item noise.

    Reseeds before each quantile pass so ``eps`` is identical across eval calls (smooth,
    comparable curves). Returns the mean over ``TIMESTEP_QUANTILES_FOR_EVAL``.
    """
    losses = []
    for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
        # Freeze eps per item: same seed every eval → same randn sequence in prepare_inputs.
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        losses.append(
            evaluate_single(
                model_engine,
                dataloader,
                eval_gradient_accumulation_steps,
                quantile,
                max_batches=max_batches,
            )
        )
    return sum(losses) / len(losses)


def generalization_probe(
    model,
    model_engine,
    val_dataloader: Any,
    train_probe_dataloader: Any,
    sink: Any,
    step: int,
    eval_gradient_accumulation_steps: int,
    disable_block_swap: bool,
    *,
    probe_batches: int | None = None,
    optimizer: Any = None,
) -> dict[str, float] | None:
    """Run the deterministic generalization probe and log ``val/loss``, ``train/probe`` and
    ``val/gap`` (= val − train_probe) via the tracking sink.

    Forward-only, ``torch.no_grad`` + isolated RNG, model put in inference (block-swap) state
    and restored to training after. No-ops gracefully if no val dataloader is available.
    Returns ``{"val_loss", "train_probe", "val_gap"}`` (rank 0) for UI surfacing, else ``None``.
    """
    if val_dataloader is None:
        return None

    if optimizer is not None and hasattr(optimizer, "eval") and callable(optimizer.eval):
        optimizer.eval()

    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    result: dict[str, float] | None = None
    start = time.time()
    try:
        with torch.no_grad(), isolate_rng():
            seed = get_rank()
            val_loss = _deterministic_loss(
                model_engine,
                val_dataloader,
                eval_gradient_accumulation_steps,
                probe_batches,
                seed,
            )
            train_probe = None
            if train_probe_dataloader is not None:
                train_probe = _deterministic_loss(
                    model_engine,
                    train_probe_dataloader,
                    eval_gradient_accumulation_steps,
                    probe_batches,
                    seed,
                )
    finally:
        empty_cuda_cache()
        model.prepare_block_swap_training()
        if optimizer is not None and hasattr(optimizer, "train") and callable(optimizer.train):
            optimizer.train()

    duration = time.time() - start
    if is_main_process():
        metrics: dict[str, float] = {"val_loss": val_loss}
        sink.scalar("val/loss", val_loss, step)
        if train_probe is not None:
            gap = val_loss - train_probe
            metrics["train_probe"] = train_probe
            metrics["val_gap"] = gap
            sink.scalar("train/probe", train_probe, step)
            sink.scalar("val/gap", gap, step)
        sink.scalar("val/probe_time_sec", duration, step)
        result = metrics
    return result
