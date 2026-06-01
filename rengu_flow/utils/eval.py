"""Evaluation over timestep quantiles; metrics to TensorBoard and optional WandB."""

from __future__ import annotations

import random
import time
from typing import Any

import torch
import numpy as np

from rengu_flow.utils.common import get_rank, is_main_process, empty_cuda_cache
from rengu_flow.utils.isolate_rng import isolate_rng
from rengu_flow.utils.pipeline import get_data_iterator_for_step

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def evaluate_single(
    model_engine,
    eval_dataloader,
    eval_gradient_accumulation_steps: int,
    quantile: float,
    pbar: Any = None,
) -> float:
    """Run eval over one full pass of eval_dataloader at a fixed timestep quantile; return mean loss."""
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0.0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(
            eval_dataloader,
            model_engine,
            num_micro_batches=eval_gradient_accumulation_steps,
        )
        loss = model_engine.eval_batch(
            iterator, num_micro_batches=eval_gradient_accumulation_steps
        ).item()
        eval_dataloader.sync_epoch()
        if pbar is not None:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break
    eval_dataloader.reset()
    return total_loss / count


def _evaluate(
    model_engine,
    eval_dataloaders: dict[str, Any],
    step: int,
    eval_gradient_accumulation_steps: int,
    tb_writer: Any,
    wandb_enable: bool = False,
) -> None:
    """Run evaluate_single per dataset and per quantile; log to TensorBoard and optionally WandB."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    pbar_total = 0
    for dl in eval_dataloaders.values():
        pbar_total += (
            len(dl) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
        )
    if is_main_process() and tqdm is not None:
        print("Running eval")
        import sys

        pbar = tqdm(total=pbar_total, disable=not sys.stderr.isatty())
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(
                model_engine,
                eval_dataloader,
                eval_gradient_accumulation_steps,
                quantile,
                pbar=pbar,
            )
            losses.append(loss)
            if is_main_process():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"{name}/loss_quantile_{quantile:.2f}", loss, step)
                if wandb_enable:
                    _wandb_log({f"{name}/loss_quantile_{quantile:.2f}": loss, "step": step})
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            if tb_writer is not None:
                tb_writer.add_scalar(f"{name}/loss", avg_loss, step)
            if wandb_enable:
                _wandb_log({f"{name}/loss": avg_loss, "step": step})

    duration = time.time() - start
    if is_main_process():
        if tb_writer is not None:
            tb_writer.add_scalar("eval/eval_time_sec", duration, step)
        if wandb_enable:
            _wandb_log({"eval/eval_time_sec": duration, "step": step})
        if pbar is not None:
            pbar.close()


def _wandb_log(metrics: dict[str, Any]) -> None:
    try:
        import wandb
        wandb.log(metrics)
    except ImportError:
        pass


def evaluate(
    model,
    model_engine,
    eval_dataloaders: dict[str, Any],
    tb_writer: Any,
    step: int,
    eval_gradient_accumulation_steps: int,
    disable_block_swap: bool,
    optimizer: Any = None,
    wandb_enable: bool = False,
) -> None:
    """Run evaluation with block-swap inference setup, isolated RNG, then restore training state."""
    if len(eval_dataloaders) == 0:
        return

    if optimizer is not None and hasattr(optimizer, "eval") and callable(optimizer.eval):
        optimizer.eval()

    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _evaluate(
            model_engine,
            eval_dataloaders,
            step,
            eval_gradient_accumulation_steps,
            tb_writer,
            wandb_enable=wandb_enable,
        )
    empty_cuda_cache()
    model.prepare_block_swap_training()

    if optimizer is not None and hasattr(optimizer, "train") and callable(optimizer.train):
        optimizer.train()
