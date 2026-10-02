"""Pure decisions of the training loop (step budget, seeding, resume, OOM agreement).

``main._run_training`` is a long, torch/DeepSpeed-bound function; the choices it makes about
*when* things happen (where the LR schedule ends, which seed a rank gets, whether a resumed run
is already finished, ...) live here instead so they are unit-tested without a GPU. Nothing in
this module imports torch at module level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def step_budget(max_steps: int | None, total_steps: int) -> int:
    """Step at which the loop actually stops: ``max_steps`` if set (it wins), else the
    epochs-derived ``total_steps``. The LR schedule horizon, the WSD decay onset and the
    resolution schedule are all measured against this one number, with or without a
    resolution schedule, so the LR tail lands exactly where the run ends."""
    return max_steps if max_steps is not None else total_steps


def data_parallel_size(world_size: int, pipeline_stages: int) -> int:
    """Replicas that each see different data: world size divided by pipeline stages.

    Every pipeline stage of one replica consumes the *same* batch, so the global batch is
    ``micro_batch * grad_accum * data_parallel_size`` — not times the full world size."""
    return max(1, int(world_size) // max(1, int(pipeline_stages)))


def examples_to_steps(key: str, n_examples: int, global_batch_size: int) -> int:
    """Convert an ``*_every_n_examples`` cadence to optimizer steps.

    Raises if the cadence is smaller than one global batch: the integer division would give 0
    steps, which either divides by zero (``step % 0``) or silently disables the feature."""
    steps = int(n_examples) // max(1, int(global_batch_size))
    if steps < 1:
        raise ValueError(
            f"{key} = {n_examples} is smaller than the global batch ({global_batch_size} examples "
            f"per optimizer step); it would convert to 0 steps. Use at least {global_batch_size}."
        )
    return steps


def rank_seed(train_seed: int, rank: int) -> int:
    """Per-rank seed: ``train_seed + rank`` so ranks draw different stochastic streams while the
    run stays reproducible for a fixed ``train_seed`` and topology."""
    return int(train_seed) + int(rank)


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (+CUDA) global RNGs."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wsd_fork_step_for_resume(onset: int | None, resumed_step: int) -> int | None:
    """WSD fork step to arm for this process, given the step the loop (re)starts at.

    The fork ('predecay' checkpoint) is saved once the loop reaches the decay onset. If the
    resumed checkpoint is already at/after the onset (``resumed_step - 1 >= onset``) the fork was
    written by the earlier process; saving again would overwrite the protected pre-decay
    checkpoint with a post-onset state. Returns ``None`` when it must not fire."""
    if onset is None or onset <= 0:
        return None
    if resumed_step - 1 >= onset:
        return None
    return onset


def run_already_complete(step: int, budget: int) -> bool:
    """True when a (resumed) run has already consumed its whole step budget: the next step
    to run is beyond it, so the loop must not train (or re-export) anything."""
    return step > budget


def can_emergency_save(world_size: int) -> bool:
    """Whether the best-effort emergency checkpoint is safe in this process.

    ``save_checkpoint`` is collective (barriers, sharded state). When one rank fails while its
    peers are elsewhere, entering it from a single rank deadlocks NCCL and turns a clean crash
    into a hang. Only a single-process run can do it unilaterally."""
    return int(world_size) <= 1


def force_constant_lr_enabled(config: dict[str, Any]) -> bool:
    """``force_constant_lr`` is a boolean switch: only a true value pins the LR on resume
    (``force_constant_lr = false`` behaves like the key being absent)."""
    return bool(config.get("force_constant_lr", False))


@dataclass
class ResumeState:
    """Where the loop restarts after (attempting) a checkpoint resume."""

    step: int = 1
    examples: int = 0
    epoch: int = 1
    resumed: bool = False


def restore_training_state(
    *,
    model_engine: Any,
    optimizer: Any,
    run_dir: str,
    resume_tag: str | None,
    reset_optimizer: bool,
    reset_dataloader: bool,
    train_dataloader: Any,
    training_ema: Any,
    parameters_to_train: list,
    epoch_schedule: Any,
    steps_per_epoch: int,
    global_batch_size: int,
    config: dict[str, Any],
    is_main: bool,
    barrier: Callable[[], None],
    log: Callable[..., None] = print,
) -> ResumeState:
    """Load the checkpoint in ``run_dir`` and compute the restart position.

    Order matters and is covered by tests: engine/optimizer load -> option re-application ->
    dataloader state -> **global RNG restore** -> EMA (re-seeded from the loaded weights when the
    checkpoint has no ``ema.pt``) -> step/examples/epoch -> LR-scheduler fast-forward.
    Nothing after this may re-seed the global RNGs (that used to undo the RNG restore).

    LR scheduler: its saved state is deliberately NOT loaded (the current config is
    authoritative); instead the fresh scheduler is stepped ``step - 1`` times. That equals the
    uninterrupted run because the loop advances the scheduler exactly once per step, including
    OOM-skipped steps (see ``main`` — skipped steps step the scheduler manually).
    """
    from rengu_flow.optim.param_groups import (
        configured_group_keys,
        reapply_param_group_options,
        snapshot_param_group_options,
    )
    from rengu_flow.training.ema import load_ema_checkpoint
    from rengu_flow.utils.rng_state import restore_rng_state

    # The current config is authoritative on resume. The optimizer's *state* (moments) is
    # restored from the checkpoint, but its hyperparameters (LR, betas, weight_decay, per-group
    # options) follow the freshly-built config. --reset_optimizer builds nothing to restore.
    load_optimizer = not reset_optimizer
    configured_group_options = None
    if load_optimizer:
        configured_group_options = snapshot_param_group_options(optimizer.param_groups)
    load_path, client_state = model_engine.load_checkpoint(
        run_dir,
        tag=resume_tag,
        load_module_strict=False,
        load_lr_scheduler_states=False,
        load_optimizer_states=load_optimizer,
    )
    if configured_group_options is not None:
        # In-place so wrapped optimizers (e.g. Nekaon) keep sharing the same param_groups list
        # with their inner optimizer — reassigning the list would break that identity.
        reapply_param_group_options(
            optimizer.param_groups,
            configured_group_options,
            configured_keys=configured_group_keys(config["optimizer"]),
        )
        del configured_group_options
    barrier()
    if load_path is None:
        if is_main:
            log("Resume requested but no checkpoint found; starting from step 1.")
        return ResumeState(step=1, examples=global_batch_size, epoch=1)

    if reset_dataloader:
        train_dataloader.epoch = client_state["custom_loader"]["epoch"]
    else:
        train_dataloader.load_state_dict(client_state["custom_loader"])
    # Restore the global RNG so post-resume augmentation/dropout/shuffling reproduce the
    # uninterrupted run's stochastic stream (exact for dataloader_num_workers=0).
    restore_rng_state(client_state.get("rng_state"))
    load_ema_checkpoint(load_path, training_ema, parameters_to_train)
    step = client_state["step"] + 1
    examples = client_state.get("examples", (step - 1) * global_batch_size) + global_batch_size
    epoch = epoch_schedule.current(step)
    if not force_constant_lr_enabled(config) and model_engine.lr_scheduler is not None:
        for _ in range(max(0, step - 1)):
            model_engine.lr_scheduler.step()
    if is_main:
        log(f"Resuming from checkpoint at epoch {epoch}, step {step}")
        effective_lrs = [group.get("lr") for group in optimizer.param_groups]
        log(
            f"[resume] effective optimizer LRs (config applied; optimizer state "
            f"preserved): {effective_lrs}",
            flush=True,
        )
        # Transparency: a changed batch/schedule re-derives steps_per_epoch, so the same
        # restored step now lands in a different epoch. Log it rather than silently drift.
        prev_spe = client_state.get("steps_per_epoch")
        if prev_spe and prev_spe != steps_per_epoch:
            log(
                f"[resume] batch/schedule changed: steps_per_epoch {prev_spe} -> "
                f"{steps_per_epoch}; step {step} now maps to epoch {epoch} of "
                f"{epoch_schedule.epochs}",
                flush=True,
            )
    return ResumeState(step=step, examples=examples, epoch=epoch, resumed=True)
