"""Optimizer param-group helpers (aligned with diffusion-pipe train.py)."""

from __future__ import annotations

import copy
from typing import Any, Callable


def snapshot_param_group_options(
    param_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Capture configured group options without retaining parameter lists."""
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in param_groups
    ]


def reapply_param_group_options(
    param_groups: list[dict[str, Any]],
    configured: list[dict[str, Any]],
) -> None:
    """Apply configured options to checkpoint-loaded groups without replacing the groups.

    Wrapped optimizers such as Nekaon deliberately share the exact ``param_groups`` list with
    their inner optimizer. Assigning a saved pre-load list to the outer wrapper breaks that
    identity: its scheduler/lookahead and its inner Adakaon then operate on different LRs. Update
    the checkpoint-loaded dictionaries in place so wrapper bindings, parameter IDs and optimizer
    state stay intact.
    """
    if len(param_groups) != len(configured):
        raise ValueError(
            "Cannot apply edited optimizer settings: the checkpoint has "
            f"{len(param_groups)} parameter groups but the current config builds "
            f"{len(configured)}. Use --reset_optimizer when the group structure changes."
        )
    for group, options in zip(param_groups, configured):
        for key in tuple(group):
            if key != "params":
                del group[key]
        group.update(options)


def _partition(
    params: list[Any], predicate: Callable[[Any], bool]
) -> tuple[list[Any], list[Any]]:
    """Split *params* into ``(matching, rest)`` by *predicate*."""
    matching: list[Any] = []
    rest: list[Any] = []
    for p in params:
        (matching if predicate(p) else rest).append(p)
    return matching, rest


def adjust_beta2_half_life(optim_config: dict[str, Any], global_batch_size: int) -> dict[str, Any]:
    """Recompute betas[1] from beta2_half_life and global batch size. Mutates and returns optim_config."""
    cfg = copy.deepcopy(optim_config)
    half_life = cfg.pop("beta2_half_life", None)
    if half_life is None:
        return cfg
    betas = list(cfg["betas"])
    if len(betas) != 2:
        raise ValueError("beta2_half_life requires optimizer.betas of length 2")
    betas[1] = 0.5 ** (global_batch_size / half_life)
    cfg["betas"] = betas
    return cfg


def split_weight_decay_param_groups(
    param_groups: list[dict[str, Any]],
    optim_type_lower: str,
) -> list[dict[str, Any]]:
    """Split each group into weight-decay and no-decay (1D / embed) subsets."""
    new_param_groups: list[dict[str, Any]] = []
    for pg in param_groups:
        pg = dict(pg)
        params = pg.pop("params")
        params_no_wd, params_wd = _partition(
            params,
            lambda p: p.ndim == 1 or getattr(p, "original_name", "").startswith("llm_adapter.embed"),
        )
        pg_no_wd = pg.copy()
        pg["params"] = params_wd
        pg_no_wd["params"] = params_no_wd
        pg_no_wd["weight_decay"] = 0
        if optim_type_lower == "genericoptim":
            pg_no_wd["muon"] = False
            pg_no_wd["adamuon"] = False
            pg_no_wd["normuon"] = False
        if params_wd:
            new_param_groups.append(pg)
        if params_no_wd:
            new_param_groups.append(pg_no_wd)
    return new_param_groups


def split_genericoptim_param_groups(
    param_groups: list[dict[str, Any]],
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split 2D vs other params for GenericOptim (diffusion-pipe). May pop proj keys into 2D group."""
    new_param_groups: list[dict[str, Any]] = []
    for pg in param_groups:
        pg = dict(pg)
        params = pg.pop("params")
        params_2d, params_other = _partition(params, lambda p: p.ndim == 2)
        pg_2d = pg.copy()
        pg_2d["params"] = params_2d
        if kwargs.get("second_moment_type") == "sn":
            pg_2d["subset_size"] = "heuristics"
        for key in ("rank", "proj_type", "update_proj_gap"):
            if key in kwargs:
                pg_2d[key] = kwargs.pop(key)
        new_param_groups.append(pg_2d)
        pg_other = pg.copy()
        pg_other["params"] = params_other
        new_param_groups.append(pg_other)
    return new_param_groups
