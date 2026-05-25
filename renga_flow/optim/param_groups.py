"""Optimizer param-group helpers (aligned with diffusion-pipe train.py)."""

from __future__ import annotations

import copy
from typing import Any


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
        params_no_wd = []
        params_wd = []
        params = pg.pop("params")
        for p in params:
            name = getattr(p, "original_name", "")
            if p.ndim == 1 or name.startswith("llm_adapter.embed"):
                params_no_wd.append(p)
            else:
                params_wd.append(p)
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
        params_2d = []
        params_other = []
        for p in params:
            if p.ndim == 2:
                params_2d.append(p)
            else:
                params_other.append(p)
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
