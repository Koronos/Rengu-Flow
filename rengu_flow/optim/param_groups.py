"""Optimizer param-group helpers (aligned with diffusion-pipe train.py)."""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable


def snapshot_param_group_options(
    param_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Capture configured group options without retaining parameter lists."""
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in param_groups
    ]


# Group keys that are always user/config-controlled: the LR (and the scheduler's base LR), the
# weight-decay split, and the per-group switches ``split_*_param_groups`` write from the config.
ALWAYS_CONFIGURED_GROUP_KEYS = frozenset(
    {
        "lr",
        "initial_lr",
        "weight_decay",
        "muon",
        "adamuon",
        "normuon",
        "rank",
        "proj_type",
        "update_proj_gap",
        "subset_size",
    }
)


def configured_group_keys(optimizer_config: dict[str, Any]) -> frozenset[str]:
    """Group keys the user controls through the ``[optimizer]`` TOML section.

    ``beta2_half_life`` is translated into ``betas`` (the key the group actually holds); ``type``
    and ``gradient_release`` are not hyperparameters. Everything else keeps its own name.
    """
    keys = {k for k in optimizer_config if k not in ("type", "gradient_release")}
    if "beta2_half_life" in keys:
        keys.discard("beta2_half_life")
        keys.add("betas")
    return frozenset(keys) | ALWAYS_CONFIGURED_GROUP_KEYS


def reapply_param_group_options(
    param_groups: list[dict[str, Any]],
    configured: list[dict[str, Any]],
    configured_keys: Iterable[str] | None = None,
) -> None:
    """Apply configured options to checkpoint-loaded groups without replacing the groups.

    Wrapped optimizers such as Nekaon deliberately share the exact ``param_groups`` list with
    their inner optimizer. Assigning a saved pre-load list to the outer wrapper breaks that
    identity: its scheduler/lookahead and its inner Adakaon then operate on different LRs. Update
    the checkpoint-loaded dictionaries in place so wrapper bindings, parameter IDs and optimizer
    state stay intact.

    Optimizers keep running state in the group dicts (Prodigy ``d``, ScheduleFree ``step``...).
    So nothing is ever deleted, and only the keys the user configured (``configured_keys``; by
    default ``ALWAYS_CONFIGURED_GROUP_KEYS`` plus the common torch hyperparameters) are overwritten
    from the freshly built optimizer — any other key keeps its checkpoint value.
    """
    if len(param_groups) != len(configured):
        raise ValueError(
            "Cannot apply edited optimizer settings: the checkpoint has "
            f"{len(param_groups)} parameter groups but the current config builds "
            f"{len(configured)}. Use --reset_optimizer when the group structure changes."
        )
    keys = (
        _DEFAULT_CONFIGURED_KEYS if configured_keys is None else frozenset(configured_keys)
    ) | ALWAYS_CONFIGURED_GROUP_KEYS
    for group, options in zip(param_groups, configured):
        group.update({k: v for k, v in options.items() if k in keys and k != "params"})


_DEFAULT_CONFIGURED_KEYS = frozenset(
    {"betas", "eps", "momentum", "dampening", "nesterov", "amsgrad"}
) | ALWAYS_CONFIGURED_GROUP_KEYS


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


_NO_DECAY_NAME_MARKERS = ("dora_scale", "magnitude", "scalar")


def is_no_weight_decay_param(p: Any) -> bool:
    """True for params that must not receive weight decay.

    Covers biases/norm scales (``ndim <= 1``, including 0-D LyCORIS ``scalar``), DoRA magnitude
    vectors (``dora_scale`` is ``1 x in``, so ``ndim`` alone misses it), and the Cosmos
    ``llm_adapter.embed`` tables. Reused by the gradient_release path so both paths share one rule.
    """
    if getattr(p, "ndim", 2) <= 1:
        return True
    name = getattr(p, "original_name", "") or ""
    if name.startswith("llm_adapter.embed"):
        return True
    return any(marker in name for marker in _NO_DECAY_NAME_MARKERS)


def split_weight_decay_param_groups(
    param_groups: list[dict[str, Any]],
    optim_type_lower: str,
) -> list[dict[str, Any]]:
    """Split each group into weight-decay and no-decay (1D / embed) subsets."""
    new_param_groups: list[dict[str, Any]] = []
    for pg in param_groups:
        pg = dict(pg)
        params = pg.pop("params")
        params_no_wd, params_wd = _partition(params, is_no_weight_decay_param)
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
                pg_2d[key] = kwargs[key]
        new_param_groups.append(pg_2d)
        pg_other = pg.copy()
        pg_other["params"] = params_other
        new_param_groups.append(pg_other)
    # Pop only after every group has received them (popping inside the loop starved later groups).
    for key in ("rank", "proj_type", "update_proj_gap"):
        kwargs.pop(key, None)
    return new_param_groups
