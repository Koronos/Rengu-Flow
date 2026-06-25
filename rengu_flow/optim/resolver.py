"""Resolve optimizer and scheduler from config. Uses optimizer registry and scheduler registry."""

import importlib
from typing import Any, Callable

import torch

from rengu_flow.registry.optimizers import get_optimizer_class

# Scheduler factory: (optimizer, config, total_steps, steps_per_epoch) -> LRScheduler | None
SchedulerFactory = Callable[
    [
        torch.optim.Optimizer,
        dict[str, Any],
        int,
        int,
    ],
    torch.optim.lr_scheduler.LRScheduler | None,
]

scheduler_registry: dict[str, SchedulerFactory] = {}


def register_scheduler(name: str) -> Callable[[SchedulerFactory], SchedulerFactory]:
    """Decorator to register a scheduler factory by name (case-insensitive)."""

    def decorator(fn: SchedulerFactory) -> SchedulerFactory:
        scheduler_registry[name.lower()] = fn
        return fn

    return decorator


def substitute_runtime_tokens(kwargs: dict[str, Any], runtime_values: dict[str, Any]) -> dict[str, Any]:
    """Replace string tokens in config values with runtime values (e.g. 'total_steps' -> int)."""
    for key, value in list(kwargs.items()):
        if isinstance(value, str) and value in runtime_values:
            kwargs[key] = runtime_values[value]
    return kwargs


def build_scheduler_runtime_values(
    config: dict[str, Any],
    *,
    total_steps: int,
    steps_per_epoch: int,
) -> dict[str, Any]:
    """Runtime placeholders for ``[lr_scheduler_args]`` (custom class paths)."""
    epochs = config.get("epochs", 1)
    effective = total_steps
    values: dict[str, Any] = {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "effective_total_steps": effective,
    }
    max_steps = config.get("max_steps")
    if max_steps is not None:
        try:
            ms = int(max_steps)
            if ms > 0:
                values["max_steps"] = ms
                values["effective_total_steps"] = min(total_steps, ms)
        except (TypeError, ValueError):
            pass
    gas = config.get("gradient_accumulation_steps")
    if gas is not None:
        try:
            values["gradient_accumulation_steps"] = int(gas)
        except (TypeError, ValueError):
            pass
    return values


class RexLR(torch.optim.lr_scheduler.LRScheduler):
    """REX schedule (Chen et al. 2021, *Revisiting Budgeted Training with an Improved Schedule*).

    Reflected-exponential decay: slower at the start, faster near the end. With remaining
    fraction ``z = clamp(1 - step / total_steps, 0, 1)`` the multiplier is
    ``z / ((1 - d) + d * z)`` (1.0 at step 0, 0.0 at ``total_steps``) and the LR for each
    param group is ``lr_min + (base_lr - lr_min) * multiplier``.

    The shape coefficient ``d`` (``rex_d``) interpolates the curve:
    ``d = 0.0`` is linear decay, ``d = 0.5`` is the canonical REX profile, and ``d -> 1.0``
    holds the LR higher for longer before a sharper final drop. Defaults to ``0.5``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        lr_min: float = 0.0,
        rex_d: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = int(total_steps)
        self.lr_min = lr_min
        self.rex_d = min(max(float(rex_d), 0.0), 1.0)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.total_steps <= 0:
            return list(self.base_lrs)
        z = 1.0 - min(max(self.last_epoch, 0), self.total_steps) / self.total_steps
        denom = (1.0 - self.rex_d) + self.rex_d * z
        factor = z / denom if denom > 0.0 else 0.0
        return [self.lr_min + (base_lr - self.lr_min) * factor for base_lr in self.base_lrs]


def _constant_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], total_steps: int, steps_per_epoch: int) -> torch.optim.lr_scheduler.ConstantLR:
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def _linear_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], total_steps: int, steps_per_epoch: int) -> torch.optim.lr_scheduler.LinearLR:
    return torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )


def _cosine_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], total_steps: int, steps_per_epoch: int) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    lr_min = config.get("lr_scheduler_args", {}).get("lr_min", 0.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr_min
    )


def _rex_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], total_steps: int, steps_per_epoch: int) -> RexLR:
    args = config.get("lr_scheduler_args", {})
    lr_min = args.get("lr_min", 0.0)
    rex_d = args.get("rex_d", 0.5)
    return RexLR(optimizer, total_steps=total_steps, lr_min=lr_min, rex_d=rex_d)


def _none_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], total_steps: int, steps_per_epoch: int) -> None:
    return None


def parse_wsd_decay_steps(decay: Any, total_steps: int) -> int:
    """Resolve the WSD ``decay`` arg to a step count.

    A **float** is a fraction of ``total_steps`` (``0.1`` → last 10%, ``1.0`` → all decay);
    an **int** is an absolute number of decay steps (``100`` → last 100 steps). Clamped to
    ``[1, total_steps]``. ``bool`` is treated as "use default" (it is an ``int`` subclass).
    """
    if isinstance(decay, bool) or decay is None:
        decay = 0.1
    if isinstance(decay, float):
        frac = min(max(decay, 0.0), 1.0)
        steps = round(total_steps * frac)
    else:
        steps = int(decay)
    return max(1, min(steps, max(1, total_steps)))


def wsd_decay_onset_step(config: dict[str, Any], total_steps: int) -> int:
    """Global training step where the WSD decay tail begins (where the fork is saved).

    Equals ``total_steps - decay_steps`` — warmup-independent, because warmup shortens the
    stable phase by the same amount it prepends (the factory sizes the stable phase over the
    post-warmup budget). The trainer saves the protected pre-decay checkpoint here.
    """
    decay = config.get("lr_scheduler_args", {}).get("decay", 0.1)
    return max(0, total_steps - parse_wsd_decay_steps(decay, total_steps))


def _wsd_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any], total_steps: int, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Warmup-Stable-Decay: hold the base LR constant, then decay over a final tail.

    The flat "stable" phase is the most *continuable* shape (no mid-run LR drop to recover
    from); the decay tail is what actually lands the model. Because the LR is still at base at
    the decay onset, the trainer auto-saves a resume checkpoint there (the "fork") so you can
    later extend the run from before the decay and re-anchor the tail to the new end.

    ``lr_scheduler_args``: ``decay`` (float = fraction of total, int = absolute steps;
    default ``0.1``), ``decay_type`` (``"rex"`` default, ``"cosine"``, ``"linear"``),
    ``rex_d`` (REX shape, default ``0.9`` — high LR held long, sharp final drop) and
    ``lr_min`` (decay floor for rex/cosine; default ``0.0``).
    """
    args = config.get("lr_scheduler_args", {})
    # Stable phase is sized over the post-warmup budget so the decay still ends exactly at
    # the run's last step (apply_warmup prepends the warmup phase on top of this scheduler).
    warmup = int(config.get("warmup_steps", 0) or 0)
    effective = max(1, total_steps - warmup)
    decay_steps = min(parse_wsd_decay_steps(args.get("decay", 0.1), total_steps), effective)
    stable_steps = max(0, effective - decay_steps)
    decay_type = str(args.get("decay_type", "rex")).lower()
    lr_min = args.get("lr_min", 0.0)

    if decay_type == "rex":
        decay_sched: torch.optim.lr_scheduler.LRScheduler = RexLR(
            optimizer, total_steps=decay_steps, lr_min=lr_min, rex_d=args.get("rex_d", 0.9)
        )
    elif decay_type == "cosine":
        decay_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=decay_steps, eta_min=lr_min
        )
    elif decay_type == "linear":
        decay_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps
        )
    else:
        raise ValueError(
            f"Unknown wsd decay_type '{decay_type}'. Use 'rex', 'cosine', or 'linear'."
        )

    if stable_steps <= 0:
        scheduler: torch.optim.lr_scheduler.LRScheduler = decay_sched
    else:
        stable = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=stable_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[stable, decay_sched], milestones=[stable_steps]
        )
    # The trainer reads this to place the protected pre-decay fork checkpoint.
    scheduler.wsd_decay_onset = stable_steps  # type: ignore[attr-defined]
    return scheduler


register_scheduler("constant")(_constant_scheduler)
register_scheduler("linear")(_linear_scheduler)
register_scheduler("cosine")(_cosine_scheduler)
register_scheduler("rex")(_rex_scheduler)
register_scheduler("wsd")(_wsd_scheduler)
register_scheduler("none")(_none_scheduler)


def resolve_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_steps: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create LR scheduler from type (registry name or qualified path)."""
    scheduler_type_lower = scheduler_type.lower()
    runtime_values = build_scheduler_runtime_values(
        config, total_steps=total_steps, steps_per_epoch=steps_per_epoch
    )
    if scheduler_type_lower in scheduler_registry:
        return scheduler_registry[scheduler_type_lower](optimizer, config, total_steps, steps_per_epoch)
    if "." in scheduler_type:
        module_path, class_name = scheduler_type.rsplit(".", 1)
        module = importlib.import_module(module_path)
        scheduler_class = getattr(module, class_name)
        scheduler_kwargs = substitute_runtime_tokens(
            dict(config.get("lr_scheduler_args", {})), runtime_values
        )
        return scheduler_class(optimizer, **scheduler_kwargs)
    raise ValueError(
        f"Unknown scheduler type '{scheduler_type}'. "
        "Use 'constant', 'linear', 'cosine', 'rex', 'wsd', 'none', or a fully-qualified path."
    )


def apply_warmup(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """If warmup_steps > 0 and scheduler is not None, wrap with SequentialLR (warmup + main)."""
    if scheduler is None or warmup_steps <= 0:
        return scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / warmup_steps, total_iters=warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps]
    )
