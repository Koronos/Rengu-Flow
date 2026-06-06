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


def resolve_optimizer_class(optim_type: str) -> type[torch.optim.Optimizer]:
    """Resolve optimizer class from string via registry or fully-qualified path."""
    return get_optimizer_class(optim_type)


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


register_scheduler("constant")(_constant_scheduler)
register_scheduler("linear")(_linear_scheduler)
register_scheduler("cosine")(_cosine_scheduler)
register_scheduler("rex")(_rex_scheduler)
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
        "Use 'constant', 'linear', 'cosine', 'rex', 'none', or a fully-qualified path."
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
