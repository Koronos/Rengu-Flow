"""Optimizer and LR scheduler resolution (torch-only for Phase 1)."""

from rengu_flow.optim.resolver import (
    RexLR,
    apply_warmup,
    build_scheduler_runtime_values,
    register_scheduler,
    resolve_scheduler,
    scheduler_registry,
    substitute_runtime_tokens,
)

__all__ = [
    "RexLR",
    "apply_warmup",
    "build_scheduler_runtime_values",
    "register_scheduler",
    "resolve_scheduler",
    "scheduler_registry",
    "substitute_runtime_tokens",
]
