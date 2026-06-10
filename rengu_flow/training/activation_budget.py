"""Compiler-driven activation checkpointing (Inductor memory-budget partitioner).

``activation_checkpointing = "auto"`` replaces the manual checkpoint wrappers
(full / selective / unsloth) with torch.compile's min-cut partitioner: each
layer's joint forward+backward graph is partitioned under a VRAM budget
(``activation_memory_budget``, 0.0-1.0), saving the activations that are most
expensive to recompute and recomputing the rest. The save/recompute split is
chosen per compiled graph from op cost estimates instead of a hand-written op
list, and the recompute is exact — same math as full or selective
checkpointing, no precision cost.

Budget semantics: 0.0 recomputes everything the partitioner can (full-AC-like
VRAM), 1.0 saves everything (no-AC speed and VRAM). The useful range sits in
between; it is a continuous dial over the same trade full/selective pick two
points on. Measured on Cosmos LoKr @1024 (RTX 4080, compile, steady state),
with full checkpointing at 0.974 s / 5.76 GB and SAC at 0.932 s / 6.56 GB:

    budget 0.1 -> 0.881 s /  6.37 GB   (faster AND smaller than SAC)
    budget 0.3 -> 0.822 s /  8.99 GB
    budget 0.5 -> 0.774 s / 11.32 GB   (speed plateau; 0.8 gains nothing)
"""

DEFAULT_BUDGET = 0.3


def resolve_auto_ac_budget(config: dict) -> float:
    """Validate config for activation_checkpointing='auto' and return the budget."""
    if not config.get("compile"):
        raise ValueError(
            "activation_checkpointing = 'auto' is implemented by torch.compile's "
            "memory-budget partitioner; set compile = true, or use "
            "activation_checkpointing = true / 'selective' instead."
        )
    budget = config.get("activation_memory_budget", DEFAULT_BUDGET)
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        raise ValueError(
            f"activation_memory_budget must be a number in [0.0, 1.0], got {budget!r}."
        ) from None
    if not 0.0 <= budget <= 1.0:
        raise ValueError(
            f"activation_memory_budget must be within [0.0, 1.0], got {budget}."
        )
    return budget


def apply_activation_memory_budget(budget: float) -> None:
    """Point Inductor's partitioner at the budget (call before compiling)."""
    import torch._functorch.config as functorch_config

    functorch_config.activation_memory_budget = budget
