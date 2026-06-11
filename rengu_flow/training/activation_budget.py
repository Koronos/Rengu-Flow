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

The budget is GLOBAL: one value, read at compile time by every graph in both
compile modes. (A per-shape scaling variant was retired — see
docs/EXPERIMENTS_GRAVEYARD.md: under compile_dynamic the single graph baked the
first shape's scaled-up budget in and the largest bucket OOMed at any
configured base.) Because the budget is a *fraction* of the saved set, its
byte translation can still overshoot on a new model/resolution/batch
combination — ``BudgetBackoff`` makes that survivable instead of fatal.
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


class BudgetBackoff:
    """Make ``activation_memory_budget`` OOM-proof: lower it and recompile.

    The budget is a *fraction* of the partitioner's no-recompute saved set, so
    its translation to bytes depends on model, resolution, batch and whatever
    else lives on the GPU — no static computation is honest across all of
    them. The empirical guarantee instead: when a training step (or its
    compile) hits CUDA OOM under ``activation_checkpointing = "auto"``, lower
    the budget by ``factor``, reset dynamo and retry; the configured value
    becomes a desired ceiling, not a crash. The settled value is logged so the
    config can be updated for the next run.
    """

    def __init__(self, base: float, *, factor: float = 0.66, max_retries: int = 4) -> None:
        self.base = base
        self.current = base
        self.factor = factor
        self.max_retries = max_retries
        self.retries = 0

    def on_oom(self) -> float | None:
        """Return the next (lower) budget to try, or None when exhausted."""
        if self.retries >= self.max_retries or self.current <= 0.0:
            return None
        self.retries += 1
        nxt = round(self.current * self.factor, 3)
        # Below ~0.05 the partitioner is effectively at full checkpointing;
        # jump straight to 0.0 so the last retry is the true floor.
        self.current = 0.0 if nxt < 0.05 else nxt
        return self.current

    def describe(self) -> str:
        return (
            f"activation_memory_budget backed off {self.base} -> {self.current} "
            f"(retry {self.retries}/{self.max_retries}); update the config to "
            "start there next run and skip the failed compiles."
        )


def nominal_micro_batch(value) -> int:
    """One representative integer for a micro-batch config that may be a dict.

    Used where a single number is needed before the dataset can report the real
    per-step average (DeepSpeed config, optimizer batch-size scaling): the mean
    of the per-resolution values, rounded — unlike the first dict entry, it does
    not depend on key order. Step/example accounting should prefer the
    dataset's ``avg_examples_per_step`` once available.
    """
    if isinstance(value, dict):
        values = [int(v) for v in value.values()]
        return max(1, round(sum(values) / len(values))) if values else 1
    return int(value)
