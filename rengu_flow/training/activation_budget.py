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


def scale_budget_for_area(base: float, latent_area: int, max_latent_area: int) -> float:
    """Per-shape budget: ``base`` applies to the largest bucket; smaller shapes scale up.

    The VRAM constraint comes from the largest shape only — a uniform budget
    makes the partitioner recompute just as aggressively on small shapes where
    activations are a fraction of the peak, slowing them for nothing. Scaling
    by area keeps the saved-activation bytes of every shape at or below
    ``base x (no-checkpoint bytes of the largest shape)``, so the peak is
    unchanged while small shapes run with little or no recompute (budget
    capped at 1.0). Applied per shape just before its first (compiling) step.

    "Area" here must be the **effective token count of the whole micro-batch**
    (batch x T x H x W of the latents): activation bytes scale with the batch
    dimension exactly like they scale with resolution, so a per-resolution
    ``micro_batch_size_per_gpu`` dict (e.g. batch 4 @512 + batch 1 @1024)
    would otherwise let a small-resolution/large-batch shape blow past the
    peak the budget was chosen for.

    Requires per-shape compiles (``compile_dynamic = false``): a single dynamic
    graph reads the budget ONCE at its compile — on whichever shape arrives
    first — so per-shape scaling would bake a small bucket's near-1.0 budget
    into the graph and OOM the largest bucket. ``main`` disables the scaling
    and applies the base budget globally when ``compile_dynamic`` is on.
    """
    if max_latent_area <= 0 or latent_area <= 0:
        return base
    return min(1.0, base * max_latent_area / latent_area)


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


def micro_batch_for_size_bucket(
    size_bucket: tuple,
    micro_batch_dict: dict,
    image_micro_batch_dict: dict,
) -> int:
    """Per-GPU micro batch a (w, h, frames) bucket will be fed with.

    Mirrors the dataset's resolution->batch rule (``dataset.py post_init``):
    image buckets (frames == 1) read the image dict, video buckets the main
    dict; a ``None`` key means "all resolutions", otherwise the numerically
    closest key to sqrt(w*h) wins.
    """
    import math

    w, h, frames = size_bucket[-3:]
    bs_dict = image_micro_batch_dict if frames == 1 else micro_batch_dict
    if None in bs_dict:
        return int(bs_dict[None])
    bucket_size = math.sqrt(w * h)
    return int(min(bs_dict.items(), key=lambda kv: abs(kv[0] - bucket_size))[1])
