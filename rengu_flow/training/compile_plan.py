"""Shape-aware torch.compile planning for multi-resolution training.

Multi-resolution / AR-bucketed datasets feed the model a fixed, enumerable set
of latent shapes (one per cached size bucket). The historical advice was
``compile_dynamic = true`` for any run where shapes vary, but dynamic kernels
are measurably slower than static ones and the Inductor disk cache cannot key
them. Worse, without ``compile_dynamic`` torch's automatic-dynamic-shapes
converts the whole model to dynamic kernels after the second distinct shape,
so multi-res runs silently lose the static-kernel speed a single-res run gets.

The plan computed here instead specializes statically per shape: it forces
``dynamic=False`` and raises torch._dynamo's recompile budgets to cover every
shape the dataset can produce. Each bucket then runs the exact kernels a
single-res run at that resolution would use, at the cost of one (disk-cached)
compile per shape.
"""

from dataclasses import dataclass, field

# torch._dynamo defaults (torch 2.x): per-code-object and process-wide entry caps.
DEFAULT_CACHE_SIZE_LIMIT = 8
DEFAULT_ACCUMULATED_LIMIT = 256
# Distinct compiled code objects in a pipeline model stay small (the repeated
# transformer block shares one); 24 gives the process-wide cap ample headroom.
_CODE_OBJECTS_HEADROOM = 24
# Margin over the exact shape count for incidental recompiles (e.g. a guard on
# a non-shape input changing once).
_SHAPE_MARGIN = 2


@dataclass
class CompilePlan:
    """Arguments for pipeline_model.compile() plus dynamo cache limits to apply."""

    kwargs: dict = field(default_factory=dict)
    cache_size_limit: int | None = None
    accumulated_cache_size_limit: int | None = None
    notes: list[str] = field(default_factory=list)


def plan_compile(config: dict, num_shapes: int | None) -> CompilePlan:
    """Build the torch.compile plan for ``config`` and a dataset with ``num_shapes``
    distinct size buckets (``None`` when unknown, e.g. synthetic data).
    """
    plan = CompilePlan()
    if mode := config.get("compile_mode"):
        plan.kwargs["mode"] = mode
    dynamic = config.get("compile_dynamic") is True
    if dynamic:
        plan.kwargs["dynamic"] = True
    elif num_shapes:
        # Defeat automatic dynamic shapes: specialize one static graph per size
        # bucket so every step runs single-res-speed kernels.
        plan.kwargs["dynamic"] = False

    if num_shapes and num_shapes > 1:
        # Budget one cache entry per shape. Without this, >8 shapes overflow
        # torch._dynamo's per-code-object cache and fall back to eager. Dynamic
        # mode recompiles per resolution bucket too, so size it the same way.
        limit = num_shapes + _SHAPE_MARGIN
        if limit > DEFAULT_CACHE_SIZE_LIMIT:
            plan.cache_size_limit = limit
        accumulated = limit * _CODE_OBJECTS_HEADROOM
        if accumulated > DEFAULT_ACCUMULATED_LIMIT:
            plan.accumulated_cache_size_limit = accumulated
        if not dynamic:
            plan.notes.append(
                f"multi-shape dataset: {num_shapes} size buckets -> static "
                "per-shape kernels (single-res speed; first step on each shape "
                "compiles once)."
            )
        if plan.cache_size_limit:
            plan.notes.append(
                f"raised torch._dynamo cache_size_limit "
                f"{DEFAULT_CACHE_SIZE_LIMIT} -> {plan.cache_size_limit} to fit "
                "every shape."
            )
    return plan


def apply_dynamo_limits(plan: CompilePlan) -> None:
    """Apply the plan's dynamo cache limits (no-op when defaults suffice)."""
    if plan.cache_size_limit is None and plan.accumulated_cache_size_limit is None:
        return
    import torch._dynamo

    cfg = torch._dynamo.config
    if plan.cache_size_limit is not None:
        cfg.cache_size_limit = max(cfg.cache_size_limit, plan.cache_size_limit)
    if plan.accumulated_cache_size_limit is not None:
        cfg.accumulated_cache_size_limit = max(
            cfg.accumulated_cache_size_limit, plan.accumulated_cache_size_limit
        )
