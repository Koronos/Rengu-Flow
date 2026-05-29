"""Discrete augmentation branch expansion for metadata rows."""

from __future__ import annotations

from itertools import product
from typing import Any

from rengu_flow.data.augmentation.errors import AugmentationConfigError


def _branches_for_strategy(name: str, strategy: dict[str, Any]) -> list[str | None]:
    """Return variant_key suffixes for one strategy (None = identity branch)."""
    if strategy.get("sampling", "probability") != "enumerated":
        return [None]
    if name == "horizontal_flip":
        return [None, "horizontal_flip:mirror"]
    raise AugmentationConfigError(
        f"Strategy {name!r} is marked enumerable but has no branch catalogue."
    )


def expand_variant_keys(resolved: dict[str, Any]) -> list[str | None]:
    """Cartesian product of enumerable strategy branches per image_spec."""
    if not resolved.get("enabled"):
        return [None]

    strategies = resolved.get("strategies") or {}
    enumerable = [
        (name, strategy)
        for name, strategy in strategies.items()
        if strategy.get("sampling") == "enumerated"
    ]
    if not enumerable:
        return [None]

    branch_lists = [_branches_for_strategy(name, s) for name, s in enumerable]
    keys: list[str | None] = []
    for combo in product(*branch_lists):
        parts = [p for p in combo if p]
        keys.append(":".join(parts) if parts else None)

    max_branches = resolved.get("max_branches_per_image")
    if max_branches is not None and len(keys) > max_branches:
        raise AugmentationConfigError(
            f"Enumerated branches ({len(keys)}) exceed max_branches_per_image "
            f"({max_branches}). Reduce enumerable strategies or raise the cap."
        )
    return keys


def parse_variant_key(variant_key: str | None) -> dict[str, str]:
    """Parse ``horizontal_flip:mirror`` into ``{strategy: branch}``."""
    if not variant_key:
        return {}
    key = str(variant_key)
    if key == "horizontal_flip:mirror":
        return {"horizontal_flip": "mirror"}
    if ":" in key:
        strategy, branch = key.split(":", 1)
        return {strategy: branch}
    return {}
