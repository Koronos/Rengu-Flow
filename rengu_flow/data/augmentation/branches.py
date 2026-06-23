"""Per-image augmentation copy expansion for metadata rows."""

from __future__ import annotations

from typing import Any


def expand_variant_keys(resolved: dict[str, Any]) -> list[str | None]:
    """Variant keys per image_spec: the pristine original plus N augmented copies.

    ``branches_per_image`` (default 1) is the number of augmented copies cached per image,
    *besides* the un-augmented original. ``None`` keys the pristine original — no augmentation
    is applied to it — and ``"1".."N"`` key the augmented copies, each deriving a distinct
    deterministic seed downstream. N = 0 (or augmentation disabled) → only the original.
    """
    if not resolved.get("enabled"):
        return [None]
    copies = int(resolved.get("branches_per_image", 1) or 0)
    return [None, *(str(i) for i in range(1, copies + 1))]
