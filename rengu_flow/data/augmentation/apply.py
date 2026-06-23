"""Apply resolved augmentation config to PIL images (and masks)."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

from rengu_flow.data.augmentation.registry import (
    GEOMETRIC_ORDER,
    PHOTOMETRIC_ORDER,
    _GEOMETRIC_FNS,
    _PHOTOMETRIC_FNS,
    _to_rgb,
)
from rengu_flow.data.cache_utils import seed_from_hash


def augmentation_seed_for_image(
    image_spec_base: tuple,
    aug_fingerprint: str,
    variant_key: str | None,
) -> int:
    return seed_from_hash((image_spec_base, aug_fingerprint, variant_key or ""))


def apply_augmentation(
    pil_image: Image.Image,
    mask: Image.Image | None,
    seed: int,
    resolved: dict[str, Any],
    variant_key: str | None = None,
) -> tuple[Image.Image, Image.Image | None]:
    """Apply active strategies in fixed order (geometric → photometric).

    ``variant_key is None`` keys the pristine original — it is returned untouched even when
    augmentation is enabled. Only the augmented copies ("1".."N") run the strategy stack.
    """
    if variant_key is None or not resolved.get("enabled") or not resolved.get("strategies"):
        return pil_image, mask

    strategies = resolved["strategies"]
    rng = random.Random(seed)
    image = pil_image
    mask_out = mask

    for name in GEOMETRIC_ORDER:
        if name not in strategies:
            continue
        entry = strategies[name]
        params = entry.get("params") or {}
        image, mask_out = _GEOMETRIC_FNS[name](image, mask_out, params, rng)

    image = _to_rgb(image)
    for name in PHOTOMETRIC_ORDER:
        if name not in strategies:
            continue
        entry = strategies[name]
        params = entry.get("params") or {}
        image = _PHOTOMETRIC_FNS[name](image, params, rng)

    return image, mask_out
