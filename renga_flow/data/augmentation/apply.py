"""Apply resolved augmentation config to PIL images (and masks)."""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

from renga_flow.data.augmentation.branches import parse_variant_key
from renga_flow.data.augmentation.registry import (
    GEOMETRIC_ORDER,
    PHOTOMETRIC_ORDER,
    _GEOMETRIC_FNS,
    _PHOTOMETRIC_FNS,
    _to_rgb,
)
from renga_flow.data.cache_utils import seed_from_hash


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
    """Apply active strategies in fixed order (geometric → photometric)."""
    if not resolved.get("enabled") or not resolved.get("strategies"):
        return pil_image, mask

    strategies = resolved["strategies"]
    forced = parse_variant_key(variant_key) if variant_key else {}
    rng = random.Random(seed)
    image = pil_image
    mask_out = mask

    for name in GEOMETRIC_ORDER:
        if name not in strategies:
            continue
        entry = strategies[name]
        params = entry.get("params") or {}
        fn = _GEOMETRIC_FNS[name]
        if name == "horizontal_flip":
            image, mask_out = fn(
                image,
                mask_out,
                params,
                rng,
                forced.get(name),
                sampling=entry.get("sampling", "probability"),
            )
        else:
            image, mask_out = fn(image, mask_out, params, rng)

    image = _to_rgb(image)
    for name in PHOTOMETRIC_ORDER:
        if name not in strategies:
            continue
        entry = strategies[name]
        params = entry.get("params") or {}
        image = _PHOTOMETRIC_FNS[name](image, params, rng)

    return image, mask_out
