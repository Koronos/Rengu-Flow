"""Helpers for image_spec tuples with optional variant_key."""

from __future__ import annotations


def image_spec_base(image_spec: tuple) -> tuple:
    """(tar, path) without variant suffix."""
    return (image_spec[0], image_spec[1])


def image_spec_variant_key(image_spec: tuple) -> str | None:
    if len(image_spec) > 2:
        vk = image_spec[2]
        return vk if vk else None
    return None


def with_variant_key(image_spec: tuple, variant_key: str | None) -> tuple:
    base = image_spec_base(image_spec)
    if variant_key:
        return (base[0], base[1], variant_key)
    return base
