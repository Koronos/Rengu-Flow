"""Per-strategy PIL/numpy augmentation implementations (MVP)."""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def _to_rgb(pil: Image.Image) -> Image.Image:
    if pil.mode == "RGB":
        return pil
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[-1])
        return bg
    return pil.convert("RGB")


def _numpy_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.randint(0, 2**31 - 1))


# --- Geometric (image + mask) ---


def horizontal_flip(
    image: Image.Image,
    mask: Image.Image | None,
    params: dict[str, Any],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image | None]:
    prob = float(params.get("probability", 0.5))
    if rng.random() < prob:
        image = ImageOps.mirror(image)
        if mask is not None:
            mask = ImageOps.mirror(mask)
    return image, mask


def small_rotation(
    image: Image.Image,
    mask: Image.Image | None,
    params: dict[str, Any],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image | None]:
    max_deg = float(params.get("max_degrees", 2.0))
    angle = rng.uniform(-max_deg, max_deg)
    fillcolor = (255, 255, 255)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fillcolor)
    if mask is not None:
        mask = mask.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)
    return image, mask


def crop_jitter(
    image: Image.Image,
    mask: Image.Image | None,
    params: dict[str, Any],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image | None]:
    frac = float(params.get("fraction", 0.02))
    w, h = image.size
    side = min(w, h)
    margin = int(side * frac)
    if margin < 1:
        return image, mask
    left = rng.randint(0, margin)
    top = rng.randint(0, margin)
    right = w - rng.randint(0, margin)
    bottom = h - rng.randint(0, margin)
    if right <= left + 1 or bottom <= top + 1:
        return image, mask
    image = image.crop((left, top, right, bottom)).resize((w, h), Image.Resampling.BICUBIC)
    if mask is not None:
        mask = mask.crop((left, top, right, bottom)).resize((w, h), Image.Resampling.BICUBIC)
    return image, mask


# --- Photometric (image only) ---


def color_jitter(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    image = _to_rgb(image)
    b = float(params.get("brightness", 0.05))
    c = float(params.get("contrast", 0.05))
    s = float(params.get("saturation", 0.05))
    h = float(params.get("hue", 0.02))
    if b > 0:
        image = ImageEnhance.Brightness(image).enhance(1.0 + rng.uniform(-b, b))
    if c > 0:
        image = ImageEnhance.Contrast(image).enhance(1.0 + rng.uniform(-c, c))
    if s > 0:
        image = ImageEnhance.Color(image).enhance(1.0 + rng.uniform(-s, s))
    if h > 0:
        arr = np.array(image).astype(np.float32)
        shift = rng.uniform(-h, h) * 255.0
        arr[..., 0] = np.clip(arr[..., 0] + shift * 0.3, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] - shift * 0.3, 0, 255)
        image = Image.fromarray(arr.astype(np.uint8))
    return image


def gamma(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    gmin = float(params.get("gamma_min", 0.95))
    gmax = float(params.get("gamma_max", 1.05))
    g = rng.uniform(gmin, gmax)
    arr = np.array(_to_rgb(image)).astype(np.float32) / 255.0
    arr = np.power(np.clip(arr, 1e-6, 1.0), g)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def jpeg_simulation(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    qmin = int(params.get("quality_min", 85))
    qmax = int(params.get("quality_max", 98))
    quality = rng.randint(min(qmin, qmax), max(qmin, qmax))
    buf = io.BytesIO()
    _to_rgb(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def temperature_tint(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    span = float(params.get("warm_cool_range", 0.05))
    t = rng.uniform(-span, span)
    arr = np.array(_to_rgb(image)).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * (1.0 + t), 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * (1.0 - t), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def chromatic_aberration(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    shift = int(params.get("shift_px", 1))
    if shift < 1:
        return _to_rgb(image)
    dx = rng.randint(-shift, shift)
    arr = np.array(_to_rgb(image))
    out = arr.copy()
    out[..., 0] = np.roll(arr[..., 0], dx, axis=1)
    out[..., 2] = np.roll(arr[..., 2], -dx, axis=1)
    return Image.fromarray(out.astype(np.uint8))


def gaussian_noise(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    sigma = float(params.get("sigma", 2.0))
    if sigma <= 0:
        return _to_rgb(image)
    arr = np.array(_to_rgb(image)).astype(np.float32)
    arr += _numpy_rng(rng).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def film_grain(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    intensity = float(params.get("intensity", 0.04))
    arr = np.array(_to_rgb(image)).astype(np.float32)
    grain = _numpy_rng(rng).normal(0, intensity * 255.0, arr.shape)
    return Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8))


def lab_jitter(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    dl = float(params.get("delta_l", 3.0))
    da = float(params.get("delta_a", 2.0))
    db = float(params.get("delta_b", 2.0))
    arr = np.array(_to_rgb(image)).astype(np.float32) / 255.0
    arr[..., 0] = np.clip(arr[..., 0] + rng.uniform(-dl, dl) / 255.0, 0, 1)
    arr[..., 1] = np.clip(arr[..., 1] + rng.uniform(-da, da) / 255.0, 0, 1)
    arr[..., 2] = np.clip(arr[..., 2] + rng.uniform(-db, db) / 255.0, 0, 1)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def split_toning(
    image: Image.Image,
    params: dict[str, Any],
    rng: random.Random,
) -> Image.Image:
    strength = float(params.get("strength", 0.15))
    arr = np.array(_to_rgb(image)).astype(np.float32)
    lum = arr.mean(axis=-1, keepdims=True) / 255.0
    shadow = lum < 0.5
    warm = np.array([1.08, 1.0, 0.92], dtype=np.float32)
    arr = arr / 255.0
    arr[shadow[..., 0]] = np.clip(
        arr[shadow[..., 0]] * (1.0 + strength * (warm - 1.0)), 0, 1
    )
    return Image.fromarray((arr * 255.0).astype(np.uint8))


GEOMETRIC_ORDER = ("crop_jitter", "small_rotation", "horizontal_flip")
PHOTOMETRIC_ORDER = (
    "color_jitter",
    "gamma",
    "jpeg_simulation",
    "temperature_tint",
    "chromatic_aberration",
    "gaussian_noise",
    "film_grain",
    "lab_jitter",
    "split_toning",
)

_GEOMETRIC_FNS = {
    "horizontal_flip": horizontal_flip,
    "small_rotation": small_rotation,
    "crop_jitter": crop_jitter,
}

_PHOTOMETRIC_FNS = {
    "color_jitter": color_jitter,
    "gamma": gamma,
    "jpeg_simulation": jpeg_simulation,
    "temperature_tint": temperature_tint,
    "chromatic_aberration": chromatic_aberration,
    "gaussian_noise": gaussian_noise,
    "film_grain": film_grain,
    "lab_jitter": lab_jitter,
    "split_toning": split_toning,
}
