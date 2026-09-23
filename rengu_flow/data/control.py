"""Control (condition) images for edit datasets: pairing, sizing and loading.

A ``[[directory]]`` with ``control_path`` trains an edit model: each target image ``stem.ext``
is paired with one or more condition images in the control folder, and its ``.txt`` holds the
edit instruction. Two naming forms (see docs/user/dataset-config.md):

* ``stem.<ext>``                         -> one control image;
* ``stem_0.<ext>, stem_1.<ext>, ...``    -> N control images, in numeric order, contiguous from 0.

Each control keeps its OWN aspect ratio (it is not cropped to the target's bucket): it is resized
to the area ``control_resolution ** 2`` and floored to the model's pixel multiple. This module is
the single source of that size — the latent cache (VAE side) and the text-encoder cache both call
:func:`load_control_image`, so the image the VAE encodes and the one the text encoder sees always
have the same size.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image

# Extensions a control image may have (anything else in the control folder is ignored).
CONTROL_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".jpe", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)

_NUMBERED_STEM = re.compile(r"^(?P<base>.+)_(?P<n>\d+)$")


class ControlPairingError(RuntimeError):
    """A target image has no valid set of control images (missing, ambiguous or gapped)."""


def control_size(width: int, height: int, resolution: int, multiple: int) -> tuple[int, int]:
    """Pixel size ``(w, h)`` of a control image of ``width x height`` at ``resolution``.

    Keeps the image's own aspect ratio: area ``resolution ** 2``, ``w = sqrt(area * ratio)``,
    ``h = w / ratio``, each floored to ``multiple`` (never below one multiple).
    """
    multiple = int(multiple)
    area = float(resolution) ** 2
    ratio = float(width) / float(height)
    w = math.sqrt(area * ratio)
    h = w / ratio
    # The epsilon keeps an exact multiple (768.0 computed as 767.9999…) from flooring one step down.
    w_out = max(1, math.floor(w / multiple + 1e-6)) * multiple
    h_out = max(1, math.floor(h / multiple + 1e-6)) * multiple
    return int(w_out), int(h_out)


def control_signature(control_dims, resolution: int, multiple: int) -> tuple:
    """Batch-grouping key of a row: ``((w_0, h_0), ..., (w_{N-1}, h_{N-1}))`` (``()`` for t2i).

    ``control_dims`` is the row's list of source ``[width, height]`` per control image.
    """
    return tuple(control_size(w, h, resolution, multiple) for w, h in control_dims)


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in img.info


def load_control_image(path, resolution: int, multiple: int) -> Image.Image:
    """Open a control image and resize it to :func:`control_size` (no crop).

    Returns an RGB image, or RGBA when the file carries alpha.
    """
    with Image.open(path) as src:
        src.load()
        mode = "RGBA" if _has_alpha(src) else "RGB"
        img = src.convert(mode)
    size = control_size(img.width, img.height, resolution, multiple)
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def read_control_dims(path) -> tuple[int, int]:
    """Source ``(width, height)`` of a control image from its header (no pixel decode)."""
    with Image.open(path) as img:
        return int(img.width), int(img.height)


def control_stamp(path) -> str:
    """Cheap content identity of a control file (size + mtime), for cache keys.

    Same make/rsync tradeoff as the metadata source signature: a rewrite with identical size and
    mtime reads as unchanged; ``--regenerate_cache`` is the escape hatch.
    """
    st = Path(path).stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def index_control_dir(control_path) -> dict:
    """Index a control folder once: ``{"exact": {stem: [paths]}, "numbered": {base: {n: [paths]}}}``.

    Only image files count. A stem can appear in both maps (``a_1.png`` is the exact control of a
    target ``a_1`` and control #1 of a target ``a``); :func:`pair_control_files` decides per target.
    """
    exact: dict[str, list[Path]] = {}
    numbered: dict[str, dict[int, list[Path]]] = {}
    for p in sorted(Path(control_path).glob("*")):
        if not p.is_file() or p.suffix.lower() not in CONTROL_IMAGE_EXTENSIONS:
            continue
        exact.setdefault(p.stem, []).append(p)
        m = _NUMBERED_STEM.match(p.stem)
        if m:
            numbered.setdefault(m.group("base"), {}).setdefault(int(m.group("n")), []).append(p)
    return {"exact": exact, "numbered": numbered}


def pair_control_files(stem: str, index: dict, *, target: str | None = None) -> list[str]:
    """Control image paths for the target ``stem`` (in order), or raise :class:`ControlPairingError`.

    ``stem.<ext>`` -> ``[it]``; otherwise every ``stem_<n>.<ext>``, numeric order, which must be
    contiguous from 0. Both forms present, neither present, or two files for the same slot is an
    error (strict pairing: an unpaired target would silently train as text-to-image).
    """
    label = target or stem
    exact = index["exact"].get(stem, [])
    numbered = index["numbered"].get(stem, {})
    if exact and numbered:
        raise ControlPairingError(
            f"Control images for {label} use both forms ({exact[0].name} and "
            f"{stem}_<n>.*); keep either '{stem}.<ext>' or '{stem}_0.<ext>, {stem}_1.<ext>, …'."
        )
    if exact:
        if len(exact) > 1:
            raise ControlPairingError(
                f"Control image for {label} is ambiguous: {', '.join(p.name for p in exact)}."
            )
        return [str(exact[0])]
    if not numbered:
        raise ControlPairingError(
            f"No control image for {label}: expected '{stem}.<ext>' or '{stem}_0.<ext>, …' "
            "in control_path."
        )
    ns = sorted(numbered)
    for n in ns:
        if len(numbered[n]) > 1:
            raise ControlPairingError(
                f"Control image #{n} for {label} is ambiguous: "
                f"{', '.join(p.name for p in numbered[n])}."
            )
    if ns != list(range(len(ns))):
        raise ControlPairingError(
            f"Control images for {label} must be numbered contiguous from 0 "
            f"({stem}_0, {stem}_1, …); found indices {ns}."
        )
    return [str(numbered[n][0]) for n in ns]
