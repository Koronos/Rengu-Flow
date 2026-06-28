"""Image-quality filtering for dataset preparation.

Flags low-quality images by two dep-free signals (numpy + Pillow only, both
already core deps):

  * sharpness -- variance of the Laplacian, computed on a long-side-512 copy so
    one threshold is portable across resolutions. Soft scans, weak upscales and
    out-of-focus shots score low.
  * resolution -- the shorter side in original pixels.

An image is flagged when ``sharpness < blur_threshold`` OR (``min_side > 0`` and
its shorter side < ``min_side``). ``action="report"`` (default) only writes the
scores into the job's ``report.json``; ``action="move"`` relocates flagged
images (and their caption sidecars) into ``<path>/low_quality/`` for review.

# ponytail: Laplacian variance is a blur heuristic, not a perceptual IQA model --
# flat/low-texture art can score low while sharp. Upgrade path if that bites:
# add a `metric="clipiqa"` branch backed by pyiqa (new dep). Not built until asked.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from rengu_flow.prep.caption_store import IMAGE_EXTENSIONS
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

LOW_QUALITY_DIR = "low_quality"


@dataclass
class QualityConfig:
    blur_threshold: float = 80.0  # calibration knob: tune against your own set
    min_side: int = 0  # flag images whose shorter side is below this (0 = off)
    action: str = "report"  # "report" (non-destructive) | "move"
    output_dir: Path | None = None  # destination for moved files (default <path>/low_quality)


def sharpness_score(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian of a 2-D grayscale array.

    Higher = sharper. Borders are cropped to avoid np.roll wrap artifacts.
    """
    a = gray.astype(np.float64)
    lap = -4.0 * a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)
    return float(lap[1:-1, 1:-1].var())


def _score_image(path: Path) -> tuple[float, int]:
    """Return (sharpness, shorter_side_px) for the image at *path*."""
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        shorter = min(w, h)
        # Long-side-512 grayscale copy so blur_threshold is resolution-portable.
        scale = 512.0 / max(w, h) if max(w, h) > 512 else 1.0
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        gray = np.asarray(im.convert("L"))
    return sharpness_score(gray), shorter


def filter_folder(
    src: str | Path,
    config: QualityConfig,
    *,
    caption_ext: str = ".txt",
    on_progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Score every image in *src* and flag/move the low-quality ones."""
    src = Path(src).resolve()
    images = sorted(
        p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    total = len(images)

    out_dir: Path | None = None
    if config.action == "move":
        out_dir = Path(config.output_dir) if config.output_dir else src / LOW_QUALITY_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "scored": 0,
        "flagged": 0,
        "moved": 0,
        "low_quality": [],
        "failed": [],
        "stopped": False,
        "action": config.action,
        "blur_threshold": config.blur_threshold,
        "min_side": config.min_side,
        "output_dir": str(out_dir) if out_dir else None,
    }

    for done, img_path in enumerate(images):
        if should_stop is not None and should_stop():
            report["stopped"] = True
            if on_progress:
                on_progress(done, total, "stopped")
            return report

        try:
            sharp, shorter = _score_image(img_path)
            report["scored"] += 1
            reasons = []
            if sharp < config.blur_threshold:
                reasons.append("blurry")
            if config.min_side > 0 and shorter < config.min_side:
                reasons.append("low_res")

            if reasons:
                report["flagged"] += 1
                report["low_quality"].append(
                    {
                        "name": img_path.name,
                        "sharpness": round(sharp, 1),
                        "min_side": shorter,
                        "reasons": reasons,
                    }
                )
                if out_dir is not None:
                    shutil.move(str(img_path), str(out_dir / img_path.name))
                    # Move the caption sidecar alongside so the dataset isn't left an orphan.
                    sidecar = img_path.with_suffix(caption_ext)
                    if sidecar.exists():
                        shutil.move(str(sidecar), str(out_dir / sidecar.name))
                    report["moved"] += 1
        except Exception as exc:
            logger.warning("quality: failed to process %s: %s", img_path.name, exc)
            report["failed"].append(img_path.name)

        if on_progress:
            on_progress(done + 1, total, img_path.name)

    return report


if __name__ == "__main__":
    # Self-check: a sharp checkerboard must out-score its blurred copy.
    rng = np.indices((64, 64)).sum(0) % 2 * 255.0  # high-frequency pattern
    blurred = (rng + np.roll(rng, 1, 0) + np.roll(rng, 1, 1) + np.roll(rng, -1, 0)) / 4.0
    assert sharpness_score(rng) > sharpness_score(blurred), "blur detection broken"
    flat = np.zeros((64, 64))
    assert sharpness_score(flat) == 0.0, "flat image must score 0"
    print("quality self-check OK")
