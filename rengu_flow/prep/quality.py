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

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from rengu_flow.prep.caption_store import IMAGE_EXTENSIONS
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

LOW_QUALITY_DIR = "low_quality"

# deepghs anime_aesthetic's 7-tier Danbooru scale, worst -> best. An image is
# flagged when its predicted label ranks below `aesthetic_min_label`.
AESTHETIC_LABELS = ("worst", "low", "normal", "good", "great", "best", "masterpiece")
_AESTHETIC_SCORER = Path(__file__).with_name("aesthetic_scorer.py")
_IQA_SCORER = Path(__file__).with_name("iqa_scorer.py")
_REPO_ROOT = Path(__file__).resolve().parents[2]  # rengu_flow/prep/ -> repo root


@dataclass
class QualityConfig:
    # "blur" (Laplacian, dep-free) | "aesthetic" (deepghs booru appeal) | "iqa" (pyiqa technical)
    metric: str = "blur"
    blur_threshold: float = 80.0  # calibration knob: tune against your own set
    min_side: int = 0  # flag images whose shorter side is below this (0 = off)
    min_detail: float = 0.0  # flag low effective resolution (pixelated/upscaled); 0 = off
    aesthetic_min_label: str = "normal"  # flag images ranked below this label
    aesthetic_model: str = ""  # imgutils model_name override ("" = its default)
    iqa_model: str = "clipiqa"  # pyiqa NR-IQA model (clipiqa/arniqa: any domain; musiq/maniqa: photos)
    iqa_threshold: float = 0.5  # flag images on the wrong side of this (model-dependent scale)
    action: str = "report"  # "report" (non-destructive) | "move"
    output_dir: Path | None = None  # destination for moved files (default <path>/low_quality)


def sharpness_score(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian of a 2-D grayscale array.

    Higher = sharper. Borders are cropped to avoid np.roll wrap artifacts.
    """
    a = gray.astype(np.float64)
    lap = -4.0 * a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)
    return float(lap[1:-1, 1:-1].var())


def detail_residual(gray: np.ndarray) -> float:
    """Mean abs residual after halving then restoring resolution (bilinear).

    Low => little real detail for the image's size: blurred OR pixelated/upscaled
    (the latter reconstructs almost perfectly from half-res). Unlike the Laplacian,
    this is not fooled by the hard block edges of nearest-neighbour upscaling.
    """
    im = Image.fromarray(gray)
    w, h = im.size
    small = im.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
    back = small.resize((w, h), Image.BILINEAR)
    return float(np.abs(gray.astype(np.float32) - np.asarray(back, np.float32)).mean())


def _score_image(path: Path, *, want_detail: bool = False) -> tuple[float, int, float]:
    """Return (sharpness, shorter_side_px, detail) for the image at *path*.

    ``detail`` is computed only when *want_detail* (it opens a larger copy);
    it is 0.0 otherwise.
    """
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        shorter = min(w, h)
        # detail wants real resolution (long-side 1024) to see pixelation; the
        # Laplacian uses a long-side-512 copy so blur_threshold stays portable.
        detail = 0.0
        if want_detail:
            dscale = 1024.0 / max(w, h) if max(w, h) > 1024 else 1.0
            dim = im.resize((max(1, round(w * dscale)), max(1, round(h * dscale)))) if dscale < 1.0 else im
            detail = detail_residual(np.asarray(dim.convert("L")))
        scale = 512.0 / max(w, h) if max(w, h) > 512 else 1.0
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        gray = np.asarray(im.convert("L"))
    return sharpness_score(gray), shorter, detail


def _iter_scorer(scorer: Path, with_pkg: str, src: Path, model_name: str):
    """Yield (path, record) from a model scorer subprocess, streaming.

    ``uv run --with`` overlays *with_pkg* on top of the project env (non-isolated),
    so the installed torch/onnxruntime (and GPU) are reused while the heavy deps
    are pinned only inside the ephemeral overlay; the project .venv is untouched.
    Each stdout line is one image's JSON result; we yield them as they arrive so
    progress stays live.
    """
    cmd = ["uv", "run", "--project", str(_REPO_ROOT), "--with", with_pkg,
           "python", str(scorer), str(src)]
    if model_name:
        cmd.append(model_name)
    # Sanitized env: a parent already under `uv run` leaks VIRTUAL_ENV / UV_* into
    # the nested uv, and a leaked PYTHONPATH would let the overlay import the
    # project's packages by accident.
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH") and not k.startswith("UV_")}
    # Child stderr -> temp file, never the inherited stderr: when the parent's stderr
    # is a pipe, `uv run` races on it at teardown and returns 120 *after* every result
    # was already emitted. The file also captures the traceback if launch truly fails.
    produced = False
    with tempfile.TemporaryFile(mode="w+") as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True, env=env)
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # stray non-JSON noise (uv notice, tqdm bar) — ignore
                produced = True
                yield Path(rec["path"]), rec
        finally:
            proc.stdout.close()  # type: ignore[union-attr]
            rc = proc.wait()
            # Only a launch failure (model load, bad deps) yields nothing; a non-zero
            # code after a full run is a uv/torch teardown artifact and is ignored.
            if rc != 0 and not produced:
                errf.seek(0)
                tail = errf.read()[-2000:]
                raise RuntimeError(f"{scorer.name} exited with code {rc} and no output:\n{tail}")


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
        "metric": config.metric,
        "scored": 0,
        "flagged": 0,
        "moved": 0,
        "low_quality": [],
        "failed": [],
        "stopped": False,
        "action": config.action,
        "output_dir": str(out_dir) if out_dir else None,
    }
    if config.metric == "blur":
        report["blur_threshold"] = config.blur_threshold
        report["min_side"] = config.min_side
        report["min_detail"] = config.min_detail
    elif config.metric == "aesthetic":
        report["aesthetic_min_label"] = config.aesthetic_min_label
    else:  # iqa
        report["iqa_model"] = config.iqa_model
        report["iqa_threshold"] = config.iqa_threshold

    def _flag(img_path: Path, info: dict) -> None:
        report["flagged"] += 1
        report["low_quality"].append({"name": img_path.name, **info})
        if out_dir is not None:
            shutil.move(str(img_path), str(out_dir / img_path.name))
            # Move the caption sidecar alongside so the dataset isn't left an orphan.
            sidecar = img_path.with_suffix(caption_ext)
            if sidecar.exists():
                shutil.move(str(sidecar), str(out_dir / sidecar.name))
            report["moved"] += 1

    if config.metric == "aesthetic":
        min_rank = AESTHETIC_LABELS.index(config.aesthetic_min_label)
        done = 0
        for img_path, rec in _iter_scorer(_AESTHETIC_SCORER, "dghs-imgutils>=0.15", src,
                                          config.aesthetic_model):
            if should_stop is not None and should_stop():
                report["stopped"] = True
                break
            done += 1
            if "error" in rec or rec.get("label") not in AESTHETIC_LABELS:
                report["failed"].append(img_path.name)
            else:
                report["scored"] += 1
                if AESTHETIC_LABELS.index(rec["label"]) < min_rank:
                    _flag(img_path, {"label": rec["label"], "percentile": rec.get("percentile"),
                                     "reasons": [f"aesthetic<{config.aesthetic_min_label}"]})
            if on_progress:
                on_progress(done, total, img_path.name)
        return report

    if config.metric == "iqa":
        thr = config.iqa_threshold
        done = 0
        for img_path, rec in _iter_scorer(_IQA_SCORER, "pyiqa", src, config.iqa_model):
            if should_stop is not None and should_stop():
                report["stopped"] = True
                break
            done += 1
            if "error" in rec or "score" not in rec:
                report["failed"].append(img_path.name)
            else:
                report["scored"] += 1
                score = rec["score"]
                # lower_better: high score = bad; else low score = bad.
                bad = score > thr if rec.get("lower_better") else score < thr
                if bad:
                    _flag(img_path, {"score": score, "reasons": [f"iqa:{config.iqa_model}"]})
            if on_progress:
                on_progress(done, total, img_path.name)
        return report

    # metric == "blur": scoring is pure CPU+IO (PIL/numpy release the GIL), so
    # score across a thread pool; flag/move + report stay on this thread (no locks).
    want_detail = config.min_detail > 0

    def _score_one(img_path: Path):
        try:
            return img_path, _score_image(img_path, want_detail=want_detail), None
        except Exception as exc:  # noqa: BLE001 — reported per-image
            return img_path, None, exc

    workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (img_path, scored, exc) in enumerate(pool.map(_score_one, images)):
            if should_stop is not None and should_stop():
                report["stopped"] = True
                break
            if exc is not None:
                logger.warning("quality: failed to process %s: %s", img_path.name, exc)
                report["failed"].append(img_path.name)
            else:
                sharp, shorter, detail = scored
                report["scored"] += 1
                reasons = []
                if sharp < config.blur_threshold:
                    reasons.append("blurry")
                if config.min_side > 0 and shorter < config.min_side:
                    reasons.append("low_res")
                if want_detail and detail < config.min_detail:
                    reasons.append("pixelated")
                if reasons:
                    info = {"sharpness": round(sharp, 1), "min_side": shorter, "reasons": reasons}
                    if want_detail:
                        info["detail"] = round(detail, 2)
                    _flag(img_path, info)
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
    # detail_residual: real detail >> nearest-neighbour upscale (pixelated) >> flat.
    noise = np.random.default_rng(0).integers(0, 255, (64, 64)).astype(np.uint8)
    pixelated = np.repeat(np.repeat(noise[::8, ::8], 8, 0), 8, 1).astype(np.uint8)
    assert detail_residual(noise) > detail_residual(pixelated) > detail_residual(flat.astype(np.uint8)), \
        "detail residual must rank detailed > pixelated > flat"
    print("quality self-check OK")
