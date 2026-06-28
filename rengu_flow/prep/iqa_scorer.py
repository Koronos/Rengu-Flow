"""Standalone technical image-quality scorer — invoked via ``uv run --with``.

Runs a learned No-Reference IQA model from pyiqa (IQA-PyTorch): CLIP-IQA,
ARNIQA, MUSIQ, MANIQA, BRISQUE, NIQE, ... These predict perceived *technical*
quality (blur, noise, compression, low effective resolution) — unlike the
deepghs aesthetic model (booru appeal) or the Laplacian heuristic (edge energy,
fooled by pixelation).

pyiqa pulls torch/timm/opencv, so it is NOT a project dependency. ``quality.py``
runs it as::

    uv run --project <repo> --with pyiqa python iqa_scorer.py <folder> [model]

A NON-isolated overlay: uv reuses the project's torch (and CUDA/GPU) and layers
pyiqa on top; the project .venv is untouched.

Each image is decoded to a first-frame RGB PIL image (GIF/palette/RGBA/CMYK/
grayscale all normalized) and handed to pyiqa as a *PIL image*, NOT a raw tensor:
pyiqa then applies each model's own preprocessing. That matters — feeding a raw
native-resolution tensor bypasses it, giving wrong scores and OOM-ing attention
models like MANIQA (which crops to 224). So inference is per-image (no cross-model
batching); the GPU is kept fed by decoding ahead in worker threads, since the
real bottleneck for these small models is serial disk decode, not GPU compute.

Score scale and direction vary by model, so each record carries ``lower_better``
and the caller thresholds accordingly. Emits one JSON object per image to stdout
as it goes (JSONL) so the caller can stream progress.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".jfif", ".avif"}

# Bound how many decoded images buffer ahead of the GPU (memory cap for huge folders).
PREFETCH_CHUNK = 64


def _emit(rec: dict) -> None:
    sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.flush()


def _list_images(target: Path) -> list:
    """Images to score: every image in *target* if it's a folder, else the paths
    listed one-per-line in *target* (a manifest, for incremental indexing)."""
    if target.is_dir():
        return sorted(
            p for p in target.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return [Path(line) for line in target.read_text().splitlines() if line.strip()]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: iqa_scorer.py <folder|manifest> [model_name]", file=sys.stderr)
        return 2
    target = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "clipiqa"

    import pyiqa
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = pyiqa.create_metric(model_name, device=device)
    lower_better = bool(getattr(metric, "lower_better", False))

    # Each pyiqa model has its own scale and direction (clipiqa 0..1 higher-better,
    # niqe ~0..100 lower-better, brisque ~0..150 lower-better, ...). Normalize every
    # raw score to a single quality on 1..100 where higher is always better, so one
    # threshold/slider means the same thing across models. score_range is a string
    # like "0, 1" or "~0, ~100".
    sr = str(getattr(metric, "score_range", "0, 1"))
    try:
        lo, hi = (float(x.strip().lstrip("~").strip()) for x in sr.split(","))
    except Exception:  # noqa: BLE001 — unparseable range, assume 0..1
        lo, hi = 0.0, 1.0
    span = (hi - lo) or 1.0

    def to_quality(raw: float) -> float:
        q = (raw - lo) / span
        if lower_better:
            q = 1.0 - q
        q = max(0.0, min(1.0, q))  # clamp outliers (niqe can blow past its range)
        return round(1.0 + 99.0 * q, 1)  # 1..100, higher = better

    def load_rgb(path: Path):
        # First-frame RGB so GIFs/palette/RGBA/CMYK/grayscale all normalize to a
        # clean 3-channel image pyiqa can preprocess. Returns (path, image) or
        # (path, exception) so decode can run in worker threads without losing
        # which file failed.
        try:
            with Image.open(path) as im:
                if getattr(im, "is_animated", False):
                    im.seek(0)
                return path, im.convert("RGB")
        except Exception as exc:  # noqa: BLE001 — reported per-image
            return path, exc

    images = _list_images(target)

    # Decode ahead in worker threads (PIL releases the GIL) so disk IO overlaps GPU
    # inference; score sequentially per image so each model's preprocessing runs.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(images), PREFETCH_CHUNK):
            for path, res in pool.map(load_rgb, images[i:i + PREFETCH_CHUNK]):
                if isinstance(res, Exception):
                    _emit({"path": str(path), "error": f"{type(res).__name__}: {res}"})
                    continue
                try:
                    with torch.no_grad():
                        score = float(metric(res).item())
                    _emit({"path": str(path), "quality": to_quality(score),
                           "score": round(score, 4)})
                except Exception as exc:  # noqa: BLE001 — keep going past a bad image
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    _emit({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    # os._exit after a clean run: torch/CUDA interpreter-shutdown teardown can race
    # with the parent closing our stdout pipe and return a spurious non-zero code
    # (120) even though every score was already emitted and flushed. A real failure
    # raises inside main() and still surfaces with a traceback + non-zero exit.
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
