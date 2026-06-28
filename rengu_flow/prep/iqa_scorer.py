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

Score scale and direction vary by model, so each record carries ``lower_better``
and the caller thresholds accordingly. Emits one JSON object per image to stdout
as it goes (JSONL) so the caller can stream progress.
"""

import json
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".jfif", ".avif"}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: iqa_scorer.py <folder> [model_name]", file=sys.stderr)
        return 2
    folder = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "clipiqa"

    import pyiqa
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = pyiqa.create_metric(model_name, device=device)
    lower_better = bool(getattr(metric, "lower_better", False))

    images = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in images:
        try:
            score = float(metric(str(path)).item())
            rec = {"path": str(path), "score": round(score, 4), "lower_better": lower_better}
        except Exception as exc:  # noqa: BLE001 — report per-image, keep going
            rec = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(rec) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
