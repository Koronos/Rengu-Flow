"""Standalone anime-aesthetic scorer — invoked via ``uv run --with``.

deepghs ``anime_aesthetic`` maps each image to the 7-tier Danbooru quality
scale (worst → low → normal → good → great → best → masterpiece) via imgutils.
imgutils pins ``numpy<2`` and drags opencv/pandas/scipy, so it must NOT be a
project dependency. ``quality.py`` runs it as::

    uv run --project <repo> --with dghs-imgutils python aesthetic_scorer.py <folder>

This is a NON-isolated overlay: uv layers imgutils on top of the project env, so
the already-installed onnxruntime-gpu (with its working CUDA libs) is reused for
the model — no second onnxruntime, GPU available — while numpy is pinned to 1.26
only inside the ephemeral overlay (the project .venv is untouched). Deliberately
NOT PEP 723 inline metadata: that would force isolation and ignore the project
env, losing the installed onnxruntime-gpu.

Do NOT import this module from the package — it is only ever a subprocess entry
point (imgutils isn't in the project env). Emits one JSON object per image to
stdout as it goes (JSONL), so the caller can stream progress.
"""

import json
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".jfif", ".avif"}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: aesthetic_scorer.py <folder> [model_name]", file=sys.stderr)
        return 2
    folder = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    from imgutils.metrics import anime_dbaesthetic  # isolated env

    images = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    kwargs = {"model_name": model_name} if model_name else {}
    for path in images:
        try:
            label, percentile, score = anime_dbaesthetic(
                str(path), fmt=("label", "percentile", "score"), **kwargs
            )
            rec = {"path": str(path), "label": label,
                   "percentile": round(float(percentile), 4), "score": round(float(score), 4)}
        except Exception as exc:  # noqa: BLE001 — report per-image, keep going
            rec = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(rec) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
