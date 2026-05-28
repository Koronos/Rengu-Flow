#!/usr/bin/env python3
"""Exit 0 when SDXL v2 cache for smoke_cc0 dataset is usable; else 1."""

from __future__ import annotations

import sys
from pathlib import Path

import toml

REPO = Path(__file__).resolve().parents[1]
DATASET_TOML = REPO / "tests/fixtures/smoke/dataset_cc0.toml"
IMAGES_DIR = REPO / "tests/fixtures/smoke_cc0/images"
MODEL = "sdxl"


def smoke_cache_dir() -> Path:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from renga_flow.data.cache_paths import resolve_directory_cache_dir

    cfg = toml.load(DATASET_TOML)
    cfg["_dataset_toml_path"] = str(DATASET_TOML.resolve())
    return resolve_directory_cache_dir(cfg, IMAGES_DIR, MODEL)


def main() -> int:
    cache_dir = smoke_cache_dir()
    if not cache_dir.is_dir():
        print(f"cache missing: {cache_dir}", file=sys.stderr)
        return 1

    meta = cache_dir / "metadata"
    if not meta.is_dir() or not any(meta.glob("*.arrow")):
        print(f"metadata incomplete: {meta}", file=sys.stderr)
        return 1

    latent_dirs = [p for p in cache_dir.glob("cache_*") if p.is_dir()]
    if not latent_dirs:
        print(f"no cache_* latent dirs under {cache_dir}", file=sys.stderr)
        return 1
    latent_ok = False
    for d in latent_dirs:
        files = list(d.rglob("*.arrow")) + list(d.rglob("*.mmap"))
        if len(files) >= 1:
            latent_ok = True
            break
    if not latent_ok:
        print(f"latent cache empty under {cache_dir}", file=sys.stderr)
        return 1

    for te in ("uncond_text_embeddings_1", "uncond_text_embeddings_2"):
        te_dir = cache_dir / te
        if not te_dir.is_dir() or not any(te_dir.iterdir()):
            print(f"text embeddings incomplete: {te_dir}", file=sys.stderr)
            return 1

    print(f"cache ready: {cache_dir}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--print-dir":
        print(smoke_cache_dir())
        raise SystemExit(0)
    raise SystemExit(main())
