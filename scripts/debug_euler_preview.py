#!/usr/bin/env python3
"""Generate one Cosmos preview image via Euler sampling (no training loop)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="tests/fixtures/smoke/train_cosmos_predict2_signals.toml",
    )
    parser.add_argument("--out", default="tmp/euler_preview_test.png")
    parser.add_argument("--prompt", default="a red apple on a wooden table, soft light")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--simulate-cache-meta",
        action="store_true",
        help="Put text_encoder and VAE on meta like post-cache training.",
    )
    parser.add_argument("--no-offload-te", action="store_true", help="Keep text encoder on GPU.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this check.", file=sys.stderr)
        return 1

    from renga_flow.config import load_config, set_config_defaults
    from renga_flow.config.local_env import apply_model_paths_from_env, load_repo_dotenv
    from renga_flow.model.cosmos_predict2.preview_sampling import generate_preview_image
    from renga_flow.registry import get_model

    repo = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path

    load_repo_dotenv()
    config = load_config(config_path)
    set_config_defaults(config)
    apply_model_paths_from_env(config)

    print("Loading Cosmos pipeline...", flush=True)
    model = get_model(config)
    model.load_diffusion_model()
    if adapter_cfg := config.get("adapter"):
        model.configure_adapter(adapter_cfg)

    if args.simulate_cache_meta:
        print("Simulating post-cache meta devices...", flush=True)
        model.text_encoder.to("meta")
        model.vae.model.to("meta")

    preview_cfg = dict(config.get("preview") or {})
    preview_cfg["num_inference_steps"] = args.steps
    preview_cfg["guidance_scale"] = args.guidance_scale
    preview_cfg["width"] = preview_cfg.get("width", 512)
    preview_cfg["height"] = preview_cfg.get("height", 512)
    if args.no_offload_te:
        preview_cfg["preview_offload_text_encoder"] = False

    model.prepare_preview_memory(preview_cfg)
    try:
        image = generate_preview_image(
            model, preview_cfg, args.prompt, step=0, seed=42
        )
    finally:
        model.restore_after_preview()

    out = Path(args.out)
    if not out.is_absolute():
        out = repo / out
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)

    arr = np.asarray(image)
    print(
        f"OK: {out} | {image.size[0]}x{image.size[1]} | "
        f"mean={arr.mean():.1f} std={arr.std():.1f} "
        f"min={arr.min()} max={arr.max()}",
        flush=True,
    )
    if arr.std() < 1.0:
        print("WARN: very low variance — image may be flat.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
