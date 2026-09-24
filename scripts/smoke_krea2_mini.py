#!/usr/bin/env python3
"""GPU smoke: Krea 2 LoRA training on a random-weight mini model (no real weights needed).

Builds a mini Krea 2 DiT + mini Qwen3-VL text encoder (and a random Qwen-Image VAE unless
``RENGU_KREA2_VAE_PATH`` is set in the environment or repo-root ``.env``) under
``tmp/krea2_mini/models/``, writes a synthetic dataset (captions of very different lengths)
and dataset/train TOMLs under ``tmp/krea2_mini/run/``, then trains in-process with the
single-device engine (no DeepSpeed; works natively on Windows or Linux, any >= 4 GB GPU).

Asserts: every logged loss is finite, ``adapter_model.safetensors`` was saved with the
official ``transformer.*`` LoRA keys, and a preview PNG was written. Prints peak VRAM and
s/it from the bench summary. Exit 0 = pass, 1 = fail, 77 = skipped (no CUDA).

    python scripts/smoke_krea2_mini.py [--steps N] [--keep] [--rebuild]

Convention: docs/developer/smoke-tests.md.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "tmp" / "krea2_mini"
MODELS = ROOT / "models"
RUN = ROOT / "run"
SMOKE_SKIP_EXIT = 77
KREA2_ENV = (
    "RENGU_KREA2_TRANSFORMER_PATH",
    "RENGU_KREA2_VAE_PATH",
    "RENGU_KREA2_TEXT_ENCODER_PATH",
    "RENGU_KREA2_CHECKPOINT_PATH",
)
TE_HIDDEN = 2560  # text-encoder width the DiT's text_hidden_dim must match
CAPTION_WORDS = [3, 5, 8, 15, 25, 40, 60, 80]  # one image per length; exercises text padding
WORDS = (
    "red blue green shiny matte wooden metallic cat dog tree house river mountain sky cloud "
    "sunset night lamp chair table glass window door flower field road car bird stone wall"
).split()


def build_models(rebuild: bool) -> None:
    """Random-weight mini DiT (diffusers folder) + mini Qwen3-VL (transformers folder)."""
    import torch

    dit_dir, te_dir = MODELS / "transformer", MODELS / "text_encoder"
    if not rebuild and (dit_dir / "config.json").is_file() and (te_dir / "config.json").is_file():
        print(f"[mini] reusing models in {MODELS.relative_to(REPO)} (--rebuild to regenerate)")
        return
    shutil.rmtree(MODELS, ignore_errors=True)
    torch.manual_seed(0)

    from rengu_flow.model.krea2.dit import Krea2Transformer2DModel

    dit = Krea2Transformer2DModel(
        in_channels=64, num_layers=8, attention_head_dim=128, num_attention_heads=8,
        num_key_value_heads=2, intermediate_size=4096, timestep_embed_dim=256,
        text_hidden_dim=TE_HIDDEN, num_text_layers=12, text_num_attention_heads=20,
        text_num_key_value_heads=20, text_intermediate_size=6912,
        num_layerwise_text_blocks=2, num_refiner_text_blocks=2,
    ).to(torch.bfloat16)
    with torch.no_grad():  # non-zero modulation tables so the random model is not degenerate
        for name, param in dit.named_parameters():
            if "scale_shift_table" in name:
                param.normal_(0, 0.02)
    dit.save_pretrained(dit_dir)
    print(f"[mini] DiT {sum(p.numel() for p in dit.parameters()) / 1e6:.0f}M params")
    del dit

    # Full 36-layer depth (the DiT taps hidden states across it), tiny width elsewhere.
    from transformers import Qwen3VLConfig, Qwen3VLModel

    from rengu_flow.model.krea2.loading import QWEN3VL_ASSETS

    base = json.loads((QWEN3VL_ASSETS / "config.json").read_text(encoding="utf-8"))
    base["text_config"].update(
        hidden_size=TE_HIDDEN, intermediate_size=512, num_attention_heads=2,
        num_key_value_heads=1, head_dim=128, num_hidden_layers=36, max_position_embeddings=4096,
    )
    base["vision_config"].update(
        depth=2, hidden_size=64, intermediate_size=128, num_heads=1,
        out_hidden_size=TE_HIDDEN, deepstack_visual_indexes=[0, 1],
    )
    cfg = Qwen3VLConfig(
        **{k: v for k, v in base.items() if k not in ("architectures", "transformers_version")}
    )
    te = Qwen3VLModel(cfg).to(torch.bfloat16)
    te.save_pretrained(te_dir)
    print(f"[mini] text encoder {sum(p.numel() for p in te.parameters()) / 1e6:.0f}M params")


def resolve_vae() -> Path:
    """RENGU_KREA2_VAE_PATH from env/.env, else a random VAE from the bundled config."""
    from rengu_flow.config.local_env import load_repo_dotenv

    load_repo_dotenv()
    env_vae = os.environ.get("RENGU_KREA2_VAE_PATH", "").strip()
    # The trainer applies RENGU_KREA2_* over [model]; drop them so the mini paths win.
    for name in KREA2_ENV:
        os.environ.pop(name, None)
    if env_vae:
        if not Path(env_vae).exists():
            raise SystemExit(f"RENGU_KREA2_VAE_PATH not found: {env_vae}")
        print("[mini] VAE: RENGU_KREA2_VAE_PATH")
        return Path(env_vae)
    vae_dir = MODELS / "vae"
    if not (vae_dir / "config.json").is_file():
        import torch
        from diffusers import AutoencoderKLQwenImage

        from rengu_flow.model.krea2.loading import VAE_CONFIG_PATH

        torch.manual_seed(0)
        config = json.loads(VAE_CONFIG_PATH.read_text(encoding="utf-8"))
        vae = AutoencoderKLQwenImage.from_config({k: v for k, v in config.items() if not k.startswith("_")})
        vae.to(torch.bfloat16).save_pretrained(vae_dir)
    print("[mini] VAE: random (bundled qwen_image_vae_config.json)")
    return vae_dir


def write_dataset(steps: int) -> Path:
    from PIL import Image, ImageDraw

    rng = random.Random(0)
    img_dir = RUN / "data" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, n_words in enumerate(CAPTION_WORDS):
        size = (640, 384) if i == len(CAPTION_WORDS) - 1 else (512, 512)  # second AR bucket
        im = Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
        draw = ImageDraw.Draw(im)
        for _ in range(12):
            x0, y0 = rng.randrange(size[0]), rng.randrange(size[1])
            draw.ellipse(
                [x0, y0, x0 + rng.randrange(20, 200), y0 + rng.randrange(20, 200)],
                fill=tuple(rng.randrange(256) for _ in range(3)),
            )
        im.save(img_dir / f"img{i}.png")
        caption = " ".join(rng.choice(WORDS) for _ in range(n_words))
        (img_dir / f"img{i}.txt").write_text(caption, encoding="utf-8")
    repeats = max(1, math.ceil(steps * 2 / len(CAPTION_WORDS)))  # GAS 2 -> 2 samples/step
    dataset = RUN / "dataset.toml"
    dataset.write_text(
        "resolutions = [512]\nenable_ar_bucket = true\nmin_ar = 0.5\nmax_ar = 2.0\n"
        "num_ar_buckets = 4\nframe_buckets = [1]\n\n"
        f"[[directory]]\npath = '{img_dir.as_posix()}'\nnum_repeats = {repeats}\n",
        encoding="utf-8",
    )
    return dataset


ADAPTERS = {
    "lora": "type = 'lora'\nrank = 16",
    "lokr": "type = 'lokr'\nrank = 6\nfactor = -1",  # quantization-aware
}


def write_train_config(dataset: Path, vae: Path, steps: int, adapter: str, nf4: bool) -> Path:
    train = RUN / "train.toml"
    quant = "transformer_4bit = true\n" if nf4 else ""
    train.write_text(
        f"""output_dir = '{(RUN / "output").as_posix()}'
dataset = '{dataset.as_posix()}'
cache_root = '{(RUN / "cache").as_posix()}'
max_steps = {steps}
epochs = 999
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 2
activation_checkpointing = true
save_every_n_epochs = 999
caching_batch_size = 4
bench = true
logging_steps = 1

[model]
type = 'krea2'
dtype = 'bfloat16'
transformer_path = '{(MODELS / "transformer").as_posix()}'
vae_path = '{vae.as_posix()}'
text_encoder_path = '{(MODELS / "text_encoder").as_posix()}'
{quant}
[adapter]
{ADAPTERS[adapter]}

[optimizer]
type = 'adamw'
lr = 1e-4

[preview]
width = 512
height = 512
num_inference_steps = 4
guidance_scale = 4.0
negative_prompt = ''
seed = 42
preview_every_n_steps = {steps}
preview_save_png = true
prompts = ['a red cat sitting on a wooden chair near a window']
""",
        encoding="utf-8",
    )
    return train


def check_run(steps: int) -> list[str]:
    """Return failure messages (empty = pass) and print the bench numbers."""
    runs = sorted(p for p in (RUN / "output").glob("*") if p.is_dir())
    if not runs:
        return ["no run directory under tmp/krea2_mini/run/output"]
    run_dir = runs[-1]
    failures: list[str] = []

    csv_path = run_dir / "bench_steps.csv"
    rows = list(csv.DictReader(csv_path.open())) if csv_path.is_file() else []
    losses = [float(r["loss"]) for r in rows]
    if len(losses) != steps:
        failures.append(f"expected {steps} bench rows, got {len(losses)}")
    if not losses or not all(math.isfinite(x) for x in losses):
        failures.append(f"non-finite or missing loss: {losses}")
    if rows:
        iters = [float(r["iter_sec"]) for r in rows]
        warm = iters[1:] or iters  # step 1 includes compile/warmup
        peak = max(float(r["cuda_peak_gb"]) for r in rows)
        print(
            f"[mini] loss first={losses[0]:.4f} last={losses[-1]:.4f} | "
            f"bench peak VRAM {peak:.2f} GB | {sum(warm) / len(warm):.3f} s/it (steps 2+)"
        )

    adapters = sorted(run_dir.glob("*/adapter_model.safetensors"))
    if not adapters:
        failures.append("no adapter_model.safetensors saved")
    else:
        import torch
        from safetensors.torch import load_file

        sd = load_file(adapters[-1])
        n_a = sum(".lora_A." in k or k.endswith((".lokr_w1", ".lokr_w1_a")) for k in sd)
        if not sd or any(not k.startswith("transformer.") for k in sd):
            failures.append("adapter keys missing the official transformer.* prefix")
        n_b = sum(".lora_B." in k or k.endswith((".lokr_w2", ".lokr_w2_a")) for k in sd)
        if not n_a or n_a != n_b:
            failures.append("adapter factor keys missing or unpaired (lora_A/B or lokr_w1/w2)")
        if not all(torch.isfinite(v.float()).all() for v in sd.values()):
            failures.append("adapter has non-finite tensors")
        print(f"[mini] adapter {adapters[-1].relative_to(REPO).as_posix()} ({n_a} adapted modules)")

    previews = sorted((run_dir / "preview").glob("*.png"))
    if not previews:
        failures.append("no preview PNG written")
    else:
        print(f"[mini] preview {previews[-1].relative_to(REPO).as_posix()}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=8, help="optimizer steps (GAS 2), default 8")
    parser.add_argument("--keep", action="store_true", help="keep tmp/krea2_mini/run (dataset, cache, output)")
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), default="lora")
    parser.add_argument("--4bit", dest="nf4", action="store_true", help="NF4 base (needs bitsandbytes)")
    parser.add_argument("--rebuild", action="store_true", help="regenerate the mini models")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    os.environ.setdefault("RENGU_ENGINE", "accelerate")  # single-device engine, no DeepSpeed
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", file=sys.stderr)
        return SMOKE_SKIP_EXIT

    if RUN.exists():  # leftovers of a previous run (fresh process: nothing holds them)
        shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    exit_code = 1
    try:
        build_models(args.rebuild)
        vae = resolve_vae()
        train = write_train_config(write_dataset(args.steps), vae, args.steps, args.adapter, args.nf4)

        from rengu_flow.main import main as train_main

        train_main(["--config", str(train)])
        failures = check_run(args.steps)
        print(f"[mini] overall peak VRAM (incl. preview) {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        exit_code = 1 if failures else 0
        print("Smoke krea2_mini OK." if exit_code == 0 else "Smoke krea2_mini FAILED.")
    finally:
        if not args.keep:
            gc.collect()  # release cache sqlite/memmap handles (Windows locks open files)
            shutil.rmtree(RUN, ignore_errors=True)
            if RUN.exists():
                print(f"[mini] note: {RUN.relative_to(REPO)} still in use; removed on the next run")
    return exit_code


if __name__ == "__main__":  # caching workers use spawn on Windows: keep the guard
    sys.exit(main())
