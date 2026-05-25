"""Training previews: sample images during training and log to TensorBoard (and optional WandB)."""

from __future__ import annotations

import time
from typing import Any

import torch

from renga_flow.utils.common import empty_cuda_cache, is_main_process


def get_preview_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the ``[preview]`` table from config, or an empty dict."""
    preview = config.get("preview")
    return preview if isinstance(preview, dict) else {}


def previews_configured(config: dict[str, Any]) -> bool:
    """True when ``[preview]`` has prompts and previews are not explicitly disabled."""
    preview_cfg = get_preview_config(config)
    if not preview_cfg.get("prompts"):
        return False
    return preview_cfg.get("enabled", True)


def normalize_preview_prompts(preview_cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (tag_name, prompt) pairs for TensorBoard image tags."""
    raw = preview_cfg.get("prompts") or []
    out: list[tuple[str, str]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            out.append((f"prompt_{i}", item))
        elif isinstance(item, dict):
            prompt = item.get("prompt") or item.get("text")
            if not prompt:
                continue
            name = item.get("name") or f"prompt_{i}"
            out.append((str(name), str(prompt)))
    return out


def should_run_previews(
    config: dict[str, Any],
    step: int,
    epoch: int,
    *,
    finished_epoch: bool = False,
    forced: bool = False,
) -> bool:
    if forced:
        return previews_configured(config)
    preview_cfg = get_preview_config(config)
    if not previews_configured(config):
        return False
    every_n_steps = preview_cfg.get("preview_every_n_steps")
    if every_n_steps is not None and step % every_n_steps == 0:
        return True
    every_n_epochs = preview_cfg.get("preview_every_n_epochs")
    if finished_epoch and every_n_epochs is not None and epoch % every_n_epochs == 0:
        return True
    return False


def _pil_to_chw_float(image) -> torch.Tensor:
    import numpy as np

    arr = np.array(image, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().div(255.0)


def _dist_barrier() -> None:
    try:
        from deepspeed import comm as dist
    except ImportError:
        return
    if dist.is_initialized():
        dist.barrier()


def run_previews(
    model: Any,
    config: dict[str, Any],
    tb_writer: Any,
    step: int,
    *,
    disable_block_swap: bool = False,
    optimizer: Any = None,
    wandb_enable: bool = False,
) -> None:
    """Generate preview images on the main process and log them to TensorBoard."""
    preview_cfg = get_preview_config(config)
    prompts = normalize_preview_prompts(preview_cfg)
    if not prompts:
        return

    if model.name != "sdxl":
        if is_main_process():
            print(f"Preview skipped: model type {model.name!r} does not support previews yet.")
        return

    _dist_barrier()
    if not is_main_process():
        _dist_barrier()
        return

    if optimizer is not None and hasattr(optimizer, "eval") and callable(optimizer.eval):
        optimizer.eval()

    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)

    start = time.time()
    try:
        _run_sdxl_previews(model, preview_cfg, prompts, tb_writer, step, wandb_enable=wandb_enable)
    finally:
        empty_cuda_cache()
        model.prepare_block_swap_training()
        if optimizer is not None and hasattr(optimizer, "train") and callable(optimizer.train):
            optimizer.train()

    _dist_barrier()
    print(f"Preview complete in {time.time() - start:.1f}s")


def _run_sdxl_previews(
    model: Any,
    preview_cfg: dict[str, Any],
    prompts: list[tuple[str, str]],
    tb_writer: Any,
    step: int,
    *,
    wandb_enable: bool = False,
) -> None:
    model.load_diffusion_model()
    pipe = model._pipeline
    device = pipe.device

    negative_prompt = preview_cfg.get("negative_prompt", "")
    width = int(preview_cfg.get("width", 1024))
    height = int(preview_cfg.get("height", 1024))
    num_inference_steps = int(preview_cfg.get("num_inference_steps", 20))
    guidance_scale = float(preview_cfg.get("guidance_scale", 7.0))
    base_seed = int(preview_cfg.get("seed", 0))
    seed_stride = int(preview_cfg.get("seed_stride", 1))

    modules = (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2)
    prev_training = [m.training for m in modules]
    for m in modules:
        m.eval()

    wandb_images: dict[str, Any] = {}
    print(f"Running preview at step {step} ({len(prompts)} prompt(s))")

    with torch.no_grad():
        for idx, (name, prompt) in enumerate(prompts):
            generator = torch.Generator(device=device).manual_seed(base_seed + step * seed_stride + idx)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            )
            image = result.images[0]
            tag = f"preview/{name}"
            if tb_writer is not None:
                tb_writer.add_image(tag, _pil_to_chw_float(image), step)
            if wandb_enable:
                try:
                    import wandb

                    wandb_images[tag] = wandb.Image(image, caption=prompt)
                except ImportError:
                    pass

    for m, was_training in zip(modules, prev_training):
        m.train(was_training)

    if wandb_enable and wandb_images:
        try:
            import wandb

            wandb.log({**wandb_images, "step": step})
        except ImportError:
            pass
