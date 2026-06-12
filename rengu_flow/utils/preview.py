"""Training previews: sample images during training and log to TensorBoard (and optional WandB)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from rengu_flow.utils.common import empty_cuda_cache, is_main_process


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


def reload_preview_config(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    sink: Any = None,
    step: int | None = None,
) -> bool:
    """Hot-reload the ``[preview]`` section from ``config_path`` into ``config`` in place.

    Used by the ``reload_config`` signal so a running job can change previews live (edit
    the TOML, then signal). Only ``[preview]`` is reloaded — model/optimizer/dataset
    cannot change mid-run. ``run_previews``/``should_run_previews`` read ``config["preview"]``
    every step, so replacing it here takes effect immediately (including ``enabled`` to
    turn previews off/on). Rank 0 reads the file and broadcasts so all ranks stay in sync.
    Returns True when the section was applied.
    """
    import toml

    from rengu_flow.utils.signal_files import _broadcast_object_list

    new_preview: dict[str, Any] | None = None
    if is_main_process():
        try:
            disk = toml.load(str(config_path))
            section = disk.get("preview")
            new_preview = section if isinstance(section, dict) else {}
            print(
                f"rengu_flow: reloaded [preview] from {config_path} "
                f"({len(new_preview)} keys)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - best-effort; a bad edit must not kill training
            print(f"rengu_flow: failed to reload [preview] from {config_path}: {e}", flush=True)
            new_preview = None
    (new_preview,) = _broadcast_object_list([new_preview])
    if new_preview is None:
        return False
    # Record the live config mutation on the run timeline (rank 0): what [preview] keys changed.
    if sink is not None and is_main_process():
        from rengu_track import EVENT_CONFIG_RELOADED, config_diff

        diff = config_diff({"preview": config.get("preview", {})}, {"preview": new_preview})
        sink.event(
            EVENT_CONFIG_RELOADED,
            step=step,
            payload={"section": "preview", "diff": diff},
        )
    config["preview"] = new_preview
    return True


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
    preview_cfg = get_preview_config(config)
    if forced:
        # An explicit preview (signal / "Preview now" button) runs whenever there are
        # prompts to render, even if previews are otherwise disabled (enabled=false) or
        # have no step/epoch schedule — the user asked for one on purpose.
        return bool(preview_cfg.get("prompts"))
    if not previews_configured(config):
        return False
    every_n_steps = preview_cfg.get("preview_every_n_steps")
    if every_n_steps is not None and step % every_n_steps == 0:
        return True
    every_n_epochs = preview_cfg.get("preview_every_n_epochs")
    if finished_epoch and every_n_epochs is not None and epoch % every_n_epochs == 0:
        return True
    return False


def _save_preview_png(
    image,
    sink: Any,
    preview_cfg: dict[str, Any],
    name: str,
    step: int,
) -> None:
    """Write ``preview/step{NNNNNNNN}_{name}.png`` under the run directory.

    The step comes first (zero-padded) so the files sort chronologically in a file browser —
    plain ``step1000`` would otherwise sort before ``step9``.
    """
    if not preview_cfg.get("preview_save_png", False):
        return
    log_dir = getattr(sink, "run_dir", None)
    if not log_dir:
        return
    out_dir = Path(log_dir) / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = str(name).replace("/", "_")
    path = out_dir / f"step{step:08d}_{safe}.png"
    image.save(path)
    print(f"rengu_flow: saved preview PNG {path}", flush=True)


def _pil_to_chw_float(image) -> torch.Tensor:
    import numpy as np

    arr = np.array(image, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().div(255.0)


def _log_preview_image(
    *,
    name: str,
    prompt: str,
    image: Any,
    sink: Any,
    preview_cfg: dict[str, Any],
    step: int,
) -> None:
    tag = f"preview/{name}"
    sink.image(tag, _pil_to_chw_float(image), step)
    _save_preview_png(image, sink, preview_cfg, name, step)


def _dist_barrier() -> None:
    from rengu_flow import distributed

    distributed.barrier()


def run_previews(
    model: Any,
    config: dict[str, Any],
    sink: Any,
    step: int,
    *,
    disable_block_swap: bool = False,
    optimizer: Any = None,
) -> None:
    """Generate preview images on the main process and log them via the tracking sink."""
    preview_cfg = get_preview_config(config)
    prompts = normalize_preview_prompts(preview_cfg)
    if not prompts:
        return

    if model.name == "sdxl":
        preview_runner = _run_sdxl_previews
        use_block_swap_hooks = True
    elif model.name in ("cosmos_predict2", "anima"):
        if config.get("pipeline_stages", 1) != 1:
            if is_main_process():
                print(
                    "Preview skipped: cosmos_predict2 previews require pipeline_stages = 1."
                )
            return
        preview_runner = _run_cosmos_previews
        use_block_swap_hooks = False
    else:
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
    if use_block_swap_hooks:
        model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)

    start = time.time()
    try:
        preview_runner(model, preview_cfg, prompts, sink, step)
    finally:
        if use_block_swap_hooks:
            empty_cuda_cache()
            model.prepare_block_swap_training()
        else:
            if hasattr(model, "restore_after_preview"):
                model.restore_after_preview()
            empty_cuda_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if optimizer is not None and hasattr(optimizer, "train") and callable(optimizer.train):
            optimizer.train()

    _dist_barrier()
    if is_main_process():
        print(f"Preview complete in {time.time() - start:.1f}s")


def _run_sdxl_previews(
    model: Any,
    preview_cfg: dict[str, Any],
    prompts: list[tuple[str, str]],
    sink: Any,
    step: int,
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
            _log_preview_image(
                name=name,
                prompt=prompt,
                image=image,
                sink=sink,
                preview_cfg=preview_cfg,
                step=step,
            )

    for m, was_training in zip(modules, prev_training):
        m.train(was_training)


def _run_cosmos_previews(
    model: Any,
    preview_cfg: dict[str, Any],
    prompts: list[tuple[str, str]],
    sink: Any,
    step: int,
) -> None:
    base_seed = int(preview_cfg.get("seed", 0))
    seed_stride = int(preview_cfg.get("seed_stride", 1))

    if hasattr(model, "prepare_preview_memory"):
        model.prepare_preview_memory(preview_cfg)

    print(f"Running preview at step {step} ({len(prompts)} prompt(s))")

    for idx, (name, prompt) in enumerate(prompts):
        seed = base_seed + step * seed_stride + idx
        image = model.generate_preview_image(preview_cfg, prompt, step, seed)
        _log_preview_image(
            name=name,
            prompt=prompt,
            image=image,
            sink=sink,
            preview_cfg=preview_cfg,
            step=step,
        )
        # Release the per-prompt VAE-decode peak before the next prompt so multi-prompt
        # previews don't accumulate toward an OOM.
        empty_cuda_cache()
