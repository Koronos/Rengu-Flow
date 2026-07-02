"""Set default values on config dict (aligned with diffusion-pipe set_config_defaults)."""

from __future__ import annotations

import os
from typing import Any

from rengu_flow.config.validation import ConfigValidationError

try:
    import torch
    _ = torch.float32  # ensure dtype attrs are loadable (e.g. avoid broken torch installs)
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

# Dtype names as in TOML -> torch.dtype when torch is available; else keep as string.
if _TORCH_AVAILABLE:
    DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8": torch.float8_e4m3fn,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
    }
else:
    DTYPE_MAP = {k: k for k in ("float32", "float16", "bfloat16", "float8", "float8_e4m3fn", "float8_e5m2")}


def set_config_defaults(config: dict[str, Any]) -> None:
    """Apply default values to config in place.

    Replicates the logic of diffusion-pipe train.set_config_defaults so that
    the same TOML files remain valid. Requires config to have 'model' (with 'dtype')
    and optionally 'adapter'. Sets a default save_every_n_epochs so configs
    without any save_* still validate.
    """
    # One save_* default so a config without any is still valid.
    config.setdefault("save_every_n_epochs", 1)
    config.setdefault("output_dir", "output")
    config.setdefault("pipeline_stages", 1)
    config.setdefault("activation_checkpointing", False)
    # Retired modes (see docs/EXPERIMENTS_GRAVEYARD.md): 'selective' (SAC) and
    # 'unsloth' were superseded by 'auto' (compile's memory-budget partitioner,
    # faster AND lighter than SAC). Degrade old configs to the safe full mode.
    if config["activation_checkpointing"] in ("selective", "unsloth"):
        print(
            f"[checkpoint] activation_checkpointing='{config['activation_checkpointing']}' was retired; "
            "falling back to full checkpointing (true). For the speed gains use "
            "activation_checkpointing='auto' with compile=true (better speed AND VRAM).",
            flush=True,
        )
        config["activation_checkpointing"] = True
    config.setdefault("reentrant_activation_checkpointing", False)
    config.setdefault("warmup_steps", 0)
    if "save_dtype" in config:
        config["save_dtype"] = DTYPE_MAP[config["save_dtype"]]

    model_config = config["model"]
    model_dtype_str = model_config["dtype"]
    model_config["dtype"] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get("transformer_dtype", None):
        model_config["transformer_dtype"] = DTYPE_MAP[transformer_dtype]
    if diffusion_model_dtype := model_config.get("diffusion_model_dtype", None):
        model_config["diffusion_model_dtype"] = DTYPE_MAP[diffusion_model_dtype]
        if str(model_config.get("type", "")).lower() in ("cosmos_predict2", "anima"):
            model_config.setdefault("transformer_dtype", model_config["diffusion_model_dtype"])
    model_config.setdefault("guidance", 1.0)
    if str(model_config.get("type", "")).lower() in ("cosmos_predict2", "sdxl", "krea2"):
        model_config.setdefault("cache_text_embeddings", True)
        if config.get("activation_checkpointing") and not config.get("blocks_to_swap"):
            config.setdefault("reentrant_activation_checkpointing", True)

    if str(model_config.get("type", "")).lower() == "krea2":
        # The 4B Qwen3-VL dominates caching and each caption otherwise re-encodes once per
        # resolution bucket. Dedup is disk-backed (mmap spill, ~zero RAM; see
        # data/manager.py), so it is safe at any dataset size.
        config.setdefault("cache_dedup_text_embeddings", True)

    if str(model_config.get("type", "")).lower() in ("cosmos_predict2", "anima", "krea2"):
        # Frozen-base quantization knobs (A/B; default-off, mutually exclusive).
        model_config.setdefault("transformer_fp8_matmul", False)
        model_config.setdefault("transformer_4bit", False)
        model_config.setdefault("fp8_matmul_dtype", "e5m2")
        if model_config["transformer_fp8_matmul"] and model_config["transformer_4bit"]:
            raise ConfigValidationError(
                "model.transformer_fp8_matmul and model.transformer_4bit are mutually "
                "exclusive; enable only one."
            )
        if model_config["fp8_matmul_dtype"] not in ("e5m2", "e4m3"):
            raise ConfigValidationError(
                "model.fp8_matmul_dtype must be 'e5m2' (default) or 'e4m3'."
            )

    if str(model_config.get("type", "")).lower() in ("cosmos_predict2", "anima", "krea2"):
        preview_cfg = config.get("preview")
        if isinstance(preview_cfg, dict):
            is_krea2 = str(model_config.get("type", "")).lower() == "krea2"
            # Krea 2 reference settings: 28 steps, CFG 4.5 (cond + g*(cond - uncond)).
            preview_cfg.setdefault("num_inference_steps", 28 if is_krea2 else 20)
            preview_cfg.setdefault("guidance_scale", 4.5 if is_krea2 else 4.0)
            preview_cfg.setdefault("negative_prompt", "")
            preview_cfg.setdefault("width", 1024)
            preview_cfg.setdefault("height", 1024)
            preview_cfg.setdefault("preview_offload_text_encoder", True)
            preview_cfg.setdefault("preview_offload_dit_for_decode", False)

    if "adapter" in config:
        adapter_config = config["adapter"]
        # Normalize dim -> rank (Kohya-style alias) so rest of code uses only "rank"
        if "rank" not in adapter_config and "dim" in adapter_config:
            adapter_config["rank"] = adapter_config["dim"]
        adapter_type = adapter_config["type"]
        if adapter_type in ("lora", "lokr") or adapter_type.startswith("lycoris_"):
            if "alpha" in adapter_config:
                raise ConfigValidationError(
                    "Remove alpha from [adapter]; rengu-flow sets alpha=rank for Comfy-compatible saves."
                )
            adapter_config["alpha"] = adapter_config["rank"]
        if adapter_type == "lora":
            adapter_config.setdefault("dropout", 0.0)
            adapter_config.setdefault("dtype", model_dtype_str)
            adapter_config["dtype"] = DTYPE_MAP[adapter_config["dtype"]]
        elif adapter_type == "lokr":
            adapter_config.setdefault("factor", -1)
            adapter_config.setdefault("decompose_both", False)
            adapter_config.setdefault("full_matrix", False)
            adapter_config.setdefault("dtype", model_dtype_str)
            adapter_config["dtype"] = DTYPE_MAP[adapter_config["dtype"]]
        elif adapter_type.startswith("lycoris_"):
            # Torch-free catalog module; per-algo tunables mirror lycoris 3.4.0.
            from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES, apply_lycoris_defaults

            if adapter_type not in LYCORIS_ADAPTER_TYPES:
                known = ", ".join(LYCORIS_ADAPTER_TYPES)
                raise ConfigValidationError(
                    f"adapter.type {adapter_type!r} is not a known LyCORIS type. Available: {known}."
                )
            apply_lycoris_defaults(adapter_config)
            adapter_config.setdefault("dtype", model_dtype_str)
            adapter_config["dtype"] = DTYPE_MAP[adapter_config["dtype"]]
        else:
            raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")

    config.setdefault("epochs", 1)
    config.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("micro_batch_size_per_gpu", 1)
    # Per-resolution micro batch (e.g. { 512 = 2, 1024 = 1 }): TOML always parses
    # table keys as strings, but the dataset's resolution->batch lookup
    # (dataset.py post_init) compares keys numerically. Normalize here so the
    # dict form actually works from a TOML file.
    for _mb_key in ("micro_batch_size_per_gpu", "image_micro_batch_size_per_gpu"):
        _mb = config.get(_mb_key)
        if isinstance(_mb, dict):
            config[_mb_key] = {
                (int(k) if isinstance(k, str) and k.isdigit() else k): int(v)
                for k, v in _mb.items()
            }
    config.setdefault("partition_method", "parameters")
    config.setdefault("partition_split", None)
    config.setdefault("lr_scheduler", "constant")
    config.setdefault("lr_scheduler_args", {})
    config.setdefault("logging_steps", 1)
    config.setdefault("eval_datasets", [])
    config.setdefault("caching_batch_size", 1)
    config.setdefault("cache_num_proc", min(8, os.cpu_count() or 1))
    config.setdefault("cache_keep_in_memory", False)
    config.setdefault("train_seed", 42)
    config.setdefault("dataloader_num_workers", 0)
    # Default on: with prefetch off the next-batch preload runs synchronously on the
    # main thread before the step timer starts — a measured ~67-72 ms/step GPU stall
    # (~9% schedule-weighted wall-clock, 18% @512) that no bench number surfaced.
    # Same data either way; see docs/DIT_TRAINING_SPEED_RESEARCH (2026-06-09 x-ray).
    config.setdefault("dataloader_prefetch", True)
    config.setdefault("dataloader_pin_memory", False)
    config.setdefault("dataloader_prefetch_factor", 2)
    config.setdefault("dataloader_persistent_workers", True)
    config.setdefault("eval_gradient_accumulation_steps", 1)
    config.setdefault("eval_every_n_steps", None)
    config.setdefault("eval_every_n_epochs", None)
    config.setdefault("eval_every_n_examples", None)
    config.setdefault("eval_before_first_step", True)
    config.setdefault("disable_block_swap_for_eval", False)
    # Deterministic generalization probe: held-out val loss + matched train probe + GAP
    # (val − train), the fast overfitting signal. Forward-only, runs on the existing eval
    # cadence. No-ops gracefully when no val set (eval_datasets) is available.
    config.setdefault("val_gap_enable", True)
    config.setdefault("val_gap_probe_batches", 8)
    config.setdefault("cache_dedup_text_embeddings", False)
    # Stream saved activations to pinned CPU RAM over side streams (see
    # training/activation_offload.py). Pairs with a raised
    # activation_memory_budget: the budget picks save-vs-recompute, the
    # offloader moves the saved ones off the GPU.
    config.setdefault("activation_offload", False)
    config.setdefault("activation_offload_min_tensor_mb", 4.0)
    config.setdefault("activation_offload_max_ram_gb", None)
    config.setdefault("activation_offload_prefetch_mb", 512.0)
    config.setdefault("compile", False)
    # OOM-proof activation budget: on CUDA OOM with activation_checkpointing
    # = "auto", lower the budget and recompile instead of crashing the run.
    config.setdefault("activation_budget_backoff", True)
    # TorchInductor/Triton disk cache for torch.compile. "auto" enables the cache
    # only when compile is on AND compile_dynamic is off (with dynamic shapes the
    # shape guards never match, so the cache misses and adds a cold-population
    # penalty). true = always enable; false = never. The cache dir must live on an
    # ext4 (255-char filename) filesystem; the worker runs a safety check and
    # disables caching if the chosen dir has a short filename limit. Default dir is
    # <repo_root>/.compile_cache (the renga-flow repo lives on ext4).
    config.setdefault("compile_disk_cache", "auto")
    config.setdefault("compile_cache_dir", None)
    config.setdefault("x_axis_examples", False)
    config.setdefault("steps_per_print", 1)
    # Experiment tracking (rengu_track). The sink fans out to the selected backends; the local
    # store is the run_dir (TB event files + run.json + run_events.jsonl). `enabled=false` is the
    # full disconnect (no-op sink). wandb is opt-in by adding "wandb" to `backends`.
    config.setdefault("tracking", {})
    tracking = config["tracking"]
    tracking.setdefault("enabled", True)
    tracking.setdefault("backends", ["manifest", "tensorboard"])
    tracking.setdefault("capture_lineage", True)
    sampler = tracking.setdefault("system_sampler", {})
    sampler.setdefault("enabled", True)
    sampler.setdefault("interval_sec", 10)
    wandb_cfg = tracking.setdefault("wandb", {})
    wandb_cfg.setdefault("project", "rengu-flow")
    wandb_cfg.setdefault("run_name", None)
    wandb_cfg.setdefault("api_key", None)

    train_cfg = config.setdefault("train", {})
    oom_skip = train_cfg.setdefault("oom_skip", {})
    oom_skip.setdefault("enabled", False)
    oom_skip.setdefault("max_consecutive", 3)
    oom_skip.setdefault("clear_cache_on_skip", True)
