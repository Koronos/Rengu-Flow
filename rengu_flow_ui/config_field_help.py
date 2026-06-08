"""Field-level help text for the config form (sourced from repo docs where noted)."""

from __future__ import annotations

from typing import Any

from rengu_flow_ui.optim_kv_defaults import (
    SCHEDULER_RUNTIME_TOKEN_HINTS,
    SCHEDULER_RUNTIME_TOKENS,
)


def scheduler_runtime_tokens_help_detail(*, intro: str) -> str:
    """Compact runtime-token glossary for FieldHelpIcon (popover / doc drawer)."""
    lines = [intro, "", "Runtime tokens (string values → integers at train time):"]
    for token in SCHEDULER_RUNTIME_TOKENS:
        lines.append(f"• {token} — {SCHEDULER_RUNTIME_TOKEN_HINTS[token]}")
    return "\n".join(lines)


# path -> {summary, detail?, doc?}
FIELD_HELP: dict[str, dict[str, str]] = {
    "dataset": {
        "summary": "Training data: one dataset TOML or several merged at run time.",
        "detail": (
            "Each entry is a library ref (rengu-flow-dataset:<id> or rengu-flow-dataset:<id>:label) "
            "or a .toml path. The optional label after the id is only for reading the TOML; "
            "training resolves by id. Multiple entries merge all [[directory]] tables."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "output_dir": {
        "summary": "Root folder for training runs (timestamped subfolders).",
        "detail": "Each run creates a dated folder under this path with checkpoints, logs, and copied config.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "run_name": {
        "summary": "Optional suffix for the run folder name.",
        "detail": "Appended to the timestamp so you can recognize experiments in output_dir.",
        "doc": "docs/user/web-ui.md",
    },
    "resume_from_checkpoint": {
        "summary": "Resume weights/optimizer from a prior run (trainer flag).",
        "detail": "Distinct from the UI “Resume folder” on jobs, which passes --resume_from_checkpoint with a run path.",
        "doc": "docs/user/web-ui.md",
    },
    "model.type": {
        "summary": "Registered pipeline (sdxl, cosmos_predict2, …).",
        "detail": "Selects which model code loads (e.g. sdxl, cosmos_predict2).",
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "model.dtype": {
        "summary": "Default dtype for weights and compute (required).",
        "detail": (
            "Common: bfloat16, float16. Cosmos: VAE, text encoder (Qwen3/T5), adapters, "
            "and sensitive DiT parts (embedders, norms, 1D params). "
            "Bulk DiT weights use this too unless transformer_dtype is set. SDXL: UNet and encoders."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "model.checkpoint_path": {
        "summary": "SDXL base model — one .safetensors file or a Diffusers folder.",
        "detail": (
            "The full pretrained SDXL you downloaded (often a single large .safetensors). "
            "This is not a LoRA and not your dataset. Diffusers-style directories are also accepted."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "model.guidance": {
        "summary": "Legacy [model] CFG scale (TOML-only; hidden in UI).",
        "detail": (
            "Parsed and defaulted in config but not used by SDXL training. "
            "Use preview.guidance_scale for sample images during training."
        ),
    },
    "model.freeze_text_encoders": {
        "summary": "Train UNet only; freeze both CLIP text encoders.",
        "detail": "Reduces VRAM and speeds adapter training when you only need visual changes.",
        "doc": "docs/user/full-model-training-sdxl.md",
    },
    "model.transformer_path": {
        "summary": "Main image model — one .safetensors file (the big checkpoint you train).",
        "detail": (
            "Example: Anima preview or another Cosmos Predict2 release as a single .safetensors on disk. "
            "This file is the core diffusion model. Do not point here at the VAE or text encoder — "
            "those have their own fields below."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "model.vae_path": {
        "summary": "Image VAE — one .safetensors that encodes/decodes pixels for training.",
        "detail": (
            "Usually the Qwen image VAE paired with Cosmos/Anima (e.g. qwen_image_vae.safetensors). "
            "Training reads your images through this file to build latent caches."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "model.llm_path": {
        "summary": "Text encoder (Qwen3) — .safetensors file or folder with weights.",
        "detail": (
            "Converts dataset captions into conditioning vectors. Required for modern Anima/Cosmos setups "
            "that use Qwen3. Set llm_adapter_lr = 0 unless you intentionally train the small LLM adapter."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "model.t5_path": {
        "summary": "Text encoder (T5) — use on older stacks instead of llm_path.",
        "detail": (
            "Single .safetensors or compatible layout. Pick either llm_path (Qwen3) or t5_path, not both, "
            "unless you know your checkpoint bundle requires a specific combination."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "model.llm_adapter_path": {
        "summary": "Optional extra weights on the text encoder (.safetensors).",
        "detail": "Leave empty unless your checkpoint pack includes a separate LLM adapter file to load.",
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "model.llm_adapter_lr": {
        "summary": "Learning rate for LLM adapter; 0 freezes it.",
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "model.cache_text_embeddings": {
        "summary": "Cache captions as embeddings once (faster training, more disk).",
        "detail": (
            "Strongly recommended for Cosmos/Anima: run --cache_only so training skips Qwen3 forward passes. "
            "Disabling only makes sense for debugging; it does not save meaningful VRAM once latents are cached."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "_has_adapter": {
        "summary": "Train LoRA/LoKr instead of full-model finetune.",
        "detail": "Uncheck when the model supports full finetune and you want all weights trainable.",
        "doc": "docs/user/full-model-training-sdxl.md",
    },
    "adapter.type": {
        "summary": "Adapter algorithm: lora or lokr.",
        "detail": "LoRA uses PEFT low-rank matrices; LoKr uses Kronecker (LyCORIS) factorization.",
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.rank": {
        "summary": "Adapter rank (capacity vs size).",
        "detail": "Higher rank = more parameters. For LoKr/LoRA, alpha defaults to rank unless set.",
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.dim": {
        "summary": "Alias for rank (Kohya-style configs).",
        "detail": (
            "Same as rank, named dim for Kohya-style configs. You must set either rank or dim. "
            "Typical values: 8, 16, 32 — higher rank means more capacity (and larger adapter)."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.init_from_existing": {
        "summary": "Start from an existing LoRA/LoKr (.safetensors or folder).",
        "detail": (
            "Path to a prior run export or downloaded adapter. "
            "Folder should contain adapter_model.safetensors (or a single .safetensors file)."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.dtype": {
        "summary": "Dtype for adapter weights (defaults to model dtype).",
        "detail": (
            "Override the precision of the adapter (LoRA/LoKr) weights. Leave unset to inherit "
            "model.dtype; set e.g. bfloat16/float16/float32 if you want the adapter to train at "
            "a different precision than the base model."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.dropout": {
        "summary": "LoRA dropout probability (default 0.0).",
        "detail": (
            "Probability of dropping adapter activations during training, in [0, 1]. Default "
            "0.0 (off). Small values (e.g. 0.05–0.1) can regularize and reduce overfitting on "
            "small datasets."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.factor": {
        "summary": "LoKr factorization hint (-1 = automatic).",
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.decompose_both": {
        "summary": "LoKr: decompose both Kronecker factors (default false).",
        "detail": (
            "LoKr only. When true, both Kronecker factors are low-rank decomposed instead of "
            "just one — more expressive but larger. Default false. Leave off unless you "
            "specifically need the extra capacity."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.full_matrix": {
        "summary": "LoKr: use full matrices for the second factor (default false).",
        "detail": (
            "LoKr only. When true, the second Kronecker factor uses full matrices instead of a "
            "low-rank approximation — higher capacity at the cost of size. Default false."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "optimizer.type": {
        "summary": "Any name the trainer can resolve: built-in registry, pytorch_optimizer, or module.Class.",
        "detail": (
            "Suggestions list common registry names. Type a custom class name if needed; "
            "required packages are installed automatically when training starts."
        ),
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "lr_scheduler": {
        "summary": "Registry name (cosine, linear, …) or a fully-qualified scheduler class.",
        "detail": (
            "Built-in names: constant, linear, cosine, rex, none. "
            "For custom classes, use Scheduler parameters and runtime token string values "
            "(total_steps, effective_total_steps, …) — open the (i) help on that field for "
            "meanings; resolved when training starts."
        ),
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "lr_scheduler_args.extra_params": {
        "summary": "Scheduler parameters (lr_min, constructor kwargs).",
        "detail": scheduler_runtime_tokens_help_detail(
            intro=(
                "Key-value rows map to [lr_scheduler_args] (not top-level warmup_steps — use the "
                "Warmup steps field). Built-in names and PyTorch class paths are pre-filled when "
                "you change scheduler type."
            ),
        ),
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "save_dtype": {
        "summary": "Dtype for exported adapter or full-model weights.",
        "detail": "Same keys as model dtype (bfloat16, float16, float32). Applied when writing save folders.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "save_full_model": {
        "summary": "Planned: export full backbone while keeping [adapter] (not implemented).",
        "detail": "Today: omit [adapter] for full-model export. See docs/spec/save-full-model-flag.md.",
        "doc": "docs/spec/save-full-model-flag.md",
    },
    "save_every_n_steps": {
        "summary": "Export model weights every N optimizer steps.",
        "detail": "Writes folders like step500 under the run directory. Can be combined with save_every_n_epochs.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "save_every_n_examples": {
        "summary": "Export every N training examples (converted to steps).",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "checkpoint_every_n_minutes": {
        "summary": "Write a resume checkpoint after this many minutes.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "max_checkpoints_to_keep": {
        "summary": "Limit DeepSpeed global_step* folders kept on disk.",
        "detail": "Oldest checkpoints are pruned; latest always points to the newest.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "max_model_exports_to_keep": {
        "summary": "Cap scheduled export folders (step*/epoch*); signal_step* exports are never auto-deleted.",
        "detail": "Works together with keep_exports_from_step when both are set.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "keep_exports_from_step": {
        "summary": "Drop scheduled exports below this training step before applying max_model_exports_to_keep.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "train_seed": {
        "summary": "RNG seed for training, tag dropout, and dataloader shuffle.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_datasets": {
        "summary": "Extra dataset(s) for periodic evaluation.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_every_n_steps": {
        "summary": "Run eval every N training steps.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_every_n_epochs": {
        "summary": "Run eval at the end of every N epochs.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_every_n_examples": {
        "summary": "Run eval every N examples (converted to steps).",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_before_first_step": {
        "summary": "Run one eval pass before the first training step.",
        "detail": "Useful for a loss baseline on validation data.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_gradient_accumulation_steps": {
        "summary": "Gradient accumulation steps used during eval.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "val_gap_enable": {
        "summary": "Deterministic held-out val loss + train-val gap (overfitting signal).",
        "detail": (
            "Forward-only probe on the existing eval cadence. The val curve uses the first "
            "eval dataset; a matched probe runs on a fixed train subset. Logs val/loss, "
            "train/probe and val/gap (val − train). No-ops if no eval dataset is configured."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "val_gap_probe_batches": {
        "summary": "Forward batches per gap probe (per timestep quantile).",
        "detail": "Smaller is faster. Keeps the probe cheap regardless of dataset size.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "preview.enabled": {
        "summary": "Generate sample images during training.",
        "doc": "docs/user/previews.md",
    },
    "preview.prompts": {
        "summary": "Preview configurations (one prompt or table per row in TOML).",
        "detail": "Each entry becomes one item in preview.prompts — use Add preview to manage the list.",
        "doc": "docs/user/previews.md",
    },
    "preview.negative_prompt": {
        "summary": "Negative prompt for preview generation.",
        "doc": "docs/user/previews.md",
    },
    "preview.width": {
        "summary": "Preview image width in pixels.",
        "doc": "docs/user/previews.md",
    },
    "preview.height": {
        "summary": "Preview image height in pixels.",
        "doc": "docs/user/previews.md",
    },
    "preview.num_inference_steps": {
        "summary": "Denoising steps per preview image.",
        "doc": "docs/user/previews.md",
    },
    "preview.guidance_scale": {
        "summary": "Classifier-free guidance scale for previews.",
        "doc": "docs/user/previews.md",
    },
    "preview.seed": {
        "summary": "Base RNG seed for preview images.",
        "doc": "docs/user/previews.md",
    },
    "preview.seed_stride": {
        "summary": "Seed offset per prompt index and training step.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_every_n_steps": {
        "summary": "Generate previews every N training steps.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_every_n_epochs": {
        "summary": "Generate previews every N epochs.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_before_first_step": {
        "summary": "Run previews once before step 1.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_offload_text_encoder": {
        "summary": "Move text encoder to CPU during Cosmos preview sampling.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_blocks_to_swap": {
        "summary": "DiT blocks on CPU between preview steps (Cosmos only).",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_save_png": {
        "summary": "Write PNG files under the run preview/ folder.",
        "doc": "docs/user/previews.md",
    },
    "monitoring.enable_wandb": {
        "summary": "Log metrics to Weights & Biases.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "monitoring.wandb_tracker_name": {
        "summary": "WandB project (entity/project) name.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "monitoring.wandb_run_name": {
        "summary": "Display name for this run in WandB.",
        "detail": "Defaults to the run directory path if omitted.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "monitoring.wandb_api_key": {
        "summary": "WandB API key (prefer WANDB_API_KEY env var).",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "model.transformer_dtype": {
        "summary": "Optional: dtype when loading the main DiT checkpoint only.",
        "detail": (
            "Defaults to model.dtype. Applies to most weights in transformer_path; "
            "VAE, text encoder, and embedder/norm layers still use model.dtype. "
            "Leave empty unless you need a different load precision (VRAM / load errors)."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "model.diffusion_model_dtype": {
        "summary": "DiT forward autocast dtype (Cosmos/Anima).",
        "detail": (
            "Sets training forward autocast (main.py). Defaults to model.dtype. "
            "When set and transformer_dtype is omitted, defaults copies this to transformer_dtype for checkpoint load."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "cache_dedup_text_embeddings": {
        "summary": "Reuse text-encoder outputs for duplicate captions during cache.",
        "detail": "Speeds --cache_only on tag-heavy datasets. Caption hash dedup in DatasetManager.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "ema_decay": {
        "summary": "EMA decay for trainable weights (CPU shadow).",
        "detail": "Float in (0, 1), e.g. 0.999. Omit to disable. Export from EMA is not automatic yet.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "disable_block_swap_for_eval": {
        "summary": "Load full backbone on GPU during eval when block swap is on.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "disable_block_swap_for_preview": {
        "summary": "Load full backbone on GPU during preview when block swap is on.",
        "doc": "docs/user/previews.md",
    },
    "optimizer.extra_params": {
        "summary": "All optimizer parameters (lr, betas, weight_decay, special keys).",
        "detail": (
            "Key-value rows merged into [optimizer] in TOML. Built-in registry names are "
            "pre-filled when you change optimizer type (lr, betas, type-specific keys). "
            "Custom class paths start empty until you add rows. See the user guide for "
            "per-optimizer parameter tables and links to PyTorch, bitsandbytes, and Prodigy docs."
        ),
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "image_micro_batch_size_per_gpu": {
        "summary": "Micro-batch for image-only steps when mixed with video.",
        "detail": "Integer or dict keyed by modality; falls back to micro_batch_size_per_gpu.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "steps_per_print": {
        "summary": "DeepSpeed console log interval (steps).",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "synthetic_num_batches": {
        "summary": "Use in-memory synthetic data for N batches (debug).",
        "detail": "Skips real dataset cache; dataset TOML is not used for training data.",
        "doc": "docs/user/dataset-config.md",
    },
    "partition_method": {
        "summary": "How model layers are split across pipeline-parallel stages.",
        "detail": (
            "Only matters with pipeline_stages > 1 (pipeline parallelism across multiple GPUs); "
            "ignored on a single stage / single GPU. parameters (default): balance VRAM by giving "
            "each stage a similar parameter count. uniform: same number of layers per stage. "
            "manual: set the exact split points yourself in partition_split. For one GPU, leave it "
            "alone — use blocks_to_swap + activation checkpointing to save VRAM instead."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "partition_split": {
        "summary": "Manual pipeline stage boundaries (layer indices).",
        "detail": (
            "Only used with pipeline_stages > 1 and partition_method = manual. JSON list of layer "
            "indices where each stage ends (e.g. [10, 20] for 3 stages)."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "reentrant_activation_checkpointing": {
        "summary": "Which PyTorch activation-checkpoint backend to use (reentrant vs non-reentrant).",
        "detail": (
            "Only applies when activation_checkpointing is true. PyTorch has two checkpoint backends: "
            "reentrant (legacy, re-enters autograd) and non-reentrant (newer, saved-tensor hooks, more "
            "flexible). Reentrant can be slightly faster on some models (cosmos_predict2 defaults it "
            "to true) but is more restrictive — leave it off unless your model benefits."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "x_axis_examples": {
        "summary": "TensorBoard/WandB x-axis uses example count instead of step.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "caching_batch_size": {
        "summary": "Batch size for the dataset latent/text cache pass.",
        "detail": "Larger values can speed cache but use more VRAM during caching.",
        "doc": "docs/user/dataset-config.md",
    },
    "cache_num_proc": {
        "summary": "CPU processes for metadata map and latent/TE cache.",
        "detail": "Default caps at 8. Lower if RAM is tight during --cache_only.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "cache_keep_in_memory": {
        "summary": "Load resumed cache slices fully into RAM.",
        "detail": "false saves RAM on large datasets; train reads still use OS page cache.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_num_workers": {
        "summary": "Subprocess workers that load batches in parallel during training.",
        "detail": (
            "0 (default) loads in the main process — fine here because cached (v2) latents read "
            "fast, so the GPU rarely waits on data. Raise to 2–4 on Linux only if you see the GPU "
            "idle waiting for batches (>0 is problematic on Windows/macOS). When kept at 0, "
            "dataloader_prefetch is the companion overlap knob."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_prefetch": {
        "summary": "Background thread that preloads the next batch while the current step trains.",
        "detail": (
            "Off by default: loading is synchronous, which is fine because cached (v2) reads are "
            "cheap and the GPU rarely waits. Only applies when dataloader_num_workers = 0. Turn it "
            "on to overlap loading the next batch (the next step's data, not the whole epoch) with "
            "the current step's compute — useful if you see the GPU idle between steps."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_pin_memory": {
        "summary": "Page-locked CPU tensors for faster CUDA copies.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_prefetch_factor": {
        "summary": "Batches prefetched per worker when num_workers > 0.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_persistent_workers": {
        "summary": "Keep DataLoader worker processes between epochs.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "max_steps": {
        "summary": "Stop training after this many optimizer steps.",
        "detail": "Overrides epoch count when set.",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "warmup_steps": {
        "summary": "Trainer-level linear LR warmup before the main scheduler.",
        "detail": (
            "Top-level warmup_steps in TOML. When > 0 and lr_scheduler is not none, training wraps "
            "the resolved scheduler with a short LinearLR warmup phase — for built-in names and "
            "fully-qualified PyTorch classes alike. Not a [lr_scheduler_args] constructor kwarg; "
            "use [lr_scheduler_args] only if your class defines its own warmup under a different "
            "parameter name."
        ),
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "epochs": {
        "summary": "Number of passes over the dataset.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "micro_batch_size_per_gpu": {
        "summary": "Samples per GPU per forward/backward micro-step.",
        "detail": (
            "Anima/Cosmos LoKR on 16 GB: 1 is the practical default; higher micro-batch raised VRAM "
            "and slowed per-step time in tuning — prefer grad accumulation for effective batch size."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "gradient_accumulation_steps": {
        "summary": "Micro-steps to accumulate before optimizer step.",
        "detail": (
            "Increases effective batch without raising micro_batch VRAM, but each optimizer step does more work "
            "(e.g. 2× accum ≈ ~2× wall time per step in Anima smokes)."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "gradient_clipping": {
        "summary": "Max gradient norm (0 disables).",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "logging_steps": {
        "summary": "Log metrics every N optimizer steps.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "pipeline_stages": {
        "summary": "DeepSpeed pipeline parallel stages (typically = num GPUs).",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "activation_checkpointing": {
        "summary": "Save VRAM by recomputing activations in the backward pass instead of storing them.",
        "detail": (
            "During the forward pass the intermediate activations are dropped and re-computed during "
            "backprop — large activation-memory savings for a small extra compute cost (an extra "
            "forward per checkpointed block). Keep it true on small GPUs; false can OOM. Values: "
            "true (full, lowest VRAM, safe default), false (fastest, highest VRAM — OOMs at high res), "
            "'selective' (SAC: keeps the expensive attention activations and recomputes only cheaper "
            "ops — quality-neutral, ~4% faster at 1024 but uses MORE VRAM than full, so it needs "
            "headroom — ~9.5 GB at 1024/batch2; NOT for small GPUs), or 'unsloth' (a faster "
            "checkpointing kernel for supported models). Tune SAC with selective_checkpoint_save_ops."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "selective_checkpoint_save_ops": {
        "summary": "Extra op types SAC keeps instead of recomputing (the VRAM/speed dial).",
        "detail": (
            "Only applies when activation_checkpointing = 'selective'. SAC always keeps the attention "
            "outputs; this comma-separated list adds more aten ops to keep resident, e.g. "
            "'mm,addmm,bmm'. More kept = less recompute (a touch faster) but MORE VRAM; empty = keep "
            "attention only (the lightest SAC). On a tight card leave empty or use "
            "activation_checkpointing = true. (Adding mm/addmm/bmm gave no extra speed at 1024 — "
            "torch.compile's partitioner already handles those — so attention-only is enough.)"
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "activation_checkpoint_interval": {
        "summary": "Checkpoint every N transformer blocks (1 = every block).",
        "detail": (
            "Only applies when activation_checkpointing is on. 1 (default) checkpoints every block "
            "(most VRAM-saving, most recompute); higher keeps more activations to recompute less. "
            "Measured neutral on Cosmos at 1024 — leave at 1 unless you have a specific reason."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "blocks_to_swap": {
        "summary": "Stream UNet/DiT blocks between CPU and GPU so only a few stay resident.",
        "detail": (
            "SDXL and Cosmos Predict2; requires pipeline_stages = 1. Works for both adapter and "
            "full-model training: adapters keep their small trainable params resident, while "
            "full-model (no [adapter]) additionally requires optimizer.gradient_release so the "
            "per-parameter step runs while the block is on the GPU. On ~8 GB cards this is the "
            "lever that makes a full SDXL fine-tune fit (e.g. blocks_to_swap = 6 → ~4.3 GB). "
            "Combine with activation_checkpointing. With gradient_release set, the Block-swap "
            "prefetch toggle appears. See the low-VRAM recipe in the doc."
        ),
        "doc": "docs/developer/vram-optimization.md",
    },
    "block_swap_prefetch": {
        "summary": "Overlap block transfers on a side CUDA stream (situational; off by default).",
        "detail": (
            "Pins the swapped blocks' CPU memory and prefetches the next block while the current "
            "one computes, to hide CPU↔GPU transfer latency. Only takes effect with "
            "optimizer.gradient_release (full-model training — adapter runs keep trainable params "
            "resident and force prefetch off) and when blocks_to_swap leaves ≥2 blocks resident, so "
            "it only helps where there is VRAM headroom (bigger GPUs / native Linux). On an 8 GB "
            "WSL2 box it is counterproductive — the extra resident block pushes past the "
            "sysmem-paging threshold and steps get slower — so leave it off there."
        ),
        "doc": "docs/developer/vram-optimization.md",
    },
    "compile": {
        "summary": "torch.compile on the DeepSpeed pipeline model (recommended for long runs).",
        "detail": (
            "Applies torch.compile to the whole pipeline model (UNet/DiT). "
            "Cosmos/Anima (≥1000 steps): after warmup, steady steps were ~0.51s vs ~0.68–0.70s without compile. "
            "Early steps are slower while Inductor/CUDA graphs build; short test runs are not representative. "
            "Tune behavior with compile_mode and compile_dynamic below."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "compile_mode": {
        "summary": "Inductor mode for torch.compile (default when unset).",
        "detail": (
            "Maps to torch.compile(mode=...). Options: 'default'; 'reduce-overhead' (CUDA-graph based, "
            "lowest per-step launch overhead — best for fixed-shape steps, costs a little extra VRAM for "
            "graph pools and can further reduce step time over default; benchmark per setup); "
            "'max-autotune' and 'max-autotune-no-cudagraphs' (longer autotuning warmup). Only applies when compile is on."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "compile_dynamic": {
        "summary": "Pass dynamic=True to torch.compile for varying input shapes.",
        "detail": (
            "Maps to torch.compile(dynamic=True). Leave off for fixed-shape training; turn on if input "
            "shapes change between steps so Inductor avoids recompiling per shape. Only applies when compile is on. "
            "Note: dynamic shapes defeat the on-disk compile cache (compile_disk_cache) — the cache only "
            "helps with fixed (static) shapes."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "compile_disk_cache": {
        "summary": "Persist torch.compile's kernels to disk so re-runs skip recompilation (static shapes only).",
        "detail": (
            "Values: 'auto' (default — enable only when compile is on AND compile_dynamic is off, where it "
            "actually helps), true (always on; warns if dynamic, since it won't help), false (never). "
            "Sets the Inductor/Triton on-disk caches so a second run with the SAME static shapes reuses the "
            "compiled kernels instead of recompiling (~30s of compile saved per run). With compile_dynamic "
            "on it is a no-op (dynamic-shape guards never match). The cache must live on an ext4-style "
            "filesystem (255-char filenames); on an encrypted home (~143-char limit) it auto-disables with a "
            "warning — set compile_cache_dir to an ext4 path. Only applies when compile is on."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "compile_cache_dir": {
        "summary": "Where the on-disk compile cache lives (default: <repo>/.compile_cache).",
        "detail": (
            "Directory for compile_disk_cache. Default is a folder beside the repo (ext4). Point it at an "
            "ext4 path if your repo/home is on an encrypted or short-filename filesystem. Only used when "
            "compile_disk_cache is active."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "checkpoint_every_n_epochs": {
        "summary": "Write DeepSpeed checkpoints every N epochs.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "save_every_n_epochs": {
        "summary": "Export adapter/model files every N epochs.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
}


def _apply_field_help(field: dict[str, Any], meta: dict[str, str]) -> None:
    if not field.get("description") and meta.get("summary"):
        field["description"] = meta["summary"]
    if meta.get("detail"):
        field["help"] = meta["detail"]
    elif meta.get("summary"):
        field["help"] = meta["summary"]
    if meta.get("doc"):
        field["doc_path"] = meta["doc"]


def enrich_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Attach help, doc_path, and ensure every field has UI help text."""
    for section in schema.get("sections", []):
        for field in section.get("fields", []):
            path = field.get("path")
            if not path:
                continue
            meta = FIELD_HELP.get(path)
            if meta:
                _apply_field_help(field, meta)
            if not field.get("help"):
                field["help"] = field.get("description") or field.get("label") or path
            if not field.get("description"):
                field["description"] = field["help"]
    return schema
