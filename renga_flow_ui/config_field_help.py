"""Field-level help text for the config form (sourced from repo docs where noted)."""

from __future__ import annotations

from typing import Any

# path -> {summary, detail?, doc?}
FIELD_HELP: dict[str, dict[str, str]] = {
    "dataset": {
        "summary": "Dataset TOML: one file with one or more [[directory]] folders.",
        "detail": "Pick from the Datasets library or type a path. Compose collections in the Datasets tab, then reference the merged .toml here.",
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
        "summary": "Guidance strength during SDXL training (CFG scale).",
        "detail": "How strongly the model follows the text vs unconditional signal. Often 1.0 for LoRA; raise for preview-like behavior.",
    },
    "model.freeze_text_encoders": {
        "summary": "Train UNet only; freeze both CLIP text encoders.",
        "detail": "Reduces VRAM and speeds adapter training when you only need visual changes.",
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
    },
    "adapter.dropout": {
        "summary": "LoRA dropout probability.",
    },
    "adapter.factor": {
        "summary": "LoKr factorization hint (-1 = automatic).",
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.decompose_both": {
        "summary": "LoKr: decompose both Kronecker factors.",
    },
    "adapter.full_matrix": {
        "summary": "LoKr: use full matrix for the second factor.",
    },
    "optimizer.type": {
        "summary": "Any name the trainer can resolve: built-in, optional deps, pytorch_optimizer, or module.Class.",
        "detail": "Suggestions are common names only. Type a custom name; the UI checks if it imports in this environment.",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "optimizer.lr": {
        "summary": "Base learning rate.",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "optimizer.weight_decay": {
        "summary": "L2-style weight decay.",
    },
    "optimizer.betas": {
        "summary": "Adam-style beta coefficients as a list.",
    },
    "optimizer.momentum": {
        "summary": "SGD momentum.",
    },
    "optimizer.gradient_release": {
        "summary": "Memory-saving optimizer mode (requires pipeline_stages = 1).",
    },
    "lr_scheduler": {
        "summary": "Registry name (cosine, linear, …) or a fully-qualified scheduler class.",
        "detail": "Suggestions are built-in schedulers. Custom class paths are validated when you save or on blur in the form.",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "warmup_steps": {
        "summary": "Linear LR warmup steps at start of training.",
    },
    "lr_scheduler_args.lr_min": {
        "summary": "Minimum LR for cosine schedules.",
    },
    "epochs": {
        "summary": "Number of passes over the dataset.",
    },
    "max_steps": {
        "summary": "Stop after this many optimizer steps (overrides epoch count).",
    },
    "micro_batch_size_per_gpu": {
        "summary": "Samples per GPU per forward/backward micro-step.",
    },
    "gradient_accumulation_steps": {
        "summary": "Micro-steps to accumulate before optimizer step.",
    },
    "gradient_clipping": {
        "summary": "Max gradient norm (0 disables).",
    },
    "logging_steps": {
        "summary": "Log metrics every N optimizer steps.",
    },
    "pipeline_stages": {
        "summary": "DeepSpeed pipeline parallel stages.",
    },
    "activation_checkpointing": {
        "summary": "Trade compute for VRAM (true, false, or unsloth).",
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "blocks_to_swap": {
        "summary": "SDXL: offload UNet blocks to CPU (not on Cosmos yet).",
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "compile": {
        "summary": "torch.compile — recommended for long Cosmos/Anima runs.",
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
    "save_dtype": {
        "summary": "Dtype for exported adapter or full-model weights.",
        "detail": "Same keys as model dtype (bfloat16, float16, float32). Applied when writing save folders.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "save_full_model": {
        "summary": "Save the full trained model, not only the small adapter file.",
        "detail": "Use for full finetune runs. With LoRA/LoKr, leave off so exports stay as adapter_model.safetensors.",
        "doc": "docs/user/checkpoint-and-save.md",
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
    "disable_block_swap_for_eval": {
        "summary": "Disable block swap during eval for stable metrics.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "preview.enabled": {
        "summary": "Generate sample images during training.",
        "doc": "docs/user/previews.md",
    },
    "preview.prompts": {
        "summary": "Prompts for preview images (strings or tables).",
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
    "disable_block_swap_for_preview": {
        "summary": "Disable block swap during preview (full GPU inference).",
        "doc": "docs/user/previews.md",
    },
    "monitoring.enable_wandb": {
        "summary": "Log metrics to Weights & Biases.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "monitoring.enable_status_file": {
        "summary": "Write status.json for the web UI (low overhead).",
        "doc": "docs/user/web-ui.md",
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
        "summary": "Optional (not used yet): forward-pass dtype.",
        "detail": (
            "Accepted in TOML but not applied by cosmos_predict2 training today. "
            "Leave unset; use model.dtype (and transformer_dtype only if needed)."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "optimizer.beta2_half_life": {
        "summary": "Recompute Adam beta2 from global batch size.",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "optimizer.kahan_buffer_offload": {
        "summary": "Offload GenericOptim Kahan buffer to CPU (saves VRAM).",
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
        "summary": "How layers are split across pipeline stages.",
        "detail": "parameters (by param count), uniform, or manual with partition_split.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "partition_split": {
        "summary": "Manual pipeline stage boundaries (layer indices).",
        "detail": "Required when partition_method = manual. JSON list of split points.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "reentrant_activation_checkpointing": {
        "summary": "Reentrant PyTorch checkpoint (recommended for Cosmos with AC on).",
        "detail": (
            "When activation_checkpointing is true, cosmos_predict2 defaults this to true. "
            "Anima LoKR smokes: ~3% faster than false, stable over long runs."
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
    "cache_format": {
        "summary": "On-disk layout for latent and text-embedding cache.",
        "detail": (
            "v2 (default): stacked bf16 tensors + SQLite metadata — faster reads, less disk. "
            "v1: legacy pickle shards. Changing format invalidates the fingerprint; run "
            "--regenerate_cache or delete cache/<model>/ latents and text_embeddings_* folders. "
            "Existing v1 dirs are auto-detected on load when unchanged."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_num_workers": {
        "summary": "Parallel workers loading cached batches during training.",
        "detail": "Try 2–4 on Linux if the GPU waits on disk. Use dataloader_prefetch instead when 0.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_prefetch": {
        "summary": "Background thread loads the next batch while training.",
        "detail": "Only applies when dataloader_num_workers is 0.",
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
    },
    "warmup_steps": {
        "summary": "Linear LR warmup steps at start of training.",
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
        "summary": "Trade compute for VRAM (true, false, or unsloth).",
        "detail": (
            "Cosmos/Anima: keep true on ~16 GB GPUs — false caused OOM in LoKR tuning. "
            "With true, reentrant_activation_checkpointing defaults to true for cosmos_predict2 (~3% faster in smokes)."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "blocks_to_swap": {
        "summary": "SDXL: offload UNet blocks to CPU (adapter training only).",
        "detail": (
            "Works on SDXL LoRA/LoKr. Not implemented for cosmos_predict2 — leave 0 or training will error. "
            "On Anima, use activation_checkpointing (and optional unsloth) instead."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "compile": {
        "summary": "torch.compile on the DeepSpeed pipeline (recommended for long runs).",
        "detail": (
            "Cosmos/Anima (≥1000 steps): after warmup, steady steps were ~0.51s vs ~0.68–0.70s without compile. "
            "Early steps are slower while graphs build; short test runs are not representative."
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
    "optimizer.weight_decay": {
        "summary": "L2-style weight decay.",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "optimizer.betas": {
        "summary": "Adam-style beta coefficients as a list.",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "optimizer.momentum": {
        "summary": "SGD momentum.",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "optimizer.gradient_release": {
        "summary": "Memory-saving optimizer mode (requires pipeline_stages = 1).",
        "doc": "docs/user/optimizer-and-scheduler.md",
    },
    "lr_scheduler_args.lr_min": {
        "summary": "Minimum LR for cosine schedules.",
        "doc": "docs/user/optimizer-and-scheduler.md",
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
