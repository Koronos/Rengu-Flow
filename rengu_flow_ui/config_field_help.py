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
        "detail": (
            "Keeps both text encoders frozen so their weights don't update — reduces VRAM and speeds up "
            "each training step. Turn off only if you need the model to respond differently to new prompt words "
            "(e.g. training a new concept with a novel trigger token)."
        ),
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
        "summary": "Finetune-only LR for the embedded Qwen3 LLM adapter; frozen by default (0), set a value to train it.",
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "model.cache_text_embeddings": {
        "summary": "Cache captions as embeddings once (faster training, more disk).",
        "detail": (
            "Strongly recommended for Cosmos/Anima: training skips the live Qwen3 forward "
            "(~22 ms/step) and frees ~1.2 GB VRAM (more activation_memory_budget headroom). "
            "For dropout regularization WITH the cache on, enable tag_dropout (and/or "
            "cached_caption_shuffle) on the dataset: K = cached_caption_variants dropout/shuffle "
            "variants per caption are baked into the cache. K = 1 bakes one fixed variant; K >= 2 "
            "rotates them across epochs without inflating them."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "_has_adapter": {
        "summary": "Train a LoRA/LoKr/LyCORIS adapter (checked) or fine-tune all weights directly (unchecked).",
        "detail": (
            "When checked, only the adapter parameters train — the base model stays frozen and the adapter file "
            "is small and portable (ComfyUI/Forge compatible). "
            "Uncheck to train all weights (full finetune); also set optimizer.gradient_release = true and "
            "blocks_to_swap if the full model does not fit in VRAM."
        ),
        "doc": "docs/user/full-model-training-sdxl.md",
    },
    "adapter.type": {
        "summary": "Adapter algorithm: PEFT LoRA, LoKr, or one of the LyCORIS networks.",
        "detail": (
            "LoRA exports the widely compatible lora.safetensors; LoKr and every LyCORIS type export "
            "adapter_model.safetensors with kohya-style keys (loads in ComfyUI). LyCORIS types use the "
            "lycoris-lora library: LoCon (classic LoRA), LoHa (Hadamard product), LoKr (Kronecker product), "
            "DoRA (LoCon + weight decomposition, often closer to full finetune at low rank), DyLoRA (trains "
            "nested ranks so you can truncate after training), GLoRA (adds input-side adaptation), Diag-OFT "
            "and BOFT (orthogonal rotations that preserve base-model weight norms)."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.rank": {
        "summary": "Controls how many parameters the adapter adds (typical values: 8, 16, 32).",
        "detail": (
            "Higher rank = more adapter parameters and larger saved file. "
            "Alpha defaults to rank (scale 1.0); raise alpha for stronger adapter effect, lower for weaker. "
            "If results look blurry or under-trained, try a higher rank; if the adapter overrides the base too aggressively, lower rank or alpha."
        ),
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
        "summary": "Probability of dropping adapter activations during training (default 0.0 = off).",
        "detail": (
            "Probability of dropping adapter activations during training, in [0, 1]. Default 0.0 (off). "
            "Small values (e.g. 0.05–0.1) can regularize and reduce overfitting on small datasets. "
            "Applies to LoRA and all LyCORIS types."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.rank_dropout": {
        "summary": "Zeroes random rank rows of the adapter each step (default 0.0 = off).",
        "detail": (
            "Per-step dropout over the adapter's rank dimension, in [0, 1]. A light value (0.05-0.1) "
            "regularizes inside the low-rank factorization; try it if a small dataset overfits before "
            "the style converges."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.module_dropout": {
        "summary": "Skips the entire adapter on a layer with this probability per step (default 0.0 = off).",
        "detail": (
            "Each step, an adapted layer keeps base-only behavior with this probability, so the frozen "
            "model stays visible during training. Use a small value if the adapter drowns out the base "
            "model's general knowledge."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.train_conv": {
        "summary": "Also adapts Conv2d layers (resnet/sampling convs), not just Linear (default off).",
        "detail": (
            "Off matches the lora/lokr targets (attention + MLP Linear layers). On attaches the network "
            "to every Conv2d in the UNet blocks too — larger file and slower steps, more grip on texture. "
            "Turn on if fine surface detail refuses to transfer at a rank that otherwise works. Not "
            "supported for lycoris_dylora."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.use_tucker": {
        "summary": "Tucker-decomposes conv kernels (only acts when Train conv layers is on).",
        "detail": (
            "Adds a small core tensor (lora_mid) between the down/up factors on non-1x1 convs, cutting "
            "conv adapter parameters. It has no effect on Linear layers, so leave it off unless train_conv "
            "is on."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.use_scalar": {
        "summary": "Adds a trained scale that starts at 0 so the weight factors can init non-zero (default off).",
        "detail": (
            "Standard init zeroes the up factor so training starts as a no-op; use_scalar instead "
            "initializes both factors and multiplies the delta by a learned scalar starting at 0. The "
            "scalar is folded into the exported tensors, so files stay loader-compatible."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.dora_wd": {
        "summary": "DoRA weight decomposition on top of this algorithm: the adapter trains the direction, a per-channel magnitude trains separately (default off).",
        "detail": (
            "Splits each adapted weight into a trained magnitude (dora_scale) and the algorithm's delta "
            "as direction. Often tracks full finetuning better at low rank, at a per-step compute cost. "
            "For plain LoRA + DoRA pick the Lycoris.DoRA type instead of toggling this."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.wd_on_output": {
        "summary": "Computes the DoRA magnitude over the output axis (default on).",
        "detail": (
            "Only matters when DoRA decomposition is active (dora_wd, or the DoRA type). On = one "
            "magnitude per output channel (lycoris default); off = per input column. Leave on unless "
            "reproducing a recipe that used the input axis."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.train_norm": {
        "summary": "Also trains the existing LayerNorm/GroupNorm weights alongside the adapter (default off).",
        "detail": (
            "Exports them as w_norm/b_norm keys, which ComfyUI loads with the rest of the file. "
            "Try it when global tone, contrast, or palette refuses to shift at a rank that otherwise "
            "works. SDXL only — the Cosmos DiT has no trainable norm weights, so the run fails fast there."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.rs_lora": {
        "summary": "Scales the adapter delta by alpha/sqrt(rank) instead of alpha/rank (default off).",
        "detail": (
            "Rank-stabilized LoRA: standard scaling shrinks the effective update as rank grows; "
            "sqrt scaling keeps it steady. Worth trying at rank 32+. The exported per-module alpha "
            "is adjusted (alpha x sqrt(rank)) so loaders reproduce the trained strength unchanged."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.target_include": {
        "summary": "Only attaches the adapter to modules whose path matches one of these glob patterns (empty = all modules).",
        "detail": (
            "Patterns match the full dotted module path (e.g. unet.down_blocks.0...attn1.to_q on SDXL, "
            "blocks.0.self_attn... on Cosmos). *attn* trains attention only — smaller file with the "
            "capacity focused on composition/identity instead of textures."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.target_exclude": {
        "summary": "Skips modules whose path matches any of these glob patterns, applied after the include list (empty = skip none).",
        "detail": (
            "Same path matching as Target include. E.g. *ff* (SDXL) or *mlp* (Cosmos) leaves the "
            "feed-forward layers untrained. If include and exclude together match nothing, the run "
            "fails at startup instead of training an empty adapter."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.unbalanced_factorization": {
        "summary": "LoKr: swaps which output factor each Kronecker side gets (default off).",
        "detail": (
            "The output dimension factors into a (small, large) pair; this gives W1 the large one "
            "instead. Capacity moves between the two Kronecker factors — a recipe-matching knob, leave "
            "off otherwise."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.block_size": {
        "summary": "DyLoRA: the rank trains in nested blocks of this size; rank must divide evenly (default 4).",
        "detail": (
            "Each step updates a random sub-rank that is a multiple of block_size, so the exported LoRA "
            "stays usable when truncated to any multiple of block_size after training. Smaller blocks give "
            "finer rank choices but noisier updates."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.constraint": {
        "summary": "OFT: caps the rotation magnitude per layer; 0 = uncapped (default).",
        "detail": (
            "Positive values bound the norm of the rotation generator (scaled by layer width) — the "
            "constrained-OFT variant. Raise it from 0 if the adapter drifts the model too far from base "
            "behavior. Exported as the file's per-module alpha."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.rescaled": {
        "summary": "OFT: adds a trained per-channel scale on top of the rotation (default off).",
        "detail": (
            "Pure OFT only rotates weights, preserving their norms. Rescaled OFT adds a learned diagonal "
            "scale (saved as 'rescale'), letting magnitudes change too — slightly more expressive, slightly "
            "less base-preserving."
        ),
        "doc": "docs/user/training-sdxl-lora-lokr.md",
    },
    "adapter.factor": {
        "summary": "Shapes the LoKr Kronecker split per layer; -1 (default) picks the most balanced pair.",
        "detail": (
            "-1 splits each dimension into near-square factors (128 -> 8x16, 512 -> 16x32). "
            "A positive value forces that factor when it divides the dimension (factor 4: 128 -> 4x32) — "
            "smaller factors shrink the adapter at some capacity cost. Leave at -1 unless you are "
            "matching a known recipe or chasing a smaller file."
        ),
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
        "summary": "Export model weights every N training examples (converted to steps using global batch size).",
        "detail": "Alternative to save_every_n_steps when you want to compare runs with different batch sizes at the same data exposure.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "checkpoint_every_n_minutes": {
        "summary": "Write a DeepSpeed resume checkpoint after this many wall-clock minutes.",
        "detail": "Use on long runs to limit the distance back to the last safe resume point if training crashes.",
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
        "summary": "RNG seed for training, tag dropout, and dataloader shuffle (default 42).",
        "detail": "Change to reproduce a run with a different random order, or to break a run that appears to be stuck in a bad local pattern.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_datasets": {
        "summary": "Held-out dataset(s) used for periodic evaluation (val/loss and val/gap).",
        "detail": (
            "Each entry is a path to a dataset TOML, or a table with name and config keys. "
            "Set at least one to enable the generalization probe — without it, val_gap_enable is a no-op."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_every_n_steps": {
        "summary": "Run a validation pass every N optimizer steps.",
        "detail": "Use this for fine-grained overfitting tracking; combine with val_gap_enable to log val/loss and val/gap.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_every_n_epochs": {
        "summary": "Run a validation pass at the end of every N epochs.",
        "detail": "Simpler alternative to eval_every_n_steps when your dataset has a well-defined epoch boundary.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_every_n_examples": {
        "summary": "Run a validation pass every N training examples (converted to steps using global batch size).",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_before_first_step": {
        "summary": "Run one eval pass before the first training step.",
        "detail": "Useful for a loss baseline on validation data.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "eval_gradient_accumulation_steps": {
        "summary": "Micro-batches accumulated per eval step (default 1).",
        "detail": "Raise to match training accumulation if your eval loss differs unexpectedly from train loss.",
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
        "summary": "Forward batches per gap probe pass (per timestep quantile); default 8.",
        "detail": (
            "Each probe runs this many forward-only batches per quantile to estimate val/loss and train/probe. "
            "Lower if the probe adds noticeable time to your eval cadence; raise for smoother, less noisy gap curves."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "preview.enabled": {
        "summary": "Generate sample images at configured intervals during training.",
        "detail": "Images are written to the run preview/ folder and shown in TensorBoard. Enable to visually track quality without stopping the run.",
        "doc": "docs/user/previews.md",
    },
    "preview.prompts": {
        "summary": "Preview configurations (one prompt or table per row in TOML).",
        "detail": "Each entry becomes one item in preview.prompts — use the Add sampling button to manage the list.",
        "doc": "docs/user/previews.md",
    },
    "preview.negative_prompt": {
        "summary": "Negative prompt applied to every preview image (SDXL).",
        "detail": "Shared across all preview prompts. Leave empty for Cosmos/Anima, which does not use a negative prompt during sampling.",
        "doc": "docs/user/previews.md",
    },
    "preview.width": {
        "summary": "Preview image width in pixels.",
        "detail": "Set to a resolution your model supports; mismatch with the trained resolution can make previews look blurry or distorted.",
        "doc": "docs/user/previews.md",
    },
    "preview.height": {
        "summary": "Preview image height in pixels.",
        "detail": "Set to a resolution your model supports; mismatch with the trained resolution can make previews look blurry or distorted.",
        "doc": "docs/user/previews.md",
    },
    "preview.num_inference_steps": {
        "summary": "Denoising steps per preview image.",
        "detail": "Fewer steps = faster preview but lower quality. Typical values: 20–30 for SDXL, 30–50 for Cosmos.",
        "doc": "docs/user/previews.md",
    },
    "preview.guidance_scale": {
        "summary": "Classifier-free guidance scale for previews.",
        "detail": "Higher values follow the prompt more strictly but can saturate colours. Typical range: 5–9 for SDXL; Cosmos ignores this.",
        "doc": "docs/user/previews.md",
    },
    "preview.seed": {
        "summary": "Base RNG seed for preview images.",
        "detail": "Fixed seed produces consistent previews across steps so changes in quality are clearly visible. Change if the seed happens to pick an unrepresentative starting noise.",
        "doc": "docs/user/previews.md",
    },
    "preview.seed_stride": {
        "summary": "Seed offset applied per prompt index and per training step.",
        "detail": "Varies the noise across multiple prompts and steps so previews do not all look identical. Leave at default unless you have a specific reason to fix the per-step seed.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_every_n_steps": {
        "summary": "Generate preview images every N optimizer steps.",
        "detail": "Lower values give more frequent quality checks at the cost of extra inference time per step. Combine with preview_before_first_step for a baseline.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_every_n_epochs": {
        "summary": "Generate preview images at the end of every N epochs.",
        "detail": "Alternative to preview_every_n_steps for epoch-based runs. Preview inference runs after the epoch save, so it does not delay checkpointing.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_before_first_step": {
        "summary": "Run one preview pass before step 1 to capture the untrained baseline.",
        "detail": "Useful for visually comparing the model's output before and after training begins.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_offload_text_encoder": {
        "summary": "Move the text encoder to CPU during Cosmos preview sampling to free VRAM.",
        "detail": "Enable if preview sampling OOMs; the transfer adds latency but lets the DiT use more VRAM during generation.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_blocks_to_swap": {
        "summary": "Number of DiT blocks to keep on CPU during Cosmos preview sampling.",
        "detail": "Reduces preview VRAM at the cost of slower sampling. Raise if preview sampling OOMs after enabling preview_offload_text_encoder.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_offload_dit_for_decode": {
        "summary": "Move the DiT to CPU during the VAE decode to save VRAM (Cosmos only). "
        "Off by default — unsafe on DeepSpeed/compiled runs (crashes the next NCCL op). "
        "Rarely needed: the preview decode is tiled, so it fits next to a resident DiT.",
        "doc": "docs/user/previews.md",
    },
    "preview.preview_save_png": {
        "summary": "Write preview images as PNG files under the run preview/ folder (on by default).",
        "detail": "Turn off only to suppress preview files on disk; images still appear in TensorBoard via the in-memory path.",
        "doc": "docs/user/previews.md",
    },
    "tracking.enabled": {
        "summary": "Master switch for experiment tracking (run.json + TB + timeline).",
        "detail": "When off, the run trains with a no-op sink and writes no tracking artifacts.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "tracking.system_sampler.enabled": {
        "summary": "Periodically sample GPU/CPU/RAM usage and log as system/* scalars.",
        "detail": "Enable to see hardware utilisation alongside training loss in TensorBoard. Turn off if the sampler adds overhead on very fast steps.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "tracking.wandb.project": {
        "summary": "WandB project name — required when 'wandb' is in tracking.backends.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "tracking.wandb.run_name": {
        "summary": "Display name for this run in WandB.",
        "detail": "Defaults to the run directory name if omitted.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "tracking.wandb.api_key": {
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
        "summary": "Load the full backbone onto GPU for eval passes even when blocks_to_swap is on.",
        "detail": "Gives more accurate eval loss at the cost of temporarily using full-model VRAM. Enable if eval results look wrong with block swap active.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "disable_block_swap_for_preview": {
        "summary": "Load the full backbone onto GPU for preview sampling even when blocks_to_swap is on.",
        "detail": "Faster previews at the cost of temporarily using full-model VRAM. Enable if preview quality looks degraded with block swap active.",
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
        "summary": "Micro-batch override for image (single-frame) buckets when the run mixes video and images.",
        "detail": (
            "Integer or per-resolution map like micro_batch_size_per_gpu (e.g. 512 -> 2, 1024 -> 1); "
            "unset = image buckets use micro_batch_size_per_gpu. Set it when image steps leave the GPU "
            "under-filled at the batch size the video buckets need."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "steps_per_print": {
        "summary": "DeepSpeed prints step timing to the console every N steps.",
        "detail": "Lower for more frequent stdout feedback; raise to reduce log noise on long runs. Does not affect TensorBoard logging (use logging_steps for that).",
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
        "summary": "Plot TensorBoard/WandB x-axis as total examples seen instead of optimizer steps.",
        "detail": "Useful when comparing runs with different batch sizes or gradient accumulation — example count normalizes the x-axis across them.",
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
            "0 (default) loads in the main process, with dataloader_prefetch (on by default) "
            "overlapping the load with compute. Raise to 2–4 on Linux only if prefetch alone "
            "leaves the GPU idle waiting for batches (>0 is problematic on Windows/macOS)."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_prefetch": {
        "summary": "Background thread that preloads the next batch while the current step trains.",
        "detail": (
            "On by default: it overlaps loading the next batch (the next step's data, not the "
            "whole epoch) with the current step's compute. Only applies when "
            "dataloader_num_workers = 0. Off, the load runs synchronously on the main thread and "
            "stalls the GPU every step (measured ~67 ms/step). Same data either way."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_pin_memory": {
        "summary": "Allocate CPU batch tensors in page-locked memory for faster host-to-GPU copies.",
        "detail": "Helps on Linux with dataloader_num_workers > 0; has little effect with the default prefetch thread. Off by default.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_prefetch_factor": {
        "summary": "Batches each DataLoader worker prefetches ahead (default 2, applies only when num_workers > 0).",
        "detail": "Raise if workers are fast but the GPU still stalls waiting for batches. Lower if RAM is tight.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "dataloader_persistent_workers": {
        "summary": "Keep DataLoader worker processes alive between epochs instead of restarting them (default on).",
        "detail": "Turn off only if you see stale worker state or memory leaks across epoch boundaries.",
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
        "summary": "Passes over the full training dataset; each pass visits every image once.",
        "detail": (
            "One epoch = one complete pass over all images at each configured resolution. "
            "Raise if the model hasn't converged; lower if val/gap is rising (overfitting signal)."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "micro_batch_size_per_gpu": {
        "summary": "Samples per GPU per forward/backward micro-step; integer or per-resolution map.",
        "detail": (
            "Per-resolution mode (e.g. 512 -> 2, 1024 -> 1) lets low resolutions batch up where the "
            "GPU is under-filled while the detail resolution stays at what fits: measured on a 16 GB "
            "4080, batching pays at 512 (bs1->bs2 is -18% per sample) and nothing at 1024 (GEMM-"
            "saturated). Buckets pick the numerically closest configured resolution."
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
        "summary": "Clips gradient norm to this value before the optimizer step (0 = off).",
        "detail": "Prevents gradient spikes from destabilizing training. Lower if loss spikes suddenly mid-run; common values are 0.5–1.0.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "logging_steps": {
        "summary": "Write train/loss and other scalars to TensorBoard every N optimizer steps.",
        "detail": "Lower for finer loss curves; raise to reduce log file growth on very long runs.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "pipeline_stages": {
        "summary": "DeepSpeed pipeline parallel stages — set to the number of GPUs when using pipeline parallelism.",
        "detail": "Default 1 (single GPU or tensor-parallel). Raise only when splitting the model across multiple GPUs; blocks_to_swap is the VRAM lever for single-GPU runs.",
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "activation_checkpointing": {
        "summary": "Save VRAM by recomputing activations in the backward pass instead of storing them.",
        "detail": (
            "During the forward pass the intermediate activations are dropped and re-computed during "
            "backprop — large activation-memory savings for a small extra compute cost (an extra "
            "forward per checkpointed block). Keep it true on small GPUs; false can OOM. Values: "
            "true (full, lowest VRAM, safe default), false (fastest, highest VRAM — OOMs at high res), "
            "or 'auto' (compile's memory-budget partitioner picks the optimal save/recompute split per "
            "graph — requires compile=true; tune with activation_memory_budget; exact recompute, no "
            "precision cost). The old 'selective' (SAC) and 'unsloth' modes were retired: 'auto' beats "
            "SAC on both speed and VRAM; legacy configs fall back to true with a warning."
        ),
        "doc": "docs/user/training-loop-and-eval.md",
    },
    "activation_memory_budget": {
        "summary": "VRAM/speed dial for activation_checkpointing='auto' (0.0-1.0, global).",
        "detail": (
            "Fraction of activation memory the compile partitioner may keep instead of recomputing: "
            "0.0 ~ full-checkpoint VRAM, 1.0 ~ no-checkpoint speed; gains plateau around 0.5. "
            "One global value for every resolution/shape in both compile modes. Recompute is exact "
            "(same math, no precision cost). Measured on Cosmos LoKr @1024 (RTX 4080, vs full "
            "checkpointing at 0.97 s / 5.8 GB): 0.1 = -9.5% step time / 6.4 GB, 0.3 = -16% / 9.0 GB "
            "(default), 0.5 = -21% / 11.3 GB. Requires compile = true. If a step OOMs, the budget "
            "backs off and recompiles instead of crashing (activation_budget_backoff, default on); "
            "the settled value is logged."
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
        "summary": "Where the on-disk compile cache lives (default: a 'compile' subdir of cache_root).",
        "detail": (
            "Directory for compile_disk_cache. Defaults to <cache_root>/compile — i.e. it sits with the "
            "dataset caches and follows a custom cache_root. Must be ext4 (255-char filenames); point it at "
            "an ext4 path if cache_root is on an encrypted/short-filename filesystem. Only used when "
            "compile_disk_cache is active."
        ),
        "doc": "docs/user/training-cosmos-predict2-lora-lokr-finetune.md",
    },
    "checkpoint_every_n_epochs": {
        "summary": "Write a DeepSpeed resume checkpoint at the end of every N epochs.",
        "detail": "Stores full optimizer and scheduler state for training resumption; not a usable inference file. Combine with max_checkpoints_to_keep to limit disk use.",
        "doc": "docs/user/checkpoint-and-save.md",
    },
    "save_every_n_epochs": {
        "summary": "Export adapter or full-model weights every N epochs (default 1).",
        "detail": "Writes an epoch1/, epoch2/, ... folder usable in ComfyUI/Forge. Reduce to save fewer intermediate files; increase max_model_exports_to_keep if you want them all kept.",
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
