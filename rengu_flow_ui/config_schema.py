"""Config form schema and registry metadata for the web UI."""

from __future__ import annotations

from typing import Any

from rengu_flow.config.defaults import DTYPE_MAP
from rengu_flow.optim.resolver import scheduler_registry
from rengu_flow.registry.model_capabilities import (
    ADAPTER_FIELD_TEMPLATES,
    capabilities_for_api,
    get_canonical_model_types,
    get_capability,
    model_capability_registry,
    normalize_model_type,
)
from rengu_flow_ui.field_visibility import attach_visibility_to_schema
from rengu_flow_ui.model_form import attach_model_section_visibility
from rengu_flow_ui.optimizer_form import attach_optimizer_visibility
from rengu_flow_ui.optim_kv_defaults import (
    OPTIMIZER_REGISTRY_KV_DEFAULTS,
    SCHEDULER_BUILTIN_KV_DEFAULTS,
    SCHEDULER_FQN_KV_DEFAULTS,
    SCHEDULER_RUNTIME_TOKENS,
    SUGGESTED_SCHEDULER_FQNS,
)
from rengu_flow_ui.scheduler_form import attach_scheduler_visibility
from rengu_flow.registry.optimizers import optimizer_options

DTYPE_OPTIONS = list(DTYPE_MAP.keys())
ACTIVATION_CHECKPOINTING_OPTIONS = [False, True, "auto"]
PARTITION_METHODS = ["parameters", "uniform", "manual"]
HAS_ADAPTER = {"field": "_has_adapter", "equals": True}

# Fields tagged "Important" in the UI — only knobs most runs should consciously set.
RECOMMENDED_PATHS: frozenset[str] = frozenset(
    {
        "lr_scheduler",
        "epochs",
        "micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
    }
)


def _field(
    path: str,
    label: str,
    ftype: str,
    *,
    default: Any = None,
    description: str = "",
    required: bool = False,
    recommended: bool = False,
    importance: str | None = None,
    options: list[Any] | None = None,
    when: dict[str, Any] | None = None,
    min_value: float | None = None,
    placeholder: str = "",
    example: Any = None,
    options_from_model: bool = False,
    options_key: str | None = None,
    when_model_has_adapter: bool = False,
    allow_custom: bool = False,
    when_capability: str | dict[str, Any] | None = None,
    show_if_set: bool = False,
    show_if_set_exclude_zero: bool = False,
    visibility: dict[str, Any] | None = None,
    max_length: int | None = None,
    runtime_tokens: list[str] | None = None,
    deepspeed_only: bool = False,
) -> dict[str, Any]:
    if required:
        imp = "required"
    elif importance in ("required", "recommended", "advanced"):
        imp = importance
    elif recommended or path in RECOMMENDED_PATHS:
        imp = "recommended"
    else:
        imp = "advanced"
    out: dict[str, Any] = {
        "path": path,
        "label": label,
        "type": ftype,
        "default": default,
        "description": description,
        "required": required,
        "recommended": imp == "recommended",
        "importance": imp,
        "options": options,
        "when": when,
        "min": min_value,
        "placeholder": placeholder,
        "example": example,
    }
    if options_from_model:
        out["options_from_model"] = True
    if options_key:
        out["options_key"] = options_key
    if when_model_has_adapter:
        out["when_model_has_adapter"] = True
    if allow_custom:
        out["allow_custom"] = True
    if when_capability:
        out["when_capability"] = when_capability
    if show_if_set:
        out["show_if_set"] = True
    if show_if_set_exclude_zero:
        out["show_if_set_exclude_zero"] = True
    if visibility is not None:
        out["visibility"] = visibility
    if max_length is not None:
        out["max_length"] = max_length
    if runtime_tokens:
        out["runtime_tokens"] = runtime_tokens
    if deepspeed_only:
        # Multi-GPU / DeepSpeed-pipeline-only knob. Dropped from the schema on hosts whose engine
        # is not 'deepspeed' (e.g. native Windows 'accelerate'); see get_schema().
        out["deepspeed_only"] = True
    return out


def _when_model(*type_ids: str) -> dict[str, Any]:
    return {"field": "model.type", "in": list(type_ids)}


def _field_from_template(spec: dict[str, Any], when: dict[str, Any] | None) -> dict[str, Any] | None:
    if spec.get("ui") is False:
        return None
    opts = spec.get("options")
    if spec.get("options_key") == "dtypes":
        opts = DTYPE_OPTIONS
    return _field(
        spec["path"],
        spec["label"],
        spec["type"],
        default=spec.get("default"),
        description=spec.get("description", ""),
        required=spec.get("required", False),
        recommended=spec.get("recommended", False),
        importance=spec.get("importance"),
        options=opts,
        when=when or spec.get("when"),
        min_value=spec.get("min"),
        placeholder=spec.get("placeholder", ""),
        example=spec.get("example"),
        when_model_has_adapter=spec.get("when_model_has_adapter", False),
        when_capability=spec.get("when_capability"),
        show_if_set=spec.get("show_if_set", False),
        show_if_set_exclude_zero=spec.get("show_if_set_exclude_zero", False),
        visibility=spec.get("visibility"),
    )


def _model_section_fields() -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = [
        _field(
            "model.type",
            "Model type",
            "select",
            required=True,
            description="Registered pipeline type (e.g. sdxl, cosmos_predict2).",
        ),
        _field(
            "model.dtype",
            "Model dtype",
            "select",
            required=True,
            options=DTYPE_OPTIONS,
            description="Default precision for VAE, text stack, adapters, and most weights.",
        ),
    ]
    for cap in model_capability_registry.values():
        when = _when_model(cap.type_id)
        for spec in cap.model_fields:
            built = _field_from_template(spec, when)
            if built is not None:
                fields.append(built)
    return fields


def _adapter_section_fields() -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = [
        _field(
            "_has_adapter",
            "Train with adapter (LoRA / LoKr / LyCORIS)",
            "boolean",
            default=True,
            importance="required",
            description="Uncheck for full-model finetune when the model supports it.",
        ),
        _field(
            "adapter.type",
            "Adapter / network type",
            "select",
            when=HAS_ADAPTER,
            required=True,
            options_from_model=True,
            description="Options depend on the selected model.",
        ),
    ]
    for spec in ADAPTER_FIELD_TEMPLATES["common"]:
        fields.append(
            _field_from_template({**spec, "when_model_has_adapter": True}, HAS_ADAPTER)
        )
    from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES

    # Build reverse map: field path -> list of kinds that use it (preserving first-seen spec)
    path_to_kinds: dict[str, list[str]] = {}
    path_to_spec: dict[str, dict] = {}
    for kind in ("lora", "lokr", *LYCORIS_ADAPTER_TYPES):
        for spec in ADAPTER_FIELD_TEMPLATES.get(kind, []):
            p = spec["path"]
            if p not in path_to_kinds:
                path_to_kinds[p] = []
                path_to_spec[p] = spec
            path_to_kinds[p].append(kind)

    # Emit each field once, with a when clause covering all its kinds
    emitted: set[str] = set()
    for kind in ("lora", "lokr", *LYCORIS_ADAPTER_TYPES):
        for spec in ADAPTER_FIELD_TEMPLATES.get(kind, []):
            p = spec["path"]
            if p in emitted:
                continue
            emitted.add(p)
            kinds_for_path = path_to_kinds[p]
            if len(kinds_for_path) == 1:
                when_kind = {"all": [HAS_ADAPTER, {"field": "adapter.type", "equals": kinds_for_path[0]}]}
            else:
                when_kind = {"all": [HAS_ADAPTER, {"field": "adapter.type", "in": kinds_for_path}]}
            f = _field_from_template(path_to_spec[p], None)
            f["when"] = when_kind
            fields.append(f)
    return fields


def _preview_section() -> dict[str, Any]:
    from rengu_flow_ui.preview_form import WHEN_COSMOS_PREVIEW

    preview_models = [c.type_id for c in model_capability_registry.values() if c.preview]
    when_preview = _when_model(*preview_models) if preview_models else {"field": "model.type", "in": []}
    return {
        "id": "preview",
        "title": "Sampling",
        "description": (
            "Sample images during training. Set the global defaults, then add one or more "
            "sampling rows below; each row is a separate prompt (and optional overrides). "
            "Global settings apply to all sampling rows unless overridden on a row."
        ),
        "when_any_model": preview_models,
        "fields": [
            _field(
                "preview.prompts",
                "Preview configurations",
                "preview_entries",
                when=when_preview,
                importance="recommended",
                description="Each entry becomes one item under preview.prompts in TOML.",
            ),
            _field(
                "preview.enabled",
                "Enabled",
                "boolean",
                default=False,
                when=when_preview,
                importance="recommended",
                description=(
                    "Off by default — generating samples during training adds significant "
                    "VRAM and time (a 1024×1024 SDXL preview can OOM on small GPUs). Enable "
                    "it explicitly when you want in-training samples."
                ),
            ),
            _field(
                "preview.preview_every_n_steps",
                "Preview every N steps",
                "integer",
                min_value=1,
                when=when_preview,
                importance="recommended",
                example=500,
            ),
            _field(
                "preview.preview_every_n_epochs",
                "Preview every N epochs",
                "integer",
                min_value=1,
                when=when_preview,
                importance="recommended",
                example=1,
            ),
            _field(
                "preview.preview_before_first_step",
                "Preview before first step",
                "boolean",
                when=when_preview,
                importance="recommended",
            ),
            _field(
                "disable_block_swap_for_preview",
                "Disable block swap for preview",
                "boolean",
                when=when_preview,
                when_capability="block_swap",
            ),
            _field("preview.negative_prompt", "Negative prompt", "string", when=when_preview, example="blurry, low quality, watermark"),
            _field("preview.width", "Width", "integer", default=1024, when=when_preview),
            _field("preview.height", "Height", "integer", default=1024, when=when_preview),
            _field("preview.num_inference_steps", "Inference steps", "integer", default=20, when=when_preview),
            _field("preview.guidance_scale", "Guidance scale", "number", default=7.0, when=when_preview),
            _field("preview.seed", "Seed", "integer", default=0, when=when_preview),
            _field("preview.seed_stride", "Seed stride", "integer", default=1, when=when_preview),
            _field(
                "preview.preview_offload_text_encoder",
                "Offload text encoder during preview",
                "boolean",
                default=True,
                when=WHEN_COSMOS_PREVIEW,
            ),
            _field(
                "preview.preview_blocks_to_swap",
                "Preview blocks to swap",
                "integer",
                default=0,
                min_value=0,
                when=WHEN_COSMOS_PREVIEW,
            ),
            _field(
                "preview.preview_offload_dit_for_decode",
                "Offload DiT to CPU for VAE decode",
                "boolean",
                default=False,
                when=WHEN_COSMOS_PREVIEW,
            ),
            _field(
                "preview.preview_save_png",
                "Save preview PNGs",
                "boolean",
                default=False,
                when=WHEN_COSMOS_PREVIEW,
            ),
        ],
    }


def get_registries() -> dict[str, Any]:
    from rengu_flow_ui import datasets_store

    dataset_picker = datasets_store.list_for_training_picker()
    optimizer_pairs = optimizer_options()
    optimizers = [value for value, _label in optimizer_pairs]
    optimizer_labels = [label for _value, label in optimizer_pairs]
    from rengu_flow_ui.preview_form import get_preview_entry_fields

    return {
        "models": get_canonical_model_types(),
        "model_capabilities": capabilities_for_api(),
        "preview_entry_fields": get_preview_entry_fields(),
        "optimizers": optimizers,
        "optimizer_labels": optimizer_labels,
        # Canonical per-type [optimizer] KV defaults so the form prefills from the
        # registry instead of a hand-maintained frontend copy (which drifts).
        "optimizer_kv_defaults": {k: dict(v) for k, v in OPTIMIZER_REGISTRY_KV_DEFAULTS.items()},
        "optimizer_allow_custom": True,
        "schedulers": sorted(set(scheduler_registry.keys()) | set(SUGGESTED_SCHEDULER_FQNS)),
        # Canonical scheduler KV defaults so the form prefills from the backend
        # instead of a hand-maintained frontend copy (builtin names + suggested FQNs).
        "scheduler_kv_defaults": {k: dict(v) for k, v in SCHEDULER_BUILTIN_KV_DEFAULTS.items()},
        "scheduler_fqn_kv_defaults": {k: dict(v) for k, v in SCHEDULER_FQN_KV_DEFAULTS.items()},
        "scheduler_allow_custom": True,
        "dtypes": DTYPE_OPTIONS,
        "activation_checkpointing": ACTIVATION_CHECKPOINTING_OPTIONS,
        "partition_methods": PARTITION_METHODS,
        "dataset_paths": dataset_picker,
    }


def get_sections() -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = [
        {
            "id": "general",
            "title": "General",
            "description": "Dataset path, output, and run naming.",
            "fields": [
                _field(
                    "dataset",
                    "Dataset",
                    "select",
                    required=True,
                    allow_custom=True,
                    recommended=True,
                    description=(
                        "One or more dataset TOMLs (library ref or file path). "
                        "Multiple entries are merged at train time (all [[directory]] blocks)."
                    ),
                ),
                _field(
                    "output_dir",
                    "Output directory",
                    "string",
                    default="output",
                    importance="advanced",
                    description="Root folder for run outputs.",
                ),
                _field(
                    "run_name",
                    "Run name",
                    "string",
                    importance="advanced",
                    description="Optional suffix for timestamped run folder.",
                    example="sdxl-lora-v1",
                ),
                _field(
                    "resume_from_checkpoint",
                    "Resume from checkpoint",
                    "boolean",
                    default=False,
                    importance="advanced",
                ),
            ],
        },
        {
            "id": "model",
            "title": "Model",
            "description": "Model type and checkpoint paths (fields depend on type).",
            "fields": _model_section_fields(),
        },
        {
            "id": "adapter",
            "title": "Adapter / network",
            "description": "LoRA or LoKr when supported; omit for full-model finetune.",
            "fields": _adapter_section_fields(),
        },
        {
            "id": "optimizer",
            "title": "Optimizer",
            "fields": [
                _field(
                    "optimizer.type",
                    "Optimizer type",
                    "select",
                    required=True,
                    allow_custom=True,
                    description="Built-in registry names, pytorch_optimizer classes, or module.Class path.",
                ),
                _field(
                    "optimizer.extra_params",
                    "Optimizer parameters",
                    "key_value_list",
                    importance="advanced",
                    description=(
                        "All [optimizer] keys (lr, betas, weight_decay, gradient_release, …). "
                        "Pre-filled when you change optimizer type; see docs for per-type tables."
                    ),
                ),
            ],
        },
        {
            "id": "scheduler",
            "title": "LR scheduler",
            "fields": [
                _field(
                    "lr_scheduler",
                    "Scheduler type",
                    "select",
                    options=None,
                    recommended=True,
                    allow_custom=True,
                    description="Registry name (cosine, rex, linear, …) or fully-qualified scheduler class.",
                ),
                _field(
                    "warmup_steps",
                    "Warmup steps",
                    "integer",
                    default=0,
                    min_value=0,
                    importance="advanced",
                    description=(
                        "Trainer-level linear LR warmup before the main scheduler (not a "
                        "[lr_scheduler_args] constructor kwarg)."
                    ),
                ),
                _field(
                    "lr_scheduler_args.extra_params",
                    "Scheduler parameters",
                    "key_value_list",
                    importance="advanced",
                    runtime_tokens=list(SCHEDULER_RUNTIME_TOKENS),
                    description=(
                        "Scheduler constructor kwargs (lr_min, total_iters, T_max, …) under "
                        "[lr_scheduler_args]."
                    ),
                ),
            ],
        },
        {
            "id": "te_cache",
            "title": "Text-embedding cache",
            "description": (
                "Cache text-encoder outputs once instead of running the encoder every "
                "step. Baking tag-dropout caption variants into the cache "
                "(cached_caption_variants / cached_caption_shuffle) is configured per "
                "dataset in the Dataset form."
            ),
            "fields": [
                _field(
                    "model.cache_text_embeddings",
                    "Cache text embeddings",
                    "boolean",
                    default=True,
                    recommended=True,
                    when=_when_model("cosmos_predict2", "sdxl"),
                    description=(
                        "Encode captions once and cache them (faster steps, more disk; "
                        "frees the text encoder from the GPU). For rotating dropout "
                        "regularization with caching on, set the dataset's "
                        "cached_caption_variants >= 2 (1 bakes a single fixed variant)."
                    ),
                ),
                _field("caching_batch_size", "Dataset cache batch size", "integer", default=1),
                _field("cache_num_proc", "Cache CPU workers", "integer", default=8, min_value=1),
                _field("cache_keep_in_memory", "Keep HF slice in RAM during cache", "boolean", default=False),
                _field(
                    "cache_dedup_text_embeddings",
                    "Dedup text embeddings on cache",
                    "boolean",
                    default=False,
                    importance="advanced",
                    description="Reuse TE outputs when captions match (tag-heavy datasets).",
                ),
            ],
        },
        {
            "id": "training",
            "title": "Training loop",
            "fields": [
                _field("epochs", "Epochs", "integer", default=1, min_value=1, recommended=True),
                _field(
                    "max_steps",
                    "Max steps",
                    "integer",
                    min_value=1,
                    importance="advanced",
                    description="Optional cap; stops after N optimizer steps.",
                    example=1000,
                ),
                _field(
                    "micro_batch_size_per_gpu",
                    "Micro batch per GPU",
                    "integer",
                    default=1,
                    min_value=1,
                    recommended=True,
                    description=(
                        "Uniform integer, or per-resolution (e.g. 512 -> 2, 1024 -> 1) to batch "
                        "up low resolutions where the GPU is under-filled."
                    ),
                ),
                _field(
                    "image_micro_batch_size_per_gpu",
                    "Image micro batch per GPU",
                    "integer",
                    min_value=1,
                    importance="advanced",
                    description=(
                        "Per-GPU micro-batch for image-only steps when mixing image+video; "
                        "falls back to micro_batch_size_per_gpu. For an advanced per-modality "
                        "dict, edit it in the TOML tab."
                    ),
                    example=2,
                ),
                _field(
                    "gradient_accumulation_steps",
                    "Gradient accumulation",
                    "integer",
                    default=1,
                    min_value=1,
                    recommended=True,
                ),
                _field(
                    "gradient_clipping",
                    "Gradient clipping",
                    "number",
                    default=1.0,
                ),
                _field(
                    "logging_steps",
                    "Logging steps",
                    "integer",
                    default=1,
                    min_value=1,
                ),
                _field(
                    "steps_per_print",
                    "DeepSpeed steps_per_print",
                    "integer",
                    default=1,
                    importance="advanced",
                ),
                _field(
                    "train_seed",
                    "Training seed",
                    "integer",
                    default=42,
                    min_value=0,
                    importance="advanced",
                ),
                _field(
                    "synthetic_num_batches",
                    "Synthetic batches (debug)",
                    "integer",
                    importance="advanced",
                    example=50,
                ),
                _field("compile", "torch.compile", "boolean", default=False, importance="advanced"),
                _field(
                    "compile_mode",
                    "torch.compile mode",
                    "select",
                    options=[
                        "default",
                        "reduce-overhead",
                        "max-autotune",
                        "max-autotune-no-cudagraphs",
                    ],
                    allow_custom=True,
                    importance="advanced",
                    when={"field": "compile", "equals": True},
                    description=(
                        "Inductor mode passed to torch.compile. 'reduce-overhead' uses CUDA graphs "
                        "to cut per-step launch overhead (best for fixed-shape steps). Leave unset "
                        "for 'default'."
                    ),
                ),
                _field(
                    "compile_dynamic",
                    "torch.compile dynamic shapes",
                    "boolean",
                    default=False,
                    importance="advanced",
                    when={"field": "compile", "equals": True},
                    description="Pass dynamic=True to torch.compile when input shapes vary between steps.",
                ),
                _field(
                    "compile_disk_cache",
                    "Persist compile cache to disk",
                    "select",
                    options=["auto", True, False],
                    allow_custom=True,
                    importance="advanced",
                    when={"field": "compile", "equals": True},
                    description=(
                        "'auto' (default) persists Inductor/Triton kernels only when compile_dynamic is off "
                        "(static shapes — where the cache actually hits; dynamic shapes are a no-op). Needs an "
                        "ext4 cache dir; auto-disables with a warning on encrypted homes."
                    ),
                ),
                _field(
                    "compile_cache_dir",
                    "Compile cache dir",
                    "string",
                    importance="advanced",
                    when={"field": "compile", "equals": True},
                    placeholder="<cache_root>/compile",
                    description="Where the on-disk compile cache lives (must be ext4/255-char). Default: a 'compile' subdir of the dataset cache_root.",
                ),
                _field("x_axis_examples", "TensorBoard x-axis = examples", "boolean"),
                _field(
                    "ema_decay",
                    "EMA decay",
                    "number",
                    importance="advanced",
                    description="Optional; CPU shadow weights updated each step. No auto-export yet.",
                    example=0.999,
                ),
                _field("dataloader_num_workers", "Train DataLoader workers", "integer", default=0, min_value=0),
                _field("dataloader_prefetch", "Prefetch next batch (thread)", "boolean", default=True),
                _field("dataloader_pin_memory", "Pin memory (CUDA)", "boolean", default=False),
                _field("dataloader_prefetch_factor", "DataLoader prefetch factor", "integer", default=2, min_value=1),
                _field(
                    "dataloader_persistent_workers",
                    "Persistent DataLoader workers",
                    "boolean",
                    default=True,
                ),
            ],
        },
        {
            "id": "memory",
            "title": "Memory savings",
            "fields": [
                _field(
                    "activation_checkpointing",
                    "Activation checkpointing",
                    "select",
                    options=ACTIVATION_CHECKPOINTING_OPTIONS,
                    description=(
                        "true = full (lowest VRAM, safe default); false = fastest but OOMs at high res; "
                        "'auto' = compile's memory-budget partitioner (needs compile=true) — continuous "
                        "VRAM/speed dial via activation_memory_budget, faster AND lighter than the "
                        "retired 'selective' mode."
                    ),
                ),
                _field(
                    "activation_memory_budget",
                    "Auto-AC memory budget",
                    "number",
                    default=0.3,
                    min_value=0.0,
                    when={"field": "activation_checkpointing", "equals": "auto"},
                    description=(
                        "Only for activation_checkpointing='auto'. 0.0 ~ full-checkpoint VRAM, 1.0 ~ "
                        "no-checkpoint speed (plateaus ~0.5). Measured @1024 LoKr: 0.1 = -9.5% step time / "
                        "6.4 GB (beats SAC on both), 0.3 = -16% / 9.0 GB, 0.5 = -21% / 11.3 GB."
                    ),
                ),
                _field(
                    "activation_checkpoint_interval",
                    "Checkpoint interval (blocks)",
                    "integer",
                    default=1,
                    min_value=1,
                    importance="advanced",
                    description="Checkpoint every N transformer blocks (1 = every block). Measured neutral on Cosmos.",
                ),
                _field(
                    "reentrant_activation_checkpointing",
                    "Reentrant checkpointing",
                    "boolean",
                    importance="advanced",
                ),
                _field(
                    "activation_offload",
                    "Activation offload",
                    "boolean",
                    default=False,
                    importance="advanced",
                    description=(
                        "Stream saved activations to pinned CPU RAM (overlapped on side streams) "
                        "instead of recomputing them — trades activation VRAM for PCIe bandwidth, not "
                        "recompute FLOPs. Pairs with a raised activation_memory_budget. Costs host RAM; "
                        "incompatible with compile_mode='reduce-overhead' (CUDA graphs)."
                    ),
                ),
                # activation_offload tuning — revealed only when activation_offload is on.
                _field(
                    "activation_offload_min_tensor_mb",
                    "Offload min tensor size (MB)",
                    "number",
                    default=4.0,
                    min_value=0.0,
                    importance="advanced",
                    when={"field": "activation_offload", "equals": True},
                    description=(
                        "Only saved activations at least this large are offloaded; smaller tensors stay "
                        "in VRAM (the per-tensor copy overhead isn't worth it). Lower it to offload more "
                        "(useful when many medium activations add up); raise it to offload only the big ones."
                    ),
                ),
                _field(
                    "activation_offload_max_ram_gb",
                    "Offload max host RAM (GB)",
                    "number",
                    min_value=0.0,
                    importance="advanced",
                    placeholder="none (no cap)",
                    when={"field": "activation_offload", "equals": True},
                    description=(
                        "Cap on pinned host RAM used for offloaded activations. Empty = no cap. Once the "
                        "cap is hit, further activations stay in VRAM (so VRAM savings taper instead of "
                        "exhausting host RAM)."
                    ),
                ),
                _field(
                    "activation_offload_prefetch_mb",
                    "Offload backward prefetch (MB)",
                    "number",
                    default=512.0,
                    min_value=0.0,
                    importance="advanced",
                    when={"field": "activation_offload", "equals": True},
                    description=(
                        "How many MB of offloaded activations to stream back to the GPU ahead of where "
                        "the backward is running, to hide the H2D latency. Larger = more overlap but more "
                        "transient VRAM for the in-flight tensors."
                    ),
                ),
                _field(
                    "blocks_to_swap",
                    "Blocks to swap",
                    "integer",
                    default=0,
                    min_value=0,
                    when_capability="block_swap",
                    # NOT deepspeed_only: the hook offloader is engine-agnostic, so adapter (LoRA/LoKr)
                    # block swap works on the single-GPU 'accelerate' engine (native Windows) too.
                    # Full-model swap still needs gradient_release (DeepSpeed) — enforced at train time.
                ),
                _field(
                    "block_swap_prefetch",
                    "Block-swap prefetch (overlap)",
                    "boolean",
                    default=False,
                    importance="advanced",
                    when_capability="block_swap",
                    # Meaningful whenever blocks are actually being swapped — overlaps the next block's
                    # H2D copy with compute via a side stream + pinned buffers. Works for both adapter
                    # (streams frozen weights, adapters stay resident) and full-model (gradient_release)
                    # runs, so it is NOT deepspeed-only. Needs >=2 blocks resident (handled at runtime).
                    when={"form_nonempty": "blocks_to_swap", "exclude_zero": True},
                ),
            ],
        },
        {
            "id": "deepspeed",
            "title": "DeepSpeed (multi-GPU)",
            "fields": [
                _field(
                    "pipeline_stages",
                    "Pipeline stages",
                    "integer",
                    default=1,
                    min_value=1,
                    deepspeed_only=True,
                ),
                _field(
                    "partition_method", "Partition method", "select",
                    options=PARTITION_METHODS, deepspeed_only=True,
                ),
                _field("partition_split", "Partition split (manual)", "json", deepspeed_only=True),
            ],
        },
        {
            "id": "checkpoint",
            "title": "Checkpoints & export",
            "fields": [
                _field("checkpoint_every_n_epochs", "Checkpoint every N epochs", "integer", min_value=1, example=1),
                _field("checkpoint_every_n_minutes", "Checkpoint every N minutes", "number", min_value=0, example=30),
                _field("max_checkpoints_to_keep", "Max checkpoints to keep", "integer", min_value=1, example=3),
                _field("max_model_exports_to_keep", "Max model exports to keep", "integer", min_value=1, example=5),
                _field("keep_exports_from_step", "Keep exports from step", "integer", min_value=0, example=500),
                _field(
                    "save_every_n_epochs",
                    "Save model every N epochs",
                    "integer",
                    default=1,
                    min_value=1,
                ),
                _field("save_every_n_steps", "Save model every N steps", "integer", min_value=1, example=500),
                _field("save_every_n_examples", "Save every N examples", "integer", min_value=1, example=1000),
                _field("save_dtype", "Save dtype", "select", options=DTYPE_OPTIONS),
            ],
        },
        {
            "id": "eval",
            "title": "Evaluation",
            "fields": [
                _field(
                    "eval_datasets",
                    "Eval datasets",
                    "json",
                    description='List of paths or [{name, config}] tables.',
                ),
                _field("eval_every_n_steps", "Eval every N steps", "integer", min_value=1, example=100),
                _field("eval_every_n_epochs", "Eval every N epochs", "integer", min_value=1, example=1),
                _field("eval_every_n_examples", "Eval every N examples", "integer", min_value=1, example=500),
                _field("eval_before_first_step", "Eval before first step", "boolean", default=True),
                _field("eval_gradient_accumulation_steps", "Eval grad accum", "integer", default=1),
                _field(
                    "val_gap_enable",
                    "Train-val gap probe",
                    "boolean",
                    default=True,
                    description="Deterministic held-out val loss + train-val gap (overfitting signal). Uses the first eval dataset; no-ops if none.",
                ),
                _field(
                    "val_gap_probe_batches",
                    "Gap probe batches",
                    "integer",
                    default=8,
                    min_value=1,
                    description="Forward batches per probe (per timestep quantile). Smaller = faster.",
                ),
                _field(
                    "disable_block_swap_for_eval",
                    "Disable block swap for eval",
                    "boolean",
                    when_capability="block_swap",
                ),
            ],
        },
        _preview_section(),
        {
            "id": "tracking",
            "title": "Tracking",
            "fields": [
                _field("tracking.enabled", "Enable tracking", "boolean", default=True),
                _field(
                    "tracking.system_sampler.enabled",
                    "Sample system metrics (GPU/CPU)",
                    "boolean",
                    default=True,
                ),
                _field("tracking.wandb.project", "WandB project", "string", default="rengu-flow"),
                _field("tracking.wandb.run_name", "WandB run name", "string", example="sdxl-lora-v1"),
                _field("tracking.wandb.api_key", "WandB API key", "string"),
            ],
        },
    ]
    return sections


def get_schema() -> dict[str, Any]:
    from rengu_flow_ui.config_field_help import enrich_schema
    from rengu_flow.engine import resolve_backend
    from rengu_flow.platform_compat import PLATFORM

    engine = resolve_backend()
    deepspeed_engine = engine == "deepspeed"

    registries = get_registries()
    sections = get_sections()
    if not deepspeed_engine:
        # Non-DeepSpeed host (e.g. native Windows 'accelerate'): drop the multi-GPU /
        # pipeline-only knobs. They raise at train time there and do nothing single-GPU.
        for section in sections:
            section["fields"] = [f for f in section["fields"] if not f.get("deepspeed_only")]
    for section in sections:
        if section["id"] == "model":
            attach_model_section_visibility(section["fields"])
        if section["id"] == "optimizer":
            attach_optimizer_visibility(section["fields"])
        if section["id"] == "scheduler":
            attach_scheduler_visibility(section["fields"])
        for field in section["fields"]:
            if field["path"] == "model.type":
                field["options"] = registries["models"]
            if field["path"] == "optimizer.type":
                # Raw alias is the value (goes to TOML); vendor-prefixed label is display.
                field["option_values"] = registries["optimizers"]
                field["options"] = registries["optimizer_labels"]
            if field["path"] == "lr_scheduler":
                field["options"] = registries["schedulers"]
            if field["path"] == "dataset":
                field["options"] = [
                    p.get("label") or p["path"]
                    for p in registries.get("dataset_paths", [])
                ]
                field["option_values"] = [p["path"] for p in registries.get("dataset_paths", [])]
            _finalize_field_importance(field)
    from rengu_flow_ui.default_config_template import default_new_config_toml

    schema = enrich_schema(
        {
            "registries": registries,
            "sections": sections,
            "default_new_config_toml": default_new_config_toml(),
            # Host training capabilities — lets the UI explain why some fields are absent.
            "host": {
                "engine": engine,
                "is_windows": PLATFORM.is_windows,
                "multi_gpu": deepspeed_engine,
            },
        }
    )
    return attach_visibility_to_schema(schema)


def _finalize_field_importance(field: dict[str, Any]) -> None:
    """Ensure importance is set consistently after options injection."""
    path = field["path"]
    if field.get("required"):
        field["importance"] = "required"
    elif field.get("importance") not in ("required", "recommended", "advanced"):
        field["importance"] = "recommended" if path in RECOMMENDED_PATHS else "advanced"
    field["recommended"] = field["importance"] == "recommended"


__all__ = ["get_schema", "get_registries", "normalize_model_type", "get_capability"]
