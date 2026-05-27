"""Config form schema and registry metadata for the web UI."""

from __future__ import annotations

from typing import Any

from renga_flow.config.defaults import DTYPE_MAP
from renga_flow.optim.resolver import scheduler_registry
from renga_flow.registry.model_capabilities import (
    ADAPTER_FIELD_TEMPLATES,
    capabilities_for_api,
    get_canonical_model_types,
    get_capability,
    model_capability_registry,
    normalize_model_type,
)
from renga_flow_ui.field_visibility import attach_visibility_to_schema
from renga_flow.registry.optimizers import (
    OPTIMIZER_ALIASES,
    VENDOR_OPTIMIZER_ALIASES,
    optimizer_registry,
)

DTYPE_OPTIONS = list(DTYPE_MAP.keys())
ACTIVATION_CHECKPOINTING_OPTIONS = [False, True, "unsloth"]
PARTITION_METHODS = ["parameters", "uniform", "manual"]
HAS_ADAPTER = {"field": "_has_adapter", "equals": True}

# Shown upfront: have trainer defaults but you should review/set them explicitly.
RECOMMENDED_PATHS: frozenset[str] = frozenset(
    {
        "output_dir",
        "adapter.type",
        "adapter.rank",
        "adapter.dim",
        "optimizer.lr",
        "lr_scheduler",
        "epochs",
        "micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "gradient_clipping",
        "logging_steps",
        "activation_checkpointing",
        "blocks_to_swap",
        "pipeline_stages",
        "save_every_n_epochs",
        "warmup_steps",
        "lr_scheduler_args.lr_min",
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
    options_from_model: bool = False,
    options_key: str | None = None,
    when_model_has_adapter: bool = False,
    allow_custom: bool = False,
    when_capability: str | dict[str, Any] | None = None,
    show_if_set: bool = False,
    show_if_set_exclude_zero: bool = False,
    visibility: dict[str, Any] | None = None,
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
            "Train with adapter (LoRA / LoKr)",
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
    for adapter_kind in ("lora", "lokr"):
        when_kind = {
            "all": [
                HAS_ADAPTER,
                {"field": "adapter.type", "equals": adapter_kind},
            ]
        }
        for spec in ADAPTER_FIELD_TEMPLATES.get(adapter_kind, []):
            f = _field_from_template(spec, None)
            f["when"] = when_kind
            fields.append(f)
    return fields


def _preview_section() -> dict[str, Any]:
    preview_models = [c.type_id for c in model_capability_registry.values() if c.preview]
    when_preview = _when_model(*preview_models) if preview_models else {"field": "model.type", "in": []}
    return {
        "id": "preview",
        "title": "Previews",
        "description": "Sample images during training (supported models only).",
        "when_any_model": preview_models,
        "fields": [
            _field("preview.enabled", "Enabled", "boolean", default=True, when=when_preview),
                _field(
                    "preview.prompts",
                    "Prompts",
                    "string_list",
                    when=when_preview,
                    description="Preview captions. Named {name, prompt} tables stay as JSON in TOML.",
                    placeholder="Type a prompt, then Enter",
                ),
            _field("preview.negative_prompt", "Negative prompt", "string", when=when_preview),
            _field("preview.width", "Width", "integer", default=1024, when=when_preview),
            _field("preview.height", "Height", "integer", default=1024, when=when_preview),
            _field("preview.num_inference_steps", "Inference steps", "integer", default=20, when=when_preview),
            _field("preview.guidance_scale", "Guidance scale", "number", default=7.0, when=when_preview),
            _field("preview.seed", "Seed", "integer", default=0, when=when_preview),
            _field("preview.seed_stride", "Seed stride", "integer", default=1, when=when_preview),
            _field("preview.preview_every_n_steps", "Preview every N steps", "integer", min_value=1, when=when_preview),
            _field("preview.preview_every_n_epochs", "Preview every N epochs", "integer", min_value=1, when=when_preview),
            _field("preview.preview_before_first_step", "Preview before first step", "boolean", when=when_preview),
            _field(
                "disable_block_swap_for_preview",
                "Disable block swap for preview",
                "boolean",
                when=when_preview,
                when_capability="block_swap",
            ),
        ],
    }


def get_registries() -> dict[str, Any]:
    from renga_flow_ui import datasets_store

    dataset_picker = datasets_store.list_for_training_picker()
    optimizers = sorted(
        set(optimizer_registry.keys())
        | set(VENDOR_OPTIMIZER_ALIASES.keys())
        | set(OPTIMIZER_ALIASES.keys())
    )
    return {
        "models": get_canonical_model_types(),
        "model_capabilities": capabilities_for_api(),
        "optimizers": optimizers,
        "optimizer_allow_custom": True,
        "schedulers": sorted(scheduler_registry.keys()),
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
                    recommended=True,
                    description="Root folder for run outputs.",
                ),
                _field(
                    "run_name",
                    "Run name",
                    "string",
                    importance="advanced",
                    description="Optional suffix for timestamped run folder.",
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
                    description="Built-in names, optional deps (adamw8bit, …), pytorch_optimizer classes, or module.Class path.",
                ),
                _field(
                    "optimizer.lr",
                    "Learning rate",
                    "number",
                    default=1e-4,
                    recommended=True,
                ),
                _field("optimizer.weight_decay", "Weight decay", "number"),
                _field(
                    "optimizer.betas",
                    "Adam betas",
                    "number_list",
                    default=[0.9, 0.999],
                    description="Two floats for Adam-style optimizers.",
                    options=[0.9, 0.99, 0.999, 0.95],
                ),
                _field("optimizer.momentum", "Momentum (SGD)", "number"),
                _field(
                    "optimizer.gradient_release",
                    "Gradient release",
                    "boolean",
                    description="Requires pipeline_stages = 1.",
                ),
                _field("optimizer.beta2_half_life", "Beta2 half-life", "number"),
                _field("optimizer.kahan_buffer_offload", "Kahan buffer offload", "boolean"),
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
                    description="Registry name (cosine, linear, …) or fully-qualified scheduler class.",
                ),
                _field("warmup_steps", "Warmup steps", "integer", default=0, min_value=0, recommended=True),
                _field(
                    "lr_scheduler_args.lr_min",
                    "Cosine lr_min",
                    "number",
                    default=0.0,
                    recommended=True,
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
                ),
                _field(
                    "micro_batch_size_per_gpu",
                    "Micro batch per GPU",
                    "integer",
                    default=1,
                    min_value=1,
                    recommended=True,
                ),
                _field(
                    "image_micro_batch_size_per_gpu",
                    "Image micro batch per GPU",
                    "json",
                    description="Dict or int for mixed modalities.",
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
                    recommended=True,
                ),
                _field(
                    "logging_steps",
                    "Logging steps",
                    "integer",
                    default=1,
                    min_value=1,
                    recommended=True,
                ),
                _field(
                    "steps_per_print",
                    "DeepSpeed steps_per_print",
                    "integer",
                    default=1,
                    importance="advanced",
                ),
                _field(
                    "synthetic_num_batches",
                    "Synthetic batches (debug)",
                    "integer",
                    importance="advanced",
                ),
                _field(
                    "pipeline_stages",
                    "Pipeline stages",
                    "integer",
                    default=1,
                    min_value=1,
                    recommended=True,
                ),
                _field("partition_method", "Partition method", "select", options=PARTITION_METHODS),
                _field("partition_split", "Partition split (manual)", "json"),
                _field(
                    "activation_checkpointing",
                    "Activation checkpointing",
                    "select",
                    options=[False, True, "unsloth"],
                    recommended=True,
                ),
                _field(
                    "reentrant_activation_checkpointing",
                    "Reentrant checkpointing",
                    "boolean",
                    importance="advanced",
                ),
                _field(
                    "blocks_to_swap",
                    "Blocks to swap",
                    "integer",
                    default=0,
                    min_value=0,
                    recommended=True,
                    when_capability="block_swap",
                ),
                _field("compile", "torch.compile", "boolean", default=False, importance="advanced"),
                _field("x_axis_examples", "TensorBoard x-axis = examples", "boolean"),
                _field("caching_batch_size", "Dataset cache batch size", "integer", default=1),
                _field("cache_num_proc", "Cache CPU workers", "integer", default=8, min_value=1),
                _field("cache_keep_in_memory", "Keep HF slice in RAM during cache", "boolean", default=False),
                _field(
                    "cache_format",
                    "Disk cache format",
                    "select",
                    options=["v2", "v1"],
                    default="v2",
                    importance="advanced",
                    description="v2 = mmap bf16 stacks (default). v1 = legacy pickle shards.",
                ),
                _field("dataloader_num_workers", "Train DataLoader workers", "integer", default=0, min_value=0),
                _field("dataloader_prefetch", "Prefetch next batch (thread)", "boolean", default=False),
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
            "id": "checkpoint",
            "title": "Checkpoints & export",
            "fields": [
                _field("checkpoint_every_n_epochs", "Checkpoint every N epochs", "integer", min_value=1),
                _field("checkpoint_every_n_minutes", "Checkpoint every N minutes", "number", min_value=0),
                _field("max_checkpoints_to_keep", "Max checkpoints to keep", "integer", min_value=1),
                _field(
                    "save_every_n_epochs",
                    "Save model every N epochs",
                    "integer",
                    default=1,
                    min_value=1,
                    recommended=True,
                ),
                _field("save_every_n_steps", "Save model every N steps", "integer", min_value=1),
                _field("save_every_n_examples", "Save every N examples", "integer", min_value=1),
                _field("save_dtype", "Save dtype", "select", options=DTYPE_OPTIONS),
                _field("save_full_model", "Save full model", "boolean"),
            ],
        },
        {
            "id": "eval",
            "title": "Evaluation",
            "flat_optional": True,
            "fields": [
                _field(
                    "eval_datasets",
                    "Eval datasets",
                    "json",
                    description='List of paths or [{name, config}] tables.',
                ),
                _field("eval_every_n_steps", "Eval every N steps", "integer", min_value=1),
                _field("eval_every_n_epochs", "Eval every N epochs", "integer", min_value=1),
                _field("eval_every_n_examples", "Eval every N examples", "integer", min_value=1),
                _field("eval_before_first_step", "Eval before first step", "boolean", default=True),
                _field("eval_gradient_accumulation_steps", "Eval grad accum", "integer", default=1),
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
            "id": "monitoring",
            "title": "Monitoring",
            "flat_optional": True,
            "fields": [
                _field("monitoring.enable_wandb", "Enable WandB", "boolean", default=False),
                _field("monitoring.enable_status_file", "Enable status.json for UI", "boolean", default=False),
                _field("monitoring.wandb_tracker_name", "WandB project", "string", default="renga-flow"),
                _field("monitoring.wandb_run_name", "WandB run name", "string"),
                _field("monitoring.wandb_api_key", "WandB API key", "string"),
            ],
        },
    ]
    return sections


def get_schema() -> dict[str, Any]:
    from renga_flow_ui.config_field_help import enrich_schema

    registries = get_registries()
    sections = get_sections()
    for section in sections:
        for field in section["fields"]:
            if field["path"] == "model.type":
                field["options"] = registries["models"]
            if field["path"] == "optimizer.type":
                field["options"] = registries["optimizers"]
            if field["path"] == "lr_scheduler":
                field["options"] = registries["schedulers"]
            if field["path"] == "dataset":
                field["options"] = [
                    p.get("label") or p["path"]
                    for p in registries.get("dataset_paths", [])
                ]
                field["option_values"] = [p["path"] for p in registries.get("dataset_paths", [])]
            _finalize_field_importance(field)
    schema = enrich_schema({"registries": registries, "sections": sections})
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
