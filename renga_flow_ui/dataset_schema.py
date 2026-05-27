"""Dataset TOML form schema for the web UI."""

from __future__ import annotations

from typing import Any

from renga_flow_ui.dataset_field_help import enrich_dataset_schema


def _field(
    path: str,
    label: str,
    ftype: str,
    *,
    default: Any = None,
    description: str = "",
    required: bool = False,
    recommended: bool = False,
    options: list[Any] | None = None,
    option_values: list[Any] | None = None,
    show_if_set: bool = False,
    show_when_field: str = "",
) -> dict[str, Any]:
    imp = "required" if required else ("recommended" if recommended else "advanced")
    out: dict[str, Any] = {
        "path": path,
        "label": label,
        "type": ftype,
        "default": default,
        "description": description,
        "required": required,
        "recommended": recommended,
        "importance": imp,
        "options": options,
    }
    if option_values is not None:
        out["option_values"] = option_values
    if show_if_set:
        out["show_if_set"] = True
    if show_when_field:
        out["show_when_field"] = show_when_field
    return out


def get_directory_fields() -> list[dict[str, Any]]:
    """Fields stored on each ``[[directory]]`` table (optional keys inherit from global root)."""
    return [
        _field(
            "path",
            "Folder path",
            "string",
            required=True,
            description="Absolute or relative path to images (and optional .txt / captions.json).",
        ),
        _field(
            "num_repeats",
            "Repeats per epoch",
            "integer",
            default=1,
            required=True,
            description="How many times this folder is repeated each epoch.",
        ),
        _field(
            "directory_caption",
            "Caption prefix / fallback",
            "string",
            description="Prepended to per-image captions, or used when an image has no caption file.",
        ),
        _field(
            "shuffle_tags",
            "Shuffle tags",
            "boolean",
            description="Shuffle delimiter-separated tags when caching (overrides global).",
        ),
        _field(
            "cache_shuffle_num",
            "Cache shuffle count",
            "integer",
            default=0,
            description="Caption shuffle/repeat count for cache augmentation (0 = off).",
        ),
        _field(
            "cache_shuffle_delimiter",
            "Tag delimiter",
            "string",
            default=", ",
            description="Delimiter between tags when shuffle tags is enabled.",
        ),
        _field(
            "shuffle_metadata",
            "Shuffle metadata order",
            "boolean",
            default=True,
            description="Shuffle image order when building metadata.",
        ),
        _field(
            "online_captions",
            "Online captions.json",
            "boolean",
            default=False,
            description="Read captions.json at train time instead of cache-only.",
        ),
        _field(
            "resolutions",
            "Resolutions override",
            "integer_list",
            description="Override global resolutions for this folder only.",
            options=[512, 640, 768, 1024, 1280, 1536],
            show_if_set=True,
        ),
        _field(
            "frame_buckets",
            "Frame buckets override",
            "integer_list",
            description="Override global frame buckets (1 = images).",
            options=[1, 9, 17, 25],
            show_if_set=True,
        ),
        _field(
            "enable_ar_bucket",
            "Enable AR buckets",
            "boolean",
            description="Override global aspect-ratio bucketing for this folder.",
            show_if_set=True,
        ),
        _field(
            "min_ar",
            "Min aspect ratio",
            "number",
            default=0.5,
            show_if_set=True,
        ),
        _field(
            "max_ar",
            "Max aspect ratio",
            "number",
            default=2.0,
            show_if_set=True,
        ),
        _field(
            "num_ar_buckets",
            "Num AR buckets",
            "integer",
            default=12,
            show_if_set=True,
        ),
        _field(
            "ar_buckets",
            "AR ratios (explicit)",
            "number_list",
            description="Override global AR list for this folder.",
            options=[0.75, 1.0, 1.25, 1.33, 1.5, 1.78, 2.0],
            show_when_field="enable_ar_bucket",
            show_if_set=True,
        ),
        _field(
            "size_buckets",
            "Size buckets",
            "json",
            description="[[width, height, frames], …] for this folder only.",
            show_if_set=True,
        ),
        _field("mask_path", "Mask folder", "string", show_if_set=True),
        _field("control_path", "Control folder", "string", show_if_set=True),
        _field("default_mask_file", "Default mask file", "string", show_if_set=True),
    ]


def get_augmentation_directory_fields() -> list[dict[str, Any]]:
    """Per-[[directory]] augmentation (nested ``augmentation`` table in TOML)."""
    from renga_flow.data.augmentation.names import MVP_PRESET_NAMES

    preset_options = sorted(MVP_PRESET_NAMES)
    return [
        _field(
            "enabled",
            "Enable augmentation",
            "boolean",
            default=False,
            description="Apply diversity transforms before VAE encode (images only).",
        ),
        _field(
            "preset",
            "Preset",
            "select",
            default="none",
            options=preset_options,
            description="Named bundle of strategies; override per strategy below.",
        ),
        _field(
            "seed_mode",
            "Seed mode",
            "select",
            default="deterministic_per_image",
            options=["deterministic_per_image"],
            show_if_set=True,
        ),
        _field(
            "variant_sampling",
            "Variant sampling",
            "select",
            default="probability",
            options=["probability", "enumerated"],
            description="How discrete strategies (e.g. flip) expand into cache rows.",
            show_if_set=True,
        ),
        _field(
            "strategies",
            "Strategy overrides (JSON)",
            "json",
            description='e.g. {"horizontal_flip": {"enabled": false}}',
            show_if_set=True,
        ),
    ]


def get_dataset_schema() -> dict[str, Any]:
    sections = [
        {
            "id": "resolutions",
            "title": "Resolutions & buckets",
            "description": "Default for every [[directory]] unless a folder overrides it below.",
            "fields": [
                _field(
                    "resolutions",
                    "Resolutions",
                    "integer_list",
                    default=[1024],
                    required=True,
                    recommended=True,
                    description="Long-side sizes (px). Add several for multi-resolution training.",
                    options=[512, 640, 768, 1024, 1280, 1536],
                ),
                _field(
                    "frame_buckets",
                    "Frame buckets",
                    "integer_list",
                    default=[1],
                    recommended=True,
                    description="1 = images only; higher values = video clip lengths.",
                    options=[1, 9, 17, 25],
                ),
                _field("enable_ar_bucket", "Enable AR buckets", "boolean", default=False, recommended=True),
                _field("min_ar", "Min aspect ratio", "number", default=0.5),
                _field("max_ar", "Max aspect ratio", "number", default=2.0),
                _field("num_ar_buckets", "Num AR buckets", "integer", default=12),
                _field(
                    "ar_buckets",
                    "AR ratios (explicit)",
                    "number_list",
                    description="Overrides min/max/num when set. Width÷height ratios.",
                    options=[0.75, 1.0, 1.25, 1.33, 1.5, 1.78, 2.0],
                    show_when_field="enable_ar_bucket",
                    show_if_set=True,
                ),
                _field(
                    "size_buckets",
                    "Size buckets (advanced)",
                    "json",
                    description="[[width, height, frames], …] — fixed sizes instead of AR bucketing.",
                    show_if_set=True,
                ),
            ],
        },
        {
            "id": "augmentation_global",
            "title": "Augmentation (global defaults)",
            "description": "Default augmentation for all folders unless a directory overrides it.",
            "fields": [
                _field(
                    "_dataset_augmentation",
                    "Global augmentation (JSON)",
                    "json",
                    description='{"enabled": false, "preset": "none"} — merged into each [[directory]].',
                    show_if_set=True,
                ),
            ],
        },
        {
            "id": "captions",
            "title": "Captions & shuffle (global defaults)",
            "description": "Default caption/shuffle behaviour for all folders. Override per folder in Directories.",
            "fields": [
                _field("shuffle_tags", "Shuffle tags", "boolean", default=False),
                _field("cache_shuffle_num", "Cache shuffle num", "integer", default=0),
                _field("cache_shuffle_delimiter", "Tag delimiter", "string", default=", "),
                _field("shuffle_metadata", "Shuffle metadata order", "boolean", default=True),
                _field("online_captions", "Online captions.json", "boolean", default=False),
                _field("subsample_ratio", "Subsample ratio", "number"),
            ],
        },
    ]
    schema = enrich_dataset_schema({"sections": sections})
    schema["directory_fields"] = enrich_directory_fields(get_directory_fields())
    schema["augmentation_directory_fields"] = _enrich_augmentation_directory_fields(
        get_augmentation_directory_fields()
    )
    return schema


def _enrich_augmentation_directory_fields(
    fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from renga_flow_ui.dataset_field_help import FIELD_HELP

    out = []
    for field in fields:
        f = dict(field)
        path = f.get("path", "")
        meta = FIELD_HELP.get(f"directory.augmentation.{path}")
        if meta:
            if not f.get("description") and meta.get("summary"):
                f["description"] = meta["summary"]
            f["help"] = meta.get("detail") or meta.get("summary")
            if meta.get("doc"):
                f["doc_path"] = meta["doc"]
        if not f.get("help"):
            f["help"] = f.get("description") or f.get("label") or path
        out.append(f)
    return out


def enrich_directory_fields(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach help text using ``directory.<key>`` entries in FIELD_HELP."""
    from renga_flow_ui.dataset_field_help import FIELD_HELP

    out = []
    for field in fields:
        f = dict(field)
        path = f.get("path", "")
        meta = FIELD_HELP.get(f"directory.{path}") or FIELD_HELP.get(path)
        if meta:
            if not f.get("description") and meta.get("summary"):
                f["description"] = meta["summary"]
            f["help"] = meta.get("detail") or meta.get("summary")
            if meta.get("doc"):
                f["doc_path"] = meta["doc"]
        if not f.get("help"):
            f["help"] = f.get("description") or f.get("label") or path
        out.append(f)
    return out


def get_directory_row_template() -> dict[str, Any]:
    return {
        "path": "",
        "num_repeats": 1,
        "directory_caption": "",
    }
