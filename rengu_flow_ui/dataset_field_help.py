"""Field help for dataset TOML form (links to docs/user/*.md)."""

from __future__ import annotations

from typing import Any

FIELD_HELP: dict[str, dict[str, str]] = {
    "resolutions": {
        "summary": "Long-side resolutions for aspect-ratio buckets.",
        "doc": "docs/user/dataset-config.md",
    },
    "frame_buckets": {
        "summary": "1 = images; higher values = video frame counts.",
        "doc": "docs/user/dataset-config.md",
    },
    "enable_ar_bucket": {
        "summary": "Bucket by aspect ratio instead of a single resolution.",
        "doc": "docs/user/dataset-config.md",
    },
    "min_ar": {"summary": "Minimum width/height ratio for AR buckets.", "doc": "docs/user/dataset-config.md"},
    "max_ar": {"summary": "Maximum width/height ratio for AR buckets.", "doc": "docs/user/dataset-config.md"},
    "num_ar_buckets": {"summary": "Number of AR buckets between min and max.", "doc": "docs/user/dataset-config.md"},
    "shuffle_tags": {
        "summary": "Shuffle comma-separated caption tags when caching.",
        "doc": "docs/user/dataset-config.md",
    },
    "cache_shuffle_num": {
        "summary": "Caption shuffle/repeat count for cache augmentation.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.path": {
        "summary": "Folder with images (and optional .txt or captions.json).",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.num_repeats": {
        "summary": "How many times this folder is repeated per epoch.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.directory_caption": {
        "summary": "Default or prefix caption for images in this folder.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.shuffle_tags": {
        "summary": "Shuffle delimiter-separated tags when caching this folder.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.cache_shuffle_num": {
        "summary": "Caption shuffle/repeat count for this folder only.",
        "doc": "docs/user/dataset-config.md",
    },
    "_dataset_augmentation": {
        "summary": "Default augmentation for all directories (merged per folder).",
        "doc": "docs/user/dataset-augmentation.md",
    },
    "directory.augmentation.enabled": {
        "summary": "Apply image diversity transforms before latent cache (images only).",
        "doc": "docs/user/dataset-augmentation.md",
    },
    "directory.augmentation.preset": {
        "summary": "Named bundle of augmentation strategies.",
        "doc": "docs/user/dataset-augmentation.md",
    },
    "directory.augmentation.seed_mode": {
        "summary": "deterministic_per_image (cache-safe) or stochastic (not supported with fixed cache).",
        "doc": "docs/user/dataset-augmentation.md",
    },
    "directory.augmentation.variant_sampling": {
        "summary": "probability = one random branch; enumerated = all flip branches in cache.",
        "doc": "docs/user/dataset-augmentation.md",
    },
    "directory.augmentation.strategies": {
        "summary": "Per-strategy parameter overrides (JSON object).",
        "doc": "docs/user/dataset-augmentation.md",
    },
    "directory.cache_shuffle_delimiter": {
        "summary": "Tag delimiter for this folder when shuffle tags is on.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.shuffle_metadata": {
        "summary": "Shuffle image order when building metadata for this folder.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.online_captions": {
        "summary": "Read captions.json at train time for this folder.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.resolutions": {
        "summary": "Override global resolutions for this folder only.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.frame_buckets": {
        "summary": "Override global frame buckets for this folder.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.enable_ar_bucket": {
        "summary": "Override global AR bucketing for this folder.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.mask_path": {
        "summary": "Per-image mask folder for this directory.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.control_path": {
        "summary": "Control images folder for this directory.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.default_mask_file": {
        "summary": "Single mask file used for all images in this folder.",
        "doc": "docs/user/dataset-config.md",
    },
    "ar_buckets": {
        "summary": "Explicit aspect-ratio list (overrides min/max/num).",
        "doc": "docs/user/dataset-config.md",
    },
    "size_buckets": {
        "summary": "Fixed [width, height, frames] buckets instead of AR bucketing.",
        "doc": "docs/user/dataset-config.md",
    },
    "cache_shuffle_delimiter": {
        "summary": "Delimiter between tags when shuffle_tags is enabled.",
        "doc": "docs/user/dataset-config.md",
    },
    "shuffle_metadata": {
        "summary": "Shuffle image order when building metadata (deterministic per folder).",
        "doc": "docs/user/dataset-config.md",
    },
    "online_captions": {
        "summary": "Read captions.json at train time instead of cache-only captions.",
        "doc": "docs/user/dataset-config.md",
    },
    "subsample_ratio": {
        "summary": "Use only a fraction of images (e.g. 0.1 for quick tests).",
        "doc": "docs/user/dataset-config.md",
    },
}


def enrich_dataset_schema(schema: dict[str, Any]) -> dict[str, Any]:
    for section in schema.get("sections", []):
        for field in section.get("fields", []):
            path = field.get("path", "")
            if not path:
                continue
            meta = FIELD_HELP.get(path)
            if meta:
                if not field.get("description") and meta.get("summary"):
                    field["description"] = meta["summary"]
                field["help"] = meta.get("detail") or meta.get("summary")
                if meta.get("doc"):
                    field["doc_path"] = meta["doc"]
            if not field.get("help"):
                field["help"] = field.get("description") or field.get("label") or path
            if not field.get("description"):
                field["description"] = field["help"]
    schema.setdefault("doc_links", [
        {"title": "Dataset configuration", "path": "docs/user/dataset-config.md"},
        {"title": "Dataset augmentation", "path": "docs/user/dataset-augmentation.md"},
        {"title": "Cache and CLI flags", "path": "docs/user/dataset-config.md"},
    ])
    return schema
