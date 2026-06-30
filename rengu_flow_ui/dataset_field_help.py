"""Field help for dataset TOML form (links to docs/user/*.md)."""

from __future__ import annotations

from typing import Any

FIELD_HELP: dict[str, dict[str, str]] = {
    "resolutions": {
        "summary": "Long-side pixel values used to build aspect-ratio buckets (e.g. [512, 768, 1024]).",
        "detail": (
            "Each value defines the longer edge for a set of buckets at that scale. "
            "Adding a resolution triggers latent caching for the new size on the next run. "
            "To stage resolutions over the run instead of mixing them, use resolution_schedule."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "frame_buckets": {
        "summary": "Frame counts for temporal buckets: 1 = images, higher = video (e.g. [1] or [1, 16, 24]).",
        "detail": "Use [1] for image-only training. Add video frame counts only when your model and dataset support video; mixing image and video in the same run requires the model to handle both modalities.",
        "doc": "docs/user/dataset-config.md",
    },
    "enable_ar_bucket": {
        "summary": "Allow images with different aspect ratios to train in separate buckets.",
        "detail": (
            "When off (default), all images are center-cropped or padded to the configured resolution. "
            "Turn on if your dataset has a wide mix of portrait and landscape images — bucketing avoids "
            "distortion and keeps more of each image's content. Requires min_ar, max_ar, and num_ar_buckets (or ar_buckets)."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "min_ar": {
        "summary": "Narrowest width/height ratio for AR buckets (e.g. 0.5 = 1:2 portrait).",
        "detail": "Images narrower than this ratio are cropped up to the limit. Required when enable_ar_bucket is on and ar_buckets is not set.",
        "doc": "docs/user/dataset-config.md",
    },
    "max_ar": {
        "summary": "Widest width/height ratio for AR buckets (e.g. 2.0 = 2:1 landscape).",
        "detail": "Images wider than this ratio are cropped down to the limit. Required when enable_ar_bucket is on and ar_buckets is not set.",
        "doc": "docs/user/dataset-config.md",
    },
    "num_ar_buckets": {
        "summary": "How many evenly spaced aspect-ratio buckets to generate between min_ar and max_ar.",
        "detail": "More buckets = less cropping per image, but more distinct shapes the model must handle. 5–10 is a common range for mixed-aspect datasets.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.path": {
        "summary": "Folder with images (and optional .txt or captions.json).",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.num_repeats": {
        "summary": "Multiplies how many times this folder's images appear per epoch.",
        "detail": (
            "1 = each image seen once per epoch. Raise to over-sample a small folder relative to larger "
            "ones — the epoch gets proportionally longer. Combine with max_images to cap the absolute count."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "directory.directory_caption": {
        "summary": "Caption used when an image has no .txt file; also prepended as a prefix when a per-image caption exists.",
        "detail": (
            "Example: set to 'style: ' and every captioned image becomes 'style: <original caption>'. "
            "Leave empty to use only per-image captions with no prefix."
        ),
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
    "directory.augmentation.branches_per_image": {
        "summary": "Augmented copies cached per image besides the pristine original (deterministic seeds). 0 = original only. Higher = more regularization and more cache disk.",
        "doc": "docs/user/dataset-augmentation.md",
    },
    "directory.augmentation.strategies": {
        "summary": "Per-strategy parameter overrides (JSON object).",
        "doc": "docs/user/dataset-augmentation.md",
    },
    "directory.shuffle_metadata": {
        "summary": "Randomize image order when building this folder's metadata (deterministic per-folder seed).",
        "detail": "On by default globally; per-folder override. Turn off only if you need a predictable, filename-sorted order in the cache.",
        "doc": "docs/user/dataset-config.md",
    },
    "directory.online_captions": {
        "summary": "Re-read captions.json from disk at training time instead of relying on cached metadata.",
        "detail": "Enable if you update captions.json between training runs without regenerating the full cache. Only applies to this folder.",
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
    "no_upscale": {
        "summary": "With size_buckets, discard images smaller than their bucket instead of upscaling them.",
        "detail": (
            "Off (default): an image smaller than its target bucket is enlarged to fill it. "
            "On: such images are dropped entirely, so only images that already meet the bucket "
            "resolution are kept (and downscaled to fit). Use it to keep training on genuine "
            "detail and avoid blurry upscaled samples. Only affects size_buckets; AR bucketing "
            "already keeps each image's native resolution."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "shuffle_metadata": {
        "summary": "Randomize image order when building metadata across all folders (deterministic per-folder seed, default on).",
        "detail": "Turn off only if you need predictable, filename-sorted metadata order for debugging or reproducing an exact cache layout.",
        "doc": "docs/user/dataset-config.md",
    },
    "online_captions": {
        "summary": "Re-read captions.json from disk at training time for all folders.",
        "detail": "Enable globally if you update caption files between runs without regenerating the full cache. Per-folder directory.online_captions overrides this.",
        "doc": "docs/user/dataset-config.md",
    },
    "subsample_ratio": {
        "summary": "Use a fraction of images per epoch. Mutually exclusive with max_images.",
        "detail": (
            "Fractional per-epoch limiter (e.g. 0.1). By default the window rotates over the "
            "whole folder across epochs (turn subsample_shuffle off to freeze it). Cannot be combined "
            "with max_images in the same place — pick one."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "max_images": {
        "summary": "Absolute image cap per folder per epoch (per size bucket).",
        "detail": (
            "Caps how many images a folder contributes each epoch. By default the window "
            "rotates over the whole folder across epochs so every image is eventually seen "
            "(balances folders of different sizes without wasting data). Folders with fewer "
            "images than the cap repeat up to it. Turn subsample_shuffle off to freeze the subset. "
            "Mutually exclusive with subsample_ratio."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "subsample_shuffle": {
        "summary": "Rotate the sampled window every epoch (on) or keep the same subset (off).",
        "detail": (
            "Applies to whichever limiter is set (subsample_ratio or max_images). On "
            "(default) advances the per-epoch window so the whole folder is eventually "
            "used; off keeps the same subset each epoch (the old static_sampling = true)."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "tag_dropout_enabled": {
        "summary": "Randomly drop tags so the model does not over-rely on any single tag.",
        "detail": (
            "Defines the dropout distribution (probability/mode/rules). How it is applied "
            "depends on the text-embedding cache:\n"
            "• cache_text_embeddings = false (live): dropout runs per sample at training time "
            "(keeps the text encoder on the GPU, ~22 ms/step + ~1.2 GB VRAM).\n"
            "• cache_text_embeddings = true (cached): the dropout is pre-baked into the embedding "
            "cache. cached_caption_variants = 1 bakes one fixed variant for the whole dataset "
            "(diffusion-pipe's default); set it >= 2 to bake that many variants that rotate across "
            "epochs without lengthening them."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "cached_caption_variants": {
        "summary": "How many tag-dropout/shuffle caption variants to bake into the TE cache.",
        "detail": (
            "Only used when cache_text_embeddings = true. K = 1 caches the caption as written when "
            "there is no dropout/shuffle, or bakes a single fixed augmented variant when there is "
            "(diffusion-pipe's default). K >= 2 samples the tag-dropout distribution (and optional "
            "tag shuffle) K times per caption, caches each variant's embedding, and rotates them "
            "across epochs — so an epoch is still one pass over the images. This is the cached-path "
            "equivalent of live tag dropout; higher K = more regularization and more cache disk."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "cached_caption_shuffle": {
        "summary": "Also shuffle tag order in each baked cached-caption variant.",
        "detail": (
            "When generating cached caption variants, randomly reorder the tags in each variant "
            "(deterministic per image/variant). Composes with tag dropout. Has effect only on "
            "the cached path with cached_caption_variants active."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "tag_dropout_probability": {
        "summary": "Default per-tag drop probability (0–1) for tags without a rule.",
        "detail": (
            "Probability that any given tag is dropped, applied to tags not matched by a rule "
            "in tag_dropout_rules. 0 = never drop; per-tag rules override this value. Only used "
            "when tag dropout is enabled."
        ),
        "doc": "docs/user/dataset-config.md",
    },
    "tag_dropout_rules": {
        "summary": "Per-tag overrides: a list of {tags, drop_probability} (or a tags_file).",
        "detail": (
            "Each rule sets a custom drop probability for specific tags, overriding "
            "tag_dropout_probability. In the UI this is JSON, e.g. "
            '[{"tags": ["hero"], "drop_probability": 0.1}]. A rule may use "tags_file" instead '
            "of an inline list, pointing to a .txt file with one tag per line."
        ),
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
