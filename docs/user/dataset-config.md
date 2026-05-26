# Dataset configuration

How to point the training config at a dataset and what the dataset TOML contains. When a dataset TOML is provided and you are not using synthetic data, the framework builds a directory-based dataset, runs the latent and text-embedding cache, and trains on that data.

## Main config: referencing a dataset

In your main TOML config, set the `dataset` key to the path of a **dataset TOML file**:

```toml
dataset = "examples/minimal_dataset.toml"
```

The path can be relative to the working directory or absolute. If you omit `dataset` or use synthetic data only (see below), training uses an in-memory synthetic dataset instead.

## Dataset TOML schema

The dataset TOML describes **directories** of images (and optionally video), **resolutions** and **buckets**, and caption behaviour.

### Directories (required for real data)

Use **`[[directory]]`** (array of tables). Each entry must have:

| Key | Required | Description | Values |
|-----|----------|-------------|--------|
| **`path`** | Yes | Folder containing images (and optional `.txt` caption files or `captions.json`). | Absolute or relative path string. |
| **`num_repeats`** | Yes | How many times this directory is repeated per epoch. | Integer &gt; 0. |

#### Optional per-directory keys

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`directory_caption`** | Single string used for captions in this directory. **When an image has no caption** (no `.txt` or `captions.json` entry): used as the full caption. **When an image has a caption**: prepended as a prefix (e.g. `"style: "` + image caption). | Any string. Use `""` (or omit) for no prefix / no fallback. | `""` |
| **`mask_path`** | Directory containing per-image mask files (same filenames as images, different extension). | Path string, or omit. | Not set. |
| **`control_path`** | Directory containing control images (e.g. for controlnet). | Path string, or omit. | Not set. |
| **`default_mask_file`** | Single mask file used for all images when no per-image mask is found. | Path to a file, or omit. | Not set. |
| **`resolutions`** | Override global resolutions for this directory only. | List of numbers, e.g. `[512, 768, 1024]`. | From global `resolutions`. |
| **`frame_buckets`** | Override global frame counts (1 = image, &gt;1 = video). | List of integers, e.g. `[1]` or `[1, 16, 24]`. | From global `frame_buckets`. |
| **`enable_ar_bucket`** | Enable aspect-ratio bucketing for this directory. | `true` or `false`. | From global (default `false`). |
| **`ar_buckets`** | Explicit list of aspect ratios (width/height) for bucketing. | List of floats, e.g. `[1.0, 1.25, 1.5]`. | If unset, derived from `min_ar` / `max_ar` / `num_ar_buckets`. |
| **`size_buckets`** | Use fixed size buckets instead of AR bucketing. Each entry is `[width, height, frames]`. | List of arrays, e.g. `[[512, 512, 1], [768, 768, 1]]`. | Not set (AR bucketing used if enabled). |
| **`cache_shuffle_num`** | Number of caption shuffles and repeats for cache (data augmentation). | Integer ≥ 0. `0` = no shuffle/repeat. | `0` (from global). |
| **`cache_shuffle_delimiter`** | Delimiter used when shuffling tags inside a caption (e.g. comma for "a, b, c"). | String, e.g. `", "`. | `", "` (from global). |
| **`shuffle_tags`** | Shuffle comma-separated (or delimiter-separated) tags in each caption. | `true` or `false`. | From global (default `false`). |

Example:

```toml
resolutions = [1024]
frame_buckets = [1]

[[directory]]
path = "/path/to/your/images"
num_repeats = 1
# optional: directory_caption = "style: portrait"
# optional: directory_caption = "style: "   # prefix when caption exists
```

### Global options (dataset TOML root)

These apply to all directories unless overridden per-directory.

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`resolutions`** | Resolution values used for aspect-ratio bucketing (longer side). | List of integers, e.g. `[512, 768, 1024]`. | Required if not using `size_buckets`. |
| **`frame_buckets`** | Frame counts: `1` = image, higher = video. | List of integers, e.g. `[1]` or `[1, 16, 24]`. | `[1]`. |
| **`enable_ar_bucket`** | Use aspect-ratio bucketing (different resolutions per aspect ratio). | `true` or `false`. | `false`. |
| **`min_ar`** | Minimum aspect ratio (width/height) when computing buckets. | Float &gt; 0. | Required if `enable_ar_bucket` and no `ar_buckets`. |
| **`max_ar`** | Maximum aspect ratio when computing buckets. | Float &gt; 0. | Required if `enable_ar_bucket` and no `ar_buckets`. |
| **`num_ar_buckets`** | Number of aspect-ratio buckets between `min_ar` and `max_ar`. | Integer &gt; 0. | Required if `enable_ar_bucket` and no `ar_buckets`. |
| **`ar_buckets`** | Explicit list of aspect ratios (overrides min_ar/max_ar/num). | List of floats. | Not set. |
| **`size_buckets`** | Fixed size buckets instead of AR; each entry `[width, height, frames]`. | List of arrays. | Not set. |
| **`shuffle_tags`** | Shuffle delimiter-separated tags in captions. | `true` or `false`. | `false`. |
| **`cache_shuffle_num`** | Caption shuffle/repeat count for cache. | Integer ≥ 0. | `0`. |
| **`cache_shuffle_delimiter`** | Delimiter for tag shuffling. | String. | `", "`. |
| **`shuffle_metadata`** | Shuffle image order when building metadata (deterministic seed from directory path). | `true` / `false` | `true` |
| **`online_captions`** | Read captions from `captions.json` at training time instead of only from cached metadata. | `true` / `false` | `false` |
| **`subsample_ratio`** | Use only a fraction of the dataset (e.g. for debugging). | Float in (0, 1]. | Not set (use full dataset). |

### Captions

- **Per-image `.txt` files:** One caption **per line**. Empty lines are skipped. If the file is empty or missing, the image uses **`directory_caption`** (if set on that `[[directory]]`) as the full caption, or an empty caption if not set.
- **`captions.json`:** If present in a directory, it is used instead of `.txt` files. Format: `{ "image1.png": ["caption1", "caption2"], ... }` (list of captions per image for multi-caption).
- **`directory_caption`:** One option for both roles: when there is no per-image caption, it is used as the full caption; when there is a caption, it is prepended as a prefix (e.g. `"style: "`).

## Dataset augmentation (planned)

Optional **image diversity** settings (colour jitter, flip, HDR-style tone mapping, etc.) are specified in [Dataset augmentation](dataset-augmentation.md). Until implemented, augmentation keys in a dataset TOML may be ignored. Augmentation interacts with **latent caching**; see that page for `seed_mode` and compatibility.

## Cache and CLI flags

Before training, the framework runs a **cache** step: it encodes images to latents and runs text encoders, then stores results on disk so training can iterate quickly.

| Flag | Purpose | When to use |
|------|---------|--------------|
| **`--cache_only`** | Run the cache step and then exit (no training). | Pre-fill cache; requires a dataset config. Does nothing with synthetic-only. |
| **`--regenerate_cache`** | Ignore existing cache and recompute. | After changing captions, images, or model. |
| **`--trust_cache`** | Use existing cache metadata and iteration order without recomputing. | Faster startup when data and model are unchanged. |

Cache is stored under each directory’s `cache/<model_name>/` (e.g. `path/cache/sdxl/`).

## Synthetic vs real data

- If the main config has **`synthetic_num_batches`** set, training uses an in-memory synthetic dataset and **does not** use the dataset TOML for data (only for copying into the run directory). No cache is run.
- If the main config has a **`dataset`** path and **no** `synthetic_num_batches`, the dataset TOML is loaded and must contain at least one `[[directory]]` with `path` and `num_repeats`. Cache runs, then training uses the cached data.

## Evaluation datasets

The main config’s **`eval_datasets`** is a list. Each entry is either:

- A path to a dataset TOML, or  
- A table with `name` and `config` (path), e.g. `{ name = "my_eval", config = "path/to/eval_dataset.toml" }`.

These are loaded for evaluation hooks; the training loop uses the main dataset (real or synthetic) for training.

## Inspecting a dataset (`--dump_dataset`)

Print one JSON line per image (path and resolved captions) without loading a model or GPU:

```bash
python -m renga_flow.main --dump_dataset examples/smoke_cc0_dataset.toml
```

Use this to verify directory paths, `.txt` captions, and `captions.json` before a long cache or training run. The official smoke dataset for docs and tests is `examples/smoke_cc0_dataset.toml` (12 CC0 images under `tests/fixtures/smoke_cc0/`).
