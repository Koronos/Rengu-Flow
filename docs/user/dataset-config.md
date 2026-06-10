# Dataset configuration

How to point the training config at a dataset and what the dataset TOML contains. When a dataset TOML is provided and you are not using synthetic data, the framework builds a directory-based dataset, runs the latent and text-embedding cache, and trains on that data.

## Main config: referencing a dataset

In your main TOML config, set the `dataset` key to the path of a **dataset TOML file**, or to a **list of paths** that are merged at train time (all `[[directory]]` tables; global options from the first file):

```toml
dataset = "examples/minimal_dataset.toml"
```

```toml
dataset = [
  "rengu-flow-dataset:1:portraits",
  "rengu-flow-dataset:2:landscapes",
]
```

The part after the second colon is optional and only helps you read the file; only the numeric library id is used when training runs.

Each path can be relative to the working directory or absolute. Library refs (`rengu-flow-dataset:<id>`) are resolved when a job is staged from the UI. If you omit `dataset` or use synthetic data only (see below), training uses an in-memory synthetic dataset instead.

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
| **`mask_path`** | Folder of per-image **mask** files paired with images in **`path`** (see [Masks and control images](#masks-and-control-images-optional-paired-folders)). | Absolute or relative path string; omit if unused. | Not set. |
| **`control_path`** | Folder of per-image **control/source** images paired with images in **`path`** for edit-style training (see [Masks and control images](#masks-and-control-images-optional-paired-folders)). **Not** for SD ControlNet-style adapters. | Absolute or relative path string; omit for normal image+caption training. | Not set. |
| **`default_mask_file`** | Single mask file used for all images when no per-image mask is found in **`mask_path`**. | Path to a file, or omit. | Not set. |
| **`resolutions`** | Override global resolutions for this directory only. | List of numbers, e.g. `[512, 768, 1024]`. | From global `resolutions`. |
| **`frame_buckets`** | Override global frame counts (1 = image, &gt;1 = video). | List of integers, e.g. `[1]` or `[1, 16, 24]`. | From global `frame_buckets`. |
| **`enable_ar_bucket`** | Enable aspect-ratio bucketing for this directory. | `true` or `false`. | From global (default `false`). |
| **`ar_buckets`** | Explicit list of aspect ratios (width/height) for bucketing. | List of floats, e.g. `[1.0, 1.25, 1.5]`. | If unset, derived from `min_ar` / `max_ar` / `num_ar_buckets`. |
| **`size_buckets`** | Use fixed size buckets instead of AR bucketing. Each entry is `[width, height, frames]`. | List of arrays, e.g. `[[512, 512, 1], [768, 768, 1]]`. | Not set (AR bucketing used if enabled). |
| **`cache_shuffle_num`** | Number of caption shuffles and repeats for cache (only when `shuffle_tags` is on). | Integer ≥ 0. Ignored if `shuffle_tags` is `false`. | `1` (from global). |
| **`cache_shuffle_delimiter`** | Delimiter used when shuffling tags inside a caption (e.g. comma for "a, b, c"). Only used when `shuffle_tags` is on. | String, e.g. `", "`. | `", "` (from global). |
| **`shuffle_tags`** | Shuffle comma-separated (or delimiter-separated) tags in each caption. | `true` or `false`. | From global (default `false`). |
| **`subsample_ratio`** | **Fractional** per-epoch limiter for this folder's rows per size bucket. Rotates each epoch by default. **Mutually exclusive with `max_images`.** | Float in (0, 1]. | `1` (all rows). |
| **`max_images`** | **Absolute** per-epoch image cap from this folder, per size bucket. Rotates each epoch by default. **Mutually exclusive with `subsample_ratio`.** | Integer &gt; 0. | Not set (no cap); inherits the dataset default if one is set. |
| **`static_sampling`** | Use the **same** images every epoch (no rotation) for whichever limiter is active. | `true` or `false`. | `false` (rotate). |

#### Per-epoch image limiting: `subsample_ratio` vs `max_images`

There are two ways to use only part of a folder each epoch — pick **one** (they are mutually exclusive in the same scope; setting both raises a config error):

- **`subsample_ratio = f`** — a **fraction** (e.g. `0.25` = a quarter of the rows). Good for quick debug runs.
- **`max_images = N`** — an **absolute count**. Good for **balancing several folders of very different sizes** (e.g. ten style folders with 10–100 images each) so no folder dominates.

Both share the same per-epoch behavior, governed by `static_sampling`:

- **Rotating (default, `static_sampling = false`).** Each epoch serves a *different* window, advancing through the whole folder and wrapping around. Over `ceil(total / limit)` epochs every image is seen — you get the limit **and** eventually use the entire dataset. Recommended.
- **Static (`static_sampling = true`).** The same first images are used every epoch; the rest of the folder is never seen. Use only when you deliberately want a fixed subset.
- **Fewer images than `max_images`.** The folder repeats its images up to `N` (repeat-to-N), so its per-epoch count matches folders that do have `N`. To over-sample a small folder beyond `N`, use `num_repeats`.
- **Per size bucket.** The limit applies per (folder, size bucket). With aspect-ratio bucketing **off** (the common case) a folder has a single bucket, so the limit is effectively per folder. With AR bucketing on, each bucket is limited independently.
- **Interaction.** `num_repeats` still multiplies the per-epoch count (`limit × num_repeats` rows). The limit keeps each epoch's length constant, so `steps_per_epoch` is unchanged — only *which* images appear rotates. (The root-level `subsample_ratio` in [Global options](#global-options-dataset-toml-root) is a separate, static trim of the combined schedule and is unaffected by `static_sampling`.)
- **Resuming / workers.** Rotation is derived from the epoch number, so resuming from a checkpoint continues the rotation correctly. With the default `dataloader_num_workers = 0` it works out of the box; with `dataloader_num_workers > 0` the loader re-creates workers at each epoch boundary so they pick up the new window.

```toml
# Global default cap; each style folder contributes 10 rotating images/epoch.
max_images = 10

[[directory]]
path = "styles/watercolor"   # 12 images → rotates through all over time
num_repeats = 1

[[directory]]
path = "styles/lineart"      # 100 images → 10 rotate in each epoch
num_repeats = 1

[[directory]]
path = "styles/rare"         # 6 images → repeated up to 10 each epoch
num_repeats = 1
static_sampling = true       # this folder: fixed subset instead of rotating
```

#### Path resolution (`path`, `mask_path`, `control_path`)

These keys are **independent folder paths**. They are not joined automatically — each points at its own directory on disk.

- **Absolute paths** — use as-is, e.g. `path = "/home/you/data/portraits"`.
- **Relative paths** — resolved from the **training working directory** (the process cwd when you launch training). For CLI and UI jobs that is usually the **repo / install root**. Relative paths in exported dataset TOMLs may also be resolved against the dataset TOML’s folder when the UI writes absolute paths for portability.

`mask_path` and `control_path` are **sibling folders** to `path`, not subfolders scanned inside it. A typical layout:

```text
my_dataset/
  targets/     ← path
  sources/     ← control_path (optional)
  masks/       ← mask_path (optional)
```

### Masks and control images (optional paired folders)

Use these only when your training recipe needs extra files per image. For standard SDXL or Cosmos Predict2 LoRA/finetune on images and captions, leave **`mask_path`**, **`control_path`**, and **`default_mask_file`** unset.

#### Masks (`mask_path`, `default_mask_file`)

Optional grayscale masks for loss weighting or masked training. Each mask is matched to an image in **`path`** by **base filename** (stem): `targets/photo.png` looks for `masks/photo.png`, `masks/photo.jpg`, etc. The extension may differ. If **`mask_path`** is set but an image has no matching file, training continues **without** a mask for that image (a warning is logged). **`default_mask_file`** is a single fallback mask used when no per-image match exists.

#### Control images (`control_path`)

Optional **paired source/control images** for edit-style datasets (target image in **`path`**, control image in **`control_path`**). This is **not** the same as loading a ControlNet adapter in the model config — it is a dataset pairing used when the model pipeline expects a control image alongside each target during caching.

- **When to set:** only when your model recipe documents paired control+target training. Omit for normal image+caption runs.
- **Pairing:** same rule as masks — match by stem. `targets/0001.png` requires `sources/0001.png` (or `0001.jpg`, etc.) inside **`control_path`**.
- **Strict:** if **`control_path`** is set, **every** image in **`path`** must have a matching control file; missing pairs raise an error at metadata build time.
- **Relation to masks:** independent. You can use masks only, control images only, both, or neither.

Example:

```toml
[[directory]]
path = "datasets/edit_set/targets"
control_path = "datasets/edit_set/sources"
num_repeats = 1
```

```text
targets/0001.png  ← train on this
sources/0001.png  ← paired control image (same stem)
targets/0002.jpg
sources/0002.png  ← extension can differ; stem must match
```

### Captions: `.txt` vs `captions.json`

| Source | Format | Behaviour |
|--------|--------|-----------|
| **Sidecar `.txt`** | One **line** = one caption variant for that image. | Multiple lines → multiple training rows per image (separate text-embedding cache entries). Empty file → one empty caption. |
| **`captions.json`** in the image folder | JSON object: filename → caption or **list of captions**. | If present, **overrides** sidecar `.txt` for that directory. Example: `{ "photo.jpg": ["tag1, style", "alt description"] }`. A single string value is treated as one caption. |
| **`directory_caption`** | One string in TOML. | Used when there is no `.txt` and no JSON entry; when a per-image caption exists, it is **prepended** as a prefix (see table above). |

Inspect resolved captions with [`--dump_dataset`](#inspecting-a-dataset---dump_dataset) before caching.

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
| **`cache_shuffle_num`** | Caption shuffle/repeat count for cache (ignored when `shuffle_tags` is `false`). | Integer ≥ 0. | `1`. |
| **`cache_shuffle_delimiter`** | Delimiter for tag shuffling (ignored when `shuffle_tags` is `false`). | String. | `", "`. |
| **`shuffle_metadata`** | Shuffle image order when building metadata (deterministic seed from directory path). | `true` / `false` | `true` |
| **`online_captions`** | Read captions from `captions.json` at training time instead of only from cached metadata. | `true` / `false` | `false` |
| **`subsample_ratio`** | Fraction of the combined training schedule (e.g. `0.25` for quick debug runs). | Float in (0, 1]. | `1` (full dataset). |
| **`max_images`** | Default absolute image cap per folder per epoch (per size bucket); rotates each epoch unless `static_sampling`. Per-folder keys override it. Mutually exclusive with a per-folder `subsample_ratio`. See [per-epoch limiting](#per-epoch-image-limiting-subsample_ratio-vs-max_images). | Integer &gt; 0. | Not set (no cap). |
| **`static_sampling`** | Default for whether the active limiter (subsample ratio or max images) uses a fixed subset every epoch instead of rotating. | `true` / `false` | `false` (rotate). |
| **`tag_dropout_enabled`** | Enable random tag dropout at training time. Requires `cache_text_embeddings = false` in the model config (captions must be encoded at training time). | `true` / `false` | `false` |
| **`tag_dropout_probability`** | Default drop probability for tags not in a rule. | Float in [0, 1]. | — |
| **`tag_dropout_mode`** | `per_tag` or `full`. | String | `per_tag` |
| **`tag_dropout_rules`** | List of `{ tags, drop_probability }` and/or `tags_file`. | Tables / JSON in UI | — |
| **`uncond_fraction`** | Fraction of samples with empty caption (CFG). | Float in [0, 1]. | `0` |
| **`tag_match_case_sensitive`** | Case-sensitive tag matching in rules. | `true` / `false` | `false` |

### Staged multi-resolution: `resolution_schedule`

By default, when you list several `resolutions` they are **mixed uniformly**
throughout training (every epoch samples all of them, weighted by image count).
The optional `[resolution_schedule]` section lets you instead **split training
progress across resolutions** — for example, train the first third at 512, the
second third at 768, and the last third at 1024 — or mix a chosen subset.

```toml
resolutions = [512, 768, 1024]

[resolution_schedule]
enabled = true

# Staged (no mixing): each resolution gets ~1/3 of the run.
[[resolution_schedule.stage]]
resolutions = [512]
fraction = 0.33

[[resolution_schedule.stage]]
resolutions = [768]
fraction = 0.33

[[resolution_schedule.stage]]
resolutions = [1024]
fraction = 0.34
```

How it works:

- **Stages run in order.** Each stage lists which `resolutions` are active and a
  `fraction` (its share of the run). Fractions are **normalized** to sum to 1, so
  `[1, 1, 2]` means 25% / 25% / 50%.
- **One resolution per stage = staged with no mixing.** Several resolutions in a
  stage = they are **mixed** during that stage (sampled in proportion to image
  count, like the default). A single stage with every resolution reproduces the
  default mixed behavior.
- **Budget.** "100% of training" is your **`max_steps`** if you set it (that
  option takes precedence); otherwise the system derives it from
  `epochs × steps_per_epoch` (measured over the full resolution set). You do not
  have to set `max_steps`.
- **Granularity.** Stage changes happen at the **exact step** that crosses a
  boundary (iteration restarts mid-epoch), so the schedule is precise even with
  very few epochs / large datasets where a single-resolution epoch can span more
  steps than a whole stage. Drive saves/evals by steps (`save_every_n_steps` /
  `eval_every_n_steps`) in this regime, since natural epoch boundaries may be rare.
- Every resolution used in a stage must also be in the dataset's `resolutions`
  (latents are still cached for all of them). The schedule also applies to AR
  buckets and frame buckets of the active resolution(s).

| Key | Description | Values | Default |
|-----|-------------|--------|---------|
| **`resolution_schedule.enabled`** | Turn the staged schedule on. | `true` / `false` | `false` (uniform mixing). |
| **`resolution_schedule.stage`** | Ordered list of stages, each `{ resolutions, fraction }`. | `[[resolution_schedule.stage]]` tables. | — |

In the **web UI dataset editor**, the **Resolution schedule** section provides a
visual editor: toggle it on, then add stages — each row picks the active
resolution(s) (from the dataset's Resolutions) and its fraction, with the
normalized percentage shown next to it. An "Edit as JSON" escape hatch is there
for advanced cases.

### Captions

- **Per-image `.txt` files:** One caption **per line**. Empty lines are skipped. If the file is empty or missing, the image uses **`directory_caption`** (if set on that `[[directory]]`) as the full caption, or an empty caption if not set.
- **`captions.json`:** If present in a directory, it is used instead of `.txt` files. Format: `{ "image1.png": ["caption1", "caption2"], ... }` (list of captions per image for multi-caption).
- **`directory_caption`:** One option for both roles: when there is no per-image caption, it is used as the full caption; when there is a caption, it is prepended as a prefix (e.g. `"style: "`).

## Dataset augmentation

Optional **image diversity** settings (colour jitter, flip, mild geometry, etc.) are configured per directory (and optional global defaults under `[dataset.augmentation]`). In the web UI, use the dataset editor **Augmentation** tab: enable augmentation once for the dataset, then optionally customize individual `[[directory]]` rows (strategy overrides). Presets and strategy names come from the same catalog as training (`GET /api/v1/augmentations`). See [Dataset augmentation](dataset-augmentation.md) for presets, `seed_mode`, and flip enumeration. Augmentation applies before latent cache; use `deterministic_per_image` for reproducible caches. **Not supported:** video folders with augmentation enabled in this release.

## Cache and CLI flags

Before training, the framework runs a **cache** step: it encodes images to latents and runs text encoders, then stores results on disk so training can iterate quickly.

| Flag | Purpose | When to use |
|------|---------|--------------|
| **`--cache_only`** | Run the cache step and then exit (no training). | Pre-fill cache; requires a dataset config. Does nothing with synthetic-only. |
| **`--regenerate_cache`** | Ignore existing cache and recompute. | After changing captions, images, or model. |
| **`--regenerate_text_cache`** | Rebuild metadata and text embeddings only; reuse latents when possible. | After changing captions or tag-dropout rules that affect TE. |
| **`--trust_cache`** | Use existing cache metadata and iteration order without recomputing. | Faster startup when data and model are unchanged. |

Cache uses **v2 only** (mmap bf16 stacks). It is stored under **`cache_root`** / `<dataset_id>` / `<directory_id>` / `<model_name>/` ( **`cache_root`** is set in the training TOML — see [Training loop](training-loop-and-eval.md); default `cache/` in the install directory). Tag dropout runs at **training time**; captions in metadata stay raw. Training **`train_seed`** is also set in the main training TOML.

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
python -m rengu_flow.main --dump_dataset tests/fixtures/smoke/dataset_cc0.toml
```

Use this to verify directory paths, `.txt` captions, and `captions.json` before a long cache or training run. The versioned smoke dataset TOML for tests and GPU smokes is `tests/fixtures/smoke/dataset_cc0.toml` (12 CC0 images under `tests/fixtures/smoke_cc0/`).
