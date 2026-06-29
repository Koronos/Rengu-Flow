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
| **`subsample_ratio`** | **Fractional** per-epoch cap on this folder's **base images** (e.g. `0.25` = a quarter). Rotates each epoch by default. **Mutually exclusive with `max_images`.** | Float in (0, 1]. | `1` (all images). |
| **`max_images`** | **Absolute** per-epoch cap on how many **base images** this folder contributes, shared across all of its resolution/AR buckets. Rotates each epoch by default. **Mutually exclusive with `subsample_ratio`.** | Integer &gt; 0. | Not set (no cap); inherits the dataset default if one is set. |
| **`subsample_shuffle`** | Rotate the sampled window each epoch for whichever limiter is active; `false` keeps the **same** images every epoch. | `true` or `false`. | `true` (rotate). |

#### Per-epoch image limiting: `subsample_ratio` vs `max_images`

There are two ways to use only part of a folder each epoch — pick **one** (they are mutually exclusive in the same scope; setting both raises a config error):

- **`subsample_ratio = f`** — a **fraction** (e.g. `0.25` = a quarter of the rows). Good for quick debug runs.
- **`max_images = N`** — an **absolute count**. Good for **balancing several folders of very different sizes** (e.g. ten style folders with 10–100 images each) so no folder dominates.

Both share the same per-epoch behavior, governed by `subsample_shuffle`:

- **Rotating (default, `subsample_shuffle = true`).** Each epoch serves a *different* window, advancing through the whole folder and wrapping around. Over `ceil(total / limit)` epochs every image is seen — you get the limit **and** eventually use the entire dataset. Recommended.
- **Static (`subsample_shuffle = false`).** The same first images are used every epoch; the rest of the folder is never seen. Use only when you deliberately want a fixed subset.
- **Fewer images than `max_images`.** The folder repeats its images up to `N` (repeat-to-N), so its per-epoch count matches folders that do have `N`. To over-sample a small folder beyond `N`, use `num_repeats`.
- **Counts base images, shared across the whole folder.** The cap is on the folder's *original* images, applied **once** across the folder — not per resolution/AR bucket. Resolution buckets and augmentation branches then expand each selected image normally. So `max_images = 50` on a 1200-image folder uses **50 base images per epoch** (× your augmentation × the active resolution), no matter how many AR/resolution buckets the folder spans — that is what balances a big folder against small ones.
- **Interaction.** `num_repeats` still multiplies the per-epoch count (`limit × num_repeats` rows). The limit keeps each epoch's length constant, so `steps_per_epoch` is unchanged — only *which* images appear rotates. (The root-level `subsample_ratio` in [Global options](#global-options-dataset-toml-root) is a separate, static trim of the combined schedule and is unaffected by `subsample_shuffle`.)
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
subsample_shuffle = false    # this folder: fixed subset instead of rotating
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
| **Sidecar `.txt`** | One **line** = one caption variant for that image. | Multiple lines → one cached text embedding per variant, rotated across epochs (an epoch still means one pass over the images — variants do **not** lengthen it). Empty file → one empty caption. |
| **`captions.json`** in the image folder | JSON object: filename → caption or **list of captions**. | If present, **overrides** sidecar `.txt` for that directory. Example: `{ "photo.jpg": ["tag1, style", "alt description"] }`. A single string value is treated as one caption. |
| **`directory_caption`** | One string in TOML. | Used when there is no `.txt` and no JSON entry; when a per-image caption exists, it is **prepended** as a prefix (see table above). |

Inspect resolved captions with [`--dump_dataset`](#inspecting-a-dataset---dump_dataset) before caching.

### Caption variants: cached tag dropout

Live `tag_dropout_enabled = true` requires `cache_text_embeddings = false`, which keeps the
text encoder on the GPU for the whole run (~22 ms/step plus ~1.2 GB VRAM that lowers the
usable `activation_memory_budget`). To get the **same dropout distribution with the text
encoder fully off the GPU**, keep `cache_text_embeddings = true` and let the caching step
bake K dropout/shuffle variants per caption into the embedding cache:

```toml
cache_text_embeddings = true       # model config (it already defaults on for Cosmos/SDXL)

# dataset TOML root (or per [[directory]]):
tag_dropout_enabled = true          # defines the dropout distribution...
tag_dropout_probability = 0.3
cached_caption_variants = 15        # ...baked this many times per caption at cache time
cached_caption_shuffle = false      # optional: also shuffle tag order per variant
```

When the text-embedding cache is built, each caption is expanded into `cached_caption_variants`
seeded dropout/shuffle samples; every sample gets its own cached embedding and the trainer
rotates them, so an epoch is still one pass over the images (variants do **not** lengthen it).
`cached_caption_variants` equal to your `epochs` is statistically sufficient (each variant is
used about once across the run). The generation is deterministic and idempotent: the cache is
keyed by caption content, so re-running reuses it until you change K, the probability/rules,
the shuffle flag, or the captions — then **only** the text-embedding cache rebuilds (latents
are untouched, a couple of minutes).

Notes:
- `cached_caption_variants` is dataset-level and must be uniform across folders (mixed counts
  fall back to 1); the per-folder `tag_dropout` rules still apply when baking each folder.
- `cached_caption_variants = 1` with dropout/shuffle bakes **one fixed augmented variant** for
  the whole dataset (this is how diffusion-pipe behaved by default — augmentation applied once at
  cache time). Use `>= 2` to get variants that rotate across epochs.
- A `.txt` with several lines (alternative descriptions) keeps all of them: each line is
  expanded K times (total `lines × K`).
- `uncond_fraction` composes with this (the cached unconditional embedding is swapped in per
  draw). Internals: `docs/developer/dataset-and-cache.md`, "Caption variants".

> The offline `scripts/generate_caption_variants.py` (pre-baking variants as extra `.txt`
> lines) still works and is equivalent, but the `cached_caption_variants` config above is the
> recommended path — no manual script run, no `.txt` rewrites.

**Undroppable tags (control lists).** A rule with `drop_probability = 0.0` pins tags so they
are *never* dropped — useful for quality/meta tags you want the model to associate with the
artifact rather than bake in (`watermark`, `signature`, `censored`, `jpeg_artifacts`, quality
tags, …). Keep these control lists in the **original Danbooru form (underscores)**, e.g.
`jpeg_artifacts`, one tag per line in a `tags_file`. Matching is underscore- and
case-insensitive, so the same list drops **both** forms (`jpeg_artifacts` and `jpeg artifacts`)
regardless of the tagger's [underscore setting](dataset-prep.md#tagging--rengu-prep-tag) — you
never need two copies.

```toml
tag_dropout_enabled = true
tag_dropout_probability = 0.3
tag_dropout_rules = [
  { tags_file = "undroppable_tags/quality.txt", drop_probability = 0.0 },
  { tags_file = "undroppable_tags/text_signature_watermark.txt", drop_probability = 0.0 },
]
```

`tags_file` is resolved relative to the dataset folder (or give an absolute path).

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
| **`shuffle_metadata`** | Shuffle image order when building metadata (deterministic seed from directory path). | `true` / `false` | `true` |
| **`online_captions`** | Read captions from `captions.json` at training time instead of only from cached metadata. | `true` / `false` | `false` |
| **`subsample_ratio`** | Fraction of the combined training schedule (e.g. `0.25` for quick debug runs). | Float in (0, 1]. | `1` (full dataset). |
| **`max_images`** | Default absolute base-image cap per folder per epoch (shared across the folder's buckets); rotates each epoch while `subsample_shuffle` is on. Per-folder keys override it. Mutually exclusive with a per-folder `subsample_ratio`. See [per-epoch limiting](#per-epoch-image-limiting-subsample_ratio-vs-max_images). | Integer &gt; 0. | Not set (no cap). |
| **`subsample_shuffle`** | Default for whether the active limiter (subsample ratio or max images) rotates per epoch (`true`) or keeps a fixed subset (`false`). Renamed from the retired `static_sampling` (inverted polarity). | `true` / `false` | `true` (rotate). |
| **`tag_dropout_enabled`** | Enable random tag dropout. Defines the drop distribution; how it is applied depends on the cache: live per sample when `cache_text_embeddings = false`, or pre-baked into the cache when `cache_text_embeddings = true` with `cached_caption_variants >= 2`. See [caption variants](#caption-variants-cached-tag-dropout). | `true` / `false` | `false` |
| **`tag_dropout_probability`** | Default drop probability for tags not in a rule. | Float in [0, 1]. | — |
| **`tag_dropout_mode`** | `per_tag` or `full`. | String | `per_tag` |
| **`tag_dropout_rules`** | List of `{ tags, drop_probability }` and/or `tags_file` (one tag per line). Use `drop_probability = 0.0` for undroppable control lists; keep tags in the original underscore form (matches both `long_hair` and `long hair`). | Tables / JSON in UI | — |
| **`cached_caption_variants`** | With `cache_text_embeddings = true`, bake this many tag-dropout/shuffle variants per caption into the embedding cache. `1` = identity if no dropout/shuffle, or one fixed baked variant with them (diffusion-pipe-style); `>= 2` rotates variants across epochs without lengthening them. Dataset-level and must be uniform across folders. See [caption variants](#caption-variants-cached-tag-dropout). | Integer ≥ 1. | `1` |
| **`cached_caption_shuffle`** | Also shuffle tag order in each baked cached-caption variant. | `true` / `false` | `false` |
| **`uncond_fraction`** | Fraction of samples with empty caption (CFG). | Float in [0, 1]. | `0` |
| **`tag_match_case_sensitive`** | Case-sensitive tag matching in rules. Underscores and spaces always match interchangeably (`long_hair` == `long hair`), independent of this flag. | `true` / `false` | `false` |

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
- **Budget — the schedule shrinks total steps.** Each resolution is only trained
  for the fraction of the run it is active, so it contributes that share of its
  steps. A resolution in every stage is trained the whole run; one in a single
  33% stage costs ~⅓ of its full-mixing steps. Example: 250 images/res at batch
  2/2/1 is 500 steps/epoch fully mixed; with stages {500,1000}/{700,1000}/{1000}
  at 40/40/20 it becomes 350 (1000 stays at 250, but 500 and 700 only cost
  0.4 × 125 = 50 each). `total_steps` is this reduced count × `epochs`; an
  "epoch" is just a `total_steps / epochs` slice for save/eval/preview cadence,
  not a full pass. Set **`max_steps`** to pin an absolute budget instead (it
  takes precedence and the stage fractions then split *it*).
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
