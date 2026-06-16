# Dataset and cache (contract and code locations)

Technical contract for dataset config, data loading, and cache in Rengu Flow. Phase 2 implements directory-based datasets, latent and text-embedding cache, and integration with the training loop.

**Implementation tracking:** Directory datasets, `DatasetManager.cache()`, latent and text-embedding cache, SDXL and Cosmos Predict2 model hooks, `--dump_dataset`, the CC0 smoke fixture, and **dataset augmentation (MVP)** are implemented. CPU/RAM tuning options and smoke A/B: [performance-cpu-ram](performance-cpu-ram.md).

## Dataset augmentation (MVP)

**Named strategies**, presets, merge rules, cache/seed modes, and UI: [Dataset augmentation (developer)](dataset-augmentation.md). Augmentation runs in the RGB path before VAE encode (`PreprocessMediaFile` via `DatasetManager.cache()`). Latent cache fingerprint includes the **fully merged** augmentation config (`rengu_flow/data/augmentation/config.py`). Images only in this release (`frame_buckets` must be `[1]` when augmentation is enabled).

User-facing summary: [Dataset augmentation (user)](../user/dataset-augmentation.md).

## Dataset TOML schema

The main config references a dataset via the `dataset` key (path to a TOML file). That TOML must contain:

- **`directory`** — List of directory configs (from TOML `[[directory]]`). Each entry: **`path`** and **`num_repeats`** (required); optional: `directory_caption`, `mask_path`, `control_path`, `default_mask_file`, `resolutions`, `frame_buckets`, `enable_ar_bucket`, `ar_buckets`, `size_buckets`, `subsample_ratio`, `max_images`, `subsample_shuffle`. See [user dataset-config](../user/dataset-config.md) for what each option does and allowed values.
- **Global options** in the same TOML: `resolutions`, `frame_buckets`, `enable_ar_bucket`, `min_ar`, `max_ar`, `num_ar_buckets`, `ar_buckets`, `size_buckets`, `shuffle_metadata`, `online_captions`, `subsample_ratio`, `max_images`, `subsample_shuffle`. Full descriptions and values are in the user doc.
- **Per-`[[directory]]` optional keys** include the same caption/bucket overrides as the UI directory editor, plus the per-epoch limiters `subsample_ratio` / `max_images` (see below). Root `subsample_ratio` is a separate static trim applied in `Dataset.post_init` on the combined iteration order.

### Per-epoch image limiting (`subsample_ratio` / `max_images`, rotating)

`subsample_ratio` (fraction) and `max_images` (absolute count) are two **mutually exclusive** per-(folder, size bucket) limiters on how many rows a folder serves per epoch. Both go through the same rotation machinery and are governed by `subsample_shuffle` (renamed from the retired `static_sampling`, inverted polarity). Neither is baked into the cached iteration order — the full pool stays cached so the served window can change per epoch (the latents cache was always full regardless, so this adds no caching cost):

- `SizeBucketDataset` keeps the full `iteration_order` (pool length `P`). The per-epoch row count is `cap = effective_sample_cap(P, max_images, subsample_ratio)` — `max_images`, else `max(1, int(P*subsample_ratio))` when `ratio < 1`, else `None` (uncapped). Per-epoch served length `M = cap or P`; `__len__` is `M * num_repeats`, constant across epochs (so `steps_per_epoch` is stable).
- `__getitem__`/`get_items_batch` map an index to a pool row via the pure helper `rotation_window_index(pos, epoch, pool_len, cap, static)`. Non-static (default) advances the window start by `cap` each epoch (wrapping) so the whole pool is covered every `ceil(P/cap)` epochs; `static` keeps offset 0; `cap > P` repeats the pool up to `cap` (repeat-to-N).
- Epoch is propagated by `set_epoch` (on `Dataset` → `ConcatenatedBatchedDataset` → `SizeBucketDataset`). `PipelineDataLoader._refresh_dataset_epoch` calls it on dataloader (re)creation and at each epoch rollover. `Dataset.rotation_active` flags whether any directory rotates; the loader re-creates the dataloader at epoch boundaries when `rotation_active` and `dataloader_num_workers > 0` so forked workers pick up the new window (with the default `num_workers = 0` the in-process dataset is updated directly). Rotation depends only on the (synced) epoch, so it is deterministic across data-parallel ranks and survives checkpoint resume.

Validation for real data: `rengu_flow.data.dataset_config.validate_dataset_config_for_real_data(dataset_config)` ensures `directory` is present and non-empty, each entry has `path` and `num_repeats` &gt; 0, any `max_images` (per-directory or global) is an integer &gt; 0, and that `subsample_ratio < 1` and `max_images` are not both set in the same scope (`_validate_sampler_exclusivity`).

## Caption sources

- **`.txt` files:** One caption **per line** (rengu-flow behaviour). Empty lines skipped; if no lines, treat as one empty caption. Stored as a list of strings for multi-caption.
- **`captions.json`:** If present in the directory, used instead of `.txt`. Format: `{ "filename": ["cap1", "cap2"], ... }`. Lists are supported (multi-caption).
- **`directory_caption`** (single option, no separate prefix): When there is no `.txt` and no `captions.json` entry for an image, this string is used as the full caption (or empty if unset). When a per-image caption exists, this string is **prepended** as a prefix. So one key covers both “fallback caption” and “prefix”.

Caption lists are flattened for text-embedding cache (one embedding per (image, caption)); iteration order picks one caption per step.

## Caption variants — how multi-caption works inside

Multiple captions per image are the cached-augmentation mechanism (tag-dropout variants that
let `cache_text_embeddings = true` keep working). They come from two equivalent sources: extra
`.txt` lines (offline `scripts/generate_caption_variants.py`), or — the recommended path —
in-pipeline baking via the dataset-level `cached_caption_variants` (see point 5). Either way the
caption column ends up with K entries per image; everything below is identical. End to end:

1. **Parsing** — `_read_captions_from_txt_per_line` stores each image's captions as a list;
   every variant gets its own cached text embedding keyed by `(image_spec, caption_number)`
   (`TextEmbeddingDataset.get_text_embeddings`). The mapping is unambiguous: one
   `(image, caption_number)` → one text → one embedding.
2. **Iteration order** (`SizeBucketDataset`, "Building iteration order") — with equal counts,
   each image's captions are first shuffled with a per-image seed, then grouped by slot:
   segment 0 holds every image once with its slot-0 variant, segment 1 with slot-1, etc. One
   full pass over the order therefore serves **every image with all K variants exactly once
   each** (verified on real cache artifacts: no image is ever stuck with one embedding, and
   no `(image, caption_number)` maps to two texts). With unequal counts the order is a flat
   shuffle instead.
3. **Epoch accounting** — variants multiply the iteration order (each is its own example),
   but they are regularization samples of the same images, not new data.
   `Dataset.caption_variants` (uniform count across all buckets, else 1) is divided out of
   `steps_per_epoch` in `main._run_training`, so an **"epoch" still means one pass over the
   images**: save/eval/preview cadence and the `epochs × steps_per_epoch` budget stay stable
   for any K, and each accounting epoch serves the next per-image variant (rotation across
   epochs). `epochs = K` consumes every variant exactly once.
4. **Why pre-baked variants instead of live tag dropout** — live per-tag dropout produces a
   unique caption per draw, forcing the text encoder to stay resident (~1.2 GB, lowering the
   usable `activation_memory_budget`) and to run every step (~22 ms — launch-overhead-bound,
   independent of caption length; see EXPERIMENTS_GRAVEYARD "Trim Cosmos text padding").
   K pre-baked variants sampled from the same dropout distribution are statistically
   equivalent up to finite-K noise (with `epochs` passes, each tag is seen present/absent in
   a Binomial(K, p) share of them), encode once at cache time (~22 ms per caption, latents
   reused), and free the TE from the GPU entirely. `uncond_fraction` composes with this
   (the cached unconditional embedding is swapped in per draw).

5. **Where the K variants come from** — `dataset.expand_caption_variants` samples
   `apply_tag_dropout` (+ optional tag shuffle) K times per base caption with a per-`(image,
   base, variant)` md5 seed. When `cache_text_embeddings` is on and any of `cached_caption_variants
   > 1`, `cached_caption_shuffle`, or `tag_dropout.enabled` holds, `SizeBucketDataset.__init__` runs it as an in-memory map over
   the loaded metadata's caption column — **before** both the TE-cache flatten and the
   iteration-order build, so both read the same expanded column. It is deterministic, so the
   text-embedding cache (keyed by a content hash of the caption column) is reused until K / the
   probability / rules / shuffle / captions change. The iteration-order cache, which only trusts
   an existence check, gets its own caption-content fingerprint sidecar
   (`iteration_order.caption_fp`) so it rebuilds on the same triggers. Latents never depend on
   captions, so they are untouched. The offline generator (`generate_caption_variants.py`) does
   the same sampling but writes K `.txt` lines instead.

## Cache

- **`cache_root`** (training TOML) — All v2 caches live under `cache_root/<dataset_hash>/<directory_hash>/<model_name>/`. Default: `cache/` at install root (gitignored). Legacy v1 / co-located `path/cache/` is not supported. If `cache_root` remains in a dataset TOML, it is deprecated: a warning is logged and the value is used only when the training config omits `cache_root`.
- **`tag_dropout_*`** / **`uncond_fraction`** — See `rengu_flow/data/tag_dropout.py`. `uncond_fraction` works in every mode (cached runs swap in the cached uncond embedding). Tag dropout rewrites the caption string: with `cache_text_embeddings = false` it is applied **live** per sample in `SizeBucketDataset._sample_from_entry`; with the cache on it is **pre-baked** into the embedding cache (`cached_caption_variants = 1` bakes a single fixed variant — diffusion-pipe's default — and `>= 2` bakes rotating variants; see "Caption variants"). Either way the dropout reaches the model, so cache-on + dropout is not rejected.
- **`rengu_flow.utils.cache_v2.CacheV2`** — Only supported format: `manifest.json` (fingerprint, tensor specs), `tensors/{key}.bin` (stacked payloads; float tensors stored as **bf16**), `meta.db` (per-index JSON for non-tensor fields and optional null tensors). Opened via **`rengu_flow.utils.cache_factory.open_disk_cache`**, which rejects legacy v1 caches (a `metadata.db` file) with an actionable error.
- **`rengu_flow.data.cache_utils._map_and_cache`** — Maps over a HuggingFace `datasets.Dataset` with a `map_fn(example, rank)`, persists results via `open_disk_cache`. Fingerprint from dataset `_fingerprint` + optional `new_fingerprint_args` + `cache_format=…`. If `map_fn is None`, loads existing cache only (used after worker cache run). Config: `cache_num_proc` (pool size, default `min(8, cpu_count)`), `cache_keep_in_memory` (default `false` for resume slices).
- **`PipelineDataLoader` prefetch** — `dataloader_prefetch=true` uses a background thread when `dataloader_num_workers=0`; `prepare_inputs` stays on the main process. See `dataloader_num_workers`, `dataloader_pin_memory`, `dataloader_prefetch_factor` in the main TOML.
- **`rengu_flow.data.manager.DatasetManager`** — Holds model (VAE, text encoders), registers datasets, and runs **`cache()`**: spawns a worker process that runs `_cache_fn` (metadata → latents → text embeddings); main processes handle GPU work via a queue. After cache, VAE/TE can be unloaded; then all ranks load datasets from cache (`cache_metadata(trust_cache=True)`, `cache_latents(None, trust_cache=True)`, `cache_text_embeddings(None, i)`).

## Dataset classes

| Class | Location | Role |
|-------|----------|------|
| **Dataset** | `rengu_flow.data.dataset` | Top-level: builds one `DirectoryDataset` per `directory` entry; `post_init(dp_rank, dp_world_size, per_device_batch_size, gradient_accumulation_steps, per_device_batch_size_image)`; `__getitem__` returns a collated batch for the data-parallel rank. |
| **DirectoryDataset** | `rengu_flow.data.dataset` | One directory: metadata (ungrouped → grouped by AR/size bucket), `cache_metadata`, `cache_latents`, `cache_text_embeddings`. Uses `_get_ungrouped_metadata`, `_metadata_map_fn` (read media, captions from .txt per line or captions.json; `directory_caption` as fallback and prefix). Optional `mask_path` / `control_path`: separate folders; files paired to images in `path` by stem. When `control_path` is set, metadata includes `control_file` and cache runs in edit mode (`DatasetManager._cache_fn`, `is_edit`). |
| **ARBucketDataset** / **SizeBucketDataset** | `rengu_flow.data.dataset` | Per (AR, frames) or per size bucket; create `SizeBucketDataset` instances, cache latents and text embeddings, build iteration order (multi-caption). Each size bucket shuffles metadata with **`seed_from_hash(size_bucket)`** so multi-resolution runs mix order per bucket (diffusion-pipe). |
| **`seed_from_hash`** | `rengu_flow.data.cache_utils` | Deterministic int seed from path or bucket key (MD5). Used for metadata shuffle and per-bucket shuffle. |
| **ConcatenatedBatchedDataset** | `rengu_flow.data.dataset` | Concatenates multiple `SizeBucketDataset` (same size bucket); `post_init` for batch sizes and DP rank; returns batches. |
| **PipelineDataLoader** | `rengu_flow.data.loader` | Wraps dataset; calls `model.prepare_inputs(batch)`; splits into micro-batches for gradient accumulation; `sync_epoch`; **`state_dict`** / **`load_state_dict`** for resume. |

Contract for the object passed to the orchestrator as the training dataset:

- `cache_metadata(regenerate_cache=..., trust_cache=...)`
- `cache_latents(map_fn, ...)`, `cache_text_embeddings(map_fn, i, ...)`
- `post_init(dp_rank, dp_world_size, per_device_batch_size, gradient_accumulation_steps, per_device_batch_size_image)`
- `__len__`, `__getitem__(idx)` → batch dict (latents, mask, caption, text-embedding keys per model)
- Attribute **`dataset_config`** (for error messages and Saver).

## Where the code lives

| Area | Location |
|------|----------|
| Config loading | `rengu_flow.config.loader`: `load_config`, `load_dataset_config`, `load_eval_dataset_config` |
| Dataset config validation | `rengu_flow.data.dataset_config`: `validate_dataset_config_for_real_data`, `DatasetConfigError` |
| Cache (disk) | `rengu_flow.utils.cache_v2`: `CacheV2` (only supported format); `rengu_flow.utils.cache_factory`: `open_disk_cache`, `detect_cache_format` (rejects legacy v1) |
| Map and cache helpers | `rengu_flow.data.cache_utils`: `_map_and_cache`, `bucket_suffix`, `dedup_and_sort` |
| Dataset hierarchy | `rengu_flow.data.dataset`: `Dataset`, `DirectoryDataset`, `SizeBucketDataset`, `ConcatenatedBatchedDataset`, `ARBucketDataset`, `TextEmbeddingDataset`, `_cache_text_embeddings`, caption helpers |
| Cache orchestration | `rengu_flow.data.manager`: `_cache_fn`, `DatasetManager` |
| Data loader | `rengu_flow.data.loader`: `PipelineDataLoader`, `split_batch` |
| Synthetic dataset | `rengu_flow.data.synthetic`: `SyntheticSDXLDataset` |
| Main flow | `rengu_flow.main`: loads dataset config; if real data: `Dataset`, `DatasetManager`, `cache()`, `--cache_only` exit; after DeepSpeed init: `train_data.post_init(...)` or synthetic, then `PipelineDataLoader`; CLI: `--cache_only`, `--regenerate_cache`, `--trust_cache`, `--dump_dataset` (see `rengu_flow.data.dump_dataset`) |
| UI dataset library | `rengu_flow_ui.datasets_store`, `library_db`, `dataset_scan`, `dataset_schema` — TOML in SQLite (`jobs.db`), compose `[[directory]]`, folder scan preview; see **`docs/developer/web-ui.md`** |

## Tests

See the full table in [Testing — Dataset and data loading tests](testing.md#dataset-and-data-loading-tests). Highlights:

- **`tests/test_dataset_config.py`** — config load and `validate_dataset_config_for_real_data`.
- **`tests/test_dataset_captions.py`** — `.txt` (one caption per line), `captions.json` (list or string), multi-caption `iteration_order`, `online_captions`.
- **`tests/test_dump_dataset.py`**, **`tests/test_smoke_cc0_dataset.py`** — `dump_dataset` and the versioned CC0 fixture.
- **`tests/test_sdxl_cache_hooks.py`**, **`tests/test_sdxl_cached_prepare_inputs.py`** — SDXL cache hooks and cached training path (mocked).
- **`tests/test_cache_v2.py`** — v2 round-trip, resume, fingerprint, factory detect.

## Model hooks for cache

Required by `DatasetManager.cache()` / `_cache_fn`.

### Cosmos Predict2 (`rengu_flow.model.cosmos_predict2.CosmosPredict2Pipeline`)

| Method | Status |
|--------|--------|
| `get_preprocess_media_file_fn` | Implemented — `PreprocessMediaFile` |
| `get_call_vae_fn` | Implemented — Wan VAE latents |
| `get_call_text_encoder_fn` | Implemented — Qwen3 + T5 token ids |
| `get_text_encoders` | Implemented when `cache_text_embeddings` (default true) |
| `model_specific_dataset_config_validation` | Requires `1` in `frame_buckets` for image training |

### SDXL (`rengu_flow.model.sdxl.SDXLPipeline`)

| Method | Status |
|--------|--------|
| `get_call_vae_fn` | Implemented |
| `get_preprocess_media_file_fn` | Implemented — `PreprocessMediaFile` (images only, 16px round) |
| `get_call_text_encoder_fn` | Implemented — per-encoder dict keys (see below) |
| `get_text_encoders` | `[text_encoder, text_encoder_2]` when `cache_text_embeddings` (default true) |

**SDXL cached embedding keys:** encoder 1 → `prompt_embeds`; encoder 2 → `prompt_embeds_2`, `pooled_prompt_embeds`. `prepare_inputs` concatenates prompt embeds for the UNet and passes cached tensors to `InitialLayer` when `cache_text_embeddings` is true.

**Smoke dataset:** `tests/fixtures/smoke_cc0/` (12 CC0 GB82 JPEGs + captions). Regenerate with `scripts/vendor_smoke_cc0.sh`. Dataset TOML: `tests/fixtures/smoke/dataset_cc0.toml`.
