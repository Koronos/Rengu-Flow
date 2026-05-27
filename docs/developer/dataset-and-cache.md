# Dataset and cache (contract and code locations)

Technical contract for dataset config, data loading, and cache in Renga Flow. Phase 2 implements directory-based datasets, latent and text-embedding cache, and integration with the training loop.

**Implementation tracking:** Directory datasets, `DatasetManager.cache()`, latent and text-embedding cache, SDXL and Cosmos Predict2 model hooks, `--dump_dataset`, the CC0 smoke fixture, and **dataset augmentation (MVP)** are implemented. CPU/RAM tuning options and smoke A/B: [performance-cpu-ram](performance-cpu-ram.md).

## Dataset augmentation (MVP)

**Named strategies**, presets, merge rules, cache/seed modes, and UI: [Dataset augmentation (developer)](dataset-augmentation.md). Augmentation runs in the RGB path before VAE encode (`PreprocessMediaFile` via `DatasetManager.cache()`). Latent cache fingerprint includes the **fully merged** augmentation config (`renga_flow/data/augmentation/config.py`). Images only in this release (`frame_buckets` must be `[1]` when augmentation is enabled).

User-facing summary: [Dataset augmentation (user)](../user/dataset-augmentation.md).

## Dataset TOML schema

The main config references a dataset via the `dataset` key (path to a TOML file). That TOML must contain:

- **`directory`** — List of directory configs (from TOML `[[directory]]`). Each entry: **`path`** and **`num_repeats`** (required); optional: `directory_caption`, `mask_path`, `control_path`, `default_mask_file`, `resolutions`, `frame_buckets`, `enable_ar_bucket`, `ar_buckets`, `size_buckets`, `cache_shuffle_num`, `cache_shuffle_delimiter`, `shuffle_tags`. See [user dataset-config](../user/dataset-config.md) for what each option does and allowed values.
- **Global options** in the same TOML: `resolutions`, `frame_buckets`, `enable_ar_bucket`, `min_ar`, `max_ar`, `num_ar_buckets`, `ar_buckets`, `size_buckets`, `shuffle_tags`, `cache_shuffle_num`, `cache_shuffle_delimiter`, `shuffle_metadata`, `online_captions`, `subsample_ratio`. Full descriptions and values are in the user doc.

Validation for real data: `renga_flow.data.dataset_config.validate_dataset_config_for_real_data(dataset_config)` ensures `directory` is present and non-empty and each entry has `path` and `num_repeats` &gt; 0.

## Caption sources

- **`.txt` files:** One caption **per line** (renga-flow behaviour). Empty lines skipped; if no lines, treat as one empty caption. Stored as a list of strings for multi-caption.
- **`captions.json`:** If present in the directory, used instead of `.txt`. Format: `{ "filename": ["cap1", "cap2"], ... }`. Lists are supported (multi-caption).
- **`directory_caption`** (single option, no separate prefix): When there is no `.txt` and no `captions.json` entry for an image, this string is used as the full caption (or empty if unset). When a per-image caption exists, this string is **prepended** as a prefix. So one key covers both “fallback caption” and “prefix”.

Caption lists are flattened for text-embedding cache (one embedding per (image, caption)); iteration order picks one caption per step.

## Cache

- **`cache_format`** (main TOML, default **`v2`**) — On-disk layout for latent and text-embedding caches. Set **`v1`** to keep legacy pickle shards. There is **no automatic migration** from v1: change format or fingerprint → regenerate cache (`--regenerate_cache` or delete `latents/` / `text_embeddings_*` under the bucket).
- **`renga_flow.utils.cache_v2.CacheV2`** — Default format: `manifest.json` (fingerprint, tensor specs), `tensors/{key}.bin` (stacked payloads; float tensors stored as **bf16**), `meta.db` (per-index JSON for non-tensor fields and optional null tensors). Opened via **`renga_flow.utils.cache_factory.open_disk_cache`** (auto-detects existing v1 vs v2).
- **`renga_flow.utils.cache.Cache`** — Legacy v1: SQLite + pickle shards (`metadata.db`, `shard_*.bin`).
- **`renga_flow.data.cache_utils._map_and_cache`** — Maps over a HuggingFace `datasets.Dataset` with a `map_fn(example, rank)`, persists results via `open_disk_cache`. Fingerprint from dataset `_fingerprint` + optional `new_fingerprint_args` + `cache_format=…`. If `map_fn is None`, loads existing cache only (used after worker cache run). Config: `cache_num_proc` (pool size, default `min(8, cpu_count)`), `cache_keep_in_memory` (default `false` for resume slices).
- **`PipelineDataLoader` prefetch** — `dataloader_prefetch=true` uses a background thread when `dataloader_num_workers=0`; `prepare_inputs` stays on the main process. See `dataloader_num_workers`, `dataloader_pin_memory`, `dataloader_prefetch_factor` in the main TOML.
- **`renga_flow.data.manager.DatasetManager`** — Holds model (VAE, text encoders), registers datasets, and runs **`cache()`**: spawns a worker process that runs `_cache_fn` (metadata → latents → text embeddings); main processes handle GPU work via a queue. After cache, VAE/TE can be unloaded; then all ranks load datasets from cache (`cache_metadata(trust_cache=True)`, `cache_latents(None, trust_cache=True)`, `cache_text_embeddings(None, i)`).

## Dataset classes

| Class | Location | Role |
|-------|----------|------|
| **Dataset** | `renga_flow.data.dataset` | Top-level: builds one `DirectoryDataset` per `directory` entry; `post_init(dp_rank, dp_world_size, per_device_batch_size, gradient_accumulation_steps, per_device_batch_size_image)`; `__getitem__` returns a collated batch for the data-parallel rank. |
| **DirectoryDataset** | `renga_flow.data.dataset` | One directory: metadata (ungrouped → grouped by AR/size bucket), `cache_metadata`, `cache_latents`, `cache_text_embeddings`. Uses `_get_ungrouped_metadata`, `_metadata_map_fn` (read media, captions from .txt per line or captions.json; `directory_caption` as fallback and prefix). |
| **ARBucketDataset** / **SizeBucketDataset** | `renga_flow.data.dataset` | Per (AR, frames) or per size bucket; create `SizeBucketDataset` instances, cache latents and text embeddings, build iteration order (multi-caption). Each size bucket shuffles metadata with **`seed_from_hash(size_bucket)`** so multi-resolution runs mix order per bucket (diffusion-pipe). |
| **`seed_from_hash`** | `renga_flow.data.cache_utils` | Deterministic int seed from path or bucket key (MD5). Used for metadata shuffle and per-bucket shuffle. |
| **ConcatenatedBatchedDataset** | `renga_flow.data.dataset` | Concatenates multiple `SizeBucketDataset` (same size bucket); `post_init` for batch sizes and DP rank; returns batches. |
| **PipelineDataLoader** | `renga_flow.data.loader` | Wraps dataset; calls `model.prepare_inputs(batch)`; splits into micro-batches for gradient accumulation; `sync_epoch`; **`state_dict`** / **`load_state_dict`** for resume. |

Contract for the object passed to the orchestrator as the training dataset:

- `cache_metadata(regenerate_cache=..., trust_cache=...)`
- `cache_latents(map_fn, ...)`, `cache_text_embeddings(map_fn, i, ...)`
- `post_init(dp_rank, dp_world_size, per_device_batch_size, gradient_accumulation_steps, per_device_batch_size_image)`
- `__len__`, `__getitem__(idx)` → batch dict (latents, mask, caption, text-embedding keys per model)
- Attribute **`dataset_config`** (for error messages and Saver).

## Where the code lives

| Area | Location |
|------|----------|
| Config loading | `renga_flow.config.loader`: `load_config`, `load_dataset_config`, `load_eval_dataset_config` |
| Dataset config validation | `renga_flow.data.dataset_config`: `validate_dataset_config_for_real_data`, `DatasetConfigError` |
| Cache (disk) | `renga_flow.utils.cache`: `Cache` (v1); `renga_flow.utils.cache_v2`: `CacheV2`; `renga_flow.utils.cache_factory`: `open_disk_cache`, `detect_cache_format` |
| Map and cache helpers | `renga_flow.data.cache_utils`: `_map_and_cache`, `bucket_suffix`, `dedup_and_sort` |
| Dataset hierarchy | `renga_flow.data.dataset`: `Dataset`, `DirectoryDataset`, `SizeBucketDataset`, `ConcatenatedBatchedDataset`, `ARBucketDataset`, `TextEmbeddingDataset`, `_cache_text_embeddings`, caption helpers |
| Cache orchestration | `renga_flow.data.manager`: `_cache_fn`, `DatasetManager` |
| Data loader | `renga_flow.data.loader`: `PipelineDataLoader`, `split_batch` |
| Synthetic dataset | `renga_flow.data.synthetic`: `SyntheticSDXLDataset` |
| Main flow | `renga_flow.main`: loads dataset config; if real data: `Dataset`, `DatasetManager`, `cache()`, `--cache_only` exit; after DeepSpeed init: `train_data.post_init(...)` or synthetic, then `PipelineDataLoader`; CLI: `--cache_only`, `--regenerate_cache`, `--trust_cache`, `--dump_dataset` (see `renga_flow.data.dump_dataset`) |
| UI dataset library | `renga_flow_ui.datasets_store`, `library_db`, `dataset_scan`, `dataset_schema` — TOML in SQLite (`jobs.db`), compose `[[directory]]`, folder scan preview; see **`docs/developer/web-ui.md`** |

## Tests

See the full table in [Testing — Dataset and data loading tests](testing.md#dataset-and-data-loading-tests). Highlights:

- **`tests/test_dataset_config.py`** — config load and `validate_dataset_config_for_real_data`.
- **`tests/test_dataset_captions.py`** — `.txt` (one caption per line), `captions.json` (list or string), multi-caption `iteration_order`, `online_captions`.
- **`tests/test_dump_dataset.py`**, **`tests/test_smoke_cc0_dataset.py`** — `dump_dataset` and the versioned CC0 fixture.
- **`tests/test_sdxl_cache_hooks.py`**, **`tests/test_sdxl_cached_prepare_inputs.py`** — SDXL cache hooks and cached training path (mocked).
- **`tests/test_cache.py`** — disk `Cache` fingerprint and shards (v1).
- **`tests/test_cache_v2.py`** — v2 round-trip, resume, fingerprint, factory detect.

## Model hooks for cache

Required by `DatasetManager.cache()` / `_cache_fn`.

### Cosmos Predict2 (`renga_flow.model.cosmos_predict2.CosmosPredict2Pipeline`)

| Method | Status |
|--------|--------|
| `get_preprocess_media_file_fn` | Implemented — `PreprocessMediaFile` |
| `get_call_vae_fn` | Implemented — Wan VAE latents |
| `get_call_text_encoder_fn` | Implemented — Qwen3 + T5 token ids |
| `get_text_encoders` | Implemented when `cache_text_embeddings` (default true) |
| `model_specific_dataset_config_validation` | Requires `1` in `frame_buckets` for image training |

### SDXL (`renga_flow.model.sdxl.SDXLPipeline`)

| Method | Status |
|--------|--------|
| `get_call_vae_fn` | Implemented |
| `get_preprocess_media_file_fn` | Implemented — `PreprocessMediaFile` (images only, 16px round) |
| `get_call_text_encoder_fn` | Implemented — per-encoder dict keys (see below) |
| `get_text_encoders` | `[text_encoder, text_encoder_2]` when `cache_text_embeddings` (default true) |

**SDXL cached embedding keys:** encoder 1 → `prompt_embeds`; encoder 2 → `prompt_embeds_2`, `pooled_prompt_embeds`. `prepare_inputs` concatenates prompt embeds for the UNet and passes cached tensors to `InitialLayer` when `cache_text_embeddings` is true.

**Smoke dataset:** `tests/fixtures/smoke_cc0/` (12 CC0 GB82 JPEGs + captions). Regenerate with `scripts/vendor_smoke_cc0.sh`. Dataset TOML: `tests/fixtures/smoke/dataset_cc0.toml`.
