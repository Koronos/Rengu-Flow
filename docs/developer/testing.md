# Testing

How to run and extend the test suite for Rengu Flow.

## Running tests

**Requirements:** Install with the dev extra so pytest is available:

```bash
pip install -e ".[dev]"
# or with uv:
uv sync
```

**Run all tests:**

```bash
pytest
# or from repo root:
uv run pytest
python -m pytest tests/
```

**Useful options:**

| Option | Description |
|--------|-------------|
| `-v` | Verbose: show each test name. |
| `-x` | Stop on first failure. |
| `-k EXPR` | Run tests whose name matches `EXPR` (e.g. `-k "config_validation"`). |
| `--tb=short` | Shorter tracebacks. |
| `tests/test_config_loader.py` | Run a single file. |
| `tests/test_config_validation.py::test_validate_config_minimal_passes` | Run one test. |

**Examples:**

```bash
# Only config-related tests
pytest tests/test_config_loader.py tests/test_config_validation.py tests/test_config_defaults.py -v

# Only tests whose name contains "adapter"
pytest -v -k adapter

# Fail fast with short tracebacks
pytest -x --tb=short
```

Tests are designed to run in a few seconds. They do **not** use GPU, real checkpoints, or DeepSpeed; config, validation, defaults, data loaders, and optimizer/scheduler resolution are tested with in-memory configs, mocks, and minimal synthetic data.

**Parity / diffusion-pipe-related modules:**

| Test file | Focus |
|-----------|--------|
| `test_param_groups.py` | `beta2_half_life`, weight-decay split |
| `test_training_metrics.py` | TensorBoard logging helpers |
| `test_loss_utils.py` | Huber / pseudo-Huber / MSE |
| `test_dataset_options.py` | `seed_from_hash`, `online_captions` |
| `test_gradient_release.py` | `GradientReleaseOptimizerWrapper` |
| `test_optim_resolver.py` | Vendor optimizers (skip if `optimum` missing) |
| `test_config_cosmos_predict2.py` | Cosmos Predict2 config validation |
| `test_cosmos_predict2_param_groups.py` | `llm_adapter_lr=0` freezes adapter params |
| `test_cosmos_predict2_assets.py` | Bundled tokenizer assets via `importlib.resources` |
| `test_preprocess_media.py` | Media bucket rounding helpers |
| `test_config_form.py` | UI schema registries, TOML round-trip, **`test_all_config_fields_have_help`** |
| `test_dataset_field_help.py` | All dataset schema fields have `help` text |
| `test_datasets_store.py` | Dataset library CRUD and `compose_datasets` |
| `test_dataset_scan.py` | Folder image count / preview aggregation |
| `test_system_stats.py` | Host metrics collector and `nvidia-smi` line parse |
| `test_job_queue.py` | Pending queue ordering and updates |
| `test_docs_reader.py` | Safe markdown path resolution |
| `test_registry_probe.py` | Optimizer/scheduler import probe |

### Web UI (control plane)

Run the UI-focused suite (no browser, no GPU):

```bash
pytest tests/test_ui_api.py tests/test_configs_store.py tests/test_config_form.py \
  tests/test_dataset_form.py tests/test_dataset_field_help.py tests/test_datasets_store.py \
  tests/test_dataset_scan.py tests/test_job_queue.py tests/test_ui_job_queue.py \
  tests/test_rengu_flow_ui.py tests/test_docs_reader.py tests/test_system_stats.py \
  tests/test_registry_probe.py -q
```

| Test file | Focus |
|-----------|--------|
| `tests/conftest.py` | `ui_data_tmp`, `ui_client`, auth fixtures |
| `tests/test_ui_api.py` | FastAPI routes: configs, datasets, jobs, schema, docs, stats |
| `tests/test_configs_store.py` | Staging paths, validation, `_safe_id` |
| `tests/test_config_form.py` | TOML ↔ form, schema help, registries |
| `tests/test_dataset_form.py` | Dataset TOML ↔ form, directory rows |
| `tests/test_ui_job_queue.py` | Queue reorder, delete pending, `try_start_next` |
| `tests/test_job_queue.py` | Enqueue + edit pending (baseline) |
| `tests/test_datasets_store.py` | Dataset library CRUD + compose |
| `tests/test_dataset_scan.py` | Folder scan / preview aggregation |
| `tests/test_system_stats.py` | Host metrics collector |
| `tests/test_docs_reader.py` | Markdown path safety |
| `tests/test_registry_probe.py` | Optimizer/scheduler import probe |
| `tests/test_rengu_flow_ui.py` | Config store CRUD smoke |

Requires `pip install -e ".[ui,dev]"` (FastAPI TestClient + httpx).

### Manual GPU smoke (Cosmos Predict2 / Anima)

Not run in CI. With `.[cosmos_predict2]` installed and real checkpoint paths:

```bash
deepspeed --num_gpus=1 --module rengu_flow.main --config my.toml --cache_only
deepspeed --num_gpus=1 --module rengu_flow.main --config my.toml
```

Repeat for `examples/minimal_config_cosmos_predict2_{lora,lokr,finetune}.toml`. Success: cache on disk, training loop runs, run dir contains `adapter_model.safetensors` (LoRA/LoKr) or `model.safetensors` (finetune).

```bash
pytest -k "param_groups or training_metrics or loss_utils or dataset_options or gradient_release"
```

## Layout

- **`tests/`** — All test modules.
- **`tests/conftest.py`** — Shared pytest fixtures (e.g. `minimal_config`, `minimal_config_copy`, `examples_dir`, `valid_toml_content`).

**Per-file focus:** Each test file targets one area (e.g. `test_config_loader.py` for config loading, `test_config_validation.py` for validation). This keeps the suite easy to navigate and extend.

## Fixtures (conftest.py)

| Fixture | Description |
|---------|-------------|
| `minimal_config` | Minimal valid config dict (model.type, model.dtype, optimizer.type, dataset). |
| `minimal_config_copy` | Deep copy of `minimal_config` for tests that mutate the config (e.g. defaults). |
| `examples_dir` | Path to the `examples/` directory. |
| `valid_toml_content` | Valid TOML string for writing temporary config files. |

Use `tmp_path` (pytest built-in) for any temporary files; avoid depending on `examples/` except in integration tests (e.g. `test_example_configs.py`).

## Conventions

- **No GPU:** Use `device="cpu"` and small shapes (e.g. 64×64, `micro_batch_size=1`, `num_batches=2`).
- **No heavy I/O:** Do not load real checkpoints or call `load_diffusion_model()` in unit tests; use mocks for model/engine when needed.
- **Parametrize:** Use `@pytest.mark.parametrize` for multiple similar cases (e.g. several config names or adapter types) instead of many nearly identical tests.
- **Success and failure:** Cover both the passing path and the failing path with the expected exception (`pytest.raises(..., match=...)` or assert on the message).

Tests that need the model registry (e.g. `test_registry.py`) may skip if optional dependencies (e.g. diffusers/huggingface_hub) are missing.

## Dataset and data loading tests

Dataset-related tests (config, loader, captions, cache hooks, smoke fixture):

| File | What it covers |
|------|----------------|
| `tests/test_dataset_config.py` | **Dataset config loading:** `load_dataset_config` (no dataset / `None`, missing file, valid TOML with `resolutions` and `directories`); `load_eval_dataset_config` (path string, dict with `name` + `config`, missing path). |
| `tests/test_data_loader.py` | **PipelineDataLoader:** empty dataset raises; `len(loader)`; iteration; thread prefetch; DataLoader kwargs; `reset()`. Uses `SyntheticSDXLDataset` and mocks. |
| `tests/test_data_split_batch.py` | **split_batch:** correct number of pieces and sizes (parametrised); tensors `None` produce empty tensors per piece. This is splitting a batch into **micro-batches** for gradient accumulation, not train/val or folder-based split. |
| `tests/test_data_synthetic.py` | **SyntheticSDXLDataset:** length, keys and shapes of `__getitem__`; device and dtypes; reproducibility (same item returns same tensors). |
| `tests/test_dataset_captions.py` | **Caption formats:** multi-line `.txt` (one caption per line), `captions.json` (list or string per image, JSON over `.txt` when both exist), `DirectoryDataset._metadata_map_fn`, `SizeBucketDataset` multi-caption `iteration_order` and `online_captions` / `caption_number`. |
| `tests/test_smoke_cc0_dataset.py` | Versioned CC0 fixture (12 jpg/txt pairs, manifest, `dump_dataset` on `tests/fixtures/smoke/dataset_cc0.toml`). |
| `tests/test_sdxl_cache_hooks.py` | SDXL `get_preprocess_media_file_fn`, `get_text_encoders`, `get_call_text_encoder_fn` (mocked). |
| `tests/test_sdxl_cached_prepare_inputs.py` | SDXL `prepare_inputs` / `InitialLayer` cached embedding path (mocked). |
| `tests/test_dump_dataset.py` | `rengu_flow.data.dump_dataset` on a temporary directory. |
| `tests/test_oom_skip.py` | `is_cuda_oom`, `OomSkipState`, `handle_oom_skip` (CPU). |
| `tests/test_cosmos_load_and_fuse.py` | Cosmos `load_and_fuse_adapter` raises `NotImplementedError` (documented). |
| `tests/test_cache_utils_config.py` | `resolve_cache_num_proc`, `_map_and_cache` `keep_in_memory`. |
| `tests/test_bench_utils.py` | `bench_mean_iter_sec_after_warmup`, `find_latest_bench_csv`. |

| `tests/test_signal_files.py` | File-based signals: `process_signals` per name, `save_quit` priority over `save`. |
| `tests/test_saver_signals.py` | `Saver.process_step` with mocked `save_checkpoint` (all signal actions). |
| `tests/test_ui_signals.py` | `send_signal` and job signals API. |
| `tests/test_genericoptim_cpu_state.py` | `GenericOptim` + `kahan_buffer_offload` CPU state roundtrip. |

**GPU smokes (optional, local):** Copy `.env.example` to `.env` (gitignored) and set `RENGU_SDXL_CHECKPOINT_PATH` / `RENGU_COSMOS_*`. Configs live under `tests/fixtures/smoke/` (not `examples/`). `scripts/run_model_smoke.sh sdxl|cosmos` vendors fixtures if needed, runs `--cache_only`, then **30** training steps. `scripts/smoke_training_signals.sh` exercises every signal file plus `genericoptim` resume (requires `pip install -e ".[optim]"`). `rengu_flow.config.local_env` applies env vars before validation. Scripts **purge** `output/`, fixture caches, and `tmp/smoke_*.log` by default (`KEEP_SMOKE_ARTIFACTS=1` / `KEEP_SMOKE_LOG=1` to retain).

**GPU smoke A/B (dataloader/cache flags):** After unit tests pass, run `scripts/smoke_perf_ab.sh sdxl` for baseline `iter_sec_mean` (steps ≥ 6 from `bench_steps.csv`), then e.g. `scripts/smoke_perf_ab.sh sdxl prefetch` to compare `dataloader_prefetch=true`. Presets: `prefetch`, `workers2`. See [performance-cpu-ram](performance-cpu-ram.md).

**POC CPU/RAM ideas (no GPU):** `python scripts/poc_cpu_ram_optimizations.py` — CI runs a short smoke via `tests/test_poc_cpu_ram_smoke.py`. Results and default policy: [poc-cpu-ram-results](poc-cpu-ram-results.md).

**Not covered:**

- **Real-data training E2E on GPU** (full `DatasetManager.cache()` + train) — hook unit tests and `smoke_cc0` fixture cover the data path without checkpoints.
- **Dataset augmentation:** `tests/test_augmentation.py`, `tests/test_dataset_form_augmentation.py`.

## Adding tests

When you add a new module or feature:

1. Prefer a **new test file** if the area is new (e.g. `test_<module>.py`), or add to the existing file that matches the topic.
2. Reuse **fixtures** from `conftest.py`; add new ones there if several tests need the same setup.
3. Use **mocks** for DeepSpeed, heavy models, or external services so the suite stays fast and runnable without GPU.
4. Use **parametrize** for multiple equivalent scenarios (e.g. different adapter types or scheduler names) to avoid test proliferation.
