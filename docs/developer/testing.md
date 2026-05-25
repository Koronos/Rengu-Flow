# Testing

How to run and extend the test suite for Renga Flow.

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

Four test files cover dataset config, the data loader, batch splitting, and the synthetic dataset:

| File | What it covers |
|------|----------------|
| `tests/test_dataset_config.py` | **Dataset config loading:** `load_dataset_config` (no dataset / `None`, missing file, valid TOML with `resolutions` and `directories`); `load_eval_dataset_config` (path string, dict with `name` + `config`, missing path). |
| `tests/test_data_loader.py` | **PipelineDataLoader:** empty dataset raises; `len(loader)`; one iteration yields correct micro-batch structure. Uses `SyntheticSDXLDataset` and mocks for model/engine. |
| `tests/test_data_split_batch.py` | **split_batch:** correct number of pieces and sizes (parametrised); tensors `None` produce empty tensors per piece. This is splitting a batch into **micro-batches** for gradient accumulation, not train/val or folder-based split. |
| `tests/test_data_synthetic.py` | **SyntheticSDXLDataset:** length, keys and shapes of `__getitem__`; device and dtypes; reproducibility (same item returns same tensors). |

**Not covered / `[TODO]` in code:**

- **`[TODO]` Real-data training E2E:** Directory datasets and `DatasetManager.cache()` exist (`tests/test_cache.py`, etc.), but SDXL still lacks `get_preprocess_media_file_fn` / `get_call_text_encoder_fn` — add integration tests when those hooks land.
- **`[TODO]` `--dump_dataset`:** CLI flag prints not implemented; no tests.
- **Dataset augmentation:** Spec only; no tests until `apply_augmentation` exists.

## Adding tests

When you add a new module or feature:

1. Prefer a **new test file** if the area is new (e.g. `test_<module>.py`), or add to the existing file that matches the topic.
2. Reuse **fixtures** from `conftest.py`; add new ones there if several tests need the same setup.
3. Use **mocks** for DeepSpeed, heavy models, or external services so the suite stays fast and runnable without GPU.
4. Use **parametrize** for multiple equivalent scenarios (e.g. different adapter types or scheduler names) to avoid test proliferation.
