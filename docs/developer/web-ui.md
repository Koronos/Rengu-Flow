# Web UI (developer guide)

Optional control plane: **`renga_flow_ui`** package + **`ui/web`** SPA + **`start-ui.sh`** at repo root.

User guide: **`docs/user/web-ui.md`**.

## Layout

| Path | Role |
|------|------|
| `renga_flow_ui/app.py` | FastAPI routes (`/api/v1/...`) |
| `renga_flow_ui/settings.py` | `RENGA_FLOW_UI_DATA`, `repo_root()`, `web_dist_dir()` |
| `renga_flow_ui/paths.py` | `resolve_repo_path()` — relative paths under repo root |
| `renga_flow_ui/library_db.py` | SQLite tables `training_configs` + `datasets` (TOML content + index columns) |
| `renga_flow_ui/configs_store.py` | Training config CRUD (via library_db), validate, staging |
| `renga_flow_ui/datasets_store.py` | Dataset CRUD (via library_db), `compose_datasets()`, picker refs |
| `renga_flow_ui/dataset_scan.py` | `scan_folder()`, `preview_dataset_config()` for UI previews |
| `renga_flow_ui/dataset_image_preview.py` | Signed tokens, list/serve images under `[[directory]]` paths |
| `renga_flow_ui/dataset_schema.py` | Dataset form schema (`get_dataset_schema`) |
| `renga_flow_ui/dataset_form.py` | Parse/render dataset TOML ↔ form (`_directories` rows) |
| `renga_flow_ui/config_schema.py` | Training form schema (`get_schema`), model capabilities |
| `renga_flow_ui/config_form.py` | Parse/render training TOML ↔ form |
| `renga_flow_ui/config_field_help.py` | `FIELD_HELP` + `enrich_schema()` — attaches `help`, `doc_path` per field |
| `renga_flow_ui/dataset_field_help.py` | Same for dataset form |
| `renga_flow_ui/docs_reader.py` | Safe read of `docs/**/*.md` for in-app help drawer |
| `renga_flow_ui/registry_probe.py` | `POST /registry/probe` — import-check optimizer/scheduler names |
| `renga_flow_ui/system_stats.py` | `GET /system/stats` — CPU/RAM/GPU via `psutil` + `nvidia-smi` |
| `renga_flow_ui/job_queue.py` | Pending job queue ordering, `try_start_next()` |
| `renga_flow_ui/jobs.py` | Subprocess launcher (`deepspeed` / `python -m renga_flow.main`) |
| `renga_flow_ui/db.py` | SQLite job registry |
| `renga_flow_ui/metrics_tb.py` | TensorBoard `EventAccumulator` (passive read, mtime cache) |
| `renga_flow_ui/tensorboard_server.py` | Spawn TensorBoard with `uv run --no-project --with tensorboard` |
| `renga_flow_ui/subprocess_util.py` | `popen_repo_subprocess()` shared by jobs and TensorBoard |
| `renga_flow_ui/runs_scanner.py` | List/discover runs under `output_dir` |
| `renga_flow_ui/signals.py` | Touch signal files via `renga_flow.utils.signal_files` constants |
| `renga_flow/control/status_file.py` | Opt-in `status.json` writer (trainer hook in `main.py`) |
| `ui/web/` | Vite + Vue 3 + Element Plus SPA; build output `ui/web/dist/` |
| `start-ui.sh` | User entrypoint: install `[ui]`, build web, `renga-flow-ui serve` |
| `scripts/start-ui-dev.sh` | Developer-only: API `--reload` + Vite on port 5173 (proxies `/api`) |

## Local development (UI)

End users run `./start-ui.sh` (built SPA from `ui/web/dist/`). While editing the Vue app or `renga_flow_ui/`:

```bash
./scripts/start-ui-dev.sh [--no-open]
```

| Process | URL |
|---------|-----|
| Vite | [http://127.0.0.1:5173](http://127.0.0.1:5173) (proxies `/api`) |
| API | [http://127.0.0.1:8765](http://127.0.0.1:8765) (`uvicorn --reload`) |

Ctrl+C stops both. No frontend build required.

## API prefix

All JSON routes: `/api/v1`. Static SPA mounted at `/` when `ui/web/dist/index.html` exists.

**Maintenance** (dev only, `RENGAFLOW_MAINTENANCE=1`): `GET /maintenance/enabled`, `GET /maintenance/status`, `POST /maintenance/database/reset`, `POST /maintenance/submodules/update`, `POST /maintenance/deps/install`. Implemented in `renga_flow_ui/maintenance.py`; UI route `/maintenance`. See **`docs/user/maintenance.md`**.

Authentication: optional `RENGA_FLOW_UI_TOKEN` middleware checks `X-Renga-Flow-Token` or `Authorization: Bearer`.

### Route groups

| Prefix | Purpose |
|--------|---------|
| `/configs`, `/configs/{id}`, `/validate`, `/schema`, `/configs/parse-toml`, `/configs/render-toml` | Training config library and form round-trip |
| `/datasets`, `/datasets/{id}`, `/datasets/schema`, `/datasets/compose`, `/datasets/preview`, `/datasets/preview-images`, `/datasets/preview-image`, `/datasets/scan-path` | Dataset TOML library, merge, folder scan, image gallery |
| `/jobs`, `/jobs/{id}`, `/jobs/import`, `/jobs/import/preview`, `/jobs/import/candidates`, `/jobs/{id}/queue/*`, `/jobs/{id}/logs`, `/jobs/{id}/metrics` | Job queue, launch, import script runs, tail logs |
| `/runs`, `/runs/{name}`, `/runs/{name}/signals`, `/runs/{name}/metrics` | Filesystem runs (no DB) |
| `/docs?path=...` | Markdown for help drawer (repo-relative under `docs/`) |
| `/registry/probe` | Optimizer/scheduler import probe |
| `/system/stats` | Host metrics for header bar |
| `/tensorboard/status`, `/tensorboard/start`, `/tensorboard/stop` | Local TensorBoard subprocess (`uv`, `--logdir=<output_dir>`) |
| `/health` | Liveness for `start-ui.sh` browser open |

## Config staging

On `POST /api/v1/jobs`:

1. Load TOML from library or inline body
2. `materialize_staging()` writes `{RENGA_FLOW_UI_DATA}/staging/{job_id}/train.toml` (default `<repo>/.renga-flow-ui`); `renga-flow-dataset:<id>` refs become `{staging}/{job_id}/{id}.dataset.toml` with absolute `[[directory]]` paths
3. Subprocess: `--config <staging>/train.toml` plus optional CLI flags from the request body (stored in `jobs.extra_args`)
4. Trainer copies config into `run_dir` (unchanged `main.py` behavior)

**Cache vs training (Train `/jobs` page):**

| JSON field | CLI flag | Default | Purpose |
|------------|----------|---------|---------|
| `cache_only` | `--cache_only` | `false` | Build dataset cache only; process exits before training. |
| `trust_cache` | `--trust_cache` | `false` | Skip cache rebuild when existing cache is valid. |
| `regenerate_cache` | `--regenerate_cache` | `false` | Force full cache rebuild. |

`cache_only` and `trust_cache` cannot both be true. The UI exposes **Build cache** actions (`cache_only`) and a **Use existing cache** checkbox on normal training launches (`trust_cache`).

## Config / dataset library (SQLite)

- Tables in `{RENGA_FLOW_UI_DATA}/jobs.db`: `training_configs`, `datasets` (same file as `jobs`)
- Content column: full TOML string; index columns: `model_type`, `dataset_ref`, `directory_count`, `meta_json`
- Library dataset reference in training TOML: `renga-flow-dataset:<id>` — resolved to a file under `staging/{job_id}/` at job start
- **`compose_datasets(target_id, source_ids)`** — merges `[[directory]]` blocks into one library record
- **`list_for_training_picker()`** — `renga-flow-dataset:…` refs from the SQLite library only (no repo `examples/` paths)
- Import/export: `POST …/import`, `GET …/{id}/export`
- Validation reuses **`validate_dataset_config_for_real_data`** from `renga_flow.data.dataset_config`

## Config sources: library vs run folder

| Source | Role |
|--------|------|
| **SQLite library** (`training_configs`, `datasets`) | Named templates, search/pagination, compose, UI forms |
| **Run folder `*.toml`** | Ground truth for a finished or in-progress training run; copied/updated by `renga_flow.main` on each start (including resume) |
| **Job row** (`jobs` table) | Queue metadata: `config_id` (library), `config_path` (staging), `resume_from`, `source_run_dir`, `run_dir` |

**Continue training:** `GET /runs/config?run_path=…` reads the run TOML; user edits (e.g. more `epochs`); `POST /jobs/continue-run` stages the new TOML and sets `resume_from` to that folder. Optional `save_to_library` copies the edited TOML into `training_configs` for reuse. The trainer resumes in the **same** `run_dir` and overwrites the snapshot TOML there with the config used for that launch.

## Job import (script runs)

- **`job_import.py`** — `preview_import`, `import_run`; heuristics in `runs_scanner.is_training_run_dir` / `pick_main_config_path`
- **`db.create_imported_job`** — finished job row with `run_dir` set; duplicate detection via `find_job_by_run_dir`
- Optional library copy: training config + `dataset.toml` (or path from config) → `renga-flow-dataset:` ref in stored config

## Default template for **New config**

Canonical file: **`renga_flow_ui/templates/default_new_config.toml`** (loaded by **`default_config_template.default_new_config_toml()`**).

- Exposed on **`GET /api/v1/schema`** as **`default_new_config_toml`** so the Vue editor (`configEditor.ts`) stays in sync with the server.
- Production-style SDXL LoRA: `dataset`, `[model]` + `checkpoint_path`, `[adapter]`, `[optimizer]`, `lr_scheduler` / `[lr_scheduler_args]`, epochs, micro-batch, `output_dir`. No **`synthetic_num_batches`** or other smoke-only keys.
- Offline fallback string in **`ui/web/src/stores/configEditor.ts`** (`FALLBACK_DEFAULT_CONFIG_TOML`) must parse to the same table as the `.toml` file — enforced by **`tests/test_default_new_config_template.py`**.

## Form schema and field help

1. **`get_schema()`** (`config_schema.py`) builds sections/fields from **`model_capability_registry`** and static training keys.
2. **`enrich_schema()`** (`config_field_help.py`) merges `FIELD_HELP[path]` → `description`, `help`, `doc_path`. Every field gets a non-empty **`help`** (fallback: label) so the Vue **`FieldHelpIcon`** always renders.
3. **`attach_visibility_to_schema()`** (`field_visibility.py`) adds a normalized **`visibility`** tree on each field (from `when`, `when_capability`, `show_if_set`, etc.).
4. **`GET /api/v1/docs`** serves markdown; **`doc_path`** on a field must be repo-relative (e.g. `docs/user/training-loop-and-eval.md`). **`docs_reader.resolve_doc_path`** rejects path traversal.

### Field visibility (single place)

Logic lives in **`renga_flow_ui/field_visibility.py`**; the Vue form mirrors it in **`ui/web/src/lib/formUtils.ts`** (`fieldVisible`, `pruneFormForModel`).

| Mechanism | Where to set it | Example |
|-----------|-----------------|--------|
| Per-model paths | `ModelCapability.model_fields` (auto `when: model.type in […]`) | `transformer_path` only for `cosmos_predict2` |
| `ui: false` on a model field spec | Same | Omit from form entirely (TOML-only keys) |
| `features` on capability | `ModelCapability.features` | `block_swap: true` → show `blocks_to_swap` via `when_capability="block_swap"` on any schema field |
| `when_capability` | `_field(...)` in `config_schema.py` or model field spec | Cross-section flags without listing model IDs |
| `show_if_set` | model field spec | Expert fields (e.g. `llm_adapter_path`) only after a value exists |
| `visibility` | explicit dict on field spec | `any` / `all` / `form_nonempty` / `capability` clauses |

When the user changes **`model.type`**, the form calls **`pruneFormForModel`** so stale `model.*` keys are removed before saving TOML.

**Validate** in the UI (`POST /api/v1/validate`) calls the same **`validate_config()`** as the trainer, including **`model_config_rules`** derived from the capability registry. See **[model-capabilities-and-validation.md](model-capabilities-and-validation.md)** for how to extend models without duplicating checks.

To add help for a new config key:

1. Add TOML field to `config_schema.py` (or `ADAPTER_FIELD_TEMPLATES` / model capability `model_fields`).
2. Add entry to **`FIELD_HELP`** in `config_field_help.py` (`summary`, optional `detail`, optional `doc`).
3. Extend **`docs/user/*.md`** with a table row (purpose, values, default) per [documentation conventions](documentation-conventions.md).
4. Add or extend **`docs/developer/*.md`** with code location (this guide + topic page).
5. Extend **`tests/test_config_form.py::test_all_config_fields_have_help`** if the field is in the schema.

Dataset form: same pattern in **`dataset_field_help.py`** and **`tests/test_dataset_field_help.py`**.

## Job queue

- **`job_queue.enqueue_job`** — assigns `queue_position`; only one job **running** at a time
- **`try_start_next()`** — called after job finish/stop; starts highest-priority pending job
- **`PATCH /jobs/{id}`** — edit pending jobs (config, GPUs, resume path, …)
- **`POST /jobs/{id}/queue/move`**, **`start-now`**

## Host metrics

**`system_stats.collect_system_stats()`** — blocking CPU sample (~0.15s), `psutil` RAM, optional `nvidia-smi` CSV for GPU util/VRAM/temp. Returned JSON: `summary` (compact) + `detail` (per-core CPU, sensors, per-GPU extras). No trainer involvement; safe to poll every 2s from the SPA header.

## Signals

`POST /api/v1/jobs/{id}/signals` or `POST /api/v1/runs/{name}/signals` → `Path(run_dir) / SIGNAL_*` via `touch()`. Same contract as [signal-files.md](signal-files.md).

## Extending

- **New signal**: add constant in `signal_files.py`, map in `renga_flow_ui/signals.py`, button in `ui/web/src/views/RunDetailView.vue`
- **New API route**: add handler in `app.py`, mirror in `ui/web/src/api.ts`
- **Trainer status fields**: extend `write_status_file()` payload and UI reader
- **New model in UI**: `@register_model("my_type")` adds the type to the model picker. Optional: `register_model_capability(ModelCapability(...))` in `renga_flow/registry/model_capabilities.py` for LoRA/LoKr/full, preview, per-model form fields. `GET /api/v1/schema` exposes `registries.model_capabilities`
- **Optional config fields**: set `importance="advanced"` on the field in `config_schema.py`; the form shows them inline with muted labels and a small “(optional)” hint (no collapse)

## Tests

Python (API / stores):

```bash
pytest tests/test_status_file.py tests/test_renga_flow_ui.py \
  tests/test_config_form.py tests/test_dataset_field_help.py \
  tests/test_job_queue.py tests/test_docs_reader.py \
  tests/test_system_stats.py tests/test_datasets_store.py -q
```

Frontend (Vitest, `ui/web/`):

```bash
cd ui/web && npm ci && npm test && npm run typecheck
```

## Dependencies

`[project.optional-dependencies] ui` in `pyproject.toml`: `fastapi`, `uvicorn`, `tensorboard` (event reading), `psutil` (host metrics).
