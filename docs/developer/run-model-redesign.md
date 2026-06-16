# Proposed redesign: unified run model + library migration mode

> **Status: PROPOSED — not yet implemented.** This document captures a design direction
> for discussion. It does **not** describe current behavior. Until implemented, the code
> still uses the two-step config-library → job flow described in
> [web-ui.md](web-ui.md) and [cli.md](../user/cli.md).

## 1. Motivation

Today Rengu Flow separates two concepts with two user steps:

1. **Config** — a TOML stored in the `training_configs` library table; created/edited in
   the config editor.
2. **Job** — a row in the `jobs` table that *references* a config by `config_id` and adds
   runtime data (pid, run_dir, output_dir, state, queue position, logs).

This forces "first author a config, then start a job from it". The desired model (as in
ai-toolkit) is **a single entity and a single step**: you create a *run* that already
carries its full configuration plus its runtime relation (where it runs, output dir,
status). There is no separate "save the config first" step.

Editing an existing run should **not** mutate the finished run: it is a **"create new"**
seeded with the previous run's configuration but **without** the previous run's runtime
data (new run_dir, fresh state, no inherited logs/checkpoints).

## 2. Proposed model

A single **Run** entity owns:

- **Configuration** — the full training TOML, inline (the same content the trainer reads).
- **Runtime relation** — output/run directory, state (`pending`/`running`/`finished`/…),
  pid, queue position, started/finished timestamps, exit code, log path.
- **Identity** — see §3.

"Edit" = clone the configuration into a new Run; the old Run stays immutable as history.

Datasets remain a separate library entity (they are shared across runs and referenced by
id), but the **config-as-separate-library-record** step goes away: a run carries its own
config. (Open question §6: keep an optional "config templates/presets" library for reuse,
or rely purely on "clone an existing run".)

## 3. Identity and naming (decided)

- **ids are always integer autoincrement.** No mixed/slug primary keys — a single id
  scheme is simpler to reason about and is the API reference key. (We explicitly rejected
  string/slug ids: "un enfoque mixto, si bien es tolerante, complica saber dónde se usa
  uno u otro".)
- **Human identification uses names, not ids.** The string-id idea was really about
  *identifying* runs/configs/datasets; that need is met by names:
  - Datasets already store a `name` column (the script-mode dataset TOML carries no name).
  - Configs/runs are identified by their `run_name` (from the TOML).
  - Names may collide; ids never do. The id is the unambiguous API handle.

## 4. Library as a TOML store + migration mode (IMPLEMENTED — Phase 4)

The SQLite DB exists mainly to avoid scattering TOML files everywhere. Conceptually each
row is **a TOML blob in one field plus a few index columns** derived from it
(`model_type`, `dataset_ref`, `run_name`, `directory_count`, timestamps…).

**Migration mode** (export/import) is the durable, schema-independent format:

- **Export**: dump every DB row to a `<id>.toml` file. The index columns are written into
  a dedicated section the trainer's validator **ignores** (e.g. `[__rengu_index]`), so the
  exported file is still a valid training/dataset TOML.
- **Import**: a separate process reads those TOML files back, **ignoring any section/keys
  it does not recognize** (forward/backward tolerant). Re-import rebuilds the index columns
  from content.

This makes the DB disposable: the TOML files are the source of truth you can re-import into
any schema version.

## 5. Schema-change guard (IMPLEMENTED — Phase 1, interim)

Until migration mode exists, a **schema version** is stored in the DB (e.g.
`PRAGMA user_version`, mirrored by `maintenance.SCHEMA_VERSION`). On startup, if an existing
DB's version differs from the code's:

- Warn the user that the schema changed and the DB is **incompatible**.
- Offer a **Yes/No**: **Yes** → wipe and recreate empty (data lost); **No** → abort and tell
  them to use the previous app version (or, once available, export→migrate).
- Fresh DBs are stamped with the current version, so this never fires in tests or first run.

## 6. Data directory location (IMPLEMENTED — Phase 1)

Move local UI state out of the hidden `.rengu-flow-ui/` into a **non-hidden, git-ignored
`data/`** folder at the repo root (overridable via `RENGU_FLOW_UI_DATA`). Easier for users
to find the DB, logs, and staging. Update `.gitignore`, `settings.py`, launcher scripts,
and docs together.

The default is `data/` everywhere (`settings.py`, `local_config.UiConfig`, and the
`rengu.local.toml.example`). Legacy values are retired without a migration map: a configured
`data_dir` of `.rengu-flow-ui` / `.renga-flow-ui` (the historical typo) is dropped to the `data/`
default by `parse_local_config_dict`, and `migrate_legacy_ui_data_dir` moves an existing hidden
folder's contents into `data/` on startup (adopt-only when `data/` has no `jobs.db` yet — it never
clobbers; a conflicting legacy folder is left for the user to delete). `LEGACY_UI_DATA_DIRNAMES`
in `local_config.py` lists the retired names.

## 7. Phased implementation (proposal)

1. **Independent, low-risk now** (no model change): data dir → `data/`; schema-version
   guard. Shippable on their own. — **DONE** (`SCHEMA_VERSION` in `db.py`, startup guard in
   `cli.py`, default dir in `settings.py`/`local_config.py`).
2. **Run model — data layer**: introduce the unified run entity (config inline on the run),
   keep datasets as-is. Update job_queue, job_import, runs_scanner.
   - **Phase 2a — DONE**: each run carries an immutable `config_content` snapshot
     (`jobs.config_content`, set in `prepare_job`, exposed in the job API). `clone_run` +
     `POST /api/v1/jobs/{id}/clone` implement "edit = create a new run with the same config
     but no previous-run data" (fresh run_dir, no resume). `SCHEMA_VERSION` bumped to 2, so
     the startup guard wipes-and-recreates incompatible local DBs (no users → no migration).
   - **Phase 2b — TODO**: collapse the create paths into a single create-and-run entry; treat
     the `training_configs` library as optional presets rather than a required first step.
3. **API + UI**: collapse the "create config then start job" flow into one create-and-run
   step; implement "edit = clone to new run".
   - **FE-1 — DONE**: `api.cloneJob` + Clone button in JobsView rows and RunDetailView
     (uses `POST /jobs/{id}/clone`). `JobRecord.config_content` exposed; `JobStartBody`
     gains optional `content` and optional `config_id`.
   - **FE-2 — DONE**: ConfigEditorView "Run now" / "Add to queue" start a run directly from
     the inline config (POST /jobs with `content`), no mandatory library save. "Save" now
     means "keep as a reusable preset".
   - **FE-3 — DONE**: the JobsView picker and PickForJobBanner now present the library as
     optional "presets" and point to "Create & run" (editor single-step) as the primary path;
     the pick-from-library flow reads as optional.
4. **Migration mode** — **DONE** (`rengu_flow_ui/library_migration.py`): `export_library` /
   `import_library` round-trip configs and datasets to `<dir>/configs|datasets/<id>.toml`
   with an appended `[__rengu_index]` table the trainer ignores. CLI: `rengu-flow-ui
   export-library <dir>` / `import-library <dir> [--overwrite]`. The schema guard message
   now points users to export before wiping. (Runs/jobs stay disk-backed and are re-imported
   via `job_import`.)

## 8. `job_import` under the new model (DONE)

Imported runs follow the decided identity rules: ids stay **integer autoincrement**, and the
run is identified by **name**. On import (`_import_config_from_run`), the dataset is stored
with `name = "<run> dataset"` and the config is labelled with `run_name = <run>`. The job
references the config by its integer library id. `test_preview_and_import_run` was rewritten to
this contract (no more `xfail`): it asserts an integer `config_id`, `config_exists(config_id)`,
the config content carries the run name, and a dataset named after the run exists.
