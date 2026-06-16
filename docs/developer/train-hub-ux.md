# Train hub UX (developer note)

The **Train hub** (unified run list combining queued/active jobs with progress and previews) is **shipped**. See the [Web UI developer guide](web-ui.md) for the full layout.

- **Nav** (`ui/web/src/App.vue`): **Runs**, **Compare**, **Datasets**, **Studio** (plus **Docs** and, when enabled, **Maintenance**). There is no separate "Configs" item — training-config editing happens inside the run form (`RunFormView.vue`).
- **Run list / progress** lives in `rengu_flow_ui/training_hub.py` (`list_training_runs`, `compute_run_progress`, `resolve_job_run_dir`), wired into `rengu_flow_ui/app.py`.
- **Launch and queue**: `ui/web/src/views/JobsView.vue` at `/runs`. Legacy `/jobs` and `/jobs/{id}` routes redirect to `/runs` (`ui/web/src/router.ts`).
