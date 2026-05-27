# Train hub UX (unified Training / Train / Runs)

Design target: one **Train** surface similar to AI-toolkit — configs and runs in one place, with live progress for the active job without overwhelming the UI.

## Information architecture

| Nav item | Role |
|----------|------|
| **Datasets** | Dataset TOML library |
| **Training** | Config TOML library (templates / recipes) |
| **Train** | Hub: launch, queue, live monitor, unified run list |
| ~~Runs~~ | Removed from nav; disk-only runs appear in Train list (`state: on_disk`) |

Run detail stays at `/jobs/:id` (managed jobs) and `/runs/:name` (filesystem-only).

## Train page layout

```
┌─────────────────────────────────────────────────────────────┐
│ Train                                    [N running] [M queue]│
├─────────────────────────────────────────────────────────────┤
│ LIVE PANEL (only when state ∈ running|stopping)              │
│  • Progress bar: step / max_steps (or epoch)                 │
│  • Loss, ETA hint, link to full detail                       │
│  • Mini-TB: line chart (train/loss + tag selector)         │
│  • Preview strip: latest PNGs from run/preview/              │
│  • Quick actions: Stop, Preview signal, Open TensorBoard     │
├─────────────────────────────────────────────────────────────┤
│ [+] New run (collapsed launch form: config, GPUs, resume)    │
├─────────────────────────────────────────────────────────────┤
│ Unified run list (single table)                              │
│  Filter: All | Active | Queued | Finished | On disk          │
│  Search: config id, run folder name                          │
│  Columns: State, Config, Run folder, Progress, Updated       │
│  Row actions: Open, Edit config, Continue, Duplicate, Queue  │
├─────────────────────────────────────────────────────────────┤
│ History: paginated; default sort = most recent first         │
└─────────────────────────────────────────────────────────────┘
```

## Anti-saturation rules

1. **Live panel** shows metrics for **at most one** run (first `running` job).
2. **History** is paginated (default 20); disk-only runs optional via filter.
3. **Mini-TB** is line charts + small preview strip — not full TensorBoard. Link opens external TB for deep history.
4. **Launch form** collapses after first use; primary CTA is “New run from config…”.

## Unified run model (`TrainingRun`)

Each list row is a `TrainingRun`:

| Field | Source |
|-------|--------|
| `key` | `job:{id}` or `disk:{folder_name}` |
| `kind` | `job` \| `disk` |
| `state` | job state or `on_disk` |
| `config_id` | jobs DB |
| `run_dir` / `run_name` | job or filesystem scan |
| `progress` | `status.json` + run TOML (`max_steps`, `epochs`) |
| `started_at` / `finished_at` | jobs DB |

## Progress & ETA

- **Primary**: `run_dir/status.json` when `monitoring.enable_status_file = true`.
- **Fallback**: last `train/loss` point from TensorBoard events.
- **Total steps**: `max_steps` from run snapshot TOML; percent = `step / max_steps`.
- **ETA** (v1): optional; requires two status samples — deferred until stable.

## Mini-TensorBoard scope

- **Where**: Train live panel + job/run detail (replace bar chart).
- **Charts**: SVG line series from `/jobs/:id/metrics` scalars.
- **Previews**: glob `run_dir/preview/*.png`, newest first.
- **Not in scope**: multi-run comparison, full TB plugin set.

## Row actions

| Action | Job | Disk run |
|--------|-----|----------|
| Open detail | ✓ | ✓ (`/runs/:name`) |
| Edit config | library config id | import / continue flow |
| Continue training | ✓ | ✓ |
| New run from config | duplicate config → launch | import config |
| Queue / Stop | ✓ | — |

## API

- `GET /api/v1/train/runs` — unified list + stats + pagination
- `GET /api/v1/train/active` — active job + progress + scalars + preview images
- `GET /api/v1/train/preview-image?run_dir=…&name=…` — serve PNG from run preview folder

Existing `/jobs`, `/runs` endpoints remain for compatibility.
