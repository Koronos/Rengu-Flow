# Implementation backlog

Canonical list of **not-yet-implemented** or **deferred** work for Rengu Flow. Developer specs may still use **`[TODO]`** inline; this file is the durable backlog.

**Design context:** [developer/architecture.md](developer/architecture.md). **Specifications:** [spec/](spec/). **Last updated:** 2026-08-09 (P5-2 shipped: Workflows engine, API and editor).

---

## How to use

- Add new items here when deferring work; keep specs accurate for *current* behaviour and link here for *planned* work.
- When an item ships, remove it from this file and update the relevant developer spec.
- Grep specs: `rg '\[TODO\]' docs/developer`

---

## P0 — Model pipeline and training core

| ID | Item | Source | Notes |
|----|------|--------|-------|
| P0-3 | **Adapter registry** | [architecture.md](developer/architecture.md) | By `adapter.type`; today branches in `sdxl.py` / `adapter_dit`. |
| P0-4 | **Generic step/epoch callback registry** | [architecture.md](developer/architecture.md) | Partial: `Saver`, eval, previews. |
| P0-5 | **`pre_train` hook registry** | [architecture.md](developer/architecture.md) | Partial: `DatasetManager.cache()` only. |
| P0-6 | **Post-training hooks** | [architecture.md](developer/architecture.md) | Hub upload, format conversion, notifications. |
| P0-7 | **More models in registry** | [architecture.md](developer/architecture.md) | Built-in: `sdxl`, `cosmos_predict2` (+ `anima`). Flux, etc. not registered. |
| P0-8 | **TOML plug-in phases** | [architecture.md](developer/architecture.md) | e.g. dataset tagging phases, `[[post_train]]` metadata handlers. |
| P0-9 | **Special optimizer cases (full parity)** | diffusion-pipe parity | Gradient release, GenericOptim edge cases beyond current resolver. |

---

## P1 — Networks and adapters

| ID | Item | Source | Notes |
|----|------|--------|-------|
| P1-1 | **New SDXL network types** (e.g. LoHA) | [networks.md](developer/networks.md) | Example: `loha_sdxl.py` + branches in `sdxl.py`, defaults, validation. |
| P1-2 | **LoRA/LoKr for non-SDXL models** (e.g. Flux) | [networks.md](developer/networks.md) | Example: `lora_flux.py`, `lokr_flux.py`, pipeline delegation. |

**Intentionally not supported (documented, not backlog):**

- Cosmos **`load_and_fuse_adapter`** — use `load_adapter_weights` ([cosmos-predict2-pipeline.md](developer/cosmos-predict2-pipeline.md)).

---

## P2 — Dataset, cache, and performance

| ID | Item | Source | Notes |
|----|------|--------|-------|
| P2-2 | **Safetensors pack per bucket** | [poc-cpu-ram-results.md](developer/poc-cpu-ram-results.md) | Alternative on-disk layout to stacked `.bin` (cache v2). |
| P2-4 | **Per-directory schedule windows** | `dataset.py:1953`, `main.py:scheduled_epoch_len` | Today the schedule is global-per-resolution (`phi` per resolution in `scheduled_epoch_len`). Generalize to a per-`[[directory]]` activation window over `[0,1]` so a folder can deactivate at a point, or two clones (aug on/off) interleave. Buckets currently merge size-buckets across all directories (`ConcatenatedBatchedDataset`), so this needs splitting that grouping + sampler composition by directory + UI/validation. The step calc already accommodates it: `phi` moves from per-resolution to per-(directory,resolution) bucket, same formula. |
| P2-3 | **Block-swap overlapped prefetch + Linux/multi-GPU validation** | [spec/sdxl-full-finetune-low-vram.md](spec/sdxl-full-finetune-low-vram.md) | `block_swap_prefetch` is opt-in/off — counterproductive on 8 GB WSL (sysmem spill); validate on native Linux / larger GPU, add GPU-buffer-reuse so pulls don't alloc fresh tensors each step. Also validate `pipeline_stages > 1` / multi-GPU + the DeepSpeed `_broadcast_model`/`PipelineModule.to` no-ops there. WSL2 was dev-only. |

---

## P3 — Config and export

| ID | Item | Source | Notes |
|----|------|--------|-------|
| P3-1 | **`save_full_model` TOML flag** | [spec/save-full-model-flag.md](spec/save-full-model-flag.md), [checkpoint-and-save.md](developer/checkpoint-and-save.md) | Not read by `Saver`; full export today = omit `[adapter]`. |

---

## P4 — Dataset augmentation

**Runtime source of truth:** `rengu_flow/data/augmentation/names.py`.

### Deferred presets (validation error if enabled)

`photo_cinematic`, `retro_scan`, `manga_print`.

### Future work

- Remaining catalogue strategies ([dataset-augmentation.md](developer/dataset-augmentation.md)).
- Video per-frame augmentation.
- Enable deferred presets once strategies exist.

---

## P5 — Workflows and control plane

**Spec:** [spec/workflows.md](spec/workflows.md). Phase order is a hard constraint: P5-1 and P5-2 (both shipped) had to land
before P5-3, or prep loses its only guarantee of not sharing the GPU with a training run
(`prep_jobs.py:1-8`).

**P5-1 (GPU lease + groundwork) and P5-2 (engine + editor) shipped 2026-08-08/09** and are no
longer listed here. What they left behind for P5-3:

- `jobs.gpu_index` exists but nothing writes it and `CUDA_VISIBLE_DEVICES` is not applied
  anywhere, so `_devices_for_job` always returns `None` (host-exclusive). Per-node GPU pinning is
  specced and plumbed but not yet wired end to end.
- The unconditional `try_start_next()` in the poller tick is still **not** there: it belongs with
  the workflow lane and lands together with the two contract comments it invalidates
  (`jobs.py:254-256`, `job_queue.py:495-496`) and a test fixing the new semantics.
- `prep_jobs`' two `start_now` call sites still bypass the lease entirely (Risk 14 in the spec) —
  a prep job can start on a GPU a training run already holds, in the window where that run's row
  is still `pending` during its own `uv sync`. Close this before P5-3 removes the shared queue.
- No "Accept current configuration" action yet (spec, Staleness): after a release that changes a
  stage default, every saved node goes amber with no way to accept them in bulk.
- `/prep/new/index` has a form now but no entry point — `PrepJobsView.vue` offers buttons for
  tag/caption/clean/quality only.

| ID | Item | Source | Notes |
|----|------|--------|-------|
| P5-3 | **Retire `kind='prep'`** | [spec/workflows.md](spec/workflows.md) | Delete `prep_jobs.py`, `POST /prep/jobs`, the `kind=='prep'` branch in `jobs.start_job`; `/prep/new/:stage` becomes a one-node-workflow shortcut. Add `kind == "train"` to `_pending_sorted()` and `has_active_runner()` plus a startup sweep, or a queued prep row left over from an upgrade launches DeepSpeed against `prep.toml`. Existing `kind='prep'` rows are left untouched (no `SCHEMA_VERSION` bump). Watch `QualityIndexView.vue:399` — the only `createPrepJob` caller outside the prep form. |

---

## Spec cross-reference (`[TODO]` in `docs/developer/`)

| File | Markers |
|------|---------|
| [networks.md](developer/networks.md) | LoHA, Flux LoRA/LoKr, non-SDXL guide |
