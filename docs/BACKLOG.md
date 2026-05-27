# Implementation backlog

Canonical list of **not-yet-implemented** or **deferred** work for Renga Flow. Developer specs may still use **`[TODO]`** inline; this file is the durable backlog.

**Design context:** [developer/architecture.md](developer/architecture.md). **Last updated:** 2026-05-26.

---

## How to use

- Add new items here when deferring work; keep specs accurate for *current* behaviour and link here for *planned* work.
- When an item ships, remove it from this file and update the relevant developer spec.
- Grep specs: `rg '\[TODO\]' docs/developer`

---

## P0 — Model pipeline and training core

| ID | Item | Source | Notes |
|----|------|--------|-------|
| P0-1 | **`enable_block_swap` on SDXL** | [model-pipeline-contract.md](developer/model-pipeline-contract.md) | `BasePipeline.enable_block_swap` still raises; `main.py` calls it when `blocks_to_swap` is set. |
| P0-2 | **`enable_block_swap` during training (Cosmos)** | [previews.md](developer/previews.md) | Preview uses `CosmosBlockOffloader`; training uses `NoopOffloader`. |
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
| P2-1 | **`cache_dedup_text_embeddings`** | [poc-cpu-ram-results.md](developer/poc-cpu-ram-results.md) | Opt-in dedup for tag-heavy datasets. |
| P2-2 | **Safetensors pack per bucket** | [poc-cpu-ram-results.md](developer/poc-cpu-ram-results.md) | Alternative on-disk layout to stacked `.bin` (cache v2). |
| P2-3 | **diffusion-pipe `utils/offloading.py` incorporation** | [dependencies-and-upstream.md](developer/dependencies-and-upstream.md) | Tied to `blocks_to_swap`; out of current “austere” scope. |

---

## P3 — Config and export

| ID | Item | Source | Notes |
|----|------|--------|-------|
| P3-1 | **`save_full_model` TOML flag** | [checkpoint-and-save.md](developer/checkpoint-and-save.md) | Not read by `Saver`; full export today = omit `[adapter]`. |

---

## P4 — Dataset augmentation

**Runtime source of truth:** `renga_flow/data/augmentation/names.py`.

### Deferred presets (validation error if enabled)

`photo_cinematic`, `retro_scan`, `manga_print`.

### Future work

- Remaining catalogue strategies ([dataset-augmentation.md](developer/dataset-augmentation.md)).
- Video per-frame augmentation.
- Enable deferred presets once strategies exist.
- **`enumerated` sampling** for continuous strategies after explicit discretisation in spec.

---

## Spec cross-reference (`[TODO]` in `docs/developer/`)

| File | Markers |
|------|---------|
| [model-pipeline-contract.md](developer/model-pipeline-contract.md) | SDXL `enable_block_swap` |
| [networks.md](developer/networks.md) | LoHA, Flux LoRA/LoKr, non-SDXL guide |
