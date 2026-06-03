# Spec: SDXL full-model fine-tuning on low VRAM (8 GB), incl. WSL2 findings

**Status:** Implemented and validated on WSL2 / 8 GB; several items flagged below **still need
validation on native Linux** (the project's real target — WSL2 is a dev convenience). **Date:**
2026-05-30. **Hardware used:** NVIDIA RTX 3000 Ada laptop, 8 GB, WSL2 (Windows 11, WDDM), torch
2.12.0+cu130, deepspeed 0.19.0.

This spec records the investigation and decisions behind enabling **full-model SDXL fine-tuning on
8 GB** and the block-swap work. For the practical lever guide see
[developer/vram-optimization.md](../developer/vram-optimization.md); for the user guide see
[user/full-model-training-sdxl.md](../user/full-model-training-sdxl.md).

## Goal

Make full-model SDXL fine-tuning (no adapter) runnable — at least as a smoke test that produces a
loadable `model.safetensors` — on an 8 GB GPU. Full SDXL UNet is ~2.6 B params; naïve fp32-Adam
full fine-tuning needs ~50 GB. The low-VRAM stack (bf16+Kahan, activation checkpointing,
gradient_release, factored optimizer, freeze+cache text encoders, block swap) was largely already
present; the blockers were bugs and environment issues, not missing features.

## Root causes found (and fixed)

1. **`expandable_segments:True` is toxic on WSL2.** PyTorch's expandable-segments allocator uses the
   CUDA VMM (`cuMemMap`/`cuMemSetAccess`), which raises `CUDA driver error: device not ready` when
   cuDNN allocates convolution workspace — so any conv model (SDXL UNet) crashes early in the
   backward. Transformer-only models (Cosmos DiT) don't hit it, which made it look model-specific.
   This was the real reason SDXL training "didn't work", not versions or gradient_release.
   - Fix: `rengu_flow.platform_compat.configure_cuda_allocator()` forces `expandable_segments:False`
     on WSL (only) before torch is imported (called from `rengu_flow/__init__`), preserving other
     allocator knobs. `scripts/lib/smoke_common.sh` and the docs are WSL-aware too.

2. **Version drift.** The repo pinned torch 2.10 but resolved deepspeed 0.19, which requires torch
   ≥ 2.11 and silently skipped its compiled ops. Bumped to **torch 2.12.0+cu130 / torchvision
   0.27.0+cu130 / deepspeed 0.19.0** and regenerated `uv.lock`. (The old "2.12 has a broken nvshmem
   closure" comment was stale; the lock resolves clean.)

3. **Full-model save path was never exercised.** With `freeze_text_encoders` + `cache_text_embeddings`
   the text encoders are not in the trained state dict, but `SDXLPipeline.save_model` assumed they
   were. Fix: source frozen TE/VAE weights from the live modules, and keep them on **CPU** (not
   `meta`) for full-model SDXL in `DatasetManager.cache` so the read succeeds.

4. **Block swap was unwired for SDXL.** `SDXLPipeline.to_layers()` flattens each UNet block into
   several pipeline layers, so the block's `forward` (where the old offloader's wait/submit lived)
   never ran — the offloader was dead code, and it was gated adapter-only besides.

## Design decisions

- **Block swap for full fine-tuning** is implemented with a new hook-based offloader
  (`HookBlockSwapOffloader`): forward-pre / full-backward-pre hooks on each block's leaf modules pull
  the block to the GPU on demand (covering activation-checkpointing recompute), LRU-evicting to keep
  `num_blocks - blocks_to_swap` resident. It is **gated to `optimizer.gradient_release`** for
  full-model training so the per-parameter optimizer step runs while the block is resident.
- **Placement / no load spike:** with block swap, `main.py` no-ops `PipelineModule.to` and (single
  rank only, `WORLD_SIZE==1`) DeepSpeed's `_broadcast_model`, so the full UNet is never hauled onto
  the GPU at init (`from_single_file` loads to CPU). After `deepspeed.initialize`,
  `prepare_block_swap_training()` places only the small non-swappable UNet parts on the GPU, pushes
  blocks to CPU, and `empty_cache()`.
- **Optimal config on 8 GB WSL** is `blocks_to_swap=6` (1 resident block, ~4.3 GB, ~11–15 s/step) —
  staying well under the WSL2 sysmem-paging threshold (~6 GB) matters more than transfer cost. See
  the measured curve in [vram-optimization.md](../developer/vram-optimization.md).
- **Overlapped prefetch** (`block_swap_prefetch`, pinned + side-stream) is implemented but **off by
  default**: on 8 GB WSL the ≥2 resident blocks it needs push back over the paging threshold, making
  it *slower*. It should help where there's VRAM headroom or no sysmem paging (native Linux / bigger
  GPU).

## WSL-specific behavior (conditioned on `platform_compat.IS_WSL`)

These exist because WSL2/WDDM differs from the Linux target; on native Linux the normal path runs:

- **`configure_cuda_allocator`** only rewrites `PYTORCH_CUDA_ALLOC_CONF` on WSL (forces
  `expandable_segments:False` + low-fragmentation defaults). On Linux, `expandable_segments:True`
  (as the example configs suggest) is fine and beneficial — left untouched.
- **`smoke_common.sh`** only avoids `expandable_segments:True` on WSL.

Not WSL-specific (apply everywhere, single-GPU): the block-swap orchestration (offloader, the
`PipelineModule.to` / `_broadcast_model` no-ops). They are correct for single-rank runs on Linux too,
but see below.

## Validation status

**Validated on WSL2 / 8 GB (this work).** Both models, all three modes, 3-step smokes (process
`cuda_peak`):

| | LoRA | LoRA + block swap | full finetune + block swap |
|---|---|---|---|
| **SDXL** (256px) | ✓ | ✓ | ✓ 4.32 GB, writes complete `model.safetensors` |
| **Cosmos** (512px) | ✓ 4.98 GB | ✓ 1.65 GB | ✓ 1.74 GB |

Block swap is now model-agnostic: `enable_block_swap` / `prepare_block_swap_training` /
`_place_for_block_swap` live in `base.py`; models declare only `get_block_swap_modules` +
`_block_swap_root_modules`. Two correctness fixes landed for adapter + Cosmos paths:
- The offloader keeps trainable params GPU-resident unless `gradient_release` (otherwise DeepSpeed's
  end-of-step grad reduction hits CPU grads) — needed for LoRA/LoKr block swap.
- `_place_for_block_swap` works at the tensor level so PEFT-wrapped models (SDXL LoRA wraps the UNet
  in a `PeftModel`) place their non-block tensors correctly.
- gradient_release registers hooks only on params that got an optimizer (skips frozen `lr=0` groups,
  e.g. Cosmos `llm_adapter_lr=0`).
Unit tests cover `configure_cuda_allocator`, the offloader (sync + swap_trainable), and placement.

**Needs validation on native Linux (handing off):**
- The whole low-VRAM full-finetune path on a real Linux GPU (no WSL2 sysmem paging) — expected
  faster; the `blocks_to_swap` sweet spot will differ (more headroom).
- **`block_swap_prefetch`** — expected to *help* on Linux / GPUs with headroom (opposite of the WSL
  result); needs measurement and the GPU-buffer-reuse refinement (current impl allocates fresh GPU
  tensors per pull → allocator churn).
- **Multi-GPU** (`pipeline_stages > 1`, data-parallel `WORLD_SIZE > 1`): block swap requires
  `pipeline_stages = 1` and full-model swap requires `gradient_release` (which requires DP=1), so
  full-model block swap is single-process by construction. The `_broadcast_model` no-op is gated to
  `WORLD_SIZE==1`; multi-GPU adapter block swap is untested.
- DeepSpeed `_broadcast_model` / `PipelineModule.to` no-ops under a non-WSL DeepSpeed build.

## Open work

- GPU-buffer-reuse + proper overlap for `block_swap_prefetch` (avoid per-pull allocation churn);
  validate on Linux/large GPU.
- Consider a `cuda_memory_fraction` knob (`set_per_process_memory_fraction`) as a WSL safety net to
  turn slow sysmem spill into a clean OOM — researched, not currently wired (it does not help the
  ~4.3 GB case and the constant ~0.1 GB shared baseline is an unavoidable WDDM context artifact).

## References

- [developer/vram-optimization.md](../developer/vram-optimization.md) — lever guide + measured curve.
- [developer/full-model-training.md](../developer/full-model-training.md),
  [developer/training-techniques.md](../developer/training-techniques.md) — block swap internals.
- [user/full-model-training-sdxl.md](../user/full-model-training-sdxl.md) — user guide.
