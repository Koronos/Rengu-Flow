# VRAM optimization (low-VRAM training)

How to fit training — especially **full-model SDXL fine-tuning** — on small GPUs, which levers exist,
how they interact, and their trade-offs. Numbers below were measured on an **8 GB RTX 3000 Ada
laptop (WSL2)** training the full SDXL UNet at 256px, batch 1, bf16. Use them as ratios, not
guarantees; absolute values depend on resolution, batch size, and GPU.

> Adapters (LoRA/LoKr) are the cheapest way to train SDXL — the base is frozen, so gradients and
> optimizer state exist only for the small adapter. This page is about the harder case: **full
> fine-tuning**, where every UNet weight is trained. See
> [full-model-training-sdxl.md](../user/full-model-training-sdxl.md) for the user guide.

## The budget problem (full SDXL UNet ≈ 2.6 B params)

| Component | Naïve (fp32 Adam) | With the levers below |
|-----------|-------------------|------------------------|
| Weights | 10.3 GB (fp32) | **5.1 GB** (bf16) |
| Gradients | 10.3 GB (all resident) | **~0** (freed per-param) |
| Optimizer state | ~21 GB (Adam m+v fp32) | **~tens of MB** (factored) or on CPU |
| Master weights | 10.3 GB (fp32 copy) | **0** (bf16 + Kahan, no copy) |
| Text encoders + VAE | resident | **off-GPU** (frozen + cached) |
| Resident UNet weights | 5.1 GB | **~1 block** with block swap |

Naïvely that's ~50 GB+. The levers bring batch-1 256px full SDXL fine-tuning into **~4.3 GB**.

## The levers

### 1. `dtype = "bfloat16"` + Kahan summation
Train weights in bf16 (half of fp32) with **no fp32 master copy**. The naïve danger is that
`bf16(weight + tiny_update)` truncates small updates to zero; the optimizers compensate with **Kahan
summation** (`GenericOptim`, `AdamW8bitKahan`) or stochastic rounding, so a bf16-only master stays
trainable. *Side effect:* very slightly noisier updates than fp32; negligible in practice.

### 2. `activation_checkpointing = true` (or `"unsloth"`)
Recompute activations in the backward instead of storing them. Big activation-memory cut. *Side
effect:* ~20-30% more compute (one extra forward). Essentially mandatory at low VRAM. The `unsloth`
variant uses a reentrant checkpoint; `true` uses the non-reentrant torch checkpoint.

### 3. `optimizer.gradient_release = true` (fused backward)
Runs each parameter's optimizer step **inside the backward** (`register_post_accumulate_grad_hook`)
and frees that gradient immediately, so the full-model gradient tensor is never resident at once —
the single biggest lever for full fine-tuning (gradients are as large as the model). *Side effects:*
incompatible with `gradient_accumulation_steps > 1`; requires data-parallel world size 1
(`pipeline_stages = num_gpus`); gradient clipping is disabled (no grads remain at step end). It is
**required** for full-model block swap (see below).

### 4. A memory-frugal optimizer
The optimizer state, not the weights, dominates naïve full fine-tuning. Options (smaller → larger
state): **Adafactor** (factored 2nd moment, no momentum → ~tens of MB), `GenericOptim`
(`second_moment_type="factored"`, `momentum_type="none"`, optional `cpu_offload`/`kahan_buffer_offload`
to push state to CPU RAM), **`AdamW8bitKahan`** (8-bit state, ~4× smaller than fp32 Adam). *Side
effects:* Adafactor/factored converge a little differently than Adam; `cpu_offload` adds CPU↔GPU
traffic per step. Note: a normal `fused=True` AdamW is fast but needs full fp32 m+v (~21 GB) — not
an option here.

### 5. Freeze + cache the text encoders, cache VAE latents
`model.freeze_text_encoders = true` trains the UNet only; `model.cache_text_embeddings = true`
pre-encodes captions so the two CLIP text encoders leave the training graph entirely. Latents are
cached too, so the VAE isn't resident during training. *Side effects:* the text encoders aren't
adapted (UNet-only training); incompatible with caption dropout / on-the-fly augmentation that would
change embeddings per step. The full checkpoint still includes the (frozen) TE/VAE weights —
`save_model` sources them from the live modules, which is why
[`DatasetManager.cache`](../../rengu_flow/data/manager.py) keeps them on CPU (not `meta`) for
full-model SDXL.

*Training the text encoders too* (`freeze_text_encoders = false`, `cache_text_embeddings = false`):
supported — they then run live each step and are placed on the GPU by block swap
(`_block_swap_root_modules` includes them when uncached) and trained via `gradient_release`. On 8 GB
this **fits but is tight** (the two CLIP encoders add ~1.6 GB resident, pushing near the WSL2 sysmem
threshold → slower, erratic steps). Prefer UNet-only (freeze + cache) for speed on small cards; TE
training has comfortable headroom on bigger GPUs (or, future, by block-swapping the TE layers too).

### 6. `blocks_to_swap = N` (block swap)
Stream blocks between CPU RAM and the GPU so only a few are resident
([`HookBlockSwapOffloader`](../../rengu_flow/training/block_swap.py)). Model-agnostic: each model
declares its swappable blocks (`get_block_swap_modules`) and roots (`_block_swap_root_modules`); the
base does the rest. SDXL's UNet has **7 swappable blocks** (3 down + mid + 3 up); `blocks_to_swap=6`
keeps **1 resident**, dropping resident weights from 5.1 GB to ~1 block. Cosmos's DiT has **~28
blocks** (`transformer.blocks`) — same mechanism, swap most of them (Cosmos LoRA + heavy swap was
~1.6 GB on 8 GB). For **adapter** training the offloader keeps the small trainable params resident
and swaps only the frozen base (so DeepSpeed's end-of-step grad reduction never sees a CPU grad);
for **full-model** training it *requires `gradient_release`* (the per-parameter step runs while the
block is on the GPU; a monolithic `optimizer.step()` would need every block resident at once).
`pipeline_stages` must be 1. *Side effect:* CPU↔GPU transfer cost per step (see the curve below).

### 7. `block_swap_prefetch = true` (overlapped transfer — situational)
Opt-in: pin the swapped blocks' CPU memory and prefetch the next block on a side CUDA stream while
the current block computes, to hide transfer latency. **On the 8 GB WSL box it is counterproductive**
(see below) and is **off by default**; it's expected to help only where there is VRAM headroom for
≥2 resident blocks (bigger GPUs, or native Linux without WSL2 sysmem paging). Needs Linux/large-GPU
validation.

## How they interact

- **gradient_release + block swap are a pair** for full fine-tuning: the swap removes weights, the
  per-parameter step removes the need to have all weights resident for the update. Neither alone
  fits full SDXL on 8 GB.
- **bf16 + Kahan + factored/8-bit optimizer** together remove the fp32-master and Adam-state bloat;
  use one frugal optimizer, don't stack momentum on top of a full second moment.
- **activation_checkpointing** composes with everything and is near-mandatory.
- **freeze + cache TEs** frees room *before* the UNet levers even start; do it first.
- **block swap vs. `cuda_memory_fraction`**: they solve different things — swap *reduces* usage;
  capping the memory fraction only changes *spill into clean OOM* (relevant on WSL, see the spec).

## Measured curve (8 GB WSL2, full SDXL UNet, 256px, batch 1)

Stack: bf16 + Kahan, `activation_checkpointing`, `gradient_release`, Adafactor, freeze + cache TEs.

| Config | Resident UNet blocks | Process VRAM (torch peak) | Step time |
|--------|----------------------|---------------------------|-----------|
| No block swap | 7 / 7 | 6.86 GB (spills) | ~70 s |
| `blocks_to_swap=2` | 5 | 6.73 GB | ~34 s |
| `blocks_to_swap=4` | 3 | 6.17 GB | ~19 s |
| `blocks_to_swap=5` | 2 | 5.87 GB | ~22 s |
| **`blocks_to_swap=6`** | **1** | **4.32 GB** | **~11–15 s** |
| `blocks_to_swap=5` + prefetch | 2 | 5.89 GB | ~30–150 s (worse) |
| *(reference) LoRA, frozen base* | — | ~5 GB | ~3 s |

**The counter-intuitive bit:** more swapping is *faster* here, because the dominant cost is **WSL2
sysmem paging**, not transfer. Above ~6 GB the Windows driver silently pages VRAM into shared system
RAM (there is no working "no sysmem fallback" toggle under WSL2), and *every* CUDA kernel then runs
slowly. Staying well under that threshold (`blocks_to_swap=6`, 4.3 GB) avoids paging entirely. The
same reason makes prefetch counterproductive on 8 GB: holding 2 resident blocks to overlap transfer
pushes back over the paging threshold. On native Linux (no sysmem paging) or a larger GPU the
trade-off flips and prefetch should help — that path needs Linux validation.

## Recommended recipe for ~8 GB, full SDXL fine-tune

```toml
activation_checkpointing = true
blocks_to_swap = 6          # 1 resident block; stay well under the spill threshold

[model]
type = "sdxl"
dtype = "bfloat16"
freeze_text_encoders = true
cache_text_embeddings = true

[optimizer]
type = "transformers.optimization.Adafactor"
lr = 1.0e-5
scale_parameter = false
relative_step = false
warmup_init = false
gradient_release = true     # required with full-model block swap
```

Result: ~4.3 GB, full UNet trained, writes a complete `model.safetensors`. See the smoke fixture
[`tests/fixtures/smoke/train_sdxl_full_finetune.toml`](../../tests/fixtures/smoke/train_sdxl_full_finetune.toml).

## WSL2 caveat (read the spec)

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **crashes cuDNN convolutions on WSL2** ("CUDA
driver error: device not ready"); rengu forces it off on WSL automatically. And WSL2's "shared GPU
memory" fallback cannot be disabled from Windows. See
[spec/sdxl-full-finetune-low-vram.md](../spec/sdxl-full-finetune-low-vram.md) for the full story and
what still needs validation on native Linux.
