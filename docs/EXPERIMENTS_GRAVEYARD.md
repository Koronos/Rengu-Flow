# Experiments graveyard — rejected / not-merged attempts (renga-flow)

> Why this file exists: so an agent (or human) does **not** re-try or accidentally merge an idea
> that was already implemented and **measured** to lose. Every entry is a real attempt with code on
> a branch (or tried in-place and reverted). Each says **what**, **why it lost**, **where the code
> is**, and **when to revisit**. "Rejected" here means *measured net-negative for our target*
> (Cosmos Predict2 LoKr, RTX 4080 16GB, multi-res bf16, quality-first) — not "bad in general".
>
> Optimizer experiments live in the **kaon** repo's graveyard, not here.

Legend: ⛔ REJECTED (measured net-negative) · ⏸ PARKED (dead here, viable on other hardware) · ↩ SUPERSEDED

---

## ⛔ / ⏸ Speed & memory levers

### ↩ SAC — `activation_checkpointing = "selective"` (retired 2026-06)
- **What:** Selective Activation Checkpointing: non-reentrant checkpoint with a `MUST_SAVE` policy for SDPA ops (+ `selective_checkpoint_save_ops` to keep more), recompute the rest.
- **Why superseded:** `activation_checkpointing = "auto"` (Inductor memory-budget partitioner, `training/activation_budget.py`) measured **better on BOTH axes** @1024 LoKr: SAC 0.932 s / 6.56 GB vs auto(0.1) 0.881 s / 6.37 GB — and auto scales to −20.6% at budget 0.5. SAC's only niche was compile-off runs (−4.3% for +0.8 GB), too thin to keep a second policy, config key, and UI field. Legacy configs degrade to `true` with a warning.
- **Where the code was:** `main.py` AC block (removed; git history has it).
- **Revisit when:** a supported model can't compile at all (auto requires `compile = true`).

### ↩ unsloth-style CPU-offload checkpointing — `activation_checkpointing = "unsloth"` (retired 2026-06)
- **What:** custom autograd.Function offloading each block's input hidden_states to CPU during forward, restoring on backward (`utils/unsloth_utils.py`).
- **Why superseded:** measured **+2.6% step time for only −0.5 GB** vs full checkpointing @1024 (non-pinned, effectively synchronous transfers). Dominated by `true` (simpler) and `auto` (faster).
- **Where the code was:** `rengu_flow/utils/unsloth_utils.py` (deleted; git history has it).
- **Revisit when:** never as-is; an *async pinned-buffer* offload overlapping PCIe with compute would be a different experiment. *(Update 2026-06: that experiment was run — see "Async pinned-buffer activation offload" below. Also rejected.)*

### ⛔ Async pinned-buffer activation offload — `activation_offload` (branch `worktree-activation-offload`)
- **What:** the "different experiment" the unsloth entry pointed at, done properly:
  `saved_tensors_hooks` streaming each large contiguous saved activation to a pooled **pinned** CPU
  buffer on a side CUDA stream during the forward (GPU storage released on the GPU timeline via
  `record_stream`), reverse-order prefetched H2D on a second stream during the backward. Composes
  with `activation_memory_budget` (budget picks save-vs-recompute, offloader moves the saved share).
- **Why rejected** (Cosmos LoKr @1024, 24-step bench, steady per-step peaks, PCIe 4.0 x16 measured
  25.5 GB/s/dir):
  - budget 0.3 + offload: 0.830 s / **8.77 GB** vs plain 0.3 at 0.810 s / 8.97 GB → **−0.2 GB for
    +2.5 % time**, strictly dominated by just lowering the budget (0.1 → 6.24 GB @ 0.854 s).
  - budget 0.5 + offload: ~1.03 s / 11.03 GB vs 0.748 s / 11.25 GB → **+35 % time for −0.2 GB**
    (bus-saturated: 5.21 GB/step each way no longer fits the step).
  - budget 1.0 + offload: **still OOM** — the cold/compile step can't stream past its own peak, so
    no previously-impossible operating point is unlocked.
  - Root causes: at useful budgets the partitioner's saved set is dominated by (a) residual-stream
    tensors the ongoing forward keeps alive anyway (offload copies them but frees nothing) and
    (b) **non-contiguous views** (fused-QKV chunks) that cannot round-trip (the compiled backward
    asserts exact sizes/strides) — the *freeable* share measured ~0.2 GB of an 8.97 GB peak; where
    the saved set IS large (high budgets), its volume exceeds what the bus can drain inside a step.
- **Hard-won facts for whoever revisits:** hooks DO fire under torch.compile (AOT routes compiled
  saves through them); side-stream staging tensors must be allocated *inside* the owning stream's
  context (cross-stream allocator race → NaN losses otherwise); frees must happen on the GPU
  timeline (`record_stream`) because under compile Python runs a full forward ahead of the GPU, so
  Python-side frees hold the whole saved set until backward.
- **Where the code is:** branch `worktree-activation-offload` —
  `rengu_flow/training/activation_offload.py` + tests + main-loop wiring, off by default.
- **Revisit when:** a host link ≥4× PCIe 4.0 (NVLink/Grace class), or a model whose saved set is
  mostly large contiguous tensors that are not alive in the residual stream.

### ⏸ fp8 matmul for the frozen DiT base — `feat/cosmos-quant`
- **What:** `torch._scaled_mm` row-wise fp8 GEMM for the ~280 frozen DiT linears (`transformer_fp8_matmul`), with a custom `_Fp8ScaledMatmul` autograd.Function (scaled_mm has no derivative) and LoKr-on-quantized-base composition.
- **Why rejected:** **~70 % SLOWER** (2.95 s vs 1.74 s/step @1024) + more VRAM. The 4080's `_scaled_mm` forces the **weight operand to e4m3** (e5m2 weight rejected) — exactly Cosmos's fp8-**unsafe** format (weight outliers > ±448). The per-step manual activation quant over 280 linears + autograd graph breaks dwarf any GEMM gain.
- **Revisit when:** Hopper-class HW (different e5m2 matmul support) or a non-fp8-sensitive model, with a *fused* fp8 path (torchao delayed scaling) — but that conflicts with the LoKr forward override here.

### ⏸ 4-bit NF4 frozen base — `feat/cosmos-quant`
- **What:** `bitsandbytes.nn.Linear4bit` for the frozen DiT (`transformer_4bit`), QLoRA-style base+Δ composition.
- **Why rejected as a *speed* lever:** 4-bit shrinks **weight** memory only; our OOM is **activation-bound**, so it can't let us drop activation checkpointing → **OOM @1024 (14.6 GB)** with AC off, and just adds dequant overhead with AC on. Conversion itself works.
- **Revisit when:** training a model too big to fit in 16 GB at all (it's a VRAM-fit lever, not a speed lever). We already fit.

### ⛔ Regional (per-block) `torch.compile` — `feat/regional-compile` (has its own `BRANCH_REJECTED_DO_NOT_MERGE.md`)
- **What:** compile each of the ~28 identical DiT blocks individually (`compile_regional`), inductor reuses one artifact.
- **Why rejected:** halves the **one-time** cold-compile spike (77→38 s) but the 28 graph-break seams cost **+2–5 %/step forever**; crossover ~1000–1600 steps → real long runs are slower overall. The "proper" `nested_compile_region` fix is also a dud here: **crashes on dynamic shapes** (torch 2.12) and gives ~no compile win on static (compile is **inductor-bound, not trace-bound**) — code removed.
- **Revisit when:** short/debug runs only (startup saving beats the per-step tax), or if torch/HW changes the compile-vs-step economics — **re-measure the crossover first.** Kept opt-in default-off.

### ⛔ Persistent inductor disk cache for dynamic shapes — *(tried in-place, reverted; no branch)*
- **What:** `TORCHINDUCTOR_FX_GRAPH_CACHE` / `AUTOGRAD_CACHE` / `CACHE_DIR` / `TRITON_CACHE_DIR` to skip recompiles across runs.
- **Why rejected:** the cache only hits on **static** shapes (static warm 72 s vs cold 217 s). Our multi-res training runs `compile_dynamic=true`, and dynamic-shape graphs **don't cache** → warm ≈ cold. Also crashes on the encrypted `/home` (eCryptfs 143-char filename limit, Errno 36) — ext4 only.
- **Kept:** the disk-cache *plumbing* (`_setup_compile_disk_cache`, default-on when `compile_dynamic=false`) is merged — it's a real win for the static case. Only the "expect it to help dynamic multi-res" hope is rejected.

### ⛔ `max-autotune` for the training compile — *(tried in-place, reverted; no branch)*
- **Why rejected:** ~0 % gain with `compile_dynamic=true`. Autotuning needs concrete shapes; dynamic shapes defeat it. Blended over the real multi-res schedule it's not worth the much longer compile.

---

## Not in this graveyard (these LANDED on `main`)
For contrast, so nobody re-litigates them: **Selective Activation Checkpointing** (`activation_checkpointing="selective"`, ~4 % @1024), the **resolution schedule / progressive-resolution curriculum**, **foreach/chunked optimizer batching**, the **val-loss train–val gap metric**, and the **AdaPNM RMS-clip divergence fix** are all merged and kept.

---

_Maintained by: update this file whenever an experiment is rejected or parked. One entry, with the measured "why". Cross-repo optimizer attempts → `K-Optimizers/docs/EXPERIMENTS_GRAVEYARD.md`._
