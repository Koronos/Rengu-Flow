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
