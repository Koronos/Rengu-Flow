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

### ↩ SAC — `activation_checkpointing = "selective"` (retired 2026-06; re-tested and re-buried 2026-06-11)
- **What:** Selective Activation Checkpointing: non-reentrant checkpoint with a `MUST_SAVE` policy for SDPA ops (+ `selective_checkpoint_save_ops` to keep more), recompute the rest.
- **Why superseded:** `activation_checkpointing = "auto"` (Inductor memory-budget partitioner, `training/activation_budget.py`) measured **better on BOTH axes** @1024 LoKr: SAC 0.932 s / 6.56 GB vs auto(0.1) 0.881 s / 6.37 GB — and auto scales to −20.6% at budget 0.5. SAC's only niche was compile-off runs (−4.3% for +0.8 GB), too thin to keep a second policy, config key, and UI field. Legacy configs degrade to `true` with a warning.
- **Re-test (2026-06-11):** resurrected on `worktree-activation-offload` and A/B'd against auto under
  the full production config (live TE, per-res batches {2,2,1}, resolution schedule, ar buckets,
  previews, dynamic compile; 261-step harness). SAC 571/1043/1000 ms (512/768/1024) and tuned SAC
  (+mm,addmm,bmm, dial verified applied) 567/1034/991 vs **auto(0.22) 509/928/892 — auto wins all
  three resolutions by ~11%**. SAC's only edge: steady peak 8.46 vs 10.85 GB — and auto buys the
  same VRAM with a lower budget (it is a continuous dial over the same trade). Verdict unchanged.
- **Where the code was:** `main.py` AC block (removed; git history has it, plus the re-test resurrection on `worktree-activation-offload`).
- **Revisit when:** a supported model can't compile at all (auto requires `compile = true`).

### ↩ unsloth-style CPU-offload checkpointing — `activation_checkpointing = "unsloth"` (retired 2026-06)
- **What:** custom autograd.Function offloading each block's input hidden_states to CPU during forward, restoring on backward (`utils/unsloth_utils.py`).
- **Why superseded:** measured **+2.6% step time for only −0.5 GB** vs full checkpointing @1024 (non-pinned, effectively synchronous transfers). Dominated by `true` (simpler) and `auto` (faster).
- **Where the code was:** `rengu_flow/utils/unsloth_utils.py` (deleted; git history has it).
- **Revisit when:** never as-is; an *async pinned-buffer* offload overlapping PCIe with compute would be a different experiment. *(Update 2026-06: that experiment was run — see "Async pinned-buffer activation offload" below. Also rejected.)*

### ↩ Per-shape activation-budget scaling — `scale_budget_for_area` (retired 2026-06)
- **What:** with `activation_checkpointing="auto"` and multi-resolution buckets, scale the
  configured budget up for small shapes (toward 1.0 at "constant peak") by latent area, applied by
  the dataloader just before each shape's first compile.
- **Why retired:** two field bombs. (1) Under `compile_dynamic` ONE graph serves every shape and
  the partitioner reads the budget once — at the first shape's compile — so the scaling **baked a
  small bucket's ~1.0 budget into the graph and the largest bucket OOMed at any configured base**
  (reported in production on a 512-first schedule @1024). (2) The area math ignored the batch
  dimension, so a per-resolution `micro_batch_size_per_gpu` dict turned the scaled budget into an
  OOM trap (fixed batch-aware first, then the whole feature retired). Superseded by: **global
  budget** (one value, one meaning, both compile modes) + **`BudgetBackoff`** (on CUDA OOM, lower
  the budget and recompile instead of crashing — `activation_budget_backoff`, default on).
- **Where the code was:** `training/activation_budget.py` + loader/main wiring (git history).
- **Revisit when:** never as-is. If small-shape recompute overhead ever matters again, it must be
  static-compile-only and batch-aware by construction — and prove it can't out-OOM the base.

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

### ⛔ Trim Cosmos text padding below 512 — *(tried in-place 2026-06-11, reverted; no branch)*
- **What:** `tokenize()` pads every caption to a fixed 512 tokens (`model/cosmos_predict2/text.py`),
  so the live TE (Qwen 0.6B) processes ~3-6× padding for typical 80-150-token captions and the DiT
  cross-attention carries the full 512 K/V context every step. Probe: env-gated bucketed padding
  (longest-in-batch rounded up to 64, cap 512), expected to recover most of the ~35 ms/step live-TE
  cost plus a slice of DiT crossattn.
- **Why rejected:** **the model depends on the padding**. The DiT cross-attention applies NO mask
  (`attn_mask_type="no_mask"` for TE; plain `scaled_dot_product_attention(q, k, v)` for torch —
  `dit.py`), so the ~400 zeroed positions act as trained-in attention sinks. Equivalence probe
  (same seed, no dropout, eager, 10 steps): |Δloss| up to 0.053 vs a 0.0013 bf16 noise floor —
  **39× above noise and systematically higher loss on every step**. Trimming is out-of-distribution
  input, not an optimization.
- **Revisit when:** training a model whose checkpoints were trained with masked/variable-length
  text context, or if a finetune deliberately re-adapts the model to short contexts (measure loss
  parity first, exactly like this probe).
- **Follow-up (same day): exact variant also rejected — zero gain.** Trimming only the *encoder
  compute* (bucketed tokenize for the Qwen side, zero-pad the embedding + mask back to 512 in
  InitialLayer; t5/DiT context untouched) IS mathematically equivalent — fp32 probe: max|Δ|=1e-4 on
  embeddings of magnitude ~63 (padding side is right; the encoder masks padding; padded outputs are
  zeroed). But the canonical-harness A/B measured **no speedup at all** (510/930/900 vs 509/928/892
  ms): a direct CUDA-event bench shows the live TE forward costs **~22 ms at ANY length** (21.8 ms
  @512 vs 22.2 ms @128, bs2) — it is kernel-launch/overhead bound, not token bound. Reverted.
  The real levers for the ~22-30 ms live-TE cost are: (a) pre-cached caption variants (already
  supported: one caption per .txt line + cache_text_embeddings=true — embeddings cached per
  caption_number, TE leaves the GPU, budget headroom returns to 0.3), or (b) cutting launch
  overhead itself (e.g. compiling the TE with mode="reduce-overhead"/CUDA graphs) — unmeasured.

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
