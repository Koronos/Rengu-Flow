# DiT training-speed research (Anima/Cosmos LoKr on RTX 4080) — living doc

> Why this exists: the optimizer (kaon) is maxed out (<1% of the step; making it 5× changed real
> iter_sec ~0%). The bottleneck is the DiT training loop. This is the running record of what we
> measured, what the levers are, and the open experiment. Keep it updated.

## Step X-ray — the dataloader cost when prefetch is OFF (MEASURED 2026-06-09)

> Context: `dataloader_prefetch` already existed and was already in use; this section just quantifies
> what it saves (the cost of running with it OFF). It is now the origin default (`ad7f48d`), so for
> current runs this is already handled — not a new lever.


Wall-clock X-ray of the real step (measured with temporary per-step instrumentation, since removed;
AdaPNM-fused LoKr, compile+dynamic, SAC, micro_batch 2, Torino set cached). The bench `iter_sec` only
times `model_engine.train_batch` (main.py t0) — but **the dataloader preload
(`get_data_iterator_for_step`) runs BEFORE t0**, so its cost is INVISIBLE in every bench number we'd
ever looked at. Splitting the step:

| res | t_data (dataloader) | t_compute (GPU 100%) | total | **fixed: `dataloader_prefetch=true`** | speedup |
|---|---|---|---|---|---|
| 512  | 66.7 ms (11.5%) | 503.5 ms | 581.5 ms | **475.7 ms** | **18.2%** |
| 768  | 69.5 ms (6.8%)  | 946.5 ms | 1015.0 ms | **932.8 ms** | **8.1%** |
| 1024 | 71.6 ms (4.0%)  | 1700.8 ms | 1773.4 ms | **1699.8 ms** | **4.2%** |

- **`t_data` is a ~FIXED ~67–72 ms/step** (does NOT scale with resolution → it's per-step iter+collate+
  H2D+eCryptfs-decrypt of the 2 cached latents/embeddings, not bandwidth). At `dataloader_num_workers=0`
  + `dataloader_prefetch=false` (the renga default AND the user's real config) it runs synchronously on
  the main thread → pure GPU-idle stall, and it even slowed `t_compute` ~6% (main thread contending with
  kernel launches).
- **Fix = overlap the feed with compute.** `dataloader_prefetch=true` (thread, workers=0) OR
  `dataloader_num_workers≥1` both collapse t_data 67→~1.5 ms AND recover the t_compute contention.
  Measured identical (512: prefetch 475.7 ms vs workers=4 477.7 ms). Thread-prefetch is simplest (no
  worker fork/shm). **Zero quality impact** — same data, just prefetched.
- **~9% wall-clock blended** (NOTE: the real schedule mixes res per stage and 1024 dominates ~60%, so the
  blend skews toward the 4% end; per-res 18/8/4% hold). `t_compute` is 100% GPU-active at all 3 res → the
  compute is saturated; the dataloader was the only slack, and only when prefetch is off.
- **Already handled:** `dataloader_prefetch=true` is the origin default (`ad7f48d`) and was already in
  use; `tag_dropout` was also wired into caption sampling (`8b1a88d`). Nothing to change for current runs.

### Multi-res + varied-AR is CLEAN — `compile_dynamic` confirmed (wlop set2, 295 imgs, ~15 ARs, 2026-06-09)
X-ray'd the realistic varied dataset with all 3 resolutions live at once (no schedule, prefetch on,
tag_dropout 0.3): 70 steps, resolution changing every step.
- **`compile_dynamic` = ONE cold compile (step 1 = 57 s) then ZERO recompiles** across 512/768/1024 ×
  ~15 aspect ratios. The varied-AR recompile-thrashing worry is a non-issue; the 57 s amortizes to
  nothing over a real run (15 epochs × ~148 steps). Per-res steady (483/954/1746 ms for 512/768/1024)
  matches the pinned single-res numbers → cost generalizes.
- **Prefetch hides the dataloader even with tag_dropout + varied ARs** (t_data ≤4 ms after step 1).
- **Every resolution is 100% GPU-compute-bound** → the loop has no remaining slack on this hardware.
  Bottom line for the real config: one-time compile + prefetch-hidden data + compute at the floor. Done.

## Per-step cost breakdown (MEASURED — `RENGU_PROF_DIR`, 1024px, batch 2, compile-off, SAC)

Attributable self-CUDA %:

| group | % of CUDA | what |
|---|---|---|
| **GEMM (`aten::mm`)** | **44%** | frozen base linears (QKV, attn-out, MLP up/down, adaLN) fwd+bwd **+ LoKr adapter matmuls**. Compute-bound bf16 tensor-core (cutlass_80 `s16816`/`s1688` gemm kernels). |
| pointwise unfused (mul/copy/add/pow/gelu/cat) | ~38% | **fused away by `torch.compile`** → this is why compile gives ~1.5×. |
| attention (flash fwd+bwd) | ~14% | FA2 (TE dispatches FA2 on Ada). |
| optimizer (AdaPNM) | <3% | kaon; isolated bench shows 0.3.0-fused = 2.4× faster than old-fused on the real LoKr set (28.7→11.9 ms), but it's <1% of the step → invisible in iter_sec. |
| layer_norm | ~3% | |

**How to reproduce:** `RENGU_PROF_DIR=/path RENGU_PROF_WAIT=12 RENGU_PROF_WARMUP=3 RENGU_PROF_ACTIVE=6 ./rengu train --config <cfg> -- --trust_cache`. Reads `kernels_self_cuda.txt`. CAVEAT: `torch.profiler`'s `record_function` mis-attributes async work (the optimizer.step line absorbed a 54s cuDNN-autotune spike → 340% bogus). For optimizer/sub-step costs use an isolated micro-bench, not the profiler. Full-step iter_sec has ~3% run-to-run variance → can't resolve sub-3% deltas.

## Levers — status

### Already on / confirmed
- **`dataloader_prefetch = true`** (or `dataloader_num_workers≥1`) — MEASURED 18/8/4% @512/768/1024
  (blended ~9%) vs running with it off. Origin default (`ad7f48d`) and already in use — already handled.
- **`torch.compile` + `compile_dynamic=true`** — MEASURED ~1.5× (1024px 1.28→0.99 on a synthetic dataset; ~1.5× on the real config too). dynamic is load-bearing with AR-buckets (static recompiles per AR shape). Eats the ~38% unfused-pointwise. KEEP.
- **Selective activation checkpointing (SAC)** — shipped; ~4% over full-ckpt; VRAM 9.5-10.3GB at batch 2/1024 (fits 16GB).
- **FA2 attention** (via Transformer-Engine `DotProductAttention`).

### INT8 mixed-precision TRAINING (torchao) — ⛔ MEASURED NEGATIVE on the 4080 (2026-06-09)
The GEMM is compute-bound bf16 tensor-core work → only two physical levers: fewer FLOPs (token
reduction = quality risk) or faster (lower-precision) tensor cores. fp8 already FAILED here (e4m3
weights outlier-unsafe). int8 mixed-precision training (`torchao.prototype.Int8MixedPrecisionTrainingConfig`,
PR #748) was the last credible GEMM lever — measured ~70% e2e on a 4090 in the torchao docs. **It does
NOT reproduce on this 4080. Three-layer evidence (`tmp/bench_int8mpt*.py`, `bench_rawint8.py`):**
1. **End-to-end (Linear fwd+bwd, torch.compiled, M=8192 DiT shapes):** int8-mpt is **0.70–0.85×** (i.e.
   15–30% *slower*) than bf16 — both for the frozen config (`grad_weight=False`) AND the full config
   (`grad_weight=True`, the exact config that got 70% on the 4090). Not a config issue.
2. **Profiler proof of *why*:** under `torch.compile`, the int8-mpt path emits the **same
   `cutlass_80_tensorop_bf16_s16816gemm` (BF16) kernels** as plain bf16 — NO int8 GEMM kernel is
   generated — *plus* extra elementwise quant kernels. So it pays the rowwise-quant overhead for zero
   int8 acceleration. inductor/torchao chose bf16 because int8 isn't a win on this hardware.
3. **Hardware-floor proof (decisive):** raw `torch._int_mm` (int8 tensor-core GEMM, **zero** quant
   overhead) vs `torch.mm` bf16: **0.84–0.95×** across all DiT shapes (8192×2048×2048 → 16384×4096×4096).
   The int8 matmul *itself* loses to bf16. Reason: the Ada bf16 cutlass `s16816` (fp32-accum) kernels
   are extremely well tuned; the int8 path (cuBLAS `_int_mm`, int32-accum) never reaches its 2× peak
   for these shapes. With the honest rowwise-quant cost folded in, effective int8 = **0.60–0.80×**.

**Verdict: there is no accessible int8 GEMM win on the 4080.** A hand-tuned Triton int8 GEMM is the only
remaining path and even a perfect 2×-peak kernel would give ≤1.2× e2e (GEMM is 44% of the step) at
best — wildly optimistic given the practical int8 path is already <1× — plus int8-training quality risk.
Not worth it. **The bf16 cutlass GEMM IS the floor on this hardware; the original "GEMM is hard to
optimize" was correct, now measured.** The remaining levers are NOT kernels (see below).

### Secondary (cheap, modest)
- `max-autotune` / `coordinate_descent_tuning` — ~2–10%, fights `compile_dynamic` (per-shape autotune).
- **Fused-QKV — ⛔ MEASURED ~0.5% e2e, not worth it (2026-06-09).** Cosmos uses SEPARATE `q_proj`/`k_proj`/`v_proj`
  (`dit.py:368-374`), so fusing is *applicable*, but: bench of 3×`Linear(2048,2048)` vs 1×`Linear(2048,6144)`
  at the real self-attn shape (M=8192, fwd+bwd, frozen, compiled) = **1.04× / 0.21ms saved per block**
  (eager = 1.00×, dead-even — these GEMMs are compute-bound at M=8192, fusion only saves launch overhead
  compile already hides). ×28 blocks ≈ 5.8ms on a ~1000ms 1024px step = **~0.5%**. Cost to capture it:
  breaks LoKr adapter targeting (config keys `q_proj`/`k_proj`/`v_proj` by name → re-key = checkpoint-incompat),
  needs fuse-at-load/unfuse-at-save of the pretrained base, and edits frozen-base model code. Skip.
- CUDA-graph / `reduce-overhead` — needs a static-shape region; blocked by multi-res dynamic shapes.

### Rejected / inapplicable (measured or structural)
- **fp8 matmul** (torch._scaled_mm) — ~70% slower on Ada (e4m3 weights Cosmos-unsafe). FA3 — Hopper-only.
- **2:4 sparsity** — ≤1.3× real on Ada + must prune the frozen base (quality risk). Skip.
- **bitsandbytes Linear8bitLt** — slower than bf16 for training. **int8_weight_only** — memory-only, no compute speedup.
- **MaskDiT / REPA / REG / HASTE** — pretraining/architecture recipes; incompatible with a frozen base.
- **Token merging (ToMe/ToMA)** — degrades SDXL-class quality; only at a measured-FID-neutral low ratio.
- **GaLore / bf16x9** — orthogonal (optimizer state) / null.

### The dominant lever is strategic, not a kernel
A 1024px step costs ~3.3× a 512px step (1.73 vs 0.52 ms in one measurement). **How much you train at
1024 vs 512/768 (the `resolution_schedule`) moves wall-clock more than any kernel.**

## CONCLUDED experiment — int8 mixed-precision training (2026-06-09)

**Step 1 (feasibility gate) FAILED → no Step 2.** Gate was >1.3× matmul; measured **0.70–0.95×**
(int8 is slower on the 4080). Three-layer evidence above. The lever is dead on this hardware. No renga
change made. See the ⛔ section above and `docs/EXPERIMENTS_GRAVEYARD.md`.

### What's actually left (no kernel free lunch remains)
After int8 (now), fp8 (earlier), 2:4 sparsity, token-merging — every *kernel* GEMM lever on the 4080 is
exhausted or quality-unsafe. The throughput levers that remain are NOT kernels:
1. **`torch.compile` (~1.5×)** — already the biggest win; keep `compile=true, compile_dynamic=true`.
2. **Strategic resolution budget** — the `resolution_schedule` (1024 vs 512/768 share) moves wall-clock
   more than any kernel could. A 1024px step ≈ 3.3× a 512px step.
3. **SAC (~4%)** — already shipped.
The optimizer (kaon) is also maxed. The honest state: the loop is near its hardware floor on the 4080.

Sources: torchao PR #748 (https://github.com/pytorch/ao/pull/748); torchao training docs
(https://docs.pytorch.org/ao/stable/workflows/training.html); PyTorch native architecture optimization
blog (https://pytorch.org/blog/pytorch-native-architecture-optimization/); torchtitan #578. (Full
source list in the chat research.)
