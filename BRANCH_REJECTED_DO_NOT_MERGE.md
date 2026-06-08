# ⛔ BRANCH REJECTED — DO NOT MERGE TO `main`

**Branch:** `feat/regional-compile`
**Status:** REJECTED (kept for reference / future hardware-software only)
**Decision date:** 2026-06-08

This branch adds `compile_regional` (per-block `torch.compile`). It is **measured net-negative for production** and must **not** be merged or enabled by default. It is kept on this branch only as a documented, opt-in (`compile_regional=true`, default off) tool for short/debug runs.

## Why rejected (all MEASURED on the real target: Cosmos LoKr, RTX 4080 16GB, multi-res, `compile_dynamic=true`)

### Naive per-block compile (`compile_regional = true`)
Compiles each of the ~28 identical DiT blocks separately → inductor compiles one and reuses it → **halves the one-time cold-compile spike**, but the 28 graph-break seams cost **more per step, forever**.

| metric | whole-model (main) | regional naive | Δ |
|---|---|---|---|
| cold compile (step 1+2) | 77.4 s | 38.1 s | **−51%** (saves ~39 s, once) |
| 512 steady | 0.5111 s | 0.5356 s | **+4.8% / step** |
| 1024 steady | 1.8268 s | 1.8649 s | **+2.1% / step** |

**Crossover** (one-time saving eaten by the per-step tax): ~1030 steps @1024, ~1600 @512, ~1200 mixed. Real training runs are thousands+ of steps → **net slower overall.** Loss is numerically identical (not a quality issue, a wall-clock one).

### "Proper" fix — `nested_compile_region` (tried, also a dud, code removed)
Idea: keep one whole-model graph, mark the repeated block so dynamo dedups it (no eager seams). Two independent walls:
1. **Crashes with dynamic shapes** (`InternalTorchDynamoError: KeyError: s<N>`, torch 2.12), with or without activation checkpointing. Multi-res needs dynamic → unusable.
2. On static it runs but cuts cold compile only **−6%** (35.0 s vs 37.2 s) because this model's compile is **inductor/kernel-codegen-bound, not trace-bound** — the wrong dedup for the bottleneck. (Naive on static = −78%, 8.3 s, +0.5% steady; but static already gets disk-cache warm-start.)

A future torch fixing the dynamic crash still would not make nested useful here — the reason is structural.

## If you are an agent considering this branch
- **Do NOT merge to `main`.** Do NOT set `compile_regional` default-true anywhere.
- Only revisit if: (a) target is short/debug runs where the ~39 s startup saving beats the per-step tax, or (b) hardware/torch changes the compile-vs-step economics — and **re-measure** the crossover before acting.
- Full context: kaon memory `regional-compile-tradeoff.md`, and `cosmos-lokr-step-profile.md` / `dit-training-speedup-research.md`.
