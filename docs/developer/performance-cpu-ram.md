# CPU, RAM, and disk cache performance

Developer notes for data loading and cache tuning. User-facing option tables live in [training-loop-and-eval](../user/training-loop-and-eval.md).

**POC benchmarks:** [poc-cpu-ram-results](poc-cpu-ram-results.md) — run `python scripts/poc_cpu_ram_optimizations.py`.

## Comparison with similar trainers

| Technique | kohya-ss | SimpleTuner | ai-toolkit | OneTrainer | renga-flow |
|-----------|----------|-------------|------------|------------|------------|
| Latents on disk | `cache_latents_to_disk` | `image_embeds` | `LatentCachingMixin` | Latent caching (data tab) | `DatasetManager.cache` → `path/cache/<model>/` |
| Train DataLoader workers | `max_data_loader_n_workers` | standard | config (0 on Win/macOS) | cache threads | `dataloader_num_workers` (default 0) |
| Prefetch / pin | ecosystem | — | no `pin_memory` ([issue #758](https://github.com/ostris/ai-toolkit/issues/758)) | pinned offload buffers | `dataloader_prefetch`, `dataloader_pin_memory` |
| Cache build parallelism | pool | `write_batch_size` | mixins | threaded cache ([issue #181](https://github.com/Nerogar/OneTrainer/issues/181)) | `cache_num_proc` + GPU queue |
| Disk cache layout | — | `compress_disk_cache` | manual cleanup | change cache dir | `cache_format` **`v2`** (mmap bf16 stacks); **`v1`** = pickle shards |

## Code locations

| Component | Path |
|-----------|------|
| Train loader + prefetch thread | `renga_flow/data/loader.py` |
| Cache map/pool | `renga_flow/data/cache_utils.py` |
| Disk cache v1 / v2 | `renga_flow/utils/cache.py`, `cache_v2.py`, `cache_factory.py` |
| Bench CSV / A/B helpers | `renga_flow/utils/bench.py` |
| Defaults | `renga_flow/config/defaults.py` |

## GPU smoke and A/B

1. **Full smoke** (cache + 30 steps, auto purge): `scripts/run_model_smoke.sh sdxl|cosmos`
2. **A/B train variants** (one shared `cache_only`, then `--trust_cache` per variant): `scripts/smoke_perf_ab.sh sdxl [prefetch] [workers2]`

Requires repo-root `.env` with checkpoint paths. Both scripts delete `output/` and `tests/fixtures/smoke_cc0/images/cache/` by default. Set `KEEP_SMOKE_ARTIFACTS=1` to inspect run dirs.

**Merge criterion:** ship a new default only if smoke A/B shows ≥3–5% lower `iter_sec_mean` (steps ≥ 6) vs baseline; otherwise keep opt-in.

## Disk hygiene (production datasets)

- Cache path: `<[[directory]].path>/cache/<model_name>/`
- Remove after experiments: `rm -rf <dataset_path>/cache/<model_name>`
- Do not commit cache or `output/` (see `.gitignore`)
