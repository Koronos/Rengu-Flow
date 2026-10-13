# Smoke tests

GPU smokes are optional, local end-to-end runs: real training on a GPU, not run in CI. Unit
tests (`pytest`, CPU + mocks) cover behavior; a smoke proves the whole path (config → cache →
train → save / preview) still runs on real hardware. This page is the contract every smoke
follows. For unit tests see [Testing](testing.md).

## Convention

| Topic | Rule |
|-------|------|
| **Naming** | `scripts/smoke_<area>[_<variant>].{sh,py}` (e.g. `smoke_mini.py`, `smoke_training_signals.sh`). Per-model runs are never new scripts: real weights = a fixture for `scripts/run_model_smoke.sh <model>`, tiny random model = the model's testkit for `scripts/smoke_mini.py --model <model>`. Older names are grandfathered (see [Follow-ups](#follow-ups)). |
| **Shared shell helpers** | `source scripts/lib/smoke_common.sh`: `require_deepspeed`, `load_smoke_dotenv`, `smoke_skip`, `setup_smoke_gpu_env`, `select_master_port_if_unset`, `purge_output_dir`, `smoke_run_phase`. |
| **Fixtures** | Training/dataset TOMLs in `tests/fixtures/smoke/`, **without** `[model]` paths. Images: the versioned CC0 set `tests/fixtures/smoke_cc0/` (`scripts/vendor_smoke_cc0.sh` downloads it on demand). Synthetic data a script generates goes under `tmp/`. |
| **Model paths** | Only from the repo-root `.env` (gitignored), as `RENGU_<MODEL>_<COMPONENT>_PATH`. `rengu_flow/config/local_env.py` maps the vars to `[model]` keys (`_MODEL_PATH_ENV`); every var is listed in `.env.example`. Shell smokes run `python -m rengu_flow.config.local_env CONFIG` (pre-check) then `load_smoke_dotenv` (exports the vars to the trainer, which never reads `.env` itself). Python smokes call `local_env.load_repo_dotenv()`. |
| **No personal paths** | Tracked files never contain absolute or home paths (drive letters + `Users`, `/home/…`, `AppData`, WSL shares). Scripts derive paths from the repo root (`$(dirname "${BASH_SOURCE[0]}")/..` or `Path(__file__).resolve().parents[1]`) and write runtime TOMLs with computed paths under `tmp/`. |
| **Artifacts** | Only under `output/` or `tmp/` (both gitignored), plus fixture caches (`**/cache/`, gitignored). Clean them by default; `KEEP_SMOKE_ARTIFACTS=1` (shell) / `--keep` (Python) keeps them, `KEEP_SMOKE_LOG=1` keeps logs on success. Logs go to `tmp/smoke_<case>_<timestamp>.log`. |
| **Exit codes** | `0` pass · `77` **skipped** — prerequisites missing (weights not in `.env`, no DeepSpeed, no CUDA); prints `SKIP: <reason>` · any other nonzero = **fail**. `77` is `SMOKE_SKIP_EXIT` in both `local_env.py` and `smoke_common.sh`. |
| **Pass criteria** | Assert on artifacts, not just "no crash": saved adapter/model file (and its key layout where an export check exists), preview PNG when previews are part of the smoke, finite loss. Print peak VRAM and s/it from the bench (`bench = true` → `<run>/bench_steps.csv`, `bench_summary.txt`). |
| **Platform** | Real-weight shell smokes: Linux/WSL2 + DeepSpeed (`deepspeed --num_gpus=1`); see [WSL/Windows workflow](wsl-windows-workflow.md). Mini/random-weight Python smokes: single-device engine (`RENGU_ENGINE=accelerate`), in-process under `if __name__ == "__main__":` (Windows caching workers use spawn), run natively on Windows or Linux. |

## Inventory

| Smoke | Command | Needs | Checks |
|-------|---------|-------|--------|
| Per-model real weights | `scripts/run_model_smoke.sh <model>[_lokr\|_full]` | Linux/WSL, DeepSpeed, `.env` weights | Runs `tests/fixtures/smoke/train_<model>[_lokr\|_full].toml` (`cosmos` is short for `cosmos_predict2`): `--cache_only`, then the fixture's `max_steps`, then the generic export check `python -m rengu_flow.networks.export_check` (model type and adapter from the config, file name and key layout from `spec.adapter_export(kind)` — e.g. SDXL LoRA: kohya `lora.safetensors` with `lora_unet_` / `lora_te1_text_model_` / `lora_te2_text_model_` + `lora_down` / `lora_up` / `alpha`; DiTs: `adapter_model.safetensors` under `spec.export_prefix`: roots, per-module weights or paired factors, LoRA key names, finite; full finetune: finite tensors). Exit 77 when the weights are missing. What each fixture sets (steps, VRAM setup, preview) is in its header comment. |
| LyCORIS per algo | `scripts/run_model_smoke.sh sdxl_lycoris_<algo>\|cosmos_lycoris_<algo>\|…_extras\|…_all` | same | 12 steps + `python -m rengu_flow.networks.lycoris_export_check`. |
| Mini (testkit) | `python scripts/smoke_mini.py --model <model> [--adapter lora\|lokr\|…\|full] [--steps 3] [--keep]` | CPU only, no weights, no DeepSpeed | Any registered model whose spec names a testkit (`rengu_flow/model/<m>/testkit.py`): tiny random model through the real loaders, then the real path: synthetic dataset per kit case → cache → `prepare_inputs` → every pipeline layer → loss/backward → optimizer; preview PNG when the spec has one; export checked with `spec.adapter_export(kind)` (adapter) or reloaded identically (full). Exit 0/1. Not a VRAM or speed measurement: real-size GPU runs are the per-model fixtures above. |
| Training signals | `scripts/smoke_training_signals.sh` | DeepSpeed, `RENGU_COSMOS_*`, `optim` extra | Each signal file during a run, then `genericoptim` resume. |
| Async export | `scripts/smoke_async_export_poc.sh` | DeepSpeed, `RENGU_SDXL_CHECKPOINT_PATH` | `step10/` + `step20/` `lora.safetensors`, `[async_export]` in log. |
| Preview once | `scripts/run_preview_once.sh` | DeepSpeed, `RENGU_COSMOS_*`, existing cache | Touches `preview`, waits for the PNG. Keeps artifacts. |
| Dataloader A/B | `scripts/smoke_perf_ab.sh sdxl\|cosmos [prefetch] [workers2]` | DeepSpeed, `.env` weights | Mean `iter_sec` (steps ≥ 6) per variant; see [CPU/RAM performance](performance-cpu-ram.md). |
| Helpers | `scripts/vendor_smoke_cc0.sh`, `scripts/smoke_cc0_cache_ready.py` | network (vendor) | Vendor the CC0 images; exit 0 when the SDXL smoke cache exists. |

`scripts/poc_cpu_ram_optimizations.py` is a CPU benchmark, not a GPU smoke (CI runs it via
`tests/test_poc_cpu_ram_smoke.py`).

## Adding a smoke

1. Real weights for a model → add a fixture `tests/fixtures/smoke/train_<model>[_<variant>].toml`
   (no `[model]` paths); `scripts/run_model_smoke.sh <model>[_<variant>]` picks it up, no case to
   add. A model with no fixture declares `SMOKE_FIXTURE_EXEMPT = "<reason>"` in its testkit.
   Without weights → the model's testkit runs in `scripts/smoke_mini.py --model <model>`.
   Anything else → a new `scripts/smoke_<area>[_<variant>].{sh,py}`.
2. New model → map its `RENGU_<MODEL>_*_PATH` vars in `local_env._MODEL_PATH_ENV` and list them in
   `.env.example`. `model_path_errors` honours the capability's `one_of` groups.
3. Derive every path from the repo root; generated configs and data go under `tmp/`.
4. Exit `77` with `SKIP: <reason>` when prerequisites are missing; nonzero on any failed check.
5. Clean `output/` / `tmp/` artifacts by default, keep them with `KEEP_SMOKE_ARTIFACTS=1` / `--keep`.
6. Add a row to the [Inventory](#inventory) table (generic runners are one row, not one per model).
7. Run `pytest tests/test_smoke_conventions.py` (no personal paths, no `[model]` paths in
   fixtures, every `RENGU_*_PATH` documented), `pytest tests/contract/test_smoke_mini.py` (every
   testkit model runs in `smoke_mini.py`, every model has a fixture or an exemption) and a privacy scan of your diff:

```bash
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u \
  | xargs grep -nIiE 'Users[\\/]|/home/|AppData|wsl\.localhost|@[a-z0-9-]+\.[a-z]{2,}|<your-username>'
```

## Follow-ups

Not changed yet (renames or behavior changes with a wider blast radius):

- **Grandfathered names:** `run_model_smoke.sh`, `run_preview_once.sh`,
  `smoke_async_export_poc.sh` (`_poc` suffix), `smoke_cc0_cache_ready.py` are referenced from many
  docs; rename only together with every reference.
- **Orphan fixture:** `train_cosmos_predict2_lokr_kaon_fused.toml` has no runner case.
- **`run_preview_once.sh`** forces `KEEP_SMOKE_ARTIFACTS=1` and requires an existing cache (no
  vendor/cache step); `smoke_perf_ab.sh` sources `.env` itself instead of `load_smoke_dotenv` and
  skips the `local_env` pre-check.
- **`smoke_cc0_cache_ready.py`** is not called by any script.
- `python -m rengu_flow.config.local_env` prints a harmless `runpy` RuntimeWarning (the module is
  imported by `rengu_flow.config` first).
