# CLI implementation (`rengu`)

## Entry points

| Script | Module |
|--------|--------|
| `rengu` | `rengu_flow.cli:main` |
| `rengu-flow` (deprecated) | `rengu_flow.cli:main` |
| `rengu-flow-ui` (deprecated) | `rengu_flow_ui.cli:main` — separate entry point; prefer `rengu ui …` |

Repo-root [`rengu`](../../rengu) runs `uv sync` if needed, then `exec .venv/bin/rengu` (requires [uv](https://docs.astral.sh/uv/) on `PATH`).

## Layout

| Module | Role |
|--------|------|
| [`rengu_flow/cli/main.py`](../../rengu_flow/cli/main.py) | Dispatcher, Linux guard, legacy `rengu --config` |
| [`rengu_flow/cli/init_cmd.py`](../../rengu_flow/cli/init_cmd.py) | `init` = local TOML + `uv sync` |
| [`rengu_flow/cli/update_cmd.py`](../../rengu_flow/cli/update_cmd.py) | `update` = `uv sync` only |
| [`rengu_flow/cli/train_cmd.py`](../../rengu_flow/cli/train_cmd.py) | `train`, `validate`, `cache`, `dump-dataset` |
| [`rengu_flow/cli/ui_cmd.py`](../../rengu_flow/cli/ui_cmd.py) | `ui start` / `serve` / `dev` / `build` / `reset-db` |
| [`rengu_flow/cli/train_launcher.py`](../../rengu_flow/cli/train_launcher.py) | `base_train_command` delegates to `select_backend(config).launch_argv` (`deepspeed --num_gpus` for `"deepspeed"`, `python -m` for `"accelerate"`) + `[training.env]` merge |
| [`rengu_flow/config/local_config.py`](../../rengu_flow/config/local_config.py) | Parse `rengu.local.toml`, apply UI env |
| [`rengu_flow/install_profiles.py`](../../rengu_flow/install_profiles.py) | Profile names → `uv sync --extra` |
| [`rengu_flow/main.py`](../../rengu_flow/main.py) | `parse_args` / `run_prepared` — still used by DeepSpeed `--module rengu_flow.main` |

## Training launcher

[`rengu_flow_ui/jobs.py`](../../rengu_flow_ui/jobs.py) imports `base_train_command` and `training_subprocess_env` from `train_launcher.py` (wrapping the former in its own `build_train_command`) so UI jobs and `rengu train` share defaults from `rengu.local.toml`.

The `"deepspeed"` backend uses `--module rengu_flow.main` (DeepSpeed's launcher flag for a module target — **not** `-m`, which it rejects). The `"accelerate"` backend uses `python -m rengu_flow.main` directly. Neither requires the `rengu` CLI to be on `PATH` inside workers.

## Extending install profiles

Add an entry to `PROFILE_EXTRAS` and labels in [`rengu_flow/install_profiles.py`](../../rengu_flow/install_profiles.py). Maintenance UI reads `DEP_PROFILES` built from the same helpers.
