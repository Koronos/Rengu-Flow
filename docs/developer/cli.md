# CLI implementation (`renga`)

## Entry points

| Script | Module |
|--------|--------|
| `renga` | `renga_flow.cli:main` |
| `renga-flow` (deprecated) | `renga_flow.cli:main` |
| `renga-flow-ui` (deprecated) | `renga_flow_ui.cli:main` — maps to `renga ui …` when invoked via wrapper |

Repo-root [`renga`](../../renga) runs `uv run renga`.

## Layout

| Module | Role |
|--------|------|
| [`renga_flow/cli/main.py`](../../renga_flow/cli/main.py) | Dispatcher, Linux guard, legacy `renga --config` |
| [`renga_flow/cli/init_cmd.py`](../../renga_flow/cli/init_cmd.py) | `init` = local TOML + `uv sync` |
| [`renga_flow/cli/update_cmd.py`](../../renga_flow/cli/update_cmd.py) | `update` = `uv sync` only |
| [`renga_flow/cli/train_cmd.py`](../../renga_flow/cli/train_cmd.py) | `train`, `validate`, `cache`, `dump-dataset` |
| [`renga_flow/cli/ui_cmd.py`](../../renga_flow/cli/ui_cmd.py) | `ui start` / `serve` / `dev` / `build` / `reset-db` |
| [`renga_flow/cli/train_launcher.py`](../../renga_flow/cli/train_launcher.py) | DeepSpeed argv + `[training.env]` merge |
| [`renga_flow/config/local_config.py`](../../renga_flow/config/local_config.py) | Parse `renga.local.toml`, apply UI env |
| [`renga_flow/install_profiles.py`](../../renga_flow/install_profiles.py) | Profile names → `uv sync --extra` |
| [`renga_flow/main.py`](../../renga_flow/main.py) | `parse_args` / `run_prepared` — still used by DeepSpeed `-m renga_flow.main` |

## Training launcher

[`renga_flow_ui/jobs.py`](../../renga_flow_ui/jobs.py) calls `build_train_command` and `training_subprocess_env` from `train_launcher.py` so UI jobs and `renga train` share defaults from `renga.local.toml`.

DeepSpeed subprocesses should continue to use `-m renga_flow.main` so the CLI does not need to be on `PATH` inside workers.

## Extending install profiles

Add an entry to `PROFILE_EXTRAS` and labels in [`renga_flow/install_profiles.py`](../../renga_flow/install_profiles.py). Maintenance UI reads `DEP_PROFILES` built from the same helpers.
