# Dependency installer (`rengu_flow/install/`)

All optional-dependency logic is centralized in the `rengu_flow.install` package. CLI/UI modules delegate to it; a few legacy import paths remain as thin re-export shims.

## Modules

- **`install/profiles.py`** — profile→extra mapping (`PROFILE_EXTRAS`), labels/descriptions, `normalize_profiles`, `uv_sync_argv` (always emits `uv sync --inexact …`), `PROFILE_IMPORT_CHECKS` (profile→modules that must import), and `PROFILE_GIT_REQUIREMENTS` (profile→pip/git specs uv can't install via extras).
- **`install/runner.py`** — subprocess wrappers: `run_uv_venv(_or_exit)`, `run_uv_sync(_or_exit)`, `run_uv_pip_install(_or_exit)`, and `require_uv`.
- **`install/state.py`** — persisted record of enabled profiles at `<repo>/.rengu-flow/installed-profiles.json` (gitignored): `read_installed_profiles`, `record_installed_profiles` (additive merge).
- **`install/manager.py`** — the on-demand API: `profile_installed` / `missing_profiles` (import probes), `ensure_profiles` (additive install of only what's missing + git requirements + records success), `self_heal` (re-ensure recorded profiles), `ensure_ui_dependencies`, and `profiles_for_config_*` / `ensure_training_extras`.

## Invariants

- **Never destructive.** `uv sync` always runs with `--inexact`; we never call exact sync. This preserves other extras and user-installed packages (e.g. custom optimizers/schedulers resolved by qualified path via `rengu_flow/registry/optimizers.py`).
- **`uv run` caveat.** `uv run <cmd>` does an exact base sync first (ignores `--inexact`); the launchers (`rengu`, `start-ui.sh`) call the venv binary directly to avoid it, and `self_heal` recovers otherwise.

## Shims (backwards-compatible import paths)

`rengu_flow/install_profiles.py`, `rengu_flow/cli/uv_cmd.py`, and `rengu_flow/cli/training_extras.py` re-export from the new package so existing imports (and tests) keep working. `rengu_flow/cli/project_venv.py` keeps the venv helpers (`venv_python`, `reexec_cli`, `sync_dependencies`, `ensure_ui_dependencies`) and pulls runners from `install.runner`.

## Registering a git/VCS-backed backend

Add the spec to `PROFILE_GIT_REQUIREMENTS` and the importable module name to `PROFILE_IMPORT_CHECKS` in `install/profiles.py`:

```python
PROFILE_IMPORT_CHECKS["myoptim"] = ("cool_optimizer",)
PROFILE_GIT_REQUIREMENTS["myoptim"] = ["git+https://github.com/acme/cool-optimizer@v1.2.0"]
```

`ensure_profiles(["myoptim"])` then installs it additively only when `cool_optimizer` isn't importable.
