"""Legacy dotenv helpers and optional env-based model path overrides (smoke/CI only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from rengu_flow.config.local_config import load_local_config, repo_root

# Smoke "skipped: prerequisites missing" exit code (autotools convention); see
# docs/developer/smoke-tests.md. scripts/lib/smoke_common.sh uses the same value.
SMOKE_SKIP_EXIT = 77

# Env var name -> config['model'] key per model type (optional override when env is set externally).
_MODEL_PATH_ENV: dict[str, dict[str, str]] = {
    "sdxl": {"checkpoint_path": "RENGU_SDXL_CHECKPOINT_PATH"},
    "cosmos_predict2": {
        "transformer_path": "RENGU_COSMOS_TRANSFORMER_PATH",
        "vae_path": "RENGU_COSMOS_VAE_PATH",
        "llm_path": "RENGU_COSMOS_LLM_PATH",
    },
    "krea2": {
        "transformer_path": "RENGU_KREA2_TRANSFORMER_PATH",
        "vae_path": "RENGU_KREA2_VAE_PATH",
        "text_encoder_path": "RENGU_KREA2_TEXT_ENCODER_PATH",
        "checkpoint_path": "RENGU_KREA2_CHECKPOINT_PATH",
    },
}


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return key, value


def load_repo_dotenv(path: Path | None = None, *, override: bool = False) -> bool:
    """Deprecated: load ``.env`` from repo root. Prefer ``rengu.local.toml`` via ``load_local_config``."""
    env_path = path if path is not None else repo_root() / ".env"
    if not env_path.is_file():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return True


def apply_model_paths_from_env(config: dict[str, Any]) -> list[str]:
    """Override ``[model]`` paths only when matching env vars are already set (CI/smoke)."""
    model = config.get("model")
    if not isinstance(model, dict):
        return []
    model_type = str(model.get("type", "")).lower()
    mapping = _MODEL_PATH_ENV.get(model_type, {})
    applied: list[str] = []
    for model_key, env_name in mapping.items():
        value = os.environ.get(env_name)
        if value:
            model[model_key] = value
            applied.append(env_name)
    return applied


def _one_of_groups(model_type: str) -> list[list[str]]:
    from rengu_flow.registry.model_capabilities import get_capability
    from rengu_flow.registry.model_config_rules import one_of_groups

    cap = get_capability(model_type)
    return one_of_groups(cap) if cap else []


def model_path_errors(config: dict[str, Any]) -> list[str]:
    """Human-readable errors for missing or invalid model paths in the training config.

    Keys in a capability ``one_of`` group (krea2: ``<component>_path`` or
    ``checkpoint_path``) only need one member set, and may be folders; other keys must be
    existing files.
    """
    model = config.get("model")
    if not isinstance(model, dict):
        return []
    model_type = str(model.get("type", "")).lower()
    mapping = _MODEL_PATH_ENV.get(model_type, {})
    groups = [g for g in _one_of_groups(model_type) if all(k in mapping for k in g)]
    grouped = {k for g in groups for k in g}

    def _is_set(key: str) -> bool:
        return bool(str(model.get(key) or "").strip())

    errors: list[str] = []
    for group in groups:
        if not any(_is_set(k) for k in group):
            errors.append(f"Set one of [model].{' / [model].'.join(group)} in your training config")
    for model_key in mapping:
        path = str(model.get(model_key) or "").strip()
        if not path:
            if model_key not in grouped:
                errors.append(f"Set [model].{model_key} in your training config")
            continue
        exists = Path(path).exists() if model_key in grouped else Path(path).is_file()
        if not exists:
            errors.append(f"[model].{model_key} not found: {path}")
    return errors


def check_config_model_paths(config_path: str | Path) -> None:
    """Load config (+ repo-root ``.env``); exit ``SMOKE_SKIP_EXIT`` if model paths are missing.

    Smoke pre-check: missing weights mean the smoke is skipped, not failed.
    """
    from rengu_flow.config.loader import load_config

    load_local_config()
    load_repo_dotenv()
    config = load_config(config_path)
    apply_model_paths_from_env(config)
    errors = model_path_errors(config)
    if errors:
        for msg in errors:
            print(f"SKIP: {msg}", file=sys.stderr)
        raise SystemExit(SMOKE_SKIP_EXIT)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m rengu_flow.config.local_env CONFIG.toml", file=sys.stderr)
        raise SystemExit(2)
    check_config_model_paths(sys.argv[1])
