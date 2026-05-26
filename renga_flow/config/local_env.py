"""Load optional repo-root `.env` and apply local model paths to config."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Env var name -> config['model'] key per model type
_MODEL_PATH_ENV: dict[str, dict[str, str]] = {
    "sdxl": {"checkpoint_path": "RENGA_SDXL_CHECKPOINT_PATH"},
    "cosmos_predict2": {
        "transformer_path": "RENGA_COSMOS_TRANSFORMER_PATH",
        "vae_path": "RENGA_COSMOS_VAE_PATH",
        "llm_path": "RENGA_COSMOS_LLM_PATH",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    """Load ``.env`` from repo root into ``os.environ``. Returns True if file was read."""
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
    """Set ``[model]`` checkpoint paths from env vars when present.

    Returns list of env var names that were applied.
    """
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


def model_path_errors(config: dict[str, Any]) -> list[str]:
    """Human-readable errors for missing or invalid model paths (after env apply)."""
    model = config.get("model")
    if not isinstance(model, dict):
        return []
    model_type = str(model.get("type", "")).lower()
    mapping = _MODEL_PATH_ENV.get(model_type, {})
    errors: list[str] = []
    for model_key, env_name in mapping.items():
        path = str(model.get(model_key, "")).strip()
        if not path:
            errors.append(f"Set {env_name} in .env (or [model].{model_key} in config)")
            continue
        if not Path(path).is_file():
            errors.append(f"{env_name} not found: {path}")
    return errors


def check_config_model_paths(config_path: str | Path) -> None:
    """Load config + .env and exit non-zero if model paths are missing (for smoke scripts)."""
    from renga_flow.config.loader import load_config

    load_repo_dotenv()
    config = load_config(config_path)
    apply_model_paths_from_env(config)
    errors = model_path_errors(config)
    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m renga_flow.config.local_env CONFIG.toml", file=sys.stderr)
        raise SystemExit(2)
    check_config_model_paths(sys.argv[1])
