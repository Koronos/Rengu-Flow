"""Load TOML config files: main config and dataset config(s)."""

import json
from pathlib import Path
from typing import Any

import toml

from renga_flow.config.dataset_merge import merge_dataset_configs


def load_config(path: str | Path, make_pickleable: bool = True) -> dict[str, Any]:
    """Load main TOML config from file.

    Args:
        path: Path to the main TOML configuration file.
        make_pickleable: If True, convert config via json.loads(json.dumps(...))
            so it is pickleable (for multiprocessing in later phases).

    Returns:
        Config dict. If make_pickleable, nested structures are plain dicts/lists.
    """
    path = Path(path)
    with open(path) as f:
        config = toml.load(f)
    if make_pickleable:
        config = json.loads(json.dumps(config))
    return config


def normalize_dataset_paths(value: Any) -> list[str]:
    """Return non-empty dataset path strings from a main-config ``dataset`` value."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def load_dataset_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Load dataset TOML(s) referenced by config['dataset'].

    ``dataset`` may be a single path string or a list of paths. Multiple paths are
    merged (all ``[[directory]]`` tables; globals from the first file), same as
    composing datasets in the UI library.

    Args:
        config: Main config dict with optional ``dataset`` key.

    Returns:
        Dataset config dict, or None if config has no usable ``dataset`` value.
    """
    paths = normalize_dataset_paths(config.get("dataset"))
    if not paths:
        return None
    loaded: list[dict[str, Any]] = []
    for dataset_path in paths:
        with open(dataset_path, encoding="utf-8") as f:
            loaded.append(toml.load(f))
    if len(loaded) == 1:
        return json.loads(json.dumps(loaded[0]))
    merged = merge_dataset_configs(loaded)
    return json.loads(json.dumps(merged))


def load_eval_dataset_config(eval_entry: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Load a single eval dataset config from a path or from a dict with 'name' and 'config'.

    Args:
        eval_entry: Either a path string or a dict with 'name' and 'config' keys.

    Returns:
        (name, dataset_config) for use in eval_datasets.
    """
    if isinstance(eval_entry, str):
        name = f"eval_{Path(eval_entry).stem}"
        with open(eval_entry) as f:
            return name, toml.load(f)
    name = eval_entry["name"]
    config_path = eval_entry["config"]
    with open(config_path) as f:
        return name, toml.load(f)
