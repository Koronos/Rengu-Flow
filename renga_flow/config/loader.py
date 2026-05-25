"""Load TOML config files: main config and dataset config(s)."""

import json
from pathlib import Path
from typing import Any

import toml


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


def load_dataset_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Load dataset TOML referenced by config['dataset'].

    Args:
        config: Main config dict; must contain 'dataset' key with a path to a TOML file.

    Returns:
        Dataset config dict, or None if config has no 'dataset' key.
    """
    dataset_path = config.get("dataset")
    if dataset_path is None:
        return None
    path = Path(dataset_path)
    with open(path) as f:
        return toml.load(f)


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
