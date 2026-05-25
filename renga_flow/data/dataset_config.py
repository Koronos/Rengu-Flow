"""Dataset TOML schema and validation for directory-based datasets."""

from __future__ import annotations


class DatasetConfigError(ValueError):
    """Raised when dataset config is invalid for real (non-synthetic) dataset."""


def validate_dataset_config_for_real_data(dataset_config: dict) -> None:
    """Ensure dataset config has at least one directory when using real data.

    Call this when building Dataset from TOML for training (not for synthetic).
    Raises DatasetConfigError if config is invalid.
    """
    if "directory" not in dataset_config:
        raise DatasetConfigError(
            "Dataset config must contain 'directory' (list of directory entries)."
        )
    directories = dataset_config["directory"]
    if not directories:
        raise DatasetConfigError(
            "Dataset config 'directory' must be non-empty for real data. "
            "Add at least one [[directory]] with 'path' and 'num_repeats'."
        )
    for i, d in enumerate(directories):
        if not isinstance(d, dict):
            raise DatasetConfigError(
                f"dataset_config['directory'][{i}] must be a dict (path, num_repeats, ...)."
            )
        if "path" not in d:
            raise DatasetConfigError(
                f"dataset_config['directory'][{i}] must contain 'path'."
            )
        if "num_repeats" not in d:
            raise DatasetConfigError(
                f"dataset_config['directory'][{i}] must contain 'num_repeats'."
            )
        if d["num_repeats"] <= 0:
            raise DatasetConfigError(
                f"dataset_config['directory'][{i}]['num_repeats'] must be > 0."
            )
