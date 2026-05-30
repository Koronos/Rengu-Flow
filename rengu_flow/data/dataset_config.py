"""Dataset TOML schema and validation for directory-based datasets."""

from __future__ import annotations

from rengu_flow.data.augmentation import (
    AugmentationConfigError,
    AugmentationStrategyNotImplementedError,
    validate_augmentation_for_directory,
)


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
        try:
            validate_augmentation_for_directory(d, dataset_config)
        except (AugmentationConfigError, AugmentationStrategyNotImplementedError) as e:
            raise DatasetConfigError(str(e)) from e

    _validate_unit_fraction(dataset_config, "tag_dropout_probability")
    _validate_unit_fraction(dataset_config, "uncond_fraction")


def _validate_unit_fraction(dataset_config: dict, key: str) -> None:
    """Raise DatasetConfigError when ``key`` is set but outside the [0, 1] range."""
    if key not in dataset_config or dataset_config[key] is None:
        return
    try:
        value = float(dataset_config[key])
    except (TypeError, ValueError):
        raise DatasetConfigError(f"{key} must be a number in [0, 1].") from None
    if not 0.0 <= value <= 1.0:
        raise DatasetConfigError(f"{key} must be in [0, 1] (got {value}).")
