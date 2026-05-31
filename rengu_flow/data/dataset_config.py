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
        _validate_positive_int(
            d, "max_images", f"dataset_config['directory'][{i}]['max_images']"
        )
        _validate_sampler_exclusivity(
            d, f"dataset_config['directory'][{i}]"
        )
        try:
            validate_augmentation_for_directory(d, dataset_config)
        except (AugmentationConfigError, AugmentationStrategyNotImplementedError) as e:
            raise DatasetConfigError(str(e)) from e

    _validate_positive_int(dataset_config, "max_images", "max_images")
    _validate_sampler_exclusivity(dataset_config, "dataset_config")
    _validate_unit_fraction(dataset_config, "tag_dropout_probability")
    _validate_unit_fraction(dataset_config, "uncond_fraction")


def _validate_sampler_exclusivity(config: dict, label: str) -> None:
    """Reject defining both per-epoch limiters (subsample_ratio < 1 and max_images) together.

    ``subsample_ratio`` (fraction) and ``max_images`` (absolute count) are two ways to limit how
    many images are used per epoch; only one may be set in the same scope (a single directory, or
    the dataset root). Checks explicitly-set keys only.
    """
    if config.get("max_images") is None or "subsample_ratio" not in config:
        return
    try:
        ratio = float(config["subsample_ratio"])
    except (TypeError, ValueError):
        return
    if ratio < 1.0:
        raise DatasetConfigError(
            f"{label}: set either 'subsample_ratio' (< 1) or 'max_images', not both — "
            "they are mutually exclusive per-epoch image limiters."
        )


def _validate_positive_int(config: dict, key: str, label: str) -> None:
    """Raise DatasetConfigError when ``key`` is set but is not an integer > 0."""
    if key not in config or config[key] is None:
        return
    value = config[key]
    # bool is an int subclass but never a valid count here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetConfigError(f"{label} must be an integer > 0 (got {value!r}).")
    if value <= 0:
        raise DatasetConfigError(f"{label} must be > 0 (got {value}).")


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
