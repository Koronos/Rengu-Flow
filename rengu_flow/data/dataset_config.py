"""Dataset TOML schema and validation for directory-based datasets."""

from __future__ import annotations

import logging

from rengu_flow.data.augmentation import (
    AugmentationConfigError,
    AugmentationStrategyNotImplementedError,
    validate_augmentation_for_directory,
)

logger = logging.getLogger(__name__)


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
    _validate_resolution_schedule(dataset_config)


def _collect_declared_resolutions(dataset_config: dict) -> set[int]:
    """Union of long-side resolutions declared globally and per directory."""
    res: set[int] = set()
    global_res = dataset_config.get("resolutions")
    if isinstance(global_res, (list, tuple)):
        res.update(int(r) for r in global_res)
    for d in dataset_config.get("directory", []) or []:
        if isinstance(d, dict) and isinstance(d.get("resolutions"), (list, tuple)):
            res.update(int(r) for r in d["resolutions"])
    return res


def _validate_resolution_schedule(dataset_config: dict) -> None:
    """Validate the optional ``[resolution_schedule]`` section.

    Checks structure (enabled flag, stages with non-empty ``resolutions`` and a
    positive ``fraction``) and that every stage resolution is one of the dataset's
    declared resolutions. Warns (does not fail) when a declared resolution is never
    used by any stage, since its cached latents would go unused.
    """
    sched = dataset_config.get("resolution_schedule")
    if sched is None:
        return
    if not isinstance(sched, dict):
        raise DatasetConfigError("resolution_schedule must be a table.")
    if not sched.get("enabled", False):
        return
    stages = sched.get("stage", sched.get("stages"))
    if not stages or not isinstance(stages, (list, tuple)):
        raise DatasetConfigError(
            "resolution_schedule.enabled is true but no [[resolution_schedule.stage]] "
            "entries were defined."
        )
    declared = _collect_declared_resolutions(dataset_config)
    used: set[int] = set()
    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            raise DatasetConfigError(f"resolution_schedule.stage[{i}] must be a table.")
        res = st.get("resolutions", st.get("resolution"))
        if res is None:
            raise DatasetConfigError(
                f"resolution_schedule.stage[{i}] must set 'resolutions'."
            )
        if not isinstance(res, (list, tuple)):
            res = [res]
        if not res:
            raise DatasetConfigError(
                f"resolution_schedule.stage[{i}].resolutions must be non-empty."
            )
        try:
            res_ints = [int(r) for r in res]
        except (TypeError, ValueError):
            raise DatasetConfigError(
                f"resolution_schedule.stage[{i}].resolutions must be integers."
            ) from None
        frac = st.get("fraction")
        if frac is None:
            raise DatasetConfigError(
                f"resolution_schedule.stage[{i}] must set 'fraction'."
            )
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            raise DatasetConfigError(
                f"resolution_schedule.stage[{i}].fraction must be a number."
            ) from None
        if frac <= 0:
            raise DatasetConfigError(
                f"resolution_schedule.stage[{i}].fraction must be > 0 (got {frac})."
            )
        for r in res_ints:
            if declared and r not in declared:
                raise DatasetConfigError(
                    f"resolution_schedule.stage[{i}] references resolution {r}, which is "
                    f"not in the dataset's resolutions {sorted(declared)}."
                )
            used.add(r)
    if declared:
        unused = declared - used
        if unused:
            logger.warning(
                "resolution_schedule: resolutions %s are cached but never used by any "
                "stage; their latents will go unused.",
                sorted(unused),
            )


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
