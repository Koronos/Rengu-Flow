"""Estimate total training steps from a dataset/training config, without caching.

Pure and torch-free so the web UI can show the step count live while a run is configured
(instead of starting a run and waiting through latent caching to find out). Mirrors the
trainer's real accounting (``Dataset.scheduled_epoch_len`` + ``main``):

  steps_per_epoch = sum over resolutions of  phi(res) * (images_at_res // global_batch(res))
  total_steps     = epochs * steps_per_epoch        (capped at max_steps if set)

where ``images_at_res`` is the per-folder served base images (after max_images / subsample_ratio)
times augmentation branches times num_repeats, and ``phi(res)`` is the fraction of the run the
resolution is active under a ``resolution_schedule`` (1.0 with no schedule). Caption variants
cancel out (an epoch is one pass over the images), so they do not enter here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def _resolution_active_fractions(schedule: Mapping | None) -> dict[int, float] | None:
    """phi per resolution from a ``[resolution_schedule]`` config, or None when inactive.

    A resolution sums the fractions of every stage it appears in (one active all run -> 1.0).
    Fractions are normalized to sum to 1, matching ``parse_resolution_schedule``.
    """
    if not isinstance(schedule, Mapping) or not schedule.get("enabled", False):
        return None
    raw_stages = schedule.get("stage", schedule.get("stages", [])) or []
    stages: list[tuple[list[int], float]] = []
    for st in raw_stages:
        if not isinstance(st, Mapping):
            continue
        res = st.get("resolutions", st.get("resolution"))
        if res is None:
            continue
        if not isinstance(res, (list, tuple)):
            res = [res]
        try:
            res_ints = [int(r) for r in res]
            frac = float(st.get("fraction", 0.0))
        except (TypeError, ValueError):
            continue
        if not res_ints or frac <= 0.0:
            continue
        stages.append((res_ints, frac))
    total = sum(f for _, f in stages)
    if total <= 0:
        return None
    phi: dict[int, float] = {}
    for res_ints, frac in stages:
        for r in res_ints:
            phi[r] = phi.get(r, 0.0) + frac / total
    return phi


def _batch_for_resolution(micro_batch, resolution: int) -> int:
    """Per-GPU micro-batch for ``resolution``: a plain int, or the nearest key of a dict.

    Matches ``ConcatenatedBatchedDataset.post_init`` (nearest key by the long side; a ``None``
    key is the default). Falls back to 1 on a malformed value.
    """
    if isinstance(micro_batch, Mapping):
        if None in micro_batch:
            try:
                return max(1, int(micro_batch[None]))
            except (TypeError, ValueError):
                return 1
        best, best_diff = 1, math.inf
        for key, bs in micro_batch.items():
            try:
                diff = abs(int(key) - resolution)
            except (TypeError, ValueError):
                continue
            if diff < best_diff:
                best, best_diff = int(bs), diff
        return max(1, best)
    try:
        return max(1, int(micro_batch))
    except (TypeError, ValueError):
        return 1


def _served_base_images(dir_cfg: Mapping, count: int, global_max, global_ratio) -> int:
    """Base images a folder serves per epoch after its cap (max_images / subsample_ratio).

    max_images wins and serves exactly that many (a smaller folder repeats up to it, matching
    FolderSubsampler); a subsample_ratio < 1 serves that fraction; otherwise the whole folder.
    """
    max_images = dir_cfg.get("max_images", global_max)
    if max_images is not None:
        try:
            return max(0, int(max_images))
        except (TypeError, ValueError):
            pass
    ratio = dir_cfg.get("subsample_ratio", global_ratio)
    try:
        ratio = float(ratio) if ratio is not None else 1.0
    except (TypeError, ValueError):
        ratio = 1.0
    if ratio < 1.0:
        return max(1, int(count * ratio)) if count else 0
    return count


def _augmentation_multiplier(dir_cfg: Mapping, dataset_config: Mapping) -> int:
    """Rows per base image from augmentation: pristine original + N branches (N+1), else 1.

    The global augmentation lives under the nested ``[dataset.augmentation]`` table (what the
    trainer reads in ``_global_augmentation_defaults``); a ``[[directory]].augmentation`` table
    overrides it. Mirror that here, with a top-level ``augmentation`` fallback for robustness.
    """
    global_aug = (dataset_config.get("dataset") or {}).get("augmentation")
    if not isinstance(global_aug, Mapping):
        global_aug = dataset_config.get("augmentation")
    global_aug = global_aug if isinstance(global_aug, Mapping) else {}
    dir_aug = dir_cfg.get("augmentation")
    dir_aug = dir_aug if isinstance(dir_aug, Mapping) else {}
    merged = {**global_aug, **dir_aug}  # directory overrides global (mirrors merge_directory_augmentation)
    if not merged.get("enabled", False):
        return 1
    try:
        branches = int(merged.get("branches_per_image", 1) or 0)
    except (TypeError, ValueError):
        branches = 0
    return max(1, branches + 1)


def estimate_total_steps(
    dataset_config: Mapping,
    training_config: Mapping,
    image_counts: Mapping[str, int],
    *,
    world_size: int = 1,
) -> dict:
    """Estimate steps_per_epoch / total_steps from config + per-folder base image counts.

    ``image_counts`` maps each directory ``path`` to its base image count (e.g. the web UI's
    ``scan_folder`` result). Returns a dict with ``steps_per_epoch``, ``total_steps``,
    ``images_per_resolution`` and ``per_resolution`` breakdown. ``max_steps`` (if set) caps
    ``total_steps``. Best-effort: missing counts are treated as 0.
    """
    resolutions = [int(r) for r in (dataset_config.get("resolutions") or []) if r is not None]
    directories = dataset_config.get("directory") or []
    global_max = dataset_config.get("max_images")
    global_ratio = dataset_config.get("subsample_ratio")

    # Images present at every resolution: each base image exists at each resolution, expanded
    # by augmentation branches and num_repeats, then summed across folders.
    images_at_res = 0
    for dir_cfg in directories:
        if not isinstance(dir_cfg, Mapping):
            continue
        count = int(image_counts.get(str(dir_cfg.get("path")), 0) or 0)
        served = _served_base_images(dir_cfg, count, global_max, global_ratio)
        aug = _augmentation_multiplier(dir_cfg, dataset_config)
        try:
            repeats = max(1, int(dir_cfg.get("num_repeats", 1) or 1))
        except (TypeError, ValueError):
            repeats = 1
        images_at_res += served * aug * repeats

    phi = _resolution_active_fractions(dataset_config.get("resolution_schedule"))
    micro_batch = training_config.get("micro_batch_size_per_gpu", 1)
    try:
        grad_accum = max(1, int(training_config.get("gradient_accumulation_steps", 1) or 1))
    except (TypeError, ValueError):
        grad_accum = 1
    world = max(1, int(world_size or 1))

    per_resolution = {}
    steps_per_epoch = 0.0
    for res in resolutions:
        global_batch = _batch_for_resolution(micro_batch, res) * grad_accum * world
        bucket_steps = images_at_res // max(1, global_batch)
        weight = 1.0 if phi is None else phi.get(res, 1.0)
        contribution = weight * bucket_steps
        per_resolution[res] = {
            "phi": weight,
            "global_batch": global_batch,
            "bucket_steps": bucket_steps,
            "contribution": contribution,
        }
        steps_per_epoch += contribution

    steps_per_epoch = max(1, round(steps_per_epoch)) if images_at_res else 0
    try:
        epochs = max(1, int(training_config.get("epochs", 1) or 1))
    except (TypeError, ValueError):
        epochs = 1
    total_steps = epochs * steps_per_epoch

    max_steps = training_config.get("max_steps")
    if max_steps is not None:
        try:
            total_steps = min(total_steps, int(max_steps))
        except (TypeError, ValueError):
            pass

    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "images_per_resolution": images_at_res,
        "epochs": epochs,
        "per_resolution": per_resolution,
    }
