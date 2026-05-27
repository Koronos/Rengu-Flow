"""Merge multiple dataset TOML configs into one (training / compose)."""

from __future__ import annotations

from typing import Any


def merge_dataset_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ``[[directory]]`` blocks; global keys come from the first config.

    Same rules as the UI library ``compose_datasets`` helper.
    """
    if not configs:
        raise ValueError("merge_dataset_configs requires at least one config")
    merged: dict[str, Any] = {}
    directories: list[dict[str, Any]] = []
    for cfg in configs:
        if not merged:
            for key, val in cfg.items():
                if key in ("directory", "name"):
                    continue
                merged[key] = val
        directories.extend(cfg.get("directory") or [])
    merged["directory"] = directories
    if "resolutions" not in merged:
        merged["resolutions"] = [1024]
    if "frame_buckets" not in merged:
        merged["frame_buckets"] = [1]
    return merged


__all__ = ["merge_dataset_configs"]
