"""Dataset TOML ↔ flat form for the UI."""

from __future__ import annotations

import json
from typing import Any

import toml

from renga_flow_ui.dataset_schema import get_directory_row_template

DIRECTORIES_KEY = "_directories"
INTEGER_LIST_KEYS = frozenset({"resolutions", "frame_buckets"})
NUMBER_LIST_KEYS = frozenset({"ar_buckets"})
JSON_LIST_KEYS = frozenset({"size_buckets"})

# Optional keys on each [[directory]] table (see docs/user/dataset-config.md).
DIRECTORY_OPTIONAL_KEYS = (
    "directory_caption",
    "mask_path",
    "control_path",
    "default_mask_file",
    "resolutions",
    "frame_buckets",
    "enable_ar_bucket",
    "min_ar",
    "max_ar",
    "num_ar_buckets",
    "ar_buckets",
    "size_buckets",
    "cache_shuffle_num",
    "cache_shuffle_delimiter",
    "shuffle_tags",
    "shuffle_metadata",
    "online_captions",
)


def _normalize_directory_lists(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    for key in INTEGER_LIST_KEYS | NUMBER_LIST_KEYS:
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    if "size_buckets" in out and isinstance(out["size_buckets"], str):
        try:
            out["size_buckets"] = json.loads(out["size_buckets"])
        except json.JSONDecodeError:
            pass
    return out


def _directory_row_for_toml(entry: dict[str, Any], global_values: dict[str, Any]) -> dict[str, Any] | None:
    path = (entry.get("path") or "").strip()
    if not path:
        return None
    row: dict[str, Any] = {
        "path": path,
        "num_repeats": int(entry.get("num_repeats") or 1),
    }
    cap = entry.get("directory_caption")
    if cap not in (None, ""):
        row["directory_caption"] = cap

    aug = entry.get("augmentation")
    if isinstance(aug, str):
        try:
            aug = json.loads(aug)
        except json.JSONDecodeError:
            aug = None
    if isinstance(aug, dict) and aug:
        aug_out = dict(aug)
        strat = aug_out.get("strategies")
        if isinstance(strat, str):
            try:
                aug_out["strategies"] = json.loads(strat)
            except json.JSONDecodeError:
                pass
        row["augmentation"] = aug_out

    for key in DIRECTORY_OPTIONAL_KEYS:
        if key == "directory_caption":
            continue
        if key not in entry:
            continue
        val = entry[key]
        if val is None or val == "":
            continue
        if key in global_values and val == global_values.get(key):
            continue
        if key in INTEGER_LIST_KEYS | NUMBER_LIST_KEYS | JSON_LIST_KEYS:
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    continue
            if val in ([], None):
                continue
        row[key] = val
    return row


def _augmentation_for_form(aug: dict[str, Any]) -> dict[str, Any]:
    out = dict(aug)
    strategies = out.get("strategies")
    if isinstance(strategies, dict):
        out["strategies"] = json.dumps(strategies, indent=2)
    return out


def parse_toml(content: str) -> dict[str, Any]:
    config = toml.loads(content)
    raw_dirs = config.get("directory") or []
    form: dict[str, Any] = {
        DIRECTORIES_KEY: [_normalize_directory_lists(d) for d in raw_dirs if isinstance(d, dict)],
    }
    dataset_section = config.get("dataset")
    if isinstance(dataset_section, dict):
        global_aug = dataset_section.get("augmentation")
        if isinstance(global_aug, dict) and global_aug:
            form["_dataset_augmentation"] = json.dumps(global_aug, indent=2)
    for entry in form[DIRECTORIES_KEY]:
        aug = entry.get("augmentation")
        if isinstance(aug, dict):
            entry["augmentation"] = _augmentation_for_form(aug)
    for key, val in config.items():
        if key == "directory":
            continue
        if key == "dataset":
            continue
        if isinstance(val, list) and key in INTEGER_LIST_KEYS | NUMBER_LIST_KEYS:
            form[key] = val
        elif isinstance(val, (dict, list)):
            form[key] = json.dumps(val, indent=2) if isinstance(val, list) else json.dumps(val)
        else:
            form[key] = val
    if not form[DIRECTORIES_KEY]:
        form[DIRECTORIES_KEY] = [get_directory_row_template()]
    return form


def form_to_toml(form: dict[str, Any]) -> str:
    config: dict[str, Any] = {}
    global_values = {
        k: form[k]
        for k in (
            *INTEGER_LIST_KEYS,
            *NUMBER_LIST_KEYS,
            *JSON_LIST_KEYS,
            "enable_ar_bucket",
            "min_ar",
            "max_ar",
            "num_ar_buckets",
            "shuffle_tags",
            "cache_shuffle_num",
            "cache_shuffle_delimiter",
            "shuffle_metadata",
            "online_captions",
        )
        if k in form and form[k] not in (None, "")
    }
    directories = form.get(DIRECTORIES_KEY) or []
    if isinstance(directories, str):
        directories = json.loads(directories)
    cleaned_dirs: list[dict[str, Any]] = []
    for entry in directories:
        if not isinstance(entry, dict):
            continue
        row = _directory_row_for_toml(entry, global_values)
        if row:
            cleaned_dirs.append(row)
    if cleaned_dirs:
        config["directory"] = cleaned_dirs

    global_aug_raw = form.get("_dataset_augmentation")
    if global_aug_raw not in (None, ""):
        try:
            global_aug = (
                json.loads(global_aug_raw)
                if isinstance(global_aug_raw, str)
                else global_aug_raw
            )
            if isinstance(global_aug, dict) and global_aug:
                config.setdefault("dataset", {})["augmentation"] = global_aug
        except json.JSONDecodeError:
            pass

    for key, val in form.items():
        if key == "_dataset_augmentation":
            continue
        if key == DIRECTORIES_KEY or val is None or val == "":
            continue
        if (
            key in INTEGER_LIST_KEYS | NUMBER_LIST_KEYS | JSON_LIST_KEYS
            or isinstance(val, str)
            and val.strip().startswith(("[", "{"))
        ):
            try:
                config[key] = json.loads(val) if isinstance(val, str) else val
            except json.JSONDecodeError:
                config[key] = val
        else:
            config[key] = val
    return toml.dumps(config)


def form_values_for_ui(form: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Fill schema defaults for keys absent from parsed dataset TOML."""
    out = dict(form)
    for section in schema.get("sections", []):
        for field in section.get("fields", []):
            path = field["path"]
            if path not in out and "default" in field:
                out[path] = field["default"]
    return out
