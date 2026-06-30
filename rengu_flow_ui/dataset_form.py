"""Dataset TOML ↔ flat form for the UI."""

from __future__ import annotations

import json
from typing import Any

import toml

from rengu_flow_ui.dataset_schema import get_dataset_schema

DIRECTORIES_KEY = "_directories"
# Written at the top of library dataset TOML for readability; never used by the trainer.
DATASET_UI_METADATA_KEYS = frozenset({"name"})
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
    "no_upscale",
    "shuffle_metadata",
    "online_captions",
    "subsample_ratio",
    "max_images",
    "subsample_shuffle",
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
    if not isinstance(entry, dict):
        return None
    path_val = entry.get("path")
    path = path_val.strip() if isinstance(path_val, str) else ""
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
        if key == "subsample_ratio":
            try:
                if float(val) >= 1.0:  # full dataset is the default; omit from TOML
                    continue
            except (TypeError, ValueError):
                continue
        if key == "max_images":
            try:
                val = int(val)
            except (TypeError, ValueError):
                continue
        if key == "subsample_shuffle" and val:
            continue  # rotating (True) is the default; omit from TOML
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


def strip_ui_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Remove UI-only top-level keys before validation or training."""
    if not isinstance(config, dict):
        return {}
    return {k: v for k, v in config.items() if k not in DATASET_UI_METADATA_KEYS}


def loads_for_training(content: str) -> dict[str, Any]:
    """Parse dataset TOML and drop UI-only keys (e.g. display ``name``)."""
    config = toml.loads(content)
    if not isinstance(config, dict):
        raise ValueError("Dataset TOML root must be a table.")
    return strip_ui_metadata(config)


def strip_display_name_from_toml(content: str) -> str:
    """Return TOML without the display-only ``name`` key."""
    stripped = (content or "").strip()
    if not stripped:
        return content or ""
    try:
        config = toml.loads(stripped)
    except Exception:
        return content
    if not isinstance(config, dict):
        return content
    cleaned = strip_ui_metadata(config)
    return toml.dumps(cleaned)


def embed_display_name(content: str, name: str | None) -> str:
    """Add or update ``name = "…"`` for human-readable TOML; identity stays the library ID."""
    label = (name or "").strip()
    body = strip_display_name_from_toml(content)
    if not label:
        return body
    if not body.strip():
        return toml.dumps({"name": label})
    try:
        config = toml.loads(body)
    except Exception:
        return content
    if not isinstance(config, dict):
        return content
    config["name"] = label
    return toml.dumps(config)


def _schema_known_root_keys(schema: dict[str, Any]) -> frozenset[str]:
    keys: set[str] = set()
    for section in schema.get("sections", []):
        for field in section.get("fields", []):
            keys.add(field["path"])
    return frozenset(keys)


def _schema_known_directory_keys(schema: dict[str, Any]) -> frozenset[str]:
    keys = {"path", "num_repeats", "augmentation", *DIRECTORY_OPTIONAL_KEYS}
    for field in schema.get("directory_fields", []):
        keys.add(field["path"])
    return frozenset(keys)


def _parse_toml_lenient_core(
    content: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Map user TOML to form fields; drop unknown keys. Does not apply schema defaults."""
    warnings: list[str] = []
    known_root = _schema_known_root_keys(schema)
    known_dir = _schema_known_directory_keys(schema)

    stripped = (content or "").strip()
    if not stripped:
        warnings.append("Empty file — nothing to show in the form builder yet.")
        return {DIRECTORIES_KEY: []}, warnings

    try:
        config = toml.loads(stripped)
    except Exception as e:
        warnings.append(f"Could not parse TOML ({e}). Fix the file or edit as raw TOML.")
        return {DIRECTORIES_KEY: []}, warnings

    if not isinstance(config, dict):
        warnings.append("Root value is not a TOML table — form builder cannot load this file.")
        return {DIRECTORIES_KEY: []}, warnings

    raw_dirs = config.get("directory")
    if raw_dirs is None:
        raw_dirs = []
    elif not isinstance(raw_dirs, list):
        warnings.append("[[directory]] is not a list — folders section not shown in the form builder.")
        raw_dirs = []

    directories: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_dirs):
        if not isinstance(raw, dict):
            warnings.append(f"Folder row {i + 1} is not a table — skipped in the form builder.")
            continue
        entry: dict[str, Any] = {}
        for key, val in raw.items():
            if key in known_dir:
                entry[key] = val
            else:
                warnings.append(f"Folder {i + 1}: key '{key}' is not supported in the form builder (still in your TOML file).")
        entry = _normalize_directory_lists(entry)
        path_val = entry.get("path")
        if isinstance(path_val, str):
            entry["path"] = path_val.strip()
        elif "path" not in entry:
            entry["path"] = ""
        if not entry.get("path"):
            warnings.append(
                f"Directory {i + 1}: path is empty — shown as not found in the form builder."
            )
        aug = entry.get("augmentation")
        if isinstance(aug, dict):
            entry["augmentation"] = _augmentation_for_form(aug)
        directories.append(entry)

    form: dict[str, Any] = {DIRECTORIES_KEY: directories}
    if not directories and isinstance(raw_dirs, list) and raw_dirs:
        warnings.append("No [[directory]] entries in the form builder (TOML unchanged).")

    dataset_section = config.get("dataset")
    if isinstance(dataset_section, dict):
        global_aug = dataset_section.get("augmentation")
        if isinstance(global_aug, dict) and global_aug:
            form["_dataset_augmentation"] = json.dumps(global_aug, indent=2)
    elif config.get("dataset") is not None:
        warnings.append("[dataset] section is not a table — not shown in the form builder.")

    for key, val in config.items():
        if key in ("directory", "dataset") or key in DATASET_UI_METADATA_KEYS:
            continue
        if key not in known_root:
            warnings.append(f"Top-level key '{key}' is not supported in the form builder (still in your TOML file).")
            continue
        if isinstance(val, list) and key in INTEGER_LIST_KEYS | NUMBER_LIST_KEYS:
            form[key] = val
        elif isinstance(val, (dict, list)):
            form[key] = json.dumps(val, indent=2) if isinstance(val, list) else json.dumps(val)
        else:
            form[key] = val

    return form, warnings


def parse_toml_to_form(
    content: str,
    schema: dict[str, Any] | None = None,
    *,
    fill_defaults: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Parse dataset TOML into a form model. Does not modify or re-serialize the source TOML."""
    schema = schema or get_dataset_schema()
    form, warnings = _parse_toml_lenient_core(content, schema)
    if fill_defaults:
        form = form_values_for_ui(form, schema)
    return form, warnings


def parse_toml(content: str) -> dict[str, Any]:
    form, _warnings = _parse_toml_lenient_core(content, get_dataset_schema())
    return form


def _complete_tag_dropout_rules(value: Any) -> list[dict[str, Any]]:
    """Return only well-formed tag-dropout rules (those with ``tags`` or ``tags_file``).

    Incomplete rows from the UI builder (empty ``tags`` and no ``tags_file``) are dropped
    so they never reach the trainer's TOML.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    rules: list[dict[str, Any]] = []
    for rule in value:
        if not isinstance(rule, dict):
            continue
        if rule.get("tags") or rule.get("tags_file"):
            rules.append(rule)
    return rules


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
            "shuffle_metadata",
            "online_captions",
            "subsample_ratio",
            "max_images",
            "subsample_shuffle",
        )
        if k in form and form[k] not in (None, "")
    }
    directories = form.get(DIRECTORIES_KEY) or []
    if isinstance(directories, str):
        try:
            directories = json.loads(directories)
        except json.JSONDecodeError:
            directories = []
    elif not isinstance(directories, list):
        directories = []
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
        if val == []:
            continue
        if key == "subsample_ratio":
            try:
                if float(val) >= 1.0:  # full dataset is the default; omit from TOML
                    continue
            except (TypeError, ValueError):
                continue
            config[key] = val
            continue
        if key == "max_images":
            try:
                config[key] = int(val)
            except (TypeError, ValueError):
                pass
            continue
        if key == "subsample_shuffle":
            if not val:  # rotating (True) is the default; omit from TOML
                config[key] = False
            continue
        if key == "tag_dropout_rules":
            rules = _complete_tag_dropout_rules(val)
            if rules:
                config[key] = rules
            continue
        if (
            key in INTEGER_LIST_KEYS | NUMBER_LIST_KEYS | JSON_LIST_KEYS
            or isinstance(val, str)
            and val.strip().startswith(("[", "{"))
        ):
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
            except json.JSONDecodeError:
                config[key] = val
            else:
                if parsed in ([], None, {}):
                    continue
                config[key] = parsed
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
