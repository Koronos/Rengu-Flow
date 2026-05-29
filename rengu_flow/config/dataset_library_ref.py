"""Parse and format ``rengu-flow-dataset:`` refs in training configs.

Refs may include an optional human-readable suffix after the numeric id::

    rengu-flow-dataset:3
    rengu-flow-dataset:3:artista 1

Only the id is used when resolving library rows or staging jobs.
"""

from __future__ import annotations

from typing import Any

DATASET_REF_PREFIX = "rengu-flow-dataset:"


def is_library_dataset_ref(value: str) -> bool:
    return isinstance(value, str) and value.strip().startswith(DATASET_REF_PREFIX)


def library_dataset_id_from_ref(value: str) -> int:
    """Return the library dataset id; ignore any ``:label`` suffix."""
    if not is_library_dataset_ref(value):
        raise ValueError(f"Not a dataset library ref: {value!r}")
    rest = value.strip()[len(DATASET_REF_PREFIX) :].strip()
    if not rest:
        raise ValueError(f"Invalid dataset library ref: {value!r}")
    id_part = rest.split(":", 1)[0].strip()
    if not id_part.isdigit():
        raise ValueError(f"Invalid dataset library ref: {value!r}")
    return int(id_part)


def library_dataset_label_from_ref(value: str) -> str | None:
    """Optional display suffix from a ref string (not authoritative)."""
    if not is_library_dataset_ref(value):
        return None
    rest = value.strip()[len(DATASET_REF_PREFIX) :].strip()
    if ":" not in rest:
        return None
    label = rest.split(":", 1)[1].strip()
    return label or None


def dataset_library_ref(dataset_id: str | int, display_name: str | None = None) -> str:
    """Build a ref for TOML; ``display_name`` is cosmetic only."""
    did = int(dataset_id)
    base = f"{DATASET_REF_PREFIX}{did}"
    label = (display_name or "").strip()
    if label:
        return f"{base}:{label}"
    return base


def canonical_dataset_ref(value: str) -> str:
    """Normalize to ``rengu-flow-dataset:<id>`` (strip display suffix)."""
    if not is_library_dataset_ref(value):
        return value.strip()
    return dataset_library_ref(library_dataset_id_from_ref(value))


def iter_dataset_path_strings(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(field_name, path_or_ref)`` for ``dataset`` and ``eval_datasets`` entries."""
    out: list[tuple[str, str]] = []

    dataset = config.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        out.append(("dataset", dataset.strip()))
    elif isinstance(dataset, list):
        for item in dataset:
            if isinstance(item, str) and item.strip():
                out.append(("dataset", item.strip()))

    eval_datasets = config.get("eval_datasets")
    if isinstance(eval_datasets, list):
        for entry in eval_datasets:
            if isinstance(entry, str) and entry.strip():
                out.append(("eval_datasets", entry.strip()))
            elif isinstance(entry, dict):
                cfg = entry.get("config")
                if isinstance(cfg, str) and cfg.strip():
                    out.append(("eval_datasets", cfg.strip()))

    return out


def collect_script_dataset_library_ref_issues(config: dict[str, Any]) -> list[str]:
    """Errors when training config uses UI library refs outside the Rengu UI."""
    issues: list[str] = []
    seen: set[tuple[str, str]] = set()

    for field, entry in iter_dataset_path_strings(config):
        if not is_library_dataset_ref(entry):
            continue
        field_label = "`dataset`" if field == "dataset" else "`eval_datasets`"
        try:
            did = library_dataset_id_from_ref(entry)
        except ValueError:
            issues.append(
                f"{field_label} entry {entry!r} looks like a UI dataset library reference "
                "but has an invalid id. Use a path to an exported dataset .toml file, "
                "or run training from the Rengu UI."
            )
            continue

        dedupe_key = (field, dataset_library_ref(did))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        export_hint = (
            f"Export dataset #{did} from the Rengu UI (Datasets → open dataset → Export) "
            "and set that .toml path in your config."
        )
        issues.append(
            f"{field_label} uses UI-only library reference {entry.strip()!r}. "
            "Refs like `rengu-flow-dataset:` are resolved only when a job is started from the UI. "
            f"For `python -m rengu_flow.main` or other script runs, {export_hint}"
        )

    return issues


__all__ = [
    "DATASET_REF_PREFIX",
    "canonical_dataset_ref",
    "collect_script_dataset_library_ref_issues",
    "dataset_library_ref",
    "is_library_dataset_ref",
    "iter_dataset_path_strings",
    "library_dataset_id_from_ref",
    "library_dataset_label_from_ref",
]
