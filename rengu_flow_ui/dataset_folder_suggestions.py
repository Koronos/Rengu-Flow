"""Aggregate folder paths from library datasets for UI suggestions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from rengu_flow_ui import datasets_store, library_db
from rengu_flow_ui.dataset_image_preview import issue_image_token
from rengu_flow_ui.dataset_scan import scan_folder


def _normalize_path_key(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _preview_token_for_scan(scan: dict[str, Any]) -> str | None:
    if not scan.get("ok"):
        return None
    root = Path(scan["path"])
    samples = scan.get("sample_files") or []
    if not samples:
        return None
    name = samples[0]
    if not isinstance(name, str):
        return None
    return issue_image_token(0, name, root)


def collect_folder_suggestions(
    *,
    exclude_dataset_id: str | int | None = None,
) -> dict[str, Any]:
    """
  Return unique folder paths used across library datasets.

  - ``suggestions``: paths that still exist on disk (for the picker strip).
  - ``missing``: paths referenced in library but folder missing (notice only).
  """
    by_path: dict[str, dict[str, Any]] = {}

    exclude_id: int | None = None
    if exclude_dataset_id is not None:
        try:
            exclude_id = library_db._coerce_record_id(exclude_dataset_id)
        except FileNotFoundError:
            exclude_id = None

    for row in library_db.list_datasets_summary():
        did = row["id"]
        if exclude_id is not None and did == exclude_id:
            continue
        try:
            text = datasets_store.read_dataset_text(did)
            config = toml.loads(text)
        except Exception:
            continue
        directories = config.get("directory") or []
        if not isinstance(directories, list):
            continue
        for entry in directories:
            if not isinstance(entry, dict):
                continue
            raw = (entry.get("path") or "").strip()
            if not raw:
                continue
            try:
                key = _normalize_path_key(raw)
            except (OSError, ValueError):
                key = raw
            rec = by_path.get(key)
            if rec is None:
                rec = {
                    "path": raw,
                    "resolved_path": key,
                    "source_datasets": [],
                }
                by_path[key] = rec
            if did not in rec["source_datasets"]:
                rec["source_datasets"].append(did)

    suggestions: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for rec in by_path.values():
        scan = scan_folder(rec["path"], count_cap=2_000)
        basename = Path(rec["path"]).name or rec["path"]
        item = {
            "path": rec["path"],
            "resolved_path": rec.get("resolved_path") or rec["path"],
            "basename": basename,
            "source_datasets": list(rec["source_datasets"]),
            "exists": bool(scan.get("ok")),
            "image_count": scan.get("image_count", 0) if scan.get("ok") else 0,
            "video_count": scan.get("video_count", 0) if scan.get("ok") else 0,
            "count_capped": bool(scan.get("count_capped")) if scan.get("ok") else False,
            "error": scan.get("error") if not scan.get("ok") else None,
            "preview_token": _preview_token_for_scan(scan),
        }
        if scan.get("ok"):
            suggestions.append(item)
        else:
            missing.append(item)

    suggestions.sort(
        key=lambda x: (-len(x["source_datasets"]), x["basename"].lower())
    )
    missing.sort(key=lambda x: x["basename"].lower())

    return {"suggestions": suggestions, "missing": missing}
