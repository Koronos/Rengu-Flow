"""Managed storage location for dataset-prep recovery artifacts.

Prep keeps recovery data — caption-edit backups, quarantined images, and originals
backed up before in-place cleaning. These live under the app data dir, keyed by the
dataset folder, instead of a hidden ``.rengu_prep`` folder inside the dataset itself.
"""

from __future__ import annotations

import os
from pathlib import Path

from rengu_flow.data.cache_paths import directory_cache_id


def _data_root() -> Path:
    """App data dir. Mirrors ``rengu_flow_ui.settings.ui_data_dir`` without importing the
    UI package: ``RENGU_FLOW_UI_DATA`` (set by the launcher / UI server and inherited by
    the prep subprocess) wins; otherwise fall back to ``<repo>/data``."""
    base = os.environ.get("RENGU_FLOW_UI_DATA")
    if base:
        return Path(base).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data"


def prep_storage_dir(folder: str | Path) -> Path:
    """Per-dataset prep root: ``<data_dir>/prep/<folder-id>``.

    Keyed by the dataset folder's absolute path so artifacts stay associated as long as
    the folder doesn't move, and discoverable under the managed data dir rather than
    hidden inside the dataset. Does not create the directory.
    """
    return _data_root() / "prep" / directory_cache_id(folder)
