"""Paginated library search."""

from __future__ import annotations

from pathlib import Path

from rengu_flow_ui import datasets_store, library_db


def test_search_datasets_paginated(ui_data_tmp: Path) -> None:
    toml = (
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/tmp/a'\nnum_repeats = 1\n"
    )
    ids = [datasets_store.insert_dataset(toml) for _ in range(5)]

    r = library_db.search_datasets(str(ids[0]), page=1, page_size=3)
    assert r["total"] >= 1
    assert len(r["items"]) >= 1
