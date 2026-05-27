"""Paginated library search."""

from __future__ import annotations

from pathlib import Path

from renga_flow_ui import configs_store, datasets_store, library_db


def test_search_configs_paginated(ui_data_tmp: Path) -> None:
    ids = []
    for _ in range(25):
        ids.append(
            configs_store.insert_config(f'dataset = "x"\n[model]\ntype = "sdxl"\n')
        )

    page1 = library_db.search_configs("", page=1, page_size=10)
    assert page1["total"] == 25
    assert len(page1["items"]) == 10

    target = ids[1]
    filtered = library_db.search_configs(str(target), page=1, page_size=50)
    assert any(item["id"] == target for item in filtered["items"])


def test_search_datasets_paginated(ui_data_tmp: Path) -> None:
    toml = (
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/tmp/a'\nnum_repeats = 1\n"
    )
    ids = [datasets_store.insert_dataset(toml) for _ in range(5)]

    r = library_db.search_datasets(str(ids[0]), page=1, page_size=3)
    assert r["total"] >= 1
    assert len(r["items"]) >= 1
