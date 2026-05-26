"""Paginated library search."""

from __future__ import annotations

from pathlib import Path

from renga_flow_ui import configs_store, datasets_store, library_db


def test_search_configs_paginated(ui_data_tmp: Path) -> None:
    for i in range(25):
        configs_store.write_config_text(
            f"cfg_{i:02d}",
            f'dataset = "x"\n[model]\ntype = "sdxl"\n',
        )

    page1 = library_db.search_configs("", page=1, page_size=10)
    assert page1["total"] == 25
    assert len(page1["items"]) == 10

    filtered = library_db.search_configs("cfg_01", page=1, page_size=50)
    assert filtered["total"] == 1
    assert filtered["items"][0]["id"] == "cfg_01"


def test_search_datasets_paginated(ui_data_tmp: Path) -> None:
    toml = (
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/tmp/a'\nnum_repeats = 1\n"
    )
    for i in range(5):
        datasets_store.write_dataset_text(f"ds_{i}", toml)

    r = library_db.search_datasets("ds_", page=1, page_size=3)
    assert r["total"] == 5
    assert len(r["items"]) == 3
