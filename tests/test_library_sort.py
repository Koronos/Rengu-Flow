"""Library list ordering (id, name, dates)."""

from __future__ import annotations

from pathlib import Path

from renga_flow_ui import library_db


def test_datasets_sort_by_name_asc(ui_data_tmp: Path) -> None:
    library_db.insert_dataset(
        "resolutions = [512]\nframe_buckets = [1]\n\n[[directory]]\npath = '/z'\nnum_repeats = 1\n",
        name="Zebra",
    )
    library_db.insert_dataset(
        "resolutions = [512]\nframe_buckets = [1]\n\n[[directory]]\npath = '/a'\nnum_repeats = 1\n",
        name="Alpha",
    )
    rows = library_db.list_datasets_summary(sort="name", order="asc")
    names = [r["name"] for r in rows]
    assert names.index("Alpha") < names.index("Zebra")


def test_datasets_sort_by_id_desc(ui_data_tmp: Path) -> None:
    a = library_db.insert_dataset(
        "resolutions = [512]\nframe_buckets = [1]\n\n[[directory]]\npath = '/a'\nnum_repeats = 1\n"
    )
    b = library_db.insert_dataset(
        "resolutions = [512]\nframe_buckets = [1]\n\n[[directory]]\npath = '/b'\nnum_repeats = 1\n"
    )
    rows = library_db.list_datasets_summary(sort="id", order="desc")
    ids = [r["id"] for r in rows]
    assert ids.index(b) < ids.index(a)
