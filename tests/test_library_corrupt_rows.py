"""Library DB uses integer primary keys (no string-id corruption)."""

from __future__ import annotations

from rengu_flow_ui import library_db


def test_dataset_ids_are_integers(ui_data_tmp) -> None:
    did = library_db.insert_dataset("resolutions = [512]\nframe_buckets = [1]\n")
    assert isinstance(did, int)
    items = library_db.list_datasets_summary()
    assert items[0]["id"] == did
    assert isinstance(items[0]["id"], int)
