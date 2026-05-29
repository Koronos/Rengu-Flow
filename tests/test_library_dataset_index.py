"""Dataset library index columns."""

from rengu_flow_ui import library_db


def test_read_returns_raw_content(ui_data_tmp) -> None:
    did = library_db.insert_dataset("1")
    assert library_db.read_dataset_text(did) == "1"
    row = library_db.search_datasets("", page=1, page_size=10)
    assert row["items"][0]["id"] == did


def test_partial_toml_indexes_directory_count(ui_data_tmp) -> None:
    toml_text = """resolutions = [1024]
frame_buckets = [1]

[[directory]]
path = "/tmp/x"
num_repeats = 1
"""
    did = library_db.insert_dataset(toml_text)
    item = library_db.search_datasets("", page=1, page_size=10)["items"][0]
    assert item["id"] == did
    assert item["directory_count"] == 1


def test_refresh_index_after_bad_write(ui_data_tmp) -> None:
    did = library_db.insert_dataset("not toml at all")
    library_db.refresh_dataset_index(did)
    item = library_db.search_datasets("", page=1, page_size=10)["items"][0]
    assert item["directory_count"] == 0
