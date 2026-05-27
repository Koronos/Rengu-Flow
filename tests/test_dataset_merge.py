"""Tests for merging multiple dataset TOML configs."""

from renga_flow.config.dataset_merge import merge_dataset_configs
from renga_flow.config.loader import load_dataset_config, normalize_dataset_paths


def test_normalize_dataset_paths() -> None:
    assert normalize_dataset_paths("  a.toml ") == ["a.toml"]
    assert normalize_dataset_paths(["a.toml", " ", "b.toml"]) == ["a.toml", "b.toml"]
    assert normalize_dataset_paths(None) == []
    assert normalize_dataset_paths([]) == []


def test_merge_dataset_configs_combines_directories() -> None:
    a = {
        "resolutions": [512],
        "frame_buckets": [1],
        "directory": [{"path": "/a", "num_repeats": 1}],
    }
    b = {
        "resolutions": [768],
        "directory": [{"path": "/b", "num_repeats": 2}],
    }
    merged = merge_dataset_configs([a, b])
    assert merged["resolutions"] == [512]
    assert len(merged["directory"]) == 2
    assert merged["directory"][1]["path"] == "/b"


def test_load_dataset_config_multiple_paths(tmp_path) -> None:
    ds1 = tmp_path / "one.toml"
    ds2 = tmp_path / "two.toml"
    ds1.write_text(
        'resolutions = [1024]\nframe_buckets = [1]\n\n[[directory]]\npath = "/one"\nnum_repeats = 1\n',
        encoding="utf-8",
    )
    ds2.write_text(
        'resolutions = [512]\n\n[[directory]]\npath = "/two"\nnum_repeats = 1\n',
        encoding="utf-8",
    )
    loaded = load_dataset_config({"dataset": [str(ds1), str(ds2)]})
    assert loaded is not None
    assert len(loaded["directory"]) == 2
    assert loaded["resolutions"] == [1024]
