"""Tests for cache_root path resolution."""

from __future__ import annotations

import logging

from rengu_flow.data.cache_paths import (
    directory_cache_id,
    resolve_cache_root,
    resolve_directory_cache_dir,
)


def test_resolve_cache_root_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = resolve_cache_root({})
    assert root.name == "cache"


def test_resolve_cache_root_from_training_config(tmp_path):
    custom = tmp_path / "my_caches"
    root = resolve_cache_root({"cache_root": str(custom)})
    assert root == custom.resolve()


def test_resolve_directory_cache_dir_structure(tmp_path):
    ds_cfg = {"_dataset_toml_path": str(tmp_path / "data.toml")}
    dir_path = tmp_path / "images"
    dir_path.mkdir()
    training = {}
    out = resolve_directory_cache_dir(
        ds_cfg, dir_path, "sdxl", training_config=training
    )
    assert out.name == "sdxl"
    assert out.parent.name == directory_cache_id(dir_path)
    assert resolve_cache_root(training) in out.parents


def test_legacy_dataset_cache_root_fallback(tmp_path, caplog):
    custom = tmp_path / "legacy_root"
    ds_cfg = {"cache_root": str(custom)}
    with caplog.at_level(logging.WARNING):
        root = resolve_cache_root({}, dataset_config=ds_cfg)
    assert root == custom.resolve()
    assert "deprecated" in caplog.text.lower()


def test_legacy_dataset_cache_root_ignored_when_training_set(tmp_path, caplog):
    train_root = tmp_path / "train_root"
    ds_root = tmp_path / "ds_root"
    ds_cfg = {"cache_root": str(ds_root)}
    with caplog.at_level(logging.WARNING):
        root = resolve_cache_root(
            {"cache_root": str(train_root)}, dataset_config=ds_cfg
        )
    assert root == train_root.resolve()
    assert "ignored" in caplog.text.lower()



def test_validate_training_cache_root_empty_string():
    from rengu_flow.config.validation import collect_validation_errors

    issues = collect_validation_errors({"cache_root": "   "})
    assert any("cache_root" in e for e in issues)
