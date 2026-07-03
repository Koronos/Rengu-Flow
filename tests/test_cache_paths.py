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

# --- dataset_cache_id stability (regression: staged TOML copies must not move the cache) ---


def test_dataset_cache_id_ignores_toml_path(tmp_path):
    """The UI stages the dataset TOML into a per-job folder; a path-keyed id regenerated
    every cache on every run. The id must depend only on the [[directory]] paths."""
    from rengu_flow.data.cache_paths import dataset_cache_id

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    base = {"resolutions": [512], "directory": [{"path": str(img_dir)}]}
    a = dataset_cache_id({**base, "_dataset_toml_path": str(tmp_path / "job1" / "d.toml")})
    b = dataset_cache_id({**base, "_dataset_toml_path": str(tmp_path / "job2" / "d.toml")})
    c = dataset_cache_id(base)  # no staged path at all
    assert a == b == c


def test_dataset_cache_id_ignores_settings_changes(tmp_path):
    """Settings (resolutions, captions, repeats) are handled by bucket dirs and content
    fingerprints downstream; they must not move the whole namespace (full regen)."""
    from rengu_flow.data.cache_paths import dataset_cache_id

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    a = dataset_cache_id({"resolutions": [512], "directory": [{"path": str(img_dir)}]})
    b = dataset_cache_id(
        {
            "resolutions": [1024],
            "cached_caption_variants": 4,
            "directory": [{"path": str(img_dir), "num_repeats": 7}],
        }
    )
    assert a == b


def test_dataset_cache_id_differs_per_directory_set(tmp_path):
    from rengu_flow.data.cache_paths import dataset_cache_id

    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir(), d2.mkdir()
    one = dataset_cache_id({"directory": [{"path": str(d1)}]})
    both = dataset_cache_id({"directory": [{"path": str(d1)}, {"path": str(d2)}]})
    # order-independent
    both_rev = dataset_cache_id({"directory": [{"path": str(d2)}, {"path": str(d1)}]})
    assert one != both
    assert both == both_rev


def test_resolve_directory_cache_dir_relocates_legacy_cache(tmp_path):
    """Caches built under the old TOML-path-keyed id are renamed to the new id once,
    preserving their contents instead of regenerating."""
    from rengu_flow.data.cache_paths import (
        _legacy_dataset_cache_id,
        dataset_cache_id,
        resolve_directory_cache_dir,
    )

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ds_cfg = {
        "directory": [{"path": str(img_dir)}],
        "_dataset_toml_path": str(tmp_path / "d.toml"),
    }
    training = {"cache_root": str(tmp_path / "root")}
    root = tmp_path / "root"
    legacy_dir = root / _legacy_dataset_cache_id(ds_cfg) / directory_cache_id(img_dir) / "krea2"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "marker.txt").write_text("cached")

    out = resolve_directory_cache_dir(ds_cfg, img_dir, "krea2", training_config=training)
    assert out == root / dataset_cache_id(ds_cfg) / directory_cache_id(img_dir) / "krea2"
    assert (out / "marker.txt").read_text() == "cached"
    assert not (root / _legacy_dataset_cache_id(ds_cfg)).exists()
