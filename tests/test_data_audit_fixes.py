"""Regression tests for the data-pipeline audit (training-workflow-audit): each test pins one
finding that was verified against the code before the fix."""

from __future__ import annotations

import gc
import json
import os
import tarfile

import datasets
import pytest
import torch
from PIL import Image

from rengu_flow.data.cache_paths import directory_cache_id, directory_config_digest
from rengu_flow.data.control import ControlPairingError
from rengu_flow.data.dataset import (
    STAMP_COLUMNS,
    Dataset,
    DirectoryDataset,
    SizeBucketDataset,
    load_captions_json,
    _read_captions_from_txt_per_line,
)
from rengu_flow.data.manager import DatasetManager
from rengu_flow.data.preprocess_media import PreprocessMediaFile, convert_crop_and_resize
from rengu_flow.engine import select_backend
from rengu_flow.utils.cache import Cache

from test_dataset_control_integration import (
    RES,
    StubPipeline,
    _build_tree,
    _cache_and_batches,
    _caption,
    _dataset_config,
    _img,
)

pytestmark = pytest.mark.no_ui_db

BASE_CFG = {
    "resolutions": [64],
    "frame_buckets": [1],
    "min_ar": 0.5,
    "max_ar": 2.0,
    "num_ar_buckets": 4,
    "enable_ar_bucket": True,
}


def _dd(path, tmp_path, dataset_config=None, **directory):
    return DirectoryDataset(
        {"path": str(path), "num_repeats": 1, "shuffle_metadata": False, **directory},
        dataset_config or BASE_CFG,
        "sdxl",
        skip_dataset_validation=True,
        training_config={"cache_root": str(tmp_path / "cache")},
    )


def _rows(dd):
    return [row for ar in dd.ar_bucket_datasets for row in ar.metadata_dataset]


# --- item 2: image / mask stamps ------------------------------------------------------------


def test_replacing_a_target_in_place_reencodes_only_that_row(tmp_path):
    _build_tree(tmp_path)
    _cache_and_batches(tmp_path)
    gc.collect()
    again, _ = _cache_and_batches(tmp_path)
    assert again.vae_calls == []  # nothing changed -> nothing re-encoded
    del again, _

    target = tmp_path / "t2i" / "a.png"
    _img(target, (64, 64), (0, 0, 255))  # same path, new pixels
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    model, _ds = _cache_and_batches(tmp_path)
    # Before the stamp the latent was keyed by path only: the stale latent was served.
    assert len(model.vae_calls) == 1, model.vae_calls
    assert model.vae_calls[0]["controls"] is None


def test_metadata_rows_carry_file_stamps(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    _img(d / "a.png", (64, 64), "red")
    dd = _dd(d, tmp_path)
    dd.cache_metadata(regenerate_cache=True, cache_num_proc=1)
    row = _rows(dd)[0]
    st = (d / "a.png").stat()
    assert row["image_stamp"] == f"{st.st_size}:{st.st_mtime_ns}"
    assert row["mask_stamp"] == ""
    assert set(STAMP_COLUMNS) <= set(row)


def test_legacy_donor_rows_without_stamps_are_still_reused(tmp_path):
    """Caches written before the stamps existed must not all be re-encoded after the upgrade:
    a donor row WITHOUT stamp keys matches on the remaining identity; a donor row WITH a
    different stamp (the file changed) does not."""
    from rengu_flow.data.cache_utils import _match_donor_rows

    def row(spec, **extra):
        return {"latents": torch.zeros(4), "image_spec": [None, spec], **extra}

    legacy = Cache(tmp_path / "legacy", "fp")
    legacy.add(row("old.png"))
    legacy.finalize_current_shard()
    stamped = Cache(tmp_path / "stamped", "fp")
    stamped.add(row("new.png", image_stamp="1:1", mask_stamp=""))
    stamped.finalize_current_shard()

    ds = datasets.Dataset.from_dict(
        {
            "image_spec": [[None, "old.png"], [None, "new.png"]],
            "image_stamp": ["9:9", "2:2"],  # new.png was rewritten since it was cached
            "mask_stamp": ["", ""],
        }
    )
    matched = _match_donor_rows(
        ds, [0, 1], ("image_spec", "image_stamp", "mask_stamp"), [legacy, stamped]
    )
    assert set(matched) == {0}  # legacy reused; the changed stamped row is re-encoded


# --- item 6 / 10: encodings, captions.json -----------------------------------------------------


def test_txt_captions_with_bom_are_decoded_as_utf8(tmp_path):
    f = tmp_path / "c.txt"
    f.write_bytes("﻿cafe señor\nsegunda\n".encode("utf-8"))
    assert _read_captions_from_txt_per_line(str(f)) == ["cafe señor", "segunda"]


def test_captions_json_accepts_bare_string_and_rejects_junk(tmp_path):
    f = tmp_path / "captions.json"
    f.write_text(json.dumps({"a.png": "one caption", "b.png": ["x", "y"]}), encoding="utf-8")
    assert load_captions_json(f) == {"a.png": ["one caption"], "b.png": ["x", "y"]}
    f.write_text(json.dumps({"a.png": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="a.png"):
        load_captions_json(f)


def test_captions_json_bare_string_flows_into_metadata(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    _img(d / "a.png", (64, 64), "red")
    (d / "captions.json").write_text(json.dumps({"a.png": "just one"}), encoding="utf-8")
    dd = _dd(d, tmp_path)
    dd.cache_metadata(regenerate_cache=True, cache_num_proc=1)
    assert _rows(dd)[0]["caption"] == ["just one"]


def test_online_captions_key_lookup_uses_the_file_name(tmp_path):
    sb = object.__new__(SizeBucketDataset)
    sb.uncond_fraction = 0.0
    sb.captions_dict = {"a.png": ["cap zero", "cap one"]}
    sb._caption_variants_expanded = False
    sb.text_embedding_datasets = []
    sb.uncond_text_embeddings = []
    sb.tag_dropout = type("T", (), {"enabled": False})()
    entry = {"image_spec": [None, str(tmp_path / "a.png")], "caption_number": 1, "caption": ""}
    out = sb._sample_from_entry(entry, {"latents": torch.zeros(1)})
    assert out["caption"] == "cap one"


# --- item 8: source signature ---------------------------------------------------------------


def test_source_signature_ignores_private_config_keys(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    _img(d / "a.png", (64, 64), "red")
    a = _dd(d, tmp_path, dataset_config={**BASE_CFG, "_dataset_toml_path": "/stage/job1/ds.toml"})
    b = _dd(d, tmp_path, dataset_config={**BASE_CFG, "_dataset_toml_path": "/stage/job2/ds.toml"})
    assert a._source_signature() == b._source_signature()
    c = _dd(d, tmp_path, dataset_config={**BASE_CFG, "resolutions": [128]})
    assert a._source_signature() != c._source_signature()


# --- item 9: cache dir collisions ------------------------------------------------------------


def test_same_path_with_different_settings_gets_its_own_cache_dir(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    _img(d / "a.png", (64, 64), "red")
    cfg = {**BASE_CFG, "directory": []}
    train = Dataset({**cfg, "directory": [{"path": str(d), "num_repeats": 1}]}, StubPipeline(),
                    skip_dataset_validation=True, training_config={"cache_root": str(tmp_path / "c")})
    evald = Dataset({**cfg, "directory": [{"path": str(d), "num_repeats": 1, "resolutions": [32]}]},
                    StubPipeline(), skip_dataset_validation=True,
                    training_config={"cache_root": str(tmp_path / "c")})
    same = Dataset({**cfg, "directory": [{"path": str(d), "num_repeats": 1}]}, StubPipeline(),
                   skip_dataset_validation=True, training_config={"cache_root": str(tmp_path / "c")})
    original = train.directory_datasets[0].cache_dir
    manager = DatasetManager(StubPipeline(), backend=select_backend({"engine": "accelerate"}))
    manager.register(train)
    manager.register(evald)
    manager.register(same)
    assert train.directory_datasets[0].cache_dir == original  # first user: existing cache kept
    assert evald.directory_datasets[0].cache_dir != original  # different settings: separated
    assert same.directory_datasets[0].cache_dir == original  # identical settings: shared


def test_directory_cache_id_unchanged_without_disambiguator(tmp_path):
    assert directory_cache_id(tmp_path) == directory_cache_id(tmp_path, None)
    assert directory_cache_id(tmp_path) != directory_cache_id(tmp_path, "x")
    assert directory_config_digest({"path": "a", "_x": 1}) == directory_config_digest({"path": "a"})


# --- item 11: edit datasets ignore non-image files -----------------------------------------


def test_edit_targets_ignore_junk_files(tmp_path):
    _img(tmp_path / "t" / "a.png", (64, 64), "red")
    _img(tmp_path / "c" / "a.png", (64, 64), "blue")
    for junk in ("Thumbs.db", "desktop.ini", "notes.toml", "a.caption"):
        (tmp_path / "t" / junk).write_text("junk", encoding="utf-8")
    dd = _dd(tmp_path / "t", tmp_path, control_path=str(tmp_path / "c"))
    dd.cache_metadata(regenerate_cache=True, cache_num_proc=1)
    assert len(_rows(dd)) == 1


def test_edit_targets_still_fail_on_a_genuinely_unpaired_image(tmp_path):
    _img(tmp_path / "t" / "a.png", (64, 64), "red")
    (tmp_path / "c").mkdir()
    dd = _dd(tmp_path / "t", tmp_path, control_path=str(tmp_path / "c"))
    with pytest.raises(ControlPairingError):
        dd.cache_metadata(regenerate_cache=True, cache_num_proc=1)


# --- item 12: tar ----------------------------------------------------------------------------


def test_tar_captions_come_from_the_tar_and_txt_members_are_not_images(tmp_path, monkeypatch):
    d = tmp_path / "shards"
    d.mkdir()
    img = tmp_path / "x.png"
    _img(img, (64, 64), "red")
    cap = tmp_path / "x.txt"
    cap.write_text("a red square\nsecond line\n", encoding="utf-8")
    with tarfile.open(d / "s.tar", "w") as tar:
        tar.add(img, arcname="x.png")
        tar.add(cap, arcname="x.txt")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    dd = _dd(d, tmp_path)
    dd.cache_metadata(regenerate_cache=True, cache_num_proc=1)
    rows = _rows(dd)
    assert len(rows) == 1  # the .txt member is not enumerated as an image
    assert rows[0]["caption"] == ["a red square", "second line"]


def test_preprocess_uses_one_tar_handle_per_thread(tmp_path):
    import threading

    img = tmp_path / "x.png"
    _img(img, (64, 64), "red")
    tar_path = tmp_path / "s.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(img, arcname="x.png")
    fn = PreprocessMediaFile({}, support_video=False)
    main_handle = fn._tar_for(str(tar_path))
    seen = {}

    def other():
        seen["handle"] = fn._tar_for(str(tar_path))

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert seen["handle"] is not main_handle
    assert fn._tar_for(str(tar_path)) is main_handle


# --- item 13: alpha ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["LA", "RGBA"])
def test_transparent_pixels_composite_on_white(mode):
    img = Image.new(mode, (8, 8), (0, 0) if mode == "LA" else (0, 0, 0, 0))
    out = convert_crop_and_resize(img, (8, 8))
    assert out.getpixel((0, 0)) == (255, 255, 255)


# --- item 7: parquet column config reaches the preprocessor -----------------------------------


def test_preprocess_honours_parquet_image_column(tmp_path):
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buf, "JPEG")
    f = tmp_path / "d.parquet"
    pq.write_table(pa.table({"pic": [buf.getvalue()], "caption": ["x"]}), f)
    fn = PreprocessMediaFile({}, support_video=False)
    fn.parquet_config_resolver = lambda path: {"parquet_image_column": "pic"}
    out = fn((str(f), "pq/0"), None, size_bucket=(64, 64, 1))
    assert out[0][2] is True


# --- item 15: caption_number is the ORIGINAL caption index -------------------------------------


def test_caption_number_indexes_the_original_caption_list(tmp_path):
    """The text-embedding cache is keyed (image, original caption index); the iteration order used
    to store the slot after a shuffle, so caption_number pointed at another caption's embedding."""
    originals = {f"img{i}.png": [f"img{i} cap{j}" for j in range(4)] for i in range(12)}
    metadata = datasets.Dataset.from_dict(
        {
            "image_spec": [[None, k] for k in originals],
            "caption": list(originals.values()),
        }
    )
    sb = SizeBucketDataset(
        metadata, {"path": str(tmp_path), "num_repeats": 1}, (512, 512, 1), tmp_path / "cache", None
    )
    sb.cache_latents(
        lambda ex, rank: {"latents": torch.zeros(len(ex["image_spec"]), 4)},
        regenerate_cache=True,
        trust_cache=False,
    )
    moved = 0
    for row in sb.iteration_order:
        name = row["image_spec"][1]
        assert row["caption"] == originals[name][row["caption_number"]]
        moved += originals[name].index(row["caption"]) != 0
    assert len(sb.iteration_order) == 48 and moved > 0


# --- item 16: nested directory roots ---------------------------------------------------------


def test_augmentation_resolver_prefers_the_most_specific_root(tmp_path):
    from types import SimpleNamespace

    outer, inner = tmp_path / "data", tmp_path / "data" / "sub"
    inner.mkdir(parents=True)

    def d(path, tag):
        return SimpleNamespace(
            _aug_enabled=True,
            directory_config={"path": str(path)},
            _resolved_augmentation={"tag": tag},
            _aug_fingerprint=tag,
        )

    ds = object.__new__(Dataset)
    ds.directory_datasets = [d(outer, "outer"), d(inner, "inner")]
    resolve = ds.get_augmentation_resolver()
    assert resolve((None, str(inner / "a.png")))[1] == "inner"
    assert resolve((None, str(outer / "b.png")))[1] == "outer"
    cfg = ds.get_directory_config_resolver()
    assert cfg(str(inner / "a.png"))["path"] == str(inner)
