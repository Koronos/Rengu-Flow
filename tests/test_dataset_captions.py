"""Dataset caption formats: per-line .txt, captions.json, and multi-caption training rows."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import datasets
import pytest
import toml
import torch

from rengu_flow.data.dataset import (
    CAPTIONS_JSON_FILE,
    DirectoryDataset,
    FolderSubsampler,
    SizeBucketDataset,
    _read_captions_from_txt_per_line,
)
from rengu_flow.data.dump_dataset import dump_dataset

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)

MINIMAL_DATASET_CONFIG = {
    "resolutions": [512],
    "frame_buckets": [1],
    "min_ar": 0.5,
    "max_ar": 2.0,
    "num_ar_buckets": 4,
}


def _write_dataset_toml(tmp_path: Path, img_dir: Path, **directory_extra) -> Path:
    directory = {"path": str(img_dir), "num_repeats": 1, **directory_extra}
    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.write_text(
        toml.dumps({**MINIMAL_DATASET_CONFIG, "directory": [directory]}),
        encoding="utf-8",
    )
    return dataset_toml


def _copy_fixture_image(dest: Path, stem: str = "sample") -> Path:
    assert FIXTURE_JPG.is_file(), "smoke_cc0 fixture required"
    out = dest / f"{stem}.jpg"
    shutil.copy(FIXTURE_JPG, out)
    return out


@pytest.fixture
def img_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    return d


def test_read_captions_from_txt_per_line_multiline(tmp_path):
    txt = tmp_path / "cap.txt"
    txt.write_text("first caption\n\nsecond caption\n  \nthird\n", encoding="utf-8")
    assert _read_captions_from_txt_per_line(str(txt)) == [
        "first caption",
        "second caption",
        "third",
    ]


def test_read_captions_from_txt_empty_file_yields_single_empty_string(tmp_path):
    txt = tmp_path / "empty.txt"
    txt.write_text("\n\n", encoding="utf-8")
    assert _read_captions_from_txt_per_line(str(txt)) == [""]


def test_dump_dataset_txt_single_caption(tmp_path, img_dir):
    _copy_fixture_image(img_dir, "a")
    (img_dir / "a.txt").write_text("solo caption\n", encoding="utf-8")
    records = dump_dataset(_write_dataset_toml(tmp_path, img_dir))
    assert len(records) == 1
    assert records[0]["captions"] == ["solo caption"]


def test_dump_dataset_txt_multiple_captions_per_image(tmp_path, img_dir):
    _copy_fixture_image(img_dir, "multi")
    (img_dir / "multi.txt").write_text(
        "tag1, character\nvariant B description\n", encoding="utf-8"
    )
    records = dump_dataset(_write_dataset_toml(tmp_path, img_dir))
    assert records[0]["captions"] == ["tag1, character", "variant B description"]


def test_dump_dataset_captions_json_list(tmp_path, img_dir):
    img = _copy_fixture_image(img_dir, "from_json")
    (img_dir / CAPTIONS_JSON_FILE).write_text(
        json.dumps({img.name: ["json cap A", "json cap B"]}),
        encoding="utf-8",
    )
    records = dump_dataset(_write_dataset_toml(tmp_path, img_dir))
    assert records[0]["captions"] == ["json cap A", "json cap B"]


def test_dump_dataset_captions_json_string_coerced_to_list(tmp_path, img_dir):
    img = _copy_fixture_image(img_dir, "str_cap")
    (img_dir / CAPTIONS_JSON_FILE).write_text(
        json.dumps({img.name: "single string caption"}),
        encoding="utf-8",
    )
    records = dump_dataset(_write_dataset_toml(tmp_path, img_dir))
    assert records[0]["captions"] == ["single string caption"]


def test_dump_dataset_captions_json_missing_key_returns_empty_string(tmp_path, img_dir):
    _copy_fixture_image(img_dir, "orphan")
    (img_dir / CAPTIONS_JSON_FILE).write_text(json.dumps({}), encoding="utf-8")
    records = dump_dataset(_write_dataset_toml(tmp_path, img_dir))
    assert records[0]["captions"] == [""]


def test_dump_dataset_json_takes_precedence_over_txt(tmp_path, img_dir):
    """When captions.json exists, dump_dataset ignores sidecar .txt (same as cache enumeration)."""
    img = _copy_fixture_image(img_dir, "both")
    (img_dir / "both.txt").write_text("from txt file\n", encoding="utf-8")
    (img_dir / CAPTIONS_JSON_FILE).write_text(
        json.dumps({img.name: ["from json"]}),
        encoding="utf-8",
    )
    records = dump_dataset(_write_dataset_toml(tmp_path, img_dir))
    assert records[0]["captions"] == ["from json"]


def test_metadata_map_fn_reads_txt_multiple_lines(tmp_path, img_dir):
    img = _copy_fixture_image(img_dir, "pic")
    txt = img_dir / "pic.txt"
    txt.write_text("alpha\nbeta\n", encoding="utf-8")
    dd = DirectoryDataset(
        {"path": str(img_dir), "num_repeats": 1, "shuffle_metadata": False},
        MINIMAL_DATASET_CONFIG,
        "sdxl",
        skip_dataset_validation=True,
    )
    fn, _ = dd._metadata_map_fn()
    batch = {
        "image_spec": [[None, str(img)]],
        "caption_file": [str(txt)],
        "mask_file": [None],
    }
    out = fn(batch)
    assert out["caption"] == [["alpha", "beta"]]


def test_metadata_map_fn_uses_embedded_json_caption_field(tmp_path, img_dir):
    img = _copy_fixture_image(img_dir, "embedded")
    dd = DirectoryDataset(
        {"path": str(img_dir), "num_repeats": 1, "shuffle_metadata": False},
        MINIMAL_DATASET_CONFIG,
        "sdxl",
        skip_dataset_validation=True,
    )
    fn, _ = dd._metadata_map_fn()
    batch = {
        "image_spec": [[None, str(img)]],
        "caption_file": [""],
        "mask_file": [None],
        "caption": [["from preloaded json", "second variant"]],
    }
    out = fn(batch)
    assert out["caption"] == [["from preloaded json", "second variant"]]


def test_retired_shuffle_keys_are_ignored_with_warning(tmp_path, img_dir, capsys):
    """shuffle_tags/cache_shuffle_num were retired: captions pass through untouched."""
    img = _copy_fixture_image(img_dir, "tags")
    (img_dir / "tags.txt").write_text("alpha, beta, gamma\n", encoding="utf-8")
    dd = DirectoryDataset(
        {
            "path": str(img_dir),
            "num_repeats": 1,
            "shuffle_metadata": False,
            "shuffle_tags": True,
            "cache_shuffle_num": 3,
        },
        MINIMAL_DATASET_CONFIG,
        "sdxl",
        skip_dataset_validation=True,
    )
    out_text = capsys.readouterr().out
    assert "retired" in out_text and "generate_caption_variants" in out_text
    fn, _ = dd._metadata_map_fn()
    batch = {
        "image_spec": [[None, str(img)]],
        "caption_file": [str(img_dir / "tags.txt")],
        "mask_file": [None],
    }
    out = fn(batch)
    assert out["caption"][0] == ["alpha, beta, gamma"]


def test_metadata_map_fn_tracks_tar_handles_for_cleanup(tmp_path, img_dir):
    """The tar handles opened while building metadata are tracked so the caller can close them."""
    import tarfile

    img = _copy_fixture_image(img_dir, "intar")
    tar_path = tmp_path / "shard.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(img, arcname="intar.jpg")

    dd = DirectoryDataset(
        {"path": str(img_dir), "num_repeats": 1},
        MINIMAL_DATASET_CONFIG,
        "sdxl",
        skip_dataset_validation=True,
    )
    fn, tarfile_map = dd._metadata_map_fn()
    assert tarfile_map == {}

    batch = {
        "image_spec": [[str(tar_path), "intar.jpg"]],
        "caption_file": [""],
        "mask_file": [None],
        "caption": [["a caption"]],
    }
    fn(batch)
    # The handle is opened and registered, still open (the map is mid-flight).
    assert str(tar_path) in tarfile_map
    assert not tarfile_map[str(tar_path)].closed

    # Simulate the caller's finally-block cleanup.
    for tar_f in tarfile_map.values():
        tar_f.close()
    assert tarfile_map[str(tar_path)].closed


def test_metadata_map_fn_directory_caption_fallback(tmp_path, img_dir):
    img = _copy_fixture_image(img_dir, "fallback")
    dd = DirectoryDataset(
        {
            "path": str(img_dir),
            "num_repeats": 1,
            "shuffle_metadata": False,
            "directory_caption": "shared folder caption",
        },
        MINIMAL_DATASET_CONFIG,
        "sdxl",
        skip_dataset_validation=True,
    )
    fn, _ = dd._metadata_map_fn()
    batch = {
        "image_spec": [[None, str(img)]],
        "caption_file": [""],
        "mask_file": [None],
    }
    out = fn(batch)
    # No per-image caption: directory_caption is the full caption, applied once (not prefixed onto itself).
    assert out["caption"] == [["shared folder caption"]]


def test_metadata_map_fn_directory_caption_prefixes_existing(tmp_path, img_dir):
    img = _copy_fixture_image(img_dir, "prefixed")
    txt = img.with_suffix(".txt")
    txt.write_text("a cat", encoding="utf-8")
    dd = DirectoryDataset(
        {
            "path": str(img_dir),
            "num_repeats": 1,
            "shuffle_metadata": False,
            "directory_caption": "style: ",
        },
        MINIMAL_DATASET_CONFIG,
        "sdxl",
        skip_dataset_validation=True,
    )
    fn, _ = dd._metadata_map_fn()
    out = fn({"image_spec": [[None, str(img)]], "caption_file": [str(txt)], "mask_file": [None]})
    # Per-image caption present: directory_caption is prepended once.
    assert out["caption"] == [["style: a cat"]]


def test_size_bucket_iteration_order_one_row_per_caption(tmp_path):
    """Equal caption counts expand iteration_order to one training row per (image, caption_index)."""
    metadata = datasets.Dataset.from_dict(
        {
            "image_spec": [[None, "a.jpg"], [None, "b.jpg"]],
            "caption": [["cap a1", "cap a2"], ["cap b1", "cap b2"]],
        }
    )
    dir_cfg = {"path": str(tmp_path), "num_repeats": 1}
    sb = SizeBucketDataset(
        metadata,
        dir_cfg,
        (512, 512, 1),
        tmp_path / "cache",
        None,
    )

    def fake_latent_map(example, rank):
        n = len(example["image_spec"])
        return {"latents": torch.zeros(n, 4)}

    sb.cache_latents(
        fake_latent_map,
        regenerate_cache=True,
        trust_cache=False,
    )
    assert len(sb.iteration_order) == 4
    numbers = sorted(row["caption_number"] for row in sb.iteration_order)
    assert numbers == [0, 0, 1, 1]
    captions = sorted(row["caption"] for row in sb.iteration_order)
    assert captions == ["cap a1", "cap a2", "cap b1", "cap b2"]


def test_size_bucket_directory_subsample_ratio(tmp_path):
    metadata = datasets.Dataset.from_dict(
        {
            "image_spec": [[None, f"img{i}.jpg"] for i in range(8)],
            "caption": [[f"cap {i}"] for i in range(8)],
        }
    )
    dir_cfg = {
        "path": str(tmp_path),
        "num_repeats": 1,
        "subsample_ratio": 0.25,
    }
    sb = SizeBucketDataset(
        metadata,
        dir_cfg,
        (512, 512, 1),
        tmp_path / "cache",
        None,
    )

    def fake_latent_map(example, rank):
        n = len(example["image_spec"])
        return {"latents": torch.zeros(n, 4)}

    sb.cache_latents(
        fake_latent_map,
        regenerate_cache=True,
        trust_cache=False,
    )
    # subsample_ratio no longer trims the cache: the full pool stays available so the
    # per-epoch window can rotate over all of it.
    assert len(sb.iteration_order) == 8

    # The cap is now applied at the folder level (shared across buckets). Inject a folder
    # subsampler over this bucket's 8 base images with the floor(8 * 0.25) = 2 cap.
    sub = FolderSubsampler(sb._base_keys, cap=2, static=False, seed=0)

    class _Dir:
        def folder_subsampler(self):
            return sub

    sb.directory_dataset = _Dir()
    assert len(sb) == 2
    # Rotating by default: consecutive epochs serve a different slice of the pool.
    sb.set_epoch(1)
    epoch1 = {sb._pool_index(i) for i in range(len(sb))}
    sb.set_epoch(2)
    epoch2 = {sb._pool_index(i) for i in range(len(sb))}
    assert epoch1 != epoch2


def test_max_images_caps_the_folder_total_not_each_bucket(tmp_path):
    """Regression: max_images caps a FOLDER's per-epoch base images, shared across its size
    buckets — not max_images per bucket (which made a multi-bucket folder contribute N x buckets)."""

    def make_bucket(names, bucket):
        sb = SizeBucketDataset(
            datasets.Dataset.from_dict(
                {"image_spec": [[None, n] for n in names], "caption": [["c"] for _ in names]}
            ),
            {"path": str(tmp_path), "num_repeats": 1},
            bucket,
            tmp_path / "cache",
            None,
        )
        sb.iteration_order = datasets.Dataset.from_dict(
            {
                "image_spec": [[None, n] for n in names],
                "latents_idx": list(range(len(names))),
                "caption": [""] * len(names),
                "caption_number": [0] * len(names),
            }
        )
        return sb

    b1 = make_bucket([f"a{i}.jpg" for i in range(5)], (512, 512, 1))
    b2 = make_bucket([f"b{i}.jpg" for i in range(5)], (768, 768, 1))  # 10 base images, 2 buckets
    sub = FolderSubsampler(list(b1._base_keys) + list(b2._base_keys), cap=4, static=False, seed=0)

    class _Dir:
        def folder_subsampler(self):
            return sub

    b1.directory_dataset = b2.directory_dataset = _Dir()
    b1.set_epoch(1)
    b2.set_epoch(1)
    # The folder serves exactly `cap` base images TOTAL across both buckets (not cap per bucket).
    assert len(b1) + len(b2) == 4


def test_size_bucket_online_captions_selects_caption_number(tmp_path):
    metadata = datasets.Dataset.from_dict(
        {
            "image_spec": [[None, "img.png"]],
            "caption": [[""]],
        }
    )

    class FakeDir:
        captions_dict = {"img.png": ["first", "second", "third"]}

        def folder_subsampler(self):
            return FolderSubsampler([], cap=None, static=False, seed=0)  # uncapped

    sb = SizeBucketDataset(
        metadata,
        {"path": str(tmp_path), "num_repeats": 1},
        (512, 512, 1),
        tmp_path / "cache",
        FakeDir(),
    )
    sb.latent_dataset = [{"latents": 0}]
    sb.iteration_order = datasets.Dataset.from_dict(
        {
            "image_spec": [[None, "img.png"]],
            "latents_idx": [0],
            "caption": [""],
            "caption_number": [2],
        }
    )
    sb.text_embedding_datasets = []
    sb.uncond_text_embeddings = []
    assert sb[0]["caption"] == "third"


def test_uniform_caption_variants_equal_counts():
    from rengu_flow.data.dataset import uniform_caption_variants

    assert uniform_caption_variants([["a", "b", "c"], ["d", "e", "f"]]) == 3


def test_uniform_caption_variants_mixed_counts_fall_back_to_one():
    from rengu_flow.data.dataset import uniform_caption_variants

    assert uniform_caption_variants([["a", "b"], ["c"]]) == 1


def test_uniform_caption_variants_single_caption():
    from rengu_flow.data.dataset import uniform_caption_variants

    assert uniform_caption_variants([["a"], ["b"]]) == 1


def test_dataset_caption_variants_aggregates_buckets():
    """Dataset.caption_variants: one uniform value across buckets, else 1."""
    from types import SimpleNamespace

    from rengu_flow.data.dataset import Dataset

    def fake(variants):
        return SimpleNamespace(caption_variants=variants)

    ds = object.__new__(Dataset)
    ds.directory_datasets = [
        SimpleNamespace(
            size_bucket_datasets=[fake(15), fake(15)],
            ar_bucket_datasets=[SimpleNamespace(size_buckets=[fake(15)])],
        )
    ]
    assert ds.caption_variants == 15

    ds.directory_datasets[0].size_bucket_datasets.append(fake(3))
    assert ds.caption_variants == 1
