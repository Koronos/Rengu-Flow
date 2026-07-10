"""Parquet-backed datasets: enumeration, captions from columns, dims fast path,
and image reads through all three consumer sites (metadata map, media preprocess)."""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from rengu_flow.data.dataset import DirectoryDataset
from rengu_flow.data.parquet_source import MEMBER_PREFIX, ParquetSource, spec_row
from rengu_flow.data.preprocess_media import PreprocessMediaFile

MINIMAL_DATASET_CONFIG = {
    "resolutions": [64],
    "frame_buckets": [1],
    "min_ar": 0.5,
    "max_ar": 2.0,
    "num_ar_buckets": 4,
    "enable_ar_bucket": True,
}

SIZES = [(64, 64), (96, 48), (48, 96), (80, 64)]
COLORS = ["red", "green", "blue", "yellow"]


def _jpeg_bytes(size, color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _write_recap_style(path):
    """LLaVA/ReCap-CC3M schema: id, image struct{bytes,path}, conversations chat."""
    rows = {
        "id": [f"img{i:03d}" for i in range(4)],
        "image": [{"bytes": _jpeg_bytes(s, c), "path": None} for s, c in zip(SIZES, COLORS)],
        "conversations": [
            [{"from": "human", "value": "describe"}, {"from": "gpt", "value": f"a {c} rectangle"}]
            for c in COLORS
        ],
    }
    pq.write_table(pa.table(rows), path)


def _write_danbooru_style(path):
    """Target schema for future scrapes: image binary, caption str, width/height."""
    rows = {
        "image": [_jpeg_bytes(s, c) for s, c in zip(SIZES, COLORS)],
        "caption": [f"a {c} rectangle" for c in COLORS],
        "width": [s[0] for s in SIZES],
        "height": [s[1] for s in SIZES],
    }
    pq.write_table(pa.table(rows), path)


def test_source_enumerates_chat_captions(tmp_path):
    f = tmp_path / "a.parquet"
    _write_recap_style(f)
    src = ParquetSource({})
    cols = src.enumerate_columns(str(f))
    assert cols["caption"] == [[f"a {c} rectangle"] for c in COLORS]
    assert "width" not in cols  # recap schema has no dims


def test_source_enumerates_plain_captions_and_dims(tmp_path):
    f = tmp_path / "b.parquet"
    _write_danbooru_style(f)
    cols = ParquetSource({}).enumerate_columns(str(f))
    assert cols["caption"] == [[f"a {c} rectangle"] for c in COLORS]
    assert cols["width"] == [s[0] for s in SIZES]
    assert cols["height"] == [s[1] for s in SIZES]


def test_source_reads_images_by_row(tmp_path):
    f = tmp_path / "a.parquet"
    _write_recap_style(f)
    src = ParquetSource({})
    for i, (size, color) in enumerate(zip(SIZES, COLORS)):
        im = Image.open(src.read_image(str(f), i))
        assert im.size == size


def _metadata_for(tmp_path, writer):
    d = tmp_path / "ds"
    d.mkdir()
    writer(d / "data.parquet")
    dd = DirectoryDataset(
        {"path": str(d), "num_repeats": 1, "shuffle_metadata": False},
        MINIMAL_DATASET_CONFIG,
        "sdxl",
        skip_dataset_validation=True,
    )
    dd.cache_metadata(regenerate_cache=True, cache_num_proc=1)
    return dd


def test_directory_dataset_metadata_from_recap_parquet(tmp_path):
    dd = _metadata_for(tmp_path, _write_recap_style)
    buckets = dd.ar_bucket_datasets
    assert len(buckets) == 4  # 4 distinct ARs -> one image per bucket
    caps, specs = [], []
    for b in buckets:
        for row in b.metadata_dataset:
            caps.append(row["caption"][0])
            specs.append(row["image_spec"])
    assert sorted(caps) == sorted(f"a {c} rectangle" for c in COLORS)
    assert all(s[0].endswith(".parquet") and s[1].startswith(MEMBER_PREFIX) for s in specs)


def test_dims_fast_path_never_reads_image_bytes(tmp_path, monkeypatch):
    """With width/height columns, the metadata stage must not touch image bytes."""
    def boom(self, path, row):
        raise AssertionError("read_image called during metadata despite dims columns")

    monkeypatch.setattr(ParquetSource, "read_image", boom)
    dd = _metadata_for(tmp_path, _write_danbooru_style)
    n = sum(len(b.metadata_dataset) for b in dd.ar_bucket_datasets)
    assert n == 4  # bucketing happened from columns alone


def test_preprocess_media_reads_parquet_spec(tmp_path):
    f = tmp_path / "a.parquet"
    _write_recap_style(f)
    fn = PreprocessMediaFile({}, support_video=False)
    out = fn((str(f), f"{MEMBER_PREFIX}1"), None, size_bucket=(96, 48, 1))
    assert len(out) == 1
    img, _mask, valid = out[0]
    assert valid
    assert tuple(img.shape)[-2:] == (48, 96)


def test_spec_row_roundtrip():
    assert spec_row(("x.parquet", f"{MEMBER_PREFIX}42")) == 42
    with pytest.raises(ValueError):
        spec_row(("x.parquet", "not-a-row"))
