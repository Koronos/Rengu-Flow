"""Tests for --dump_dataset / rengu_flow.data.dump_dataset."""

import json
from pathlib import Path

import pytest
import toml

from rengu_flow.data.dump_dataset import dump_dataset


def test_dump_dataset_tmp_dir(tmp_path, capsys):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (img_dir / "a.txt").write_text("hello world\n", encoding="utf-8")
    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.write_text(
        toml.dumps(
            {
                "resolutions": [512],
                "frame_buckets": [1],
                "directory": [{"path": str(img_dir), "num_repeats": 1}],
            }
        ),
        encoding="utf-8",
    )
    records = dump_dataset(dataset_toml)
    assert len(records) == 1
    assert records[0]["captions"] == ["hello world"]
    line = capsys.readouterr().out.strip()
    parsed = json.loads(line)
    assert parsed["image"].endswith("a.jpg")
