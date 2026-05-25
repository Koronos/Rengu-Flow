"""Tests for DeepSpeed checkpoint retention pruning."""

from pathlib import Path

from renga_flow.utils.saver import _prune_old_checkpoints


def test_prune_old_checkpoints_keeps_newest(tmp_path: Path):
    for step in (100, 200, 300, 400):
        (tmp_path / f"global_step{step}").mkdir()
        (tmp_path / f"global_step{step}" / "dummy.txt").write_text("x")

    _prune_old_checkpoints(tmp_path, 2)

    remaining = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert remaining == {"global_step300", "global_step400"}


def test_prune_old_checkpoints_noop_when_unlimited(tmp_path: Path):
    (tmp_path / "global_step1").mkdir()
    (tmp_path / "global_step2").mkdir()
    _prune_old_checkpoints(tmp_path, None)
    assert len(list(tmp_path.iterdir())) == 2
