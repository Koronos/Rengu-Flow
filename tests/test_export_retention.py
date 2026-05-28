"""Tests for model export retention pruning."""

from pathlib import Path

from renga_flow.utils.saver import _prune_old_exports


def test_prune_intersection_min_step_and_max_keep(tmp_path: Path):
    for name in ("step500", "step1000", "step1500", "step2000"):
        (tmp_path / name).mkdir()
    config = {"keep_exports_from_step": 1000, "max_model_exports_to_keep": 2}
    _prune_old_exports(tmp_path, config, steps_per_epoch=100)
    remaining = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert remaining == {"step1500", "step2000"}


def test_signal_step_not_pruned(tmp_path: Path):
    (tmp_path / "step100").mkdir()
    (tmp_path / "step200").mkdir()
    (tmp_path / "signal_step300").mkdir()
    config = {"max_model_exports_to_keep": 1}
    _prune_old_exports(tmp_path, config, steps_per_epoch=10)
    remaining = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert "signal_step300" in remaining
    assert len([n for n in remaining if n.startswith("step")]) == 1
