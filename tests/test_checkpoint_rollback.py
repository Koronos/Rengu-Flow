"""Tests for checkpoint rollback on failed save."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rengu_flow.utils.saver import Saver


def test_save_checkpoint_returns_false_on_enospc(tmp_path: Path):
    args = MagicMock()
    args.config = str(tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("# test")
    model_engine = MagicMock()
    model_engine.save_checkpoint.side_effect = OSError(28, "No space left on device")
    train_dataloader = MagicMock()
    train_dataloader.state_dict.return_value = {"epoch": 1}
    saver = Saver(
        args,
        {},
        True,
        tmp_path,
        MagicMock(),
        train_dataloader,
        model_engine,
        MagicMock(),
    )
    (tmp_path / "global_step1").mkdir()
    with patch("rengu_flow.utils.saver.dist") as mock_dist:
        mock_dist.barrier = MagicMock()
        with patch("rengu_flow.utils.saver.is_main_process", return_value=True):
            ok = saver.save_checkpoint(5, 50)
    assert ok is False
