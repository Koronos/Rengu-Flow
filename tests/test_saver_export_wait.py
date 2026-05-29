"""Tests for export wait / continue recovery loop."""

from unittest.mock import MagicMock, patch


from rengu_flow.utils.signal_files import ExportRecoveryAction
from rengu_flow.utils.saver import Saver


def test_save_model_retries_after_continue(tmp_path):
    args = MagicMock()
    args.config = str(tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("# test")
    model_engine = MagicMock()
    model_engine.grid.get_data_parallel_rank.return_value = 0
    model_engine.grid.get_pipe_parallel_rank.return_value = 0
    train_dataloader = MagicMock()
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
    calls = {"n": 0}

    def flaky_save(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")

    with patch("rengu_flow.utils.saver.dist") as mock_dist:
        mock_dist.barrier = MagicMock()
        with patch("rengu_flow.utils.saver.is_main_process", return_value=True):
            with patch.object(saver, "_save_model_once", side_effect=flaky_save):
                with patch(
                    "rengu_flow.utils.saver.wait_for_export_recovery",
                    return_value=ExportRecoveryAction.CONTINUE,
                ):
                    assert saver.save_model("step9") is True
    assert calls["n"] == 2
