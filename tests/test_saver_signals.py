"""Tests for Saver.process_step integration with file-based training signals."""

from unittest.mock import MagicMock, patch

import pytest

from renga_flow.utils.signal_files import (
    SIGNAL_EXPORT_MODEL,
    SIGNAL_EXPORT_MODEL_QUIT,
    SIGNAL_PREVIEW,
    SIGNAL_SAVE,
    SIGNAL_SAVE_QUIT,
)
from renga_flow.utils.saver import Saver


@pytest.fixture
def mock_save_checkpoint():
    with patch.object(Saver, "save_checkpoint", return_value=True) as mock_ckpt:
        yield mock_ckpt


@pytest.fixture
def saver_bundle(tmp_path):
    args = MagicMock()
    args.config = str(tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("# test")
    config = {}
    model_engine = MagicMock()
    train_dataloader = MagicMock()
    train_dataloader.state_dict.return_value = {"epoch": 1}
    train_dataloader.epoch = 1
    saver = Saver(
        args,
        config,
        is_adapter=True,
        save_root=tmp_path,
        model=MagicMock(),
        train_dataloader=train_dataloader,
        model_engine=model_engine,
        pipeline_model=MagicMock(),
    )
    return saver, model_engine


@patch("renga_flow.utils.saver._need_to_checkpoint", return_value=False)
def test_process_step_no_signals(_mock_ckpt, saver_bundle):
    saver, model_engine = saver_bundle
    with patch.object(saver, "save_model", return_value=True) as save_model:
        checkpointed, saved, signals = saver.process_step(10, 100)
    assert checkpointed is False
    assert saved is False
    assert not signals.should_checkpoint
    assert not signals.should_quit
    assert not signals.should_export_model
    assert not signals.should_export_quit
    assert not signals.should_preview
    model_engine.save_checkpoint.assert_not_called()
    save_model.assert_not_called()


@patch("renga_flow.utils.saver._need_to_checkpoint", return_value=False)
def test_process_step_save(_mock_ckpt, saver_bundle, mock_save_checkpoint, tmp_path):
    saver, model_engine = saver_bundle
    (tmp_path / SIGNAL_SAVE).touch()
    checkpointed, saved, signals = saver.process_step(7, 70)
    assert checkpointed is True
    assert saved is False
    assert signals.should_checkpoint is True
    assert signals.should_quit is False
    mock_save_checkpoint.assert_called_once_with(7, 70)
    model_engine.save_checkpoint.assert_not_called()


@patch("renga_flow.utils.saver._need_to_checkpoint", return_value=False)
def test_process_step_save_quit(_mock_ckpt, saver_bundle, mock_save_checkpoint, tmp_path):
    saver, model_engine = saver_bundle
    (tmp_path / SIGNAL_SAVE_QUIT).touch()
    with pytest.raises(SystemExit):
        saver.process_step(3, 30)
    mock_save_checkpoint.assert_called_once_with(3, 30)


@patch("renga_flow.utils.saver._need_to_checkpoint", return_value=False)
def test_process_step_export_model(_mock_ckpt, saver_bundle, tmp_path):
    saver, model_engine = saver_bundle
    (tmp_path / SIGNAL_EXPORT_MODEL).touch()
    with patch.object(saver, "save_model", return_value=True) as save_model:
        checkpointed, saved, signals = saver.process_step(12, 120)
    assert checkpointed is False
    assert saved is True
    assert signals.should_export_model is True
    save_model.assert_called_once_with("signal_step12")
    model_engine.save_checkpoint.assert_not_called()


@patch("renga_flow.utils.saver._need_to_checkpoint", return_value=False)
def test_process_step_export_model_quit(_mock_ckpt, saver_bundle, tmp_path):
    saver, model_engine = saver_bundle
    (tmp_path / SIGNAL_EXPORT_MODEL_QUIT).touch()
    with patch.object(saver, "save_model", return_value=True) as save_model:
        with pytest.raises(SystemExit):
            saver.process_step(4, 40)
    save_model.assert_called_once_with("signal_step4")
    model_engine.save_checkpoint.assert_not_called()


@patch("renga_flow.utils.saver._need_to_checkpoint", return_value=False)
def test_process_step_save_and_preview(_mock_ckpt, saver_bundle, mock_save_checkpoint, tmp_path):
    saver, model_engine = saver_bundle
    (tmp_path / SIGNAL_SAVE).touch()
    (tmp_path / SIGNAL_PREVIEW).touch()
    checkpointed, saved, signals = saver.process_step(9, 90)
    assert checkpointed is True
    assert signals.should_preview is True
    mock_save_checkpoint.assert_called_once_with(9, 90)
