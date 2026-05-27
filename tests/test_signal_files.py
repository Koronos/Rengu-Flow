"""Tests for file-based training signals."""

from pathlib import Path

from renga_flow.utils.signal_files import (
    SIGNAL_EXPORT_MODEL,
    SIGNAL_EXPORT_MODEL_QUIT,
    SIGNAL_PREVIEW,
    SIGNAL_SAVE,
    SIGNAL_SAVE_QUIT,
    SignalResult,
    process_signals,
)


def test_process_signals_save(tmp_path):
    (tmp_path / SIGNAL_SAVE).touch()
    result = process_signals(tmp_path)
    assert result.should_checkpoint is True
    assert result.should_quit is False
    assert result.should_export_model is False
    assert not (tmp_path / SIGNAL_SAVE).exists()


def test_process_signals_save_quit(tmp_path):
    (tmp_path / SIGNAL_SAVE_QUIT).touch()
    result = process_signals(tmp_path)
    assert result.should_checkpoint is True
    assert result.should_quit is True
    assert not (tmp_path / SIGNAL_SAVE_QUIT).exists()


def test_process_signals_export_model(tmp_path):
    (tmp_path / SIGNAL_EXPORT_MODEL).touch()
    result = process_signals(tmp_path)
    assert result.should_checkpoint is False
    assert result.should_export_model is True
    assert result.should_export_quit is False
    assert not (tmp_path / SIGNAL_EXPORT_MODEL).exists()


def test_process_signals_export_model_quit(tmp_path):
    (tmp_path / SIGNAL_EXPORT_MODEL_QUIT).touch()
    result = process_signals(tmp_path)
    assert result.should_export_model is True
    assert result.should_export_quit is True
    assert not (tmp_path / SIGNAL_EXPORT_MODEL_QUIT).exists()


def test_process_signals_empty(tmp_path):
    result = process_signals(tmp_path)
    assert result == SignalResult(False, False, False, False, False)


def test_process_signals_preview(tmp_path):
    (tmp_path / SIGNAL_PREVIEW).touch()
    result = process_signals(tmp_path)
    assert result.should_preview is True
    assert not (tmp_path / SIGNAL_PREVIEW).exists()


def test_process_signals_save_quit_takes_priority_over_save(tmp_path):
    (tmp_path / SIGNAL_SAVE).touch()
    (tmp_path / SIGNAL_SAVE_QUIT).touch()
    result = process_signals(tmp_path)
    assert result.should_checkpoint is True
    assert result.should_quit is True
    assert not (tmp_path / SIGNAL_SAVE).exists()
    assert not (tmp_path / SIGNAL_SAVE_QUIT).exists()
