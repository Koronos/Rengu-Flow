"""Tests for file-based training signals."""


from rengu_flow.utils.signal_files import (
    SIGNAL_CONTINUE,
    SIGNAL_EXPORT_MODEL,
    SIGNAL_EXPORT_MODEL_QUIT,
    SIGNAL_PREVIEW,
    SIGNAL_RELOAD_CONFIG,
    SIGNAL_SAVE,
    SIGNAL_SAVE_QUIT,
    ExportRecoveryAction,
    SignalResult,
    process_signals,
    wait_for_export_recovery,
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
    assert result == SignalResult(False, False, False, False, False, False)


def test_process_signals_preview(tmp_path):
    (tmp_path / SIGNAL_PREVIEW).touch()
    result = process_signals(tmp_path)
    assert result.should_preview is True
    assert not (tmp_path / SIGNAL_PREVIEW).exists()


def test_process_signals_reload_config(tmp_path):
    (tmp_path / SIGNAL_RELOAD_CONFIG).touch()
    result = process_signals(tmp_path)
    assert result.should_reload_config is True
    assert not (tmp_path / SIGNAL_RELOAD_CONFIG).exists()


def test_clear_stale_signals(tmp_path):
    from rengu_flow.utils.signal_files import clear_stale_signals

    (tmp_path / SIGNAL_SAVE_QUIT).touch()  # e.g. left by a force-stop that killed the process
    (tmp_path / SIGNAL_PREVIEW).touch()
    (tmp_path / "latest").write_text("global_step5", encoding="utf-8")  # not a signal

    removed = clear_stale_signals(tmp_path)
    assert set(removed) == {SIGNAL_SAVE_QUIT, SIGNAL_PREVIEW}
    assert not (tmp_path / SIGNAL_SAVE_QUIT).exists()
    assert not (tmp_path / SIGNAL_PREVIEW).exists()
    assert (tmp_path / "latest").exists()  # non-signal files are left alone
    assert clear_stale_signals(tmp_path) == []  # nothing left to clear


def test_wait_for_export_recovery_continue(tmp_path):
    (tmp_path / SIGNAL_CONTINUE).touch()
    action = wait_for_export_recovery(tmp_path)
    assert action == ExportRecoveryAction.CONTINUE
    assert not (tmp_path / SIGNAL_CONTINUE).exists()


def test_process_signals_save_quit_takes_priority_over_save(tmp_path):
    (tmp_path / SIGNAL_SAVE).touch()
    (tmp_path / SIGNAL_SAVE_QUIT).touch()
    result = process_signals(tmp_path)
    assert result.should_checkpoint is True
    assert result.should_quit is True
    assert not (tmp_path / SIGNAL_SAVE).exists()
    assert not (tmp_path / SIGNAL_SAVE_QUIT).exists()
