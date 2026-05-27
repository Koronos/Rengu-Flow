"""Tests for UI TensorBoard launcher (uv --no-project)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from renga_flow_ui import tensorboard_server
from renga_flow_ui.settings import repo_root


@pytest.fixture(autouse=True)
def _reset_tensorboard_process() -> None:
    proc = tensorboard_server._proc
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=1)
        except Exception:
            pass
    tensorboard_server._proc = None
    tensorboard_server._meta = {}
    yield
    tensorboard_server._proc = None
    tensorboard_server._meta = {}


def test_resolve_output_dir_relative() -> None:
    p = tensorboard_server.resolve_output_dir("output")
    assert p == (repo_root() / "output").resolve()


def test_start_requires_output_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("renga_flow_ui.settings.repo_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        tensorboard_server.start_tensorboard("missing-output")


def test_start_uv_missing(tmp_path, monkeypatch) -> None:
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr("renga_flow_ui.settings.repo_root", lambda: tmp_path)
    monkeypatch.setattr(tensorboard_server.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="uv"):
        tensorboard_server.start_tensorboard("output")


def test_start_and_stop_mocked(tmp_path, monkeypatch) -> None:
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr("renga_flow_ui.settings.repo_root", lambda: tmp_path)
    monkeypatch.setattr(tensorboard_server, "logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(tensorboard_server.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(tensorboard_server, "pick_free_port", lambda *a, **k: 6007)
    monkeypatch.setattr(tensorboard_server.time, "sleep", lambda _: None)

    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 4242

    def _finish(*_a, **_k):
        proc.poll.return_value = 0

    proc.terminate.side_effect = _finish
    proc.wait.side_effect = _finish
    proc.kill.side_effect = _finish

    @contextmanager
    def _fake_conn(*_a, **_k):
        yield object()

    monkeypatch.setattr(tensorboard_server.socket, "create_connection", _fake_conn)

    with patch.object(tensorboard_server.subprocess, "Popen", return_value=proc) as popen:
        result = tensorboard_server.start_tensorboard("output")

    assert result["running"] is True
    assert result["port"] == 6007
    assert "6007" in result["url"]
    cmd = popen.call_args[0][0]
    assert "tensorboard" in cmd
    assert any(arg.startswith("--logdir=") and str(out.resolve()) in arg for arg in cmd)

    stopped = tensorboard_server.stop_tensorboard()
    assert stopped["stopped"] is True
    proc.terminate.assert_called_once()
