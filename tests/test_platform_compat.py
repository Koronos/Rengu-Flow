"""Tests for rengu_flow.platform_compat. No GPU/DeepSpeed; mock sys.platform via the flag."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from rengu_flow import platform_compat as pc

# These are pure platform-layer unit tests; the UI sqlite autouse fixture is irrelevant.
pytestmark = pytest.mark.no_ui_db


@pytest.fixture
def as_windows(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pc, "PLATFORM", pc.WindowsPlatform())


@pytest.fixture
def as_posix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pc, "PLATFORM", pc.Platform())


# --- venv layout ---------------------------------------------------------


def test_venv_bin_dir_windows(as_windows):
    assert pc.venv_bin_dir(Path("/proj/.venv")) == Path("/proj/.venv/Scripts")


def test_venv_bin_dir_posix(as_posix):
    assert pc.venv_bin_dir(Path("/proj/.venv")) == Path("/proj/.venv/bin")


def test_venv_exe_windows_appends_exe(as_windows):
    assert pc.venv_exe(Path("/proj/.venv"), "python") == Path("/proj/.venv/Scripts/python.exe")
    assert pc.venv_exe(Path("/proj/.venv"), "rengu") == Path("/proj/.venv/Scripts/rengu.exe")


def test_venv_exe_posix_no_suffix(as_posix):
    assert pc.venv_exe(Path("/proj/.venv"), "python") == Path("/proj/.venv/bin/python")
    assert pc.venv_exe(Path("/proj/.venv"), "tensorboard") == Path("/proj/.venv/bin/tensorboard")


# --- popen kwargs --------------------------------------------------------


def test_popen_kwargs_windows(as_windows):
    kw = pc.popen_kwargs_new_group()
    assert "creationflags" in kw
    assert kw["creationflags"] == pc._CREATE_NEW_PROCESS_GROUP
    assert "start_new_session" not in kw


def test_popen_kwargs_posix(as_posix):
    kw = pc.popen_kwargs_new_group()
    assert kw == {"start_new_session": True}


# --- find_free_port ------------------------------------------------------


def test_find_free_port_returns_bindable_port():
    port = pc.find_free_port(38500, 50)
    assert 38500 <= port < 38550
    # The returned port must actually be bindable.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def test_find_free_port_skips_occupied(monkeypatch):
    # Occupy the first candidate port; find_free_port must skip it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        taken = occupied.getsockname()[1]
        got = pc.find_free_port(taken, 20)
        assert got != taken
        assert taken < got < taken + 20


# --- pid_alive -----------------------------------------------------------


def test_pid_alive_none():
    assert pc.pid_alive(None) is False


def test_pid_alive_self():
    # Run on the native platform (Windows→psutil, POSIX→os.kill); both must see our own PID.
    import os

    assert pc.pid_alive(os.getpid()) is True


def test_pid_alive_dead_pid():
    # PID 2**31-1 is essentially never a live process, on either platform.
    assert pc.pid_alive(2**31 - 1) is False


# --- http_health_ok ------------------------------------------------------


def test_http_health_ok_bad_url_is_false():
    # Nothing is listening here → URLError → False (fast, no network).
    assert pc.http_health_ok("http://127.0.0.1:1/health", timeout=0.2) is False


def test_http_health_ok_parses_status(monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    assert pc.http_health_ok("http://example/health") is True


# --- open_browser --------------------------------------------------------


def test_open_browser_prints_fallback_when_open_fails(monkeypatch, capsys):
    monkeypatch.setattr("webbrowser.open", lambda url: False)
    pc.open_browser("http://localhost:9999/")
    assert "Open in browser" in capsys.readouterr().out


def test_open_browser_no_fallback_on_success(monkeypatch, capsys):
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    pc.open_browser("http://localhost:9999/")
    assert "Open in browser" not in capsys.readouterr().out
