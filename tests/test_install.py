"""Tests for the centralized additive dependency installer (rengu_flow.install)."""

from __future__ import annotations

import pytest

from rengu_flow.install import manager
from rengu_flow.install.runner import run_uv_pip_install
from rengu_flow.install.state import read_installed_profiles, record_installed_profiles


# --- persisted state ---------------------------------------------------------------------------

def test_installed_profiles_roundtrip_and_additive_merge(tmp_path):
    assert read_installed_profiles(tmp_path) == []
    record_installed_profiles(["ui", "cosmos"], root=tmp_path)
    assert read_installed_profiles(tmp_path) == ["ui", "cosmos"]
    # Merge is additive and order-preserving; duplicates are ignored.
    record_installed_profiles(["ui", "optim"], root=tmp_path)
    assert read_installed_profiles(tmp_path) == ["ui", "cosmos", "optim"]


def test_read_installed_profiles_drops_unknown(tmp_path):
    """A state file can outlive the set of known profiles (a dropped/renamed library leaves a
    stale name). Unknown names are silently dropped so self_heal never raises; known ones survive,
    order-preserved and de-duped."""
    import json

    from rengu_flow.install.state import installed_profiles_path

    path = installed_profiles_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 'koptim' (a removed library's profile) and 'bogus' are no longer valid -> dropped.
    path.write_text(
        json.dumps({"profiles": ["ui", "koptim", "bogus", "kaon", "kaon"]}), encoding="utf-8"
    )
    assert read_installed_profiles(tmp_path) == ["ui", "kaon"]


def test_install_state_migrates_from_legacy_hidden_dir(tmp_path):
    """The record moves out of the retired hidden .rengu-flow/ into the visible data/ on access."""
    import json

    from rengu_flow.install.state import LEGACY_STATE_DIRNAME, installed_profiles_path

    legacy = tmp_path / LEGACY_STATE_DIRNAME
    legacy.mkdir()
    (legacy / "installed-profiles.json").write_text(
        json.dumps({"profiles": ["ui", "cosmos"]}), encoding="utf-8"
    )

    # Reading triggers the one-time migration.
    assert read_installed_profiles(tmp_path) == ["ui", "cosmos"]
    assert installed_profiles_path(tmp_path).is_file()  # now under data/
    assert not legacy.exists()  # hidden folder removed

    # A subsequent record keeps working against the new location.
    record_installed_profiles(["optim"], root=tmp_path)
    assert read_installed_profiles(tmp_path) == ["ui", "cosmos", "optim"]


# --- run_uv_pip_install (additive git/VCS specs) -----------------------------------------------

def test_run_uv_pip_install_builds_argv(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    class _Proc:
        returncode = 0

    monkeypatch.setattr(
        "rengu_flow.install.runner.subprocess.run",
        lambda cmd, cwd=None: calls.append(cmd) or _Proc(),
    )
    rc = run_uv_pip_install(["git+https://example.com/x@v1"], root=tmp_path)
    assert rc == 0
    assert calls == [["uv", "pip", "install", "git+https://example.com/x@v1"]]


def test_run_uv_pip_install_noop_on_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rengu_flow.install.runner.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    assert run_uv_pip_install([], root=tmp_path) == 0


# --- ensure_profiles ---------------------------------------------------------------------------

def _wire(monkeypatch, installed: set[str]):
    """Make the manager's probes/installers operate on an in-memory ``installed`` set."""
    monkeypatch.setattr(manager, "profile_installed", lambda p: p in installed)
    monkeypatch.setattr(manager, "require_uv", lambda: None)
    synced: list[list[str]] = []

    def fake_sync(profiles, root=None):
        synced.append(list(profiles))
        installed.update(profiles)  # a normal extra is satisfied by uv sync

    monkeypatch.setattr("rengu_flow.cli.project_venv.sync_dependencies", fake_sync)
    return synced


def test_ensure_profiles_noop_when_present_but_records(tmp_path, monkeypatch):
    synced = _wire(monkeypatch, {"ui"})
    missing = manager.ensure_profiles(["ui"], root=tmp_path)
    assert missing == []
    assert synced == []  # nothing installed when already importable
    assert read_installed_profiles(tmp_path) == ["ui"]  # still recorded for self-heal


def test_ensure_profiles_installs_missing_and_records(tmp_path, monkeypatch):
    synced = _wire(monkeypatch, set())
    missing = manager.ensure_profiles(["cosmos"], root=tmp_path)
    assert missing == ["cosmos"]
    assert synced == [["cosmos"]]
    assert read_installed_profiles(tmp_path) == ["cosmos"]


def test_ensure_profiles_installs_git_requirement(tmp_path, monkeypatch):
    installed: set[str] = set()
    synced = _wire(monkeypatch, installed)
    # uv sync does NOT satisfy this profile; only the git spec does.
    monkeypatch.setattr("rengu_flow.cli.project_venv.sync_dependencies", lambda p, root=None: synced.append(list(p)))
    monkeypatch.setattr(manager, "PROFILE_GIT_REQUIREMENTS", {"optim": ["git+https://x/y@v1"]})
    pip_calls: list[list[str]] = []

    def fake_pip(specs, root=None):
        pip_calls.append(list(specs))
        installed.add("optim")

    monkeypatch.setattr(manager, "run_uv_pip_install_or_exit", fake_pip)

    missing = manager.ensure_profiles(["optim"], root=tmp_path)
    assert missing == ["optim"]
    assert pip_calls == [["git+https://x/y@v1"]]
    assert read_installed_profiles(tmp_path) == ["optim"]


def test_ensure_profiles_raises_when_still_missing(tmp_path, monkeypatch):
    # sync claims to run but the module never imports -> hard error.
    monkeypatch.setattr(manager, "profile_installed", lambda p: False)
    monkeypatch.setattr(manager, "require_uv", lambda: None)
    monkeypatch.setattr("rengu_flow.cli.project_venv.sync_dependencies", lambda p, root=None: None)
    with pytest.raises(SystemExit):
        manager.ensure_profiles(["cosmos"], root=tmp_path)


def test_self_heal_restores_recorded_profiles(tmp_path, monkeypatch):
    record_installed_profiles(["cosmos"], root=tmp_path)
    synced = _wire(monkeypatch, set())
    restored = manager.self_heal(root=tmp_path)
    assert restored == ["cosmos"]
    assert synced == [["cosmos"]]


def test_self_heal_noop_when_nothing_recorded(tmp_path, monkeypatch):
    synced = _wire(monkeypatch, set())
    assert manager.self_heal(root=tmp_path) == []
    assert synced == []
