"""Version reporting: single source of truth (module), CLI, and the /version API endpoint."""

from __future__ import annotations

import re

from rengu_flow import version as version_mod


def test_package_version_is_real_not_sentinel():
    # Installed metadata or the pyproject fallback must yield a real version, never the sentinel.
    v = version_mod.package_version()
    assert v and v != "0.0.0+unknown"
    assert re.match(r"^\d+\.\d+", v)


def test_package_version_prefers_pyproject_over_stale_metadata(monkeypatch):
    """In a source checkout the just-pulled pyproject wins over a stale editable-install record,
    so the UI reflects a `git pull` even when uv skips reinstalling the project."""
    version_mod.package_version.cache_clear()
    monkeypatch.setattr(version_mod, "_version_from_pyproject", lambda: "9.9.9")
    monkeypatch.setattr(
        version_mod, "_dist_version", lambda _name: "0.0.1"  # stale installed metadata
    )
    try:
        assert version_mod.package_version() == "9.9.9"
    finally:
        version_mod.package_version.cache_clear()


def test_package_version_falls_back_to_metadata_for_wheel_install(monkeypatch):
    """No pyproject beside the package (wheel install) -> use the installed distribution metadata."""
    version_mod.package_version.cache_clear()
    monkeypatch.setattr(version_mod, "_version_from_pyproject", lambda: None)
    monkeypatch.setattr(version_mod, "_dist_version", lambda _name: "1.2.3")
    try:
        assert version_mod.package_version() == "1.2.3"
    finally:
        version_mod.package_version.cache_clear()


def test_version_string_includes_commit_when_in_checkout():
    s = version_mod.version_string()
    assert s.startswith(version_mod.package_version())


def test_version_info_shape():
    info = version_mod.version_info()
    assert set(info) == {"version", "commit", "kaon"}
    assert info["version"] == version_mod.package_version()


def test_git_revision_none_outside_checkout(tmp_path):
    # A directory with no .git -> None (graceful, never raises).
    assert version_mod.git_revision(tmp_path) is None


def test_cli_version_flag_prints_and_exits(capsys):
    from rengu_flow.cli.main import main

    main(["--version"])  # short-circuits before platform/config setup
    out = capsys.readouterr().out
    assert "rengu-flow" in out
    assert version_mod.package_version() in out


def test_cli_version_subcommand(capsys):
    from rengu_flow.cli.main import main

    main(["version"])
    out = capsys.readouterr().out
    assert "rengu-flow" in out
    assert "kaon:" in out


def test_api_version_endpoint(ui_client):
    r = ui_client.get("/api/v1/version")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"version", "commit", "kaon"}
    assert body["version"] == version_mod.package_version()
