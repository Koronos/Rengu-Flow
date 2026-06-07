"""Version reporting: single source of truth (module), CLI, and the /version API endpoint."""

from __future__ import annotations

import re

from rengu_flow import version as version_mod


def test_package_version_is_real_not_sentinel():
    # Installed metadata or the pyproject fallback must yield a real version, never the sentinel.
    v = version_mod.package_version()
    assert v and v != "0.0.0+unknown"
    assert re.match(r"^\d+\.\d+", v)


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
