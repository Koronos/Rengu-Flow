"""CLI behavior: `rengu update` git pull + `rengu ui` defaulting to start."""

from __future__ import annotations

import argparse
import subprocess

import rengu_flow.cli.update_cmd as update_cmd
from rengu_flow.cli.main import _build_parser


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


def _make_upstream(path):
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    (path / "code.py").write_text("v1\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-m", "v1")


def _stub_sync(monkeypatch, calls):
    monkeypatch.setattr(update_cmd, "sync_dependencies", lambda profiles, **k: calls.setdefault("sync", profiles))
    monkeypatch.setattr(update_cmd, "record_installed_profiles", lambda profiles, **k: None)
    monkeypatch.setattr(update_cmd, "read_installed_profiles", lambda *a, **k: [])
    monkeypatch.setattr(update_cmd, "reexec_cli", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "rebuild_web", lambda profiles, **k: calls.setdefault("rebuild", profiles))


def test_git_pull_skips_when_not_a_git_checkout(tmp_path, capsys):
    # No .git (and never destructive): returns False and explains it only synced deps.
    assert update_cmd.git_pull(tmp_path) is False
    assert "skipping code update" in capsys.readouterr().out


def test_update_pulls_from_repo_by_default(monkeypatch):
    calls = {}
    monkeypatch.setattr(update_cmd, "git_pull", lambda *a, **k: calls.setdefault("pull", True))
    _stub_sync(monkeypatch, calls)
    update_cmd.run(argparse.Namespace(profiles=["base"], all_extras=False, no_pull=False))
    assert calls.get("pull") is True
    assert calls.get("sync") == ["base"]


def test_update_no_pull_only_syncs(monkeypatch):
    calls = {}
    monkeypatch.setattr(update_cmd, "git_pull", lambda *a, **k: calls.setdefault("pull", True))
    _stub_sync(monkeypatch, calls)
    update_cmd.run(argparse.Namespace(profiles=["ui"], all_extras=False, no_pull=True))
    assert "pull" not in calls
    assert calls.get("sync") == ["ui"]


def test_update_refreshes_previously_installed_profiles(monkeypatch):
    # kaon was installed earlier; a plain `rengu update` must refresh it too (so a bumped commit
    # pin in the pulled pyproject is applied) without the user re-listing it.
    calls = {}
    monkeypatch.setattr(update_cmd, "git_pull", lambda *a, **k: None)
    _stub_sync(monkeypatch, calls)
    monkeypatch.setattr(update_cmd, "read_installed_profiles", lambda *a, **k: ["kaon"])
    update_cmd.run(argparse.Namespace(profiles=["base"], all_extras=False, no_pull=False))
    assert "kaon" in calls.get("sync")
    assert "base" in calls.get("sync")


def test_update_leaves_uninstalled_git_profiles_untouched(monkeypatch):
    # Nothing optional installed -> a plain update never pulls in kaon for someone who never
    # enabled it.
    calls = {}
    monkeypatch.setattr(update_cmd, "git_pull", lambda *a, **k: None)
    _stub_sync(monkeypatch, calls)  # read_installed_profiles -> []
    update_cmd.run(argparse.Namespace(profiles=["base"], all_extras=False, no_pull=False))
    assert "kaon" not in calls.get("sync")


def test_update_recompiles_ui_after_sync(monkeypatch):
    calls = {}
    monkeypatch.setattr(update_cmd, "git_pull", lambda *a, **k: None)
    _stub_sync(monkeypatch, calls)
    update_cmd.run(argparse.Namespace(profiles=["base"], all_extras=False, no_pull=False))
    assert calls.get("rebuild") == ["base"]


def test_rebuild_web_skips_when_no_dist_and_ui_not_requested(tmp_path, monkeypatch, capsys):
    # Training-only machine: no built dist, base profile -> never invokes the Node build.
    built = {"called": False}
    import rengu_flow.cli.ui_cmd as ui_cmd

    monkeypatch.setattr(ui_cmd, "_build_web", lambda *a, **k: built.__setitem__("called", True))
    update_cmd.rebuild_web(["base"], root=tmp_path)
    assert built["called"] is False


def test_rebuild_web_runs_when_dist_exists(tmp_path, monkeypatch):
    built = {"force": None}
    import rengu_flow.cli.ui_cmd as ui_cmd

    dist = tmp_path / "ui" / "web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(ui_cmd, "_build_web", lambda root, *, force=False: built.__setitem__("force", force))
    update_cmd.rebuild_web(["base"], root=tmp_path)
    assert built["force"] is True


def test_rebuild_web_warns_instead_of_crashing_without_node(tmp_path, monkeypatch, capsys):
    import rengu_flow.cli.ui_cmd as ui_cmd

    def _no_node(*a, **k):
        raise SystemExit("rengu: npm not found.")

    monkeypatch.setattr(ui_cmd, "_build_web", _no_node)
    # ui requested -> attempts a build, but a missing toolchain must not abort the update.
    update_cmd.rebuild_web(["ui"], root=tmp_path)
    assert "Skipped UI rebuild" in capsys.readouterr().out


def test_update_repo_url_is_canonical():
    assert update_cmd.REPO_URL == "https://github.com/Koronos/Rengu-Flow"


def test_force_flag_parses():
    args = _build_parser().parse_args(["update", "--force"])
    assert args.force is True
    assert _build_parser().parse_args(["update"]).force is False


def test_force_pull_resets_to_upstream_keeping_untracked_db(tmp_path, monkeypatch):
    # A clone with local tracked "noise" + an untracked jobs.db. --force must update the code
    # and preserve the untracked DB (never `git clean`).
    upstream = tmp_path / "upstream"
    _make_upstream(upstream)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(upstream), str(clone))
    _git(clone, "config", "user.email", "t@t")
    _git(clone, "config", "user.name", "t")

    # Upstream advances...
    (upstream / "code.py").write_text("v2\n", encoding="utf-8")
    _git(upstream, "add", "-A")
    _git(upstream, "commit", "-m", "v2")

    # ...while the clone has an uncommitted tracked edit that blocks a fast-forward,
    data_dir = clone / "data"
    data_dir.mkdir()
    db = data_dir / "jobs.db"
    db.write_text("PRECIOUS", encoding="utf-8")
    (clone / "code.py").write_text("local noise\n", encoding="utf-8")

    monkeypatch.setattr(update_cmd, "REPO_URL", str(upstream))
    assert update_cmd.git_pull(clone, force=True) is True

    assert (clone / "code.py").read_text(encoding="utf-8") == "v2\n"
    assert db.read_text(encoding="utf-8") == "PRECIOUS"  # untracked DB untouched


def test_plain_pull_does_not_touch_untracked_db_on_failure(tmp_path, monkeypatch, capsys):
    # Non-force pull is blocked by local changes: it must abort cleanly and keep the DB.
    upstream = tmp_path / "upstream"
    _make_upstream(upstream)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(upstream), str(clone))

    (upstream / "code.py").write_text("v2\n", encoding="utf-8")
    _git(upstream, "add", "-A")
    _git(upstream, "commit", "-m", "v2")

    db = clone / "data" / "jobs.db"
    db.parent.mkdir()
    db.write_text("PRECIOUS", encoding="utf-8")
    (clone / "code.py").write_text("local noise\n", encoding="utf-8")

    monkeypatch.setattr(update_cmd, "REPO_URL", str(upstream))
    assert update_cmd.git_pull(clone, force=False) is False
    assert "rengu update --force" in capsys.readouterr().out
    assert db.read_text(encoding="utf-8") == "PRECIOUS"


def test_ui_bare_invocation_defaults_to_start():
    args = _build_parser().parse_args(["ui"])
    assert args.command == "ui"
    # No subcommand on the namespace -> run() resolves it to "start".
    assert (args.ui_command or "start") == "start"


def test_ui_explicit_subcommands_still_parse():
    for sub in ("start", "serve", "dev", "build", "reset-db"):
        args = _build_parser().parse_args(["ui", sub])
        assert args.ui_command == sub
