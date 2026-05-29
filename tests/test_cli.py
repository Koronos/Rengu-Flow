"""Tests for the rengu CLI (no uv subprocess)."""

import argparse
from pathlib import Path

import pytest

import importlib

cli_main_mod = importlib.import_module("rengu_flow.cli.main")
from rengu_flow.install_profiles import normalize_profiles, uv_sync_argv


def test_normalize_profiles_all():
    names = normalize_profiles(["all"])
    assert "base" in names
    assert "ui" in names
    assert "cosmos" in names


def test_uv_sync_argv_ui_extra():
    argv = uv_sync_argv(["ui"])
    assert argv[:2] == ["uv", "sync"]
    assert "--extra" in argv
    assert "ui" in argv


def test_legacy_train_dispatch(monkeypatch):
    called = {}

    def fake_run_prepared(args):
        called["config"] = args.config

    def fake_parse(argv):
        assert "--config" in argv
        idx = argv.index("--config")
        return argparse.Namespace(config=argv[idx + 1], dump_dataset=None)

    monkeypatch.setattr("rengu_flow.main.parse_args", fake_parse)
    monkeypatch.setattr("rengu_flow.main.run_prepared", fake_run_prepared)
    monkeypatch.setattr(cli_main_mod.platform, "require_linux", lambda: None)
    monkeypatch.setattr(cli_main_mod, "load_local_config", lambda: None)
    monkeypatch.setattr(cli_main_mod, "apply_local_config_to_environ", lambda: None)
    monkeypatch.setattr(
        "rengu_flow.cli.training_extras.ensure_training_extras",
        lambda *_a, **_k: [],
    )

    cli_main_mod.main(["--config", "my.toml"])
    assert called["config"] == "my.toml"


def test_install_alias_maps_to_init():
    argv = cli_main_mod._normalize_argv(["install", "ui"])
    assert argv == ["init", "ui"]


def test_sync_dependencies_requires_uv(monkeypatch):
    from rengu_flow.cli import project_venv

    monkeypatch.setattr(project_venv, "require_uv", lambda: (_ for _ in ()).throw(SystemExit(1)))
    with pytest.raises(SystemExit):
        project_venv.sync_dependencies(["ui"])


def test_sync_dependencies_calls_uv_sync(tmp_path, monkeypatch):
    from rengu_flow.cli import project_venv

    root = tmp_path
    (root / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\n', encoding="utf-8"
    )
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("", encoding="utf-8")
    synced: list[list[str]] = []

    monkeypatch.setattr(project_venv, "require_uv", lambda: None)
    monkeypatch.setattr(project_venv, "repo_root", lambda: root)
    monkeypatch.setattr(project_venv, "run_uv_venv", lambda _root=None: 0)
    monkeypatch.setattr(
        project_venv,
        "run_uv_sync_or_exit",
        lambda profiles: synced.append(profiles),
    )

    project_venv.ensure_ui_dependencies(root=root)
    assert synced == [["ui"]]


def test_train_launcher_builds_deepspeed_cmd(tmp_path, monkeypatch):
    from rengu_flow.cli.train_launcher import build_train_command
    from rengu_flow.config.local_config import LocalConfig, TrainingConfig, load_local_config

    monkeypatch.setattr(
        "rengu_flow.cli.train_launcher.ensure_local_config_loaded",
        lambda: LocalConfig(root=tmp_path, training=TrainingConfig(num_gpus=1, master_port=29500)),
    )
    monkeypatch.setattr("rengu_flow.cli.train_launcher.which", lambda _: "/usr/bin/deepspeed")

    cfg = tmp_path / "train.toml"
    cfg.write_text("dataset = \"x.toml\"\n", encoding="utf-8")
    cmd = build_train_command(cfg)
    assert cmd[0].endswith("deepspeed")
    assert "--num_gpus=1" in cmd
    assert "rengu_flow.main" in cmd
