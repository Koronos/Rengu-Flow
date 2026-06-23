"""Tests for the rengu CLI (no uv subprocess)."""

import argparse

import pytest

import importlib

cli_main_mod = importlib.import_module("rengu_flow.cli.main")
from rengu_flow.install.profiles import normalize_profiles, uv_sync_argv


def test_normalize_profiles_all():
    names = normalize_profiles(["all"])
    assert "base" in names
    assert "ui" in names
    assert "cosmos" in names


def test_uv_sync_argv_ui_extra():
    argv = uv_sync_argv(["ui"])
    assert argv[:2] == ["uv", "sync"]
    # Additive: must never run an exact sync (which would remove other extras / custom packages).
    assert "--inexact" in argv
    assert "--extra" in argv
    assert "ui" in argv


def test_uv_sync_argv_base_is_inexact():
    from rengu_flow.platform_compat import PLATFORM

    argv = uv_sync_argv(["base"])
    assert argv[:3] == ["uv", "sync", "--inexact"]  # additive sync, never exact
    # A forced reinstall of just the editable project package keeps installed metadata in step after
    # a version bump — but it's skipped on Windows (it would replace the running rengu.exe -> WinError 32).
    assert ("--reinstall-package" in argv) == (not PLATFORM.is_windows)


def test_uv_sync_argv_skips_reinstall_on_windows(monkeypatch):
    """On Windows the force-reinstall must be dropped: a sync driven from within `rengu ui` cannot
    delete the running rengu.exe (os error 32). POSIX keeps it (can replace a running executable)."""
    import rengu_flow.platform_compat as pc
    from types import SimpleNamespace

    monkeypatch.setattr(pc, "PLATFORM", SimpleNamespace(is_windows=True))
    win = uv_sync_argv(["ui"])
    assert "--reinstall-package" not in win
    # Skips reinstalling the editable project so the running rengu.exe is never replaced.
    assert "--no-install-project" in win

    monkeypatch.setattr(pc, "PLATFORM", SimpleNamespace(is_windows=False))
    assert uv_sync_argv(["base"]) == ["uv", "sync", "--inexact", "--reinstall-package", "rengu-flow"]


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
    monkeypatch.setattr(cli_main_mod.platform, "require_supported_platform", lambda: None)
    monkeypatch.setattr(cli_main_mod, "load_local_config", lambda: None)
    monkeypatch.setattr(cli_main_mod, "apply_local_config_to_environ", lambda: None)
    monkeypatch.setattr(cli_main_mod, "ensure_local_config_file", lambda *_a, **_k: None)
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
    # Create the venv python at the platform-correct location (Scripts/ on Windows, bin/).
    py = project_venv.venv_python(root)
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
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


@pytest.fixture
def deepspeed_engine(monkeypatch):
    """Force the deepspeed engine: base_train_command only uses the DeepSpeed launcher there.
    On a non-deepspeed host (e.g. native Windows default 'accelerate') it runs the module
    directly, which is what test_base_train_command_accelerate_bypasses_launcher checks."""
    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")


def test_base_train_command_accelerate_bypasses_launcher(tmp_path, monkeypatch):
    from rengu_flow.cli import train_launcher

    monkeypatch.setenv("RENGU_ENGINE", "accelerate")
    monkeypatch.setattr(train_launcher, "which", lambda _: "/usr/bin/deepspeed")
    cmd = train_launcher.base_train_command(tmp_path / "t.toml", num_gpus=1)
    # accelerate is single-GPU: run the module directly, never the DeepSpeed launcher.
    assert cmd[1] == "-m" and cmd[2] == "rengu_flow.main"
    assert not cmd[0].endswith("deepspeed")


def test_train_engine_flag_forces_backend(tmp_path, monkeypatch):
    """`rengu train --engine accelerate` must force accelerate even with deepspeed on PATH
    (e.g. testing accelerate on Linux): it sets RENGU_ENGINE so the launcher picks `python -m`."""
    import argparse

    from rengu_flow.cli import train_cmd, train_launcher
    from rengu_flow.config.local_config import LocalConfig, TrainingConfig

    monkeypatch.delenv("RENGU_ENGINE", raising=False)  # records cleanup -> teardown restores
    cfg = LocalConfig(root=tmp_path, training=TrainingConfig(num_gpus=1, master_port=29500))
    monkeypatch.setattr(train_cmd, "ensure_local_config_loaded", lambda: cfg)
    monkeypatch.setattr(train_launcher, "ensure_local_config_loaded", lambda: cfg)
    monkeypatch.setattr(train_cmd, "ensure_training_extras", lambda *a, **k: None)
    monkeypatch.setattr(train_launcher, "which", lambda _: "/usr/bin/deepspeed")  # deepspeed available
    captured: dict = {}
    monkeypatch.setattr(
        train_cmd, "run_training_with_progress",
        lambda cmd, env=None, cwd=None: (captured.update(cmd=cmd, env=env), 0)[1],
    )

    cfgfile = tmp_path / "t.toml"
    cfgfile.write_text('dataset = "x.toml"\n', encoding="utf-8")
    args = argparse.Namespace(config=str(cfgfile), engine="accelerate", num_gpus=None,
                              master_port=None, resume_from_checkpoint=None, extra=[])
    with pytest.raises(SystemExit):
        train_cmd.run_train(args)
    import os
    assert os.environ["RENGU_ENGINE"] == "accelerate"
    assert "-m" in captured["cmd"] and not captured["cmd"][0].endswith("deepspeed")
    assert captured["env"].get("RENGU_ENGINE") == "accelerate"  # propagated to the subprocess


def test_train_launcher_builds_deepspeed_cmd(tmp_path, monkeypatch, deepspeed_engine):
    from rengu_flow.cli.train_launcher import build_train_command
    from rengu_flow.config.local_config import LocalConfig, TrainingConfig

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
    # DeepSpeed's launcher needs --module (NOT -m) to run a module target. Guard the contract.
    assert "--module" in cmd
    assert "-m" not in cmd
    assert cmd[cmd.index("--module") + 1] == "rengu_flow.main"
    assert any(a.startswith("--master_port=") for a in cmd)


def test_base_train_command_uses_module(tmp_path, monkeypatch, deepspeed_engine):
    from rengu_flow.cli import train_launcher

    monkeypatch.setattr(train_launcher, "which", lambda _: "/usr/bin/deepspeed")
    cmd = train_launcher.base_train_command(tmp_path / "t.toml", num_gpus=2, master_port=29500)
    assert cmd[0].endswith("deepspeed")
    assert "--num_gpus=2" in cmd
    assert "--master_port=29500" in cmd
    assert "--module" in cmd and "-m" not in cmd
    assert cmd[cmd.index("--module") + 1] == "rengu_flow.main"


def test_base_train_command_python_fallback(tmp_path, monkeypatch):
    from rengu_flow.cli import train_launcher

    monkeypatch.setattr(train_launcher, "which", lambda _: None)
    cmd = train_launcher.base_train_command(tmp_path / "t.toml", num_gpus=1)
    # Without deepspeed on PATH, fall back to `python -m rengu_flow.main`.
    assert cmd[1] == "-m"
    assert cmd[2] == "rengu_flow.main"


def test_ui_job_command_uses_module(tmp_path, monkeypatch, deepspeed_engine):
    """The web UI launcher must share the same --module contract as the CLI (RF-01/RF-02)."""
    from rengu_flow.cli import train_launcher
    from rengu_flow_ui import jobs as ui_jobs
    from rengu_flow_ui.jobs import build_train_command as ui_build

    monkeypatch.setattr(train_launcher, "which", lambda _: "/usr/bin/deepspeed")
    # jobs.py binds _pick_master_port by direct import — patch THAT binding, not the
    # train_launcher attribute, or the real picker runs and the asserted port drifts
    # whenever 29500 is busy (e.g. a live training run).
    monkeypatch.setattr(ui_jobs, "_pick_master_port", lambda _req: 29500)
    cmd = ui_build(tmp_path / "train.toml", num_gpus=2)
    assert cmd[0].endswith("deepspeed")
    assert "--num_gpus=2" in cmd
    assert "--master_port=29500" in cmd
    assert "--module" in cmd and "-m" not in cmd
    assert cmd[cmd.index("--module") + 1] == "rengu_flow.main"
