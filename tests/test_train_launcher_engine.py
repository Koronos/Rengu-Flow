"""Tests that base_train_command delegates to the engine backend's launch_argv."""
from pathlib import Path

from rengu_flow.cli.train_launcher import base_train_command


def test_accelerate_uses_python_m(monkeypatch):
    monkeypatch.setenv("RENGU_ENGINE", "accelerate")
    cmd = base_train_command(Path("cfg.toml"), num_gpus=1, master_port=29500)
    assert cmd[1:3] == ["-m", "rengu_flow.main"]


def test_deepspeed_uses_launcher(monkeypatch):
    # Simulate the deepspeed launcher being present so we assert the real launcher shape
    # (not the python -m fallback that fires when `which("deepspeed")` is None).
    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")
    monkeypatch.setattr("rengu_flow.engine.deepspeed_pipe.which", lambda _: "/usr/bin/deepspeed")
    cmd = base_train_command(Path("cfg.toml"), num_gpus=2, master_port=29500)
    assert cmd[0] == "/usr/bin/deepspeed"
    assert "--num_gpus=2" in cmd
    assert "--module" in cmd and "rengu_flow.main" in cmd
