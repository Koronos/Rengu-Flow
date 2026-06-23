"""TrainingBackend factory + capability surface (CPU-only, no torch/deepspeed needed)."""
import importlib
import sys

import pytest


def test_select_backend_resolution(monkeypatch):
    from rengu_flow.engine import select_backend
    monkeypatch.delenv("RENGU_ENGINE", raising=False)
    assert select_backend({"engine": "accelerate"}).name == "accelerate"
    assert select_backend({"engine": "deepspeed"}).name == "deepspeed"
    monkeypatch.setenv("RENGU_ENGINE", "accelerate")
    assert select_backend({"engine": "deepspeed"}).name == "accelerate"  # env wins


def test_select_backend_unknown():
    from rengu_flow.engine import select_backend
    with pytest.raises(SystemExit):
        select_backend({"engine": "nope"})


def test_capabilities():
    from rengu_flow.engine import select_backend
    acc = select_backend({"engine": "accelerate"})
    ds = select_backend({"engine": "deepspeed"})
    assert acc.is_distributed is False and ds.is_distributed is True
    assert acc.supports_gradient_release is False and ds.supports_gradient_release is True


def test_base_is_torch_free():
    # base.py must import without torch/deepspeed (CLI launch + config validation path).
    # Run in a subprocess so sys.modules manipulation doesn't pollute torch global state for
    # other tests (torch's triton registration is process-global and cannot be undone).
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules.pop('torch', None); sys.modules.pop('deepspeed', None);"
         "import importlib; importlib.import_module('rengu_flow.engine.base');"
         "assert 'deepspeed' not in sys.modules"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).parent.parent)},
    )
    assert result.returncode == 0, result.stderr


def test_launch_argv_accelerate():
    from rengu_flow.engine import select_backend
    argv = select_backend({"engine": "accelerate"}).launch_argv(
        {"engine": "accelerate"}, config_path="cfg.toml", num_gpus=1, master_port=29500
    )
    assert argv[1:3] == ["-m", "rengu_flow.main"]
    assert "--config" in argv and "cfg.toml" in argv
