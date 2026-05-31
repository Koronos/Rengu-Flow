"""Unit tests for the WSL-safe CUDA allocator configuration (rengu_flow.platform_compat)."""

from __future__ import annotations

from rengu_flow.platform_compat import configure_cuda_allocator


def _pairs(conf: str) -> dict[str, str]:
    return dict(p.split(":", 1) for p in conf.split(",") if ":" in p)


def test_non_wsl_is_noop() -> None:
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    assert configure_cuda_allocator(is_wsl=False, env=env, log=False) is None
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_wsl_forces_expandable_segments_false() -> None:
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    out = configure_cuda_allocator(is_wsl=True, env=env, log=False)
    pairs = _pairs(out)
    assert pairs["expandable_segments"] == "False"
    assert "expandable_segments:True" not in out


def test_wsl_adds_low_fragmentation_defaults() -> None:
    env: dict[str, str] = {}
    pairs = _pairs(configure_cuda_allocator(is_wsl=True, env=env, log=False))
    assert pairs["expandable_segments"] == "False"
    assert pairs["garbage_collection_threshold"] == "0.8"
    assert pairs["max_split_size_mb"] == "256"


def test_wsl_preserves_user_knobs() -> None:
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:64"}
    pairs = _pairs(configure_cuda_allocator(is_wsl=True, env=env, log=False))
    # user's explicit knob wins over our default; expandable still neutralized
    assert pairs["max_split_size_mb"] == "64"
    assert pairs["expandable_segments"] == "False"


def test_wsl_empty_env_is_safe() -> None:
    env: dict[str, str] = {}
    out = configure_cuda_allocator(is_wsl=True, env=env, log=False)
    assert "expandable_segments:False" in out
