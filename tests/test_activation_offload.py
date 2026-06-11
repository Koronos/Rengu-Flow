"""Unit tests for ActivationOffloader (saved-tensor offload to pinned CPU RAM).

Policy, pooling and the full pack/unpack round-trip are exercised on CPU via
``sync=True`` (same code paths minus the CUDA side streams); the overlapped
transport gets a gradient-correctness test that runs only when CUDA is
available.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rengu_flow.training.activation_offload import ActivationOffloader, _PinnedPool


def _offloader(**kw) -> ActivationOffloader:
    kw.setdefault("sync", True)
    kw.setdefault("verbose", False)
    kw.setdefault("min_tensor_mb", 0.0)
    return ActivationOffloader(**kw)


# ----------------------------------------------------------------------- policy
def test_small_tensors_stay() -> None:
    off = _offloader(min_tensor_mb=1.0)
    packed = off._pack(torch.randn(8))  # tiny
    assert packed[0] == "keep"
    assert off.kept_count == 1 and off.packed_count == 0


def test_param_tensors_stay() -> None:
    p = nn.Parameter(torch.randn(1024))
    off = _offloader(params_provider=lambda: [p])
    with off.step():
        assert off._pack(p.data)[0] == "keep"
        assert off._pack(torch.randn(1024))[0] == "cpu"


def test_non_contiguous_tensors_stay() -> None:
    off = _offloader()
    t = torch.randn(64, 64).t()
    assert not t.is_contiguous()
    assert off._pack(t)[0] == "keep"


def test_ram_cap_stops_offload() -> None:
    off = _offloader(max_ram_gb=4096 / (1 << 30))  # 4 KB cap
    a = torch.randn(512)  # 2 KB
    assert off._pack(a)[0] == "cpu"
    assert off._pack(torch.randn(512))[0] == "cpu"  # pool now 4 KB
    assert off._pack(torch.randn(512))[0] == "keep"  # over cap


# ------------------------------------------------------------------ round-trip
def test_pack_unpack_round_trip_values() -> None:
    off = _offloader()
    t = torch.randn(33, 7)
    packed = off._pack(t.clone())
    assert packed[0] == "cpu"
    out = off._unpack(packed)
    assert out.shape == t.shape and out.dtype == t.dtype
    torch.testing.assert_close(out, t)


def test_unpack_twice_raises() -> None:
    off = _offloader()
    packed = off._pack(torch.randn(256))
    off._unpack(packed)
    with pytest.raises(RuntimeError, match="unpacked twice"):
        off._unpack(packed)


def test_step_recycles_buffers_and_counts() -> None:
    off = _offloader()
    with off.step():
        packed = off._pack(torch.randn(256))
        off._unpack(packed)
        off._pack(torch.randn(256))  # never unpacked -> recycled at step end
    assert off.steps == 1
    assert off.packed_count == 2
    assert off._records == []
    # both buffers back in the pool: a third take allocates nothing new
    allocated = off._pool.allocated_bytes
    with off.step():
        off._pack(torch.randn(256))
    assert off._pool.allocated_bytes == allocated


def test_backward_through_hooks_matches_baseline_cpu() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))
    x = torch.randn(4, 64)
    model.zero_grad()
    model(x).pow(2).mean().backward()
    baseline = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    off = _offloader(params_provider=model.parameters)
    with off.step():
        model(x).pow(2).mean().backward()
    assert off.packed_count > 0
    for p, ref in zip(model.parameters(), baseline):
        torch.testing.assert_close(p.grad, ref)


# ----------------------------------------------------------------------- pool
def test_pinned_pool_reuses_by_key() -> None:
    pool = _PinnedPool(pin=False)
    a = pool.take(128, torch.float32)
    pool.give(a)
    b = pool.take(128, torch.float32)
    assert b is a
    c = pool.take(128, torch.float16)  # different dtype -> new buffer
    assert c is not a
    assert pool.allocated_bytes == 128 * 4 + 128 * 2


# --------------------------------------------------------------------- config
def test_from_config_disabled_returns_none() -> None:
    assert ActivationOffloader.from_config({}) is None
    assert ActivationOffloader.from_config({"activation_offload": False}) is None


def test_from_config_requires_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        ActivationOffloader.from_config({"activation_offload": True})


def test_from_config_rejects_reduce_overhead(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="reduce-overhead"):
        ActivationOffloader.from_config(
            {"activation_offload": True, "compile_mode": "reduce-overhead"}
        )


def test_from_config_validates_numbers(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="min_tensor_mb"):
        ActivationOffloader.from_config(
            {"activation_offload": True, "activation_offload_min_tensor_mb": -1}
        )
    with pytest.raises(ValueError, match="max_ram_gb"):
        ActivationOffloader.from_config(
            {"activation_offload": True, "activation_offload_max_ram_gb": 0}
        )


def test_set_config_defaults_includes_offload_keys(minimal_config_copy) -> None:
    from rengu_flow.config.defaults import set_config_defaults

    set_config_defaults(minimal_config_copy)
    assert minimal_config_copy["activation_offload"] is False
    assert minimal_config_copy["activation_offload_min_tensor_mb"] == 4.0
    assert minimal_config_copy["activation_offload_max_ram_gb"] is None
    assert minimal_config_copy["activation_offload_prefetch_mb"] == 512.0


# ----------------------------------------------------------------- CUDA (real)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_async_backward_matches_baseline_cuda() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256)
    ).cuda()
    x = torch.randn(64, 256, device="cuda")
    model.zero_grad()
    model(x).pow(2).mean().backward()
    baseline = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    off = ActivationOffloader(
        min_tensor_mb=0.0, prefetch_mb=1.0, params_provider=model.parameters, verbose=False
    )
    for _ in range(3):  # several steps: exercises pool recycling across steps
        model.zero_grad()
        with off.step():
            model(x).pow(2).mean().backward()
    assert off.packed_count > 0
    torch.cuda.synchronize()
    for p, ref in zip(model.parameters(), baseline):
        torch.testing.assert_close(p.grad, ref)
