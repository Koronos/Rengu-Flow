"""Frozen-DiT quantization (fp8 scaled matmul / 4-bit NF4) + LoKr-on-quantized-base wiring.

CPU-only. The fp8 ``_scaled_mm`` and bitsandbytes 4-bit kernels may be GPU-only at runtime;
those forward paths are guarded and skipped (not failed) when the kernel is unavailable on CPU.
"""

import pytest

try:
    import torch
    from torch import nn

    from rengu_flow.training import quantize_dit as q
    from rengu_flow.networks.lokr_vendored import _apply_lokr_vendored
except ImportError as e:  # heavy deps missing
    pytest.skip(f"Cannot import torch/networks: {e}", allow_module_level=True)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)


class _FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layer1 = nn.Linear(d, d * 2, bias=False)
        self.layer2 = nn.Linear(d * 2, d, bias=False)


class Block(nn.Module):  # name must be in ADAPTER_TARGET_MODULES for LoKr targeting
    def __init__(self, d):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _FFN(d)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 3 * d, bias=False))


class TinyDiT(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.x_embedder = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList([Block(d), Block(d)])
        self.final_layer = nn.Linear(d, d, bias=False)


def _freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_target_selection_skips_embedders_final_adaln():
    m = TinyDiT()
    targets = {n for n, mod in m.named_modules() if q._is_quantizable_block_linear(n, mod)}
    expect = set()
    for bi in (0, 1):
        for leaf in ("q_proj", "k_proj", "v_proj", "output_proj"):
            expect.add(f"blocks.{bi}.self_attn.{leaf}")
        expect.add(f"blocks.{bi}.mlp.layer1")
        expect.add(f"blocks.{bi}.mlp.layer2")
    assert targets == expect
    assert not any(("x_embedder" in t or "final_layer" in t or "adaln" in t) for t in targets)


def test_resolve_fp8_dtype():
    assert q.resolve_fp8_dtype("e5m2") is torch.float8_e5m2
    assert q.resolve_fp8_dtype("e4m3") is torch.float8_e4m3fn
    with pytest.raises(ValueError):
        q.resolve_fp8_dtype("e3m4")


def test_fp8_convert_keeps_frozen_and_subclasses_linear():
    m = _freeze(TinyDiT())
    n = q.convert_dit_to_fp8_matmul(m, fp8_dtype=torch.float8_e5m2)
    assert n == 12
    lin = m.blocks[0].self_attn.q_proj
    assert isinstance(lin, q.Fp8MatmulLinear) and isinstance(lin, nn.Linear)
    assert lin.weight.requires_grad is False
    assert all(not p.requires_grad for p in m.parameters())  # no trainable scales added


def test_fp8_weight_quant_is_reasonable():
    lin = nn.Linear(32, 48, bias=False)
    fl = q.Fp8MatmulLinear(lin, weight_fp8_dtype=torch.float8_e5m2)
    deq = fl.weight_fp8.float() * fl.weight_scale
    rel = (deq - lin.weight.detach().float()).abs().mean() / lin.weight.detach().float().abs().mean()
    assert rel < 0.10  # e5m2 has 2 mantissa bits; ~4% in practice


def test_fp8_forward_or_gpu_only():
    lin = nn.Linear(32, 48, bias=False)
    fl = q.Fp8MatmulLinear(lin, weight_fp8_dtype=torch.float8_e5m2)
    x = torch.randn(4, 32)
    try:
        y = fl(x)
    except (RuntimeError, ValueError, NotImplementedError) as e:
        pytest.skip(f"fp8 _scaled_mm GPU-only on this device: {e}")
    assert y.shape == (4, 48)
    ref = x @ lin.weight.detach().t()
    rel = (y.float() - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.15


def test_4bit_convert_keeps_frozen_and_subclasses_linear():
    bnb = pytest.importorskip("bitsandbytes")
    m = _freeze(TinyDiT())
    try:
        n = q.convert_dit_to_4bit(m, compute_dtype=torch.bfloat16)
    except (RuntimeError, AssertionError, NotImplementedError) as e:
        pytest.skip(f"bnb 4-bit quantize GPU-only on this device: {e}")
    assert n == 12
    lin = m.blocks[0].self_attn.q_proj
    assert isinstance(lin, bnb.nn.Linear4bit) and isinstance(lin, nn.Linear)
    assert lin.weight.requires_grad is False


@pytest.mark.parametrize("scheme", ["fp8", "4bit"])
def test_lokr_on_quantized_base_wiring(scheme):
    m = _freeze(TinyDiT())
    if scheme == "fp8":
        q.convert_dit_to_fp8_matmul(m, fp8_dtype=torch.float8_e5m2)
    else:
        pytest.importorskip("bitsandbytes")
        try:
            q.convert_dit_to_4bit(m, compute_dtype=torch.bfloat16)
        except (RuntimeError, AssertionError, NotImplementedError) as e:
            pytest.skip(f"bnb 4-bit GPU-only: {e}")

    cfg = {"type": "lokr", "rank": 4, "alpha": 4, "factor": -1,
           "decompose_both": False, "full_matrix": False, "dtype": torch.float32}
    _apply_lokr_vendored(m, list(m.blocks), cfg, "")

    lin = m.blocks[0].self_attn.q_proj
    assert hasattr(lin, "_lokr_scale")  # LoKr injected onto the quantized linear
    assert q.base_linear_of(lin) is not None  # quantized base path detected
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all("lokr_" in n for n in trainable)  # only LoKr trains

    x = torch.randn(2, 16)
    try:
        y = lin(x)
    except (RuntimeError, ValueError, AssertionError, NotImplementedError) as e:
        pytest.skip(f"quantized forward GPU-only on this device: {e}")
    assert y.shape == (2, 16)


def test_lokr_delta_decomposition_matches_fused():
    """base(x) + F.linear(x, diff) == F.linear(x, W + diff) (the QLoRA composition)."""
    import torch.nn.functional as F

    W = torch.randn(8, 6)
    diff = torch.randn(8, 6) * 0.01
    x = torch.randn(3, 6)
    a = F.linear(x, W) + F.linear(x, diff)
    b = F.linear(x, W + diff)
    assert torch.allclose(a, b, atol=1e-5)
