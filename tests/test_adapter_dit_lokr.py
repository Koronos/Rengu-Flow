"""LoKr on the Cosmos Predict2 DiT path: adapter_dit.configure must resolve target module
*names* (from _collect_target_linears) into nn.Module containers before handing them to the
vendored LoKr helper. Regression test for the str-vs-module crash:

    File "rengu_flow/networks/lokr_vendored.py", line 131, in _apply_lokr_vendored
        for sub in container.modules():
    AttributeError: 'str' object has no attribute 'modules'
"""

import pytest

try:
    import torch
    from torch import nn

    from rengu_flow.networks import adapter_dit
except ImportError as e:  # heavy deps (torch/peft) missing
    pytest.skip(f"Cannot import networks/torch: {e}", allow_module_level=True)


class Block(nn.Module):
    """Mimics a Cosmos DiT transformer block (one of adapter_dit.ADAPTER_TARGET_MODULES)."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 8)  # not under a target Block => must stay untouched
        self.blocks = nn.ModuleList([Block(), Block()])


def test_collect_target_linears_returns_names_under_blocks():
    """Sanity: _collect_target_linears yields *string* names of Linears inside target Blocks."""
    model = TinyDiT()
    names = adapter_dit._collect_target_linears(model, adapter_dit.ADAPTER_TARGET_MODULES)
    assert all(isinstance(n, str) for n in names)
    assert set(names) == {
        "blocks.0.attn",
        "blocks.0.mlp",
        "blocks.1.attn",
        "blocks.1.mlp",
    }


def test_configure_lokr_on_dit_does_not_crash_and_injects():
    """adapter_dit.configure(..., lokr) must inject LoKr params onto the Linears under Blocks
    without raising AttributeError (the str-vs-module bug)."""
    model = TinyDiT()
    cfg = {"type": "lokr", "rank": 4, "alpha": 4, "factor": -1, "dtype": torch.float32}

    peft_config, adapter_type = adapter_dit.configure(model, cfg)

    assert peft_config is None
    assert adapter_type == "lokr"

    param_names = [n for n, _ in model.named_parameters()]
    # LoKr params landed on the target Linears (inside the module tree).
    assert any(n.startswith("blocks.0.attn.lokr_") for n in param_names), param_names
    assert any(n.startswith("blocks.1.mlp.lokr_") for n in param_names), param_names

    # The non-target Linear (self.embed) was NOT adapted.
    assert not any(n.startswith("embed.lokr_") for n in param_names), param_names
    assert not hasattr(model.embed, "_lokr_scale")

    # Base weights frozen; LoKr params train.
    assert model.blocks[0].attn.weight.requires_grad is False
    assert any(p.requires_grad for n, p in model.named_parameters() if "lokr_" in n)

    # Forward through an adapted block still works.
    y = model.blocks[0].attn(torch.randn(2, 8))
    assert y.shape == (2, 8)


def test_lokr_uses_logical_dims_not_weight_shape():
    """Regression: a quantized (bnb-style) linear stores its weight packed as
    (out*in/2, 1); LoKr must factorize in/out_features, not weight.shape."""
    import torch
    from torch import nn

    from rengu_flow.networks.lokr_vendored import _inject_lokr_into_linear

    lin = nn.Linear(64, 64, bias=False)
    # Simulate a post-quantization packed weight without needing bitsandbytes.
    lin.weight = nn.Parameter(torch.zeros(64 * 64 // 2, 1, dtype=torch.uint8), requires_grad=False)
    assert lin.weight.shape != (64, 64) and lin.out_features == 64
    _inject_lokr_into_linear(lin, rank=4, alpha=4, dtype=torch.float32)
    total = 1
    for d in lin.lokr_w1.shape:
        total *= d
    assert max(lin.lokr_w1.shape) <= 64, lin.lokr_w1.shape  # factorized from 64, not 2048


def test_vec_trick_matches_dense_kron_delta():
    """The factored delta forward must equal the dense kron(w1, w2) reference exactly
    (same math, no materialization) for every factor-structure combination."""
    import torch.nn.functional as F

    from rengu_flow.networks.lokr_vendored import (
        _inject_lokr_into_linear,
        _lokr_delta_forward,
        _lokr_factors,
    )

    torch.manual_seed(0)
    for full_matrix, factor, out_dim, in_dim in [
        (True, 6, 36, 48),      # full W1 + full W2 (the krea2 full_matrix regime)
        (False, -1, 32, 64),    # low-rank W2
        (False, 4, 64, 48),
    ]:
        lin = torch.nn.Linear(in_dim, out_dim, bias=False)
        _inject_lokr_into_linear(lin, rank=4, alpha=4, factor=factor, full_matrix=full_matrix)
        for p in lin.parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, std=0.1)
        x = torch.randn(3, 5, in_dim)

        w1, w2 = _lokr_factors(lin)
        dense = torch.kron(w1, w2).reshape(out_dim, in_dim) * lin._lokr_scale
        ref = F.linear(x, dense)
        got = _lokr_delta_forward(lin, x)
        assert torch.allclose(got, ref, atol=1e-5), (full_matrix, factor, (got - ref).abs().max())

        # And gradients flow to the factors through the factored path.
        got.sum().backward()
        grads = [p.grad for n, p in lin.named_parameters() if "lokr_" in n]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)
