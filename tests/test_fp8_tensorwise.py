"""CPU-only unit tests for the fp8 tensorwise frozen-base path (quantize_dit):
quantization fidelity, krea2 scope selection, PEFT LoRA composition and gradient flow
through the (CPU-fallback) fp8 base."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rengu_flow.model.krea2.dit import Krea2Transformer2DModel
from rengu_flow.model.krea2.pipeline import QUANT_LEAF_NAMES, QUANT_SKIP_SUBSTRINGS
from rengu_flow.training.quantize_dit import (
    Fp8TensorwiseLinear,
    base_linear_of,
    convert_dit_to_fp8_tensorwise,
)


@pytest.fixture
def tiny_model() -> Krea2Transformer2DModel:
    torch.manual_seed(0)
    return Krea2Transformer2DModel(
        in_channels=16,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        timestep_embed_dim=16,
        text_hidden_dim=24,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=48,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(4, 2, 2),
    )


def test_fp8_linear_fidelity_and_storage():
    torch.manual_seed(1)
    lin = nn.Linear(32, 16, bias=False)
    q = Fp8TensorwiseLinear(lin, grad_mode="bf16")
    assert q.weight.dtype == torch.float8_e4m3fn  # fp8 IS the storage
    assert not q.weight.requires_grad
    x = torch.randn(4, 32)
    ref, out = lin(x), q(x)
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 0.06, f"e4m3 tensorwise quant error too high: {rel:.4f}"


def test_fp8_linear_grad_flows_to_input():
    lin = nn.Linear(8, 8, bias=True)
    q = Fp8TensorwiseLinear(lin)
    x = torch.randn(2, 8, requires_grad=True)
    q(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_convert_scope_on_tiny_krea2(tiny_model):
    scope = {"leaf_names": QUANT_LEAF_NAMES, "skip_substrings": QUANT_SKIP_SUBSTRINGS}
    n = convert_dit_to_fp8_tensorwise(tiny_model, grad_mode="bf16", **scope)
    assert n == 16  # 2 blocks x (q, k, v, gate, out.0, ff.gate, ff.up, ff.down)
    for name, mod in tiny_model.named_modules():
        if isinstance(mod, Fp8TensorwiseLinear):
            assert "text_fusion" not in name and "final_layer" not in name
    blk = tiny_model.transformer_blocks[0]
    assert isinstance(blk.attn.to_q, Fp8TensorwiseLinear)
    assert base_linear_of(blk.attn.to_q) is not None
    # text fusion stays bf16-precision nn.Linear
    assert not isinstance(tiny_model.text_fusion.refiner_blocks[0].attn.to_q, Fp8TensorwiseLinear)


def test_peft_lora_composes_and_trains_on_fp8_base(tiny_model):
    import peft

    scope = {"leaf_names": QUANT_LEAF_NAMES, "skip_substrings": QUANT_SKIP_SUBSTRINGS}
    convert_dit_to_fp8_tensorwise(tiny_model, **scope)
    targets = [
        name for name, mod in tiny_model.named_modules()
        if isinstance(mod, Fp8TensorwiseLinear)
    ]
    cfg = peft.LoraConfig(r=4, lora_alpha=4, target_modules=targets)
    model = peft.get_peft_model(tiny_model, cfg)
    # Mirror configure_adapter's recast: trainables to the adapter dtype.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
            assert p.dtype == torch.float32

    blk = tiny_model.transformer_blocks[0]
    x = torch.randn(2, 6, 32)
    out = blk.attn.to_q(x)  # peft-wrapped fp8 linear: base + lora delta
    loss = out.float().sum()
    loss.backward()
    # Only block 0's to_q ran; lora_A's grad is zero while lora_B is zero-init, but
    # lora_B's grad must be real and nonzero (grad_B = A x^T . grad_out).
    grads = {
        n: p.grad
        for n, p in model.named_parameters()
        if p.requires_grad and "transformer_blocks.0.attn.to_q" in n
    }
    assert grads and all(g is not None for g in grads.values())
    assert any(g.abs().sum() > 0 for g in grads.values())
