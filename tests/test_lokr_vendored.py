"""RF-05: vendored SDXL LoKr must inject params onto each nn.Linear (in the module tree, so
DeepSpeed places and trains them)."""

import torch
from torch import nn

from rengu_flow.networks import lokr_vendored


def test_vendored_lokr_injects_into_modulelist_linears():
    root = nn.Module()
    # ModuleList => numeric child names ("0", "1") — the case that used to crash.
    root.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
    cfg = {"rank": 4, "alpha": 4, "factor": -1, "dtype": torch.float32}

    lokr_vendored._apply_lokr_vendored(root, [root.blocks], cfg, "unet.")

    # LoKr params are registered ON the linears (children of the module tree), not on a
    # separate side object — this is what lets DeepSpeed move/train them.
    param_names = [n for n, _ in root.named_parameters()]
    assert any("lokr_" in n for n in param_names), param_names
    assert any(n.startswith("blocks.0.lokr_") for n in param_names), param_names

    # Base weights are frozen; LoKr params train.
    assert root.blocks[0].weight.requires_grad is False
    assert any(p.requires_grad for n, p in root.named_parameters() if "lokr_" in n)

    # Forward still runs (delta applied) and shape is preserved.
    y = root.blocks[0](torch.randn(2, 8))
    assert y.shape == (2, 8)


def test_vendored_lokr_skips_double_injection():
    root = nn.Module()
    root.blocks = nn.ModuleList([nn.Linear(8, 8)])
    cfg = {"rank": 4, "alpha": 4, "factor": -1, "dtype": torch.float32}
    lokr_vendored._apply_lokr_vendored(root, [root.blocks], cfg, "unet.")
    before = sum(1 for n, _ in root.named_parameters() if "lokr_" in n)
    lokr_vendored._apply_lokr_vendored(root, [root.blocks], cfg, "unet.")
    after = sum(1 for n, _ in root.named_parameters() if "lokr_" in n)
    assert before == after  # guard against re-injecting on an already-adapted linear


def test_lokr_plain_forward_casts_float32_input_to_bf16_weight():
    """Regression: qwen_image21 LoRA/LoKr smoke found this on real bf16 weights (2026-09-22).

    A bf16 base ``nn.Linear`` fed a float32 activation (e.g. Qwen-Image 2.1's img_in, whose
    latents are cached in float32) used to silently type-promote the output to float32 instead
    of erroring — torch >= 2.x matmul does dtype promotion rather than raising. PEFT's LoRA
    casts the input to the base layer's dtype before calling it; the vendored LoKr forward
    (``_lokr_forward_plain``) did not, so a downstream bf16-typed op (e.g. an index_put building
    the joint hidden-state sequence) crashed with "Index put requires the source and destination
    dtypes match, got BFloat16 for the destination and Float for the source."
    """
    linear = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    cfg = {"rank": 4, "alpha": 4, "factor": -1, "dtype": torch.float32}
    lokr_vendored._apply_lokr_vendored(nn.Sequential(linear), [nn.Sequential(linear)], cfg, "")

    x = torch.randn(2, 8, dtype=torch.float32)
    y = linear(x)
    assert y.dtype == torch.bfloat16
