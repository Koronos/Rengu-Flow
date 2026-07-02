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
