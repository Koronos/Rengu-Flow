"""Regression guards for user-reported incidents that had no dedicated test yet.

Each test pins one production failure mode:
  * preview memory controls hidden for Krea 2 (gated to cosmos-only in the UI),
  * preview OOM without preview block swap dying with a bare error (no actionable hint),
  * an explicit 16-bit adapter dtype silently pairing with a plain optimizer (the
    bf16-stall class) without the startup NOTE.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _preview_field_variants(path):
    from rengu_flow_ui.config_schema import get_sections

    return [
        f for s in get_sections() for f in s["fields"] if f["path"] == path
    ]


def _models_in_when(field):
    when = field.get("when") or {}
    conds = when.get("all", [when]) if isinstance(when, dict) else []
    for cond in conds:
        if isinstance(cond, dict) and cond.get("field") == "model.type":
            return cond.get("in") or []
    return None  # ungated


def test_preview_memory_controls_visible_for_krea2():
    """Regression: preview_blocks_to_swap / preview_offload_text_encoder were gated to
    cosmos_predict2 only, hiding exactly the knob that fixes Krea 2's preview OOM."""
    for path in ("preview.preview_blocks_to_swap", "preview.preview_offload_text_encoder"):
        variants = _preview_field_variants(path)
        assert variants, f"{path} missing from schema"
        models = _models_in_when(variants[0])
        assert models is not None and set(models) >= {"cosmos_predict2", "krea2"}, (
            f"{path} gated to {models}; both DiT preview runners honor it"
        )
    # krea2's preview never reads dit-for-decode — it must stay cosmos-only (no dead knob).
    dit_decode = _preview_field_variants("preview.preview_offload_dit_for_decode")
    assert _models_in_when(dit_decode[0]) == ["cosmos_predict2"]
    # save_png is model-agnostic (utils/preview.py): ungated.
    save_png = _preview_field_variants("preview.preview_save_png")
    assert _models_in_when(save_png[0]) is None


def test_preview_oom_prints_block_swap_hint(capsys):
    """Regression: a preview OOM with no preview block swap skipped the preview with a
    bare error; it must point at preview_blocks_to_swap (training memory unaffected)."""
    from rengu_flow.utils import preview as preview_mod

    oom = torch.cuda.OutOfMemoryError("CUDA out of memory (preview)")

    def exploding_runner(model, preview_cfg, prompts, sink, step):
        raise oom

    model = SimpleNamespace(name="krea2", restore_after_preview=lambda: None)
    config = {
        "pipeline_stages": 1,
        "preview": {"prompts": ["a portrait"], "enabled": True},
    }
    orig = preview_mod._run_cosmos_previews
    preview_mod._run_cosmos_previews = exploding_runner
    try:
        preview_mod.run_previews(model, config, sink=SimpleNamespace(), step=10)
    finally:
        preview_mod._run_cosmos_previews = orig
    out = capsys.readouterr().out
    assert "preview at step 10 failed" in out
    assert "preview_blocks_to_swap" in out  # the actionable hint
    # With the knob already set, no redundant hint.
    config["preview"]["preview_blocks_to_swap"] = 16
    preview_mod._run_cosmos_previews = exploding_runner
    try:
        preview_mod.run_previews(model, config, sink=SimpleNamespace(), step=11)
    finally:
        preview_mod._run_cosmos_previews = orig
    assert "preview_blocks_to_swap" not in capsys.readouterr().out.replace(
        "preview_blocks_to_swap: 16", ""
    ).split("step 11", 1)[-1]


def test_explicit_16bit_adapter_dtype_prints_note(capsys):
    """Regression guard for the bf16-adapter stall class: an explicit 16-bit
    adapter.dtype must print the rounding NOTE pointing at Kahan optimizers."""
    from rengu_flow.config import set_config_defaults

    cfg = {
        "dataset": "examples/minimal_krea2_dataset.toml",
        "model": {
            "type": "krea2",
            "dtype": "bfloat16",
            "transformer_path": "x",
            "vae_path": "v",
            "text_encoder_path": "t",
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "adapter": {"type": "lokr", "rank": 4, "dtype": "bfloat16"},
    }
    set_config_defaults(cfg)
    out = capsys.readouterr().out
    assert "adapter.dtype is 16-bit" in out
    assert "adamw8bitkahan" in out
    assert cfg["adapter"]["dtype"] == torch.bfloat16  # explicit value still honored

    # set_config_defaults is not idempotent (dtype strings become torch dtypes):
    # build the default-dtype config fresh.
    cfg2 = {
        "dataset": "examples/minimal_krea2_dataset.toml",
        "model": {
            "type": "krea2",
            "dtype": "bfloat16",
            "transformer_path": "x",
            "vae_path": "v",
            "text_encoder_path": "t",
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "adapter": {"type": "lokr", "rank": 4},
    }
    set_config_defaults(cfg2)
    assert "adapter.dtype is 16-bit" not in capsys.readouterr().out  # fp32 default: silent


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_reentrant_checkpoint_reaches_adapter_grad(use_reentrant):
    """Regression guard for the silent adapter stall on the quantized + block-swapped
    path: reentrant activation checkpointing (auto-enabled for blocks_to_swap +
    transformer_4bit) must still deliver gradient to trainable params inside a
    checkpointed window. It only inspects top-level tensor args for requires_grad, so
    the inter-layer tuple must be passed unpacked (SequentialPipe.forward) or the
    boundary tensors' requires_grad is hidden and the segment is severed from autograd."""
    from functools import partial
    from torch.utils.checkpoint import checkpoint

    from rengu_flow.engine.single_device import SequentialPipe

    class Layer(torch.nn.Module):
        """Tuple-in/tuple-out like krea2's TransformerLayer: frozen base + trainable adapter."""

        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(8, 8, bias=False)
            self.base.weight.requires_grad_(False)  # frozen base (like a 4bit-quantized DiT block)
            self.adapter = torch.nn.Linear(8, 8, bias=False)  # only trainable param ("LoKr")

        def forward(self, inputs):
            hidden, extra = inputs
            return (self.base(hidden) + self.adapter(hidden), extra)

    layer = Layer()
    pipe = SequentialPipe(
        [layer],
        loss_fn=lambda out, label: (out[0] - label).pow(2).sum(),
        activation_checkpoint_interval=1,
        checkpointable_layers=None,  # empty filter => everything is checkpointed
        activation_checkpoint_func=partial(checkpoint, use_reentrant=use_reentrant),
    )

    # InitialLayer forces requires_grad on the floating boundary tensor before the window.
    hidden = torch.randn(2, 8, requires_grad=True)
    extra = torch.zeros(1)
    out = pipe((hidden, extra))
    pipe.loss_fn(out, torch.zeros(2, 8)).backward()

    assert layer.adapter.weight.grad is not None, "checkpointed adapter got no gradient"
    assert layer.adapter.weight.grad.abs().sum() > 0
