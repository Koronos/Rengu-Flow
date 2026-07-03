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
