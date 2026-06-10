"""Tests for compiler-driven activation checkpointing (activation_budget.py)."""

from __future__ import annotations

import pytest

from rengu_flow.training.activation_budget import (
    DEFAULT_BUDGET,
    apply_activation_memory_budget,
    resolve_auto_ac_budget,
)


def test_requires_compile():
    with pytest.raises(ValueError, match="compile = true"):
        resolve_auto_ac_budget({"activation_checkpointing": "auto"})


def test_default_budget():
    cfg = {"activation_checkpointing": "auto", "compile": True}
    assert resolve_auto_ac_budget(cfg) == DEFAULT_BUDGET == 0.3


def test_explicit_budget_passthrough():
    cfg = {"compile": True, "activation_memory_budget": 0.25}
    assert resolve_auto_ac_budget(cfg) == 0.25
    cfg["activation_memory_budget"] = 0  # int 0 is a valid edge
    assert resolve_auto_ac_budget(cfg) == 0.0
    cfg["activation_memory_budget"] = 1
    assert resolve_auto_ac_budget(cfg) == 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.5, "high", None])
def test_rejects_out_of_range_or_non_numeric(bad):
    cfg = {"compile": True, "activation_memory_budget": bad}
    with pytest.raises(ValueError, match="activation_memory_budget"):
        resolve_auto_ac_budget(cfg)


def test_apply_sets_functorch_config():
    import torch._functorch.config as fc

    before = fc.activation_memory_budget
    try:
        apply_activation_memory_budget(0.3)
        assert fc.activation_memory_budget == 0.3
    finally:
        fc.activation_memory_budget = before


def test_scale_budget_for_area():
    from rengu_flow.training.activation_budget import scale_budget_for_area

    # Largest bucket keeps the base budget.
    assert scale_budget_for_area(0.1, 128 * 128, 128 * 128) == pytest.approx(0.1)
    # Quarter-area shape (512 vs 1024) scales 4x.
    assert scale_budget_for_area(0.1, 64 * 64, 128 * 128) == pytest.approx(0.4)
    # Tiny shapes cap at 1.0 (no recompute).
    assert scale_budget_for_area(0.1, 16 * 16, 128 * 128) == 1.0
    # Degenerate inputs fall back to the base.
    assert scale_budget_for_area(0.3, 0, 128 * 128) == 0.3
    assert scale_budget_for_area(0.3, 64 * 64, 0) == 0.3


def test_loader_applies_per_shape_budget(capsys):
    import torch
    import torch._functorch.config as fc

    from rengu_flow.data.loader import PipelineDataLoader

    loader = object.__new__(PipelineDataLoader)
    loader.announce_new_shapes = True
    loader._seen_latent_shapes = set()
    loader.auto_budget_base = 0.1
    loader.auto_budget_max_latent_area = 1 * 128 * 128  # T*H*W of the largest bucket

    def batch(h, w):
        return ((torch.zeros(1, 16, 1, h, w), torch.zeros(1)), (torch.zeros(1), None))

    before = fc.activation_memory_budget
    try:
        loader._maybe_announce_shape(batch(128, 128))  # largest -> base
        assert fc.activation_memory_budget == pytest.approx(0.1)
        loader._maybe_announce_shape(batch(64, 64))  # quarter area -> 0.4
        assert fc.activation_memory_budget == pytest.approx(0.4)
        out = capsys.readouterr().out
        assert "activation budget 0.10" in out and "activation budget 0.40" in out

        # Budget application is independent of the announce flag (non-main ranks).
        loader2 = object.__new__(PipelineDataLoader)
        loader2.announce_new_shapes = False
        loader2._seen_latent_shapes = set()
        loader2.auto_budget_base = 0.1
        loader2.auto_budget_max_latent_area = 1 * 128 * 128
        loader2._maybe_announce_shape(batch(32, 32))
        assert fc.activation_memory_budget == 1.0  # capped
        assert capsys.readouterr().out == ""
    finally:
        fc.activation_memory_budget = before
