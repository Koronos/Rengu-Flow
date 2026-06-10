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
