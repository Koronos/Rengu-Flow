"""Tests for compiler-driven activation checkpointing (activation_budget.py)."""

from __future__ import annotations

import pytest

from rengu_flow.training.activation_budget import (
    DEFAULT_BUDGET,
    BudgetBackoff,
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


# ----------------------------------------------------------------- backoff
def test_budget_backoff_lowers_geometrically():
    b = BudgetBackoff(0.5, factor=0.66, max_retries=4)
    assert b.on_oom() == pytest.approx(0.33)
    assert b.on_oom() == pytest.approx(0.218)
    assert b.on_oom() == pytest.approx(0.144)
    assert b.on_oom() == pytest.approx(0.095)
    assert b.on_oom() is None  # retries exhausted
    assert "0.5 -> 0.095" in b.describe()


def test_budget_backoff_floors_at_zero():
    b = BudgetBackoff(0.06, max_retries=4)
    # 0.06 * 0.66 = 0.0396 < 0.05 -> jump to the true floor (full checkpointing)
    assert b.on_oom() == 0.0
    # At the floor there is nothing left to give back.
    assert b.on_oom() is None


def test_budget_backoff_exhausts():
    b = BudgetBackoff(0.9, max_retries=1)
    assert b.on_oom() is not None
    assert b.on_oom() is None


def test_nominal_micro_batch():
    from rengu_flow.training.activation_budget import nominal_micro_batch

    assert nominal_micro_batch(2) == 2
    assert nominal_micro_batch({512: 2, 1024: 1}) == 2  # mean 1.5 rounds to 2
    assert nominal_micro_batch({512: 4, 768: 2, 1024: 1}) == 2  # mean 2.33 -> 2
    assert nominal_micro_batch({}) == 1
    # Unlike first-dict-value, key order must not matter.
    assert nominal_micro_batch({1024: 1, 512: 4, 768: 2}) == nominal_micro_batch(
        {512: 4, 768: 2, 1024: 1}
    )


def test_dataset_avg_examples_per_step():
    from rengu_flow.data.dataset import Dataset

    class _Bucket:
        def __init__(self, images, steps):
            self.iteration_order = list(range(images))
            self._steps = steps

        def __len__(self):
            return self._steps

    ds = Dataset.__new__(Dataset)
    ds.post_init_called = True
    # 512-bucket: 96 images at global batch 2 -> 48 steps; 1024: 40 at 1 -> 40 steps.
    ds.buckets = [_Bucket(96, 48), _Bucket(40, 40)]
    assert ds.avg_examples_per_step() == (96 + 40) / (48 + 40)


def test_loader_announces_new_shapes_once(capsys):
    import torch

    from rengu_flow.data.loader import PipelineDataLoader

    loader = object.__new__(PipelineDataLoader)
    loader.announce_new_shapes = True
    loader._seen_latent_shapes = set()

    def batch(h, w):
        return ((torch.zeros(1, 16, 1, h, w), torch.zeros(1)), (torch.zeros(1), None))

    loader._maybe_announce_shape(batch(128, 128))
    loader._maybe_announce_shape(batch(128, 128))  # repeat -> silent
    loader._maybe_announce_shape(batch(64, 64))
    out = capsys.readouterr().out
    assert out.count("new latent shape") == 2

    # Off (the dynamic-compile mode): never prints.
    loader2 = object.__new__(PipelineDataLoader)
    loader2.announce_new_shapes = False
    loader2._seen_latent_shapes = set()
    loader2._maybe_announce_shape(batch(32, 32))
    assert capsys.readouterr().out == ""
