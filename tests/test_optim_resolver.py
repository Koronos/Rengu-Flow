"""Tests for optimizer and scheduler resolution."""

import copy

import pytest
import torch

from renga_flow.optim.resolver import (
    apply_warmup,
    build_scheduler_runtime_values,
    register_scheduler,
    resolve_optimizer_class,
    resolve_scheduler,
    scheduler_registry,
    substitute_runtime_tokens,
)
from renga_flow.registry.optimizers import optimizer_registry, register_optimizer


@pytest.mark.parametrize("name, expected_cls", [
    ("adamw", torch.optim.AdamW),
    ("sgd", torch.optim.SGD),
    ("adam", torch.optim.Adam),
    ("AdamW", torch.optim.AdamW),
    ("torch.optim.AdamW", torch.optim.AdamW),
], ids=["adamw", "sgd", "adam", "case_insensitive", "fully_qualified"])
def test_resolve_optimizer_class(name, expected_cls):
    assert resolve_optimizer_class(name) is expected_cls


def test_resolve_optimizer_class_unknown_raises():
    with pytest.raises(ValueError) as exc_info:
        resolve_optimizer_class("adamw_8bit")
    assert "Unknown" in str(exc_info.value) or "adamw" in str(exc_info.value).lower()


@pytest.mark.parametrize("name, expected_name", [
    ("genericoptim", "GenericOptim"),
    ("automagic", "Automagic"),
    ("GenericOptim", "GenericOptim"),
], ids=["genericoptim", "automagic", "case_insensitive"])
def test_resolve_vendor_optimizers(name, expected_name):
    pytest.importorskip("optimum")
    klass = resolve_optimizer_class(name)
    assert klass.__name__ == expected_name


def test_resolve_pytorch_optimizer_prodigy_if_installed():
    pytest.importorskip("pytorch_optimizer")
    klass = resolve_optimizer_class("Prodigy")
    import pytorch_optimizer

    assert klass is pytorch_optimizer.Prodigy


def test_resolve_prodigy_alias():
    pytest.importorskip("pytorch_optimizer")
    import pytorch_optimizer

    klass = resolve_optimizer_class("prodigy")
    assert klass is pytorch_optimizer.Prodigy


def test_substitute_runtime_tokens_replaces_matching():
    kwargs = {"total_iters": "total_steps", "other": "unchanged"}
    runtime = {"total_steps": 100}
    result = substitute_runtime_tokens(copy.deepcopy(kwargs), runtime)
    assert result["total_iters"] == 100
    assert result["other"] == "unchanged"


def test_substitute_runtime_tokens_leaves_non_strings():
    kwargs = {"factor": 0.5}
    result = substitute_runtime_tokens(copy.deepcopy(kwargs), {"factor": 1.0})
    assert result["factor"] == 0.5


def test_build_scheduler_runtime_values_without_max_steps():
    values = build_scheduler_runtime_values(
        {"epochs": 3}, total_steps=300, steps_per_epoch=100
    )
    assert values["effective_total_steps"] == 300
    assert "max_steps" not in values


@pytest.mark.parametrize("scheduler_name, expected_type", [
    ("constant", torch.optim.lr_scheduler.ConstantLR),
    ("linear", torch.optim.lr_scheduler.LinearLR),
    ("none", type(None)),
], ids=["constant", "linear", "none"])
def test_resolve_scheduler(scheduler_name, expected_type):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    config = {"epochs": 1}
    sched = resolve_scheduler(scheduler_name, optimizer, config, total_steps=10, steps_per_epoch=10)
    if expected_type is type(None):
        assert sched is None
    else:
        assert sched is not None
        assert isinstance(sched, expected_type)


def test_resolve_scheduler_cosine_uses_lr_scheduler_args():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    config = {"epochs": 1, "lr_scheduler_args": {"lr_min": 0.01}}
    sched = resolve_scheduler("cosine", optimizer, config, total_steps=10, steps_per_epoch=10)
    assert sched is not None
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.eta_min == 0.01


def test_resolve_scheduler_cosine_default_lr_min():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    config = {"epochs": 1}
    sched = resolve_scheduler("cosine", optimizer, config, total_steps=10, steps_per_epoch=10)
    assert sched is not None
    assert sched.eta_min == 0.0


def test_resolve_scheduler_unknown_raises():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    config = {"epochs": 1}
    with pytest.raises(ValueError) as exc_info:
        resolve_scheduler("unknown", optimizer, config, total_steps=10, steps_per_epoch=10)
    assert "Unknown" in str(exc_info.value) or "cosine" in str(exc_info.value).lower()


@pytest.mark.parametrize("sched_none, warmup_steps", [(True, 5), (False, 0)], ids=["scheduler_none", "warmup_zero"])
def test_apply_warmup_returns_same_when_no_effective_warmup(sched_none, warmup_steps):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    sched = None
    if not sched_none:
        config = {"epochs": 1}
        sched = resolve_scheduler("constant", optimizer, config, total_steps=10, steps_per_epoch=10)
    result = apply_warmup(optimizer, sched, warmup_steps)
    assert result is sched


def test_apply_warmup_returns_sequential():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    config = {"epochs": 1}
    sched = resolve_scheduler("constant", optimizer, config, total_steps=10, steps_per_epoch=10)
    result = apply_warmup(optimizer, sched, 3)
    assert result is not None
    assert isinstance(result, torch.optim.lr_scheduler.SequentialLR)


def test_register_optimizer_custom_resolved_by_resolve_optimizer_class():
    """Custom optimizer registered via register_optimizer is resolved by resolve_optimizer_class."""
    register_optimizer("custom_test_adam")(torch.optim.Adam)
    try:
        klass = resolve_optimizer_class("custom_test_adam")
        assert klass is torch.optim.Adam
        klass = resolve_optimizer_class("Custom_Test_Adam")
        assert klass is torch.optim.Adam
    finally:
        del optimizer_registry["custom_test_adam"]


def test_register_scheduler_custom_resolved_by_resolve_scheduler():
    """Custom scheduler registered via register_scheduler is used by resolve_scheduler."""
    def _custom_constant(optimizer, config, total_steps, steps_per_epoch):
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5)

    register_scheduler("custom_test_constant")(_custom_constant)
    try:
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        config = {"epochs": 1}
        sched = resolve_scheduler("custom_test_constant", opt, config, total_steps=10, steps_per_epoch=10)
        assert sched is not None
        assert isinstance(sched, torch.optim.lr_scheduler.ConstantLR)
        assert sched.factor == 0.5
    finally:
        del scheduler_registry["custom_test_constant"]
