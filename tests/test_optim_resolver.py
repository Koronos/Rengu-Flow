"""Tests for optimizer and scheduler resolution."""

import copy

import pytest
import torch

from rengu_flow.optim.resolver import (
    RexLR,
    apply_warmup,
    build_scheduler_runtime_values,
    register_scheduler,
    resolve_optimizer_class,
    resolve_scheduler,
    scheduler_registry,
    substitute_runtime_tokens,
)
from rengu_flow.registry.optimizers import optimizer_registry, register_optimizer


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


def test_resolve_scheduler_rex_returns_rexlr():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    config = {"epochs": 1}
    sched = resolve_scheduler("rex", optimizer, config, total_steps=10, steps_per_epoch=10)
    assert isinstance(sched, RexLR)
    assert sched.total_steps == 10
    assert sched.lr_min == 0.0


def test_resolve_scheduler_rex_uses_lr_scheduler_args():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    config = {"epochs": 1, "lr_scheduler_args": {"lr_min": 0.0001}}
    sched = resolve_scheduler("rex", optimizer, config, total_steps=10, steps_per_epoch=10)
    assert isinstance(sched, RexLR)
    assert sched.lr_min == 0.0001


def test_rex_profile_endpoints_and_monotonic():
    base_lr, lr_min, total = 1e-3, 0.0, 10
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    config = {"epochs": 1, "lr_scheduler_args": {"lr_min": lr_min}}
    sched = resolve_scheduler("rex", optimizer, config, total_steps=total, steps_per_epoch=total)
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(total):
        optimizer.step()
        sched.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    # Starts at base_lr, ends at lr_min, monotonically non-increasing.
    assert lrs[0] == pytest.approx(base_lr)
    assert lrs[-1] == pytest.approx(lr_min, abs=1e-9)
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))
    # Midpoint multiplier matches d/(0.5+0.5d) with d=0.5 -> 2/3.
    assert lrs[total // 2] == pytest.approx(base_lr * (0.5 / (0.5 + 0.5 * 0.5)), rel=1e-6)


def test_rex_d_default_is_canonical_rex():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = resolve_scheduler("rex", optimizer, {"epochs": 1}, total_steps=10, steps_per_epoch=10)
    assert isinstance(sched, RexLR)
    assert sched.rex_d == 0.5


def test_rex_d_zero_equals_linear_decay():
    base_lr, total = 1e-3, 10
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    config = {"epochs": 1, "lr_scheduler_args": {"lr_min": 0.0, "rex_d": 0.0}}
    sched = resolve_scheduler("rex", optimizer, config, total_steps=total, steps_per_epoch=total)
    lrs = []
    for step in range(total + 1):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        sched.step()
    # d=0 -> factor = z = 1 - step/total (pure linear ramp to 0).
    for step, lr in enumerate(lrs):
        assert lr == pytest.approx(base_lr * (1.0 - step / total), abs=1e-9)


def test_rex_d_clamped_to_unit_interval():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    config = {"epochs": 1, "lr_scheduler_args": {"rex_d": 5.0}}
    sched = resolve_scheduler("rex", optimizer, config, total_steps=10, steps_per_epoch=10)
    assert sched.rex_d == 1.0


def test_rex_respects_lr_min_floor():
    base_lr, lr_min, total = 1e-3, 2e-4, 5
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    config = {"epochs": 1, "lr_scheduler_args": {"lr_min": lr_min}}
    sched = resolve_scheduler("rex", optimizer, config, total_steps=total, steps_per_epoch=total)
    for _ in range(total + 3):  # step past the end
        optimizer.step()
        sched.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_min, abs=1e-9)


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


def test_optimizer_display_labels_vendor_prefix():
    """Library/vendored optimizers render as Vendor.Name; rengu's own registry entries
    (torch-backed) carry no prefix."""
    from rengu_flow.registry.optimizers import optimizer_display_label, optimizer_options

    # rengu's own registry → no prefix
    assert optimizer_display_label("adamw") == "adamw"
    assert optimizer_display_label("sgd") == "sgd"
    # external libraries → Vendor.ClassName
    assert optimizer_display_label("adamw8bit") == "Bitsandbytes.AdamW8bit"
    assert optimizer_display_label("prodigy") == "PytorchOptimizer.Prodigy"
    assert optimizer_display_label("adakaon") == "Kaon.Adakaon"
    assert optimizer_display_label("stableadamw") == "Optimi.StableAdamW"
    assert optimizer_display_label("offload") == "Torchao.CPUOffloadOptimizer"
    # vendored from diffusion-pipe → DiffusionPipe.ClassName
    assert optimizer_display_label("genericoptim") == "DiffusionPipe.GenericOptim"
    assert optimizer_display_label("automagic") == "DiffusionPipe.Automagic"
    # unknown / custom name passes through unchanged
    assert optimizer_display_label("my.custom.Path") == "my.custom.Path"
    # options() returns (value, label) pairs and every value maps to its label
    pairs = optimizer_options()
    assert ("adakaon", "Kaon.Adakaon") in pairs
    assert all(label == optimizer_display_label(value) for value, label in pairs)
