"""Tests for optimizer and scheduler resolution."""

import copy

import pytest
import torch

from rengu_flow.optim.resolver import (
    RexLR,
    apply_warmup,
    build_scheduler_runtime_values,
    register_scheduler,
    resolve_scheduler,
    scheduler_registry,
    substitute_runtime_tokens,
)
from rengu_flow.registry.optimizers import get_optimizer_class, optimizer_registry, register_optimizer


@pytest.mark.parametrize("name, expected_cls", [
    ("adamw", torch.optim.AdamW),
    ("sgd", torch.optim.SGD),
    ("adam", torch.optim.Adam),
    ("AdamW", torch.optim.AdamW),
    ("torch.optim.AdamW", torch.optim.AdamW),
], ids=["adamw", "sgd", "adam", "case_insensitive", "fully_qualified"])
def test_get_optimizer_class(name, expected_cls):
    assert get_optimizer_class(name) is expected_cls


def test_get_optimizer_class_unknown_raises():
    with pytest.raises(ValueError) as exc_info:
        get_optimizer_class("adamw_8bit")
    assert "Unknown" in str(exc_info.value) or "adamw" in str(exc_info.value).lower()


@pytest.mark.parametrize("name, expected_name", [
    ("genericoptim", "GenericOptim"),
    ("automagic", "Automagic"),
    ("GenericOptim", "GenericOptim"),
], ids=["genericoptim", "automagic", "case_insensitive"])
def test_get_vendor_optimizer_class(name, expected_name):
    pytest.importorskip("optimum")
    klass = get_optimizer_class(name)
    assert klass.__name__ == expected_name


def test_get_pytorch_optimizer_prodigy_if_installed():
    pytest.importorskip("pytorch_optimizer")
    klass = get_optimizer_class("Prodigy")
    import pytorch_optimizer

    assert klass is pytorch_optimizer.Prodigy


def test_get_prodigy_alias():
    pytest.importorskip("pytorch_optimizer")
    import pytorch_optimizer

    klass = get_optimizer_class("prodigy")
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


def test_register_optimizer_custom_resolved_by_get_optimizer_class():
    """Custom optimizer registered via register_optimizer is resolved by get_optimizer_class."""
    register_optimizer("custom_test_adam")(torch.optim.Adam)
    try:
        klass = get_optimizer_class("custom_test_adam")
        assert klass is torch.optim.Adam
        klass = get_optimizer_class("Custom_Test_Adam")
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

    # torch built-ins in the registry → Torch.ClassName
    assert optimizer_display_label("adamw") == "Torch.AdamW"
    assert optimizer_display_label("sgd") == "Torch.SGD"
    assert optimizer_display_label("adam") == "Torch.Adam"
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


# --- wsd scheduler (constant + decay tail) -------------------------------------

from rengu_flow.optim.resolver import parse_wsd_decay_steps, wsd_decay_onset_step  # noqa: E402


@pytest.mark.parametrize("decay, total, expected", [
    (0.1, 1000, 100),     # float -> fraction
    (0.2, 100, 20),
    (1.0, 100, 100),      # float 1.0 -> all decay
    (100, 1000, 100),     # int -> absolute steps
    (5, 3, 3),            # int clamped to total
    (True, 100, 10),      # bool -> default 0.1
    (None, 100, 10),      # None -> default 0.1
], ids=["frac10", "frac20", "frac100", "int100", "clamp", "bool", "none"])
def test_parse_wsd_decay_steps(decay, total, expected):
    assert parse_wsd_decay_steps(decay, total) == expected


def test_wsd_decay_onset_step():
    cfg = {"lr_scheduler_args": {"decay": 0.2}}
    assert wsd_decay_onset_step(cfg, 100) == 80  # stable = total - decay


def _collect_lrs(scheduler, optimizer, total_steps):
    """Step optimizer-then-scheduler (correct order) and record the LR used at each step."""
    p = optimizer.param_groups[0]["params"][0]
    p.grad = torch.zeros_like(p)
    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_wsd_flat_then_decays():
    base = 1.0
    total = 100
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base)
    cfg = {"lr_scheduler_args": {"decay": 0.2, "decay_type": "rex", "rex_d": 0.9, "lr_min": 0.0}}
    sched = resolve_scheduler("wsd", opt, cfg, total, steps_per_epoch=10)
    assert sched.wsd_decay_onset == 80
    lrs = _collect_lrs(sched, opt, total)
    # stable phase: flat at base through the decay onset
    assert all(abs(lr - base) < 1e-9 for lr in lrs[:80]), lrs[:5]
    # decay phase: starts at base (smooth handoff) and is non-increasing
    decay = lrs[80:]
    assert abs(decay[0] - base) < 1e-9
    assert all(decay[i + 1] <= decay[i] + 1e-9 for i in range(len(decay) - 1))
    # the tail reaches lr_min by the end (read after the final step)
    assert opt.param_groups[0]["lr"] < 1e-6


def test_wsd_all_decay_when_fraction_one():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    cfg = {"lr_scheduler_args": {"decay": 1.0}}
    sched = resolve_scheduler("wsd", opt, cfg, 50, steps_per_epoch=10)
    assert sched.wsd_decay_onset == 0  # no stable phase


def test_wsd_warmup_keeps_decay_at_run_end():
    # warmup-aware: decay onset stays at total - decay_steps regardless of warmup.
    cfg = {"warmup_steps": 20, "lr_scheduler_args": {"decay": 0.1}}
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = resolve_scheduler("wsd", opt, cfg, 200, steps_per_epoch=10)
    # internal stable = effective(180) - decay(20) = 160; global onset helper = 200 - 20 = 180.
    assert sched.wsd_decay_onset == 160
    assert wsd_decay_onset_step(cfg, 200) == 180


def test_wsd_with_warmup_flattens_and_runs():
    """WSD + warmup must NOT nest SequentialLR (torch can't step nested SequentialLR). apply_warmup
    flattens it to warmup -> stable -> decay; the run steps to the end and has the right shape."""
    import warnings

    from rengu_flow.optim.resolver import apply_warmup

    base, total, warmup = 1.0, 200, 20
    cfg = {"warmup_steps": warmup, "lr_scheduler": "wsd",
           "lr_scheduler_args": {"decay": 0.1, "rex_d": 0.9, "lr_min": 0.0}}
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base)
    opt.param_groups[0]["params"][0].grad = torch.zeros(1)
    sched = apply_warmup(opt, resolve_scheduler("wsd", opt, cfg, total, 10), warmup)

    # Flattened: a single SequentialLR of 3 phases, no nested SequentialLR.
    assert not any(
        isinstance(s, torch.optim.lr_scheduler.SequentialLR) for s in sched._schedulers
    )
    assert list(sched._milestones) == [warmup, 180]  # warmup end, decay onset (total - decay)
    assert sched.wsd_decay_onset == 180

    lrs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(total):  # must not raise NotImplementedError
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
    assert lrs[0] < base and abs(lrs[warmup] - base) < 1e-9          # warmup ramps to base
    assert all(abs(lr - base) < 1e-9 for lr in lrs[warmup:180])       # flat stable phase
    assert lrs[180:] == sorted(lrs[180:], reverse=True)              # decay is non-increasing
    assert opt.param_groups[0]["lr"] < 1e-6                           # tail reaches ~lr_min


def test_wsd_extend_reanchors_decay():
    """Mirrors main.py's extend path: rebuild for a longer horizon + fast-forward to the fork
    step → LR stays flat past the original end and only decays at the new end."""
    base = 1.0
    cfg = {"lr_scheduler_args": {"decay": 0.2, "rex_d": 0.9, "lr_min": 0.0}}
    # Original run was total=100 (stable 80, fork at step 80). Extend to total=200.
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base)
    p = opt.param_groups[0]["params"][0]
    p.grad = torch.zeros_like(p)
    sched = resolve_scheduler("wsd", opt, cfg, 200, steps_per_epoch=10)
    assert sched.wsd_decay_onset == 160  # new decay onset, not the old 80

    fork_step = 80
    for _ in range(fork_step):          # fast-forward to the resumed (fork) step
        opt.step()
        sched.step()
    # at the resumed fork step the LR is still base — extending did NOT re-drop it
    assert abs(opt.param_groups[0]["lr"] - base) < 1e-9

    lrs = []
    for _ in range(200 - fork_step):    # continue to the new end
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    # global steps 80..159 (first 80 collected) stay flat — well past the original 100
    assert all(abs(x - base) < 1e-9 for x in lrs[:80]), lrs[:5]
    # past the new onset (160) it decays, reaching lr_min by 200
    assert lrs[81] < base
    assert opt.param_groups[0]["lr"] < 1e-6


def test_wsd_unknown_decay_type_raises():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError, match="decay_type"):
        resolve_scheduler("wsd", opt, {"lr_scheduler_args": {"decay_type": "bogus"}}, 100, 10)
