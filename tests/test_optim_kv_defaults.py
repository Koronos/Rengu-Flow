"""Tests for optimizer/scheduler KV default maps."""

from rengu_flow_ui.optim_kv_defaults import (
    OPTIMIZER_REGISTRY_KV_DEFAULTS,
    SCHEDULER_RUNTIME_TOKEN_HINTS,
    SCHEDULER_RUNTIME_TOKENS,
    optimizer_extra_params_defaults,
    scheduler_kv_defaults,
)
from rengu_flow_ui.optimizer_form import defaults_for_optimizer_type_change
from rengu_flow_ui.scheduler_form import defaults_for_scheduler_type_change


def test_scheduler_runtime_token_hints_cover_all_tokens() -> None:
    assert set(SCHEDULER_RUNTIME_TOKEN_HINTS) == set(SCHEDULER_RUNTIME_TOKENS)
    assert "min(total_steps" in SCHEDULER_RUNTIME_TOKEN_HINTS["effective_total_steps"]


def test_optimizer_adamw_defaults_include_lr_and_betas() -> None:
    kv = optimizer_extra_params_defaults("adamw")
    assert kv["lr"] == 1e-4
    assert kv["betas"] == [0.9, 0.999]


def test_optimizer_genericoptim_defaults() -> None:
    kv = optimizer_extra_params_defaults("genericoptim")
    assert kv["muon"] is False
    assert "adamuon" in kv
    assert kv["lr"] == 1e-4


def test_scheduler_cosine_fqn_defaults_use_tokens() -> None:
    kv = scheduler_kv_defaults("torch.optim.lr_scheduler.CosineAnnealingLR")
    assert kv["T_max"] == "effective_total_steps"
    assert kv["eta_min"] == 0.0
    assert "warmup_steps" not in kv


def test_scheduler_builtin_cosine_defaults() -> None:
    kv = scheduler_kv_defaults("cosine")
    assert kv["lr_min"] == 0.0
    assert "warmup_steps" not in kv


def test_scheduler_none_has_no_warmup() -> None:
    assert scheduler_kv_defaults("none") == {}


def test_type_change_includes_kv_defaults() -> None:
    out = defaults_for_optimizer_type_change("genericoptim")
    assert out["optimizer.extra_params"] == OPTIMIZER_REGISTRY_KV_DEFAULTS["genericoptim"]

    sched = defaults_for_scheduler_type_change("torch.optim.lr_scheduler.StepLR")
    assert sched["lr_scheduler_args.extra_params"] == scheduler_kv_defaults(
        "torch.optim.lr_scheduler.StepLR"
    )

    linear = defaults_for_scheduler_type_change("linear")
    assert linear["lr_scheduler_args.extra_params"] == scheduler_kv_defaults("linear")
