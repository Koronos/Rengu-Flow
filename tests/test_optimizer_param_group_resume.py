"""Regression tests for applying edited optimizer groups on checkpoint resume.

The current config's optimizer hyperparameters (LR, betas, weight_decay, ...) take effect on
resume while the checkpoint's optimizer *state* (moments) is preserved. This is done by
reapplying the configured param-group options in place, which must not break wrapped optimizers
(e.g. Nekaon) that share the exact param_groups list with an inner optimizer.
"""

from __future__ import annotations

import pytest

from rengu_flow.optim.param_groups import (
    configured_group_keys,
    reapply_param_group_options,
    snapshot_param_group_options,
)


def test_reapply_updates_loaded_groups_in_place_for_wrapped_optimizer() -> None:
    param = object()
    configured_groups = [
        {"params": [param], "lr": 3e-6, "betas": (0.5, 0.999), "weight_decay": 0.1}
    ]
    configured = snapshot_param_group_options(configured_groups)
    assert "params" not in configured[0]

    # Simulate the checkpoint-loaded state: old LR/betas plus a stray key from the checkpoint.
    loaded_groups = [
        {
            "params": [param],
            "lr": 1e-6,
            "betas": (0.2, 0.99),
            "weight_decay": 0.0,
            "checkpoint_only": True,
        }
    ]
    # A wrapped optimizer shares the identical list object with its inner optimizer.
    wrapper_groups = loaded_groups
    inner_groups = loaded_groups

    reapply_param_group_options(wrapper_groups, configured)

    # List and dict identity preserved (in-place update, not reassignment).
    assert wrapper_groups is inner_groups
    assert wrapper_groups[0] is inner_groups[0]
    # Params untouched; configured options win; checkpoint-only keys are never deleted.
    assert wrapper_groups[0]["params"] == [param]
    assert wrapper_groups[0]["lr"] == 3e-6
    assert wrapper_groups[0]["betas"] == (0.5, 0.999)
    assert wrapper_groups[0]["weight_decay"] == 0.1
    assert wrapper_groups[0]["checkpoint_only"] is True


def test_reapply_rejects_changed_group_structure() -> None:
    with pytest.raises(ValueError, match="reset_optimizer"):
        reapply_param_group_options(
            [{"params": [object()], "lr": 1e-6}],
            [{"lr": 1e-6}, {"lr": 2e-6}],
        )


@pytest.mark.parametrize(
    ("name", "state_keys"),
    [
        ("prodigy", ("d", "step")),
        ("kprodigy", ("d", "k")),
        ("schedulefree", ("step", "weight_sum", "lr_max")),
    ],
)
def test_reapply_preserves_optimizer_owned_state_keys(name, state_keys) -> None:
    import torch

    from rengu_flow.registry.optimizers import get_optimizer_class

    try:
        cls = get_optimizer_class(name)
    except Exception:
        pytest.skip(f"{name} unavailable")
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = cls([{"params": [p], "lr": 1.0}])
    for _ in range(5):
        opt.zero_grad()
        (p**2).sum().backward()
        opt.step()
    trained = {k: opt.param_groups[0][k] for k in state_keys}
    sd = opt.state_dict()

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = cls([{"params": [p2], "lr": 2.0}])
    snap = snapshot_param_group_options(opt2.param_groups)
    opt2.load_state_dict(sd)
    reapply_param_group_options(opt2.param_groups, snap)

    assert {k: opt2.param_groups[0][k] for k in state_keys} == trained
    assert opt2.param_groups[0]["lr"] == 2.0  # configured hyperparameter still wins


def test_reapply_keeps_arbitrary_state_key_and_applies_edited_toml_values() -> None:
    import torch

    class FakeOpt(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.99)):
            super().__init__(
                params,
                dict(lr=lr, weight_decay=weight_decay, betas=betas, foo_state=0.0),
            )

        def step(self, closure=None):
            for g in self.param_groups:
                g["foo_state"] += 1.0

    p = torch.nn.Parameter(torch.randn(3))
    opt = FakeOpt([p], lr=1e-3, weight_decay=0.01)
    for _ in range(3):
        opt.step()
    sd = opt.state_dict()
    assert sd["param_groups"][0]["foo_state"] == 3.0

    # The TOML was edited between runs: new lr / weight_decay.
    opt2 = FakeOpt([torch.nn.Parameter(p.detach().clone())], lr=5e-4, weight_decay=0.1)
    snap = snapshot_param_group_options(opt2.param_groups)
    opt2.load_state_dict(sd)
    assert opt2.param_groups[0]["lr"] == 1e-3  # checkpoint value until reapplied
    keys = configured_group_keys({"type": "fake", "lr": 5e-4, "weight_decay": 0.1})
    reapply_param_group_options(opt2.param_groups, snap, configured_keys=keys)

    g = opt2.param_groups[0]
    assert g["foo_state"] == 3.0  # unknown state key survives
    assert g["lr"] == 5e-4 and g["weight_decay"] == 0.1  # edited TOML applies


def test_configured_group_keys_translates_beta2_half_life() -> None:
    keys = configured_group_keys({"type": "adamw", "beta2_half_life": 100, "lr": 1e-4})
    assert "betas" in keys and "beta2_half_life" not in keys and "type" not in keys
