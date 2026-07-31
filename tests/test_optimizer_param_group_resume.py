"""Regression tests for applying edited optimizer groups on checkpoint resume."""

from __future__ import annotations

import pytest

from rengu_flow.optim.param_groups import (
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

    loaded_groups = [
        {
            "params": [param],
            "lr": 1e-6,
            "betas": (0.2, 0.99),
            "weight_decay": 0.0,
            "checkpoint_only": True,
        }
    ]
    wrapper_groups = loaded_groups
    inner_groups = loaded_groups

    reapply_param_group_options(wrapper_groups, configured)

    assert wrapper_groups is inner_groups
    assert wrapper_groups[0] is inner_groups[0]
    assert wrapper_groups[0]["params"] == [param]
    assert wrapper_groups[0]["lr"] == 3e-6
    assert wrapper_groups[0]["betas"] == (0.5, 0.999)
    assert wrapper_groups[0]["weight_decay"] == 0.1
    assert "checkpoint_only" not in wrapper_groups[0]


def test_reapply_rejects_changed_group_structure() -> None:
    with pytest.raises(ValueError, match="full optimizer reset"):
        reapply_param_group_options(
            [{"params": [object()], "lr": 1e-6}],
            [{"lr": 1e-6}, {"lr": 2e-6}],
        )
