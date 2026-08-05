"""Regression tests for applying edited optimizer groups on checkpoint resume.

The current config's optimizer hyperparameters (LR, betas, weight_decay, ...) take effect on
resume while the checkpoint's optimizer *state* (moments) is preserved. This is done by
reapplying the configured param-group options in place, which must not break wrapped optimizers
(e.g. Nekaon) that share the exact param_groups list with an inner optimizer.
"""

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
    # Params untouched; configured options win; stray checkpoint key dropped.
    assert wrapper_groups[0]["params"] == [param]
    assert wrapper_groups[0]["lr"] == 3e-6
    assert wrapper_groups[0]["betas"] == (0.5, 0.999)
    assert wrapper_groups[0]["weight_decay"] == 0.1
    assert "checkpoint_only" not in wrapper_groups[0]


def test_reapply_rejects_changed_group_structure() -> None:
    with pytest.raises(ValueError, match="reset_optimizer"):
        reapply_param_group_options(
            [{"params": [object()], "lr": 1e-6}],
            [{"lr": 1e-6}, {"lr": 2e-6}],
        )
