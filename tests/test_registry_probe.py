"""Tests for optimizer/scheduler availability probing."""

import pytest

from rengu_flow_ui.registry_probe import (
    probe_optimizer,
    probe_resolution,
    probe_scheduler,
    resolution_errors,
)


def test_probe_optimizer_adamw() -> None:
    r = probe_optimizer("adamw")
    assert r["available"] is True
    assert "AdamW" in r["resolved_class"]


def test_probe_optimizer_unknown() -> None:
    r = probe_optimizer("not_a_real_optimizer_xyz_123")
    assert r["available"] is False
    assert "error" in r


def test_probe_optimizer_qualified_real() -> None:
    r = probe_optimizer("torch.optim.AdamW")
    assert r["available"] is True
    assert "AdamW" in r["resolved_class"]


def test_probe_optimizer_rejects_non_optimizer_class() -> None:
    # A scheduler class imports fine but is NOT an optimizer — must be flagged,
    # not reported as available (catches a scheduler pasted into the optimizer field).
    r = probe_optimizer("torch.optim.lr_scheduler.CosineAnnealingLR")
    assert r["available"] is False
    assert "not a torch.optim.Optimizer" in r["error"]


def test_probe_scheduler_cosine() -> None:
    r = probe_scheduler("cosine")
    assert r["available"] is True
    assert r["source"] == "registry"


def test_probe_scheduler_qualified_bad() -> None:
    r = probe_scheduler("no.such.module.Scheduler")
    assert r["available"] is False


def test_resolution_errors_ignores_missing_optimizer_extra() -> None:
    r = probe_optimizer("adamw8bit")
    if r["available"]:
        pytest.skip("bitsandbytes already installed")
    res = {"optimizer": r}
    assert resolution_errors(res) == []


def test_probe_optimizer_prodigy() -> None:
    pytest.importorskip("pytorch_optimizer")
    r = probe_optimizer("prodigy")
    assert r["available"] is True
    assert r["source"] == "optional_dependency"
    assert "Prodigy" in r["resolved_class"]


def test_resolution_errors_ignores_missing_prodigy_extra() -> None:
    r = probe_optimizer("prodigy")
    if r["available"]:
        pytest.skip("pytorch-optimizer already installed")
    res = {"optimizer": r}
    assert resolution_errors(res) == []


def test_probe_resolution_minimal_config() -> None:
    config = {
        "optimizer": {"type": "adamw"},
        "lr_scheduler": "cosine",
    }
    res = probe_resolution(config)
    assert res["optimizer"]["available"]
    assert res["scheduler"]["available"]
    assert resolution_errors(res) == []
