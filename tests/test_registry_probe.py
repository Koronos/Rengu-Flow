"""Tests for optimizer/scheduler availability probing."""

import pytest

from renga_flow_ui.registry_probe import (
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


def test_probe_scheduler_cosine() -> None:
    r = probe_scheduler("cosine")
    assert r["available"] is True
    assert r["source"] == "registry"


def test_probe_scheduler_qualified_bad() -> None:
    r = probe_scheduler("no.such.module.Scheduler")
    assert r["available"] is False


def test_probe_resolution_minimal_config() -> None:
    config = {
        "optimizer": {"type": "adamw"},
        "lr_scheduler": "cosine",
    }
    res = probe_resolution(config)
    assert res["optimizer"]["available"]
    assert res["scheduler"]["available"]
    assert resolution_errors(res) == []
