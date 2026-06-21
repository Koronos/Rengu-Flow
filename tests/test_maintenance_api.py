"""Maintenance API: always disabled (machinery removed)."""

from __future__ import annotations


def test_maintenance_disabled_by_default(ui_client) -> None:
    r = ui_client.get("/api/v1/maintenance/enabled")
    assert r.status_code == 200
    assert r.json()["enabled"] is False
