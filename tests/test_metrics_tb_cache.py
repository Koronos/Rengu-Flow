"""Tests for TensorBoard scalar read cache in the UI."""

from __future__ import annotations

from rengu_flow_ui import metrics_tb


def test_read_scalars_caches_until_mtime_changes(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    metrics_tb.invalidate_scalars_cache()
    loads: list[int] = []

    def fake_load(_run_dir):
        loads.append(1)
        return {"train/loss": [{"step": 1, "value": 0.5, "wall_time": 0.0}]}

    mtime = {"v": 1.0}

    monkeypatch.setattr(metrics_tb, "_load_scalars", fake_load)
    monkeypatch.setattr(metrics_tb, "_latest_event_mtime", lambda _rd: mtime["v"])

    a = metrics_tb.read_scalars(run)
    b = metrics_tb.read_scalars(run)
    assert a == b
    assert len(loads) == 1

    mtime["v"] = 2.0
    c = metrics_tb.read_scalars(run)
    assert c == a
    assert len(loads) == 2


def test_invalidate_scalars_cache(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    metrics_tb.invalidate_scalars_cache()
    loads: list[int] = []

    monkeypatch.setattr(
        metrics_tb,
        "_load_scalars",
        lambda _rd: loads.append(1) or {"train/x": []},
    )
    monkeypatch.setattr(metrics_tb, "_latest_event_mtime", lambda _rd: 1.0)

    metrics_tb.read_scalars(run)
    metrics_tb.invalidate_scalars_cache(run)
    metrics_tb.read_scalars(run)
    assert len(loads) == 2
