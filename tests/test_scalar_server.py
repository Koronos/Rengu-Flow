"""Scalar reads via the TensorBoard Rust data server, with EventAccumulator fallback."""

import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_ui_db


def _write_run(tmp_path: Path) -> tuple[Path, Path]:
    """Make ``<out>/run20260101`` with a manifest (run_id == dir name) and 5 loss scalars."""
    pytest.importorskip("torch")
    pytest.importorskip("tensorboard")
    from rengu_track.backends.tensorboard import TensorBoardBackend
    from rengu_track.run import RunManifest, write_manifest

    out = tmp_path / "out"
    run = out / "run20260101"
    run.mkdir(parents=True)
    write_manifest(run, RunManifest(run_id="run20260101", name="r"))
    backend = TensorBoardBackend(run)
    for i in range(5):
        backend.scalar("train/loss", 1.0 / (i + 1), step=i)
    backend.close()
    return out, run


def test_scalar_server_reads_our_events(tmp_path: Path) -> None:
    """The Rust data server parses event files we wrote and returns the scalar series."""
    from rengu_track import scalar_server

    out, _ = _write_run(tmp_path)
    try:
        # The server loads asynchronously after start(); poll briefly for the tiny run to land.
        data: dict = {}
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                data = scalar_server.read_scalars(
                    out, ["run20260101"], tags=["train/loss"], max_points=600
                )
            except scalar_server.DataServerUnavailable as exc:
                pytest.skip(f"data server unavailable on this platform: {exc}")
            if data.get("run20260101", {}).get("train/loss"):
                break
            time.sleep(0.3)
    finally:
        scalar_server.shutdown()

    pts = data["run20260101"]["train/loss"]
    assert [p["step"] for p in pts] == [0, 1, 2, 3, 4]
    assert pts[0]["value"] == pytest.approx(1.0)


def test_series_for_tag_end_to_end(tmp_path: Path) -> None:
    """series_for_tag returns the run's series keyed by run_id (data server OR fallback path)."""
    from rengu_track import reader, scalar_server

    out, _ = _write_run(tmp_path)
    reader.invalidate_scalars_cache()
    try:
        series = reader.series_for_tag(out, ["run20260101"], "train/loss", max_points=600)
    finally:
        scalar_server.shutdown()
        reader.invalidate_scalars_cache()
    assert [p["step"] for p in series["run20260101"]] == [0, 1, 2, 3, 4]


def test_scalars_for_run_end_to_end(tmp_path: Path) -> None:
    """scalars_for_run returns one run's tags (the detail-view board path) via data server/fallback."""
    from rengu_track import reader, scalar_server

    out, run = _write_run(tmp_path)
    reader.invalidate_scalars_cache()
    try:
        scalars = reader.scalars_for_run(run, "train/", max_points=600)
    finally:
        scalar_server.shutdown()
        reader.invalidate_scalars_cache()
    assert [p["step"] for p in scalars["train/loss"]] == [0, 1, 2, 3, 4]


def test_scalars_for_run_falls_back_and_filters_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On data-server failure it uses EventAccumulator, and the tag_prefix still filters."""
    from rengu_track import reader, scalar_server

    run = tmp_path / "out" / "run"
    run.mkdir(parents=True)

    def _unavailable(*_a, **_k):
        raise scalar_server.DataServerUnavailable("forced")

    monkeypatch.setattr(scalar_server, "read_scalars", _unavailable)
    monkeypatch.setattr(
        reader,
        "_load_scalars",
        lambda _rd: {
            "train/loss": [{"step": 0, "value": 0.5, "wall_time": 0.0}],
            "val/loss": [{"step": 0, "value": 0.9, "wall_time": 0.0}],
        },
    )
    monkeypatch.setattr(reader, "_latest_event_mtime", lambda _rd: 1.0)
    reader.invalidate_scalars_cache()

    out = reader.scalars_for_run(run, "train/", max_points=600)
    assert "train/loss" in out
    assert "val/loss" not in out  # prefix filter applied on the fallback path too


def test_series_for_tag_falls_back_when_server_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the data server can't be used, series_for_tag uses the EventAccumulator path instead."""
    from rengu_track import reader, scalar_server

    out = tmp_path / "out"
    run = out / "run"
    run.mkdir(parents=True)  # no manifest → run_id falls back to the dir name "run"

    def _unavailable(*_a, **_k):
        raise scalar_server.DataServerUnavailable("forced")

    monkeypatch.setattr(scalar_server, "read_scalars", _unavailable)
    monkeypatch.setattr(
        reader,
        "_load_scalars",
        lambda _rd: {"train/loss": [{"step": 0, "value": 0.5, "wall_time": 0.0}]},
    )
    monkeypatch.setattr(reader, "_latest_event_mtime", lambda _rd: 1.0)
    reader.invalidate_scalars_cache()

    series = reader.series_for_tag(out, ["run"], "train/loss")
    assert series["run"] == [{"step": 0, "value": 0.5, "wall_time": 0.0}]
