"""Unit tests for rengu_track.reader (cross-run comparison assembly)."""

import pytest

from rengu_track import append_event, reader
from rengu_track.run import RunManifest, write_manifest

pytestmark = pytest.mark.no_ui_db


def _make_run(root, run_id, *, lr, status="finished"):
    run_dir = root / run_id
    run_dir.mkdir()
    manifest = RunManifest(
        run_id=run_id,
        name=run_id,
        status=status,
        config={"optimizer": {"lr": lr}, "model": {"type": "sdxl"}},
    )
    # flat hparams are derived on write via build path; set them explicitly here.
    from rengu_track.run import flatten_hparams

    manifest.hparams_flat = flatten_hparams(manifest.config)
    manifest.summary = {"best_loss": lr * 10}
    write_manifest(run_dir, manifest)
    append_event(run_dir, "run_started", step=0)
    return run_dir


def test_run_row_and_missing(tmp_path):
    run_dir = _make_run(tmp_path, "run-a", lr=1e-4)
    row = reader.run_row(run_dir)
    assert row["run_id"] == "run-a"
    assert row["hparams"]["optimizer.lr"] == 1e-4
    assert row["summary"]["best_loss"] == 1e-3

    (tmp_path / "no-manifest").mkdir()
    assert reader.run_row(tmp_path / "no-manifest") is None


def test_list_run_dirs_only_manifested(tmp_path):
    _make_run(tmp_path, "run-a", lr=1e-4)
    _make_run(tmp_path, "run-b", lr=2e-4)
    (tmp_path / "junk").mkdir()
    dirs = reader.list_run_dirs(tmp_path)
    assert [d.name for d in dirs] == ["run-a", "run-b"]


def test_compare_runs_columns_and_timelines(tmp_path):
    _make_run(tmp_path, "run-a", lr=1e-4)
    _make_run(tmp_path, "run-b", lr=2e-4)
    payload = reader.compare_runs(reader.list_run_dirs(tmp_path))

    assert {r["run_id"] for r in payload["runs"]} == {"run-a", "run-b"}
    cols = {c["key"]: c["varies"] for c in payload["columns"]}
    # lr differs across runs, model is identical.
    assert cols["optimizer.lr"] is True
    assert cols["model.type"] is False
    # one timeline event per run.
    assert payload["timelines"]["run-a"][0]["type"] == "run_started"
    # series present (empty dict per run when there are no TB event files).
    assert set(payload["series"]) == {"run-a", "run-b"}
