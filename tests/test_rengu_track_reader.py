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


def test_compare_runs_is_metadata_only_by_default(tmp_path):
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
    # No event-file parsing by default → no series key, just the metric-name union.
    assert "series" not in payload
    assert "metrics" in payload


def test_compare_runs_metrics_union_from_manifest_tags(tmp_path):
    from rengu_track.run import RunManifest, write_manifest

    a = tmp_path / "run-a"
    a.mkdir()
    ma = RunManifest(run_id="run-a", name="run-a", scalar_tags=["train/loss", "val/loss"])
    write_manifest(a, ma)
    b = tmp_path / "run-b"
    b.mkdir()
    mb = RunManifest(run_id="run-b", name="run-b", scalar_tags=["train/loss", "system/vram_used_gb"])
    write_manifest(b, mb)

    payload = reader.compare_runs(reader.list_run_dirs(tmp_path))
    assert payload["metrics"] == ["system/vram_used_gb", "train/loss", "val/loss"]


def test_downsample_keeps_endpoints_and_caps(tmp_path):
    series = [{"step": i, "value": float(i), "wall_time": 0.0} for i in range(1000)]
    out = reader._downsample(series, 100)
    assert len(out) <= 100
    assert out[0]["step"] == 0
    assert out[-1]["step"] == 999
    # short series pass through unchanged.
    assert reader._downsample(series[:5], 100) == series[:5]


def test_read_scalars_max_points(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    reader.invalidate_scalars_cache()
    full = {"train/loss": [{"step": i, "value": float(i)} for i in range(1000)]}
    monkeypatch.setattr(reader, "_load_scalars", lambda _rd: full)
    monkeypatch.setattr(reader, "_latest_event_mtime", lambda _rd: 1.0)

    assert len(reader.read_scalars(run)["train/loss"]) == 1000  # no cap by default
    capped = reader.read_scalars(run, max_points=50)["train/loss"]
    assert len(capped) <= 50
    assert capped[0]["step"] == 0 and capped[-1]["step"] == 999


def test_concurrent_reads_parse_once(tmp_path, monkeypatch):
    """Concurrent per-metric requests for the same run must trigger a single event-file parse.

    The Compare view fires one series request per metric at once for the same run set; without
    deduping they'd each launch a full parse in parallel and saturate the box.
    """
    import threading

    run = tmp_path / "run"
    run.mkdir()
    reader.invalidate_scalars_cache()

    calls = {"n": 0}
    started = threading.Event()
    release = threading.Event()

    def slow_load(_rd):
        calls["n"] += 1
        started.set()
        # Hold the (locked) load open so every other thread is in flight and would stampede.
        release.wait(timeout=5)
        return {"train/loss": [{"step": 0, "value": 0.5}], "val/loss": [{"step": 0, "value": 0.9}]}

    monkeypatch.setattr(reader, "_load_scalars", slow_load)
    monkeypatch.setattr(reader, "_latest_event_mtime", lambda _rd: 1.0)

    results: list[int] = []

    def worker():
        results.append(len(reader.read_scalars(run).get("train/loss", [])))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    assert started.wait(timeout=5)  # first load has the lock; the rest are queued behind it
    release.set()
    for t in threads:
        t.join(timeout=5)

    assert calls["n"] == 1  # parsed once, not once per concurrent request
    assert results == [1] * 8  # every caller still got the data


def test_series_for_single_tag(tmp_path, monkeypatch):
    _make_run(tmp_path, "run-a", lr=1e-4)
    reader.invalidate_scalars_cache()
    monkeypatch.setattr(
        reader,
        "_load_scalars",
        lambda _rd: {"train/loss": [{"step": 0, "value": 0.5}], "val/loss": [{"step": 0, "value": 0.9}]},
    )
    monkeypatch.setattr(reader, "_latest_event_mtime", lambda _rd: 1.0)

    out = reader.series_for([tmp_path / "run-a"], "train/loss")
    assert out["run-a"] == [{"step": 0, "value": 0.5}]


def test_preview_images_parses_step_and_prompt(tmp_path):
    run = tmp_path / "run"
    (run / "preview").mkdir(parents=True)
    for fn in (
        "step00000100_portrait.png",
        "step00000500_portrait.png",
        "step00000500_landscape.png",
        "ungrouped.png",
    ):
        (run / "preview" / fn).write_bytes(b"x")

    items = reader.preview_images(run)
    keyed = {(it["prompt"], it["step"]) for it in items}
    assert ("portrait", 100) in keyed
    assert ("portrait", 500) in keyed
    assert ("landscape", 500) in keyed
    assert ("ungrouped", None) in keyed  # non-conforming name → step None, prompt = stem
    # Newest step first; unparsed sorts last (so the cap keeps recent frames).
    assert items[0]["step"] == 500
    assert items[-1]["step"] is None


def test_preview_images_missing_dir(tmp_path):
    assert reader.preview_images(tmp_path / "nope") == []


def test_compare_runs_config_fallback_for_unmanifested(tmp_path):
    _make_run(tmp_path, "run-a", lr=1e-4)  # has run.json
    legacy = tmp_path / "legacy"  # no run.json — only reachable via fallback
    legacy.mkdir()

    def fallback(run_dir):
        from pathlib import Path as P

        return {
            "run_id": P(run_dir).name,
            "name": P(run_dir).name,
            "status": "imported",
            "created_at": "",
            "updated_at": "",
            "hparams": {"optimizer.lr": 3e-4, "model.type": "sdxl"},
            "summary": {},
            "system_summary": {},
            "lineage": {},
            "hardware": {},
            "tags": [],
            "last_scalars": {},
        }

    payload = reader.compare_runs([tmp_path / "run-a", legacy], config_fallback=fallback)
    assert {r["run_id"] for r in payload["runs"]} == {"run-a", "legacy"}
    cols = {c["key"]: c["varies"] for c in payload["columns"]}
    assert cols["optimizer.lr"] is True  # 1e-4 vs 3e-4
    # legacy has no tag index → curated defaults offered so its curves can still load lazily.
    assert "train/loss" in payload["metrics"]
