"""Integration test for the /runs/compare endpoint (rengu_track cross-run comparison)."""

from __future__ import annotations

from pathlib import Path

from rengu_track.run import RunManifest, flatten_hparams, write_manifest


def _make_run(root: Path, run_id: str, lr: float) -> None:
    run_dir = root / run_id
    run_dir.mkdir()
    manifest = RunManifest(
        run_id=run_id,
        name=run_id,
        status="finished",
        config={"optimizer": {"lr": lr}, "model": {"type": "sdxl"}},
    )
    manifest.hparams_flat = flatten_hparams(manifest.config)
    manifest.summary = {"best_loss": lr * 10}
    write_manifest(run_dir, manifest)


def test_compare_endpoint_metadata_only(ui_client, tmp_path: Path) -> None:
    _make_run(tmp_path, "run-a", 1e-4)
    _make_run(tmp_path, "run-b", 2e-4)

    resp = ui_client.get(f"/api/v1/runs/compare?output_dir={tmp_path}")
    assert resp.status_code == 200
    data = resp.json()

    assert {r["run_id"] for r in data["runs"]} == {"run-a", "run-b"}
    cols = {c["key"]: c["varies"] for c in data["columns"]}
    assert cols["optimizer.lr"] is True
    assert cols["model.type"] is False
    assert set(data["timelines"]) == {"run-a", "run-b"}
    # On-demand: the comparison payload carries no series, only the metric-name union.
    assert "series" not in data
    assert "metrics" in data


def test_compare_endpoint_selected_runs(ui_client, tmp_path: Path) -> None:
    _make_run(tmp_path, "run-a", 1e-4)
    _make_run(tmp_path, "run-b", 2e-4)

    resp = ui_client.get(f"/api/v1/runs/compare?runs=run-a&output_dir={tmp_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert [r["run_id"] for r in data["runs"]] == ["run-a"]


def test_series_endpoint(ui_client, tmp_path: Path) -> None:
    _make_run(tmp_path, "run-a", 1e-4)
    _make_run(tmp_path, "run-b", 2e-4)

    # Missing tag → 400.
    assert ui_client.get(f"/api/v1/runs/series?output_dir={tmp_path}").status_code == 400

    resp = ui_client.get(f"/api/v1/runs/series?tag=train/loss&output_dir={tmp_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tag"] == "train/loss"
    # No TB event files in these synthetic runs → empty series per run, correct shape.
    assert set(data["series"]) == {"run-a", "run-b"}


def test_run_previews_endpoint(ui_client, tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    (run / "preview").mkdir(parents=True)
    (run / "preview" / "step00000100_portrait.png").write_bytes(b"x")

    assert ui_client.get(f"/api/v1/runs/nope/previews?output_dir={tmp_path}").status_code == 404

    resp = ui_client.get(f"/api/v1/runs/run-a/previews?output_dir={tmp_path}")
    assert resp.status_code == 200
    previews = resp.json()["previews"]
    assert len(previews) == 1
    assert previews[0]["step"] == 100
    assert previews[0]["prompt"] == "portrait"


def test_compare_endpoint_discovers_unmanifested_runs(ui_client, tmp_path: Path) -> None:
    # A run with no run.json — just a config TOML (trained before tracking) — must still appear.
    legacy = tmp_path / "20260101_10-00-00_legacy"
    legacy.mkdir()
    (legacy / "train.toml").write_text(
        '[optimizer]\nlr = 0.0003\n[model]\ntype = "sdxl"\n', encoding="utf-8"
    )

    resp = ui_client.get(f"/api/v1/runs/compare?output_dir={tmp_path}")
    assert resp.status_code == 200
    data = resp.json()
    row = next((r for r in data["runs"] if r["run_id"] == legacy.name), None)
    assert row is not None
    assert row["status"] == "imported"
    assert row["hparams"]["optimizer.lr"] == 0.0003
    # curated metrics offered so the legacy run's curves can load lazily.
    assert "train/loss" in data["metrics"]
