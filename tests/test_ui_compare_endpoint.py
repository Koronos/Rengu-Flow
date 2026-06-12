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


def test_compare_endpoint_all_runs(ui_client, tmp_path: Path) -> None:
    _make_run(tmp_path, "run-a", 1e-4)
    _make_run(tmp_path, "run-b", 2e-4)

    resp = ui_client.get(f"/api/v1/runs/compare?output_dir={tmp_path}")
    assert resp.status_code == 200
    data = resp.json()

    assert {r["run_id"] for r in data["runs"]} == {"run-a", "run-b"}
    cols = {c["key"]: c["varies"] for c in data["columns"]}
    assert cols["optimizer.lr"] is True
    assert cols["model.type"] is False
    assert set(data["series"]) == {"run-a", "run-b"}
    assert set(data["timelines"]) == {"run-a", "run-b"}


def test_compare_endpoint_selected_runs(ui_client, tmp_path: Path) -> None:
    _make_run(tmp_path, "run-a", 1e-4)
    _make_run(tmp_path, "run-b", 2e-4)

    resp = ui_client.get(f"/api/v1/runs/compare?runs=run-a&output_dir={tmp_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert [r["run_id"] for r in data["runs"]] == ["run-a"]
