"""HTTP API tests for the control plane (FastAPI TestClient)."""

from __future__ import annotations

from pathlib import Path

import pytest

from renga_flow_ui import configs_store, db, job_queue

MINIMAL_TOML = """
dataset = "examples/minimal_dataset.toml"
output_dir = "output"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"

[optimizer]
type = "adamw"
lr = 1.0e-4

epochs = 1
micro_batch_size_per_gpu = 1
synthetic_num_batches = 50
"""


def test_health(ui_client) -> None:
    r = ui_client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_schema_and_dataset_schema(ui_client) -> None:
    r = ui_client.get("/api/v1/schema")
    assert r.status_code == 200
    data = r.json()
    assert "sections" in data
    assert "sdxl" in data["registries"]["models"]

    r2 = ui_client.get("/api/v1/datasets/schema")
    assert r2.status_code == 200
    schema = r2.json()
    assert "directory_fields" in schema
    assert any(s["id"] == "resolutions" for s in schema["sections"])


def test_configs_search_paginated(ui_client, ui_data_tmp: Path) -> None:
    for _ in range(3):
        configs_store.insert_config(MINIMAL_TOML)
    r = ui_client.get("/api/v1/configs", params={"page": 1, "page_size": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 3
    assert len(body["items"]) == 2
    assert body["page"] == 1

    r2 = ui_client.get("/api/v1/configs")
    assert r2.status_code == 200
    assert "configs" in r2.json()


def test_config_crud_via_api(ui_client, ui_data_tmp: Path) -> None:
    r = ui_client.post("/api/v1/configs", json={"content": MINIMAL_TOML})
    assert r.status_code == 200
    cid = r.json()["id"]
    assert isinstance(cid, int)

    r = ui_client.get(f"/api/v1/configs/{cid}")
    assert r.status_code == 200
    assert "sdxl" in r.json()["content"]

    r = ui_client.get("/api/v1/configs")
    ids = [c["id"] for c in r.json()["configs"]]
    assert cid in ids

    updated = MINIMAL_TOML + '\nrun_name = "test"\n'
    r = ui_client.put(f"/api/v1/configs/{cid}", json={"content": updated})
    assert r.status_code == 200

    r = ui_client.post(f"/api/v1/configs/{cid}/duplicate")
    assert r.status_code == 200
    dup_id = r.json()["id"]
    assert dup_id != cid

    r = ui_client.delete(f"/api/v1/configs/{cid}")
    assert r.status_code == 200


def test_config_parse_render_toml(ui_client) -> None:
    r = ui_client.post("/api/v1/configs/parse-toml", json={"content": MINIMAL_TOML})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["form"]["_has_adapter"] is False
    assert body["form"]["model.type"] == "sdxl"

    r2 = ui_client.post("/api/v1/configs/render-toml", json={"form": body["form"]})
    assert r2.status_code == 200
    assert r2.json()["ok"] is True
    assert "sdxl" in r2.json()["content"]


def test_validate_endpoint(ui_client) -> None:
    r = ui_client.post("/api/v1/validate", json={"content": MINIMAL_TOML})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r2 = ui_client.post("/api/v1/validate", json={"content": "bad toml {"})
    assert r2.status_code == 200
    assert r2.json()["ok"] is False


def test_docs_endpoint(ui_client) -> None:
    r = ui_client.get("/api/v1/docs", params={"path": "docs/user/web-ui.md"})
    assert r.status_code == 200
    assert "Web UI" in r.json()["content"]


def test_system_stats(ui_client) -> None:
    r = ui_client.get("/api/v1/system/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "summary" in body
    assert "detail" in body


def test_dataset_library_api(ui_client, ui_data_tmp: Path) -> None:
    ds_toml = (
        "resolutions = [1024]\nframe_buckets = [1]\n\n"
        "[[directory]]\npath = '/tmp/x'\nnum_repeats = 1\n"
    )
    r = ui_client.post(
        "/api/v1/datasets",
        json={"content": ds_toml, "name": "Test portraits"},
    )
    assert r.status_code == 200
    body = r.json()
    lib_id = body["id"]
    assert isinstance(lib_id, int)
    assert body["name"] == "Test portraits"

    r = ui_client.get(f"/api/v1/datasets/{lib_id}")
    assert r.status_code == 200
    assert r.json()["name"] == "Test portraits"

    r = ui_client.post(
        "/api/v1/datasets/preview",
        json={"content": ds_toml},
    )
    assert r.status_code == 200
    preview = r.json()
    assert preview.get("ok") is True or "preview" in preview

    r = ui_client.post("/api/v1/datasets/compose", json={"source_ids": [lib_id]})
    assert r.status_code == 200
    merged_id = r.json()["id"]
    assert isinstance(merged_id, int)
    assert merged_id != lib_id

    r = ui_client.get("/api/v1/datasets/folder-suggestions")
    assert r.status_code == 200
    body = r.json()
    assert "suggestions" in body
    assert "missing" in body


def test_registry_probe(ui_client) -> None:
    r = ui_client.post(
        "/api/v1/registry/probe",
        json={"optimizer": "adamw", "scheduler": "cosine"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "optimizer" in body
    assert body["optimizer"]["available"] is True


def test_jobs_enqueue_mocked(ui_client, ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_cfg = configs_store.insert_config(MINIMAL_TOML)

    def fake_start(job: db.JobRecord) -> int:
        db.update_job(job.id, state="running", pid=12345)
        return 12345

    monkeypatch.setattr("renga_flow_ui.jobs.start_job", fake_start)
    monkeypatch.setattr("renga_flow_ui.jobs.poll_job", lambda job_id: db.get_job(job_id))

    r = ui_client.post(
        "/api/v1/jobs",
        json={"config_id": job_cfg, "num_gpus": 1, "enqueue": True},
    )
    assert r.status_code == 200
    job = r.json()
    assert job["state"] in ("running", "pending")
    assert job["config_id"] == job_cfg

    r2 = ui_client.get(f"/api/v1/jobs/{job['id']}")
    assert r2.status_code == 200


def test_tensorboard_status(ui_client) -> None:
    r = ui_client.get("/api/v1/tensorboard/status")
    assert r.status_code == 200
    assert r.json()["running"] is False


def test_tensorboard_start_missing_dir(ui_client) -> None:
    r = ui_client.post(
        "/api/v1/tensorboard/start",
        json={"output_dir": "definitely_not_an_output_dir_xyz"},
    )
    assert r.status_code == 404


def test_auth_token_required(ui_client_auth) -> None:
    client, headers = ui_client_auth
    r = client.get("/api/v1/configs")
    assert r.status_code == 401
    assert "Invalid token" in r.json()["detail"]

    r2 = client.get("/api/v1/configs", headers=headers)
    assert r2.status_code == 200
