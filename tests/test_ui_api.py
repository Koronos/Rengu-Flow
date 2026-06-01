"""HTTP API tests for the control plane (FastAPI TestClient)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rengu_flow_ui import db

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

    r3 = ui_client.get("/api/v1/augmentations")
    assert r3.status_code == 200
    aug = r3.json()
    assert "presets" in aug
    assert "strategies" in aug
    assert any(p["name"] == "easy" for p in aug["presets"])


def test_config_render_toml_merge_preserves_scheduler(ui_client) -> None:
    example = (Path(__file__).resolve().parents[1] / "examples" / "minimal_config_lora_sdxl.toml").read_text(
        encoding="utf-8"
    )
    parse_r = ui_client.post("/api/v1/configs/parse-toml", json={"content": example})
    assert parse_r.status_code == 200
    form = parse_r.json()["form"]
    render_r = ui_client.post(
        "/api/v1/configs/render-toml",
        json={"form": form, "base_content": example},
    )
    assert render_r.status_code == 200
    rendered = render_r.json()["content"]
    assert "lr_scheduler" in rendered
    assert "checkpoint_path" in rendered


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


def test_validate_adamw8bit_ok_without_optimizer_dependency_errors(ui_client) -> None:
    """Validate must not fail on missing bitsandbytes; extras install at train start."""
    content = MINIMAL_TOML.replace('type = "adamw"', 'type = "adamw8bit"')
    r = ui_client.post("/api/v1/validate", json={"content": content})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    errors = body.get("errors") or []
    joined = " ".join(str(e) for e in errors).lower()
    assert "bitsandbytes" not in joined
    assert "optional dependency" not in joined
    assert "pip install" not in joined


def test_docs_endpoint(ui_client) -> None:
    r = ui_client.get("/api/v1/docs", params={"path": "docs/user/web-ui.md"})
    assert r.status_code == 200
    assert "Web UI" in r.json()["content"]


def test_docs_path_traversal_returns_404(ui_client) -> None:
    r = ui_client.get("/api/v1/docs", params={"path": "docs/../../../etc/passwd"})
    assert r.status_code == 404
    assert r.json()["detail"] == "Documentation file not found"


def test_docs_encoded_traversal_returns_404(ui_client) -> None:
    r = ui_client.get(
        "/api/v1/docs",
        params={"path": "docs/user/%2e%2e/%2e%2e/README.md"},
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "Documentation file not found"


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

    r = ui_client.get("/api/v1/datasets?page=1&page_size=20&sort=id&order=desc")
    assert r.status_code == 200
    page_body = r.json()
    assert "items" in page_body
    assert "total" in page_body
    assert any(row["id"] == lib_id for row in page_body["items"])

    r = ui_client.get("/api/v1/datasets")
    assert r.status_code == 200
    list_body = r.json()
    assert "datasets" in list_body
    assert "picker" in list_body
    assert any(row["id"] == lib_id for row in list_body["datasets"])

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


def test_dataset_scan_path(ui_client, tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("caption")
    r = ui_client.post("/api/v1/datasets/scan-path", json={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["image_count"] == 1
    assert body["caption_txt_files"] == 1

    r2 = ui_client.post(
        "/api/v1/datasets/scan-path", json={"path": "/nonexistent/rengu_flow_scan"}
    )
    assert r2.status_code == 200
    assert r2.json()["ok"] is False


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
    def fake_start(job: db.JobRecord) -> int:
        db.update_job(job.id, state="running", pid=12345)
        return 12345

    monkeypatch.setattr("rengu_flow_ui.jobs.start_job", fake_start)
    monkeypatch.setattr("rengu_flow_ui.jobs.poll_job", lambda job_id: db.get_job(job_id))

    r = ui_client.post(
        "/api/v1/jobs",
        json={"content": MINIMAL_TOML, "num_gpus": 1, "enqueue": True},
    )
    assert r.status_code == 200
    job = r.json()
    assert job["state"] in ("running", "pending")

    r_cache = ui_client.post(
        "/api/v1/jobs",
        json={"content": MINIMAL_TOML, "num_gpus": 1, "cache_only": True, "enqueue": True},
    )
    assert r_cache.status_code == 200
    assert "--cache_only" in r_cache.json()["extra_args"]

    r2 = ui_client.get(f"/api/v1/jobs/{job['id']}")
    assert r2.status_code == 200


def test_jobs_draft_enqueue_reorder_seed_http(
    ui_client, ui_data_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Never actually launch a runner during this test.
    monkeypatch.setattr("rengu_flow_ui.job_queue.try_start_next", lambda: None)

    # Save for later -> a "new" draft: stored config, no queue slot, not started.
    r = ui_client.post(
        "/api/v1/jobs",
        json={"content": MINIMAL_TOML, "num_gpus": 2, "save_for_later": True, "trust_cache": True},
    )
    assert r.status_code == 200
    draft = r.json()
    assert draft["state"] == "new"
    assert draft["queue_position"] is None
    assert draft["num_gpus"] == 2
    assert draft["trust_cache"] is True

    # Promote the draft into the pending queue.
    r = ui_client.post(f"/api/v1/jobs/{draft['id']}/enqueue")
    assert r.status_code == 200
    assert r.json()["state"] == "pending"

    # Queue two more, then reorder the whole pending list.
    ids = [draft["id"]]
    for _ in range(2):
        rr = ui_client.post("/api/v1/jobs", json={"content": MINIMAL_TOML, "enqueue": True})
        assert rr.status_code == 200
        ids.append(rr.json()["id"])
    reordered = list(reversed(ids))
    r = ui_client.post(
        "/api/v1/jobs/queue/reorder", json={"ids": [int(x) for x in reordered]}
    )
    assert r.status_code == 200
    queue = r.json()["queue"]
    assert [str(j["id"]) for j in queue] == [str(x) for x in reordered]

    # Seed a new run from an existing one: config content is preserved.
    r = ui_client.get(f"/api/v1/jobs/{draft['id']}/seed")
    assert r.status_code == 200
    assert 'type = "sdxl"' in r.json()["content"]

    # A run with no folder yet has no checkpoints.
    r = ui_client.get(f"/api/v1/jobs/{draft['id']}/checkpoints")
    assert r.status_code == 200
    assert r.json()["checkpoints"] == []


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
    r = client.get("/api/v1/jobs")
    assert r.status_code == 401
    assert "Invalid token" in r.json()["detail"]

    r2 = client.get("/api/v1/jobs", headers=headers)
    assert r2.status_code == 200
