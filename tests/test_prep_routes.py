"""Tag-editor REST flow: session -> stage -> diff -> commit (backup) -> restore."""

import shutil
from pathlib import Path

import pytest

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)


@pytest.fixture
def img_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name, caption in (
        ("a.jpg", "1girl, long hair, watermark\nA girl with long hair.\n"),
        ("b.jpg", "1girl, short hair\n"),
        ("c.jpg", "2girls, watermark\n"),
    ):
        shutil.copy(FIXTURE_JPG, d / name)
        (d / name).with_suffix(".txt").write_text(caption)
    return d


def _open_session(ui_client, img_dir):
    res = ui_client.post("/api/v1/prep/tags/sessions", json={"path": str(img_dir)})
    assert res.status_code == 200, res.text
    return res.json()["session_id"]


def test_session_lifecycle_stage_diff_commit_restore(ui_client, img_dir):
    sid = _open_session(ui_client, img_dir)

    stats = ui_client.get(f"/api/v1/prep/tags/sessions/{sid}/stats").json()
    by_tag = {t["tag"]: t["count"] for t in stats["tags"]}
    assert by_tag["1girl"] == 2 and by_tag["watermark"] == 2

    # Stage: remove watermark everywhere; nothing on disk yet.
    res = ui_client.post(
        f"/api/v1/prep/tags/sessions/{sid}/ops",
        json={"ops": [{"op": "remove", "tags": ["watermark"]}]},
    )
    assert res.status_code == 200
    assert res.json()["changed_count"] == 2
    assert "watermark" in (img_dir / "a.txt").read_text()

    diff = ui_client.get(f"/api/v1/prep/tags/sessions/{sid}/diff").json()
    assert diff["total"] == 2
    keys = {e["key"] for e in diff["entries"]}
    assert keys == {"a.jpg", "c.jpg"}

    commit = ui_client.post(f"/api/v1/prep/tags/sessions/{sid}/commit").json()
    assert commit["backup"]
    assert sorted(commit["files_written"]) == ["a.txt", "c.txt"]
    assert "watermark" not in (img_dir / "a.txt").read_text()
    # NL caption line survived the bulk edit.
    assert "A girl with long hair." in (img_dir / "a.txt").read_text()

    backups = ui_client.get(
        "/api/v1/prep/tags/backups", params={"path": str(img_dir)}
    ).json()["backups"]
    assert backups[0]["name"] == commit["backup"]

    restore = ui_client.post(
        "/api/v1/prep/tags/restore",
        json={"path": str(img_dir), "backup": commit["backup"]},
    )
    assert restore.status_code == 200
    assert "watermark" in (img_dir / "a.txt").read_text()


def test_undo_pops_staged_op(ui_client, img_dir):
    sid = _open_session(ui_client, img_dir)
    ui_client.post(
        f"/api/v1/prep/tags/sessions/{sid}/ops",
        json={"ops": [{"op": "add", "tags": ["masterpiece"], "scope": "line1"}]},
    )
    summary = ui_client.post(f"/api/v1/prep/tags/sessions/{sid}/undo").json()
    assert summary["staged_ops"] == []
    assert summary["changed_count"] == 0


def test_query_returns_previews(ui_client, img_dir):
    sid = _open_session(ui_client, img_dir)
    res = ui_client.post(
        f"/api/v1/prep/tags/sessions/{sid}/query",
        json={"filter": {"any": ["watermark"]}},
    ).json()
    assert sorted(res["keys"]) == ["a.jpg", "c.jpg"]
    token = res["previews"]["a.jpg"]
    img = ui_client.get("/api/v1/datasets/preview-image", params={"t": token})
    assert img.status_code == 200
    assert img.headers["content-type"].startswith("image/")


def test_quarantine_commit_and_restore(ui_client, img_dir):
    sid = _open_session(ui_client, img_dir)
    ui_client.post(
        f"/api/v1/prep/tags/sessions/{sid}/ops",
        json={"ops": [{"op": "quarantine", "filter": {"all": ["2girls"]}}]},
    )
    summary = ui_client.get(f"/api/v1/prep/tags/sessions/{sid}").json()
    assert summary["quarantine_pending"] == ["c.jpg"]

    commit = ui_client.post(f"/api/v1/prep/tags/sessions/{sid}/commit").json()
    assert commit["quarantined"] == ["c.jpg"]
    assert not (img_dir / "c.jpg").exists()

    batches = ui_client.get(
        "/api/v1/prep/tags/quarantine", params={"path": str(img_dir)}
    ).json()["batches"]
    assert batches[0]["images"] == ["c.jpg"]
    ui_client.post(
        "/api/v1/prep/tags/quarantine/restore",
        json={"path": str(img_dir), "batch": batches[0]["name"]},
    )
    assert (img_dir / "c.jpg").exists()
    assert (img_dir / "c.txt").read_text() == "2girls, watermark\n"


def test_errors(ui_client, img_dir):
    assert ui_client.get("/api/v1/prep/tags/sessions/nope").status_code == 404
    assert (
        ui_client.post(
            "/api/v1/prep/tags/sessions", json={"path": str(img_dir / "missing")}
        ).status_code
        == 404
    )
    sid = _open_session(ui_client, img_dir)
    # Commit with nothing staged -> 400; bad op -> 400.
    assert ui_client.post(f"/api/v1/prep/tags/sessions/{sid}/commit").status_code == 400
    assert (
        ui_client.post(
            f"/api/v1/prep/tags/sessions/{sid}/ops",
            json={"ops": [{"op": "explode"}]},
        ).status_code
        == 400
    )


# --- prep jobs -------------------------------------------------------------------


def test_create_prep_job_enqueues_kind_prep(ui_client, img_dir):
    res = ui_client.post(
        "/api/v1/prep/jobs",
        json={"stage": "tag", "config": {"path": str(img_dir)}, "start_now": False},
    )
    assert res.status_code == 200, res.text
    job = res.json()
    assert job["kind"] == "prep"
    assert job["state"] == "pending"
    assert job["extra_args"] == "tag"
    assert Path(job["config_path"]).is_file()
    assert "path = " in Path(job["config_path"]).read_text()
    assert job["run_dir"] and Path(job["run_dir"]).is_dir()

    # Prep jobs don't pollute the Runs queue (default kind=train)...
    train_jobs = ui_client.get("/api/v1/jobs").json()["jobs"]
    assert all(j["kind"] == "train" for j in train_jobs)
    # ...but show up under kind=prep.
    prep_jobs = ui_client.get("/api/v1/jobs", params={"kind": "prep"}).json()["jobs"]
    assert [j["id"] for j in prep_jobs] == [job["id"]]


def test_create_prep_job_validates_stage_and_path(ui_client, img_dir):
    bad_stage = ui_client.post(
        "/api/v1/prep/jobs", json={"stage": "explode", "config": {"path": str(img_dir)}}
    )
    assert bad_stage.status_code == 400
    bad_path = ui_client.post(
        "/api/v1/prep/jobs", json={"stage": "tag", "config": {"path": "/nope/missing"}}
    )
    assert bad_path.status_code == 404


def test_prep_job_stays_pending_while_train_runs(ui_client, img_dir, monkeypatch):
    from rengu_flow_ui import db

    # Simulate an active training run holding the single-runner queue.
    train = db.create_job(config_path="/tmp/x.toml", log_path="/tmp/x.log", state="running")
    started = []
    monkeypatch.setattr("rengu_flow_ui.jobs.start_job", lambda job, **k: started.append(job.id))

    res = ui_client.post(
        "/api/v1/prep/jobs",
        json={"stage": "tag", "config": {"path": str(img_dir)}, "start_now": True},
    )
    assert res.status_code == 200
    assert res.json()["state"] == "pending"
    assert started == []  # the queue gate held: nothing launched
    db.delete_job(train.id)


def test_build_prep_command_shape(tmp_path):
    from rengu_flow_ui.jobs import build_prep_command

    cmd = build_prep_command(tmp_path / "prep.toml", stage="caption", job_dir=tmp_path / "j")
    assert cmd[1:4] == ["-m", "rengu_flow.cli", "prep"]
    assert "caption" in cmd
    assert "--job-dir" in cmd and str(tmp_path / "j") in cmd


def test_prep_job_report_endpoint(ui_client, img_dir):
    import json as _json

    job = ui_client.post(
        "/api/v1/prep/jobs", json={"stage": "tag", "config": {"path": str(img_dir)}}
    ).json()
    res = ui_client.get(f"/api/v1/prep/jobs/{job['id']}/report")
    assert res.json() == {"report": None}
    Path(job["run_dir"], "report.json").write_text(_json.dumps({"tagged": 3}))
    res = ui_client.get(f"/api/v1/prep/jobs/{job['id']}/report")
    assert res.json()["report"]["tagged"] == 3


def test_size_query_and_quarantine_by_keys(ui_client, img_dir):
    from PIL import Image as PILImage

    # One genuinely tiny image among the normal fixtures.
    PILImage.new("RGB", (100, 80), (50, 50, 50)).save(img_dir / "tiny.jpg")
    (img_dir / "tiny.txt").write_text("lowres\n")

    sid = _open_session(ui_client, img_dir)
    res = ui_client.post(
        f"/api/v1/prep/tags/sessions/{sid}/size-query", json={"below": 256}
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["keys"] == ["tiny.jpg"]
    assert data["sizes"]["tiny.jpg"] == [100, 80]
    assert "tiny.jpg" in data["previews"]

    # Missing params -> 400.
    assert (
        ui_client.post(f"/api/v1/prep/tags/sessions/{sid}/size-query", json={}).status_code
        == 400
    )

    ui_client.post(
        f"/api/v1/prep/tags/sessions/{sid}/ops",
        json={"ops": [{"op": "quarantine", "keys": data["keys"]}]},
    )
    commit = ui_client.post(f"/api/v1/prep/tags/sessions/{sid}/commit").json()
    assert commit["quarantined"] == ["tiny.jpg"]
    assert not (img_dir / "tiny.jpg").exists()


def test_prep_jobs_hidden_from_train_history_and_active(ui_client, img_dir):
    from rengu_flow_ui import db

    job = ui_client.post(
        "/api/v1/prep/jobs", json={"stage": "tag", "config": {"path": str(img_dir)}}
    ).json()
    db.update_job(job["id"], state="running", pid=999999)

    runs = ui_client.get("/api/v1/train/runs").json()
    assert all(str(r.get("id")) != str(job["id"]) for r in runs["items"])
    assert runs["stats"]["running"] == 0  # the running prep job doesn't count

    active = ui_client.get("/api/v1/train/active").json()
    assert not active or active.get("run") in (None, {})
    db.update_job(job["id"], state="stopped", pid=None)


def test_requeue_terminal_prep_job(ui_client, img_dir, monkeypatch):
    from pathlib import Path as P

    from rengu_flow.utils.signal_files import SIGNAL_SAVE_QUIT
    from rengu_flow_ui import db

    job = ui_client.post(
        "/api/v1/prep/jobs", json={"stage": "tag", "config": {"path": str(img_dir)}}
    ).json()
    # Pending -> cannot requeue.
    assert ui_client.post(f"/api/v1/prep/jobs/{job['id']}/requeue").status_code == 400

    db.update_job(job["id"], state="stopped", exit_code=1, finished_at="2026-01-01")
    P(job["run_dir"], SIGNAL_SAVE_QUIT).touch()  # leftover stop signal

    started = []
    monkeypatch.setattr("rengu_flow_ui.jobs.start_job", lambda j, **k: started.append(j.id))
    res = ui_client.post(
        f"/api/v1/prep/jobs/{job['id']}/requeue", json={"start_now": True}
    )
    assert res.status_code == 200, res.text
    requeued = res.json()
    assert requeued["state"] in ("pending", "running")
    assert requeued["exit_code"] is None and requeued["finished_at"] is None
    assert not P(job["run_dir"], SIGNAL_SAVE_QUIT).exists()  # signal cleared
    assert started == [job["id"]]

    # Train jobs are rejected.
    train = db.create_job(config_path="/tmp/x.toml", log_path="/tmp/x.log", state="stopped")
    assert ui_client.post(f"/api/v1/prep/jobs/{train.id}/requeue").status_code == 400
    db.delete_job(train.id)


def test_logs_endpoint_serves_terminal_job_log(ui_client, img_dir):
    from rengu_flow_ui import db

    job = ui_client.post(
        "/api/v1/prep/jobs", json={"stage": "tag", "config": {"path": str(img_dir)}}
    ).json()
    Path(job["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["log_path"]).write_text("boom traceback here\nexits with return code = 1\n")
    db.update_job(job["id"], state="failed", exit_code=1)

    res = ui_client.get(f"/api/v1/jobs/{job['id']}/logs", params={"offset": 0})
    assert res.status_code == 200
    assert "boom traceback here" in res.json()["chunk"]


def test_caption_prompt_preview_is_model_native(ui_client):
    # ToriiGate: native trained format, not the instruction composition.
    res = ui_client.post(
        "/api/v1/prep/caption-prompts/preview",
        json={"caption": {"model": "toriigate-0.5", "character_name": "miku"}},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["native_format"] is True
    assert data["prompt"].startswith("# Captioning format:")
    assert "make sure to use them: [miku]" in data["prompt"]
    assert "# Booru tags for the image\n[1girl, long hair, smile]" in data["prompt"]

    # JoyCaption: instruction composition.
    res = ui_client.post(
        "/api/v1/prep/caption-prompts/preview",
        json={"caption": {"model": "joycaption-beta-one", "prompt_modifiers": ["medium_neutral"]}},
    )
    data = res.json()
    assert data["native_format"] is False
    assert "never mention or hint at the medium" in data["prompt"].lower()

    # Custom prompt wins and is shown as-is (plus grounding for toriigate).
    res = ui_client.post(
        "/api/v1/prep/caption-prompts/preview",
        json={"caption": {"model": "toriigate-0.5", "prompt": "My custom."}},
    )
    data = res.json()
    assert data["prompt"].startswith("My custom.")
    assert data["native_format"] is False


def test_caption_prompt_options_expose_sampling_defaults(ui_client):
    res = ui_client.get("/api/v1/prep/caption-prompts").json()
    assert res["sampling_defaults"]["toriigate-0.5"] == {"temperature": 0.5, "top_p": 1.0}
    assert res["sampling_defaults"]["joycaption-beta-one"] == {"temperature": 0.6, "top_p": 0.9}
