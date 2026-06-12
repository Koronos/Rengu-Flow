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
