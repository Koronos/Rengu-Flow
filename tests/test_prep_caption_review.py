"""Caption editor REST surface: list a folder's captions (with edit-set controls), save one image.

Every write goes through ``rengu_flow.prep.caption_store`` — the same reader/writer the tag editor
and the prep stages use — so these tests check the files on disk, not the store's internals.
"""

import json
import shutil
from pathlib import Path

import pytest

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)
LIST = "/api/v1/prep/captions"
SAVE = "/api/v1/prep/captions/save"


@pytest.fixture
def img_dir(tmp_path):
    d = tmp_path / "targets"
    d.mkdir()
    for name, caption in (
        ("a.jpg", "Make it black and white.\nA girl, grayscale.\n"),
        ("b.jpg", "Remove the man on the left.\n"),
        ("c.jpg", None),  # no caption yet
    ):
        shutil.copy(FIXTURE_JPG, d / name)
        if caption is not None:
            (d / name).with_suffix(".txt").write_text(caption, encoding="utf-8")
    return d


@pytest.fixture
def control_dir(tmp_path):
    """``a`` pairs with one control, ``b`` with two (stem_0, stem_1), ``c`` with none."""
    d = tmp_path / "controls"
    d.mkdir()
    for name in ("a.jpg", "b_1.jpg", "b_0.jpg"):
        shutil.copy(FIXTURE_JPG, d / name)
    return d


def _list(ui_client, path, **params):
    res = ui_client.get(LIST, params={"path": str(path), **params})
    assert res.status_code == 200, res.text
    return res.json()


def _save(ui_client, path, key, lines, **extra):
    return ui_client.post(SAVE, json={"path": str(path), "key": key, "lines": lines, **extra})


# ------------------------------------------------------------------------------ list


def test_list_returns_every_line_and_a_preview_token(ui_client, img_dir):
    body = _list(ui_client, img_dir)
    assert body["total"] == 3 and body["image_count"] == 3
    assert body["uncaptioned_count"] == 1
    assert body["read_only"] is False and body["active"] == []
    items = {it["key"]: it for it in body["items"]}
    assert items["a.jpg"]["lines"] == ["Make it black and white.", "A girl, grayscale."]
    assert items["c.jpg"]["lines"] == []
    # The token serves through the existing preview-image route.
    img = ui_client.get("/api/v1/datasets/preview-image", params={"t": items["a.jpg"]["token"]})
    assert img.status_code == 200 and img.content == FIXTURE_JPG.read_bytes()


def test_list_paginates_searches_and_filters(ui_client, img_dir):
    page = _list(ui_client, img_dir, limit=2, offset=0)
    assert [it["key"] for it in page["items"]] == ["a.jpg", "b.jpg"]
    assert page["total"] == 3 and page["limit"] == 2
    page2 = _list(ui_client, img_dir, limit=2, offset=2)
    assert [it["key"] for it in page2["items"]] == ["c.jpg"]

    # Search is case-insensitive over every line and the file name.
    assert [it["key"] for it in _list(ui_client, img_dir, q="GRAYSCALE")["items"]] == ["a.jpg"]
    assert [it["key"] for it in _list(ui_client, img_dir, q="b.jp")["items"]] == ["b.jpg"]
    only = _list(ui_client, img_dir, filter="uncaptioned")
    assert [it["key"] for it in only["items"]] == ["c.jpg"] and only["total"] == 1
    # Folder-wide counts do not depend on the filter.
    assert only["image_count"] == 3 and only["uncaptioned_count"] == 1


def test_list_pairs_controls_in_order_and_reports_unpaired(ui_client, img_dir, control_dir):
    body = _list(ui_client, img_dir, control_path=str(control_dir))
    items = {it["key"]: it for it in body["items"]}
    assert [c["name"] for c in items["a.jpg"]["controls"]] == ["a.jpg"]
    assert [c["name"] for c in items["b.jpg"]["controls"]] == ["b_0.jpg", "b_1.jpg"]
    assert items["c.jpg"]["controls"] == [] and "c.jpg" in items["c.jpg"]["unpaired"]
    assert items["a.jpg"]["unpaired"] is None
    assert body["unpaired_count"] == 1

    token = items["b.jpg"]["controls"][1]["token"]
    img = ui_client.get("/api/v1/datasets/preview-image", params={"t": token})
    assert img.status_code == 200

    only = _list(ui_client, img_dir, control_path=str(control_dir), filter="unpaired")
    assert [it["key"] for it in only["items"]] == ["c.jpg"]


def test_list_json_and_auto_format(ui_client, img_dir):
    for txt in img_dir.glob("*.txt"):
        txt.unlink()
    (img_dir / "captions.json").write_text(
        json.dumps({"a.jpg": ["from json"], "b.jpg": "one\ntwo"}), encoding="utf-8"
    )
    for fmt in ("json", "auto"):
        body = _list(ui_client, img_dir, format=fmt)
        items = {it["key"]: it["lines"] for it in body["items"]}
        assert body["format"] == "json"
        assert items == {"a.jpg": ["from json"], "b.jpg": ["one", "two"], "c.jpg": []}
    assert _list(ui_client, img_dir, format="sidecar")["items"][0]["lines"] == []


def test_list_errors(ui_client, img_dir, tmp_path):
    assert ui_client.get(LIST, params={"path": str(tmp_path / "nope")}).status_code == 404
    assert ui_client.get(LIST, params={"path": str(img_dir), "filter": "x"}).status_code == 400
    missing_controls = {"path": str(img_dir), "control_path": str(tmp_path / "nope")}
    assert ui_client.get(LIST, params=missing_controls).status_code == 404


# ------------------------------------------------------------------------------ save


def test_save_sidecar_writes_lines_and_deletes_empty(ui_client, img_dir):
    res = _save(ui_client, img_dir, "b.jpg", ["Remove the man on the right.", " ", "variant 2 "])
    assert res.status_code == 200, res.text
    assert res.json()["lines"] == ["Remove the man on the right.", "variant 2"]
    assert (img_dir / "b.txt").read_text(encoding="utf-8") == (
        "Remove the man on the right.\nvariant 2\n"
    )
    # Untouched neighbours stay byte-identical.
    assert (img_dir / "a.txt").read_text(encoding="utf-8").startswith("Make it black")

    assert _save(ui_client, img_dir, "c.jpg", ["new instruction"]).status_code == 200
    assert (img_dir / "c.txt").read_text(encoding="utf-8") == "new instruction\n"

    assert _save(ui_client, img_dir, "a.jpg", []).status_code == 200
    assert not (img_dir / "a.txt").exists()


def test_save_custom_extension(ui_client, img_dir):
    res = _save(ui_client, img_dir, "a.jpg", ["tagged"], ext="caption")
    assert res.status_code == 200, res.text
    assert (img_dir / "a.caption").read_text(encoding="utf-8") == "tagged\n"
    assert (img_dir / "a.txt").read_text(encoding="utf-8").startswith("Make it black")


def test_save_json_keeps_other_entries(ui_client, img_dir):
    (img_dir / "captions.json").write_text(
        json.dumps({"a.jpg": ["keep me"], "b.jpg": ["old"]}), encoding="utf-8"
    )
    res = _save(ui_client, img_dir, "b.jpg", ["new", "second"], format="json")
    assert res.status_code == 200, res.text
    data = json.loads((img_dir / "captions.json").read_text(encoding="utf-8"))
    assert data["b.jpg"] == ["new", "second"]
    assert data["a.jpg"] == ["keep me"]
    # The sidecars are not the json layout's files: never touched.
    assert (img_dir / "b.txt").read_text(encoding="utf-8") == "Remove the man on the left.\n"


def test_save_backup_snapshots_before_writing(ui_client, img_dir):
    from rengu_flow.prep.caption_store import CaptionStore

    res = _save(ui_client, img_dir, "b.jpg", ["changed"], backup=True)
    assert res.status_code == 200, res.text
    backup = res.json()["backup"]
    assert [b["name"] for b in CaptionStore.list_backups(img_dir)] == [backup]
    # The snapshot holds the caption as it was BEFORE this save; restoring brings it back.
    CaptionStore.restore_snapshot(img_dir, backup)
    assert (img_dir / "b.txt").read_text(encoding="utf-8") == "Remove the man on the left.\n"

    # Without backup=true a save takes no snapshot.
    assert _save(ui_client, img_dir, "b.jpg", ["again"]).json()["backup"] is None
    assert len(CaptionStore.list_backups(img_dir)) == 1


def test_save_conflict_when_caption_changed_on_disk(ui_client, img_dir):
    (img_dir / "b.txt").write_text("written by someone else\n", encoding="utf-8")
    res = _save(
        ui_client, img_dir, "b.jpg", ["mine"], expected=["Remove the man on the left."]
    )
    assert res.status_code == 409
    assert (img_dir / "b.txt").read_text(encoding="utf-8") == "written by someone else\n"
    # Matching ``expected`` (what the editor loaded) saves.
    ok = _save(ui_client, img_dir, "b.jpg", ["mine"], expected=["written by someone else"])
    assert ok.status_code == 200


@pytest.mark.parametrize(
    "key", ["../outside.jpg", "..\\outside.jpg", "sub/a.jpg", "nope.jpg", "a.txt"]
)
def test_save_rejects_keys_outside_the_folder(ui_client, img_dir, key):
    from rengu_flow.prep.caption_store import CaptionStore

    outside = img_dir.parent / "outside.jpg"
    shutil.copy(FIXTURE_JPG, outside)
    res = _save(ui_client, img_dir, key, ["pwned"], backup=True)
    assert res.status_code == 404
    assert not (img_dir.parent / "outside.txt").exists()
    assert not (img_dir / "sub").exists()
    # Rejected before anything happens: not even the backup snapshot.
    assert CaptionStore.list_backups(img_dir) == []


@pytest.mark.parametrize("ext", ["/../../evil", ".txt/../x", ".t..x", ".a b"])
def test_rejects_unsafe_caption_extensions(ui_client, img_dir, ext):
    assert ui_client.get(LIST, params={"path": str(img_dir), "ext": ext}).status_code == 400
    res = _save(ui_client, img_dir, "a.jpg", ["x"], ext=ext)
    assert res.status_code == 400
    assert sorted(p.name for p in img_dir.iterdir()) == ["a.jpg", "a.txt", "b.jpg", "b.txt", "c.jpg"]


# ------------------------------------------------------------------------------ active writers


def _prep_job(img_dir, state):
    import toml

    from rengu_flow_ui import db

    return db.create_job(
        config_path="",
        log_path="",
        state=state,
        extra_args="edit_caption",
        config_content=toml.dumps({"path": str(img_dir)}),
        kind="prep",
    )


def test_running_prep_job_on_the_folder_makes_it_read_only(ui_client, img_dir, tmp_path):
    from rengu_flow_ui import db

    job = _prep_job(img_dir, "running")
    body = _list(ui_client, img_dir)
    assert body["read_only"] is True
    assert body["active"] == [
        {"kind": "job", "id": str(job.id), "stage": "edit_caption", "label": f"job #{job.id}"}
    ]
    res = _save(ui_client, img_dir, "b.jpg", ["racing the job"])
    assert res.status_code == 409
    assert (img_dir / "b.txt").read_text(encoding="utf-8") == "Remove the man on the left.\n"

    # Once it finishes the folder is writable again.
    db.update_job(job.id, state="finished")
    assert _list(ui_client, img_dir)["read_only"] is False
    assert _save(ui_client, img_dir, "b.jpg", ["ok"]).status_code == 200


def test_jobs_on_other_folders_or_not_running_do_not_block(ui_client, img_dir, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    _prep_job(other, "running")
    _prep_job(img_dir, "pending")
    _prep_job(img_dir, "finished")
    assert _list(ui_client, img_dir)["read_only"] is False
    assert _save(ui_client, img_dir, "b.jpg", ["ok"]).status_code == 200


def test_running_workflow_prep_step_on_the_folder_blocks_saves(ui_client, img_dir):
    """A workflow step runs prep with no job row — its node dir's prep.toml names the folder."""
    import toml

    from rengu_flow_ui import workflow_db

    wf = workflow_db.create_workflow("wf", "{}")
    node_dir = workflow_db.node_dir(wf.id, "n1")
    node_dir.mkdir(parents=True)
    (node_dir / "prep.toml").write_text(toml.dumps({"path": str(img_dir)}), encoding="utf-8")

    def _running(state: dict) -> None:
        state["status"] = "running"
        state["nodes"] = {"n1": {"status": "running"}}

    workflow_db.mutate_state(wf.id, _running)
    body = _list(ui_client, img_dir)
    assert body["read_only"] is True
    assert body["active"][0]["kind"] == "workflow"
    assert _save(ui_client, img_dir, "b.jpg", ["x"]).status_code == 409

    def _done(state: dict) -> None:
        state["status"] = "done"
        state["nodes"] = {"n1": {"status": "done"}}

    workflow_db.mutate_state(wf.id, _done)
    assert _save(ui_client, img_dir, "b.jpg", ["x"]).status_code == 200
