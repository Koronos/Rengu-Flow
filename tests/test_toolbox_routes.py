import pytest

from rengu_flow_ui import toolbox


def _create(ui_client, **body):
    body.setdefault("name", "My Tool")
    res = ui_client.post("/api/v1/toolbox/tools", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_crud_works_even_when_execution_disabled(ui_client, monkeypatch):
    from rengu_flow.config import local_config as lc

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: False)

    created = _create(ui_client, name="Sumar", script="def run(a, b):\n    return a+b\n")
    tool_id = created["id"]

    listed = ui_client.get("/api/v1/toolbox/tools").json()
    assert any(t["id"] == tool_id for t in listed)

    got = ui_client.get(f"/api/v1/toolbox/tools/{tool_id}").json()
    assert got["script"].startswith("def run(")

    upd = ui_client.put(
        f"/api/v1/toolbox/tools/{tool_id}", json={"description": "updated"}
    )
    assert upd.status_code == 200 and upd.json()["description"] == "updated"

    deleted = ui_client.delete(f"/api/v1/toolbox/tools/{tool_id}")
    assert deleted.status_code == 200


def test_run_returns_409_when_execution_disabled(ui_client, monkeypatch):
    from rengu_flow.config import local_config as lc

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: False)
    created = _create(ui_client, name="Sumar", script="def run():\n    return 1\n")
    res = ui_client.post(f"/api/v1/toolbox/tools/{created['id']}/run", json={"values": {}})
    assert res.status_code == 409, res.text


def test_enabled_endpoint_reflects_toggle(ui_client, monkeypatch):
    from rengu_flow.config import local_config as lc

    monkeypatch.setattr(lc, "toolbox_enabled", lambda: True)
    assert ui_client.get("/api/v1/toolbox/enabled").json() == {"enabled": True}


def test_traversal_tool_id_is_rejected(ui_client, ui_data_tmp):
    """C1 regression: tool_id containing '..' must not escape the toolbox dir."""
    from rengu_flow_ui import settings

    # Create a legitimate tool so the data dir is populated.
    _create(ui_client, name="Safe Tool")

    # Confirm data dir exists and has content.
    data_dir = settings.ui_data_dir()
    assert data_dir.is_dir()

    # Direct unit-level guard: tool_dir("..") must raise KeyError.
    with pytest.raises(KeyError):
        toolbox.tool_dir("..")

    # HTTP-level: Starlette's TestClient may normalize %2e%2e before the route
    # receives it, so the 404 may come from path normalization rather than our
    # guard — either way the traversal is blocked and the data dir is intact.
    delete_res = ui_client.delete("/api/v1/toolbox/tools/%2e%2e")
    assert delete_res.status_code == 404

    get_res = ui_client.get("/api/v1/toolbox/tools/%2e%2e")
    assert get_res.status_code == 404

    # The data dir must be intact (toolbox dir, staging, logs, jobs.db survive).
    assert data_dir.is_dir()
    assert (data_dir / "toolbox").is_dir()
    assert (data_dir / "staging").is_dir()
    assert (data_dir / "logs").is_dir()
    assert (data_dir / "jobs.db").is_file()
