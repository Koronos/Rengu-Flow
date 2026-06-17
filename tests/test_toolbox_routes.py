import pytest


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
