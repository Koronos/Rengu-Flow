"""rengu.local.toml read/write via tomlkit (settings_store)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rengu_flow_ui import settings_store
from rengu_flow_ui.settings_store import SettingsError

EXAMPLE = """\
# top comment
[ui]
host = "127.0.0.1"
port = 8765
public = false
data_dir = "data"
# token comment
# token = "change-me"

[maintenance]
enabled = false
allow_pip = false

[training]
num_gpus = 1
master_port = 29500
extra_args = ""

[training.env]
# keep me
NCCL_P2P_DISABLE = "1"
"""


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "rengu.local.toml"
    p.write_text(EXAMPLE, encoding="utf-8")
    return p


def test_read_settings_groups_fields(cfg_file: Path) -> None:
    s = settings_store.read_settings(cfg_file)
    assert s["exists"] is True
    assert s["editable"]["training"]["num_gpus"] == 1
    assert s["editable"]["training"]["env"] == {"NCCL_P2P_DISABLE": "1"}
    assert s["editable"]["maintenance"] == {"enabled": False}
    assert s["restartRequired"]["ui"]["public"] is False
    assert s["restartRequired"]["ui"]["token"] is None
    assert s["readOnly"]["ui"] == {"host": "127.0.0.1", "port": 8765, "data_dir": "data"}
    assert s["readOnly"]["maintenance"] == {"allow_pip": False}
    assert s["readOnly"]["toolbox"] == {"enabled": False}


def test_read_settings_missing_file_uses_defaults(tmp_path: Path) -> None:
    s = settings_store.read_settings(tmp_path / "absent.toml")
    assert s["exists"] is False
    assert s["editable"]["training"]["num_gpus"] == 1
    assert s["readOnly"]["ui"]["port"] == 8765


def test_write_preserves_comments_and_untouched_keys(cfg_file: Path) -> None:
    settings_store.write_settings({"training": {"num_gpus": 2}}, cfg_file)
    text = cfg_file.read_text(encoding="utf-8")
    assert "# top comment" in text
    assert "# keep me" in text
    assert "num_gpus = 2" in text
    # untouched key kept
    assert 'master_port = 29500' in text


def test_write_replaces_env_table(cfg_file: Path) -> None:
    out = settings_store.write_settings(
        {"training": {"env": {"FOO": "bar"}}}, cfg_file
    )
    assert out["editable"]["training"]["env"] == {"FOO": "bar"}
    assert "NCCL_P2P_DISABLE" not in cfg_file.read_text(encoding="utf-8")


def test_write_token_empty_string_clears_key(cfg_file: Path) -> None:
    out = settings_store.write_settings({"ui": {"token": ""}}, cfg_file)
    assert out["restartRequired"]["ui"]["token"] is None


def test_write_rejects_bad_num_gpus(cfg_file: Path) -> None:
    with pytest.raises(SettingsError):
        settings_store.write_settings({"training": {"num_gpus": 0}}, cfg_file)


def test_write_rejects_bad_port(cfg_file: Path) -> None:
    with pytest.raises(SettingsError):
        settings_store.write_settings({"training": {"master_port": 70000}}, cfg_file)


def test_write_rejects_non_editable_key(cfg_file: Path) -> None:
    with pytest.raises(SettingsError):
        settings_store.write_settings({"ui": {"host": "0.0.0.0"}}, cfg_file)


def test_write_creates_file_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "new.toml"
    out = settings_store.write_settings({"maintenance": {"enabled": True}}, target)
    assert target.is_file()
    assert out["editable"]["maintenance"]["enabled"] is True


def test_apply_maintenance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RENGUFLOW_MAINTENANCE", raising=False)
    monkeypatch.delenv("RENGUFLOW_MAINTENANCE_ALLOW_PIP", raising=False)
    settings = {
        "editable": {"maintenance": {"enabled": True}},
        "readOnly": {"maintenance": {"allow_pip": False}},
    }
    settings_store.apply_maintenance_env(settings)
    import os

    assert os.environ["RENGUFLOW_MAINTENANCE"] == "1"
    assert os.environ["RENGUFLOW_MAINTENANCE_ALLOW_PIP"] == "0"


def test_write_rejects_allow_pip(cfg_file: Path) -> None:
    # allow_pip is read-only — patching it must be rejected, not silently written.
    with pytest.raises(SettingsError):
        settings_store.write_settings({"maintenance": {"allow_pip": True}}, cfg_file)
