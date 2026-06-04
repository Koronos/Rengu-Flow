"""Tests for rengu.local.toml loading."""

import os
from pathlib import Path

from rengu_flow.config.local_config import (
    apply_local_config_to_environ,
    init_local_config_file,
    load_local_config,
    parse_local_config_dict,
)


def test_parse_local_config_applies_ui_and_training_env():
    root = Path("/tmp/rengu-root")
    data = {
        "ui": {"host": "0.0.0.0", "port": 9000, "data_dir": "ui-data"},
        "maintenance": {"enabled": True, "allow_pip": True},
        "training": {
            "num_gpus": 2,
            "master_port": 29600,
            "extra_args": "--validate-only",
            "env": {"NCCL_P2P_DISABLE": "1"},
        },
    }
    cfg = parse_local_config_dict(data, root=root)
    assert cfg.ui.port == 9000
    assert cfg.training.num_gpus == 2
    assert cfg.training.env["NCCL_P2P_DISABLE"] == "1"
    assert cfg.ui_data_dir() == (root / "ui-data").resolve()


def test_init_local_config_file(tmp_path):
    from rengu_flow.config.local_config import LOCAL_CONFIG_EXAMPLE

    example = tmp_path / LOCAL_CONFIG_EXAMPLE
    example.write_text("[ui]\nhost = \"127.0.0.1\"\nport = 8765\n", encoding="utf-8")
    dest = init_local_config_file(root=tmp_path)
    assert dest.is_file()
    assert init_local_config_file(root=tmp_path) == dest


def test_ensure_local_config_file_creates_from_example(tmp_path):
    from rengu_flow.config.local_config import (
        LOCAL_CONFIG_EXAMPLE,
        ensure_local_config_file,
        local_config_path,
    )

    (tmp_path / LOCAL_CONFIG_EXAMPLE).write_text(
        '[ui]\nport = 8765\n', encoding="utf-8"
    )
    dest = ensure_local_config_file(root=tmp_path)
    assert dest == local_config_path(tmp_path)
    assert dest.is_file()
    # Idempotent: an existing file is left untouched.
    dest.write_text("# edited\n", encoding="utf-8")
    ensure_local_config_file(root=tmp_path)
    assert dest.read_text(encoding="utf-8") == "# edited\n"


def test_ensure_local_config_file_falls_back_to_defaults(tmp_path):
    # No example present (user went hardcore): still produces a valid config with the UI port
    # and a training section.
    import toml

    from rengu_flow.config.local_config import ensure_local_config_file

    dest = ensure_local_config_file(root=tmp_path)
    assert dest is not None and dest.is_file()
    data = toml.load(dest)
    assert data["ui"]["port"] == 8765
    assert "training" in data


def test_load_local_config_missing_returns_none(tmp_path):
    assert load_local_config(root=tmp_path) is None


def test_apply_local_config_to_environ(tmp_path, monkeypatch):
    path = tmp_path / "rengu.local.toml"
    path.write_text(
        """
[ui]
host = "10.0.0.1"
port = 8888
data_dir = ".rengu-flow-ui"
""",
        encoding="utf-8",
    )
    for key in ("RENGU_FLOW_UI_HOST", "RENGU_FLOW_UI_PORT", "RENGU_FLOW_UI_DATA"):
        monkeypatch.delenv(key, raising=False)
    cfg = load_local_config(path=path, root=tmp_path)
    assert cfg is not None
    apply_local_config_to_environ(cfg)
    assert os.environ["RENGU_FLOW_UI_HOST"] == "10.0.0.1"
