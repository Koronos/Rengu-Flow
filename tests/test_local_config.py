"""Tests for renga.local.toml loading."""

import os
from pathlib import Path

from renga_flow.config.local_config import (
    apply_local_config_to_environ,
    init_local_config_file,
    load_local_config,
    parse_local_config_dict,
)


def test_parse_local_config_applies_ui_and_training_env():
    root = Path("/tmp/renga-root")
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
    from renga_flow.config.local_config import LOCAL_CONFIG_EXAMPLE

    example = tmp_path / LOCAL_CONFIG_EXAMPLE
    example.write_text("[ui]\nhost = \"127.0.0.1\"\nport = 8765\n", encoding="utf-8")
    dest = init_local_config_file(root=tmp_path)
    assert dest.is_file()
    assert init_local_config_file(root=tmp_path) == dest


def test_load_local_config_missing_returns_none(tmp_path):
    assert load_local_config(root=tmp_path) is None


def test_apply_local_config_to_environ(tmp_path, monkeypatch):
    path = tmp_path / "renga.local.toml"
    path.write_text(
        """
[ui]
host = "10.0.0.1"
port = 8888
data_dir = ".renga-flow-ui"
""",
        encoding="utf-8",
    )
    for key in ("RENGA_FLOW_UI_HOST", "RENGA_FLOW_UI_PORT", "RENGA_FLOW_UI_DATA"):
        monkeypatch.delenv(key, raising=False)
    cfg = load_local_config(path=path, root=tmp_path)
    assert cfg is not None
    apply_local_config_to_environ(cfg)
    assert os.environ["RENGA_FLOW_UI_HOST"] == "10.0.0.1"
