from pathlib import Path

from rengu_flow.config import local_config as lc


def test_toolbox_enabled_defaults_false_when_section_absent(tmp_path: Path):
    cfg = lc.parse_local_config_dict({}, root=tmp_path)
    assert cfg.toolbox.enabled is False


def test_toolbox_enabled_reads_truthy_values(tmp_path: Path):
    cfg = lc.parse_local_config_dict({"toolbox": {"enabled": "on"}}, root=tmp_path)
    assert cfg.toolbox.enabled is True

    cfg2 = lc.parse_local_config_dict({"toolbox": {"enabled": True}}, root=tmp_path)
    assert cfg2.toolbox.enabled is True
