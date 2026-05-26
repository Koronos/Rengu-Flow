"""Tests for host system metrics collector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from renga_flow_ui import system_stats


def test_build_summary_from_parts() -> None:
    cpu = {"available": True, "percent": 42.5, "temp_c": 55.0}
    ram = {"available": True, "percent": 60.0, "used_gb": 12.0, "total_gb": 32.0}
    gpus = {
        "available": True,
        "devices": [
            {
                "index": 0,
                "util_percent": 90.0,
                "vram_used_gb": 10.0,
                "vram_total_gb": 24.0,
                "vram_percent": 41.7,
                "temp_c": 70.0,
            }
        ],
    }
    s = system_stats._build_summary(cpu, ram, gpus)
    assert s["cpu_percent"] == 42.5
    assert s["ram_used_gb"] == 12.0
    assert len(s["gpus"]) == 1
    assert s["gpus"][0]["temp_c"] == 70.0


def test_parse_nvidia_smi_line() -> None:
    row = system_stats._parse_nvidia_smi_line("0, NVIDIA A100, 65, 88, 12000, 40960, 45")
    assert row is not None
    assert row["index"] == 0
    assert row["util_percent"] == 88.0
    assert row["vram_total_gb"] == 40.0


@patch("renga_flow_ui.system_stats.shutil.which", return_value=None)
def test_collect_without_nvidia(mock_which: MagicMock) -> None:
    with patch("renga_flow_ui.system_stats._collect_cpu") as mock_cpu:
        with patch("renga_flow_ui.system_stats._collect_ram") as mock_ram:
            mock_cpu.return_value = {"available": True, "percent": 10.0, "temp_c": 40.0}
            mock_ram.return_value = {
                "available": True,
                "percent": 50.0,
                "used_gb": 8.0,
                "total_gb": 16.0,
            }
            out = system_stats.collect_system_stats(sample_cpu=False)
    assert out["summary"]["cpu_percent"] == 10.0
    assert out["summary"]["gpus"] == []
    assert any("nvidia-smi" in w for w in out["detail"]["warnings"])
