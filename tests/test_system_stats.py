"""Tests for host system metrics collector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rengu_track import system_stats


@pytest.fixture(autouse=True)
def _reset_gpu_device_cache():
    """Reset the module-level GPU device cache before and after each test.

    Without a teardown reset, a test that populates the cache (e.g.
    test_reset_device_cache) leaks a fake device list into the shared xdist
    worker process, corrupting any later test — including in other modules —
    that calls list_gpu_devices()/enumerate_devices().
    """
    system_stats.reset_device_cache()
    yield
    system_stats.reset_device_cache()


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


@patch("rengu_track.system_stats.shutil.which", return_value=None)
def test_collect_without_nvidia(mock_which: MagicMock) -> None:
    with patch("rengu_track.system_stats._collect_cpu") as mock_cpu:
        with patch("rengu_track.system_stats._collect_ram") as mock_ram:
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


def test_list_gpu_devices_shape() -> None:
    """Test that list_gpu_devices returns correct dict shape with fake _collect_gpus."""
    system_stats.reset_device_cache()
    with patch("rengu_track.system_stats._collect_gpus") as mock_collect:
        mock_collect.return_value = {
            "available": True,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA A100",
                    "temp_c": 45.0,
                    "util_percent": 50.0,
                    "vram_total_gb": 40.0,
                    "vram_used_gb": 20.0,
                    "vram_percent": 50.0,
                },
                {
                    "index": 1,
                    "name": "NVIDIA H100",
                    "temp_c": 55.0,
                    "util_percent": 75.0,
                    "vram_total_gb": 80.0,
                    "vram_used_gb": 60.0,
                    "vram_percent": 75.0,
                },
            ],
            "backend": "nvidia-smi",
        }
        result = system_stats.list_gpu_devices()

    assert len(result) == 2
    assert result[0]["index"] == 0
    assert result[0]["name"] == "NVIDIA A100"
    assert result[0]["vram_total_gb"] == 40.0
    assert result[1]["index"] == 1
    assert result[1]["name"] == "NVIDIA H100"
    assert result[1]["vram_total_gb"] == 80.0
    # Ensure only three keys are present
    assert set(result[0].keys()) == {"index", "name", "vram_total_gb"}
    assert set(result[1].keys()) == {"index", "name", "vram_total_gb"}


def test_list_gpu_devices_empty() -> None:
    """Test that list_gpu_devices returns [] when no devices available."""
    system_stats.reset_device_cache()
    with patch("rengu_track.system_stats._collect_gpus") as mock_collect:
        mock_collect.return_value = {
            "available": False,
            "devices": [],
            "backend": None,
            "error": "nvidia-smi not found",
        }
        result = system_stats.list_gpu_devices()

    assert result == []


def test_list_gpu_devices_caching() -> None:
    """Test that list_gpu_devices caches and does not re-invoke _collect_gpus."""
    system_stats.reset_device_cache()
    with patch("rengu_track.system_stats._collect_gpus") as mock_collect:
        mock_collect.return_value = {
            "available": True,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA A100",
                    "vram_total_gb": 40.0,
                }
            ],
            "backend": "nvidia-smi",
        }
        # First call
        result1 = system_stats.list_gpu_devices()
        assert len(result1) == 1
        assert mock_collect.call_count == 1

        # Second call (should use cache)
        result2 = system_stats.list_gpu_devices()
        assert len(result2) == 1
        assert mock_collect.call_count == 1  # Still 1, not called again


def test_reset_device_cache() -> None:
    """Test that reset_device_cache forces re-enumeration."""
    system_stats.reset_device_cache()
    with patch("rengu_track.system_stats._collect_gpus") as mock_collect:
        mock_collect.return_value = {
            "available": True,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA A100",
                    "vram_total_gb": 40.0,
                }
            ],
            "backend": "nvidia-smi",
        }
        # First call
        result1 = system_stats.list_gpu_devices()
        assert mock_collect.call_count == 1

        # Reset cache
        system_stats.reset_device_cache()

        # Second call (should re-invoke _collect_gpus)
        result2 = system_stats.list_gpu_devices()
        assert mock_collect.call_count == 2  # Called again after reset


def test_list_gpu_devices_returns_copy_not_cached_object() -> None:
    """A caller mutating the returned list/dicts must not corrupt the module cache."""
    system_stats.reset_device_cache()
    with patch("rengu_track.system_stats._collect_gpus") as mock_collect:
        mock_collect.return_value = {
            "available": True,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA A100",
                    "vram_total_gb": 40.0,
                }
            ],
            "backend": "nvidia-smi",
        }
        result1 = system_stats.list_gpu_devices()
        # Mutate both the returned list and one of its dicts.
        result1.append({"index": 99, "name": "Fake", "vram_total_gb": 1.0})
        result1[0]["name"] = "Mutated"

        result2 = system_stats.list_gpu_devices()

    assert len(result2) == 1
    assert result2[0]["name"] == "NVIDIA A100"
    assert result1 is not result2
    assert result1[0] is not result2[0]
