"""Unit tests for rengu_track lineage capture and the system-metrics sampler."""

from unittest.mock import patch

import pytest

from rengu_track import lineage
from rengu_track.sampler import SystemSampler

pytestmark = pytest.mark.no_ui_db


# --- lineage ----------------------------------------------------------------------------------


def test_command_shape():
    cmd = lineage.command()
    assert isinstance(cmd["argv"], list)
    assert cmd["python"]
    assert cmd["executable"]


def test_environment_is_dict():
    env = lineage.environment()
    assert isinstance(env, dict)
    # torch is installed in the test venv, so it should be captured.
    assert "torch" in env


def test_capture_bundle_keys_and_torch_free():
    bundle = lineage.capture()
    assert set(bundle) == {"git", "command", "environment"}
    # The repo is a git worktree, so git lineage resolves.
    assert bundle["git"]["available"] is True
    assert len(bundle["git"]["commit"]) >= 7


def test_git_lineage_non_repo(tmp_path):
    info = lineage.git_lineage(str(tmp_path))
    assert info["available"] is False


def test_hardware_shape():
    hw = lineage.hardware()
    # torch present → available True; GPU presence is environment-dependent (not asserted).
    assert hw["available"] is True
    assert "torch_version" in hw


# --- sampler ----------------------------------------------------------------------------------


class _FakeSink:
    def __init__(self):
        self.scalars = []
        self.summaries = []

    def scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def summary(self, metrics):
        self.summaries.append(metrics)


_STATS = {
    "summary": {
        "cpu_percent": 30.0,
        "ram_used_gb": 12.0,
        "gpus": [
            {"util_percent": 80.0, "vram_used_gb": 6.0, "temp_c": 65.0},
        ],
    },
    "detail": {"gpus": {"devices": [{"power_w": 120.0}]}},
}


def test_sampler_sample_once_pushes_system_scalars():
    sink = _FakeSink()
    sampler = SystemSampler(sink, interval_sec=10, step_fn=lambda: 7)
    with patch("rengu_track.sampler.collect_system_stats", return_value=_STATS):
        sampler._sample_once()

    tags = {tag: (value, step) for tag, value, step in sink.scalars}
    assert tags["system/cpu_percent"] == (30.0, 7)
    assert tags["system/ram_used_gb"] == (12.0, 7)
    assert tags["system/gpu_util_percent"] == (80.0, 7)
    assert tags["system/vram_used_gb"] == (6.0, 7)
    assert tags["system/gpu_temp_c"] == (65.0, 7)
    assert tags["system/gpu_power_w"] == (120.0, 7)


def test_sampler_aggregates_peak_and_mean():
    sink = _FakeSink()
    sampler = SystemSampler(sink, interval_sec=10)
    hi = {
        "summary": {"gpus": [{"util_percent": 100.0, "vram_used_gb": 8.0}]},
        "detail": {"gpus": {"devices": []}},
    }
    with patch("rengu_track.sampler.collect_system_stats", side_effect=[_STATS, hi]):
        sampler._sample_once()  # vram 6, util 80
        sampler._sample_once()  # vram 8, util 100

    sampler.stop()  # no thread started → just flushes aggregates
    assert sink.summaries
    agg = sink.summaries[-1]
    assert agg["system/peak_vram_gb"] == 8.0
    assert agg["system/mean_gpu_util_percent"] == 90.0


def test_sampler_collect_failure_is_silent():
    sink = _FakeSink()
    sampler = SystemSampler(sink)
    with patch("rengu_track.sampler.collect_system_stats", side_effect=RuntimeError("boom")):
        sampler._sample_once()  # must not raise
    assert sink.scalars == []
