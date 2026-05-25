"""Tests for preview config helpers (no GPU / no pipeline call)."""

from renga_flow.utils.preview import (
    normalize_preview_prompts,
    previews_configured,
    should_run_previews,
)
from renga_flow.utils.signal_files import SIGNAL_PREVIEW, process_signals


def test_previews_configured_requires_prompts():
    assert not previews_configured({})
    assert previews_configured({"preview": {"prompts": ["a cat"]}})
    assert not previews_configured({"preview": {"prompts": ["x"], "enabled": False}})


def test_normalize_preview_prompts():
    cfg = {
        "prompts": [
            "simple",
            {"name": "portrait", "prompt": "1girl"},
            {"text": "landscape only"},
        ]
    }
    assert normalize_preview_prompts(cfg) == [
        ("prompt_0", "simple"),
        ("portrait", "1girl"),
        ("prompt_2", "landscape only"),
    ]


def test_should_run_previews_schedules():
    config = {"preview": {"prompts": ["test"], "preview_every_n_steps": 100}}
    assert should_run_previews(config, 100, 1)
    assert not should_run_previews(config, 99, 1)
    assert should_run_previews(config, 5, 2, finished_epoch=True, forced=False) is False
    config["preview"]["preview_every_n_epochs"] = 1
    assert should_run_previews(config, 5, 1, finished_epoch=True)


def test_process_signals_preview(tmp_path):
    (tmp_path / SIGNAL_PREVIEW).touch()
    result = process_signals(tmp_path)
    assert result.should_preview
    assert not (tmp_path / SIGNAL_PREVIEW).exists()
