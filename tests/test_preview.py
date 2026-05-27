"""Tests for preview config helpers (no GPU / no pipeline call)."""

from renga_flow.utils.preview import (
    normalize_preview_prompts,
    previews_configured,
    should_run_previews,
)


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


def test_run_previews_cosmos_dispatches_to_generate_preview_image():
    from unittest.mock import MagicMock, patch

    from PIL import Image

    from renga_flow.utils.preview import run_previews

    model = MagicMock()
    model.name = "cosmos_predict2"
    model.prepare_preview_memory = MagicMock()
    model.generate_preview_image = MagicMock(return_value=Image.new("RGB", (8, 8)))
    model.restore_after_preview = MagicMock()

    config = {
        "pipeline_stages": 1,
        "preview": {"prompts": ["test scene"], "seed": 0},
    }
    tb = MagicMock()

    with patch("renga_flow.utils.preview.is_main_process", return_value=True):
        with patch("renga_flow.utils.preview._dist_barrier"):
            with patch("renga_flow.utils.preview.empty_cuda_cache"):
                run_previews(model, config, tb, step=5)

    model.prepare_preview_memory.assert_called_once()
    model.generate_preview_image.assert_called_once()
    model.restore_after_preview.assert_called_once()
    tb.add_image.assert_called_once()


def test_run_previews_cosmos_skips_when_pipeline_stages_not_one(capsys):
    from unittest.mock import MagicMock, patch

    from renga_flow.utils.preview import run_previews

    model = MagicMock()
    model.name = "cosmos_predict2"
    model.generate_preview_image = MagicMock()

    config = {
        "pipeline_stages": 2,
        "preview": {"prompts": ["test scene"]},
    }

    with patch("renga_flow.utils.preview.is_main_process", return_value=True):
        with patch("renga_flow.utils.preview._dist_barrier"):
            run_previews(model, config, None, step=1)

    model.generate_preview_image.assert_not_called()
    assert "pipeline_stages = 1" in capsys.readouterr().out


def test_should_run_previews_schedules():
    config = {"preview": {"prompts": ["test"], "preview_every_n_steps": 100}}
    assert should_run_previews(config, 100, 1)
    assert not should_run_previews(config, 99, 1)
    assert should_run_previews(config, 5, 2, finished_epoch=True, forced=False) is False
    config["preview"]["preview_every_n_epochs"] = 1
    assert should_run_previews(config, 5, 1, finished_epoch=True)
