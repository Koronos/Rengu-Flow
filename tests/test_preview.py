"""Tests for preview config helpers (no GPU / no pipeline call)."""

import toml

from rengu_flow.utils.preview import (
    normalize_preview_prompts,
    previews_configured,
    reload_preview_config,
    should_run_previews,
)


def test_reload_preview_config_replaces_section_in_place(tmp_path):
    cfg = {"preview": {"prompts": ["old"], "preview_every_n_steps": 50}, "epochs": 5}
    path = tmp_path / "train.toml"
    path.write_text(
        toml.dumps(
            {"preview": {"prompts": ["new", "second"], "preview_every_n_steps": 10, "enabled": True}}
        ),
        encoding="utf-8",
    )
    assert reload_preview_config(cfg, path) is True
    assert cfg["preview"]["prompts"] == ["new", "second"]
    assert cfg["preview"]["preview_every_n_steps"] == 10
    # Disabling live: enabled=false in the file -> previews stop being configured.
    path.write_text(toml.dumps({"preview": {"prompts": ["x"], "enabled": False}}), encoding="utf-8")
    assert reload_preview_config(cfg, path) is True
    assert previews_configured(cfg) is False


def test_reload_preview_config_missing_section_becomes_empty(tmp_path):
    cfg = {"preview": {"prompts": ["old"]}}
    path = tmp_path / "train.toml"
    path.write_text(toml.dumps({"epochs": 3}), encoding="utf-8")  # no [preview]
    assert reload_preview_config(cfg, path) is True
    assert cfg["preview"] == {}


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

    from rengu_flow.utils.preview import run_previews

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

    with patch("rengu_flow.utils.preview.is_main_process", return_value=True):
        with patch("rengu_flow.utils.preview._dist_barrier"):
            with patch("rengu_flow.utils.preview.empty_cuda_cache"):
                run_previews(model, config, tb, step=5)

    model.prepare_preview_memory.assert_called_once()
    model.generate_preview_image.assert_called_once()
    model.restore_after_preview.assert_called_once()
    tb.add_image.assert_called_once()


def test_run_previews_cosmos_skips_when_pipeline_stages_not_one(capsys):
    from unittest.mock import MagicMock, patch

    from rengu_flow.utils.preview import run_previews

    model = MagicMock()
    model.name = "cosmos_predict2"
    model.generate_preview_image = MagicMock()

    config = {
        "pipeline_stages": 2,
        "preview": {"prompts": ["test scene"]},
    }

    with patch("rengu_flow.utils.preview.is_main_process", return_value=True):
        with patch("rengu_flow.utils.preview._dist_barrier"):
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


def test_forced_preview_ignores_enabled_flag():
    # An explicit (forced) preview runs even when previews are disabled, as long as there
    # are prompts to render — the signal must not be silently consumed.
    disabled = {"preview": {"prompts": ["a"], "enabled": False, "preview_every_n_steps": 100}}
    assert should_run_previews(disabled, 7, 1, forced=True) is True
    assert should_run_previews(disabled, 7, 1, forced=False) is False  # disabled => no schedule
    # Nothing to render without prompts.
    assert should_run_previews({"preview": {"enabled": True}}, 7, 1, forced=True) is False
