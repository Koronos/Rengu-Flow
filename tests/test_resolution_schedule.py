"""Tests for the staged multi-resolution schedule (resolution_schedule)."""

from __future__ import annotations

import pytest

from rengu_flow.data.dataset import Dataset, parse_resolution_schedule
from rengu_flow.data.dataset_config import (
    DatasetConfigError,
    validate_dataset_config_for_real_data,
)


# --- parse_resolution_schedule ------------------------------------------------


def test_parse_absent_or_disabled_is_inactive():
    assert parse_resolution_schedule({}) == (False, [], [])
    assert parse_resolution_schedule(
        {"resolution_schedule": {"enabled": False, "stage": [{"resolutions": [512], "fraction": 1}]}}
    ) == (False, [], [])


def test_parse_normalizes_fractions_and_builds_cum():
    active, stages, cum = parse_resolution_schedule(
        {
            "resolution_schedule": {
                "enabled": True,
                "stage": [
                    {"resolutions": [512], "fraction": 1},
                    {"resolutions": [768], "fraction": 1},
                    {"resolutions": [1024], "fraction": 2},
                ],
            }
        }
    )
    assert active is True
    assert [s[0] for s in stages] == [frozenset({512}), frozenset({768}), frozenset({1024})]
    # fractions normalized to sum 1.0 (1/4, 1/4, 2/4)
    assert [pytest.approx(s[1]) for s in stages] == [0.25, 0.25, 0.5]
    assert cum == [pytest.approx(0.25), pytest.approx(0.5), pytest.approx(1.0)]


def test_parse_accepts_stages_alias_and_scalar_resolution():
    active, stages, _ = parse_resolution_schedule(
        {"resolution_schedule": {"enabled": True, "stages": [{"resolution": 768, "fraction": 1}]}}
    )
    assert active is True
    assert stages[0][0] == frozenset({768})


def test_parse_drops_stages_with_nonpositive_fraction():
    active, stages, _ = parse_resolution_schedule(
        {
            "resolution_schedule": {
                "enabled": True,
                "stage": [
                    {"resolutions": [512], "fraction": 0},
                    {"resolutions": [768], "fraction": 1},
                ],
            }
        }
    )
    assert active is True
    assert [s[0] for s in stages] == [frozenset({768})]


# --- stage selection math -----------------------------------------------------


def _schedule_dataset(stages_cfg, target):
    """Build a bare Dataset wired only with the fields _stage_for_step needs."""
    ds = object.__new__(Dataset)
    ds.schedule_active, ds._schedule_stages, ds._schedule_cum_frac = parse_resolution_schedule(
        {"resolution_schedule": {"enabled": True, "stage": stages_cfg}}
    )
    ds._schedule_target = target
    return ds


def test_stage_for_step_thirds():
    ds = _schedule_dataset(
        [
            {"resolutions": [512], "fraction": 0.33},
            {"resolutions": [768], "fraction": 0.33},
            {"resolutions": [1024], "fraction": 0.34},
        ],
        target=90,
    )
    assert ds._stage_for_step(1) == 0
    assert ds._stage_for_step(30) == 0
    assert ds._stage_for_step(31) == 1
    assert ds._stage_for_step(60) == 1
    assert ds._stage_for_step(61) == 2
    assert ds._stage_for_step(90) == 2
    # past the end clamps to the last stage
    assert ds._stage_for_step(10_000) == 2
    assert ds._active_resolutions_for_stage(0) == frozenset({512})
    assert ds._active_resolutions_for_stage(2) == frozenset({1024})


def test_stage_for_step_without_target_is_stage_zero():
    ds = _schedule_dataset([{"resolutions": [512], "fraction": 1}], target=None)
    assert ds._stage_for_step(123) == 0


# --- iteration-order filtering / set_epoch stage switching --------------------


class _FakeBucket:
    def __init__(self, resolution, n):
        self.resolution = resolution
        self._n = n
        self.epoch = None

    def __len__(self):
        return self._n

    def set_epoch(self, epoch):
        self.epoch = epoch


def _schedule_dataset_with_buckets(stages_cfg, target, current_step):
    ds = _schedule_dataset(stages_cfg, target)
    ds.dataset_config = {}
    ds.buckets = [_FakeBucket(512, 4), _FakeBucket(768, 4), _FakeBucket(1024, 4)]
    ds.post_init_called = True
    ds.current_step = current_step
    ds._active_stage = ds._stage_for_step(current_step)
    ds.iteration_order = ds._build_iteration_order(
        ds._active_resolutions_for_stage(ds._active_stage)
    )
    return ds


def test_build_iteration_order_filters_to_active_resolution():
    stages = [
        {"resolutions": [512], "fraction": 0.33},
        {"resolutions": [768], "fraction": 0.33},
        {"resolutions": [1024], "fraction": 0.34},
    ]
    ds = _schedule_dataset_with_buckets(stages, target=90, current_step=1)
    # Stage 0 -> only bucket index 0 (resolution 512) appears.
    assert {b for b, _ in ds.iteration_order} == {0}
    assert len(ds.iteration_order) == 4


def test_set_epoch_switches_stage_when_progress_crosses_boundary():
    stages = [
        {"resolutions": [512], "fraction": 0.33},
        {"resolutions": [768], "fraction": 0.33},
        {"resolutions": [1024], "fraction": 0.34},
    ]
    ds = _schedule_dataset_with_buckets(stages, target=90, current_step=1)
    assert {b for b, _ in ds.iteration_order} == {0}

    # Advance into stage 1 and trigger an epoch rollover.
    ds.current_step = 40
    ds.set_epoch(5)
    assert ds._active_stage == 1
    assert {b for b, _ in ds.iteration_order} == {1}  # resolution 768
    assert all(b.epoch == 5 for b in ds.buckets)  # epoch still propagated

    # Advance into the final stage.
    ds.current_step = 80
    ds.set_epoch(9)
    assert ds._active_stage == 2
    assert {b for b, _ in ds.iteration_order} == {2}  # resolution 1024


def test_update_active_stage_switches_by_step_within_one_epoch():
    """Step-accurate switching: stage changes mid-epoch without an epoch rollover."""
    stages = [
        {"resolutions": [512], "fraction": 0.33},
        {"resolutions": [768], "fraction": 0.33},
        {"resolutions": [1024], "fraction": 0.34},
    ]
    ds = _schedule_dataset_with_buckets(stages, target=90, current_step=1)
    assert {b for b, _ in ds.iteration_order} == {0}

    # Still in stage 0 -> no change reported.
    assert ds.update_active_stage(20) is False
    assert ds._active_stage == 0

    # Cross into stage 1 -> change reported and order rebuilt (no set_epoch call).
    assert ds.update_active_stage(40) is True
    assert ds._active_stage == 1
    assert {b for b, _ in ds.iteration_order} == {1}

    # Cross into stage 2.
    assert ds.update_active_stage(80) is True
    assert {b for b, _ in ds.iteration_order} == {2}

    # Same stage again -> no change.
    assert ds.update_active_stage(85) is False


def test_update_active_stage_noop_when_schedule_inactive():
    ds = object.__new__(Dataset)
    ds.schedule_active = False
    ds.post_init_called = True
    assert ds.update_active_stage(123) is False


def test_loader_refresh_for_step_restarts_only_on_stage_change():
    """PipelineDataLoader.refresh_for_step restarts iteration iff the stage changed."""
    from rengu_flow.data.loader import PipelineDataLoader

    class _FakeDataset:
        def __init__(self):
            self.changes = {40, 80}  # steps at which the stage flips

        def update_active_stage(self, step):
            return step in self.changes

    loader = object.__new__(PipelineDataLoader)
    loader.dataset = _FakeDataset()
    calls = []
    loader._restart_iteration = lambda: calls.append(True)

    loader.refresh_for_step(1)
    loader.refresh_for_step(39)
    assert calls == []  # no boundary crossed yet
    loader.refresh_for_step(40)
    loader.refresh_for_step(80)
    assert len(calls) == 2  # restarted once per stage change


def test_build_iteration_order_falls_back_when_stage_empty():
    stages = [{"resolutions": [512], "fraction": 1}]
    ds = _schedule_dataset_with_buckets(stages, target=10, current_step=1)
    # A resolution with no matching bucket -> fall back to all buckets, not empty.
    order = ds._build_iteration_order(frozenset({99999}))
    assert len(order) == 12  # all three fake buckets (4 each)


# --- validation ---------------------------------------------------------------


def _base_dataset_config(extra):
    cfg = {
        "resolutions": [512, 768, 1024],
        "directory": [{"path": "/data", "num_repeats": 1}],
    }
    cfg.update(extra)
    return cfg


def test_validation_accepts_valid_schedule():
    cfg = _base_dataset_config(
        {
            "resolution_schedule": {
                "enabled": True,
                "stage": [
                    {"resolutions": [512], "fraction": 0.5},
                    {"resolutions": [768, 1024], "fraction": 0.5},
                ],
            }
        }
    )
    validate_dataset_config_for_real_data(cfg)  # should not raise


def test_validation_rejects_unknown_resolution():
    cfg = _base_dataset_config(
        {"resolution_schedule": {"enabled": True, "stage": [{"resolutions": [333], "fraction": 1}]}}
    )
    with pytest.raises(DatasetConfigError, match="not in the dataset's resolutions"):
        validate_dataset_config_for_real_data(cfg)


def test_validation_rejects_enabled_without_stages():
    cfg = _base_dataset_config({"resolution_schedule": {"enabled": True}})
    with pytest.raises(DatasetConfigError, match="no .*stage.* entries"):
        validate_dataset_config_for_real_data(cfg)


def test_validation_rejects_missing_fraction():
    cfg = _base_dataset_config(
        {"resolution_schedule": {"enabled": True, "stage": [{"resolutions": [512]}]}}
    )
    with pytest.raises(DatasetConfigError, match="must set 'fraction'"):
        validate_dataset_config_for_real_data(cfg)


def test_validation_disabled_schedule_is_ignored():
    cfg = _base_dataset_config(
        {"resolution_schedule": {"enabled": False, "stage": [{"resolutions": [333], "fraction": 1}]}}
    )
    validate_dataset_config_for_real_data(cfg)  # disabled => not checked


# --- UI form round-trip -------------------------------------------------------


def test_form_round_trips_resolution_schedule():
    import json

    from rengu_flow_ui.dataset_form import form_to_toml, loads_for_training, parse_toml

    schedule = {
        "enabled": True,
        "stage": [
            {"resolutions": [512], "fraction": 0.33},
            {"resolutions": [768], "fraction": 0.33},
            {"resolutions": [1024], "fraction": 0.34},
        ],
    }
    form = {
        "resolutions": [512, 768, 1024],
        "resolution_schedule": json.dumps(schedule),
        "_directories": [{"path": "/data", "num_repeats": 1}],
    }
    toml_text = form_to_toml(form)
    cfg = loads_for_training(toml_text)
    assert cfg["resolution_schedule"] == schedule
    # And parsing back into the form yields an equivalent JSON string.
    form_back = parse_toml(toml_text)
    assert json.loads(form_back["resolution_schedule"]) == schedule
