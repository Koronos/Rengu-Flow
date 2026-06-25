"""Tests for the torch-free pre-caching step estimate (UI + early training banner)."""

from __future__ import annotations

from rengu_flow.data.step_estimate import estimate_total_steps


def test_original_worked_example_350_per_epoch():
    """5 folders x 50 images, res 500/700/1000 at batch 2/2/1, schedule 40/40/20, no aug."""
    dataset_config = {
        "resolutions": [500, 700, 1000],
        "directory": [{"path": f"/d{i}", "num_repeats": 1} for i in range(5)],
        "resolution_schedule": {
            "enabled": True,
            "stage": [
                {"resolutions": [500, 1000], "fraction": 0.4},
                {"resolutions": [700, 1000], "fraction": 0.4},
                {"resolutions": [1000], "fraction": 0.2},
            ],
        },
    }
    training_config = {
        "epochs": 1,
        "micro_batch_size_per_gpu": {500: 2, 700: 2, 1000: 1},
    }
    counts = {f"/d{i}": 50 for i in range(5)}
    out = estimate_total_steps(dataset_config, training_config, counts)
    # 1.0*(250//1) + 0.4*(250//2) + 0.4*(250//2) = 250 + 50 + 50 = 350
    assert out["images_per_resolution"] == 250
    assert out["steps_per_epoch"] == 350
    assert out["total_steps"] == 350


def test_users_config_18900_at_18_epochs_with_aug_and_cap():
    """User's real config: max_images=50 (5 folders), branches_per_image=2 (x3 aug),
    res 512/768/1024 at batch 2/2/1, schedule 40/40/20, 18 epochs -> 18900 (not 56700)."""
    dataset_config = {
        "resolutions": [512, 768, 1024],
        "max_images": 50,
        "directory": [{"path": f"/d{i}", "num_repeats": 1} for i in range(5)],
        # Global augmentation lives under the nested [dataset.augmentation] table (what the
        # UI writes and the trainer reads), not a top-level "augmentation" key.
        "dataset": {"augmentation": {"enabled": True, "branches_per_image": 2}},
        "resolution_schedule": {
            "enabled": True,
            "stage": [
                {"resolutions": [512, 1024], "fraction": 0.4},
                {"resolutions": [1024, 768], "fraction": 0.4},
                {"resolutions": [1024], "fraction": 0.2},
            ],
        },
    }
    training_config = {
        "epochs": 18,
        "micro_batch_size_per_gpu": {512: 2, 768: 2, 1024: 1},
    }
    # The 1200-image folder is capped to 50 like the rest; even a folder smaller than the cap
    # repeats up to it, so all five serve 50 -> 250 base, x3 aug = 750 per resolution.
    counts = {"/d0": 1200, "/d1": 50, "/d2": 50, "/d3": 50, "/d4": 50}
    out = estimate_total_steps(dataset_config, training_config, counts)
    assert out["images_per_resolution"] == 750
    # 1.0*(750//1) + 0.4*(750//2) + 0.4*(750//2) = 750 + 150 + 150 = 1050
    assert out["steps_per_epoch"] == 1050
    assert out["total_steps"] == 18900


def test_augmentation_top_level_fallback():
    """A top-level `augmentation` key (not nested under [dataset]) is still honored."""
    out = estimate_total_steps(
        {
            "resolutions": [512],
            "directory": [{"path": "/d", "num_repeats": 1}],
            "augmentation": {"enabled": True, "branches_per_image": 2},
        },
        {"epochs": 1, "micro_batch_size_per_gpu": 1},
        {"/d": 10},
    )
    assert out["images_per_resolution"] == 30  # 10 base x 3 (original + 2 branches)


def test_ui_form_round_trip_estimate(tmp_path):
    """End-to-end through the UI serialization: a dataset form with nested global augmentation
    survives form_to_toml -> loads_for_training and is counted (regression for the 6300 vs 18900
    bug, where the nested [dataset.augmentation] was dropped)."""
    import json

    from rengu_flow_ui.dataset_form import form_to_toml, loads_for_training

    form = {
        "resolutions": [512, 768, 1024],
        "max_images": 50,
        "_dataset_augmentation": json.dumps(
            {"enabled": True, "preset": "easy", "branches_per_image": 2}
        ),
        "resolution_schedule": json.dumps(
            {
                "enabled": True,
                "stage": [
                    {"resolutions": [512, 1024], "fraction": 0.4},
                    {"resolutions": [1024, 768], "fraction": 0.4},
                    {"resolutions": [1024], "fraction": 0.2},
                ],
            }
        ),
        "_directories": [{"path": f"/d{i}", "num_repeats": 1} for i in range(5)],
    }
    dataset_config = loads_for_training(form_to_toml(form))
    out = estimate_total_steps(
        dataset_config,
        {"epochs": 18, "micro_batch_size_per_gpu": {512: 2, 768: 2, 1024: 1}},
        {f"/d{i}": 50 for i in range(5)},
    )
    assert out["images_per_resolution"] == 750
    assert out["steps_per_epoch"] == 1050
    assert out["total_steps"] == 18900


def test_no_schedule_counts_all_resolutions_fully():
    dataset_config = {
        "resolutions": [512, 1024],
        "directory": [{"path": "/d", "num_repeats": 1}],
    }
    training_config = {"epochs": 2, "micro_batch_size_per_gpu": 2}
    out = estimate_total_steps(dataset_config, training_config, {"/d": 100})
    # (100//2) + (100//2) = 50 + 50 = 100 per epoch, x2 epochs
    assert out["steps_per_epoch"] == 100
    assert out["total_steps"] == 200


def test_max_steps_caps_total():
    dataset_config = {"resolutions": [512], "directory": [{"path": "/d", "num_repeats": 1}]}
    training_config = {"epochs": 100, "micro_batch_size_per_gpu": 1, "max_steps": 500}
    out = estimate_total_steps(dataset_config, training_config, {"/d": 100})
    assert out["total_steps"] == 500


def test_num_repeats_and_grad_accum_and_world():
    dataset_config = {"resolutions": [512], "directory": [{"path": "/d", "num_repeats": 3}]}
    training_config = {
        "epochs": 1,
        "micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
    }
    # 100 images x num_repeats 3 = 300; global batch = 2*2*2(world) = 8 -> 300//8 = 37
    out = estimate_total_steps(dataset_config, training_config, {"/d": 100}, world_size=2)
    assert out["images_per_resolution"] == 300
    assert out["steps_per_epoch"] == 37


def test_empty_counts_yields_zero():
    out = estimate_total_steps(
        {"resolutions": [512], "directory": [{"path": "/d", "num_repeats": 1}]},
        {"epochs": 5, "micro_batch_size_per_gpu": 1},
        {},
    )
    assert out["steps_per_epoch"] == 0
    assert out["total_steps"] == 0
