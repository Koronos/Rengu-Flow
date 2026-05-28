"""Augmentation catalog exposed to the UI."""

from renga_flow.data.augmentation.ui_schema import get_augmentation_catalog


def test_strategy_parameters_include_help() -> None:
    catalog = get_augmentation_catalog()
    flip = next(s for s in catalog["strategies"] if s["name"] == "horizontal_flip")
    assert flip.get("help")
    prob = next(p for p in flip["parameters"] if p["path"] == "probability")
    assert prob.get("help")


def test_augmentation_catalog_includes_mvp_strategies() -> None:
    catalog = get_augmentation_catalog()
    names = {s["name"] for s in catalog["strategies"]}
    assert "color_jitter" in names
    assert "horizontal_flip" in names
    assert all(s["implemented"] for s in catalog["strategies"])


def test_augmentation_catalog_presets_from_training_code() -> None:
    catalog = get_augmentation_catalog()
    preset_names = {p["name"] for p in catalog["presets"]}
    assert "easy" in preset_names
    assert "photo_safe" in preset_names
    easy = next(p for p in catalog["presets"] if p["name"] == "easy")
    assert easy["available"] is True
    assert "color_jitter" in easy["strategies"]
