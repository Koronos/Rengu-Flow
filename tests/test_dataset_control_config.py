"""Edit-dataset config rules: validation, pairing at metadata build, preflight."""

from __future__ import annotations

import pytest
from PIL import Image

from rengu_flow.config.preflight import _dataset_directory_issues
from rengu_flow.data.augmentation import AugmentationConfigError, validate_augmentation_for_directory
from rengu_flow.data.control import ControlPairingError
from rengu_flow.data.dataset import DirectoryDataset
from rengu_flow.data.dataset_config import DatasetConfigError, validate_dataset_config_for_real_data

pytestmark = pytest.mark.no_ui_db


def _img(path, size=(64, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (1, 2, 3)).save(path)


def test_augmentation_with_control_path_is_rejected():
    directory = {
        "path": "t",
        "control_path": "c",
        "num_repeats": 1,
        "augmentation": {"enabled": True, "preset": "easy"},
    }
    with pytest.raises(AugmentationConfigError, match="control_path"):
        validate_augmentation_for_directory(directory, {})
    with pytest.raises(DatasetConfigError, match="control_path"):
        validate_dataset_config_for_real_data({"directory": [directory]})


def test_augmentation_without_control_path_still_allowed():
    directory = {"path": "t", "num_repeats": 1, "augmentation": {"enabled": True, "preset": "easy"}}
    validate_augmentation_for_directory(directory, {})


@pytest.mark.parametrize("value", [0, -1, 1.5, "1024", True])
def test_control_resolution_must_be_positive_int(value):
    directory = {"path": "t", "control_path": "c", "num_repeats": 1, "control_resolution": value}
    cfg = {"directory": [directory]}
    with pytest.raises(DatasetConfigError, match="control_resolution"):
        validate_dataset_config_for_real_data(cfg)


def test_control_resolution_valid():
    validate_dataset_config_for_real_data(
        {"directory": [{"path": "t", "control_path": "c", "num_repeats": 1, "control_resolution": 768}]}
    )


def _directory(tmp_path, **extra):
    cfg = {"path": str(tmp_path / "t"), "control_path": str(tmp_path / "c"), "num_repeats": 1, **extra}
    return DirectoryDataset(
        cfg, {"resolutions": [64]}, "m", training_config={"cache_root": str(tmp_path / "cache")}
    )


def test_uncond_fraction_with_control_path_is_rejected(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "c").mkdir()
    with pytest.raises(ValueError, match="uncond_fraction"):
        _directory(tmp_path, uncond_fraction=0.1)


def test_control_resolution_defaults_to_bucket(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "c").mkdir()
    assert _directory(tmp_path).control_resolution_for(512) == 512
    assert _directory(tmp_path, control_resolution=1024).control_resolution_for(512) == 1024


def test_metadata_build_pairs_numbered_controls(tmp_path):
    _img(tmp_path / "t" / "x.png")
    _img(tmp_path / "c" / "x_1.png", (96, 64))
    _img(tmp_path / "c" / "x_0.png", (64, 96))
    dd = _directory(tmp_path)
    dd.cache_metadata()
    md = dd.ar_bucket_datasets[0].metadata_dataset
    row = md[0]
    assert row["control_file"] == [str(tmp_path / "c" / "x_0.png"), str(tmp_path / "c" / "x_1.png")]
    assert row["control_dims"] == [[64, 96], [96, 64]]
    assert len(row["control_stamp"]) == 2


def test_metadata_build_rejects_unpaired_target(tmp_path):
    _img(tmp_path / "t" / "x.png")
    _img(tmp_path / "t" / "y.png")
    _img(tmp_path / "c" / "x.png")
    with pytest.raises(ControlPairingError, match="y.png"):
        _directory(tmp_path).cache_metadata()


def test_preflight_reports_pairing_problems(tmp_path):
    _img(tmp_path / "t" / "ok.png")
    _img(tmp_path / "t" / "missing.png")
    _img(tmp_path / "t" / "gap.png")
    _img(tmp_path / "t" / "both.png")
    (tmp_path / "t" / "ok.txt").write_text("caption")
    _img(tmp_path / "c" / "ok.png")
    _img(tmp_path / "c" / "gap_0.png")
    _img(tmp_path / "c" / "gap_2.png")
    _img(tmp_path / "c" / "both.png")
    _img(tmp_path / "c" / "both_0.png")
    ds_toml = tmp_path / "ds.toml"
    ds_toml.write_text(
        "[[directory]]\n"
        f"path = '{(tmp_path / 't').as_posix()}'\n"
        f"control_path = '{(tmp_path / 'c').as_posix()}'\n"
        "num_repeats = 1\n"
    )
    issues = _dataset_directory_issues(ds_toml)
    text = "\n".join(issues)
    assert len(issues) == 3, issues
    assert "missing.png" in text and "gap.png" in text and "both.png" in text
    assert "ok.png" not in text


def test_preflight_quiet_for_valid_edit_dataset(tmp_path):
    _img(tmp_path / "t" / "a.png")
    _img(tmp_path / "c" / "a_0.png")
    _img(tmp_path / "c" / "a_1.png")
    ds_toml = tmp_path / "ds.toml"
    ds_toml.write_text(
        "[[directory]]\n"
        f"path = '{(tmp_path / 't').as_posix()}'\n"
        f"control_path = '{(tmp_path / 'c').as_posix()}'\n"
        "num_repeats = 1\n"
    )
    assert _dataset_directory_issues(ds_toml) == []


def test_ui_form_roundtrip_keeps_control_keys():
    """The dataset form drops [[directory]] keys it does not list; the edit keys must survive."""
    import toml

    from rengu_flow_ui.dataset_form import form_to_toml, parse_toml_to_form
    from rengu_flow_ui.dataset_schema import get_dataset_schema

    raw = (
        "resolutions = [1024]\n\n[[directory]]\npath = '/data/t'\nnum_repeats = 1\n"
        "control_path = '/data/c'\ncontrol_resolution = 768\n"
    )
    form, warnings = parse_toml_to_form(raw)
    assert not [w for w in warnings if "control" in w], warnings
    (directory,) = toml.loads(form_to_toml(form))["directory"]
    assert directory["control_path"] == "/data/c"
    assert directory["control_resolution"] == 768
    fields = {f["path"]: f for f in get_dataset_schema()["directory_fields"]}
    assert fields["control_resolution"]["type"] == "integer"


def test_ui_help_covers_control_keys():
    from rengu_flow_ui.dataset_field_help import FIELD_HELP

    for key in ("directory.control_path", "directory.control_resolution"):
        assert FIELD_HELP[key]["summary"]
