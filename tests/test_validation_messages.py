"""Validation API returns readable errors (not bare KeyError names)."""

from rengu_flow.config.dataset_library_ref import dataset_library_ref
from rengu_flow.config.validation import (
    collect_validation_errors,
    format_validation_issues,
    validate_config,
    ConfigValidationError,
)
from rengu_flow_ui.run_staging import validate_toml_text


def test_validate_empty_content_lists_issues() -> None:
    r = validate_toml_text("")
    assert r["ok"] is False
    assert "errors" in r
    assert len(r["errors"]) >= 2
    assert any("empty" in e.lower() for e in r["errors"])
    assert not any(e.strip("'\"") == "model" and len(e) < 12 for e in r["errors"])


def test_validate_rejects_optimizer_betas_wrong_length() -> None:
    r = validate_toml_text(
        """
dataset = "d.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
lr = 1e-4
betas = [0.9, 0.95, 0.99]
"""
    )
    assert r["ok"] is False
    assert any("exactly two" in e for e in r["errors"])


def test_validate_missing_model_section() -> None:
    r = validate_toml_text('dataset = "d.toml"\n[optimizer]\ntype = "adamw"\n')
    assert r["ok"] is False
    assert any("model" in e.lower() for e in r["errors"])


def test_collect_multiple_missing_sections() -> None:
    issues = collect_validation_errors({})
    assert len(issues) >= 3
    assert format_validation_issues(issues).startswith("Fix the following:")


def test_invalid_cache_format() -> None:
    issues = collect_validation_errors(
        {
            "dataset": "d.toml",
            "model": {"type": "sdxl", "dtype": "bfloat16"},
            "optimizer": {"type": "adamw", "lr": 1e-4},
            "cache_format": "pickle",
        }
    )
    assert any("cache_format" in e for e in issues)


def _minimal_training_config(dataset: str | list[str]) -> dict:
    return {
        "dataset": dataset,
        "model": {"type": "sdxl", "dtype": "bfloat16"},
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }


def test_ui_library_ref_allowed_in_ui_validate() -> None:
    ref = dataset_library_ref(3, "artista 1")
    issues = collect_validation_errors(_minimal_training_config(ref))
    assert not any("rengu-flow-dataset" in e for e in issues)


def test_ui_library_ref_rejected_in_script_validate() -> None:
    ref = dataset_library_ref(3, "artista 1")
    issues = collect_validation_errors(_minimal_training_config(ref), for_script=True)
    assert any("Export dataset #3" in e for e in issues)
    assert any("rengu-flow-dataset" in e for e in issues)


def test_script_validate_config_raises_on_library_ref() -> None:
    ref = dataset_library_ref(1)
    try:
        validate_config(_minimal_training_config(ref), for_script=True)
    except ConfigValidationError as e:
        assert "Export" in str(e)
    else:
        raise AssertionError("expected ConfigValidationError")
