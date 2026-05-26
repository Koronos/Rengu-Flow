"""Validation API returns readable errors (not bare KeyError names)."""

from renga_flow.config.validation import collect_validation_errors, format_validation_issues
from renga_flow_ui.configs_store import validate_toml_text


def test_validate_empty_content_lists_issues() -> None:
    r = validate_toml_text("")
    assert r["ok"] is False
    assert "errors" in r
    assert len(r["errors"]) >= 2
    assert any("empty" in e.lower() for e in r["errors"])
    assert not any(e.strip("'\"") == "model" and len(e) < 12 for e in r["errors"])


def test_validate_missing_model_section() -> None:
    r = validate_toml_text('dataset = "d.toml"\n[optimizer]\ntype = "adamw"\n')
    assert r["ok"] is False
    assert any("model" in e.lower() for e in r["errors"])


def test_collect_multiple_missing_sections() -> None:
    issues = collect_validation_errors({})
    assert len(issues) >= 3
    assert format_validation_issues(issues).startswith("Fix the following:")
