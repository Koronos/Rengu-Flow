"""Tests for training run folder naming."""

from __future__ import annotations

import pytest

from rengu_flow.config.validation import collect_validation_errors
from rengu_flow.run_naming import (
    build_run_folder_name,
    collect_run_name_validation_errors,
    sanitize_run_name,
)


def test_build_run_folder_name_timestamp_only() -> None:
    assert build_run_folder_name(None, timestamp="20250217_14-30-00") == "20250217_14-30-00"
    assert build_run_folder_name("", timestamp="20250217_14-30-00") == "20250217_14-30-00"
    assert build_run_folder_name("   ", timestamp="20250217_14-30-00") == "20250217_14-30-00"


def test_build_run_folder_name_with_label() -> None:
    assert (
        build_run_folder_name("sdxl-lora-v1", timestamp="20250217_14-30-00")
        == "sdxl-lora-v1_20250217_14-30-00"
    )


def test_sanitize_run_name() -> None:
    assert sanitize_run_name("  my run!!  ") == "my_run"
    assert sanitize_run_name("foo/bar") == "foo_bar"
    assert sanitize_run_name("///") == ""


def test_collect_run_name_validation_rejects_slashes() -> None:
    issues = collect_run_name_validation_errors({"run_name": "bad/name"})
    assert issues
    assert "/" in issues[0] or "slash" in issues[0].lower()


def test_collect_validation_includes_run_name() -> None:
    issues = collect_validation_errors({"run_name": "a/b"})
    assert any("run_name" in i for i in issues)


@pytest.mark.parametrize(
    "raw",
    ["x" * 81, 123, {"nested": True}],
)
def test_run_name_validation_bad_values(raw) -> None:
    issues = collect_run_name_validation_errors({"run_name": raw})
    assert issues
