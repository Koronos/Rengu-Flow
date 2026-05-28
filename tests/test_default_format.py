"""Tests for default value display formatting."""

import math

import pytest

from renga_flow_ui.default_format import (
    format_default_number,
    format_default_value,
    format_scientific,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1e-4, "1e-4"),
        (1e-6, "1e-6"),
        (1e-7, "1e-7"),
        (1e-3, "0.001"),
        (0.01, "0.01"),
        (0.5, "0.5"),
        (7.0, "7"),
        (1.0, "1"),
        (0.0, "0"),
        (0.999, "0.999"),
        (1.5e-3, "1.5e-3"),
    ],
)
def test_format_default_number(value: float, expected: str) -> None:
    assert format_default_number(value) == expected


def test_format_default_number_large_uses_scientific() -> None:
    assert format_default_number(100_000.0) == "1e+5"


def test_format_default_number_many_decimal_places() -> None:
    assert format_default_number(0.00001) == "1e-5"


def test_format_scientific_zero() -> None:
    assert format_scientific(0.0) == "0"


def test_format_default_value_bool_and_json() -> None:
    assert format_default_value(True) == "true"
    assert format_default_value([0.9, 0.999]) == "[0.9,0.999]"


def test_format_default_number_nan() -> None:
    assert format_default_number(math.nan) == "nan"
