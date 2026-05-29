"""Format numeric defaults for UI labels and TOML-friendly string pre-fill."""

from __future__ import annotations

import json
import math
from typing import Any

# |n| below this uses scientific notation (matches common lr/epsilon magnitudes).
_SMALL_ABS_THRESHOLD = 1e-3
_LARGE_ABS_THRESHOLD = 1e4


def format_scientific(n: float) -> str:
    """TOML-parseable scientific string (e.g. 1e-4, 1.5e-3)."""
    if n == 0.0:
        return "0"
    sign = "-" if n < 0 else ""
    x = abs(n)
    exp = int(math.floor(math.log10(x)))
    mant = x / (10.0**exp)
    mant = round(mant, 12)
    if abs(mant - round(mant)) < 1e-9:
        mant_s = str(int(round(mant)))
    else:
        mant_s = f"{mant:.12f}".rstrip("0").rstrip(".")
    if exp < 0:
        return f"{sign}{mant_s}e{exp}"
    if exp > 0:
        return f"{sign}{mant_s}e+{exp}"
    return f"{sign}{mant_s}"


def _fixed_decimal_places(n: float) -> tuple[str, int]:
    if n == 0.0:
        return "0", 0
    abs_n = abs(n)
    if abs_n >= 1:
        decimals = 12
    else:
        exp = int(math.floor(math.log10(abs_n)))
        decimals = max(0, -exp) + 6
    s = f"{n:.{decimals}f}".rstrip("0").rstrip(".")
    if "." not in s:
        return s, 0
    return s, len(s.split(".", 1)[1])


def format_default_number(n: float) -> str:
    """Format a numeric default for display / KV pre-fill."""
    if not math.isfinite(n):
        return str(n)
    if n == 0.0:
        return "0"

    abs_n = abs(n)
    use_sci = abs_n >= _LARGE_ABS_THRESHOLD or (abs_n > 0 and abs_n < _SMALL_ABS_THRESHOLD)
    if not use_sci and float(n).is_integer() and abs(n) < 1e15:
        return str(int(n))

    if use_sci:
        return format_scientific(n)

    fixed, places = _fixed_decimal_places(n)
    if places > 3:
        return format_scientific(n)
    return fixed


def format_default_value(val: Any) -> str:
    """Format any schema default for UI hint text."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if isinstance(val, int):
            return str(val)
        return format_default_number(float(val))
    if isinstance(val, (list, dict)):
        return json.dumps(val, separators=(",", ":"))
    return str(val)
