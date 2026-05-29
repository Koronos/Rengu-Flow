"""Default TOML seeded when the web UI creates a new training config."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "default_new_config.toml"


def default_new_config_toml() -> str:
    """Production-style SDXL LoRA template (no synthetic/debug keys)."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")
