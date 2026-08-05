"""RF-09: preview sampling is opt-in (off by default) so a New config doesn't emit
[preview] enabled=true and OOM small GPUs on a quick run."""

from pathlib import Path

from rengu_flow_ui.config_schema import get_schema


def _find_field(schema: dict, path: str) -> dict | None:
    for section in schema["sections"]:
        for f in section.get("fields", []):
            if f.get("path") == path:
                return f
    return None


def test_preview_disabled_by_default(ui_data_tmp: Path) -> None:
    field = _find_field(get_schema(), "preview.enabled")
    assert field is not None, "preview.enabled field missing from schema"
    assert field.get("default") is False


def test_preview_seed_is_fixed_across_steps_by_default(ui_data_tmp: Path) -> None:
    field = _find_field(get_schema(), "preview.seed_stride")
    assert field is not None, "preview.seed_stride field missing from schema"
    assert field.get("default") == 0
