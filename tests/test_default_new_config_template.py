"""Default TOML for web UI 'New config' — production-style, no synthetic data."""

from __future__ import annotations

import re

import toml

from rengu_flow.config.validation import collect_validation_errors
from rengu_flow_ui.config_schema import get_schema
from rengu_flow_ui.default_config_template import default_new_config_toml


def test_default_new_config_toml_file() -> None:
    text = default_new_config_toml()
    assert "synthetic_num_batches" not in text
    assert "path/to/" not in text
    assert 'dataset = ""' in text
    assert 'checkpoint_path = ""' in text
    assert "[adapter]" in text
    assert 'type = "lora"' in text

    config = toml.loads(text)
    assert config["dataset"] == ""
    assert config["model"]["type"] == "sdxl"
    assert config["model"]["checkpoint_path"] == ""
    assert config["optimizer"]["type"] == "adamw"
    assert config.get("lr_scheduler") == "cosine"
    assert config.get("optimizer", {}).get("lr_scheduler") is None
    assert config["adapter"]["rank"] == 16

    issues = collect_validation_errors(config)
    assert any("dataset" in i for i in issues)


def test_schema_includes_default_new_config_toml() -> None:
    schema = get_schema()
    assert "default_new_config_toml" in schema
    assert schema["default_new_config_toml"] == default_new_config_toml()


def test_schema_default_new_config_via_api(ui_client) -> None:
    r = ui_client.get("/api/v1/schema")
    assert r.status_code == 200
    data = r.json()
    text = data["default_new_config_toml"]
    assert "synthetic_num_batches" not in text
    assert 'dataset = ""' in text
    assert 'checkpoint_path = ""' in text


def test_fallback_ts_template_matches_python_file() -> None:
    """Keep ui/web FALLBACK_DEFAULT_CONFIG_TOML aligned with the canonical template."""
    from pathlib import Path

    ts_path = Path(__file__).resolve().parents[1] / "ui" / "web" / "src" / "stores" / "configEditor.ts"
    ts_text = ts_path.read_text(encoding="utf-8")
    match = re.search(
        r'export const FALLBACK_DEFAULT_CONFIG_TOML = `([^`]*)`;',
        ts_text,
        re.DOTALL,
    )
    assert match, "FALLBACK_DEFAULT_CONFIG_TOML not found in configEditor.ts"
    fallback = match.group(1).strip() + "\n"
    canonical = default_new_config_toml()
    # Comments only in the .toml file; compare parsed structure.
    assert toml.loads(fallback) == toml.loads(canonical)


def test_default_new_config_form_roundtrip() -> None:
    from rengu_flow_ui.config_form import form_to_toml, parse_toml

    form = parse_toml(default_new_config_toml())
    assert form.get("lr_scheduler") == "cosine"
    out = form_to_toml(form)
    again = parse_toml(out)
    assert again.get("lr_scheduler") == "cosine"
