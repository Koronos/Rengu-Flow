"""Training form schema for krea2: its scheduler knobs are advanced and link the Krea 2 doc (the
form checks every model shares live in tests/contract/test_spec_contract.py)."""

from __future__ import annotations

import pytest

from rengu_flow_ui.config_schema import get_schema
from rengu_flow_ui.field_visibility import field_visible

pytestmark = pytest.mark.no_ui_db

KREA_DOC = "docs/user/training-krea2.md"


@pytest.fixture
def schema(monkeypatch):
    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")
    return get_schema()


def _visible(schema, form) -> dict[str, dict]:
    caps = schema["registries"]["model_capabilities"]
    return {
        field["path"]: field
        for section in schema["sections"]
        for field in section["fields"]
        if field_visible(field, form, caps)
    }


@pytest.mark.parametrize(
    "path",
    [
        "model.shift",
        "model.sigmoid_scale",
        "model.timestep_sample_method",
        "model.transformer_fp8_matmul",
        "model.transformer_dtype",
    ],
)
def test_krea2_fields_have_krea_help_and_scheduler_knobs_are_advanced(schema, path):
    visible = _visible(schema, {"model.type": "krea2", "_has_adapter": True})
    field = visible[path]
    assert field.get("doc_path") == KREA_DOC, path
    if path in ("model.shift", "model.sigmoid_scale", "model.timestep_sample_method"):
        assert field["importance"] == "advanced"
