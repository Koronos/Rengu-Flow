"""Training form schema for krea2: dead knobs hidden, component paths discoverable, help present."""

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


@pytest.mark.parametrize("path", ["adapter.train_conv", "adapter.use_tucker", "adapter.train_norm"])
@pytest.mark.parametrize("model_type,shown", [("krea2", False), ("cosmos_predict2", False), ("sdxl", True)])
def test_conv_and_norm_lycoris_knobs_hidden_on_linear_only_models(schema, path, model_type, shown):
    """Krea 2's DiT has no Conv modules and only custom Krea2RMSNorm (LyCORIS' train_norm
    matches affine LayerNorm/GroupNorm only), so these knobs are no-ops there."""
    form = {"model.type": model_type, "_has_adapter": True, "adapter.type": "lycoris_locon"}
    assert (path in _visible(schema, form)) is shown


@pytest.mark.parametrize("model_type", ["krea2", "sdxl", "cosmos_predict2"])
def test_adapter_dim_alias_not_a_ui_field(schema, model_type):
    visible = _visible(schema, {"model.type": model_type, "_has_adapter": True, "adapter.type": "lora"})
    assert "adapter.rank" in visible
    assert "adapter.dim" not in visible


@pytest.mark.parametrize("model_type,shown", [("krea2", False), ("cosmos_predict2", True)])
def test_diffusion_model_dtype_hidden_for_krea2(schema, model_type, shown):
    visible = _visible(schema, {"model.type": model_type, "_has_adapter": True})
    assert ("model.diffusion_model_dtype" in visible) is shown


def test_krea2_checkpoint_path_discoverable_and_component_paths_one_of(schema):
    visible = _visible(schema, {"model.type": "krea2", "_has_adapter": True})
    assert "model.checkpoint_path" in visible  # visible without being set first
    for path in ("model.transformer_path", "model.vae_path", "model.text_encoder_path"):
        assert visible[path]["required"] is False, path  # one_of with checkpoint_path


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
