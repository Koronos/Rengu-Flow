"""Model section TOML-only paths and pruning."""

from __future__ import annotations

from renga_flow_ui.config_schema import get_schema
from renga_flow_ui.field_visibility import field_visible, prune_form_for_model
from renga_flow_ui.model_form import hidden_model_paths_by_type, toml_only_model_paths_for_type


def test_hidden_model_paths_by_type() -> None:
    hidden = hidden_model_paths_by_type()
    assert hidden["cosmos_predict2"] == ["model.diffusion_model_dtype"]
    assert hidden["sdxl"] == ["model.guidance"]


def test_diffusion_model_dtype_not_in_schema() -> None:
    schema = get_schema()
    paths = {f["path"] for s in schema["sections"] for f in s["fields"]}
    assert "model.diffusion_model_dtype" not in paths
    assert "model.guidance" not in paths


def test_prune_drops_toml_only_keys_for_cosmos() -> None:
    form = {
        "model.type": "cosmos_predict2",
        "model.dtype": "bfloat16",
        "model.transformer_path": "/t",
        "model.diffusion_model_dtype": "float16",
    }
    pruned = prune_form_for_model(form)
    assert "model.diffusion_model_dtype" not in pruned
    assert pruned["model.transformer_path"] == "/t"


def test_prune_drops_guidance_for_sdxl() -> None:
    form = {
        "model.type": "sdxl",
        "model.dtype": "bfloat16",
        "model.checkpoint_path": "/x.safetensors",
        "model.guidance": 7.0,
    }
    pruned = prune_form_for_model(form)
    assert "model.guidance" not in pruned
    assert pruned["model.checkpoint_path"] == "/x.safetensors"


def test_toml_only_paths_empty_for_unknown_type() -> None:
    assert toml_only_model_paths_for_type("unknown_model") == frozenset()


def test_cosmos_model_fields_visible_not_toml_only() -> None:
    schema = get_schema()
    caps = schema["registries"]["model_capabilities"]
    dtype_field = next(
        f
        for s in schema["sections"]
        if s["id"] == "model"
        for f in s["fields"]
        if f["path"] == "model.transformer_dtype"
    )
    form = {"model.type": "cosmos_predict2"}
    assert field_visible(dtype_field, form, caps) is True
