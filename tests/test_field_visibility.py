"""Tests for centralized config form field visibility."""

from __future__ import annotations

from rengu_flow_ui.config_schema import get_schema
from rengu_flow_ui.field_visibility import field_visible, prune_form_for_model
from rengu_flow_ui.config_form import parse_toml, form_to_toml


def _paths(schema) -> set[str]:
    return {f["path"] for s in schema["sections"] for f in s["fields"]}


def test_cosmos_schema_excludes_t5_and_includes_llm() -> None:
    schema = get_schema()
    paths = _paths(schema)
    assert "model.t5_path" not in paths
    assert "model.llm_path" in paths
    assert "model.transformer_path" in paths
    assert "model.checkpoint_path" in paths
    # diffusion_model_dtype is a TOML-only (ui: false) expert key for cosmos — hidden
    # from the web form, like sdxl's model.guidance below.
    assert "model.diffusion_model_dtype" not in paths
    assert "model.guidance" not in paths


def test_blocks_to_swap_visible_when_model_supports_block_swap() -> None:
    schema = get_schema()
    field = next(f for s in schema["sections"] for f in s["fields"] if f["path"] == "blocks_to_swap")
    assert field.get("when_capability") == "block_swap"
    caps = schema["registries"]["model_capabilities"]

    cosmos_form = {"model.type": "cosmos_predict2", "_has_adapter": True}
    sdxl_form = {"model.type": "sdxl", "_has_adapter": True}
    assert field_visible(field, cosmos_form, caps) is True
    assert field_visible(field, sdxl_form, caps) is True


def test_block_swap_prefetch_gated_by_capability_blocks_and_gradient_release() -> None:
    schema = get_schema()
    field = next(
        f for s in schema["sections"] for f in s["fields"] if f["path"] == "block_swap_prefetch"
    )
    assert field.get("when_capability") == "block_swap"
    caps = schema["registries"]["model_capabilities"]

    gr = {"optimizer.extra_params": {"gradient_release": True}}

    # No blocks being swapped → hidden even with gradient_release.
    assert field_visible(field, {"model.type": "sdxl", "blocks_to_swap": 0, **gr}, caps) is False
    # Swapping blocks but no gradient_release (e.g. adapter run) → backend ignores prefetch → hidden.
    assert field_visible(field, {"model.type": "sdxl", "blocks_to_swap": 6}, caps) is False
    assert (
        field_visible(
            field,
            {"model.type": "sdxl", "blocks_to_swap": 6,
             "optimizer.extra_params": {"gradient_release": False}},
            caps,
        )
        is False
    )
    # Blocks swapped + gradient_release truthy → shown (SDXL and Cosmos).
    assert field_visible(field, {"model.type": "sdxl", "blocks_to_swap": 6, **gr}, caps) is True
    assert field_visible(field, {"model.type": "cosmos_predict2", "blocks_to_swap": 20, **gr}, caps) is True
    # String "true" from the KV widget is treated as truthy.
    assert (
        field_visible(
            field,
            {"model.type": "sdxl", "blocks_to_swap": 6,
             "optimizer.extra_params": {"gradient_release": "true"}},
            caps,
        )
        is True
    )


def test_llm_adapter_fields_hidden_until_set() -> None:
    schema = get_schema()
    caps = schema["registries"]["model_capabilities"]
    path_field = next(f for f in schema["sections"] if f["id"] == "model")
    path_field = next(
        f for s in schema["sections"] if s["id"] == "model" for f in s["fields"] if f["path"] == "model.llm_adapter_path"
    )
    lr_field = next(
        f for s in schema["sections"] if s["id"] == "model" for f in s["fields"] if f["path"] == "model.llm_adapter_lr"
    )
    empty = {"model.type": "cosmos_predict2"}
    assert field_visible(path_field, empty, caps) is False
    assert field_visible(lr_field, empty, caps) is False
    with_path = {**empty, "model.llm_adapter_path": "/tmp/a.safetensors"}
    assert field_visible(path_field, with_path, caps) is True
    assert field_visible(lr_field, with_path, caps) is True


def test_llm_adapter_lr_shown_for_cosmos_finetune() -> None:
    schema = get_schema()
    caps = schema["registries"]["model_capabilities"]
    lr_field = next(
        f for s in schema["sections"] if s["id"] == "model" for f in s["fields"] if f["path"] == "model.llm_adapter_lr"
    )
    base = {"model.type": "cosmos_predict2", "model.llm_path": "/l"}
    # Finetune (no adapter network): LLM adapter LR is controllable.
    assert field_visible(lr_field, {**base, "_has_adapter": False}, caps) is True
    # LoRA/LoKr: get_param_groups isn't used, so the knob is hidden even with a text encoder set.
    assert field_visible(lr_field, {**base, "_has_adapter": True}, caps) is False


def test_prune_drops_other_model_paths() -> None:
    form = {
        "model.type": "sdxl",
        "model.checkpoint_path": "/x.safetensors",
        "model.transformer_path": "/should-go.safetensors",
        "model.vae_path": "/v.safetensors",
    }
    pruned = prune_form_for_model(form)
    assert "model.checkpoint_path" in pruned
    assert "model.transformer_path" not in pruned
    assert "model.vae_path" not in pruned


def test_prune_roundtrip_toml() -> None:
    toml_in = """
dataset = "d.toml"
[model]
type = "cosmos_predict2"
dtype = "bfloat16"
transformer_path = "/t"
vae_path = "/v"
llm_path = "/l"
[optimizer]
type = "adamw"
"""
    form = parse_toml(toml_in)
    form["model.type"] = "sdxl"
    form["model.checkpoint_path"] = "/sdxl.safetensors"
    pruned = prune_form_for_model(form)
    out = form_to_toml(pruned)
    assert "transformer_path" not in out
    assert "checkpoint_path" in out
