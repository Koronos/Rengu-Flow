"""Training form schema for qwen_image21: which fields the UI shows, their help and defaults."""

from __future__ import annotations

import pytest

from rengu_flow.config import set_config_defaults
from rengu_flow_ui.config_schema import get_schema
from rengu_flow_ui.field_visibility import field_visible

pytestmark = pytest.mark.no_ui_db

QWEN_DOC = "docs/user/training-qwen-image21.md"


@pytest.fixture
def schema(monkeypatch):
    # Full superset regardless of host OS (native Windows drops the DeepSpeed-only fields).
    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")
    return get_schema()


def _visible(schema, form) -> dict[str, dict]:
    caps = schema["registries"]["model_capabilities"]
    out: dict[str, dict] = {}
    for section in schema["sections"]:
        for field in section["fields"]:
            if field_visible(field, form, caps):
                assert field["path"] not in out, f"two visible fields share path {field['path']}"
                out[field["path"]] = field
    return out


def _form(**extra):
    return {"model.type": "qwen_image21", "_has_adapter": True, **extra}


def test_qwen_image21_is_a_selectable_model(schema) -> None:
    assert "qwen_image21" in schema["registries"]["models"]
    cap = schema["registries"]["model_capabilities"]["qwen_image21"]
    assert cap["features"]["block_swap"] is True
    assert cap["features"]["preview"] is True


def test_qwen_image21_component_fields_visible(schema) -> None:
    visible = _visible(schema, _form())
    assert visible["model.diffusers_path"]["required"] is True
    for path in (
        "model.text_encoder_offload",
        "model.transformer_fp8_matmul",
        "model.transformer_4bit",
        "model.transformer_dtype",
        "model.shift",
        "model.sigmoid_scale",
        "model.timestep_sample_method",
        "blocks_to_swap",
        "preview.enabled",
        "preview.preview_blocks_to_swap",
        "preview.preview_offload_text_encoder",
        "disable_block_swap_for_preview",
    ):
        assert path in visible, path
    assert visible["model.text_encoder_offload"]["options"] == ["auto", "stream", "none"]
    assert visible["model.text_encoder_offload"]["default"] == "auto"


def test_qwen_image21_overrides_are_expert_fields_until_set(schema) -> None:
    overrides = ("model.transformer_path", "model.vae_path", "model.text_encoder_path", "model.processor_path")
    visible = _visible(schema, _form())
    assert not set(overrides) & set(visible)
    filled = _visible(schema, _form(**{p: "/x" for p in overrides}))
    assert set(overrides) <= set(filled)


def test_qwen_image21_hides_other_models_fields(schema) -> None:
    visible = _visible(schema, _form())
    for path in (
        "tread.drop_ratio",  # the pipeline raises on [tread]
        "model.cache_text_embeddings",  # always on; false is a config error
        "video_clip_mode",  # image-only model
        "preview.preview_offload_dit_for_decode",  # cosmos-only preview knob
        "model.checkpoint_path",
        "model.max_sequence_length",
        "model.fp8_matmul_dtype",
        "model.llm_path",
    ):
        assert path not in visible, path
    # TREAD's dependent fields stay hidden even with a stale drop_ratio in the form.
    stale = _visible(schema, _form(**{"tread.drop_ratio": 0.5}))
    assert not {p for p in stale if p.startswith("tread.")}


def test_qwen_image21_frozen_base_quant_only_with_adapter(schema) -> None:
    finetune = _visible(schema, _form(_has_adapter=False))
    assert "model.transformer_fp8_matmul" not in finetune
    assert "model.transformer_4bit" not in finetune
    assert "model.fp8_grad_mode" not in _visible(schema, _form())
    assert "model.fp8_grad_mode" in _visible(schema, _form(**{"model.transformer_fp8_matmul": True}))


def test_qwen_image21_preview_defaults_match_trainer_defaults(schema) -> None:
    """The form shows the sampler defaults the trainer actually applies for this model."""
    visible = _visible(schema, _form())
    cfg = {"model": {"type": "qwen_image21", "dtype": "bfloat16"}, "preview": {}}
    set_config_defaults(cfg)
    assert visible["preview.num_inference_steps"]["default"] == cfg["preview"]["num_inference_steps"] == 28
    assert visible["preview.guidance_scale"]["default"] == cfg["preview"]["guidance_scale"] == 1.0


@pytest.mark.parametrize("model_type", ["cosmos_predict2", "krea2"])
def test_dit_preview_defaults_match_trainer_defaults(schema, model_type) -> None:
    visible = _visible(schema, {"model.type": model_type, "_has_adapter": True})
    cfg = {"model": {"type": model_type, "dtype": "bfloat16"}, "preview": {}}
    set_config_defaults(cfg)
    assert visible["preview.num_inference_steps"]["default"] == cfg["preview"]["num_inference_steps"]
    assert visible["preview.guidance_scale"]["default"] == cfg["preview"]["guidance_scale"]


def test_sdxl_preview_defaults_unchanged(schema) -> None:
    visible = _visible(schema, {"model.type": "sdxl", "_has_adapter": True})
    assert visible["preview.num_inference_steps"]["default"] == 20
    assert visible["preview.guidance_scale"]["default"] == 7.0


@pytest.mark.parametrize(
    "path",
    [
        "model.diffusers_path",
        "model.transformer_path",
        "model.vae_path",
        "model.text_encoder_path",
        "model.processor_path",
        "model.text_encoder_offload",
        "model.transformer_fp8_matmul",
        "model.transformer_4bit",
        "model.fp8_grad_mode",
        "model.transformer_dtype",
        "model.shift",
        "model.sigmoid_scale",
        "model.timestep_sample_method",
        "preview.num_inference_steps",
        "preview.guidance_scale",
    ],
)
def test_qwen_image21_fields_link_the_qwen_doc(schema, path) -> None:
    visible = _visible(schema, _form(**{
        "model.transformer_path": "/x",
        "model.vae_path": "/x",
        "model.text_encoder_path": "/x",
        "model.processor_path": "/x",
        "model.transformer_fp8_matmul": True,
    }))
    field = visible[path]
    assert field["doc_path"] == QWEN_DOC, path
    assert field["help"] and field["help"] != field["label"], path


@pytest.mark.parametrize(
    ("path", "expect"),
    [
        ("model.diffusers_path", "dir"),
        ("model.processor_path", "dir"),
        # folder or single .safetensors
        ("model.transformer_path", "any"),
        ("model.vae_path", "any"),
        ("model.text_encoder_path", "any"),
    ],
)
def test_qwen_image21_path_fields_declare_file_or_dir(schema, path, expect) -> None:
    visible = _visible(schema, _form(**{path: "/x"}))
    assert visible[path]["path_expect"] == expect


def test_block_swap_and_preview_help_mention_qwen(schema) -> None:
    visible = _visible(schema, _form())
    assert "Qwen-Image 2.1" in visible["blocks_to_swap"]["help"]
    assert "Qwen-Image 2.1" in visible["preview.preview_blocks_to_swap"]["description"]
    assert "Qwen-Image 2.1" in visible["preview.preview_offload_text_encoder"]["description"]
