"""SDXL adapter exports name the text encoders canonically (``text_model``), whatever the
transformers version: ComfyUI / A1111 only recognise ``lora_te1_text_model_encoder_layers_N_...``
and silently drop the encoder half otherwise. Loads also accept the flat TE1 names that
transformers-5 exports wrote. Real tiny CLIP encoders (TE1 is flat under transformers 5)."""

from __future__ import annotations

import re

import pytest
import safetensors.torch
import torch

from rengu_flow.model import sdxl_testkit as kit
from rengu_flow.model.testkit_common import ADAPTER_CONFIGS
from rengu_flow.networks import sdxl_te_names

pytestmark = pytest.mark.no_ui_db

COMFY_TE_KEY = re.compile(
    r"^lora_te[12]_(text_model_encoder_layers_\d+_(self_attn_(q|k|v|out)_proj|mlp_fc[12])|text_projection)\."
)
DOTTED_TE_KEY = re.compile(
    r"^text_encoder(_2)?\.(text_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])|text_projection)\."
)
LEGACY_FLAT = re.compile(r"text_model[_.](?=encoder)")


def _pipeline(tmp_path, adapter):
    if adapter.startswith("lycoris_"):
        pytest.importorskip("lycoris")
    cfg = kit.tiny_config(tmp_path, adapter=dict(ADAPTER_CONFIGS[adapter]))
    return kit.tiny_pipeline(cfg)


def _export(p, save_dir):
    """save_adapter of the trainable weights, moved off their init, read back."""
    with torch.no_grad():
        for net in kit.adapter_networks(p).values():
            for prm in net.parameters():
                if prm.requires_grad:
                    prm.add_(torch.randn_like(prm) * 0.01)
    state = {
        prm.original_name.replace(".default", ""): prm.detach().clone()
        for net in kit.adapter_networks(p).values()
        for prm in net.parameters()
        if prm.requires_grad
    }
    p.save_adapter(save_dir, state)
    (file,) = save_dir.glob("*.safetensors")
    return file, safetensors.torch.load_file(str(file))


def _te_keys(exported):
    return [k for k in exported if "te1" in k or "te2" in k or k.startswith("text_encoder")]


@pytest.mark.parametrize("adapter", list(ADAPTER_CONFIGS))
def test_export_names_both_text_encoders_canonically(adapter, tmp_path):
    p = _pipeline(tmp_path, adapter)
    _file, exported = _export(p, tmp_path / "export")
    te = _te_keys(exported)
    assert any("te1" in k or k.startswith("text_encoder.") for k in te)
    assert any("te2" in k or k.startswith("text_encoder_2.") for k in te)
    pattern = DOTTED_TE_KEY if adapter == "lokr" else COMFY_TE_KEY
    bad = [k for k in te if not pattern.match(k)]
    assert not bad, bad[:4]


@pytest.mark.parametrize("adapter", list(ADAPTER_CONFIGS))
@pytest.mark.parametrize("legacy", [False, True])
def test_loaded_weights_equal_the_exported_ones_canonical_or_legacy_flat(adapter, legacy, tmp_path):
    p = _pipeline(tmp_path, adapter)
    file, exported = _export(p, tmp_path / "export")
    src = file
    if legacy:
        flat = {
            (LEGACY_FLAT.sub("", k, count=1) if k.startswith(("lora_te1_text_model_", "text_encoder.text_model.")) else k): v
            for k, v in exported.items()
        }
        assert flat.keys() != exported.keys(), "no TE1 key was made legacy"
        src = tmp_path / "legacy" / file.name
        src.parent.mkdir()
        safetensors.torch.save_file(flat, str(src))
    q = _pipeline(tmp_path, adapter)
    q.load_adapter_weights(src.parent)
    state = {
        prm.original_name.replace(".default", ""): prm.detach().clone()
        for net in kit.adapter_networks(q).values()
        for prm in net.parameters()
        if prm.requires_grad
    }
    q.save_adapter(tmp_path / "again", state)
    (again_file,) = (tmp_path / "again").glob("*.safetensors")
    again = safetensors.torch.load_file(str(again_file))
    assert again.keys() == exported.keys()
    for key, value in exported.items():
        assert torch.equal(again[key], value), key


@pytest.mark.parametrize(
    "key, canonical",
    [
        ("lora_te1_encoder_layers_0_mlp_fc1.alpha", "lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha"),
        ("lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha", "lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha"),
        ("lora_te2_text_projection.alpha", "lora_te2_text_projection.alpha"),
        ("lora_unet_down_blocks_0.alpha", "lora_unet_down_blocks_0.alpha"),
    ],
)
def test_canonical_flat(key, canonical):
    assert list(sdxl_te_names.canonical_flat({key: 1})) == [canonical]


@pytest.mark.parametrize(
    "key, canonical",
    [
        ("text_encoder.encoder.layers.0.mlp.fc1.alpha", "text_encoder.text_model.encoder.layers.0.mlp.fc1.alpha"),
        ("text_encoder_2.text_model.encoder.layers.0.mlp.fc1.alpha", "text_encoder_2.text_model.encoder.layers.0.mlp.fc1.alpha"),
        ("text_encoder_2.text_projection.alpha", "text_encoder_2.text_projection.alpha"),
        ("unet.down_blocks.0.alpha", "unet.down_blocks.0.alpha"),
    ],
)
def test_canonical_dotted(key, canonical):
    assert list(sdxl_te_names.canonical_dotted({key: 1})) == [canonical]
