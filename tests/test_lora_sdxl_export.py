"""SDXL PEFT LoRA export/import: kohya-style keys with no PEFT wrapper fragments (CPU, tiny unet)."""

from __future__ import annotations

import diffusers
import pytest
import safetensors.torch
import torch
from torch import nn

from rengu_flow.networks import lora_sdxl

pytestmark = pytest.mark.no_ui_db

_UNET_CFG = dict(
    sample_size=8,
    in_channels=4,
    out_channels=4,
    layers_per_block=1,
    block_out_channels=(32, 64),
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=32,
    attention_head_dim=4,
    norm_num_groups=8,
)


class _TE(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)


def _build(dtype=torch.float32):
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(**_UNET_CFG)
    cfg = {"rank": 4, "alpha": 4, "dropout": 0.0, "dtype": dtype}
    return lora_sdxl.configure(unet, _TE(), _TE(), cfg)


def _export_state(modules):
    sd = {}
    for prefix, m in zip(("unet.", "text_encoder.", "text_encoder_2."), modules):
        for n, p in m.named_parameters():
            if p.requires_grad:
                sd[prefix + n.replace(".default", "")] = p.detach()
    return sd


def test_export_keys_are_kohya_without_peft_wrapper(tmp_path):
    modules = _build()
    lora_sdxl.save(tmp_path, _export_state(modules), {"type": "lora"})
    keys = list(safetensors.torch.load_file(tmp_path / "lora.safetensors"))
    assert keys
    assert not any("base_model" in k for k in keys)
    assert any(k.startswith("lora_unet_down_blocks_0_") for k in keys)
    assert any(k.startswith("lora_te1_") for k in keys)
    assert any(k.startswith("lora_te2_") for k in keys)
    assert all(k.startswith(("lora_unet_", "lora_te1_", "lora_te2_")) for k in keys)


def test_export_round_trips_into_wrapped_modules_keeping_dtype(tmp_path):
    src = _build()
    for m in src:
        for p in m.parameters():
            if p.requires_grad:
                p.data = torch.randn_like(p)
    lora_sdxl.save(tmp_path, _export_state(src), {"type": "lora"})

    dst = _build(dtype=torch.bfloat16)
    lora_sdxl.load_into_wrapped(*dst, tmp_path)

    src_sd, dst_sd = _export_state(src), _export_state(dst)
    assert src_sd.keys() == dst_sd.keys()
    for k, v in src_sd.items():
        assert dst_sd[k].dtype == torch.bfloat16
        torch.testing.assert_close(dst_sd[k].float(), v.to(torch.bfloat16).float())


def test_load_accepts_legacy_exports_with_wrapper_fragment(tmp_path):
    src = _build()
    for p in src[0].parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p)
    lora_sdxl.save(tmp_path, _export_state(src), {"type": "lora"})
    path = tmp_path / "lora.safetensors"
    legacy = {
        k.replace("lora_unet_", "lora_unet_base_model_model_", 1).replace(
            "lora_te1_", "lora_te1_base_model_model_", 1
        ).replace("lora_te2_", "lora_te2_base_model_model_", 1): v
        for k, v in safetensors.torch.load_file(path).items()
    }
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    safetensors.torch.save_file(legacy, legacy_dir / "lora.safetensors")
    dst = _build()
    lora_sdxl.load_into_wrapped(*dst, legacy_dir)
    for k, v in _export_state(src).items():
        torch.testing.assert_close(_export_state(dst)[k], v)


def test_load_rejects_unknown_keys(tmp_path):
    safetensors.torch.save_file({"lora_unet_nope.lora_down.weight": torch.zeros(1)}, tmp_path / "x.safetensors")
    with pytest.raises(RuntimeError, match="do not match"):
        lora_sdxl.load_into_wrapped(*_build(), tmp_path)


def test_sdxl_pipeline_init_from_existing_wraps_in_adapter_dtype(tmp_path):
    from types import SimpleNamespace

    from rengu_flow.model.sdxl import SDXLPipeline

    src = _build()
    for m in src:
        for p in m.parameters():
            if p.requires_grad:
                p.data = torch.randn_like(p)
    lora_sdxl.save(tmp_path, _export_state(src), {"type": "lora"})

    torch.manual_seed(0)
    pipe = SimpleNamespace(
        unet=diffusers.UNet2DConditionModel(**_UNET_CFG), text_encoder=_TE(), text_encoder_2=_TE()
    )
    model = SDXLPipeline(
        {"model": {"type": "sdxl", "dtype": "float32", "checkpoint_path": "x"}, "optimizer": {}}
    )
    model._pipeline = pipe
    model.configure_adapter(
        {"type": "lora", "rank": 4, "alpha": 4, "dtype": torch.bfloat16, "init_from_existing": str(tmp_path)}
    )
    model.load_adapter_weights(tmp_path)

    trainable = [p for m in (pipe.unet, pipe.text_encoder, pipe.text_encoder_2) for p in m.parameters() if p.requires_grad]
    assert trainable and all(p.dtype == torch.bfloat16 for p in trainable)
    got = _export_state((pipe.unet, pipe.text_encoder, pipe.text_encoder_2))
    for k, v in _export_state(src).items():
        torch.testing.assert_close(got[k].float(), v.to(torch.bfloat16).float())
