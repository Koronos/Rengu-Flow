"""SDXL LoRA load / fuse into an *unwrapped* pipeline (``lora_sdxl.load`` / ``fuse``), numerically.

A real tiny SDXL (``sdxl_testkit``: diffusers UNet + real CLIPTextModel / CLIPTextModelWithProjection)
exports a LoRA whose UNet, TE1 and TE2 factors are all non-zero; the file is loaded back through
``load_adapter_weights`` (no adapter configured: the ``lora_sdxl.load`` path) and fused through
``load_and_fuse_adapter``, and every module's effective delta ``alpha / rank * up @ down`` must
match the file. No mocks: the diffusers / transformers / PEFT versions installed are the ones
under test (transformers 5 once dropped TE2's factors silently on this path).
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from rengu_flow.model import sdxl_testkit as kit
from rengu_flow.networks import sdxl_te_names

pytestmark = pytest.mark.no_ui_db

ROOTS = {"lora_unet_": "unet", "lora_te1_": "text_encoder", "lora_te2_": "text_encoder_2"}


def _export(tmp_path):
    """A trained-looking LoRA export of the tiny SDXL (every factor random, none zero)."""
    src = kit.tiny_pipeline(kit.tiny_config(tmp_path, adapter={"type": "lora", "rank": 2, "alpha": 2}))
    g = torch.Generator().manual_seed(7)
    state = {}
    for net in kit.adapter_networks(src).values():
        for p in net.parameters():
            if p.requires_grad:
                p.data = torch.randn(p.shape, generator=g) * 0.1
                state[p.original_name.replace(".default", "")] = p.detach().clone()
    out = tmp_path / "export"
    src.save_adapter(out, state)
    return out, load_file(str(out / "lora.safetensors"))


def _file_deltas(state) -> dict:
    """``{kohya module: alpha / rank * up @ down}`` of a canonical kohya file."""
    deltas = {}
    for key, down in state.items():
        if key.endswith(".lora_down.weight"):
            base = key[: -len(".lora_down.weight")]
            up = state[base + ".lora_up.weight"]
            alpha = float(state.get(base + ".alpha", torch.tensor(down.shape[0])))
            deltas[base] = (alpha / down.shape[0]) * (up.flatten(1).float() @ down.flatten(1).float())
    return deltas


def _module(pipe, kohya_base):
    """The live module a canonical kohya module name addresses."""
    for prefix, attr in ROOTS.items():
        if kohya_base.startswith(prefix):
            root = getattr(pipe, attr)
            flat = kohya_base[len(prefix):]
            for name, module in root.named_modules():
                adaptable = isinstance(module, (nn.Linear, nn.Conv2d)) or hasattr(module, "base_layer")
                if not adaptable or "lora_" in name or "base_layer" in name:
                    continue
                kohya = sdxl_te_names.canonical_flat({prefix + name.replace(".", "_"): None})
                if next(iter(kohya)) == kohya_base:
                    return module
            raise KeyError(kohya_base)
    raise KeyError(kohya_base)


def _loaded_delta(module) -> torch.Tensor:
    a = module.lora_A["default"].weight
    b = module.lora_B["default"].weight
    return module.scaling["default"] * (b.flatten(1).float() @ a.flatten(1).float())


def _fresh(tmp_path):
    return kit.tiny_pipeline(kit.tiny_config(tmp_path))


def _assert_loaded(pipe, state):
    deltas = _file_deltas(state)
    assert {k.split("_")[1] for k in deltas} == {"unet", "te1", "te2"}
    for base, expected in deltas.items():
        assert expected.abs().sum() > 0, base
        torch.testing.assert_close(_loaded_delta(_module(pipe, base)), expected, msg=base)


def test_load_puts_every_unet_te1_te2_factor_in_place(tmp_path):
    path, state = _export(tmp_path)
    pipe = _fresh(tmp_path)
    pipe.load_adapter_weights(path)  # no adapter configured: lora_sdxl.load
    _assert_loaded(pipe, state)


def test_load_accepts_legacy_flat_te1_keys(tmp_path):
    """transformers-5 exports briefly wrote TE1 without ``text_model`` (``lora_te1_encoder_...``)."""
    path, state = _export(tmp_path)
    flat = {
        ("lora_te1_" + k[len("lora_te1_text_model_"):] if k.startswith("lora_te1_text_model_") else k): v
        for k, v in state.items()
    }
    assert any(k.startswith("lora_te1_encoder_") for k in flat)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    save_file(flat, str(legacy / "lora.safetensors"))
    pipe = _fresh(tmp_path)
    pipe.load_adapter_weights(legacy)
    _assert_loaded(pipe, state)


def test_fuse_adds_exactly_the_file_delta_to_every_base_weight(tmp_path):
    path, state = _export(tmp_path)
    pipe = _fresh(tmp_path)
    before = {base: _module(pipe, base).weight.detach().clone() for base in _file_deltas(state)}
    pipe.load_and_fuse_adapter(path)
    for net in (pipe.unet, pipe.text_encoder, pipe.text_encoder_2):
        assert not any("lora_" in n for n, _ in net.named_parameters()), "LoRA layers left after fuse"
    for base, delta in _file_deltas(state).items():
        weight = _module(pipe, base).weight.detach()
        torch.testing.assert_close((weight - before[base]).flatten(1).float(), delta, msg=base)


@pytest.mark.parametrize(
    "defect",
    ["unknown_module", "missing_up", "foreign_suffix"],
)
def test_load_fails_loudly_on_keys_it_cannot_place(tmp_path, defect):
    path, state = _export(tmp_path)
    bad = dict(state)
    victim = next(k for k in bad if k.startswith("lora_te2_") and k.endswith(".lora_up.weight"))
    if defect == "unknown_module":
        bad["lora_te2_text_model_encoder_layers_9_mlp_fc1.lora_down.weight"] = torch.ones(2, 24)
        bad["lora_te2_text_model_encoder_layers_9_mlp_fc1.lora_up.weight"] = torch.ones(32, 2)
    elif defect == "missing_up":
        del bad[victim]
    else:
        bad[victim.replace(".lora_up.weight", ".dora_scale")] = torch.ones(1, 24)
    broken = tmp_path / "broken"
    broken.mkdir()
    save_file(bad, str(broken / "lora.safetensors"))
    with pytest.raises(RuntimeError, match="LoRA"):
        _fresh(tmp_path).load_adapter_weights(broken)
