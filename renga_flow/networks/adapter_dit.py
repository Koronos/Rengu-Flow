"""LoRA / LoKr adapters for DiT models (Cosmos Predict2)."""

from __future__ import annotations

import re
from pathlib import Path

import peft
import safetensors
import torch
from torch import nn

from renga_flow.networks.lokr_sdxl import _apply_lokr_vendored, _inject_lokr_into_linear
from renga_flow.utils.common import is_main_process

ADAPTER_TARGET_MODULES = ("Block", "TransformerBlock")


def _collect_target_linears(transformer, target_module_names):
    names = set()
    for name, module in transformer.named_modules():
        if module.__class__.__name__ not in target_module_names:
            continue
        for full_name, submodule in module.named_modules(prefix=name):
            if isinstance(submodule, nn.Linear):
                names.add(full_name)
    return list(names)


def configure(transformer, adapter_config):
    target_linear_modules = _collect_target_linears(transformer, ADAPTER_TARGET_MODULES)
    adapter_type = adapter_config["type"]
    if adapter_type == "lora":
        peft_config = peft.LoraConfig(
            r=adapter_config["rank"],
            lora_alpha=adapter_config["alpha"],
            lora_dropout=adapter_config.get("dropout", 0.0),
            bias="none",
            target_modules=target_linear_modules,
        )
        lora_model = peft.get_peft_model(transformer, peft_config)
        if is_main_process():
            lora_model.print_trainable_parameters()
        return peft_config, adapter_type
    if adapter_type == "lokr":
        for p in transformer.parameters():
            p.requires_grad_(False)
        _apply_lokr_vendored(transformer, target_linear_modules, adapter_config, "")
        return None, adapter_type
    raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")


def save(save_dir, state_dict, adapter_config, peft_config=None):
    save_dir = Path(save_dir)
    if adapter_config["type"] == "lokr":
        lokr_modules = set()
        for k in list(state_dict.keys()):
            if ".lokr_w1" in k or ".lokr_w2" in k:
                module_name = k.rsplit(".lokr_", 1)[0]
                lokr_modules.add(module_name)
        alpha_value = adapter_config["alpha"]
        for module_name in lokr_modules:
            state_dict[f"{module_name}.alpha"] = torch.tensor(float(alpha_value))
        state_dict = {"diffusion_model." + k: v for k, v in state_dict.items()}
        safetensors.torch.save_file(
            state_dict, save_dir / "adapter_model.safetensors", metadata={"format": "pt"}
        )
    else:
        if peft_config is not None:
            peft_config.save_pretrained(save_dir)
        state_dict = {"diffusion_model." + k: v for k, v in state_dict.items()}
        safetensors.torch.save_file(
            state_dict, save_dir / "adapter_model.safetensors", metadata={"format": "pt"}
        )


def load_weights(transformer, adapter_path):
    if is_main_process():
        print(f"Loading adapter weights from path {adapter_path}")
    safetensors_files = list(Path(adapter_path).glob("*.safetensors"))
    if len(safetensors_files) == 0:
        raise RuntimeError(f"No safetensors file found in {adapter_path}")
    if len(safetensors_files) > 1:
        raise RuntimeError(f"Multiple safetensors files found in {adapter_path}")
    adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
    is_lokr = any("lokr_" in k for k in adapter_state_dict.keys())
    modified_state_dict = {}
    model_parameters = set(name for name, _ in transformer.named_parameters())
    for k, v in adapter_state_dict.items():
        k = re.sub(r"^(transformer|diffusion_model)\.", "", k)
        if is_lokr:
            if k.endswith(".alpha"):
                continue
        else:
            k = re.sub(r"\.weight$", ".default.weight", k)
        if k not in model_parameters:
            raise RuntimeError(f"modified_state_dict key {k} is not in the model parameters")
        modified_state_dict[k] = v
    transformer.load_state_dict(modified_state_dict, strict=False)
