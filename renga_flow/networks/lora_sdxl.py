"""LoRA adapter for SDXL (PEFT). Configure, save (Kohya), load."""

from pathlib import Path

import diffusers
import peft
import safetensors
from torch import nn

from renga_flow.utils.common import is_main_process


def _target_linear_names(containers):
    """Return list of module names that are nn.Linear under the given containers."""
    names = []
    for container in containers:
        for name, submodule in container.named_modules():
            if isinstance(submodule, nn.Linear):
                names.append(name)
    return names


def configure(unet, text_encoder, text_encoder_2, adapter_config):
    """Apply LoRA (PEFT) to unet, text_encoder, text_encoder_2. Set original_name and dtype."""
    rank = adapter_config["rank"]
    alpha = adapter_config["alpha"]
    dropout = adapter_config.get("dropout", 0.0)
    dtype = adapter_config.get("dtype")

    def add_to_module(module, containers, prefix):
        target_modules = _target_linear_names(containers)
        peft_config = peft.LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_modules,
        )
        module.add_adapter(peft_config)
        for name, p in module.named_parameters():
            p.original_name = prefix + name
            if p.requires_grad and dtype is not None:
                p.data = p.data.to(dtype)
        if is_main_process():
            module.print_trainable_parameters()

    add_to_module(
        unet,
        [unet.down_blocks, unet.mid_block, unet.up_blocks],
        "unet.",
    )
    add_to_module(text_encoder, [text_encoder], "text_encoder.")
    add_to_module(text_encoder_2, [text_encoder_2], "text_encoder_2.")


def save(save_dir, state_dict, adapter_config):
    """Save LoRA state_dict in Kohya format to save_dir/lora.safetensors."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    kohya_sd = diffusers.utils.state_dict_utils.convert_state_dict_to_kohya(state_dict)
    safetensors.torch.save_file(kohya_sd, save_dir / "lora.safetensors", metadata={"format": "pt"})


def load(pipeline, adapter_path):
    """Load LoRA weights from adapter_path (dir with .safetensors) into pipeline."""
    adapter_path = Path(adapter_path)
    if is_main_process():
        print(f"Loading LoRA adapter weights from {adapter_path}")
    files = list(adapter_path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors file found in {adapter_path}")
    if len(files) > 1:
        raise RuntimeError(f"Multiple .safetensors in {adapter_path}; use a single file or directory with one")
    state = safetensors.torch.load_file(files[0])
    pipeline.load_lora_weights(state, adapter_name="default")
