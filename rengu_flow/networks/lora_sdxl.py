"""LoRA adapter for SDXL (PEFT). Configure, save (Kohya), load."""

from pathlib import Path

import diffusers
import peft
import safetensors
from torch import nn

from rengu_flow.utils.common import is_main_process
from rengu_flow.utils.save_io import atomic_save_safetensors


def _target_linear_names(containers):
    """Return list of module names that are nn.Linear under the given containers."""
    names = []
    for container in containers:
        for name, submodule in container.named_modules():
            if isinstance(submodule, nn.Linear):
                names.append(name)
    return names


def _wrap_lora(module, containers, prefix, adapter_config):
    rank = adapter_config["rank"]
    alpha = adapter_config["alpha"]
    dropout = adapter_config.get("dropout", 0.0)
    dtype = adapter_config.get("dtype")
    target_modules = _target_linear_names(containers)
    peft_config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
    lora_module = peft.get_peft_model(module, peft_config)
    for name, p in lora_module.named_parameters():
        p.original_name = prefix + name
        if p.requires_grad and dtype is not None:
            p.data = p.data.to(dtype)
    if is_main_process():
        lora_module.print_trainable_parameters()
    return lora_module


def configure(unet, text_encoder, text_encoder_2, adapter_config):
    """Apply LoRA (PEFT) to unet, text_encoder, text_encoder_2. Returns wrapped modules."""
    unet = _wrap_lora(
        unet,
        [unet.down_blocks, unet.mid_block, unet.up_blocks],
        "unet.",
        adapter_config,
    )
    text_encoder = _wrap_lora(text_encoder, [text_encoder], "text_encoder.", adapter_config)
    text_encoder_2 = _wrap_lora(text_encoder_2, [text_encoder_2], "text_encoder_2.", adapter_config)
    return unet, text_encoder, text_encoder_2


_PEFT_WRAPPER = "base_model.model."
_LEGACY_PEFT_WRAPPER = "base_model_model_"


def _strip_peft_wrapper(name):
    """Drop PEFT's ``base_model.model.`` wrapper so keys match the bare module paths.

    ``get_peft_model`` nests the module under ``base_model.model`` and the training-time
    ``original_name`` includes it (``unet.base_model.model.down_blocks...``); exported keys must not.
    """
    return name.replace(_PEFT_WRAPPER, "", 1)


def _to_kohya(state_dict):
    return diffusers.utils.state_dict_utils.convert_state_dict_to_kohya(
        {_strip_peft_wrapper(k): v for k, v in state_dict.items()}
    )


def save(save_dir, state_dict, adapter_config):
    """Save LoRA state_dict in Kohya format to save_dir/lora.safetensors."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    kohya_sd = _to_kohya(state_dict)
    atomic_save_safetensors(save_dir / "lora.safetensors", kohya_sd)


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


def _read_single_safetensors(adapter_path):
    adapter_path = Path(adapter_path)
    files = list(adapter_path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors file found in {adapter_path}")
    if len(files) > 1:
        raise RuntimeError(f"Multiple .safetensors in {adapter_path}; use a single file or directory with one")
    return safetensors.torch.load_file(files[0])


def load_into_wrapped(unet, text_encoder, text_encoder_2, adapter_path):
    """Load a Kohya-format LoRA into the already PEFT-wrapped modules (``init_from_existing``).

    Copies into the existing trainable ``lora_A/B`` parameters, so they keep the adapter dtype and
    the training wrapper (frozen base, original_name, optimizer groups) stays intact. Exports from
    older versions carried a spurious ``base_model_model_`` fragment; it is accepted and dropped.
    """
    state = _read_single_safetensors(adapter_path)
    state = {
        k.replace(_LEGACY_PEFT_WRAPPER, "", 1): v
        for k, v in state.items()
        if not k.endswith(".alpha")
    }
    params = {}
    for prefix, module in (
        ("unet.", unet),
        ("text_encoder.", text_encoder),
        ("text_encoder_2.", text_encoder_2),
    ):
        for name, p in module.named_parameters():
            if "lora_" in name:
                params[prefix + name.replace(".default", "")] = p
    # Convert the whole dict at once (the converter infers the layout from the full key set) and
    # map each kohya key back to its parameter through the tensor identity.
    views = {k: p.detach() for k, p in params.items()}
    by_id = {id(v): params[k] for k, v in views.items()}
    targets = {
        k: by_id[id(v)] for k, v in _to_kohya(views).items() if not k.endswith(".alpha")
    }
    unexpected = [k for k in state if k not in targets]
    if unexpected:
        raise RuntimeError(
            f"LoRA keys in {adapter_path} do not match the model ({len(unexpected)} unexpected, "
            f"e.g. {unexpected[:3]})"
        )
    for k, v in state.items():
        p = targets[k]
        p.data.copy_(v.to(device=p.device, dtype=p.dtype))
    if is_main_process():
        print(f"Loaded {len(state)}/{len(targets)} LoRA tensors from {adapter_path}")
