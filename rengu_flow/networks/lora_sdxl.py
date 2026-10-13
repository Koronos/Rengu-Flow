"""LoRA adapter for SDXL (PEFT). Configure, save (Kohya), load."""

import re
from pathlib import Path

import diffusers
import peft
import safetensors
import torch
from torch import nn

from rengu_flow.networks import sdxl_te_names
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
    kohya = diffusers.utils.state_dict_utils.convert_state_dict_to_kohya(
        {_strip_peft_wrapper(k): v for k, v in state_dict.items()}
    )
    # Always the canonical ``lora_te1_text_model_...`` names (ComfyUI / A1111), whatever the
    # transformers version's CLIPTextModel looks like.
    return sdxl_te_names.canonical_flat(kohya)


def save(save_dir, state_dict, adapter_config):
    """Save LoRA state_dict in Kohya format to save_dir/lora.safetensors."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    kohya_sd = _to_kohya(state_dict)
    atomic_save_safetensors(save_dir / "lora.safetensors", kohya_sd)


# Kohya weight names of one LoRA module (``<module>.<name>``).
_DOWN, _UP, _ALPHA = "lora_down.weight", "lora_up.weight", "alpha"
_ROOTS = (("unet.", "unet"), ("text_encoder.", "text_encoder"), ("text_encoder_2.", "text_encoder_2"))


def _kohya_module_map(named_roots):
    """``{canonical kohya module name: (root, module path)}`` of every Linear/Conv2d under the roots.

    Built through the same ``_to_kohya`` the export uses (one probe key per module), so the names
    match whatever the module paths look like in the installed diffusers / transformers.
    """
    probes, owners = {}, {}
    for prefix, root in named_roots:
        for name, module in root.named_modules():
            if name and isinstance(module, (nn.Linear, nn.Conv2d)):
                probe = torch.empty(1)
                probes[f"{prefix}{name}.lora_A.weight"] = probe
                owners[id(probe)] = (root, name)
    suffix = "." + _DOWN
    return {k[: -len(suffix)]: owners[id(v)] for k, v in _to_kohya(probes).items() if k.endswith(suffix)}


def _read_kohya(adapter_path):
    """The file's weights under canonical kohya names (legacy ``base_model_model_`` fragment and
    flat transformers-5 TE1 names accepted)."""
    state = _read_single_safetensors(adapter_path)
    return sdxl_te_names.canonical_flat({k.replace(_LEGACY_PEFT_WRAPPER, "", 1): v for k, v in state.items()})


def load(pipeline, adapter_path):
    """Load a Kohya-format LoRA (``lora_unet_*`` / ``lora_te1_*`` / ``lora_te2_*``) into the
    *unwrapped* UNet and text encoders of ``pipeline``: PEFT LoRA layers are injected in place on
    exactly the modules the file names (per-module rank and alpha) and the factors are copied in.

    Not ``pipeline.load_lora_weights``: with transformers 5 the text encoders' adapter load goes
    through transformers' key conversion, which strips ``text_model.`` for every
    ``clip_text_model`` config, also for CLIPTextModelWithProjection (TE2) that still has the
    submodule; TE2's factors then stay at their init (``lora_B`` = 0) with no error. Every key of
    the file must land on a module, otherwise this raises (nothing is loaded silently partial).
    The loaded factors are inference weights (``requires_grad`` False); the base modules keep their
    own ``requires_grad``.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    adapter_path = Path(adapter_path)
    if is_main_process():
        print(f"Loading LoRA adapter weights from {adapter_path}")
    state = _read_kohya(adapter_path)
    roots = [(prefix, getattr(pipeline, attr)) for prefix, attr in _ROOTS]
    for prefix, root in roots:
        if any(isinstance(m, BaseTunerLayer) for m in root.modules()):
            raise RuntimeError(
                f"{prefix.rstrip('.')} already carries LoRA layers; load into a configured adapter with "
                "load_into_wrapped, or fuse the loaded one first"
            )
    modules = _kohya_module_map(roots)
    entries, unexpected = {}, []
    for key, value in state.items():
        base, _, weight = key.partition(".")
        if base in modules and weight in (_DOWN, _UP, _ALPHA):
            entries.setdefault(base, {})[weight] = value
        else:
            unexpected.append(key)
    incomplete = sorted(base for base, parts in entries.items() if not {_DOWN, _UP} <= parts.keys())
    if unexpected or incomplete or not entries:
        raise RuntimeError(
            f"LoRA file in {adapter_path} does not match the SDXL model: {len(unexpected)} keys name no "
            f"module / weight (e.g. {sorted(unexpected)[:3]}), {len(incomplete)} modules lack a "
            f"lora_down/lora_up pair (e.g. {incomplete[:3]})"
        )

    per_root = {}
    for base, parts in entries.items():
        root, name = modules[base]
        per_root.setdefault(id(root), (root, {}))[1][name] = parts
    loaded = 0
    for root, targets in per_root.values():
        ranks = {name: int(parts[_DOWN].shape[0]) for name, parts in targets.items()}
        alphas = {name: float(parts[_ALPHA]) if _ALPHA in parts else float(ranks[name]) for name, parts in targets.items()}
        rank = max(set(ranks.values()), key=list(ranks.values()).count)
        alpha = max(set(alphas.values()), key=list(alphas.values()).count)
        config = peft.LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=sorted(targets),
            rank_pattern={re.escape(n): r for n, r in ranks.items() if r != rank},
            alpha_pattern={re.escape(n): a for n, a in alphas.items() if a != alpha},
            bias="none",
        )
        trainable = {name: p.requires_grad for name, p in root.named_parameters()}
        peft.inject_adapter_in_model(config, root, adapter_name="default")
        for name, p in root.named_parameters():
            p.requires_grad_(trainable.get(name.replace(".base_layer.", "."), False) and "lora_" not in name)
        for name, parts in targets.items():
            layer = root.get_submodule(name)
            for factor, weight in ((layer.lora_A["default"].weight, parts[_DOWN]), (layer.lora_B["default"].weight, parts[_UP])):
                if factor.numel() != weight.numel():
                    raise RuntimeError(
                        f"LoRA file in {adapter_path}: {name} factor of shape {tuple(weight.shape)} does not fit "
                        f"{tuple(factor.shape)}"
                    )
                with torch.no_grad():
                    factor.copy_(weight.reshape(factor.shape).to(device=factor.device, dtype=factor.dtype))
            loaded += 1
    if is_main_process():
        print(f"Loaded {loaded} LoRA modules ({len(state)} tensors) from {adapter_path}")


def fuse(pipeline):
    """Merge the LoRA layers ``load`` injected into the base weights and remove them, leaving the
    plain modules (no adapter layers, no ``peft_config``)."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    for _, attr in _ROOTS:
        root = getattr(pipeline, attr)
        for name, module in list(root.named_modules()):
            if isinstance(module, BaseTunerLayer):
                module.merge(safe_merge=True)
                parent, _, child = name.rpartition(".")
                setattr(root.get_submodule(parent) if parent else root, child, module.get_base_layer())
        if hasattr(root, "peft_config"):
            del root.peft_config


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
    state = sdxl_te_names.canonical_flat(
        {
            k.replace(_LEGACY_PEFT_WRAPPER, "", 1): v
            for k, v in state.items()
            if not k.endswith(".alpha")
        }
    )
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
