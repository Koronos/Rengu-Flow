"""LyCORIS library networks for DiT models (Cosmos Predict2) over ``lycoris_attach``.

Same adapter targets as ``adapter_dit`` (every Linear inside Block/TransformerBlock
modules) and the same export convention as the cosmos LoRA/LoKr files:
``diffusion_model.`` prefix with dotted module paths (not kohya-flat), so the files
load wherever the existing cosmos adapters do.
"""

import re

import safetensors.torch
from pathlib import Path

from rengu_flow.networks import lycoris_attach
from rengu_flow.networks.adapter_dit import ADAPTER_TARGET_MODULES
from rengu_flow.networks.lycoris_meta import create_lycoris_kwargs
from rengu_flow.utils.common import is_main_process
from rengu_flow.utils.save_io import atomic_save_safetensors

EXPORT_PREFIX = "diffusion_model."


def _block_containers(transformer, targets=ADAPTER_TARGET_MODULES):
    # The LLM adapter (Qwen3 conditioning) also holds TransformerBlock modules, but
    # it is frozen by default (llm_adapter_lr = 0) and degrades easily, so it is not
    # an adapter target — only the diffusion blocks are. Skipping it also avoids a
    # DeepSpeed checkpoint-save crash for DyLoRA, whose renamed ParameterList state
    # dict trips exclude_frozen_parameters on those frozen submodules.
    out, seen = [], set()
    for name, module in transformer.named_modules():
        if module.__class__.__name__ not in targets:
            continue
        if "llm_adapter" in name:
            continue
        # Skip blocks nested inside an already-selected container (no double-attach).
        if any(name == s or name.startswith(s + ".") for s in seen):
            continue
        out.append(module)
        seen.add(name)
    return out


def configure(transformer, adapter_config, targets=ADAPTER_TARGET_MODULES):
    """Attach the configured lycoris algorithm to every Linear in the DiT blocks."""
    containers = _block_containers(transformer, targets)
    if not containers:
        raise RuntimeError(
            f"No adapter target blocks ({'/'.join(targets)}) found in transformer"
        )
    # The lycoris backend matches by exact class name "Linear", so quantized linears
    # (Fp8MatmulLinear / bnb Linear4bit) are silently skipped — only the built-in
    # `lokr` is quant-aware. Fail loudly rather than train a partial adapter.
    quantized = {
        type(m).__name__
        for c in containers
        for m in c.modules()
        if type(m).__name__ in ("Fp8MatmulLinear", "Linear4bit")
    }
    if quantized:
        raise RuntimeError(
            f"LyCORIS adapters cannot train on a quantized base ({', '.join(sorted(quantized))}); "
            "they skip quantized layers. Use adapter.type = 'lokr' (quantization-aware)."
        )
    # Prefix "" — cosmos original_name is the raw transformer-relative param name.
    lycoris_attach.configure_roots([(transformer, containers, "")], adapter_config)


def save(save_dir, state_dict, adapter_config):
    """Save the adapter snapshot in the cosmos convention (dotted diffusion_model.* keys)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    exported = lycoris_attach.save_transform(
        state_dict, adapter_config, {"": EXPORT_PREFIX}, flat=False
    )
    algo, _ = create_lycoris_kwargs(adapter_config)
    metadata = {
        "format": "pt",
        "rengu_adapter_type": str(adapter_config["type"]),
        "lycoris_algo": algo,
        "rank": str(adapter_config["rank"]),
    }
    atomic_save_safetensors(save_dir / "adapter_model.safetensors", exported, metadata)


def load(transformer, adapter_path):
    """Load an exported cosmos lycoris file into the attached adapters."""
    adapter_path = Path(adapter_path)
    files = sorted(adapter_path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors file found in {adapter_path}")
    if is_main_process():
        print(f"Loading LyCORIS adapter weights from {adapter_path}")
    state = safetensors.torch.load_file(files[0])
    # Accept the same prefix variants the cosmos lora/lokr loader does.
    state = {re.sub(r"^(transformer|diffusion_model)\.", "", k): v for k, v in state.items()}
    lycoris_attach.load_into([(transformer, "")], state, flat=False)
