"""Model-agnostic attachment of LyCORIS library networks (DeepSpeed-pipeline safe).

``create_lycoris()`` hangs every adapter module off a separate network object —
outside the model tree ``to_layers()`` flattens, so DeepSpeed would never place or
train those params. Each lycoris module, however, keeps its wrapped layer in a
plain (unregistered) ``org_module`` list and rebinds ``forward`` as an instance
attribute, so the module itself is indifferent to where it lives in the tree.
``configure_roots()`` exploits that: it re-parents each lycoris module onto the
very layer it wraps (``setattr(layer, "lycoris_adapter", module)``) and discards
the network container. Params then ride the pipeline layers, and the existing
``original_name``-keyed param-group/export plumbing works unchanged.

Adding lycoris networks to another model is a thin wrapper over this module: map
its submodules to ``(module, containers, prefix)`` roots for configure and to
``(module, kohya_prefix)`` roots for load/fuse (see ``lycoris_sdxl``).
"""

import torch

from rengu_flow.networks.lycoris_meta import (
    ALPHA_IS_CONSTRAINT,
    SCALAR_FOLD_TARGETS,
    create_lycoris_kwargs,
)
from rengu_flow.utils.common import is_main_process

ATTACH_ATTR = "lycoris_adapter"
_PRESET_NAME = "rengu"


def _register_preset(train_conv: bool, lora_prefix: str = "lycoris", train_norm: bool = False) -> str:
    """Register a complete preset under a fixed name and return that name.

    ``LycorisNetwork.apply_preset`` mutates class-level state key by key, so a
    partial preset would inherit whatever a previous caller left behind — every
    key is therefore always written.
    """
    import lycoris.config

    target_module = ["Linear", "Conv2d"] if train_conv else ["Linear"]
    if train_norm:
        # NormModule only supports affine LayerNorm/GroupNorm; the train_norm kwarg
        # routes these classes to it instead of the main algorithm.
        target_module += ["LayerNorm", "GroupNorm"]
    lycoris.config.PRESET[_PRESET_NAME] = {
        "enable_conv": train_conv,
        "target_module": target_module,
        "target_name": [],
        "module_algo_map": {},
        "name_algo_map": {},
        "lora_prefix": lora_prefix,
        "use_fnmatch": False,
        "exclude_name": [],
    }
    return _PRESET_NAME


def configure_roots(roots, adapter_config):
    """Attach a lycoris network to each root, re-parented into the model tree.

    ``roots`` is a list of ``(module, containers, prefix)``: the module whose
    params get ``original_name = prefix + name``, and the containers under it that
    actually receive adapters (e.g. only the unet's down/mid/up blocks).
    """
    from fnmatch import fnmatch

    from lycoris import create_lycoris

    algo, algo_kwargs = create_lycoris_kwargs(adapter_config)
    rank = adapter_config["rank"]
    alpha = adapter_config["alpha"]
    dtype = adapter_config.get("dtype")
    train_norm = bool(adapter_config.get("train_norm", False))
    rs_lora = bool(adapter_config.get("rs_lora", False))
    include = adapter_config.get("target_include") or ["*"]
    exclude = adapter_config.get("target_exclude") or []
    preset = _register_preset(adapter_config.get("train_conv", False), train_norm=train_norm)

    total_modules = 0
    norm_modules = 0
    filtered_out = 0
    for module, containers, prefix in roots:
        first = next(module.parameters(), None)
        if first is not None and first.device.type == "meta":
            # Frozen-out submodel (e.g. text encoders with cached embeddings are
            # moved to meta after caching): its params never join the pipeline, and
            # DoRA init must read real base weights. Skip instead of attaching
            # dead adapters.
            if is_main_process():
                print(f"LyCORIS: skipping {prefix.rstrip('.')} (on meta device, frozen out of training)")
            continue
        for p in module.parameters():
            p.requires_grad_(False)
        # Full dotted path per wrapped layer, for target_include/exclude globs.
        path_of = {id(m): prefix + name for name, m in module.named_modules()}
        for container in containers:
            if container is None:
                continue
            net = create_lycoris(
                container,
                1.0,
                linear_dim=rank,
                linear_alpha=alpha,
                algo=algo,
                preset=preset,
                bypass_mode=False,
                **algo_kwargs,
            )
            for lora in net.loras:
                if getattr(lora, "not_supported", False):
                    continue
                org = lora.org_module[0]
                path = path_of[id(org)]
                if not any(fnmatch(path, pat) for pat in include) or any(
                    fnmatch(path, pat) for pat in exclude
                ):
                    filtered_out += 1
                    continue
                if hasattr(org, ATTACH_ATTR):
                    raise RuntimeError(
                        f"Module already has a {ATTACH_ATTR}; configure_roots called twice?"
                    )
                if rs_lora and type(lora).__name__ == "LoConModule":
                    _apply_rs_lora(lora)
                lora.apply_to()
                lora.requires_grad_(True)
                setattr(org, ATTACH_ATTR, lora)
                total_modules += 1
                if type(lora).__name__ == "NormModule":
                    norm_modules += 1
            del net
        for name, p in module.named_parameters():
            p.original_name = prefix + name
            if p.requires_grad and dtype is not None:
                p.data = p.data.to(dtype)

    if total_modules == 0:
        raise RuntimeError(
            "LyCORIS attached 0 modules — target_include/target_exclude filtered "
            f"everything out ({filtered_out} candidates dropped)."
        )
    if train_norm and norm_modules == 0:
        raise RuntimeError(
            "train_norm = true but no trainable LayerNorm/GroupNorm matched. The "
            "Cosmos DiT has no affine norm weights (train_norm is SDXL-only); on "
            "SDXL check your target_include/target_exclude patterns."
        )
    if is_main_process():
        trainable = sum(
            p.numel() for module, _, _ in roots for p in module.parameters() if p.requires_grad
        )
        total = sum(p.numel() for module, _, _ in roots for p in module.parameters())
        extras = []
        if norm_modules:
            extras.append(f"{norm_modules} norm")
        if filtered_out:
            extras.append(f"{filtered_out} filtered out by targets")
        suffix = f" ({', '.join(extras)})" if extras else ""
        print(f"LyCORIS ({algo}): attached {total_modules} modules{suffix}")
        print(
            f"  trainable params: {trainable:,d} || all params: {total:,d} "
            f"|| trainable%: {100 * trainable / total:.4f}"
        )


def _apply_rs_lora(lora):
    """Rank-stabilized scaling on a LoConModule created without it.

    ``create_lycoris`` does not forward ``rs_lora`` to the modules, so reproduce
    its two effects post-init (locon.py: r_factor = sqrt(lora_dim)):
    ``scale = alpha / sqrt(r)`` and the exported-alpha buffer ``alpha * sqrt(r)``.
    """
    import math

    r = lora.lora_dim
    cfg_alpha = float(lora.alpha.item())  # buffer still holds the plain config alpha
    lora.rs_lora = True
    lora.scale = cfg_alpha / math.sqrt(r)
    lora.alpha.fill_(cfg_alpha * math.sqrt(r))


def iter_attached_adapters(module):
    """Yield ``(module_path, lycoris_module)`` for every adapter attached under module."""
    for name, sub in module.named_modules():
        if name == ATTACH_ATTR or name.endswith("." + ATTACH_ATTR):
            path = name[: -len(ATTACH_ATTR)].rstrip(".")
            yield path, sub


def _module_name(prefix, module_path, flat):
    """Exported per-module name: kohya-flat (dots -> underscores, SDXL convention)
    or dotted (Comfy DiT convention, e.g. ``diffusion_model.blocks.0...``)."""
    return prefix + (module_path.replace(".", "_") if flat else module_path)


def _sorted_list_params(collected, list_name):
    keys = [k for k in collected if k.startswith(list_name + ".")]
    keys.sort(key=lambda k: int(k.rsplit(".", 1)[1]))
    return [collected.pop(k) for k in keys]


def save_transform(state_dict, adapter_config, prefix_map, *, flat=True):
    """Snapshot ({original_name: tensor}) -> kohya-flat lycoris state dict.

    Replicates each module's ``custom_state_dict``: per-module ``alpha`` (rank, or
    the constraint for the OFT family), DyLoRA block lists concatenated into
    ``lora_up/down``, and a trained ``use_scalar`` folded into its target tensors.
    """
    adapter_type = adapter_config["type"]
    algo, _ = create_lycoris_kwargs(adapter_config)
    if adapter_type in ALPHA_IS_CONSTRAINT:
        alpha_value = float(adapter_config["constraint"])
    elif adapter_config.get("rs_lora"):
        # Loaders compute scale = alpha / rank; rank-stabilized training used
        # alpha / sqrt(rank), so export alpha * sqrt(rank) (matches the upstream
        # rs_lora alpha buffer).
        import math

        alpha_value = float(adapter_config["alpha"]) * math.sqrt(adapter_config["rank"])
    else:
        alpha_value = float(adapter_config["alpha"])

    marker = "." + ATTACH_ATTR + "."
    per_module: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if marker not in key:
            raise RuntimeError(f"Unexpected non-lycoris key in adapter snapshot: {key}")
        module_path, param_name = key.split(marker, 1)
        root = next((p for p in prefix_map if module_path.startswith(p)), None)
        if root is None:
            raise RuntimeError(f"Adapter key {key} matches no known root prefix")
        name = _module_name(prefix_map[root], module_path[len(root):], flat)
        per_module.setdefault(name, {})[param_name] = tensor

    out = {}
    for mod_name, collected in per_module.items():
        if algo == "dylora":
            ups = _sorted_list_params(collected, "up_list")
            downs = _sorted_list_params(collected, "down_list")
            collected["lora_up.weight"] = torch.cat(ups, dim=1)
            collected["lora_down.weight"] = torch.cat(downs, dim=0).reshape(
                int(adapter_config["rank"]), -1
            )
        scalar = collected.pop("scalar", None)
        if scalar is not None:
            for target in SCALAR_FOLD_TARGETS[algo]:
                if target in collected:
                    collected[target] = collected[target] * scalar.to(collected[target].dtype)
        for param_name, tensor in collected.items():
            out[f"{mod_name}.{param_name}"] = tensor.contiguous()
        if set(collected) <= {"w_norm", "b_norm"}:
            # train_norm module: upstream NormModule state dicts carry no alpha.
            continue
        out[f"{mod_name}.alpha"] = torch.tensor(alpha_value)
    return out


def load_into(roots, state_dict, *, flat=True):
    """Load an exported lycoris state dict into already-attached adapters.

    ``roots`` is a list of ``(module, prefix)``; ``flat`` selects the same naming
    style used at save time. Strict: every attached param must be fed from the
    file, except ``scalar`` (folded away at export — reset to 1 like the upstream
    load hook) — and every consumed key is tracked so leftovers raise instead of
    silently not loading.
    """
    consumed = set()
    missing = []
    for module, prefix in roots:
        for module_path, lora in iter_attached_adapters(module):
            flat_name = _module_name(prefix, module_path, flat)
            consumed.add(f"{flat_name}.alpha")
            if type(lora).__name__ == "DyLoraModule":
                _load_dylora(lora, flat_name, state_dict, consumed, missing)
                continue
            for pname, p in lora.named_parameters():
                if pname == "scalar":
                    p.data.copy_(torch.ones_like(p.data))
                    continue
                key = f"{flat_name}.{pname}"
                tensor = state_dict.get(key)
                if tensor is None:
                    missing.append(key)
                    continue
                p.data.copy_(tensor.to(p.device, p.dtype))
                consumed.add(key)
    if missing:
        raise RuntimeError(
            f"Adapter file is missing {len(missing)} expected keys, e.g. {missing[:3]}"
        )
    leftover = [k for k in state_dict if k not in consumed]
    if leftover:
        raise RuntimeError(
            f"Adapter file has {len(leftover)} keys that match no attached module, "
            f"e.g. {leftover[:3]}"
        )


def _load_dylora(lora, flat, state_dict, consumed, missing):
    """Upstream DyLoraModule.load_state_dict is a no-op; split the concatenated
    export back into the block lists manually."""
    up_key, down_key = f"{flat}.lora_up.weight", f"{flat}.lora_down.weight"
    up, down = state_dict.get(up_key), state_dict.get(down_key)
    if up is None or down is None:
        missing.extend(k for k, v in ((up_key, up), (down_key, down)) if v is None)
        return
    block = lora.block_size
    down = down.reshape(down.shape[0], -1)
    for i, p in enumerate(lora.up_list):
        p.data.copy_(up[:, i * block : (i + 1) * block].to(p.device, p.dtype))
    for i, p in enumerate(lora.down_list):
        p.data.copy_(down[i * block : (i + 1) * block].to(p.device, p.dtype))
    consumed.update((up_key, down_key))


def fuse_all(modules):
    """Merge every attached adapter into its base layer and detach it."""
    fused = 0
    for module in modules:
        for _path, lora in list(iter_attached_adapters(module)):
            lora.merge_to(1.0)
            lora.restore()
            delattr(lora.org_module[0], ATTACH_ATTR)
            fused += 1
    if is_main_process():
        print(f"LyCORIS: fused {fused} adapter modules into base weights")
    return fused


def fuse_weights_into(roots, state_dict):
    """Merge a kohya-flat lycoris file directly into base weights (no configure).

    Builds modules straight from the weights via ``create_lycoris_from_weights``
    (which infers the algorithm per module from its key family), merges, and drops
    them. ``roots`` is a list of ``(module, kohya_prefix)``.
    """
    from lycoris import create_lycoris_from_weights
    from lycoris.wrapper import LycorisNetwork

    train_conv = any(k.endswith(".lora_mid.weight") for k in state_dict)
    merged = 0
    for module, kohya_prefix in roots:
        # create_lycoris_from_weights matches modules as f"{LORA_PREFIX}_{name}".
        preset = _register_preset(train_conv, lora_prefix=kohya_prefix.rstrip("_"))
        import lycoris.config

        LycorisNetwork.apply_preset(lycoris.config.PRESET[preset])
        net, _ = create_lycoris_from_weights(1.0, None, module, weights_sd=state_dict)
        net.merge_to(1.0)
        merged += len(net.loras)
        del net
    if is_main_process():
        print(f"LyCORIS: merged {merged} modules from weights into base model")
    return merged
