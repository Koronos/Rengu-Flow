"""Programmatic verification of exported LyCORIS adapter files.

Usage:
    python -m rengu_flow.networks.lycoris_export_check <file.safetensors> --algo lycoris_loha
    ... --base-checkpoint /path/sdxl.safetensors   # optional deep rebuild (slow, manual)

Checks, in order: key families per algorithm (required present, no foreign keys,
either-groups consistent), kohya prefixes (``lora_unet_`` present, only known
prefixes), every tensor finite, ``alpha`` per module, a nonzero training delta on
the algorithm's effective tensor, and a structural parse of every module through
``lycoris.modules.get_module``. Exit code 0 = all pass.
"""

import argparse
import sys
from pathlib import Path

import safetensors
import safetensors.torch
import torch

from rengu_flow.networks.lycoris_meta import LYCORIS_ADAPTER_TYPES, WEIGHT_KEY_FAMILIES

# adapter type -> class name get_module must resolve every module to. DyLoRA
# deploys as plain up/down matrices, so its export parses as LoCon by design.
EXPECTED_MODULE_CLASS = {
    "lycoris_locon": "LoConModule",
    "lycoris_dora": "LoConModule",
    "lycoris_dylora": "LoConModule",
    "lycoris_loha": "LohaModule",
    "lycoris_lokr": "LokrModule",
    "lycoris_glora": "GLoRAModule",
    "lycoris_diag_oft": "DiagOFTModule",
    "lycoris_boft": "ButterflyOFTModule",
}

# Per export style: required module prefix and all accepted prefixes. "kohya"
# (SDXL) modules are flat (no dots), so keys split at the first dot; "cosmos"
# modules keep dotted paths, so keys split by matching known weight-name suffixes.
STYLE_PREFIXES = {
    "kohya": ("lora_unet_", ("lora_unet_", "lora_te1_", "lora_te2_")),
    "cosmos": ("diffusion_model.", ("diffusion_model.",)),
}


def _family_weight_names(fam):
    names = {"alpha", *fam["required"], *fam["optional"]}
    for group in fam["either"]:
        for alt in group:
            names.update(alt)
    return names


def check_export(path, adapter_type, style="kohya"):
    """Return a list of failure messages (empty = file passes)."""
    fam = WEIGHT_KEY_FAMILIES[adapter_type]
    required_prefix, known_prefixes = STYLE_PREFIXES[style]
    state = safetensors.torch.load_file(path)
    failures = []

    weight_names = sorted(_family_weight_names(fam), key=len, reverse=True)
    per_module: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if style == "kohya":
            module_name, _, sub = key.partition(".")
        else:
            module_name = sub = ""
            for w in weight_names:
                if key.endswith("." + w):
                    module_name, sub = key[: -len(w) - 1], w
                    break
        if not sub or not module_name:
            failures.append(f"key matches no known module/weight split: {key}")
            continue
        per_module.setdefault(module_name, {})[sub] = tensor

    if not any(m.startswith(required_prefix) for m in per_module):
        failures.append(f"no {required_prefix}* modules in file")
    unknown_prefix = [m for m in per_module if not m.startswith(known_prefixes)]
    if unknown_prefix:
        failures.append(f"unknown module prefixes: {unknown_prefix[:3]}")

    allowed = set(fam["required"]) | set(fam["optional"])
    for group in fam["either"]:
        for alt in group:
            allowed.update(alt)
    deltas = fam["delta"] if isinstance(fam["delta"], tuple) else (fam["delta"],)

    zero_delta = []
    for module_name, subs in per_module.items():
        missing = set(fam["required"]) - set(subs)
        if missing:
            failures.append(f"{module_name}: missing required keys {sorted(missing)}")
        foreign = set(subs) - allowed
        if foreign:
            failures.append(f"{module_name}: foreign keys {sorted(foreign)}")
        for group in fam["either"]:
            hits = [alt for alt in group if set(alt) <= set(subs)]
            if len(hits) != 1:
                failures.append(
                    f"{module_name}: expected exactly one of {group}, found {len(hits)}"
                )
        alpha = subs.get("alpha")
        if alpha is not None and (alpha.numel() != 1 or not torch.isfinite(alpha).all()):
            failures.append(f"{module_name}: malformed alpha {alpha}")
        for sub, tensor in subs.items():
            if not torch.isfinite(tensor).all():
                failures.append(f"{module_name}.{sub}: non-finite values")
        if not any(s in subs and subs[s].abs().sum() > 0 for s in deltas):
            zero_delta.append(module_name)
    if zero_delta:
        failures.append(
            f"{len(zero_delta)}/{len(per_module)} modules have a zero training delta "
            f"on {deltas}, e.g. {zero_delta[:3]}"
        )

    # Structural parse: the library itself must recognize every module.
    from lycoris.modules import get_module

    expected_cls = EXPECTED_MODULE_CLASS[adapter_type]
    for module_name in per_module:
        cls, params = get_module(state, module_name)
        if cls is None:
            failures.append(f"{module_name}: lycoris get_module cannot parse")
        elif cls.__name__ != expected_cls:
            failures.append(
                f"{module_name}: parsed as {cls.__name__}, expected {expected_cls}"
            )
    return failures, len(per_module)


def deep_rebuild(path, base_checkpoint):
    """Merge the file into a real SDXL checkpoint via the library (manual, slow)."""
    import diffusers

    from rengu_flow.networks import lycoris_sdxl

    pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
        base_checkpoint, torch_dtype=torch.float32, add_watermarker=False
    )
    lycoris_sdxl.load_and_fuse(pipe, Path(path).parent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--algo", required=True, choices=LYCORIS_ADAPTER_TYPES)
    parser.add_argument("--style", default="kohya", choices=sorted(STYLE_PREFIXES))
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)

    with safetensors.safe_open(args.file, framework="pt") as f:
        metadata = f.metadata() or {}
    declared = metadata.get("rengu_adapter_type")
    if declared and declared != args.algo:
        print(f"FAIL: file metadata declares {declared}, checking as {args.algo}")
        return 1

    failures, n_modules = check_export(args.file, args.algo, style=args.style)
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1

    if args.base_checkpoint is not None:
        deep_rebuild(args.file, args.base_checkpoint)
        print(f"deep rebuild against {args.base_checkpoint}: merged OK")

    print(f"OK: {args.file.name} [{args.algo}] — {n_modules} modules pass all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
