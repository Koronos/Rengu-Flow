"""Generic export check of a training run's output, driven by the model spec.

    python -m rengu_flow.networks.export_check --config <train.toml> --output <output_dir>

Used by ``scripts/run_model_smoke.sh`` (and ``scripts/smoke_mini.py``) instead of one shell
function per model. The model type and adapter kind come from the training TOML; the export layout
from ``spec.adapter_export(kind)`` (``registry/model_spec.AdapterExport``): the file name
``save_adapter`` writes, the key style and the key roots. Adapter export: the newest such file,
every key under one of the roots (the first one present), and per style:

- dotted (DiTs, SDXL LoKr): LoRA keys exactly ``lora_A.weight`` / ``lora_B.weight`` / ``alpha``
  (when roots are declared), LoRA/LoKr factors paired;
- kohya (SDXL LoRA / LyCORIS): flat module names, each module with exactly its family's weights
  (LoRA: ``lora_down.weight`` + ``lora_up.weight`` + ``alpha``; LyCORIS: ``lycoris_meta``'s
  ``WEIGHT_KEY_FAMILIES``).

Every tensor finite. Full finetune: the newest non-adapter ``.safetensors`` is non-empty and
finite. Exit 0 = ok, 1 = failed. LyCORIS exports also have a deeper library-level check
(``rengu_flow.networks.lycoris_export_check``).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from rengu_flow.networks.lycoris_meta import NORM_MODULE_FAMILY, WEIGHT_KEY_FAMILIES

_LORA_SUFFIX = re.compile(r"\.(lora_A\.weight|lora_B\.weight|alpha)$")
_KOHYA_LORA_FAMILY = {"required": ("lora_down.weight", "lora_up.weight", "alpha"), "optional": (), "either": ()}


def _family_problems(module: str, weights: set[str], family: dict) -> list[str]:
    allowed = {*family["required"], *family["optional"]}
    for group in family["either"]:
        for alt in group:
            allowed.update(alt)
    problems = []
    if missing := set(family["required"]) - weights:
        problems.append(f"{module}: missing {sorted(missing)}")
    if foreign := weights - allowed:
        problems.append(f"{module}: unexpected weights {sorted(foreign)}")
    for group in family["either"]:
        if sum(set(alt) <= weights for alt in group) != 1:
            problems.append(f"{module}: expected exactly one of {group}")
    return problems


def _kohya_problems(state: dict[str, Any], adapter_type: str) -> list[str]:
    """Flat ``<module>.<weight>`` keys, each module with exactly its family's weights."""
    per_module: dict[str, set[str]] = {}
    problems = []
    for key in state:
        module, _, weight = key.partition(".")
        if not weight:
            problems.append(f"key without a <module>.<weight> split: {key}")
            continue
        per_module.setdefault(module, set()).add(weight)
    family = _KOHYA_LORA_FAMILY if adapter_type == "lora" else WEIGHT_KEY_FAMILIES.get(adapter_type)
    if family is not None:
        for module, weights in sorted(per_module.items()):
            problems += _family_problems(module, weights, NORM_MODULE_FAMILY if "w_norm" in weights else family)
    return problems


def adapter_state_problems(state: dict[str, Any], layout, adapter_type: str) -> list[str]:
    """Problems of an exported adapter ``state`` against its ``AdapterExport`` layout (empty = ok)."""
    import torch

    if not state:
        return ["the adapter export is empty"]
    problems: list[str] = []
    roots = tuple(layout.roots)
    if roots:
        bad = [k for k in state if not k.startswith(roots)]
        if bad:
            problems.append(f"keys outside the {roots} roots: {bad[:4]}")
        if not any(k.startswith(roots[0]) for k in state):
            problems.append(f"no {roots[0]!r} keys")
    if layout.style == "kohya":
        found = _kohya_problems(state, adapter_type)
        problems += found[:6] + ([f"... {len(found) - 6} more"] if len(found) > 6 else [])
    else:
        if roots and adapter_type == "lora":
            bad = [k for k in state if not _LORA_SUFFIX.search(k)]
            if bad:
                problems.append(f"unexpected LoRA key names: {bad[:4]}")
        n_a = sum(k.endswith((".lora_A.weight", ".lora_down.weight", ".lokr_w1", ".lokr_w1_a")) for k in state)
        n_b = sum(k.endswith((".lora_B.weight", ".lora_up.weight", ".lokr_w2", ".lokr_w2_a")) for k in state)
        if n_a != n_b:
            problems.append(f"unpaired adapter factors ({n_a} A-side vs {n_b} B-side)")
    nonfinite = [k for k, v in state.items() if not torch.isfinite(v.float()).all()]
    if nonfinite:
        problems.append(f"non-finite tensors: {nonfinite[:4]}")
    return problems


def _newest(root: Path, pattern: str, exclude: str = "") -> Path | None:
    files = sorted(
        (p for p in root.rglob(pattern) if not exclude or p.name != exclude),
        key=lambda p: (p.stat().st_mtime, str(p)),
    )
    return files[-1] if files else None


def check_run(config_path: str | Path, output_dir: str | Path) -> str:
    """Check the newest export under ``output_dir``; returns a summary or raises ``AssertionError``."""
    import toml
    import torch
    from safetensors.torch import load_file

    from rengu_flow.registry.model_capabilities import model_capability_registry

    config = toml.load(config_path)
    model_type = str(config.get("model", {}).get("type", "")).lower()
    spec = model_capability_registry.get(model_type)
    assert spec is not None, f"unknown model type {model_type!r}"
    adapter_type = str(config.get("adapter", {}).get("type", "")).lower()
    root = Path(output_dir)
    if adapter_type:
        layout = spec.adapter_export(adapter_type)
        file = _newest(root, layout.file)
        assert file is not None, f"no {layout.file} under {root} ({spec.type_id} {adapter_type} export)"
        state = load_file(str(file))
        problems = adapter_state_problems(state, layout, adapter_type)
        assert not problems, f"{file}: " + "; ".join(problems)
        return f"{file} ({spec.type_id}, {adapter_type}, {layout.style}): {len(state)} tensors, roots {layout.roots}"
    file = _newest(root, "*.safetensors", exclude="adapter_model.safetensors")
    assert file is not None, f"no exported model .safetensors under {root}"
    state = load_file(str(file))
    assert state, f"{file} is empty"
    nonfinite = [k for k, v in state.items() if not torch.isfinite(v.float()).all()]
    assert not nonfinite, f"{file}: non-finite tensors: {nonfinite[:4]}"
    return f"{file} ({spec.type_id}, full finetune): {len(state)} tensors"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the training TOML the run used")
    parser.add_argument("--output", required=True, help="the run's output directory")
    args = parser.parse_args(argv)
    try:
        print(f"export OK: {check_run(args.config, args.output)}")
    except AssertionError as e:
        print(f"ERROR: export check failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
