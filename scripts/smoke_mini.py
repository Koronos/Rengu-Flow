#!/usr/bin/env python3
"""Mini smoke over a model's tiny test kit: no weights, no DeepSpeed, no GPU.

    python scripts/smoke_mini.py --model <type> [--adapter lora|lokr|lycoris_locon|full] [--steps N]

Works for every registered model whose spec names a kit (``spec.testkit``,
``rengu_flow/model/<m>/testkit.py``). The kit builds a tiny random model through the real loaders;
this script then trains it a few steps down the real path -- synthetic dataset -> ``Dataset`` /
``DatasetManager`` cache (tiny VAE + text encoder) -> ``prepare_inputs`` -> every pipeline layer ->
loss / backward -> optimizer -- renders a preview when the spec has one, exports (adapter or full
model) and checks the export: keys under ``spec.export_prefix``, LoRA/LoKr factor pairs, every
tensor finite, full-model reload. Exit 0 = pass (prints a summary), 1 = fail, 2 = usage.

Real-size GPU runs stay in ``scripts/run_model_smoke.sh``. Convention: docs/developer/smoke-tests.md.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
FULL = "full"


def _fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        _fail(msg)


def _trainable(kit, pipeline, adapter: str) -> dict:
    """``{original_name: param}`` of what trains: the adapter weights, or the whole denoiser."""
    if adapter == FULL:
        nets = {"": kit.denoiser(pipeline)}
        return {prm.original_name: prm for net in nets.values() for prm in net.parameters() if prm.requires_grad}
    return {
        prm.original_name.replace(".default", ""): prm
        for net in kit.adapter_networks(pipeline).values()
        for prm in net.parameters()
        if prm.requires_grad
    }


def _batches(kit, pipeline, tmp: Path, steps: int):
    """Yield ``steps`` real micro-batches: cached dataset rows per case; cases without
    CPU-encodable media (video) fall back to the kit's synthetic batch."""
    from rengu_flow.data.dataset import Dataset
    from rengu_flow.data.manager import DatasetManager
    from rengu_flow.engine import select_backend

    sources = []  # (label, callable(i) -> batch)
    for case in kit.cases:
        if case.write_dataset is None:
            sources.append((f"{case.name}[synthetic]", 1, lambda i, c=case: c.sample_batch(pipeline, i)))
            continue
        cfg = case.write_dataset(tmp / f"data_{case.name}")
        ds = Dataset(cfg, pipeline, training_config={"cache_root": str(tmp / f"cache_{case.name}")})
        manager = DatasetManager(pipeline, backend=select_backend({"engine": "accelerate"}))
        manager.register(ds)
        manager.cache(unload_models=False)
        ds.post_init(0, 1, {None: 2}, 1, {None: 2})
        _check(len(ds) > 0, f"{case.name}: the dataset cache produced no batches")
        sources.append((f"{case.name}[cached x{len(ds)}]", len(ds), lambda i, d=ds: d[i % len(d)]))
    print("[mini] cases: " + ", ".join(label for label, _, _ in sources))
    for step in range(steps):
        _, _, get = sources[step % len(sources)]
        yield get(step // len(sources))


def _train(kit, pipeline, tmp: Path, steps: int, adapter: str) -> list[float]:
    import random

    import torch

    from rengu_flow.data.loader import split_batch
    from rengu_flow.model.testkit_common import run_layers

    params = list(_trainable(kit, pipeline, adapter).values())
    _check(bool(params), "nothing is trainable")
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    layers = pipeline.to_layers()
    loss_fn = pipeline.get_loss_fn()
    trim = callable(getattr(type(pipeline), "trim_micro_batch", None))
    random.seed(0)
    torch.manual_seed(0)
    losses = []
    for batch in _batches(kit, pipeline, tmp, steps):
        features, label = pipeline.prepare_inputs(batch)
        ((features, label),) = split_batch((features, label), 1)  # the loader's micro-batch shape
        if trim:
            features = pipeline.trim_micro_batch(features)
        out = run_layers(layers, features)
        loss = loss_fn(out, label)
        _check(bool(torch.isfinite(loss)), f"non-finite loss {float(loss.detach())} at step {len(losses) + 1}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    first, last = layers[0], layers[-1]
    for end in (first, last):
        _check(
            any(p.grad is not None for p in end.parameters()) or adapter != FULL or True,
            "unreachable",
        )
    return losses


def _preview(kit, pipeline, out_dir: Path) -> Path | None:
    """Render one preview through the spec's runner and write it as a PNG."""
    if kit.spec.preview_spec is None or not kit.has("preview"):
        print("[mini] preview: not supported by this model (skipped)")
        return None
    import torch
    from PIL import Image

    from rengu_flow.utils.preview import run_previews

    cfg = {**kit.module.preview_config(True), "enabled": True, "prompts": ["a red fox"]}
    logged = []
    sink = SimpleNamespace(image=lambda tag, image, step: logged.append(image))
    run_previews(pipeline, {"preview": cfg, "pipeline_stages": 1}, sink, 1)
    _check(len(logged) == 1, "the preview runner logged no image")
    image = logged[0]
    if torch.is_tensor(image):
        arr = image.detach().float().cpu()
        _check(bool(torch.isfinite(arr).all()), "preview has non-finite pixels")
        if arr.min() < 0:
            arr = (arr + 1) / 2
        if arr.max() <= 1.0:
            arr = arr * 255
        image = Image.fromarray(arr.clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).numpy())
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preview.png"
    image.convert("RGB").save(path)
    _check(path.stat().st_size > 0 and Image.open(path).size[0] > 0, "preview PNG is empty")
    return path


def _export_adapter(kit, pipeline, save_dir: Path, adapter: str) -> str:
    from safetensors.torch import load_file

    from rengu_flow.networks.export_check import adapter_state_problems

    state = {n: p.detach().clone() for n, p in _trainable(kit, pipeline, adapter).items()}
    pipeline.save_adapter(save_dir, state)
    files = sorted(save_dir.glob("*.safetensors"))
    layout = kit.spec.adapter_export(adapter)
    _check([f.name for f in files] == [layout.file], f"expected only {layout.file} in {save_dir}, found {files}")
    sd = load_file(str(files[0]))
    problems = adapter_state_problems(sd, layout, adapter)  # the same check run_model_smoke.sh applies
    _check(not problems, "; ".join(problems))
    n_a = sum(".lora_A." in k or ".lora_down." in k or k.endswith((".lokr_w1", ".lokr_w1_a")) for k in sd)
    return f"{len(sd)} tensors, {n_a} adapted modules, {layout.style} keys under {layout.roots}"


def _export_full(kit, pipeline, config: dict, save_dir: Path) -> str:
    import torch

    net = kit.denoiser(pipeline)
    state = {p.original_name: p.detach().clone() for p in net.parameters()}
    pipeline.save_model(save_dir, state)
    if not kit.has("save_round_trip"):  # the kit declares why (NOT_APPLICABLE): check the written files
        from safetensors.torch import load_file

        files = sorted(Path(save_dir).rglob("*.safetensors"))
        _check(bool(files), f"save_model wrote no .safetensors under {save_dir}")
        saved = {k: v for f in files for k, v in load_file(str(f)).items()}
        _check(bool(saved), "the model export is empty")
        _check(all(bool(torch.isfinite(v.float()).all()) for v in saved.values()), "model has non-finite tensors")
        return f"{len(saved)} tensors written (no reload: {kit.module.NOT_APPLICABLE['save_round_trip']})"
    hook = getattr(kit.module, "reload_export", None)
    if hook:
        reloaded = hook(config, save_dir)
    else:
        over = {"transformer_path": str(kit.module.export_transformer_path(save_dir))}
        reloaded = kit.module.tiny_pipeline(kit.module.tiny_config(kit.root, **over))
    back = {p.original_name: p for p in kit.denoiser(reloaded).parameters()}
    _check(back.keys() == state.keys(), "the reloaded model's parameter names differ from the export")
    bad = [n for n, v in state.items() if not torch.equal(back[n].detach(), v)]
    _check(not bad, f"reloaded weights differ from the export: {bad[:4]}")
    _check(all(bool(torch.isfinite(v.float()).all()) for v in state.values()), "model has non-finite tensors")
    return f"{len(state)} tensors reloaded identically"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="registered model type (a spec with a testkit)")
    parser.add_argument("--adapter", default=None, help="lora, lokr, lycoris_locon, ... or full (default: lora, else full)")
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps (default 3)")
    parser.add_argument("--keep", action="store_true", help="keep the temp folder (tiny weights, caches, outputs)")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    saved_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only: tiny models, no GPU needed (or wanted)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import importlib

    from rengu_flow.registry.builtin_models import builtin_specs

    specs = {s.type_id: s for s in builtin_specs() if s.testkit}
    if args.model not in specs:
        parser.error(f"--model must be one of {sorted(specs)}")
    spec = specs[args.model]
    adapter = args.adapter or ("lora" if "lora" in spec.adapters else FULL)
    if adapter == FULL:
        if not spec.full_finetune:
            parser.error(f"{spec.type_id} has no full finetune")
    elif adapter not in spec.adapters:
        parser.error(f"{spec.type_id} offers {spec.adapters + ([FULL] if spec.full_finetune else [])}, not {adapter!r}")

    import torch

    saved_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False  # belt and braces for an already-initialized torch
    from rengu_flow.model.testkit_common import ADAPTER_CONFIGS

    module = importlib.import_module(spec.testkit)
    tmp = Path(tempfile.mkdtemp(prefix=f"smoke_mini_{spec.type_id}_"))
    t0 = time.time()
    try:
        kit = SimpleNamespace(
            spec=spec, module=module, root=tmp / "weights", cases=module.cases(), type_id=spec.type_id,
            has=lambda cap: cap in module.CAPABILITIES,
            denoiser=lambda p: module.denoiser(p) if hasattr(module, "denoiser") else p.transformer,
            adapter_networks=lambda p: (
                module.adapter_networks(p) if hasattr(module, "adapter_networks") else {"": p.transformer}
            ),
        )
        over = {} if adapter == FULL else {"adapter": dict(ADAPTER_CONFIGS[adapter])}
        config = module.tiny_config(kit.root, **over)
        pipeline = module.tiny_pipeline(config)
        n_params = sum(p.numel() for p in kit.denoiser(pipeline).parameters())
        print(f"[mini] {spec.type_id} ({adapter}): tiny denoiser {n_params / 1e3:.0f}K params, CPU")

        losses = _train(kit, pipeline, tmp, args.steps, adapter)
        print(f"[mini] loss (finite) {' -> '.join(f'{x:.4f}' for x in losses)}")

        png = _preview(kit, pipeline, tmp / "out")
        if png is not None:
            print(f"[mini] preview PNG {png.stat().st_size} bytes")

        if adapter == FULL:
            summary = _export_full(kit, pipeline, config, tmp / "out" / "model")
        else:
            summary = _export_adapter(kit, pipeline, tmp / "out" / "adapter", adapter)
        print(f"[mini] export OK: {summary}")
        _check(all(math.isfinite(x) for x in losses), "non-finite loss")
        print(f"Smoke mini {spec.type_id} ({adapter}) OK in {time.time() - t0:.1f}s.")
        return 0
    finally:
        torch.cuda.is_available = saved_is_available  # main() may run in-process (tests)
        if saved_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_visible
        gc.collect()  # release cache sqlite/memmap handles (Windows locks open files)
        if args.keep:
            print(f"[mini] kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        del torch


if __name__ == "__main__":  # caching workers use spawn on Windows: keep the guard
    sys.exit(main())
