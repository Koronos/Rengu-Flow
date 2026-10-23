"""Numerical parity of rengu's MOSS-Image port against the official ``pipeline_moss_image.py``.

Not part of pytest: needs a Hugging Face snapshot of ``OpenMOSS-Team/MOSS-Image-Base`` (or -SFT; the
folder holding ``transformer/``, ``text_encoder/``, ``vae/``, ``tokenizer/`` and the reference
``pipeline_moss_image.py``). Run it in an ephemeral overlay so rengu's environment is untouched. The
reference VAE class needs ``diffusers >= 0.38`` (the release's small decoder, ``decoder_block_out_channels``),
rengu's loader does not::

    uv run --no-sync --with diffusers==0.38.0 python scripts/parity_moss_image.py \\
        --snapshot ~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/<sha>

Everything runs on the CPU. Memory: the DiT check holds two fp32 copies (~41 GB RAM); ``--checks`` runs a
subset. ``--checks sigmas,keys`` needs only the small files of the snapshot (``pipeline_moss_image.py`` and
``transformer/diffusion_pytorch_model.safetensors.index.json``), not the weights.

Checks (max abs error and relative error ``max|a-b| / max|b|``; tolerance **1e-3**, float32; the text
encoder runs in bf16 in both implementations, where the expected error is 0):

* ``te``     text encoder rows for several prompts (empty, a layout-JSON one, a long one, multi-line)
             against the reference ``encode_prompt`` over the full ``Qwen3_5Model``;
* ``vae``    encode (posterior mode -> 2x2 patchify -> bn normalization) against diffusers' Flux2
             pipeline formulas over the reference VAE, and decode against the reference decode snippet;
* ``dit``    the DiT forward in float32 against the reference ``MOSSImageTransformer`` — rengu's per-sample
             ``forward``, the batched padded training layers (captions of different lengths in one batch),
             and the negative branch with ``skip_layers = (22,)``;
* ``sigmas`` the preview sigma / timestep schedule against the official ``MOSSImageScheduler`` for several
             step counts;
* ``keys``   every module of an exported LoRA / LoKr / LyCORIS adapter (rengu's own exporters over the tiny
             kit, or ``--adapter FILE``) maps through ComfyUI's stock LoRA key map for MOSS-Image
             (``comfy/lora.py`` ``model_lora_keys_unet``: ``lora_unet_<path_with_underscores>``,
             ``diffusion_model.<path>`` and bare ``<path>``; separate ``q`` / ``k`` / ``v`` also on the fused
             ``proj.qkv`` Comfy-Org checkpoint), re-derived here from the official key list (no ComfyUI
             code is imported or copied; it is GPL-3.0).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TOLERANCE = 1e-3
PROMPTS = [
    "",
    "a red fox sitting on a mossy rock at dusk",
    '{"summary": {"caption": "a cat on a sofa"}, "items": [{"label": "cat", "caption": "grey cat", "bbox": [-200, -100, 100, 150]}]}',
    "newest, masterpiece, 1girl, solo, fennec ears, long blonde wavy hair, blue eyes, big fluffy tail, smile, forest, sunlight, " * 6,
    "first line\nsecond line\nthird line",
]


def report(rows: list[tuple[str, torch.Tensor, torch.Tensor]]) -> bool:
    ok = True
    print(f"{'quantity':64s} {'max abs err':>12s} {'rel err':>10s} {'ref max':>10s}")
    for name, ours, ref in rows:
        ours, ref = ours.detach().float().cpu(), ref.detach().float().cpu()
        if ours.shape != ref.shape:
            print(f"{name:64s} SHAPE MISMATCH {tuple(ours.shape)} vs {tuple(ref.shape)}")
            ok = False
            continue
        err = (ours - ref).abs().max().item()
        scale = ref.abs().max().item()
        rel = err / max(scale, 1e-12)
        flag = "" if rel < TOLERANCE else "  <-- OUT OF TOLERANCE"
        print(f"{name:64s} {err:12.3e} {rel:10.3e} {scale:10.3e}{flag}")
        ok &= rel < TOLERANCE
    return ok


def load_reference(snapshot: Path):
    """The official module, imported from the snapshot's own ``pipeline_moss_image.py``."""
    path = snapshot / "pipeline_moss_image.py"
    if not path.is_file():
        path = snapshot / "transformer" / "pipeline_moss_image.py"
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_moss_image"] = module
    spec.loader.exec_module(module)
    return module


# ---- text encoder ------------------------------------------------------------------------------


def check_text(args, ref, state) -> bool:
    from transformers import AutoModel, AutoTokenizer

    from rengu_flow.model.moss_image import loading
    from rengu_flow.model.moss_image.text import encode_prompts

    snap = args.snapshot
    tokenizer = AutoTokenizer.from_pretrained(snap / "tokenizer")
    ours_tokenizer = loading.load_tokenizer(None)  # the bundled copy must tokenize identically
    full = AutoModel.from_pretrained(snap / "text_encoder", dtype=torch.bfloat16).eval()
    encoder = loading.load_text_encoder(snap / "text_encoder", torch.bfloat16)
    rows = []
    state["embeds"] = []
    with torch.no_grad():
        embeds, mask = encode_prompts(encoder, ours_tokenizer, PROMPTS, "cpu")
        for i, prompt in enumerate(PROMPTS):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
            want = full(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
            ours_ids = ours_tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).input_ids
            assert torch.equal(ours_ids, inputs.input_ids), "bundled tokenizer differs from the snapshot's"
            rows.append((f"TE last_hidden_state {prompt[:26]!r} ({want.shape[0]} tokens)", embeds[i][mask[i]], want))
            state["embeds"].append(want.float())
    del full, encoder
    return report(rows)


# ---- VAE ---------------------------------------------------------------------------------------


def check_vae(args, ref, state) -> bool:
    from diffusers import AutoencoderKLFlux2, Flux2Pipeline

    from rengu_flow.model.moss_image import loading
    from rengu_flow.model.moss_image.pipeline import bn_stats, patchify, unpatchify
    from rengu_flow.model.moss_image.preview_sampling import _decode

    snap = args.snapshot
    h, w = args.size
    g = torch.Generator().manual_seed(1)
    image = torch.rand(1, 3, h, w, generator=g) * 2 - 1
    ours = loading.load_vae(snap / "vae")
    try:
        official = AutoencoderKLFlux2.from_pretrained(snap / "vae", torch_dtype=torch.float32).eval()
    except Exception as exc:  # diffusers < 0.38 cannot build the small decoder
        print(f"reference VAE class could not load the release vae/ ({type(exc).__name__}); rerun with diffusers>=0.38")
        return False
    rows = []
    with torch.no_grad():
        mode = official.encode(image).latent_dist.mode()
        eps = official.config.batch_norm_eps
        want = (Flux2Pipeline._patchify_latents(mode) - official.bn.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
            official.bn.running_var.view(1, -1, 1, 1) + eps
        )
        mean, std = bn_stats(ours, "cpu", torch.float32)
        got = (patchify(ours.encode(image).latent_dist.mode()) - mean) / std
        rows.append(("VAE encode (mode, patchify, bn-normalized latents)", got, want))
        # The reference decode snippet of MOSSImagePipeline.__call__.
        z = want
        zz = z * (official.bn.running_var + eps).sqrt().view(1, -1, 1, 1) + official.bn.running_mean.view(1, -1, 1, 1)
        b, c, hh, ww = zz.shape
        zz = zz.reshape(b, c // 4, 2, 2, hh, ww).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * hh, 2 * ww)
        reference = official.decode(zz, return_dict=False)[0].clamp(-1, 1)
        rows.append(("VAE decode (same latents, untiled)", _decode_untiled(ours, z, mean, std, _decode), reference))
    return report(rows)


def _decode_untiled(vae, z, mean, std, decode):
    from rengu_flow.model.moss_image import preview_sampling

    old = preview_sampling.PREVIEW_TILE
    preview_sampling.PREVIEW_TILE = 1 << 20  # the preview tiles above 512 px; parity compares the plain decode
    try:
        return decode(vae, z * std + mean)
    finally:
        preview_sampling.PREVIEW_TILE = old


# ---- DiT ---------------------------------------------------------------------------------------


def check_dit(args, ref, state) -> bool:
    from rengu_flow.model.moss_image import loading
    from rengu_flow.model.moss_image.layers import FinalLayer, InitialLayer, TransformerLayer

    snap = args.snapshot
    h, w = (args.size[0] // 16, args.size[1] // 16)
    official = ref.MOSSImageTransformer.from_pretrained(snap / "transformer", torch_dtype=torch.float32).eval()
    ours = loading.load_transformer(snap / "transformer", torch.float32).eval()
    g = torch.Generator().manual_seed(0)
    lengths = [int(e.shape[0]) for e in state.get("embeds", [])][:3] or [20, 7, 33]
    embeds = state.get("embeds") or [torch.randn(n, 4096, generator=g) for n in lengths]
    embeds = embeds[:3] if len(embeds) >= 3 else embeds
    lengths = [int(e.shape[0]) for e in embeds]
    batch = len(embeds)
    longest = max(lengths)
    context = torch.zeros(batch, longest, 4096)
    mask = torch.zeros(batch, longest, dtype=torch.bool)
    for i, e in enumerate(embeds):
        context[i, : e.shape[0]] = e.float()
        mask[i, : e.shape[0]] = True
    latents = torch.randn(batch, 128, h, w, generator=g)
    sigma = torch.tensor([0.35, 0.78, 0.99][:batch])
    t1000 = sigma * 1000
    rows = []
    with torch.no_grad():
        want = official(latents, t1000, context, mask, return_dict=False)[0]
        rows.append((f"DiT fp32 forward, per-sample (lengths {lengths})", ours(latents, t1000, context, mask, return_dict=False)[0], want))
        x = (latents, sigma.view(-1, 1), context, mask)
        for layer in [InitialLayer(ours), *[TransformerLayer(b, i) for i, b in enumerate(ours.layers)], FinalLayer(ours)]:
            x = layer(x)
        rows.append(("DiT fp32 training layers, padded batch", x, want))
        want_skip = official(latents, t1000, context, mask, return_dict=False, skip_layers=(22,))[0]
        got_skip = ours(latents, t1000, context, mask, return_dict=False, skip_layers=(22,))[0]
        rows.append(("DiT fp32 forward, skip_layers=(22,) (negative branch)", got_skip, want_skip))
    return report(rows)


# ---- sigmas ------------------------------------------------------------------------------------


def check_sigmas(args, ref, state) -> bool:
    from rengu_flow.model.moss_image.preview_sampling import reference_sigmas

    rows = []
    for steps in (1, 2, 10, 30, 50, 100, 999):
        scheduler = ref.MOSSImageScheduler()
        scheduler.set_timesteps(steps)
        ours = reference_sigmas(steps)
        rows.append((f"sigmas, {steps} steps ({len(scheduler.sigmas)} values)", ours, scheduler.sigmas.float()))
        rows.append((f"timesteps (sigma*1000), {steps} steps", ours[:-1] * 1000, scheduler.timesteps.float()))
    return report(rows)


# ---- adapter keys through ComfyUI's key map -----------------------------------------------------

ADAPTER_SUFFIX = re.compile(
    r"\.(lora_A\.weight|lora_B\.weight|lora_down\.weight|lora_up\.weight|alpha|dora_scale|"
    r"lokr_w1|lokr_w2|lokr_w1_a|lokr_w1_b|lokr_w2_a|lokr_w2_b|lokr_t2|hada_w1_a|hada_w1_b|hada_w2_a|hada_w2_b|"
    r"hada_t1|hada_t2|diff|diff_b)$"
)


def comfy_key_map(official_keys: list[str], fused: bool) -> set[str]:
    """The names ComfyUI's ``model_lora_keys_unet`` accepts for MOSS-Image (re-derived from its mapping
    rules): per ``diffusion_model.<path>.weight`` of the loaded checkpoint, ``lora_unet_<path with _>``,
    ``diffusion_model.<path>`` and the bare ``<path>``; a fused ``<block>.proj.qkv.weight`` additionally
    addresses ``<block>.proj.{q,k,v}`` under the same three spellings. ``fused`` = Comfy-Org's layout."""
    model_keys = []
    for key in official_keys:
        m = re.fullmatch(r"(layers\.\d+\.proj)\.[qkv]\.weight", key)
        if fused and m:
            model_keys.append(f"diffusion_model.{m[1]}.qkv.weight")
        else:
            model_keys.append(f"diffusion_model.{key}")
    names = set()
    for key in set(model_keys):
        if not key.endswith(".weight"):
            continue
        path = key[len("diffusion_model.") : -len(".weight")]
        names |= {f"lora_unet_{path.replace('.', '_')}", f"diffusion_model.{path}", path}
        if path.endswith(".proj.qkv"):
            base = path[: -len(".qkv")]
            for proj in "qkv":
                sub = f"{base}.{proj}"
                names |= {f"lora_unet_{sub.replace('.', '_')}", f"diffusion_model.{sub}", sub}
    return names


def adapter_modules(state: dict) -> tuple[set[str], list[str]]:
    """``(distinct module names, keys with an unknown suffix)`` of an adapter state dict."""
    modules, unknown = set(), []
    for key in state:
        if not ADAPTER_SUFFIX.search(key):
            unknown.append(key)
        modules.add(ADAPTER_SUFFIX.sub("", key))
    return modules, unknown


def tiny_exports() -> dict[str, dict]:
    """rengu's own exports (LoRA / LoKr / LyCORIS over the tiny kit) as ``{adapter: state_dict}``."""
    import tempfile

    from safetensors.torch import load_file

    from rengu_flow.model.moss_image import testkit
    from rengu_flow.model.testkit_common import ADAPTER_CONFIGS

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, config in ADAPTER_CONFIGS.items():
            pipeline = testkit.tiny_pipeline(testkit.tiny_config(Path(tmp) / name, adapter=dict(config)))
            state = {
                prm.original_name.replace(".default", ""): prm.detach().clone()
                for prm in pipeline.transformer.parameters()
                if prm.requires_grad
            }
            save_dir = Path(tmp) / f"out_{name}"
            pipeline.save_adapter(save_dir, state)
            out[name] = load_file(str(next(save_dir.glob("*.safetensors"))))
    return out


def check_keys(args, ref, state) -> bool:
    index = json.loads((args.snapshot / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text())
    official = [k for k in index["weight_map"] if k != "rope.bands"]
    maps = {"diffusers layout": comfy_key_map(official, fused=False), "Comfy-Org fused layout": comfy_key_map(official, fused=True)}
    if args.adapter:
        from safetensors.torch import load_file

        exports = {args.adapter.name: load_file(str(args.adapter))}
    else:
        exports = tiny_exports()
    ok = True
    print(f"{'adapter file':32s} {'layout':24s} {'keys':>6s} {'modules':>8s} {'not mapped':>11s} {'unknown suffix':>15s}")
    for name, sd in exports.items():
        modules, unknown = adapter_modules(sd)
        for layout, mapping in maps.items():
            missing = sorted(m for m in modules if m not in mapping)
            print(f"{name:32s} {layout:24s} {len(sd):6d} {len(modules):8d} {len(missing):11d} {len(unknown):15d}")
            for item in (missing + unknown)[:5]:
                print("    ", item)
            ok &= not missing and not unknown and bool(modules)
    return ok


CHECKS = {"te": check_text, "vae": check_vae, "dit": check_dit, "sigmas": check_sigmas, "keys": check_keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, type=Path, help="HF snapshot folder (transformer/, text_encoder/, vae/, tokenizer/)")
    ap.add_argument("--checks", default="te,vae,dit,sigmas,keys", help=f"comma list of {sorted(CHECKS)} (te runs first: dit reuses its embeddings)")
    ap.add_argument("--size", type=int, nargs=2, default=(256, 384), help="image H W for the VAE / DiT checks (multiples of 16)")
    ap.add_argument("--adapter", type=Path, help="a rengu adapter_model.safetensors for the keys check (default: tiny-kit exports)")
    args = ap.parse_args()
    args.snapshot = args.snapshot.expanduser()

    ref = load_reference(args.snapshot)
    state: dict = {}
    results = {}
    for name in args.checks.split(","):
        print(f"\n== {name} ==")
        results[name] = CHECKS[name.strip()](args, ref, state)
    print("\n" + ", ".join(f"{name}: {'OK' if ok else 'FAIL'}" for name, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
