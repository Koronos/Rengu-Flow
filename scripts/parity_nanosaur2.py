"""Numerical parity of rengu's Nanosaur2 port against the official ComfyUI custom node.

Not part of pytest: needs a ComfyUI checkout with ``custom_nodes/nanosaur2_support`` (the node
from ``well9472/Nanosaur2-670M``) plus ComfyUI's runtime deps, and the three release files.
Run it in an ephemeral overlay so rengu's environment is untouched, e.g.::

    uv run --no-sync --with comfy-kitchen==0.2.37 --with torchsde --with av --with psutil \\
        python scripts/parity_nanosaur2.py --comfy-root /path/to/ComfyUI \\
        --dit .../nanosaur2_diffusion_model.safetensors \\
        --text-encoder .../nanosaur2_text_encoder.safetensors --vae .../nanosaur2_vae.safetensors

Compares (max abs error and relative error ``max|a-b| / max|b|``):

* text encoder: conditioning and ``token_weights`` for several prompts (empty, emphasis);
* VAE encode and decode;
* DiT forward (float32) with and without ``sparse_skip`` (mixed rows), against both the
  reference training path (``in_training``) and its inference path (fused kernels);
* the preview sigma schedule against ComfyUI's ``simple`` scheduler over the model's sampling;
* with ``--lora``: a rengu-exported adapter goes through ComfyUI's LoRA loader
  (``LoraLoaderModelOnly`` path: ``model_lora_keys_unet`` + ``load_lora``) with every key consumed
  and one patch per adapted Linear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PROMPTS = [
    "",
    "newest, masterpiece, 1girl, solo, (fennec ears:1.3), long blonde wavy hair, blue eyes, big fluffy tail, smile, forest, sunlight",
    "a (red:0.5) fox sitting on ((a mossy rock:1.2):1.5) at dusk",
    "oldest, low quality",
]


def report(rows: list[tuple[str, torch.Tensor, torch.Tensor]]) -> bool:
    ok = True
    print(f"{'quantity':58s} {'max abs err':>12s} {'rel err':>10s} {'ref max':>10s}")
    for name, ours, ref in rows:
        ours, ref = ours.detach().float().cpu(), ref.detach().float().cpu()
        if ours.shape != ref.shape:
            print(f"{name:58s} SHAPE MISMATCH {tuple(ours.shape)} vs {tuple(ref.shape)}")
            ok = False
            continue
        err = (ours - ref).abs().max().item()
        scale = ref.abs().max().item()
        rel = err / max(scale, 1e-12)
        print(f"{name:58s} {err:12.3e} {rel:10.3e} {scale:10.3e}")
        ok &= rel < 1e-3
    return ok


def check_comfy_lora(args, ref_nodes) -> bool:
    """Load the adapter the way ComfyUI's LoraLoaderModelOnly does; every key must map."""
    import logging

    import comfy.lora
    import comfy.utils

    sd = comfy.utils.load_torch_file(args.dit)
    config = ref_nodes.Nanosaur2ModelConfig(ref_nodes.detect_unet_config(sd))
    config.set_inference_dtype(torch.float32, None)
    model = config.get_model(sd)
    key_map = comfy.lora.model_lora_keys_unet(model, {})
    lora = comfy.utils.load_torch_file(args.lora)
    unused = []
    handler = logging.Handler()
    handler.emit = lambda record: unused.append(record.getMessage())
    logging.getLogger().addHandler(handler)
    try:
        patches = comfy.lora.load_lora(lora, key_map)
    finally:
        logging.getLogger().removeHandler(handler)
    not_loaded = [m for m in unused if "not loaded" in m]
    adapted = {k.rsplit(".lora_A.", 1)[0] for k in lora if ".lora_A." in k}
    print(f"ComfyUI LoRA load: {len(patches)} patched weights for {len(adapted)} adapted modules, "
          f"{len(not_loaded)} keys not loaded")
    for m in not_loaded[:5]:
        print("   ", m)
    return not not_loaded and len(patches) == len(adapted) > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-root", required=True, type=Path)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--size", type=int, nargs=2, default=(512, 384), help="image H W for VAE/DiT checks")
    ap.add_argument("--lora", help="rengu adapter_model.safetensors to load through ComfyUI's LoRA loader")
    args = ap.parse_args()

    sys.path[:0] = [str(args.comfy_root), str(args.comfy_root / "custom_nodes"), str(REPO)]
    import comfy.model_management as mm
    import comfy.ops
    import comfy.utils
    from nanosaur2_support import model as ref_model
    from nanosaur2_support import nodes as ref_nodes

    from rengu_flow.model.nanosaur2 import loading, text
    from rengu_flow.model.nanosaur2.dit import Nanosaur2Transformer2DModel, detect_config
    from rengu_flow.model.nanosaur2.preview_sampling import flow_sigmas

    device = torch.device(args.device)
    rows: list = []
    torch.manual_seed(0)

    # ---- text encoder --------------------------------------------------------------------
    clip = ref_nodes.load_clip(args.text_encoder)
    te = loading.load_text_encoder(args.text_encoder, torch.float32).to(device)
    spm = loading.load_tokenizer(args.text_encoder)
    ours_embeds, ours_weights = text.encode_prompts(te, spm, PROMPTS, device)
    for i, prompt in enumerate(PROMPTS):
        out = clip.encode_from_tokens(clip.tokenize(prompt), return_dict=True)
        label = repr(prompt[:28])
        rows.append((f"TE cond      {label}", ours_embeds[i], out["cond"][0]))
        rows.append((f"TE weights   {label}", ours_weights[i], out["token_weights"][0]))
    del te, clip

    # ---- VAE -------------------------------------------------------------------------------
    h, w = args.size
    image = torch.rand(1, 3, h, w, generator=torch.Generator().manual_seed(1)) * 2 - 1
    sd = comfy.utils.load_torch_file(args.vae)
    ref_vae = ref_nodes.Nanosaur2VAE(latent_channels=sd["latent_mean"].shape[1]).eval()
    ref_vae.load_state_dict(sd)
    ref_vae = ref_vae.to(device, torch.float32)
    ours_vae = loading.load_vae(args.vae, torch.float32).to(device)
    with torch.no_grad():
        z_ref = ref_vae.encode(image.to(device))
        z_ours = ours_vae.encode(image.to(device))
        rows.append(("VAE encode (latents)", z_ours, z_ref))
        rows.append(("VAE decode (same latents)", ours_vae.decode(z_ref), ref_vae.decode(z_ref)))
    latents = z_ref
    del ref_vae, ours_vae

    # ---- DiT -------------------------------------------------------------------------------
    sd = comfy.utils.load_torch_file(args.dit)
    ref = ref_model.Nanosaur2Transformer2DModel(
        **ref_nodes.detect_unet_config(sd), dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init
    )
    ref.load_state_dict(sd)
    ref = ref.to(device).eval()
    ours = Nanosaur2Transformer2DModel(**detect_config(sd))
    ours.load_state_dict(sd)
    ours = ours.to(device, torch.float32).eval()
    del sd

    b = 2
    x = torch.randn(b, *latents.shape[1:], generator=torch.Generator().manual_seed(2)).to(device)
    t = torch.tensor([0.83, 0.21], device=device)
    texts = ours_embeds[1], ours_embeds[2]
    weights = ours_weights[1], ours_weights[2]
    length = max(e.shape[0] for e in texts)
    context = torch.zeros(b, length, texts[0].shape[1])
    token_weights = torch.zeros(b, length)
    for i in range(b):
        context[i, : texts[i].shape[0]] = texts[i]
        token_weights[i, : weights[i].shape[0]] = weights[i]
    context, token_weights = context.to(device), token_weights.to(device)
    with torch.no_grad():
        for skip in (None, [False, True], [True, True]):
            v_ours = ours(x, t, context, token_weights, sparse_skip=skip)
            for in_training in (True, False):
                mm.in_training = in_training
                path = "train path" if in_training else "infer path"
                v_ref = ref(x, t, context, token_weights, sparse_skip=skip)
                rows.append((f"DiT v  sparse_skip={skip} ({path})", v_ours, v_ref))

    # ---- sampling schedule ---------------------------------------------------------------
    import comfy.model_sampling
    import comfy.samplers

    class _Cfg:
        sampling_settings = ref_nodes.Nanosaur2ModelConfig.sampling_settings

    class _Sampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    sampling = _Sampling(_Cfg())
    for steps in (30, 50):
        ref_sigmas = comfy.samplers.calculate_sigmas(sampling, "simple", steps)
        rows.append((f"preview sigmas ({steps} steps, simple, shift 3)", flow_sigmas(steps), ref_sigmas))

    ok = report(rows)
    if args.lora:
        ok &= check_comfy_lora(args, ref_nodes)
    print("PARITY OK" if ok else "PARITY FAILED (rel err >= 1e-3 somewhere)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
