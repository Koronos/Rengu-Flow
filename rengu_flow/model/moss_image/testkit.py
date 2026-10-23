"""Tiny random MOSS-Image for CPU tests (contract suite, model tests). See
``rengu_flow/model/testkit_common.py`` for the kit interface. No pytest here.

Written as a diffusers-layout folder (``model.diffusers_path``): ``transformer/`` (tiny DiT,
official key names), ``vae/`` (tiny FLUX.2 VAE with a *different* decoder width, like the release's
small decoder) and ``text_encoder/`` (tiny Qwen3.5 with the real vocabulary, transformers layout).
No ``tokenizer/``: the bundled Qwen3.5 tokenizer is used, as for a release download without it.
The text lengths of the cases mix within one batch (the batched padded path must equal the
reference's per-sample forward).
"""

from __future__ import annotations

import re
from pathlib import Path

from rengu_flow.model.testkit_common import (
    ALL_CAPABILITIES,
    ModelCase,
    prepare_adapter_config,
    randomize_,
    training_config,
    write_t2i_tree,
)

TEXT_DIM = 32
VAE_LATENT = 2  # 32 in the release; x4 after the 2x2 patchify = the DiT's input channels
LATENT_CH = 4 * VAE_LATENT
NOT_APPLICABLE: dict[str, str] = {}  # every contract check applies
CAPABILITIES = ALL_CAPABILITIES - NOT_APPLICABLE.keys()
VAE_INPUT_SHAPE = (1, 3, 64, 64)

DIT_CONFIG = dict(
    dim=32,
    ffn_dim=48,
    num_heads=2,
    num_layers=3,
    text_dim=TEXT_DIM,
    freq_dim=16,
    latent_channels=LATENT_CH,
    rope_base=10000.0,
)

VAE_CONFIG = dict(
    block_out_channels=[8, 8, 16, 16],
    decoder_block_out_channels=[4, 8, 8, 8],
    layers_per_block=1,
    latent_channels=VAE_LATENT,
    norm_num_groups=4,
)

_LINEAR_NAME = re.compile(r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)")


def tiny_dit(seed: int = 0):
    from rengu_flow.model.moss_image.dit import MOSSImageTransformer

    import torch

    model = MOSSImageTransformer(**DIT_CONFIG)
    randomize_(model, 0.1, seed)
    # Sharp attention: with the QK norms at ~0.1 the logits are ~0 and neither the RoPE positions nor the
    # text key mask would show in the output (a wrong position must change the prediction).
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                p.copy_(2.0 + 0.3 * torch.randn(p.shape, generator=g))
    return model.eval()


def tiny_vae(seed: int = 0):
    """A tiny ``AutoencoderKLFlux2`` with a narrower decoder (built the way ``loading`` builds it)."""
    import torch
    from diffusers import AutoencoderKLFlux2
    from diffusers.models.autoencoders.vae import Decoder

    from rengu_flow.model.moss_image.loading import _config_kwargs, VAE_CONFIG_PATH

    vae_kwargs = _config_kwargs(VAE_CONFIG_PATH)
    vae_kwargs.update(VAE_CONFIG)
    decoder_channels = vae_kwargs.pop("decoder_block_out_channels")
    torch.manual_seed(seed)
    vae = AutoencoderKLFlux2.from_config(vae_kwargs)
    vae.decoder = Decoder(
        in_channels=vae_kwargs["latent_channels"],
        out_channels=3,
        up_block_types=vae_kwargs["up_block_types"],
        block_out_channels=tuple(decoder_channels),
        layers_per_block=vae_kwargs["layers_per_block"],
        norm_num_groups=vae_kwargs["norm_num_groups"],
        act_fn=vae_kwargs["act_fn"],
        mid_block_add_attention=True,
    )
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        vae.bn.running_mean.copy_(torch.randn(vae.bn.running_mean.shape, generator=g) * 0.2)
        vae.bn.running_var.copy_(torch.rand(vae.bn.running_var.shape, generator=g) + 0.5)
    return vae.eval().requires_grad_(False)


def vae_config_json() -> dict:
    """The tiny VAE's ``config.json`` (release layout, ``decoder_block_out_channels`` included)."""
    from rengu_flow.model.moss_image.loading import VAE_CONFIG_PATH, _config_kwargs

    return {**_config_kwargs(VAE_CONFIG_PATH), **VAE_CONFIG}


def tiny_text_config():
    import json

    from transformers import Qwen3_5Config

    from rengu_flow.model.moss_image.loading import TEXT_ENCODER_ASSETS

    raw = json.loads((TEXT_ENCODER_ASSETS / "config.json").read_text())
    raw["dtype"] = "float32"
    text = raw["text_config"]
    text.update(
        hidden_size=TEXT_DIM,
        intermediate_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5,
        dtype="float32",
    )
    text["rope_parameters"].update(partial_rotary_factor=0.5, mrope_section=[2, 1, 1])
    raw["vision_config"].update(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=TEXT_DIM, num_position_embeddings=16, dtype="float32"
    )
    return Qwen3_5Config(**{k: v for k, v in raw.items() if k != "architectures"})


def tiny_text_encoder(seed: int = 0):
    """A tiny ``Qwen3_5Model`` (vision tower included, as the release's ``text_encoder/``)."""
    import torch
    from transformers import Qwen3_5Model

    torch.manual_seed(seed)
    model = Qwen3_5Model(tiny_text_config()).eval().float()
    with torch.no_grad():  # exercise the (default zero/one-initialized) norms too
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(0.05 * torch.randn_like(p))
    return model


def write_text_encoder_folder(folder: Path, model) -> None:
    """A transformers-layout ``text_encoder/`` (config.json + weights keyed ``language_model.*`` /
    ``visual.*``, as the release's)."""
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / "model.safetensors"))
    model.config.save_pretrained(folder)


def write_bundle(root: Path) -> Path:
    """The tiny diffusers-layout MOSS-Image folder under ``root`` (reused when present)."""
    import json

    from safetensors.torch import save_file

    root = Path(root)
    if (root / "text_encoder" / "config.json").is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    tiny_dit().save_pretrained(root / "transformer")
    (root / "vae").mkdir(exist_ok=True)
    (root / "vae" / "config.json").write_text(json.dumps(vae_config_json()), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in tiny_vae().state_dict().items()}, str(root / "vae" / "diffusion_pytorch_model.safetensors"))
    write_text_encoder_folder(root / "text_encoder", tiny_text_encoder())
    return root


# ---- kit interface ----------------------------------------------------------------------------


def tiny_config(tmp_path, **over) -> dict:
    import torch

    model = {
        "type": "moss_image",
        "dtype": torch.float32,
        "diffusers_path": str(write_bundle(Path(tmp_path) / "moss_image")),
        "text_encoder_offload": "none",
    }
    return training_config(model, over)


def tiny_pipeline(config):
    from rengu_flow.model.moss_image.pipeline import MOSSImagePipeline

    pipeline = MOSSImagePipeline(config)
    pipeline.load_diffusion_model()
    if config.get("adapter"):
        config["adapter"] = prepare_adapter_config(config["adapter"], config["model"]["dtype"])
        pipeline.configure_adapter(config["adapter"])
    return pipeline


def monolithic_forward(pipeline, features):
    """The reference forward over ``prepare_inputs`` features: each sample at its true text length."""
    noisy, sigma, text, text_mask = features[:4]
    return pipeline.transformer(noisy, sigma.view(-1) * 1000.0, text, text_mask, return_dict=False)[0]


def export_transformer_path(save_dir) -> Path:
    """``save_model`` writes a diffusers transformer folder (what ``model.transformer_path`` reads)."""
    return Path(save_dir)


def adapter_scope(pipeline, adapter: str) -> list[str]:
    """The default adapter scope: the blocks' attention + feed-forward linears (``layers.N.proj.*``)."""
    from rengu_flow.networks import adapter_dit

    names = adapter_dit._collect_target_linears(pipeline.transformer, tuple(pipeline.adapter_target_modules))
    return [n for n in names if _LINEAR_NAME.fullmatch(n)]


def preview_config(cfg: bool) -> dict:
    return {
        "width": 64,
        "height": 64,
        "num_inference_steps": 3,
        "guidance_scale": 3.0 if cfg else 1.0,
        "negative_prompt": "low quality",
    }


# ---- cases ------------------------------------------------------------------------------------


def _t2i_batch(pipeline, seed: int, lengths=(5, 2, 7, 3)) -> dict:
    import torch

    g = torch.Generator().manual_seed(seed)
    b = len(lengths)
    return {
        "latents": torch.randn(b, LATENT_CH, 4, 6, generator=g),
        "mask": None,
        "prompt_embeds": [torch.randn(n, TEXT_DIM, generator=g) for n in lengths],
        "text_mask": [torch.ones(n, dtype=torch.bool) for n in lengths],
    }


def cases() -> tuple[ModelCase, ...]:
    return (
        ModelCase(
            "t2i",
            write_dataset=lambda root: write_t2i_tree(root, 64),
            sample_batch=_t2i_batch,
            expected_grids=frozenset({(4, 4), (3, 6), (6, 3)}),
        ),
    )
