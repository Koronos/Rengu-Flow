"""Tiny random Krea 2 for CPU tests (contract suite, model tests). See
``rengu_flow/model/testkit_common.py`` for the kit interface. No pytest here.

Written as a diffusers-layout bundle folder (``model.checkpoint_path``): ``transformer/`` (tiny
DiT, diffusers ``save_pretrained``), ``vae/`` (tiny ``AutoencoderKLQwenImage``), ``text_encoder/``
(tiny Qwen3-VL, transformers layout) and ``model_index.json`` with the text-encoder taps the tiny
encoder has. The tokenizer is the bundled Qwen3-VL one.
"""

from __future__ import annotations

import json
from pathlib import Path

from rengu_flow.model.testkit_common import (
    ALL_CAPABILITIES,
    ModelCase,
    prepare_adapter_config,
    randomize_,
    tiny_qwen3vl,
    training_config,
    write_qwen3vl_folder,
    write_t2i_tree,
)

TEXT_DIM = 32
SELECT_LAYERS = (1, 2, 3)  # every layer of the 3-layer tiny encoder
LATENT_CH = 4  # VAE z_dim; the DiT packs 2x2 -> in_channels 16
MAX_SEQUENCE_LENGTH = 16
NOT_APPLICABLE: dict[str, str] = {}  # every contract check applies
CAPABILITIES = ALL_CAPABILITIES - NOT_APPLICABLE.keys()
VAE_INPUT_SHAPE = (1, 3, 64, 64)

DIT_CONFIG = dict(
    in_channels=LATENT_CH * 4,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=64,
    timestep_embed_dim=16,
    text_hidden_dim=TEXT_DIM,
    num_text_layers=len(SELECT_LAYERS),
    text_num_attention_heads=2,
    text_num_key_value_heads=2,
    text_intermediate_size=48,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    axes_dims_rope=(4, 2, 2),
)


def tiny_dit(seed: int = 0):
    from rengu_flow.model.krea2.dit import Krea2Transformer2DModel

    model = Krea2Transformer2DModel(**DIT_CONFIG)
    randomize_(model, 0.1, seed)
    return model.eval()


def tiny_vae(seed: int = 0):
    import torch
    from diffusers import AutoencoderKLQwenImage

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    vae = AutoencoderKLQwenImage(
        base_dim=8,
        z_dim=LATENT_CH,
        num_res_blocks=1,
        latents_mean=(torch.randn(LATENT_CH, generator=g) * 0.1).tolist(),
        latents_std=(torch.rand(LATENT_CH, generator=g) + 0.5).tolist(),
    )
    return vae.eval().requires_grad_(False)


def write_bundle(root: Path) -> Path:
    """The tiny diffusers-layout Krea 2 folder under ``root`` (reused when present)."""
    root = Path(root)
    if (root / "model_index.json").is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    tiny_dit().save_pretrained(root / "transformer")
    tiny_vae().save_pretrained(root / "vae")
    write_qwen3vl_folder(root / "text_encoder", tiny_qwen3vl(hidden=TEXT_DIM, layers=max(SELECT_LAYERS)))
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Krea2Pipeline", "text_encoder_select_layers": list(SELECT_LAYERS)}),
        encoding="utf-8",
    )
    return root


# ---- kit interface ----------------------------------------------------------------------------


def tiny_config(tmp_path, **over) -> dict:
    import torch

    model = {
        "type": "krea2",
        "dtype": torch.float32,
        "checkpoint_path": str(write_bundle(Path(tmp_path) / "krea2")),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }
    return training_config(model, over)


def tiny_pipeline(config):
    from rengu_flow.model.krea2.pipeline import Krea2Pipeline

    pipeline = Krea2Pipeline(config)
    pipeline.load_diffusion_model()
    if config.get("adapter"):
        config["adapter"] = prepare_adapter_config(config["adapter"], config["model"]["dtype"])
        pipeline.configure_adapter(config["adapter"])
    return pipeline


def monolithic_forward(pipeline, features):
    from rengu_flow.model.krea2.dit import pack_latents, prepare_position_ids, unpack_latents

    noisy, t, embeds, mask = features[:4]
    h, w = noisy.shape[-2] // 2, noisy.shape[-1] // 2
    out = pipeline.transformer(
        pack_latents(noisy),
        embeds,
        t.view(-1),
        prepare_position_ids(embeds.shape[1], h, w, noisy.device),
        encoder_attention_mask=mask,
    )
    return unpack_latents(out, h, w)


def export_transformer_path(save_dir) -> Path:
    """``save_model`` writes a diffusers transformer folder (what ``model.transformer_path`` reads)."""
    return Path(save_dir)


def preview_config(cfg: bool) -> dict:
    return {
        "width": 64,
        "height": 48,
        "num_inference_steps": 3,
        "guidance_scale": 4.0 if cfg else 0.0,
        "negative_prompt": "low quality",
    }


def _t2i_batch(pipeline, seed: int, lengths=(5, 2, 7, 3)) -> dict:
    import torch

    g = torch.Generator().manual_seed(seed)
    b = len(lengths)
    return {
        "latents": torch.randn(b, LATENT_CH, 8, 12, generator=g),
        "mask": None,
        "prompt_embeds": [torch.randn(n, len(SELECT_LAYERS), TEXT_DIM, generator=g) for n in lengths],
        "text_mask": [torch.ones(n, dtype=torch.bool) for n in lengths],
    }


def cases() -> tuple[ModelCase, ...]:
    return (
        ModelCase(
            "t2i",
            write_dataset=lambda root: write_t2i_tree(root, 64),
            sample_batch=_t2i_batch,
            expected_grids=frozenset({(8, 8), (6, 12), (12, 6)}),
        ),
    )
