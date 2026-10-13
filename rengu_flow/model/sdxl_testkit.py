"""Tiny random SDXL for CPU tests (contract suite). See ``rengu_flow/model/testkit_common.py``
for the kit interface. No pytest here.

The SDXL loader is ``StableDiffusionXLPipeline.from_single_file``, which needs the original SDXL
configs (Hub / HF cache), so a tiny checkpoint cannot go through ``load_diffusion_model`` offline.
``tiny_pipeline`` builds the real ``SDXLPipeline`` around a real diffusers
``StableDiffusionXLPipeline`` of tiny components instead (SDXL-topology UNet with text-time
conditioning, 8x VAE, both CLIP encoders with character-level tokenizers written under
``tmp_path``) and applies what ``load_diffusion_model`` does after loading. The export round trip
reads ``save_model``'s single file back through ``from_single_file`` with the tiny pipeline's
saved configs (``config=``), i.e. the same diffusers conversion a real checkpoint goes through.
"""

from __future__ import annotations

import json
from pathlib import Path

from rengu_flow.model.testkit_common import (
    ALL_CAPABILITIES,
    SEAM_CAPTIONS,
    ModelCase,
    prepare_adapter_config,
    randomize_,
    training_config,
    write_t2i_tree,
)

H1, H2, PROJ = 16, 24, 8  # CLIP-L / CLIP-G widths, pooled projection
TIME_EMBED = 8
# Contract checks that do not apply to this model, and why.
NOT_APPLICABLE = {
    "disk_loaders": "from_single_file needs the SDXL configs from the Hub; the tiny pipeline is "
    "built in memory (the export round trip still reads a real single file)",
    "lazy_text_encoder": "the text encoders live in the single checkpoint and load with it",
    "flow_target": "SDXL trains a DDPM eps / v-prediction target, not rectified flow",
    "dit_preview": "previews render through the diffusers StableDiffusionXLPipeline "
    "(utils/preview runner 'sdxl'), which has no generate_preview_image / prompt memo",
}
CAPABILITIES = ALL_CAPABILITIES - NOT_APPLICABLE.keys()
# Contract assertions this model fails today (the test xfails with the reason; it fails once the
# gap closes, so the entry must then go).
KNOWN_GAPS: dict[str, str] = {}
VAE_INPUT_SHAPE = (1, 3, 64, 64)
# Character-level tokenizers: one token per non-space character, so a caption past 75 of them
# spans two 77-token CLIP chunks (ragged rows of 77 and 154).
LONG_CAPTION = (
    "a very long caption about a red fox that sits on a mossy rock at dusk in the old forest "
    "beside a cold river"
)
RAGGED_CAPTIONS = (*SEAM_CAPTIONS[:3], LONG_CAPTION)
# Kohya-flat export names of the adapter roots (lora_sdxl / lycoris_sdxl).
KOHYA_PREFIX = {"unet.": "lora_unet_", "text_encoder.": "lora_te1_", "text_encoder_2.": "lora_te2_"}

# SDXL's topology at tiny widths: 3 levels, 2 resnets per down block (3 per up block), linear
# attention projections — the single-file export conversion (``convert_unet_state_dict``, VAE
# likewise) is written for exactly this layout, so the export round trip needs it.
UNET_CONFIG = dict(
    sample_size=8,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(16, 32, 32),
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    transformer_layers_per_block=(1, 1, 2),
    cross_attention_dim=H1 + H2,
    attention_head_dim=4,
    use_linear_projection=True,
    norm_num_groups=8,
    addition_embed_type="text_time",
    addition_time_embed_dim=TIME_EMBED,
    projection_class_embeddings_input_dim=6 * TIME_EMBED + PROJ,
)


def write_tokenizer(folder: Path, pad_token: str) -> None:
    """A character-level CLIP tokenizer (printable ASCII) — SDXL's two tokenizers differ in pad."""
    folder.mkdir(parents=True, exist_ok=True)
    chars = [chr(i) for i in range(33, 127)]
    vocab = chars + [c + "</w>" for c in chars] + ["<|startoftext|>", "<|endoftext|>"]
    (folder / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}), encoding="utf-8")
    (folder / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (folder / "pad_token.txt").write_text(pad_token, encoding="utf-8")


def _tokenizer(folder: Path):
    from transformers import CLIPTokenizer

    pad = (folder / "pad_token.txt").read_text(encoding="utf-8")
    return CLIPTokenizer(str(folder / "vocab.json"), str(folder / "merges.txt"), pad_token=pad, model_max_length=77)


def tiny_diffusers_pipeline(root: Path, v_pred: bool = False):
    import diffusers
    import torch
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    from rengu_flow.model.sdxl import CLIP_CONTEXT_LENGTH, build_noise_scheduler

    root = Path(root)
    if not (root / "tokenizer_2" / "vocab.json").is_file():
        write_tokenizer(root / "tokenizer", "<|endoftext|>")  # SDXL tokenizer: pad == eos
        write_tokenizer(root / "tokenizer_2", "!")  # SDXL tokenizer_2: pad == "!" (id 0)
    tok1, tok2 = _tokenizer(root / "tokenizer"), _tokenizer(root / "tokenizer_2")
    torch.manual_seed(0)
    common = dict(
        vocab_size=len(tok1),
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=CLIP_CONTEXT_LENGTH,
        bos_token_id=tok1.bos_token_id,
        eos_token_id=tok1.eos_token_id,
    )
    te1 = CLIPTextModel(CLIPTextConfig(hidden_size=H1, **common))
    te2 = CLIPTextModelWithProjection(CLIPTextConfig(hidden_size=H2, projection_dim=PROJ, **common))
    unet = diffusers.UNet2DConditionModel(**UNET_CONFIG)
    vae = diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=2,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=64,
    )
    for i, module in enumerate((te1, te2, unet, vae)):
        randomize_(module, 0.05, i)
    return diffusers.StableDiffusionXLPipeline(
        vae=vae.eval(),
        text_encoder=te1,
        text_encoder_2=te2,
        tokenizer=tok1,
        tokenizer_2=tok2,
        unet=unet,
        scheduler=build_noise_scheduler(v_pred),
        add_watermarker=False,
    )


# ---- kit interface ----------------------------------------------------------------------------


def tiny_config(tmp_path, **over) -> dict:
    import torch

    model = {
        "type": "sdxl",
        "dtype": torch.float32,
        "checkpoint_path": str(Path(tmp_path) / "sdxl" / "unused_checkpoint.safetensors"),
        "cache_text_embeddings": True,
    }
    return training_config(model, over)


def _finish_loading(p, pipe):
    """What ``load_diffusion_model`` does after ``from_single_file``."""
    from rengu_flow.model.sdxl import build_noise_scheduler

    pipe.scheduler = build_noise_scheduler(p.v_pred, p.zero_terminal_snr)
    pipe.upcast_vae()
    pipe.unet.train()
    pipe.text_encoder.train()
    pipe.text_encoder_2.train()
    p._pipeline = pipe
    p._set_param_original_name()
    if p.config.get("adapter"):
        p.config["adapter"] = prepare_adapter_config(p.config["adapter"], p.model_config["dtype"])
        p.configure_adapter(p.config["adapter"])
    return p


def tiny_pipeline(config):
    from rengu_flow.model.sdxl import SDXLPipeline

    p = SDXLPipeline(config)
    # The tiny tokenizers live next to the (never read) checkpoint path.
    pipe = tiny_diffusers_pipeline(Path(config["model"]["checkpoint_path"]).parent, p.v_pred)
    return _finish_loading(p, pipe)


def reload_export(config, save_dir):
    """``save_model``'s single file read back through diffusers ``from_single_file`` (the SDXL
    loader), with the tiny pipeline's configs standing in for the Hub ones."""
    import diffusers

    from rengu_flow.model.sdxl import SDXLPipeline
    from rengu_flow.utils.diffusers_tf5_compat import apply_diffusers_transformers_v5_single_file_patch

    root = Path(config["model"]["checkpoint_path"]).parent
    configs = root / "diffusers_configs"
    if not (configs / "model_index.json").is_file():
        tiny_diffusers_pipeline(root).save_pretrained(configs)
    p = SDXLPipeline(config)
    apply_diffusers_transformers_v5_single_file_patch()
    pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
        str(Path(save_dir) / "model.safetensors"),
        config=str(configs),
        torch_dtype=config["model"]["dtype"],
        add_watermarker=False,
        local_files_only=True,
    )
    return _finish_loading(p, pipe)


def denoiser(pipeline):
    """The trained network (the contract's ``pipeline.transformer`` for DiTs)."""
    return pipeline.unet


def adapter_networks(pipeline) -> dict:
    """The adapted networks by ``original_name`` prefix: the UNet and both text encoders."""
    return {"unet.": pipeline.unet, "text_encoder.": pipeline.text_encoder, "text_encoder_2.": pipeline.text_encoder_2}


def adapter_scope(pipeline, adapter: str) -> list[str]:
    """Every ``nn.Linear`` of the UNet's down/mid/up blocks and of both text encoders (the roots
    ``lora_sdxl`` / ``lokr_vendored`` / ``lycoris_sdxl`` attach to), as the export names them:
    kohya-flat for LoRA and LyCORIS, dotted ``unet.`` / ``text_encoder*.`` paths for LoKr, with the
    canonical ``text_model`` text-encoder names whatever the transformers version (a flat
    transformers-5 CLIPTextModel has none of its own)."""
    from torch import nn

    from rengu_flow.networks import sdxl_te_names

    unet = pipeline.unet
    roots = {
        "unet.": [("down_blocks", unet.down_blocks), ("mid_block", unet.mid_block), ("up_blocks", unet.up_blocks)],
        "text_encoder.": [("", pipeline.text_encoder)],
        "text_encoder_2.": [("", pipeline.text_encoder_2)],
    }
    names = []
    for prefix, containers in roots.items():
        for base, container in containers:
            for name, module in container.named_modules():
                if isinstance(module, nn.Linear):
                    path = ".".join(part for part in (base, name) if part)
                    names.append(prefix + path if adapter == "lokr" else KOHYA_PREFIX[prefix] + path.replace(".", "_"))
    canonical = sdxl_te_names.canonical_dotted if adapter == "lokr" else sdxl_te_names.canonical_flat
    return sorted(canonical({name: None for name in names}))


def monolithic_forward(pipeline, features):
    noisy, timesteps, encoder_hidden_states, pooled, add_time_ids = features[:5]
    return pipeline.unet(
        noisy,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
    ).sample


def _t2i_batch(pipeline, seed: int, chunks=(1, 2, 1)) -> dict:
    """Cached SDXL rows: per-caption CLIP chunks of 77 tokens (a long caption has two)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    b = len(chunks)
    return {
        "latents": torch.randn(b, 4, 8, 8, generator=g),
        "mask": torch.ones(b, 64, 64),  # SDXL's loss reads the mask (the dataset always yields one)
        "prompt_embeds": [torch.randn(77 * n, H1, generator=g) for n in chunks],
        "prompt_embeds_2": [torch.randn(77 * n, H2, generator=g) for n in chunks],
        "pooled_prompt_embeds": torch.randn(b, PROJ, generator=g),
    }


def preview_config(cfg: bool) -> dict:
    """``[preview]`` sampler keys of a tiny preview (64 px = an 8x8 latent)."""
    return {
        "width": 64,
        "height": 64,
        "num_inference_steps": 2,
        "guidance_scale": 5.0 if cfg else 1.0,
        "negative_prompt": "low quality",
    }


def _write_t2i_tree(root: Path) -> dict:
    """The shared t2i tree with one caption past a CLIP chunk (ragged 77 / 154 rows)."""
    config = write_t2i_tree(root, 64)
    (Path(root) / "wide_b.txt").write_text(LONG_CAPTION, encoding="utf-8")
    return config


def cases() -> tuple[ModelCase, ...]:
    return (
        ModelCase(
            "t2i",
            write_dataset=_write_t2i_tree,
            sample_batch=_t2i_batch,
            expected_grids=frozenset({(8, 8), (6, 12), (12, 6)}),
        ),
    )
