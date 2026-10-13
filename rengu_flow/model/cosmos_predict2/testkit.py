"""Tiny random Cosmos Predict2 (Anima) for CPU tests (contract suite). See
``rengu_flow/model/testkit_common.py`` for the kit interface. No pytest here.

Only the text encoder goes through the real loader (``text.load_text_stack`` over a tiny Qwen3
``llm_path`` folder with the bundled tokenizer). The DiT and the VAE cannot: the DiT loader infers
the architecture from three known release widths (``config.get_dit_config``) and the Wan VAE
loader builds the fixed 96-wide release VAE, and ``__init__`` parks the VAE stats on CUDA. So
``tiny_pipeline`` builds the real ``CosmosPredict2Pipeline`` around a tiny ``MiniTrainDIT`` (with a
tiny LLM adapter) and a tiny ``WanVAE_`` instead of reading them from disk; the checks that need
the disk loaders are not in ``CAPABILITIES``. Previews run on CPU (the sampler picks cuda:0 only
when CUDA is available).
"""

from __future__ import annotations

from pathlib import Path

from rengu_flow.model.testkit_common import (
    ALL_CAPABILITIES,
    ModelCase,
    prepare_adapter_config,
    randomize_,
    training_config,
    write_t2i_tree,
)

TEXT_DIM = 32  # tiny Qwen3 width = LLM adapter source dim
CTX = 16  # cross-attention width (LLM adapter target dim)
LATENT_CH = 16  # Wan VAE z_dim (the latent stats are 16-channel)
# Contract checks that do not apply to this model, and why.
NOT_APPLICABLE = {
    "disk_loaders": "the DiT loader sizes the model from the release widths (config.get_dit_config) "
    "and the Wan VAE loader builds the fixed release VAE; the tiny DiT/VAE are built in memory",
    "lazy_text_encoder": "CosmosPredict2Pipeline.__init__ loads the text stack eagerly "
    "(load_text_stack) and parks the VAE stats on CUDA",
    "ragged_text": "the text cache stores fixed 512-token rows with attention masks "
    "(text.tokenize pads every caption to 512: the LLM adapter's t5 ids are aligned to it)",
    "save_round_trip": "the export reloads only through load_diffusion_model, which builds the "
    "release-sized DiT and 1024-wide LLM adapter (config.get_dit_config): a tiny export cannot",
}
CAPABILITIES = ALL_CAPABILITIES - NOT_APPLICABLE.keys()
VAE_INPUT_SHAPE = (1, 3, 1, 64, 64)  # (B, C, T, H, W): the VAE fn takes video-shaped pixels
# lycoris_dit skips the LLM adapter's TransformerBlocks on purpose (frozen by default via
# llm_adapter_lr = 0, degrades easily, DyLoRA checkpoint-save crash); lora/lokr wrap them but
# get_param_groups freezes them unless llm_adapter_lr > 0, so the default trained scope matches.
ADAPTER_SCOPE_EXCLUDES = {
    adapter: ("llm_adapter.",)
    for adapter in ("lycoris_locon", "lycoris_loha", "lycoris_lokr", "lycoris_dylora", "lycoris_glora",
                    "lycoris_diag_oft", "lycoris_boft")
}


def preview_config(cfg: bool) -> dict:
    """A tiny ``[preview]`` section with classifier-free guidance on or off."""
    return {
        "width": 64,
        "height": 48,
        "num_inference_steps": 3,
        "guidance_scale": 3.0 if cfg else 1.0,
        "negative_prompt": "low quality",
    }


def tiny_dit(seed: int = 0):
    import torch

    from rengu_flow.model.cosmos_predict2.dit import MiniTrainDIT
    from rengu_flow.model.cosmos_predict2.llm_adapter import LLMAdapter

    torch.manual_seed(seed)
    model = MiniTrainDIT(
        max_img_h=64,
        max_img_w=64,
        max_frames=4,
        in_channels=LATENT_CH,
        out_channels=LATENT_CH,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=32,
        num_blocks=2,
        num_heads=2,
        crossattn_emb_channels=CTX,
        pos_emb_cls="rope3d",
        use_adaln_lora=True,
        adaln_lora_dim=8,
        rope_enable_fps_modulation=False,  # as config.get_dit_config sets for the releases
    )
    # A small stand-in for the 1024-d Anima LLM adapter (same module, tiny dims).
    model.llm_adapter = LLMAdapter(
        source_dim=TEXT_DIM, target_dim=CTX, model_dim=CTX, num_layers=1, num_heads=2, self_attn=True
    )
    randomize_(model, 0.1, seed)
    return model


def tiny_vae():
    """A ``WanVAE`` (the pipeline's wrapper: latent stats + ``.model``) around a tiny ``WanVAE_``."""
    import torch

    from rengu_flow.model.cosmos_predict2.vae import WanVAE
    from rengu_flow.model.cosmos_predict2.wan_vae import WanVAE_

    torch.manual_seed(0)
    vae = object.__new__(WanVAE)
    vae.dtype, vae.device = torch.float32, "cpu"
    vae.mean = torch.tensor(_WAN_MEAN)
    vae.std = torch.tensor(_WAN_STD)
    vae.scale = [vae.mean, 1.0 / vae.std]
    vae.model = (
        WanVAE_(
            dim=8,
            z_dim=LATENT_CH,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
        )
        .eval()
        .requires_grad_(False)
    )
    return vae


# The release VAE's latent stats (vae.WanVAE.__init__).
_WAN_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
_WAN_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


def write_llm_folder(folder: Path) -> Path:
    """A tiny Qwen3 causal LM folder (``model.llm_path``) with the bundled Qwen3 tokenizer."""
    import json
    import shutil

    folder = Path(folder)
    if (folder / "config.json").is_file():
        return folder
    import torch
    import transformers

    from rengu_flow.model.cosmos_predict2.paths import qwen3_config_dir

    torch.manual_seed(0)
    base = json.loads((Path(qwen3_config_dir()) / "config.json").read_text(encoding="utf-8"))
    base.update(
        hidden_size=TEXT_DIM, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
    )
    base.pop("layer_types", None)
    config = transformers.Qwen3Config(**{k: v for k, v in base.items() if k not in ("architectures", "transformers_version")})
    model = transformers.Qwen3ForCausalLM(config)
    randomize_(model, 0.1, 1)
    model.save_pretrained(folder)
    for name in ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
        shutil.copy(Path(qwen3_config_dir()) / name, folder / name)
    return folder


# ---- kit interface ----------------------------------------------------------------------------


def tiny_config(tmp_path, **over) -> dict:
    import torch

    model = {
        "type": "cosmos_predict2",
        "dtype": torch.float32,
        "llm_path": str(write_llm_folder(Path(tmp_path) / "cosmos_predict2" / "llm")),
        # Not read (the tiny DiT / VAE are built in memory, see the module docstring).
        "transformer_path": str(Path(tmp_path) / "cosmos_predict2" / "unused_dit.safetensors"),
        "vae_path": str(Path(tmp_path) / "cosmos_predict2" / "unused_vae.safetensors"),
    }
    return training_config(model, over)


def tiny_pipeline(config):
    from rengu_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline
    from rengu_flow.model.cosmos_predict2.text import load_text_stack

    # CosmosPredict2Pipeline.__init__ + load_diffusion_model, with the tiny DiT / VAE in place of
    # the release-only loaders.
    p = object.__new__(CosmosPredict2Pipeline)
    p._init_pipeline(config)
    p.vae = tiny_vae()
    p.tokenizer, p.t5_tokenizer, p.text_encoder, p.is_generic_llm, p.name = load_text_stack(p.model_config)
    p.text_encoder.requires_grad_(False)
    p.transformer = tiny_dit()
    p.use_llm_adapter = True
    p._maybe_quantize_frozen_dit()
    p._init_transformer_params()
    if config.get("adapter"):
        config["adapter"] = prepare_adapter_config(config["adapter"], config["model"]["dtype"])
        p.configure_adapter(config["adapter"])
    return p


def monolithic_forward(pipeline, features):
    import torch

    x, t, embeds, attn_mask, t5_ids, t5_mask = features[:6]
    model = pipeline.transformer
    crossattn = model.llm_adapter(
        source_hidden_states=embeds,
        target_input_ids=t5_ids,
        target_attention_mask=t5_mask,
        source_attention_mask=attn_mask,
    )
    crossattn[~t5_mask.bool()] = 0
    padding_mask = torch.zeros(x.shape[0], 1, x.shape[3], x.shape[4], dtype=x.dtype)
    return model(x, t.view(-1, 1), crossattn, padding_mask=padding_mask)


def _batch(frames: int):
    def batch(pipeline, seed: int, lengths=(5, 2, 7)) -> dict:
        import torch

        g = torch.Generator().manual_seed(seed)
        b, seq = len(lengths), 512  # text.tokenize pads every caption to 512
        attn = torch.zeros(b, seq, dtype=torch.long)
        t5 = torch.zeros(b, seq, dtype=torch.long)
        embeds = torch.zeros(b, seq, TEXT_DIM)
        for i, n in enumerate(lengths):
            attn[i, :n] = 1
            t5[i, : n + 1] = 1
            embeds[i, :n] = torch.randn(n, TEXT_DIM, generator=g)
        return {
            "latents": torch.randn(b, LATENT_CH, frames, 4, 6, generator=g),
            "mask": None,
            "prompt_embeds": embeds,
            "attn_mask": attn,
            "t5_input_ids": torch.randint(0, 32000, (b, seq), generator=g) * t5,
            "t5_attn_mask": t5,
        }

    return batch


def cases() -> tuple[ModelCase, ...]:
    return (
        ModelCase(
            "t2i",
            write_dataset=lambda root: {**write_t2i_tree(root, 64), "frame_buckets": [1]},
            sample_batch=_batch(1),
            expected_grids=frozenset({(8, 8), (6, 12), (12, 6)}),
        ),
        # Video: 2 latent frames through prepare_inputs / the layers. No dataset writer (the CPU
        # suite does not encode video files), so the seam check covers images only.
        ModelCase("video", write_dataset=None, sample_batch=_batch(2)),
    )
