"""Tiny random Nanosaur2 for CPU tests (contract suite, model tests). See
``rengu_flow/model/testkit_common.py`` for the kit interface. No pytest here.

The three release files are written tiny, in the release formats ``loading`` reads: the DiT with
the real SPRINT layout (6 blocks), the DINOv2/LDM VAE, and Gemma3 with an embedded SentencePiece
tokenizer under ``spiece_model``.
"""

from __future__ import annotations

import io
from pathlib import Path

from rengu_flow.model.testkit_common import (
    ALL_CAPABILITIES,
    ModelCase,
    prepare_adapter_config,
    training_config,
    write_t2i_tree,
)

TEXT_DIM = 32
LATENT_CH = 8
SPM_VOCAB = 80
NOT_APPLICABLE: dict[str, str] = {}  # every contract check applies
CAPABILITIES = ALL_CAPABILITIES - NOT_APPLICABLE.keys()
VAE_INPUT_SHAPE = (1, 3, 64, 64)

FILE_NAMES = {
    "transformer_path": "nanosaur2_diffusion_model.safetensors",
    "vae_path": "nanosaur2_vae.safetensors",
    "text_encoder_path": "nanosaur2_text_encoder.safetensors",
}


# ---- tiny components --------------------------------------------------------------------------


def tiny_dit(seed: int = 0):
    import torch

    from rengu_flow.model.nanosaur2.dit import Nanosaur2Transformer2DModel

    torch.manual_seed(seed)
    model = Nanosaur2Transformer2DModel(
        in_channels=LATENT_CH,
        hidden_size=32,
        num_heads=2,
        num_blocks=6,
        num_text_blocks=1,
        mlp_hidden=48,
        text_embed_dim=TEXT_DIM,
    )
    # Checkpoint-like weights: random everything (offsets, norms) so no path is an identity.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    return model.eval()


def tiny_vae(seed: int = 0):
    import torch

    from rengu_flow.model.nanosaur2.vae import Nanosaur2VAE

    torch.manual_seed(seed)
    vae = Nanosaur2VAE(
        latent_channels=LATENT_CH, dino_dim=64, dino_heads=1, dino_layers=6, decoder_ch=32, decoder_ch_mult=(1, 1, 1, 1, 1)
    )
    with torch.no_grad():
        vae.latent_mean.copy_(torch.randn_like(vae.latent_mean) * 0.1)
        vae.latent_std.copy_(torch.rand_like(vae.latent_std) + 0.5)
    return vae.eval().requires_grad_(False)


def tiny_spm_proto() -> bytes:
    import sentencepiece as spm

    words = "a red fox sits on the mossy rock at dusk girl ears tail smile forest sunlight cat dog blue sky".split()
    sentences = [" ".join(words[i:] + words[:i]) for i in range(len(words))] * 4
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_writer=buf,
        vocab_size=SPM_VOCAB,
        model_type="bpe",
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        byte_fallback=False,
        minloglevel=2,
    )
    return buf.getvalue()


def tiny_gemma_state_dict(vocab: int = SPM_VOCAB) -> dict:
    import torch
    from transformers import Gemma3TextConfig, Gemma3TextModel

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=vocab,
        hidden_size=TEXT_DIM,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    model = Gemma3TextModel(config)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()}


def write_tiny_files(root: Path) -> dict[str, str]:
    """The three release files, tiny, under ``root`` (reused when present)."""
    import torch
    from safetensors.torch import save_file

    root = Path(root)
    paths = {key: root / name for key, name in FILE_NAMES.items()}
    if not all(p.is_file() for p in paths.values()):
        root.mkdir(parents=True, exist_ok=True)
        te_sd = tiny_gemma_state_dict()
        te_sd["spiece_model"] = torch.frombuffer(bytearray(tiny_spm_proto()), dtype=torch.uint8)
        save_file({k: v.contiguous() for k, v in tiny_dit().state_dict().items()}, str(paths["transformer_path"]))
        save_file({k: v.contiguous() for k, v in tiny_vae().state_dict().items()}, str(paths["vae_path"]))
        save_file(te_sd, str(paths["text_encoder_path"]))
    return {k: str(v) for k, v in paths.items()}


# ---- kit interface ----------------------------------------------------------------------------


def tiny_config(tmp_path, **over) -> dict:
    import torch

    model = {"type": "nanosaur2", "dtype": torch.float32, **write_tiny_files(Path(tmp_path) / "nanosaur2")}
    return training_config(model, over)


def tiny_pipeline(config):
    from rengu_flow.model.nanosaur2.pipeline import Nanosaur2Pipeline

    pipeline = Nanosaur2Pipeline(config)
    pipeline.load_diffusion_model()
    if config.get("adapter"):
        config["adapter"] = prepare_adapter_config(config["adapter"], config["model"]["dtype"])
        pipeline.configure_adapter(config["adapter"])
    return pipeline


def monolithic_forward(pipeline, features):
    x, t, context, weights, skip = features[:5]
    return pipeline.transformer(x, t.view(-1), context, weights, sparse_skip=skip.tolist())


def export_transformer_path(save_dir) -> Path:
    """Where ``save_model`` writes the DiT (what ``model.transformer_path`` reloads)."""
    from rengu_flow.model.nanosaur2.pipeline import EXPORT_FILE_NAME

    return Path(save_dir) / EXPORT_FILE_NAME


def preview_config(cfg: bool) -> dict:
    """A tiny ``[preview]`` section with classifier-free guidance on or off."""
    return {
        "width": 64,
        "height": 48,
        "num_inference_steps": 3,
        "guidance_scale": 3.0 if cfg else 1.0,
        "guidance_mode": "alternate",
        "negative_prompt": "low quality",
    }


def _t2i_batch(pipeline, seed: int, lengths=(5, 1, 3, 2)) -> dict:
    import torch

    g = torch.Generator().manual_seed(seed)
    b = len(lengths)
    weights = [torch.ones(n) for n in lengths]
    weights[0][min(1, lengths[0] - 1)] = 1.3  # an emphasized token
    return {
        "latents": torch.randn(b, LATENT_CH, 4, 6, generator=g),
        "mask": None,
        "prompt_embeds": [torch.randn(n, TEXT_DIM, generator=g) for n in lengths],
        "token_weights": weights,
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
