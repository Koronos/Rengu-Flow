"""Tiny random Qwen-Image 2.1 for CPU tests (contract suite, model tests). See
``rengu_flow/model/testkit_common.py`` for the kit interface. No pytest here.

Written as a diffusers-layout folder (``model.diffusers_path``): ``transformer/`` (tiny DiT),
``vae/`` (tiny RGBA 16x ``AutoencoderKLQwenImage21``) and ``text_encoder/`` (tiny Qwen3-VL with a
vision tower, transformers layout). No ``processor/``: the bundled tokenizer and image-processor
config are used, as for a release download without it. Cases: text-to-image, and edit with one
(square) or two (4:3 + 3:4) condition images.
"""

from __future__ import annotations

from pathlib import Path

from rengu_flow.model.testkit_common import (
    ALL_CAPABILITIES,
    ModelCase,
    prepare_adapter_config,
    randomize_,
    tiny_qwen3vl,
    training_config,
    write_image,
    write_qwen3vl_folder,
    write_t2i_tree,
)

TEXT_DIM = 32
LATENT_CH = 8
NOT_APPLICABLE: dict[str, str] = {}  # every contract check applies
CAPABILITIES = ALL_CAPABILITIES - NOT_APPLICABLE.keys()
# No real-weights smoke fixture (tests/fixtures/smoke/train_qwen_image21.toml); the tiny kit runs in
# scripts/smoke_mini.py. Declared so the drift test knows it is deliberate.
SMOKE_FIXTURE_EXEMPT = "no real-weights smoke fixture yet (.env.example: \"no smoke fixture yet\"); covered by smoke_mini.py"
VAE_INPUT_SHAPE = (1, 3, 64, 64)
# Smallest bucket whose 4:3 / 3:4 controls stay >= 256x256 (the processor's minimum) at 32-px
# multiples.
EDIT_RES = 320

DIT_CONFIG = dict(
    in_channels=LATENT_CH,
    out_channels=LATENT_CH,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=2,
    context_in_dim=TEXT_DIM,
    mlp_ratio=3,
    axes_dims_rope=(2, 2, 4),
)


def tiny_dit(seed: int = 0):
    from rengu_flow.model.qwen_image21.dit import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel(**DIT_CONFIG)
    randomize_(model, 0.1, seed)
    return model.eval()


def tiny_vae(seed: int = 0):
    import torch

    from rengu_flow.model.qwen_image21.vae import AutoencoderKLQwenImage21

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    vae = AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=12,
        z_dim=LATENT_CH,
        latents_mean=(torch.randn(LATENT_CH, generator=g) * 0.1).tolist(),
        latents_std=(torch.rand(LATENT_CH, generator=g) + 0.5).tolist(),
    )
    return vae.eval().requires_grad_(False)


def write_bundle(root: Path) -> Path:
    """The tiny diffusers-layout Qwen-Image 2.1 folder under ``root`` (reused when present)."""
    root = Path(root)
    if (root / "text_encoder" / "config.json").is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    tiny_dit().save_pretrained(root / "transformer")
    tiny_vae().save_pretrained(root / "vae")
    write_qwen3vl_folder(root / "text_encoder", tiny_qwen3vl(hidden=TEXT_DIM, layers=3))
    return root


# ---- kit interface ----------------------------------------------------------------------------


def tiny_config(tmp_path, **over) -> dict:
    import torch

    model = {
        "type": "qwen_image21",
        "dtype": torch.float32,
        "diffusers_path": str(write_bundle(Path(tmp_path) / "qwen_image21")),
        "text_encoder_offload": "none",
    }
    return training_config(model, over)


def tiny_pipeline(config):
    from rengu_flow.model.qwen_image21.pipeline import QwenImage21Pipeline

    pipeline = QwenImage21Pipeline(config)
    pipeline.load_diffusion_model()
    if config.get("adapter"):
        config["adapter"] = prepare_adapter_config(config["adapter"], config["model"]["dtype"])
        pipeline.configure_adapter(config["adapter"])
    return pipeline


def monolithic_forward(pipeline, features):
    """The transformer's own forward over ``prepare_inputs`` features: ``[prompt | target]``
    (t2i) or ``[prompt with condition slots | condition latents | target]`` (edit)."""
    import torch

    from rengu_flow.model.qwen_image21.dit import build_t2i_img_mask, pack_latents, t2i_img_shapes, unpack_latents

    noisy, t, text, text_mask = features[:4]
    b, _, h, w = noisy.shape
    encoder_mask = None if bool(text_mask.all()) else text_mask
    if len(features) == 4:
        hidden = pack_latents(noisy)
        img_shapes = t2i_img_shapes(h, w, b)
        img_mask = build_t2i_img_mask(text.shape[1], h, w, b)
    else:
        img_mask, control_latents, layout = features[4:7]
        dims = list(layout.shape[1:-1])
        grids = [(dims[i], dims[i + 1]) for i in range(0, len(dims), 2)]
        hidden = torch.cat([control_latents, pack_latents(noisy)], dim=1)
        img_shapes = [[*[(1, gh, gw) for gh, gw in grids], (1, h, w)]] * b
    out = pipeline.transformer(hidden, text, t.view(-1), img_shapes, img_mask, encoder_mask)
    return unpack_latents(out[:, -h * w :], h, w)


def export_transformer_path(save_dir) -> Path:
    """``save_model`` writes a diffusers transformer folder (what ``model.transformer_path`` reads)."""
    return Path(save_dir)


def preview_config(cfg: bool) -> dict:
    return {
        "width": 64,
        "height": 64,
        "num_inference_steps": 3,
        "guidance_scale": 3.0 if cfg else 1.0,
        "negative_prompt": "low quality",
    }


# ---- cases ------------------------------------------------------------------------------------

# Condition latent grids of the synthetic edit batches (each 2x2 group is one vision slot).
CONDITION_GRIDS = {1: [(2, 4)], 2: [(2, 4), (4, 2)]}


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


def _edit_batch(n_images: int):
    """Cached-row edit batch: per-sample prompt rows with the condition images' vision slots at
    the same place in every row (right padding), N condition latents and the target latents."""

    def batch(pipeline, seed: int, text_lens=(9, 7, 4)) -> dict:
        import torch

        g = torch.Generator().manual_seed(seed)
        grids = CONDITION_GRIDS[n_images]
        prefix = [False, False]  # "user\n<image1>" tokens before the first vision slot
        for i, (gh, gw) in enumerate(grids):
            prefix += [False] + [True] * (gh * gw // 4) + [False]  # vision_start, slots, vision_end
            if i + 1 < len(grids):
                prefix += [False, False]
        embeds, masks, pads = [], [], []
        for n_text in text_lens:
            row_pad = torch.tensor(prefix + [False] * n_text)
            embeds.append(torch.randn(row_pad.numel(), TEXT_DIM, generator=g))
            masks.append(torch.ones(row_pad.numel(), dtype=torch.bool))
            pads.append(row_pad)
        b = len(text_lens)
        out = {
            "latents": torch.randn(b, LATENT_CH, 4, 6, generator=g),
            "mask": None,
            "prompt_embeds": embeds,
            "text_mask": masks,
            "image_pad_mask": pads,
        }
        for i, (gh, gw) in enumerate(grids):
            out[f"control_latents_{i}"] = torch.randn(b, LATENT_CH, 1, gh, gw, generator=g)
        return out

    return batch


def _write_edit_tree(n_images: int):
    def write(root: Path) -> dict:
        # Four distinct token lengths: every batch of two mixes caption lengths (ragged rows).
        captions = ("red", "make it a much longer red instruction, keep the light", "merge the two", "a blue sky at dusk")
        for i, caption in enumerate(captions):
            stem = f"s{i}"
            write_image(root / "targets" / f"{stem}.png", (EDIT_RES, EDIT_RES), 10 + i)
            (root / "targets" / f"{stem}.txt").write_text(caption, encoding="utf-8")
            if n_images == 1:
                write_image(root / "controls" / f"{stem}.png", (EDIT_RES, EDIT_RES), 20 + i)
            else:
                write_image(root / "controls" / f"{stem}_0.png", (400, 300), 40 + i)  # 4:3
                write_image(root / "controls" / f"{stem}_1.png", (300, 400), 50 + i)  # 3:4
        return {
            "resolutions": [EDIT_RES],
            "enable_ar_bucket": False,
            "directory": [
                {"path": str(root / "targets"), "control_path": str(root / "controls"), "num_repeats": 1}
            ],
        }

    return write


def cases() -> tuple[ModelCase, ...]:
    square = (EDIT_RES // 16, EDIT_RES // 16)
    return (
        ModelCase(
            "t2i",
            write_dataset=lambda root: write_t2i_tree(root, 128),
            sample_batch=_t2i_batch,
            expected_grids=frozenset({(8, 8), (6, 12), (12, 6)}),
        ),
        ModelCase(
            "edit_1",
            write_dataset=_write_edit_tree(1),
            sample_batch=_edit_batch(1),
            expected_grids=frozenset({square}),
            controls=1,
        ),
        ModelCase(
            "edit_2",
            write_dataset=_write_edit_tree(2),
            sample_batch=_edit_batch(2),
            expected_grids=frozenset({square}),
            controls=2,
        ),
    )
