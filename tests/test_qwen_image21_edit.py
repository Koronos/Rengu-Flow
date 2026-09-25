"""CPU-only tests for Qwen-Image 2.1 image-conditioned (edit) training and previews: the
image-conditioned prompt encode against a literal copy of the reference pipeline's ti2i branch
(tiny Qwen3-VL with a vision tower + deepstack, real Qwen3-VL processor), the lazy vision tower,
condition-latent VAE encoding, the edit training inputs, pipeline-layer parity with the
monolithic transformer (N = 1 and 2 condition images), gradients through activation
checkpointing, and edit previews (KV cache over the condition prefix)."""

from __future__ import annotations

import copy
import glob
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from rengu_flow.model.dit_common import pad_text_embeddings
from rengu_flow.model.dit_common.streaming import LazyStreamedEncoderWithCompanion
from rengu_flow.model.qwen_image21 import loading, preview_sampling
from rengu_flow.model.qwen_image21.dit import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
    build_t2i_img_mask,
    pack_latents,
    t2i_img_shapes,
    unpack_latents,
)
from rengu_flow.model.qwen_image21.layers import FinalLayer, InitialLayer, TransformerLayer, layout_segments
from rengu_flow.model.qwen_image21.pipeline import (
    ADAPTER_LAYER_GROUPS,
    ADAPTER_TARGET_MODULES,
    QwenImage21Pipeline,
    calculate_shift,
)
from rengu_flow.model.qwen_image21.text import (
    drop_index,
    encode_prompts,
    encode_prompts_with_images,
    ti2i_template,
    vision_language_model,
)
from rengu_flow.networks import adapter_dit
from rengu_flow.training.block_swap import NoopOffloader

pytestmark = pytest.mark.no_ui_db

IN_CH = 8
CTX = 12
GRID = (4, 6)  # target latent grid (h, w)


# ---- tiny Qwen3-VL with vision --------------------------------------------------------------


def _tiny_qwen3vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=64,
            vocab_size=151936,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            rope_theta=5000000,
        ),
        # Real patch/merge geometry (16 px patches, 2x2 merge) so the release processor fits;
        # deepstack from two of the three vision blocks into the first text layers.
        vision_config=dict(
            depth=3,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=32,
            deepstack_visual_indexes=[0, 2],
        ),
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    with torch.no_grad():  # exercise the (default zero/one-initialized) norms too
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(0.05 * torch.randn_like(p))
    return model


def _snapshot_processor_dir() -> str | None:
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hits = glob.glob(os.path.join(home, "hub", "models--Qwen--Qwen-Image-2.1", "snapshots", "*", "processor"))
    return hits[0] if hits and os.path.isfile(os.path.join(hits[0], "preprocessor_config.json")) else None


PROCESSOR_SOURCES = [
    pytest.param("bundled", id="bundled_config"),
    pytest.param(
        "snapshot",
        id="release_processor",
        marks=pytest.mark.skipif(_snapshot_processor_dir() is None, reason="Qwen-Image-2.1 snapshot not in the HF cache"),
    ),
]


def _processor(source: str):
    if source == "snapshot":
        from transformers import Qwen3VLProcessor

        # The release's processor/ folder, loaded exactly as diffusers loads it.
        return Qwen3VLProcessor.from_pretrained(_snapshot_processor_dir())
    return loading.load_processor(None, loading.load_tokenizer(None))


def _images():
    g = torch.Generator().manual_seed(3)
    rgb = Image.fromarray((torch.rand(256, 256, 3, generator=g) * 255).to(torch.uint8).numpy(), "RGB")
    rgba = (torch.rand(256, 320, 4, generator=g) * 255).to(torch.uint8)
    rgba[..., 3] = torch.where(rgba[..., 3] > 128, 255, 0)
    return [rgb, Image.fromarray(rgba.numpy(), "RGBA")]


def _reference_ti2i_prompt_embeds(text_encoder, processor, prompt, image):
    """Literal copy of ``QwenImage21Pipeline._get_qwen_prompt_embeds`` (diffusers main 6256aa7)
    on its image-conditioned path, with the ``__init__``-derived ``_drop_idx`` / ``_img_token_id``."""
    from PIL import Image as PILImage

    sys_prompt = "Comprehend and analyze the provided prompt."
    prompt_template_ti2i = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    sys_message = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
    sys_tokens = processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
    _drop_idx = len(sys_tokens[0])
    _img_token_id = processor.tokenizer.encode("<|image_pad|>")[0]

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [" " if not p else p for p in prompt]
    prompts = []
    condition_pil_list = []
    for t in prompt:
        n_imgs = len(image)
        replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
        for i in range(2, n_imgs + 1):
            replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
        template = prompt_template_ti2i.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", replace)
        prompts.append(template.format(t))
    for _ in prompt:
        for img in image:
            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(img)
            if img.mode == "RGBA":
                white = PILImage.new("RGB", img.size, (255, 255, 255))
                white.paste(img, mask=img.getchannel("A"))
                img = white
            condition_pil_list.append(img)

    processor_kwargs = {"text": prompts, "padding": True, "padding_side": "left", "return_tensors": "pt"}
    processor_kwargs["images"] = condition_pil_list
    model_inputs = processor(**processor_kwargs)

    forward_kwargs = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "output_hidden_states": True,
    }
    if hasattr(model_inputs, "pixel_values"):
        forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
    if hasattr(model_inputs, "mm_token_type_ids"):
        forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

    text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = text_encoder(**forward_kwargs)
    finally:
        handle.remove()
    hidden_states = outputs.hidden_states[-1]

    bool_mask = model_inputs.attention_mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_hidden_states = list(torch.split(selected, valid_lengths.tolist(), dim=0))
    split_hidden_states = [e[_drop_idx:] for e in split_hidden_states]
    image_pad_mask = [
        (sample_ids[sample_mask.bool()] == _img_token_id)
        for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
    ]
    image_pad_mask = [e[_drop_idx:] for e in image_pad_mask]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max(e.size(0) for e in split_hidden_states)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])
    return prompt_embeds, encoder_attention_mask, image_pad_mask, _drop_idx


EDIT_PROMPTS = ["make it snowy", "", "Replace the sky with a sunset, 日本語 \"quotes\"\n\nand keep the rest"]


@pytest.mark.parametrize("source", PROCESSOR_SOURCES)
@pytest.mark.parametrize("n_images", [1, 2])
def test_encode_prompts_with_images_matches_reference_pipeline(source, n_images):
    encoder = _tiny_qwen3vl()
    processor = _processor(source)
    images = _images()[:n_images]
    expected, expected_mask, expected_pads, ref_drop_idx = _reference_ti2i_prompt_embeds(
        encoder, processor, EDIT_PROMPTS, images
    )
    assert drop_index(processor.tokenizer) == ref_drop_idx == 14
    slots = sum((im.size[0] // 32) * (im.size[1] // 32) for im in images)
    assert expected_pads.sum(1).tolist() == [slots] * len(EDIT_PROMPTS)

    # Full model, the composed shell (separately loaded text decoder + vision tower) and one row
    # at a time all agree with the reference bit for bit.
    rows = [images] * len(EDIT_PROMPTS)
    full = encode_prompts_with_images(encoder, processor, EDIT_PROMPTS, rows, device="cpu")
    shell = vision_language_model(
        copy.deepcopy(encoder.model.language_model), copy.deepcopy(encoder.model.visual), encoder.config
    )
    composed = encode_prompts_with_images(shell, processor, EDIT_PROMPTS, rows, device="cpu")
    for embeds, mask, pads in (full, composed):
        assert torch.equal(embeds, expected)
        assert torch.equal(mask, expected_mask.bool())
        assert torch.equal(pads, expected_pads.bool())
    for i, prompt in enumerate(EDIT_PROMPTS):
        e, m, pads = encode_prompts_with_images(shell, processor, [prompt], [images], device="cpu")
        n = int(m.sum())
        assert torch.equal(pads[0, :n], expected_pads[i, :n].bool())
        assert torch.allclose(e[0, :n], expected[i, :n], atol=1e-5)


def test_ti2i_template_numbers_the_placeholders():
    t = ti2i_template(3).format("x")
    assert "<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|>" in t
    assert t.count("<|image_pad|>") == 3 and t.endswith("x<|im_end|>\n<|im_start|>assistant\n")


def _save_sharded(model, folder: Path):
    from safetensors.torch import save_file

    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    folder.mkdir()
    save_file(sd, str(folder / "model.safetensors"))
    model.config.save_pretrained(folder)


def test_vision_tower_loads_from_the_text_encoder_checkpoint(tmp_path):
    full = _tiny_qwen3vl()
    _save_sharded(full, tmp_path / "text_encoder")
    vision = loading.load_vision_encoder(tmp_path / "text_encoder", torch.float32)
    assert len(vision.deepstack_merger_list) == 2 and not any(p.requires_grad for p in vision.parameters())
    ref = full.model.visual.state_dict()
    assert all(torch.equal(v, ref[k]) for k, v in vision.state_dict().items())
    # A text-only file has no vision tower: clear configuration error.
    from safetensors.torch import save_file

    text_only = tmp_path / "text_only.safetensors"
    save_file({k: v.contiguous() for k, v in full.state_dict().items() if "visual" not in k}, str(text_only))
    from rengu_flow.config.validation import ConfigValidationError

    with pytest.raises(ConfigValidationError, match="vision tower"):
        loading.load_vision_encoder(text_only, torch.float32)


# ---- pipeline text-encoder fn ---------------------------------------------------------------


def _stub_pipeline(**model_extra) -> QwenImage21Pipeline:
    p = object.__new__(QwenImage21Pipeline)
    p.config = {"model": {"dtype": torch.float32, **model_extra}}
    p.model_config = p.config["model"]
    return p


def _te_pipeline(encoder, calls):
    p = _stub_pipeline()
    p.tokenizer = loading.load_tokenizer(None)
    p.drop_idx = 14
    p._processor = _processor("bundled")
    p._vlm_shell = None
    p._preview_embed_cache = {}
    p.text_encoder = LazyStreamedEncoderWithCompanion(
        lambda: (calls.append("text"), encoder.model.language_model)[1],
        layers_of=lambda m: m.layers,
        companion_loader=lambda: (calls.append("vision"), encoder.model.visual)[1],
    )
    return p


def _rows(embeds, mask):
    """Per-row valid tokens of a right-padded ``(B, L, ...)`` batch."""
    return [e[m] for e, m in zip(embeds, mask)]


def test_text_encoder_fn_t2i_is_unchanged_and_edit_adds_image_pad_mask(monkeypatch):
    """Every output is one row per caption at its own valid length (no False tails from the
    caching batch's longest caption), in the reference encode's values."""
    encoder = _tiny_qwen3vl()
    monkeypatch.setattr(loading, "load_qwen3vl_config", lambda _path: encoder.config)
    calls = []
    p = _te_pipeline(encoder, calls)
    p.model_config["diffusers_path"] = "unused"
    fn = p.get_call_text_encoder_fn(p.text_encoder)
    tokenizer = p.tokenizer

    # Text-to-image captions: the old values per row, and the vision tower is never read.
    captions = ["a cat", "a much longer caption about a dog"]
    plain = fn(captions, [False, False])
    ref, ref_mask = encode_prompts(encoder.model.language_model, tokenizer, captions, device="cpu")
    assert not bool(ref_mask.all())  # the reference batch is padded
    assert set(plain) == {"prompt_embeds", "text_mask"}
    assert [e.shape[0] for e in plain["prompt_embeds"]] == ref_mask.sum(1).tolist()
    for got, want in zip(plain["prompt_embeds"], _rows(ref, ref_mask)):
        assert torch.equal(got, want)
    assert all(bool(m.all()) for m in plain["text_mask"])
    padded, padded_mask = pad_text_embeddings(plain["prompt_embeds"], plain["text_mask"])
    assert torch.equal(padded, ref) and torch.equal(padded_mask, ref_mask)
    assert len(fn(["a cat"], [False], [None])["prompt_embeds"]) == 1
    assert calls == ["text"]

    # Edit captions: the reference encode + image_pad_mask; the vision tower loads once.
    images = _images()
    out = fn(["make it snowy", "add a hat"], [False, False], [images[:1], images[:1]])
    ref_e, ref_m, ref_p, _ = _reference_ti2i_prompt_embeds(encoder, p._processor, ["make it snowy", "add a hat"], images[:1])
    ref_m = ref_m.bool()
    for got, want in zip(out["prompt_embeds"], _rows(ref_e, ref_m)):
        assert torch.equal(got, want)
    for got, want in zip(out["image_pad_mask"], _rows(ref_p.bool(), ref_m)):
        assert torch.equal(got, want)
    out = fn(["x"], [False], [images])
    assert int(out["image_pad_mask"][0].sum()) == 64 + 80
    assert calls == ["text", "vision"]
    assert p.text_encoder.companion is encoder.model.visual
    # The shell never keeps the encoder alive: unloading it must be able to free the weights.
    assert p._vlm_shell.language_model is None and p._vlm_shell.visual is None

    # A mixed batch: the t2i row equals its text-only encode, with an all-False image_pad_mask.
    mixed = fn(["make it snowy", "a cat"], [False, False], [images[:1], None])
    ref_cat, _ = encode_prompts(encoder.model.language_model, tokenizer, ["a cat"], device="cpu")
    assert torch.equal(mixed["prompt_embeds"][1], ref_cat[0])
    assert [m.shape[0] for m in mixed["text_mask"]] == [e.shape[0] for e in mixed["prompt_embeds"]]
    assert [m.shape[0] for m in mixed["image_pad_mask"]] == [e.shape[0] for e in mixed["prompt_embeds"]]
    assert not mixed["image_pad_mask"][1].any() and int(mixed["image_pad_mask"][0].sum()) == 64

    # Condition images the processor would resize (area < 256x256) are refused.
    small = Image.new("RGB", (128, 128))
    with pytest.raises(ValueError, match="vision slots"):
        fn(["x"], [False], [[small]])


def test_companion_follows_encoder_placement_and_unload():
    calls = []
    encoder = _tiny_qwen3vl()
    enc = LazyStreamedEncoderWithCompanion(
        lambda: encoder.model.language_model,
        layers_of=lambda m: m.layers,
        companion_loader=lambda: (calls.append(1), copy.deepcopy(encoder.model.visual))[1],
    )
    enc.to("cpu")
    assert enc.companion is None and calls == []  # placement never loads it
    enc.load_companion()
    enc.load_companion()
    assert calls == [1]
    # parameters() still only probes the encoder (the data layer's placement check).
    assert len(list(enc.parameters())) == len(list(encoder.model.language_model.parameters()))
    enc.to("meta")
    assert enc.companion is None and not enc.is_loaded


def test_unloading_a_streamed_encoder_returns_the_pinned_host_cache(monkeypatch):
    """Regression (2ea6427): _stop_streaming() clears the masters, so unload() must read whether
    it had pinned masters first, or the ~16 GB of pinned blocks stay in torch's host cache."""
    calls = []
    enc = LazyStreamedEncoderWithCompanion(
        lambda: nn.Linear(2, 2), layers_of=lambda m: [], companion_loader=lambda: nn.Linear(2, 2)
    )
    enc.load()
    enc._masters[0] = torch.zeros(1)  # as if streaming from pinned masters
    enc._stream_device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append(1), raising=False)
    enc.to("meta")
    assert calls == [1] and not enc.is_loaded


# ---- VAE fn ---------------------------------------------------------------------------------


class _Dist:
    def __init__(self, x):
        self.x = x

    def sample(self):
        return self.x

    def mode(self):
        return self.x + 1.0  # distinguishable from sample()


class _StubVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(latents_mean=[1.0] * 64, latents_std=[2.0] * 64)
        self.seen = []

    def encode(self, x):
        self.seen.append(x)
        b, _, _, h, w = x.shape
        return SimpleNamespace(latent_dist=_Dist(torch.full((b, 64, 1, h // 16, w // 16), 5.0)))


def test_vae_fn_encodes_condition_images_with_alpha_mode_and_normalization():
    p = _stub_pipeline()
    p.vae = _StubVae()
    fn = p.get_call_vae_fn(p.vae)
    target = torch.zeros(2, 3, 32, 64)
    controls = [torch.zeros(2, 3, 1, 64, 32), torch.zeros(2, 4, 1, 32, 32)]
    out = fn(target, controls)
    assert set(out) == {"latents", "control_latents_0", "control_latents_1"}
    assert out["latents"].shape == (2, 64, 2, 4) and torch.all(out["latents"] == 2.5)  # mode: (6 - 1) / 2
    assert out["control_latents_0"].shape == (2, 64, 1, 4, 2)
    assert out["control_latents_1"].shape == (2, 64, 1, 2, 2)
    assert torch.all(out["control_latents_0"] == 2.5)  # mode: (6 - 1) / 2
    seen = p.vae.seen
    assert seen[1].shape == (2, 4, 1, 64, 32) and torch.all(seen[1][:, 3] == 1.0)  # opaque alpha added
    assert seen[2].shape == (2, 4, 1, 32, 32)  # RGBA passes through
    assert set(fn(target)) == {"latents"}  # t2i: one argument, unchanged output


# ---- training inputs and layers ---------------------------------------------------------------


def tiny_transformer(seed: int = 0) -> QwenImage21Transformer2DModel:
    torch.manual_seed(seed)
    return QwenImage21Transformer2DModel(
        in_channels=IN_CH,
        out_channels=IN_CH,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        context_in_dim=CTX,
        mlp_ratio=3,
        axes_dims_rope=(2, 2, 4),
    ).eval()


@pytest.fixture
def tiny_model() -> QwenImage21Transformer2DModel:
    return tiny_transformer()


# Condition latent grids per case (latent tokens; each 2x2 group is one vision slot).
CONDITION_GRIDS = {1: [(2, 4)], 2: [(2, 4), (4, 2)]}


def _edit_batch(n_images: int, text_lens=(9, 7), seed: int = 1):
    """Cached-row style edit batch: per-sample prompt rows (vision slots inside the prompt at the
    same place in every row, right padding), N condition latents and the target latents."""
    g = torch.Generator().manual_seed(seed)
    grids = CONDITION_GRIDS[n_images]
    prefix = [False, False]  # "user\n<image1>" tokens before the first vision slot
    for i, (gh, gw) in enumerate(grids):
        prefix += [False] + [True] * (gh * gw // 4) + [False]  # vision_start, slots, vision_end
        if i + 1 < len(grids):
            prefix += [False, False]  # " <image2>"
    embeds, masks, pads = [], [], []
    for n_text in text_lens:
        row_pad = torch.tensor(prefix + [False] * n_text)
        embeds.append(torch.randn(row_pad.numel(), CTX, generator=g))
        masks.append(torch.ones(row_pad.numel(), dtype=torch.bool))
        pads.append(row_pad)
    b = len(text_lens)
    inputs = {
        "latents": torch.randn(b, IN_CH, *GRID, generator=g),
        "mask": None,
        "prompt_embeds": embeds,
        "text_mask": masks,
        "image_pad_mask": pads,
    }
    for i, (gh, gw) in enumerate(grids):
        inputs[f"control_latents_{i}"] = torch.randn(b, IN_CH, 1, gh, gw, generator=g)
    return inputs


def _layers(model):
    return (
        [InitialLayer(model)]
        + [TransformerLayer(b, i, NoopOffloader()) for i, b in enumerate(model.transformer_blocks)]
        + [FinalLayer(model)]
    )


def _run_layers(model, x):
    for layer in _layers(model):
        x = layer(x)
    return x


def _monolithic_edit(model, inputs, noisy, t):
    from rengu_flow.model.dit_common import pad_text_embeddings

    text, text_mask = pad_text_embeddings(inputs["prompt_embeds"], inputs["text_mask"])
    pads, _ = pad_text_embeddings(inputs["image_pad_mask"], inputs["image_pad_mask"])
    b, _, h, w = noisy.shape
    keys = sorted(k for k in inputs if k.startswith("control_latents_"))
    conds = [inputs[k] for k in keys]
    hidden = torch.cat([*[pack_latents(c) for c in conds], pack_latents(noisy)], dim=1)
    img_shapes = [[*[(1, c.shape[-2], c.shape[-1]) for c in conds], (1, h, w)]] * b
    img_mask = torch.cat([pads.bool(), torch.ones(b, h * w // 4, dtype=torch.bool)], dim=1)
    out = model(hidden, text, t.view(-1), img_shapes, img_mask, None if text_mask.all() else text_mask)
    return unpack_latents(out[:, -h * w :], h, w)


@pytest.mark.parametrize("n_images", [1, 2])
@pytest.mark.parametrize("text_lens", [(9, 7), (8, 8)], ids=["padded", "all_valid"])
def test_edit_layers_match_monolithic_forward(tiny_model, n_images, text_lens):
    inputs = _edit_batch(n_images, text_lens)
    p = _stub_pipeline()
    torch.manual_seed(0)
    features, (target, mask) = p.prepare_inputs(inputs)
    assert len(features) == 7 and all(torch.is_tensor(f) for f in features)
    noisy, t = features[0], features[1]
    with torch.no_grad():
        expected = _monolithic_edit(tiny_model, inputs, noisy, t)
        actual = _run_layers(tiny_model, features)
    assert actual.shape == target.shape == noisy.shape
    assert torch.allclose(actual, expected, atol=1e-5)


def test_edit_prepare_inputs_noises_only_the_target_with_the_target_shift():
    inputs = _edit_batch(2)
    p = _stub_pipeline()
    q = 0.3
    (noisy, t, embeds, text_mask, img_mask, control_latents, layout), (target, mask) = p.prepare_inputs(
        inputs, timestep_quantile=q
    )
    latents = inputs["latents"]
    t4 = t.view(-1, 1, 1, 1)
    assert torch.allclose(noisy, (1 - t4) * latents + t4 * (target + latents), atol=1e-6)
    # Condition latents are clean, packed in order.
    expected = torch.cat([pack_latents(inputs["control_latents_0"]), pack_latents(inputs["control_latents_1"])], 1)
    assert torch.equal(control_latents, expected)
    assert layout.shape == (2, 2, 4, 4, 2, 0) and layout.numel() == 0  # (B, grids..., 0)
    # Shift from the target's own sequence length (the reference's latents.shape[1]).
    import math

    raw = torch.sigmoid(torch.distributions.Normal(0, 1).icdf(torch.tensor(q)))
    mu = calculate_shift(GRID[0] * GRID[1])
    assert torch.allclose(t.view(-1), (math.exp(mu) / (math.exp(mu) + (1 / raw - 1))).expand(2).float(), atol=1e-6)
    # img_mask = the cached image_pad_mask (padded) + one slot per 2x2 target group.
    assert img_mask.shape == (2, embeds.shape[1] + GRID[0] * GRID[1] // 4)
    assert int(img_mask[0].sum()) == 2 + 2 + 6 and mask is None
    assert text_mask[1].tolist()[-2:] == [False, False]


def test_edit_inputs_reject_inconsistent_caches():
    p = _stub_pipeline()
    stale = _edit_batch(1)
    del stale["image_pad_mask"]
    with pytest.raises(ValueError, match="image_pad_mask"):
        p.prepare_inputs(stale)
    wrong = _edit_batch(1)
    wrong["control_latents_0"] = torch.randn(2, IN_CH, 1, 4, 4)  # 4 slots, the prompt has 2
    with pytest.raises(ValueError, match="vision slots"):
        p.prepare_inputs(wrong)


def test_layout_round_trips_the_model_segments(tiny_model):
    inputs = _edit_batch(2)
    features, _ = _stub_pipeline().prepare_inputs(inputs)
    layout = InitialLayer(tiny_model)(features)[5]
    prefix_len, segments = layout_segments(layout)
    # The model's own prefill structure for the same inputs.
    b, _, h, w = features[0].shape
    hidden = torch.cat([features[5], pack_latents(features[0])], 1)
    grids = features[6].shape[1:-1]
    shapes = [[(1, grids[0], grids[1]), (1, grids[2], grids[3]), (1, h, w)]] * b
    prepared = tiny_model.prepare_inputs(hidden, features[2], features[1].view(-1), shapes, features[4], features[3])
    assert segments == prepared.segments and prefix_len == prepared.prefix_len
    assert [s[2] for s in segments] == [True, False, True, False, True]


def test_t2i_layer_path_is_bit_identical_to_the_previous_implementation(tiny_model):
    """The t2i tuple and outputs are unchanged by the edit support (same 4-tuple input, same
    ``(text_len, h, w, 0)`` layout, same numbers as the text-only formulation)."""
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(2, IN_CH, *GRID, generator=g)
    text = torch.randn(2, 5, CTX, generator=g)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3:] = False
    t = torch.rand(2, generator=g)
    out = InitialLayer(tiny_model)((latents, t.view(-1, 1), text, mask))
    assert out[5].shape == (5, *GRID, 0)
    with torch.no_grad():
        actual = _run_layers(tiny_model, (latents, t.view(-1, 1), text, mask))
        # Previous formulation: one causal text segment, target mask position >= text_len.
        prepared = tiny_model.prepare_inputs(
            pack_latents(latents), text, t, t2i_img_shapes(*GRID, 2), build_t2i_img_mask(5, *GRID, 2), mask
        )
        hidden = prepared.hidden_states
        for block in tiny_model.transformer_blocks:
            hidden = block(
                hidden,
                prepared.modulation,
                rotary_emb=prepared.rotary_emb,
                target_token_mask=torch.arange(hidden.shape[1]) >= 5,
                segments=[(0, 5, True)],
                key_valid=prepared.key_valid,
            )
        target = hidden[:, 5:]
        expected = unpack_latents(
            tiny_model.proj_out(tiny_model.norm_out(target, prepared.temb, torch.ones(target.shape[1], dtype=torch.bool))),
            *GRID,
        )
    assert torch.equal(actual, expected)


ADAPTER_CONFIGS = [
    pytest.param({"type": "lora", "rank": 4, "alpha": 4, "dropout": 0.0, "dtype": torch.float32}, id="lora"),
    pytest.param(
        {"type": "lokr", "rank": 4, "alpha": 4, "factor": -1, "decompose_both": False, "full_matrix": False, "dtype": torch.float32},
        id="lokr",
    ),
]


def _adapted(tiny_model, adapter_cfg):
    """A copy of the tiny DiT with a (seeded) adapter whose zero-initialized factors are nudged
    off zero, so every adapter parameter sits on a live gradient path."""
    model = copy.deepcopy(tiny_model).train()
    for name, p in model.named_parameters():
        p.original_name = name
        p.requires_grad_(False)
    torch.manual_seed(0)
    adapter_dit.configure(model, adapter_cfg, targets=ADAPTER_TARGET_MODULES, layer_groups=ADAPTER_LAYER_GROUPS)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01)
    return model


@pytest.mark.parametrize("adapter_cfg", ADAPTER_CONFIGS)
def test_edit_gradients_reach_every_adapter_through_the_layers(tiny_model, adapter_cfg):
    model = _adapted(tiny_model, adapter_cfg)
    features, _ = _stub_pipeline().prepare_inputs(_edit_batch(2))
    _run_layers(model, features).square().mean().backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None or not p.grad.abs().sum()]
    assert trainable and not missing, missing


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_edit_activation_checkpointing_keeps_adapter_gradients(tiny_model, use_reentrant):
    """Regression guard (krea2 9c29395): checkpointed TransformerLayers must still deliver the
    adapter gradient in edit mode, under both reentrant modes, and match the plain run."""
    from torch.utils.checkpoint import checkpoint

    from rengu_flow.engine.single_device import SequentialPipe

    cfg = {"type": "lora", "rank": 4, "alpha": 4, "dropout": 0.0, "dtype": torch.float32}
    features, _ = _stub_pipeline().prepare_inputs(_edit_batch(2))
    grads = []
    for ac in (False, True):
        model = _adapted(tiny_model, cfg)
        pipe = SequentialPipe(
            _layers(model),
            loss_fn=None,
            activation_checkpoint_interval=1 if ac else 0,
            checkpointable_layers=["TransformerLayer"],
            activation_checkpoint_func=partial(checkpoint, use_reentrant=use_reentrant),
        )
        pipe(features).square().mean().backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
    inner = [n for n in grads[1] if "transformer_blocks" in n]
    assert inner and all(grads[1][n].abs().sum() > 0 for n in inner)
    assert all(torch.allclose(grads[0][n], grads[1][n], atol=1e-6) for n in grads[0])


# ---- previews -----------------------------------------------------------------------------------


def _preview_stub(model):
    return SimpleNamespace(transformer=model, _preview_offloader=None)


def test_denoise_step_edit_matches_forward_and_kv_cache(tiny_model):
    inputs = _edit_batch(2)
    features, _ = _stub_pipeline().prepare_inputs(inputs)
    noisy, t, text, text_mask, img_mask, controls, layout = features
    b, _, h, w = noisy.shape
    grids = layout.shape[1:-1]
    shapes = [[(1, grids[0], grids[1]), (1, grids[2], grids[3]), (1, h, w)]] * b
    target = pack_latents(noisy)
    n = target.shape[1]
    stub = _preview_stub(tiny_model)
    with torch.no_grad():
        expected = tiny_model(torch.cat([controls, target], 1), text, t.view(-1), shapes, img_mask, text_mask)[:, -n:]
        cache = QwenImage21KVCache(2)
        args = (text, t.view(-1), shapes, img_mask, text_mask)
        first = preview_sampling.denoise_step(stub, torch.cat([controls, target], 1), *args, cache, "extract", target_len=n)
        other = target + 0.1
        cached = preview_sampling.denoise_step(
            stub, torch.cat([controls, other], 1), text, t.view(-1) * 0.5, *args[2:], cache, "cached", target_len=n
        )
        fresh = tiny_model(torch.cat([controls, other], 1), text, t.view(-1) * 0.5, shapes, img_mask, text_mask)[:, -n:]
    assert first.shape == (b, n, IN_CH)
    assert torch.allclose(first, expected, atol=1e-5)
    assert torch.allclose(cached, fresh, atol=1e-5)


class _EditPreviewPipe(QwenImage21Pipeline):
    """Tiny end-to-end edit-preview harness: stub VAE + tiny DiT + stub prompt embeddings whose
    vision slots follow the condition images the sampler loaded."""

    def __init__(self, model):
        self.transformer = model
        self.model_config = {"dtype": torch.float32}
        self.encoded = []

        class _Vae(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(latents_mean=[0.0] * IN_CH, latents_std=[1.0] * IN_CH)
                self.use_tiling = False
                self.encoded = []

            @property
            def dtype(self):
                return self.w.dtype

            def enable_tiling(self, *args):
                self.use_tiling = True

            def encode(self, x):
                self.encoded.append(tuple(x.shape))
                b, _, _, hh, ww = x.shape
                z = torch.nn.functional.avg_pool2d(x[:, :, 0], 16).repeat(1, 2, 1, 1)[:, :IN_CH].unsqueeze(2)
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))

            def decode(self, z):
                img = torch.tanh(z[:, :4].repeat_interleave(16, -1).repeat_interleave(16, -2))
                return SimpleNamespace(sample=img)

        self.vae = _Vae()

    def ensure_vae_for_preview(self):
        pass

    def preview_edit_prompt_embeds(self, prompts, control_paths, images, resolution, preview_cfg, device):
        self.encoded.append((list(prompts), [im.size for im in images], resolution))
        g = torch.Generator().manual_seed(len(prompts))
        out = []
        for i in range(len(prompts)):
            pads = [False, False]
            for im in images:
                pads += [False] + [True] * ((im.size[0] // 32) * (im.size[1] // 32)) + [False]
            pads += [False] * (3 + i)
            pads = torch.tensor(pads)
            out.append((torch.randn(pads.numel(), CTX, generator=g), pads))
        return out


@pytest.mark.parametrize("guidance", [1.0, 3.0], ids=["no_cfg", "cfg"])
def test_generate_edit_preview_end_to_end(tiny_model, tmp_path, guidance):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = tmp_path / "control.png"
    Image.new("RGB", (200, 100), (200, 30, 30)).save(src)
    pipe = _EditPreviewPipe(tiny_model.to(device))
    cfg = {"width": 128, "height": 128, "num_inference_steps": 2, "guidance_scale": guidance, "negative_prompt": ""}
    image = preview_sampling.generate_preview_image(pipe, cfg, "make it snowy", 0, 1, control_images=[str(src)])
    # Condition: 2:1 aspect at 128^2 area floored to 32 px -> 160x64; output takes that aspect.
    assert pipe.encoded == [(["make it snowy", ""] if guidance > 1 else ["make it snowy"], [(160, 64)], 128)]
    assert pipe.vae.encoded == [(1, 4, 1, 64, 160)]  # RGB + opaque alpha, the resized size
    assert image.size == (160, 64) and image.mode == "RGB"


def test_preview_prompt_entries_parse_control_images():
    from rengu_flow.utils.preview import normalize_preview_prompts, preview_prompt_entries

    cfg = {
        "prompts": [
            "plain",
            {"name": "snow", "prompt": "make it snowy", "control_images": ["a.png", "b.png"]},
            {"prompt": "one", "control_images": "c.png"},
            {"name": "empty"},
        ]
    }
    assert preview_prompt_entries(cfg) == [
        ("prompt_0", "plain", []),
        ("snow", "make it snowy", ["a.png", "b.png"]),
        ("prompt_2", "one", ["c.png"]),
    ]
    assert normalize_preview_prompts(cfg) == [("prompt_0", "plain"), ("snow", "make it snowy"), ("prompt_2", "one")]


def test_run_previews_passes_control_images_only_to_edit_prompts(monkeypatch):
    from rengu_flow.utils import preview as preview_mod

    calls = []

    class _Model:
        name = "qwen_image21"

        def generate_preview_image(self, cfg, prompt, step, seed, control_images=None):
            calls.append((prompt, control_images))
            return Image.new("RGB", (8, 8))

    monkeypatch.setattr(preview_mod, "_log_preview_image", lambda **kw: None)
    cfg = {"prompts": ["a cat", {"prompt": "make it snowy", "control_images": ["x.png"]}]}
    preview_mod._run_cosmos_previews(_Model(), cfg, preview_mod.normalize_preview_prompts(cfg), None, 0)
    assert calls == [("a cat", None), ("make it snowy", ["x.png"])]


# ---- dataset -> model seam ------------------------------------------------------------------

SEAM_RES = 320  # smallest bucket whose 4:3 / 3:4 controls stay >= 256x256 at 32-px multiples
SEAM_Z = 8


def _seam_tree(root):
    """Mixed dataset like the smoke run: t2i (N=0), edit with one square control (N=1), and edit
    with two controls of distinct aspect ratios (N=2)."""

    def img(path, size, seed):
        path.parent.mkdir(parents=True, exist_ok=True)
        g = torch.Generator().manual_seed(seed)
        Image.fromarray((torch.rand(size[1], size[0], 3, generator=g) * 255).to(torch.uint8).numpy()).save(path)

    for i, stem in enumerate(("a", "b")):
        img(root / "t2i" / f"{stem}.png", (SEAM_RES, SEAM_RES), i)
        (root / "t2i" / f"{stem}.txt").write_text(f"t2i {stem}", encoding="utf-8")
    # Edit captions of different token lengths in one bucket, cached one row at a time: a row
    # longer than the first one (edit1) and one shorter (edit2) — image_pad_mask must be ragged.
    for i, (stem, caption) in enumerate((("p", "red"), ("q", "make q a much longer red instruction"))):
        img(root / "edit1" / "targets" / f"{stem}.png", (SEAM_RES, SEAM_RES), 10 + i)
        (root / "edit1" / "targets" / f"{stem}.txt").write_text(caption, encoding="utf-8")
        img(root / "edit1" / "controls" / f"{stem}.png", (SEAM_RES, SEAM_RES), 20 + i)
    for i, (stem, caption) in enumerate((("u", "merge u with the second image, keep the light"), ("v", "merge"))):
        img(root / "edit2" / "targets" / f"{stem}.png", (SEAM_RES, SEAM_RES), 30 + i)
        (root / "edit2" / "targets" / f"{stem}.txt").write_text(caption, encoding="utf-8")
        img(root / "edit2" / "controls" / f"{stem}_0.png", (400, 300), 40 + i)  # 4:3
        img(root / "edit2" / "controls" / f"{stem}_1.png", (300, 400), 50 + i)  # 3:4


def _seam_pipeline(monkeypatch) -> QwenImage21Pipeline:
    """The real pipeline (VAE fn, text-encoder fn, prepare_inputs, to_layers) over tiny models:
    a tiny RGBA 16x VAE, a tiny Qwen3-VL with vision tower and the release processor geometry,
    and a tiny DiT whose widths match both."""
    from rengu_flow.model.qwen_image21.vae import AutoencoderKLQwenImage21

    encoder = _tiny_qwen3vl()
    monkeypatch.setattr(loading, "load_qwen3vl_config", lambda _path: encoder.config)
    p = _te_pipeline(encoder, [])
    p.model_config["diffusers_path"] = "unused"
    p.config["tread"] = None
    p._init_block_swap_state()
    torch.manual_seed(0)
    p.vae = AutoencoderKLQwenImage21(
        base_dim=8, decoder_base_dim=12, z_dim=SEAM_Z, latents_mean=[0.0] * SEAM_Z, latents_std=[1.0] * SEAM_Z
    ).eval().requires_grad_(False)
    p.transformer = QwenImage21Transformer2DModel(
        in_channels=SEAM_Z,
        out_channels=SEAM_Z,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        context_in_dim=encoder.config.text_config.hidden_size,
        mlp_ratio=3,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    return p


@pytest.mark.parametrize("accumulation", [1, 2])
def test_mixed_dataset_batches_run_through_the_layers(tmp_path, monkeypatch, accumulation):
    """Every batch of a real mixed dataset (N = 0, 1, 2 with distinct control aspects), cached by
    the real DatasetManager through the real VAE / text-encoder fns, survives the loader's
    micro-batch split and runs ``InitialLayer`` -> blocks -> ``FinalLayer``. Regressions: the
    edit ``control_layout`` had no batch dim, so ``split_batch`` sliced the first control grid
    (smoke: "img_shapes accounts for 1040 image tokens but image_pad_mask marks 2032"); and the
    per-row ``image_pad_mask`` was cached as a fixed-width stack, not ragged like the embeddings."""
    import gc

    from rengu_flow.data.dataset import Dataset
    from rengu_flow.data.loader import split_batch
    from rengu_flow.data.manager import DatasetManager
    from rengu_flow.engine import select_backend

    _seam_tree(tmp_path)
    gc.collect()
    p = _seam_pipeline(monkeypatch)
    cfg = {
        "resolutions": [SEAM_RES],
        "enable_ar_bucket": False,
        "directory": [
            {"path": str(tmp_path / "t2i"), "num_repeats": 1},
            *(
                {
                    "path": str(tmp_path / name / "targets"),
                    "control_path": str(tmp_path / name / "controls"),
                    "num_repeats": 1,
                }
                for name in ("edit1", "edit2")
            ),
        ],
    }
    ds = Dataset(cfg, p, training_config={"cache_root": str(tmp_path / "cache")})
    manager = DatasetManager(p, backend=select_backend({"engine": "accelerate"}))
    manager.register(ds)
    manager.cache(unload_models=False)
    ds.post_init(0, 1, {None: 2}, 1, {None: 2})

    layers = p.to_layers()
    seen = set()
    for i in range(len(ds)):
        batch = ds[i]
        grids = tuple(
            tuple(batch[f"control_latents_{j}"].shape[-2:])
            for j in range(sum(k.startswith("control_latents_") for k in batch))
        )
        seen.add(grids)
        torch.manual_seed(i)
        features, label = p.prepare_inputs(batch)
        for micro_features, (target, _mask) in split_batch((features, label), accumulation):
            x = micro_features
            with torch.no_grad():
                for layer in layers:
                    x = layer(x)
            assert x.shape == target.shape and torch.isfinite(x).all()

    square = (SEAM_RES // 16, SEAM_RES // 16)
    assert seen == {(), (square,), ((256 // 16, 352 // 16), (352 // 16, 256 // 16))}


# ---- control validation before any encode ---------------------------------------------------


def _row(target, *sizes):
    from rengu_flow.data.control import ControlRow

    return ControlRow(target, tuple(sizes))


def test_validate_control_rows_accepts_the_limits():
    from rengu_flow.model.qwen_image21.layers import MAX_CONDITION_IMAGES

    p = object.__new__(QwenImage21Pipeline)
    p.validate_control_rows([_row("a.png", (256, 256)), _row("b.png", *[(512, 128)] * MAX_CONDITION_IMAGES)])


@pytest.mark.parametrize(
    "rows, fragments",
    [
        ([_row("small.png", (224, 256))], ["small.png", "224x256", "256x256", "control_resolution"]),
        ([_row("many.png", *[(256, 256)] * 11)], ["many.png", "11 control images", "at most 10", "fewer control images"]),
    ],
)
def test_validate_control_rows_rejects_each_rule(rows, fragments):
    p = object.__new__(QwenImage21Pipeline)
    with pytest.raises(ValueError) as exc:
        p.validate_control_rows(rows)
    for fragment in fragments:
        assert fragment in str(exc.value)


def test_validate_control_rows_aggregates_the_culprits():
    p = object.__new__(QwenImage21Pipeline)
    rows = [_row(f"t{i}.png", (128, 128)) for i in range(8)] + [_row("ok.png", (256, 256))]
    with pytest.raises(ValueError) as exc:
        p.validate_control_rows(rows)
    msg = str(exc.value)
    assert all(f"t{i}.png" in msg for i in range(5))
    assert "t5.png" not in msg and "ok.png" not in msg
    assert "3 more" in msg and "8 edit row" in msg
