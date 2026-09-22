"""CPU-only unit tests for the vendored Qwen-Image 2.1 transformer and VAE: T2I forward shape,
gradient flow, gradient checkpointing parity, block-causal masking (text is causal and never
sees the image), right-padding invariance, prefix KV-cache parity, and tiny VAE round trip."""

from __future__ import annotations

import pytest
import torch

from rengu_flow.model.qwen_image21.dit import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
    build_t2i_img_mask,
    pack_latents,
    t2i_img_shapes,
    unpack_latents,
)
from rengu_flow.model.qwen_image21.vae import AutoencoderKLQwenImage21

pytestmark = pytest.mark.no_ui_db

IN_CH = 8
CTX = 12
GRID = (4, 6)  # latent grid (h, w): 24 target tokens, 6 VLM slots


@pytest.fixture
def tiny_model() -> QwenImage21Transformer2DModel:
    torch.manual_seed(0)
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


def _inputs(batch: int = 2, text_len: int = 5, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    h, w = GRID
    latents = torch.randn(batch, IN_CH, h, w, generator=g)
    text = torch.randn(batch, text_len, CTX, generator=g)
    timestep = torch.rand(batch, generator=g)
    return latents, text, timestep


def _run(model, latents, text, timestep, text_mask=None, **kwargs):
    batch, _, h, w = latents.shape
    return model(
        hidden_states=pack_latents(latents),
        encoder_hidden_states=text,
        timestep=timestep,
        img_shapes=t2i_img_shapes(h, w, batch),
        img_mask=build_t2i_img_mask(text.shape[1], h, w, batch),
        encoder_hidden_states_mask=text_mask,
        **kwargs,
    )


def test_pack_unpack_roundtrip():
    latents = torch.randn(2, IN_CH, *GRID)
    packed = pack_latents(latents)
    assert packed.shape == (2, GRID[0] * GRID[1], IN_CH)
    assert torch.equal(unpack_latents(packed, *GRID), latents)
    assert torch.equal(pack_latents(latents.unsqueeze(2)), packed)


def test_build_t2i_img_mask_layout():
    mask = build_t2i_img_mask(5, 4, 6, batch_size=3)
    assert mask.shape == (3, 5 + 6) and mask.dtype == torch.bool
    assert not mask[:, :5].any() and mask[:, 5:].all()
    with pytest.raises(ValueError):
        build_t2i_img_mask(5, 3, 6, batch_size=1)


def test_forward_t2i_shape(tiny_model):
    latents, text, timestep = _inputs()
    out = _run(tiny_model, latents, text, timestep)
    n_img = GRID[0] * GRID[1]
    assert out.shape == (2, text.shape[1] + n_img, IN_CH)
    assert torch.isfinite(out).all()


def test_gradient_reaches_img_in_and_blocks(tiny_model):
    tiny_model.train()
    latents, text, timestep = _inputs()
    out = _run(tiny_model, latents, text, timestep)
    out[:, -GRID[0] * GRID[1] :].square().mean().backward()
    for name in ("img_in.weight", "txt_in.in_layer.weight", "modulation.1.weight", "proj_out.weight"):
        grad = tiny_model.get_parameter(name).grad
        assert grad is not None and grad.abs().sum() > 0, name
    for block in tiny_model.transformer_blocks:
        for lin in (block.attn.to_q, block.attn.to_k, block.img_mlp.proj):
            assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0


def test_gradient_checkpointing_matches(tiny_model):
    tiny_model.train()
    latents, text, timestep = _inputs()
    latents.requires_grad_(True)

    out_ref = _run(tiny_model, latents, text, timestep)
    out_ref.square().mean().backward()
    grads_ref = {n: p.grad.clone() for n, p in tiny_model.named_parameters()}
    lat_grad_ref = latents.grad.clone()

    tiny_model.zero_grad()
    latents.grad = None
    tiny_model.enable_gradient_checkpointing()
    assert tiny_model.gradient_checkpointing
    out_ckpt = _run(tiny_model, latents, text, timestep)
    out_ckpt.square().mean().backward()

    torch.testing.assert_close(out_ckpt, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(latents.grad, lat_grad_ref)
    for n, p in tiny_model.named_parameters():
        torch.testing.assert_close(p.grad, grads_ref[n], msg=n)


@torch.no_grad()
def test_text_tokens_are_causal_and_ignore_the_image(tiny_model):
    latents, text, timestep = _inputs(batch=1, text_len=6)
    text_len = text.shape[1]
    base = _run(tiny_model, latents, text, timestep)

    # A different image and timestep leave every text token untouched (causal prefix + t=0 modulation).
    other = _run(tiny_model, torch.randn_like(latents), text, torch.rand(1))
    torch.testing.assert_close(other[:, :text_len], base[:, :text_len], rtol=0, atol=1e-6)
    assert not torch.allclose(other[:, text_len:], base[:, text_len:])

    # Perturbing text token 3 leaves tokens 0..2 untouched but changes 3.. and the image.
    text2 = text.clone()
    text2[:, 3] += 1.0
    pert = _run(tiny_model, latents, text2, timestep)
    torch.testing.assert_close(pert[:, :3], base[:, :3], rtol=0, atol=1e-6)
    assert not torch.allclose(pert[:, 3:text_len], base[:, 3:text_len])
    assert not torch.allclose(pert[:, text_len:], base[:, text_len:])


@torch.no_grad()
def test_image_tokens_attend_bidirectionally(tiny_model):
    latents, text, timestep = _inputs(batch=1)
    base = _run(tiny_model, latents, text, timestep)
    lat2 = latents.clone()
    lat2[..., -1, -1] += 1.0  # last image token
    pert = _run(tiny_model, lat2, text, timestep)
    first_img = text.shape[1]
    assert not torch.allclose(pert[:, first_img], base[:, first_img])


@torch.no_grad()
def test_right_padding_does_not_change_image_output(tiny_model):
    n_img = GRID[0] * GRID[1]
    latents, text, timestep = _inputs(batch=2, text_len=7)
    # Standalone forwards: sample 0 with 7 valid tokens, sample 1 with only 4.
    ref0 = _run(tiny_model, latents[:1], text[:1], timestep[:1])
    ref1 = _run(tiny_model, latents[1:], text[1:, :4], timestep[1:])

    padded = text.clone()
    padded[1, 4:] = 1e3 * torch.randn(3, CTX)  # garbage in the padded slots
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 4:] = False
    out = _run(tiny_model, latents, padded, timestep, text_mask=mask)

    torch.testing.assert_close(out[:1, -n_img:], ref0[:, -n_img:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[1:, -n_img:], ref1[:, -n_img:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[1:, :4], ref1[:, :4], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_all_valid_mask_is_a_noop(tiny_model):
    latents, text, timestep = _inputs()
    ref = _run(tiny_model, latents, text, timestep)
    out = _run(tiny_model, latents, text, timestep, text_mask=torch.ones(2, text.shape[1], dtype=torch.bool))
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-6)


@torch.no_grad()
def test_kv_cache_matches_full_forward(tiny_model):
    n_img = GRID[0] * GRID[1]
    latents, text, _ = _inputs(batch=1)
    cache = QwenImage21KVCache(len(tiny_model.transformer_blocks))
    t0, t1 = torch.tensor([0.9]), torch.tensor([0.4])

    first = _run(tiny_model, latents, text, t0, kv_cache=cache, kv_cache_mode="extract")
    torch.testing.assert_close(first, _run(tiny_model, latents, text, t0), rtol=0, atol=0)

    lat1 = torch.randn_like(latents)
    cached = _run(tiny_model, lat1, text, t1, kv_cache=cache, kv_cache_mode="cached")
    assert cached.shape == (1, n_img, IN_CH)
    full = _run(tiny_model, lat1, text, t1)
    torch.testing.assert_close(cached, full[:, -n_img:], rtol=1e-5, atol=1e-5)


def test_kv_cache_mode_validation(tiny_model):
    latents, text, timestep = _inputs(batch=1)
    with pytest.raises(ValueError):
        _run(tiny_model, latents, text, timestep, kv_cache_mode="cached")
    with pytest.raises(ValueError):
        _run(tiny_model, latents, text, timestep, kv_cache=QwenImage21KVCache(2), kv_cache_mode="bogus")


@torch.no_grad()
def test_vae_tiny_encode_decode_shapes():
    torch.manual_seed(0)
    z_dim = 8
    vae = AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=12,
        z_dim=z_dim,
        latents_mean=[0.0] * z_dim,
        latents_std=[1.0] * z_dim,
    ).eval()
    image = torch.rand(1, 4, 1, 32, 32) * 2 - 1  # RGBA, one frame
    posterior = vae.encode(image).latent_dist
    latents = posterior.mode()
    assert latents.shape == (1, z_dim, 1, 2, 2)  # 16x spatial compression
    decoded = vae.decode(latents).sample
    assert decoded.shape == image.shape
    assert decoded.abs().max() <= 1.0
