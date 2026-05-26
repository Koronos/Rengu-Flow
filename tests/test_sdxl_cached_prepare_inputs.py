"""SDXL prepare_inputs and InitialLayer with cached text embeddings."""

from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("diffusers", exc_type=ImportError)

from renga_flow.model.sdxl import InitialLayer, SDXLPipeline


@pytest.fixture
def sdxl_config():
    return {
        "model": {
            "type": "sdxl",
            "dtype": "float32",
            "checkpoint_path": "/fake/model.safetensors",
            "cache_text_embeddings": True,
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }


def _mock_pipeline():
    pipe = MagicMock()
    pipe.scheduler.config.num_train_timesteps = 1000
    pipe.scheduler.add_noise = lambda latents, noise, t: latents + noise * 0.01
    pipe.scheduler.get_velocity = lambda latents, noise, t: noise
    pipe.vae_scale_factor = 8
    pipe.text_encoder_2.config.projection_dim = 1280
    pipe.unet.num_upsamplers = 3
    return pipe


def test_prepare_inputs_cached_branch(sdxl_config):
    model = SDXLPipeline(sdxl_config)
    pipe = _mock_pipeline()
    model._pipeline = pipe
    model._get_add_time_ids = MagicMock(return_value=torch.zeros(1, 6))

    latents = torch.randn(2, 4, 8, 8)
    inputs = {
        "latents": latents,
        "mask": torch.ones(2, 8, 8),
        "prompt_embeds": torch.randn(2, 10, 768),
        "prompt_embeds_2": torch.randn(2, 10, 1280),
        "pooled_prompt_embeds": torch.randn(2, 1280),
    }
    features, label = model.prepare_inputs(inputs)
    noisy, timesteps, enc, pooled, add_time = features
    assert noisy.shape == latents.shape
    assert enc.shape[-1] == 768 + 1280
    assert pooled.shape == (2, 1280)
    assert add_time.shape[0] == 2


def test_initial_layer_uses_cached_embeddings(sdxl_config):
    pipe = _mock_pipeline()
    layer = InitialLayer(pipe, cache_text_embeddings=True)
    enc = torch.randn(1, 5, 2048)
    pooled = torch.randn(1, 1280)
    add_time = torch.randn(1, 6)
    sample = torch.randn(1, 4, 8, 8)
    timestep = torch.tensor([50])

    layer.get_text_conditioning = MagicMock()
    layer.conv_in = MagicMock(return_value=sample)
    unet = MagicMock()
    unet.num_upsamplers = 3
    unet.get_time_embed = MagicMock(return_value=torch.randn(1, 320))
    unet.time_embedding = MagicMock(return_value=torch.randn(1, 1280))
    unet.get_aug_embed = MagicMock(return_value=None)
    unet.process_encoder_hidden_states = MagicMock(side_effect=lambda **kw: kw["encoder_hidden_states"])
    pipe.unet = unet

    layer.forward((sample, timestep, enc, pooled, add_time))
    layer.get_text_conditioning.assert_not_called()
