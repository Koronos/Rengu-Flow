"""SDXL cache hooks: preprocess, text encoders, call_text_encoder_fn (CPU mocks)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

diffusers = pytest.importorskip("diffusers", exc_type=ImportError)

from rengu_flow.data.preprocess_media import PreprocessMediaFile
from rengu_flow.model.sdxl import SDXLPipeline


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
        "resolutions": [512],
    }


def test_get_preprocess_media_file_fn(sdxl_config):
    model = SDXLPipeline(sdxl_config)
    fn = model.get_preprocess_media_file_fn()
    assert isinstance(fn, PreprocessMediaFile)
    assert fn.support_video is False
    assert fn.round_height == 16
    assert fn.round_width == 16


def test_get_text_encoders_when_cache_enabled(sdxl_config):
    model = SDXLPipeline(sdxl_config)
    te1 = MagicMock()
    te2 = MagicMock()
    pipe = MagicMock()
    pipe.text_encoder = te1
    pipe.text_encoder_2 = te2
    model._pipeline = pipe
    assert model.get_text_encoders() == [te1, te2]


def test_get_text_encoders_empty_when_cache_disabled(sdxl_config):
    sdxl_config["model"]["cache_text_embeddings"] = False
    model = SDXLPipeline(sdxl_config)
    assert model.get_text_encoders() == []


def test_get_call_text_encoder_fn_returns_dict_keys(sdxl_config):
    model = SDXLPipeline(sdxl_config)
    pipe = MagicMock()
    pipe.tokenizer = MagicMock()
    pipe.tokenizer_2 = MagicMock()
    te1 = MagicMock()
    te2 = MagicMock()
    pipe.text_encoder = te1
    pipe.text_encoder_2 = te2
    model._pipeline = pipe

    hidden = torch.randn(1, 4, 8)
    pooled = torch.randn(1, 4)

    with patch.object(
        model,
        "_encode_prompt_embeds_batch",
        side_effect=[hidden, (hidden, pooled)],
    ) as mock_encode:
        fn1 = model.get_call_text_encoder_fn(te1)
        out1 = fn1(["a caption"], False)
        assert "prompt_embeds" in out1
        fn2 = model.get_call_text_encoder_fn(te2)
        out2 = fn2(["a caption"], False)
        assert "prompt_embeds_2" in out2 and "pooled_prompt_embeds" in out2
        assert mock_encode.call_count == 2


def test_vae_fn_caches_distribution_mode(sdxl_config):
    """The cached latent is the posterior mode (mean), scaled like before — not one frozen
    random draw reused every epoch."""
    from types import SimpleNamespace

    class Dist:
        def mode(self):
            return torch.full((1, 4, 2, 2), 3.0)

        def sample(self):
            raise AssertionError("latent_dist.sample() must not be used for caching")

    vae = SimpleNamespace(
        device="cpu",
        dtype=torch.float32,
        config=SimpleNamespace(scaling_factor=0.5, shift_factor=1.0),
        encode=lambda x: SimpleNamespace(latent_dist=Dist()),
    )
    out = SDXLPipeline(sdxl_config).get_call_vae_fn(vae)(torch.zeros(1, 3, 16, 16))
    assert torch.equal(out["latents"], torch.full((1, 4, 2, 2), 1.0))  # (3 - 1) * 0.5
