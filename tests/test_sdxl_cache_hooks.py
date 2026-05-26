"""SDXL cache hooks: preprocess, text encoders, call_text_encoder_fn (CPU mocks)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

diffusers = pytest.importorskip("diffusers", exc_type=ImportError)

from renga_flow.data.preprocess_media import PreprocessMediaFile
from renga_flow.model.sdxl import SDXLPipeline


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
