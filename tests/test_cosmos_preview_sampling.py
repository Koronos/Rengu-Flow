"""Tests for Cosmos Predict2 preview sampling (no GPU checkpoints)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from renga_flow.model.cosmos_predict2.block_offload import CosmosBlockOffloader
from renga_flow.model.cosmos_predict2.preview_sampling import (
    PreviewPromptData,
    apply_timestep_shift,
    build_timestep_schedule,
    encode_preview_prompt,
    round_preview_pixels,
)


def test_build_timestep_schedule_monotonic_and_endpoints():
    model_config = {}
    sched = build_timestep_schedule(10, 32, 32, model_config, torch.device("cpu"))
    assert sched.shape == (11,)
    assert sched[0] == pytest.approx(1.0)
    assert sched[-1] == pytest.approx(0.0)
    assert torch.all(sched[:-1] >= sched[1:])


def test_apply_timestep_shift_with_flux_shift():
    t = torch.tensor([0.5, 1.0])
    out = apply_timestep_shift(t, 64, 64, {"flux_shift": True})
    assert out.shape == t.shape
    assert not torch.allclose(out, t)


def test_round_preview_pixels_multiple_of_16():
    assert round_preview_pixels(500) == 496
    assert round_preview_pixels(512) == 512


def test_encode_preview_prompt_mocked():
    pipeline = MagicMock()
    pipeline.tokenizer = MagicMock()
    pipeline.t5_tokenizer = MagicMock()
    pipeline.text_encoder = MagicMock()
    be = MagicMock()
    be.input_ids = torch.zeros(1, 4, dtype=torch.long)
    be.attention_mask = torch.ones(1, 4, dtype=torch.long)
    with patch("renga_flow.model.cosmos_predict2.preview_sampling.tokenize", return_value=be):
        with patch(
            "renga_flow.model.cosmos_predict2.preview_sampling.compute_text_embeddings",
            return_value=torch.zeros(1, 4, 8),
        ):
            data = encode_preview_prompt(pipeline, "a cat", torch.device("cpu"))
    assert isinstance(data, PreviewPromptData)
    assert data.crossattn_emb.shape == (1, 4, 8)


def test_euler_sample_latents_integrates_forward_transformer():
    from renga_flow.model.cosmos_predict2.preview_sampling import euler_sample_latents

    pipeline = MagicMock()
    pipeline.model_config = {"dtype": "float32"}
    pipeline.use_llm_adapter = False
    pipeline._preview_offloader = None

    prompt_data = PreviewPromptData(
        crossattn_emb=torch.zeros(1, 2, 4),
        attn_mask=torch.ones(1, 2),
        t5_input_ids=torch.zeros(1, 2, dtype=torch.long),
        t5_attn_mask=torch.ones(1, 2),
    )

    with patch(
        "renga_flow.model.cosmos_predict2.preview_sampling.latent_shape_from_pixels",
        return_value=(16, 1, 32, 32),
    ):
        with patch(
            "renga_flow.model.cosmos_predict2.preview_sampling.forward_transformer",
            return_value=torch.zeros(1, 16, 1, 32, 32),
        ) as fwd:
            out = euler_sample_latents(
                pipeline, prompt_data, 512, 512, 3, 42, torch.device("cpu")
            )
    assert out.shape == (1, 16, 1, 32, 32)
    assert fwd.call_count == 3


def test_offload_text_encoder_after_encode_moves_to_cpu():
    from renga_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline

    pipeline = MagicMock(spec=CosmosPredict2Pipeline)
    p = torch.nn.Parameter(torch.zeros(1))
    pipeline.text_encoder = torch.nn.Module()
    pipeline.text_encoder.register_parameter("w", p)
    pipeline.text_encoder.to("cuda")
    pipeline._preview_restore_state = {}

    CosmosPredict2Pipeline.offload_text_encoder_after_encode(
        pipeline, {"preview_offload_text_encoder": True}
    )
    assert next(pipeline.text_encoder.parameters()).device.type == "cpu"


def test_latent_shape_from_pixels_matches_wan_factor():
    pipeline = MagicMock()
    pipeline.vae.model.z_dim = 16
    c, t, h, w = __import__(
        "renga_flow.model.cosmos_predict2.preview_sampling", fromlist=["latent_shape_from_pixels"]
    ).latent_shape_from_pixels(pipeline, 512, 512, torch.device("cpu"))
    assert (c, t, h, w) == (16, 1, 64, 64)


def test_euler_sample_latents_applies_cfg():
    from renga_flow.model.cosmos_predict2.preview_sampling import euler_sample_latents

    pipeline = MagicMock()
    pipeline.model_config = {"dtype": "float32"}
    pipeline.use_llm_adapter = False
    pipeline._preview_offloader = None

    cond = PreviewPromptData(
        crossattn_emb=torch.ones(1, 2, 4),
        attn_mask=torch.ones(1, 2),
        t5_input_ids=torch.zeros(1, 2, dtype=torch.long),
        t5_attn_mask=torch.ones(1, 2),
    )
    uncond = PreviewPromptData(
        crossattn_emb=torch.zeros(1, 2, 4),
        attn_mask=torch.ones(1, 2),
        t5_input_ids=torch.zeros(1, 2, dtype=torch.long),
        t5_attn_mask=torch.ones(1, 2),
    )

    with patch(
        "renga_flow.model.cosmos_predict2.preview_sampling.latent_shape_from_pixels",
        return_value=(16, 1, 8, 8),
    ):
        with patch(
            "renga_flow.model.cosmos_predict2.preview_sampling.forward_transformer",
            side_effect=lambda _p, x, _t, _e: torch.full(
                (x.shape[0], 16, 1, 8, 8), 2.0 if x.shape[0] == 1 else 1.0
            ),
        ) as fwd:
            euler_sample_latents(
                pipeline,
                cond,
                512,
                512,
                1,
                0,
                torch.device("cpu"),
                uncond_prompt_data=uncond,
                guidance_scale=4.0,
            )
    assert fwd.call_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for block offloader device moves")
def test_cosmos_block_offloader_moves_blocks():
    blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    for block in blocks:
        block.to("cuda")
    offloader = CosmosBlockOffloader(blocks, blocks_to_swap=3, device="cuda")
    offloader.wait_for_block(0)
    assert next(blocks[0].parameters()).device.type == "cuda"
    offloader.submit_move_blocks_forward(0)
    assert next(blocks[0].parameters()).device.type == "cpu"
    offloader.teardown()
