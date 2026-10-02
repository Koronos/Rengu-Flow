"""SDXL prepare_inputs and InitialLayer with cached text embeddings."""

from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("diffusers", exc_type=ImportError)

from rengu_flow.model.sdxl import InitialLayer, SDXLPipeline


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
        "prompt_embeds": torch.randn(2, 77, 768),
        "prompt_embeds_2": torch.randn(2, 77, 1280),
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


# --------------------------------------------------------------------------------------------
# Data <-> model seam WITHOUT doubles: real CLIP tokenizers (tiny vocab), real tiny CLIP text
# encoders, the real ``Dataset._collate`` and the real ``SDXLPipeline.prepare_inputs`` /
# ``InitialLayer.get_text_conditioning``. Captions differ in length (empty uncond, short,
# longer than one 75-token chunk), which used to crash the cached path at collate/torch.cat.
# --------------------------------------------------------------------------------------------

import json
from types import SimpleNamespace

from rengu_flow.data.dataset import Dataset
from rengu_flow.model.sdxl import (
    CLIP_CONTEXT_LENGTH,
    InitialLayer as _InitialLayer,
    build_noise_scheduler,
    pad_cached_text_embeds,
    tokenize_clip_chunks,
)

pytest.importorskip("transformers", exc_type=ImportError)

SHORT = "a cat"
LONG = " ".join(["a"] * 80)  # 80 one-char words -> 80 tokens: two chunks
CAPTIONS = [SHORT, "", LONG]  # short, uncond (empty), long
H1, H2, PROJ = 16, 24, 8


def _make_tokenizer(tmp_path, name, pad_token):
    from transformers import CLIPTokenizer

    # Printable ASCII, "!" first: id 0, the pad of the real tokenizer_2.
    chars = [chr(i) for i in range(33, 127)]
    vocab = chars + [c + "</w>" for c in chars] + ["<|startoftext|>", "<|endoftext|>"]
    d = tmp_path / name
    d.mkdir()
    (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}))
    (d / "merges.txt").write_text("#version: 0.2\n")
    tok = CLIPTokenizer(
        str(d / "vocab.json"), str(d / "merges.txt"), pad_token=pad_token, model_max_length=77
    )
    return tok, len(vocab)


@pytest.fixture
def clip_stack(tmp_path):
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    tok1, vocab = _make_tokenizer(tmp_path, "tok1", "<|endoftext|>")  # SDXL tokenizer: pad == eos
    tok2, _ = _make_tokenizer(tmp_path, "tok2", "!")  # SDXL tokenizer_2: pad == "!" (id 0)
    torch.manual_seed(0)
    common = dict(
        vocab_size=vocab,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=CLIP_CONTEXT_LENGTH,
        bos_token_id=tok1.bos_token_id,
        eos_token_id=tok1.eos_token_id,
    )
    te1 = CLIPTextModel(CLIPTextConfig(hidden_size=H1, **common)).eval()
    te2 = CLIPTextModelWithProjection(CLIPTextConfig(hidden_size=H2, projection_dim=PROJ, **common)).eval()
    return SimpleNamespace(tok1=tok1, tok2=tok2, te1=te1, te2=te2)


def _model(clip_stack, *, cache_text_embeddings, clip_skip=None, v_pred=False):
    cfg = {
        "model": {
            "type": "sdxl",
            "dtype": "float32",
            "checkpoint_path": "/fake/model.safetensors",
            "cache_text_embeddings": cache_text_embeddings,
            "clip_skip": clip_skip,
            "v_pred": v_pred,
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }
    model = SDXLPipeline(cfg)
    model._pipeline = SimpleNamespace(
        scheduler=build_noise_scheduler(v_pred),
        vae_scale_factor=8,
        tokenizer=clip_stack.tok1,
        tokenizer_2=clip_stack.tok2,
        text_encoder=clip_stack.te1,
        text_encoder_2=clip_stack.te2,
        _get_add_time_ids=lambda o, c, t, dtype, text_encoder_projection_dim: torch.tensor(
            [[*o, *c, *t]], dtype=dtype
        ),
    )
    return model


def _cache_rows(model, clip_stack, captions, batch_size):
    """Rows as the cache layer hands them back: the map fn encodes ``batch_size`` captions per
    call (per text encoder) and the output is unbatched row by row (``cache_utils.unbatch_iter``)."""
    fn1 = model.get_call_text_encoder_fn(clip_stack.te1)
    fn2 = model.get_call_text_encoder_fn(clip_stack.te2)
    rows = []
    for start in range(0, len(captions), batch_size):
        chunk = captions[start : start + batch_size]
        with torch.no_grad():
            out = {**fn1(chunk, False), **fn2(chunk, False)}
        for i, caption in enumerate(chunk):
            row = {key: out[key][i] for key in out}
            rows.append({"latents": torch.randn(4, 8, 8), "mask": None, "caption": caption, **row})
    return rows


@pytest.mark.parametrize("tok_name,pad_id", [("tok1", None), ("tok2", 0)])
def test_clip_chunks_are_padded_to_77_with_the_tokenizers_own_pad(clip_stack, tok_name, pad_id):
    tok = getattr(clip_stack, tok_name)
    pad = tok.eos_token_id if pad_id is None else pad_id
    assert tok.pad_token_id == pad
    ids = tokenize_clip_chunks(CAPTIONS, tok)
    assert ids.dtype == torch.int64
    assert ids.shape == (3, 2 * CLIP_CONTEXT_LENGTH)  # batch filled to the longest caption's 2 chunks
    short = ids[0, :CLIP_CONTEXT_LENGTH].tolist()
    n = len(tok(SHORT, add_special_tokens=False).input_ids)
    assert short[0] == tok.bos_token_id and short[n + 1] == tok.eos_token_id
    assert set(short[n + 2 :]) == {pad}
    empty = ids[1, :CLIP_CONTEXT_LENGTH].tolist()
    assert empty[:2] == [tok.bos_token_id, tok.eos_token_id] and set(empty[2:]) == {pad}
    # 80 tokens: first chunk holds 75 then eos, the second the remaining 5
    assert ids[2, CLIP_CONTEXT_LENGTH - 1].item() == tok.eos_token_id
    long_second = ids[2, CLIP_CONTEXT_LENGTH:].tolist()
    assert long_second[0] == tok.bos_token_id and long_second[6] == tok.eos_token_id


@pytest.mark.parametrize("batch_size", [1, 3], ids=["caching_batch_1", "caching_batch_3"])
def test_cached_ragged_captions_flow_through_real_collate_into_prepare_inputs(clip_stack, batch_size):
    model = _model(clip_stack, cache_text_embeddings=True)
    rows = _cache_rows(model, clip_stack, CAPTIONS, batch_size)  # used to crash in torch.cat at batch>1
    assert [r["prompt_embeds"].shape[0] for r in rows] == [77, 77, 154]
    batch = Dataset._collate(None, rows)
    assert isinstance(batch["prompt_embeds"], list)  # ragged: the collate cannot stack

    (noisy, timesteps, enc, pooled, add_time), (target, mask) = model.prepare_inputs(batch)
    assert enc.shape == (3, 2 * CLIP_CONTEXT_LENGTH, H1 + H2)
    assert pooled.shape == (3, PROJ)
    assert noisy.shape[0] == add_time.shape[0] == 3
    for i, row in enumerate(rows):
        n = row["prompt_embeds"].shape[0]
        expected = torch.cat([row["prompt_embeds"], row["prompt_embeds_2"]], dim=-1)
        torch.testing.assert_close(enc[i, :n], expected)
        assert not enc[i, n:].any()  # shorter rows: zero chunks


def test_cached_equal_length_captions_still_stack(clip_stack):
    model = _model(clip_stack, cache_text_embeddings=True)
    rows = _cache_rows(model, clip_stack, [SHORT, "a dog"], 2)
    batch = Dataset._collate(None, rows)
    assert torch.is_tensor(batch["prompt_embeds"])
    (_, _, enc, pooled, _), _ = model.prepare_inputs(batch)
    assert enc.shape == (2, CLIP_CONTEXT_LENGTH, H1 + H2) and pooled.shape == (2, PROJ)


def test_uncached_ragged_captions_flow_through_collate_and_initial_layer(clip_stack):
    cached = _model(clip_stack, cache_text_embeddings=True)
    rows = _cache_rows(cached, clip_stack, CAPTIONS, 1)
    live = _model(clip_stack, cache_text_embeddings=False)
    batch = Dataset._collate(None, rows)
    (_, _, ids1, ids2, _), _ = live.prepare_inputs(batch)
    assert ids1.shape == ids2.shape == (3, 2 * CLIP_CONTEXT_LENGTH) and ids1.dtype == torch.int64

    layer = _InitialLayer.__new__(_InitialLayer)
    torch.nn.Module.__init__(layer)
    layer.clip_skip = None
    layer.tokenizer, layer.tokenizer_2 = clip_stack.tok1, clip_stack.tok2
    layer.text_encoder, layer.text_encoder_2 = clip_stack.te1, clip_stack.te2
    with torch.no_grad():
        enc, pooled = layer.get_text_conditioning(ids1, ids2)
    assert enc.shape == (3, 2 * CLIP_CONTEXT_LENGTH, H1 + H2) and pooled.shape == (3, PROJ)
    # Live == cached on each row's own chunks (live encodes the filler chunks; the cache zero-fills them).
    for i, row in enumerate(rows):
        n = row["prompt_embeds"].shape[0]
        expected = torch.cat([row["prompt_embeds"], row["prompt_embeds_2"]], dim=-1)
        torch.testing.assert_close(enc[i, :n], expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(pooled[i], row["pooled_prompt_embeds"], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("clip_skip", [None, 1])
def test_clip_skip_selects_the_hidden_layer(clip_stack, clip_skip):
    model = _model(clip_stack, cache_text_embeddings=True, clip_skip=clip_skip)
    ids = tokenize_clip_chunks([SHORT], clip_stack.tok1)
    with torch.no_grad():
        out, _ = model._encode_prompt_embeds_from_input_ids(ids, clip_stack.tok1, clip_stack.te1)
        hs = clip_stack.te1(ids, output_hidden_states=True).hidden_states
    torch.testing.assert_close(out, hs[-2 if clip_skip is None else -(clip_skip + 2)])


def test_stale_unpadded_text_cache_is_rejected_with_a_clear_error():
    stale = torch.randn(2, 12, 768)  # pre-padding caches stored n_tokens + 2 positions
    with pytest.raises(ValueError, match="regenerate_text_cache"):
        pad_cached_text_embeds(stale)
