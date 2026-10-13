"""CPU-only tests for Nanosaur2 on tiny random models: DiT forward and SPRINT path-drop, pipeline
layers vs the monolithic forward (subset and pipe-parallel modes), text emphasis/tokenizer and the
Gemma3 encode, VAE geometry and loaders, the training recipe (timesteps, loss weight, path-drop
flags), adapter scope and the ComfyUI export keys, full-model export, previews, and the
dataset -> model seam through the real DatasetManager / cache / loader split / layers.

Numerical parity with the official ComfyUI node on the real weights lives in
``scripts/parity_nanosaur2.py`` (needs ComfyUI + the release files)."""

from __future__ import annotations

import io
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from rengu_flow.model.nanosaur2 import loading, preview_sampling, text
from rengu_flow.model.nanosaur2.dit import Nanosaur2Transformer2DModel, detect_config
from rengu_flow.model.nanosaur2.layers import (
    FinalLayer,
    InitialLayer,
    SparseEndLayer,
    SparseStartLayer,
    TransformerLayer,
)
from rengu_flow.model.nanosaur2.pipeline import (
    ADAPTER_LAYER_GROUPS,
    ADAPTER_TARGET_MODULES,
    LOGIT_MEAN,
    Nanosaur2Pipeline,
    pad_text,
)
from rengu_flow.model.nanosaur2.vae import Nanosaur2VAE, detect_vae_config
from rengu_flow.networks import adapter_dit

pytestmark = pytest.mark.no_ui_db

TEXT_DIM = 32
LATENT_CH = 8
# The reference LoRA scope (nanosaur2_support/train_lora.py).
REFERENCE_TARGETS = ("qkv_x", "kv_y", "proj", "qkv", "w12", "w3")


# ---- tiny components --------------------------------------------------------------------------


def tiny_dit(seed: int = 0) -> Nanosaur2Transformer2DModel:
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


def tiny_vae(seed: int = 0) -> Nanosaur2VAE:
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
        vocab_size=80,
        model_type="bpe",
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        byte_fallback=False,
        minloglevel=2,
    )
    return buf.getvalue()


def tiny_gemma_state_dict(vocab: int) -> dict:
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


@pytest.fixture(scope="module")
def tiny_files(tmp_path_factory):
    """The three release files, tiny: DiT, VAE, and the Gemma3 text encoder + spiece_model."""
    root = tmp_path_factory.mktemp("nanosaur2")
    proto = tiny_spm_proto()
    te_sd = tiny_gemma_state_dict(vocab=80)
    te_sd["spiece_model"] = torch.frombuffer(bytearray(proto), dtype=torch.uint8)
    paths = {
        "transformer_path": root / "nanosaur2_diffusion_model.safetensors",
        "vae_path": root / "nanosaur2_vae.safetensors",
        "text_encoder_path": root / "nanosaur2_text_encoder.safetensors",
    }
    save_file({k: v.contiguous() for k, v in tiny_dit().state_dict().items()}, str(paths["transformer_path"]))
    save_file({k: v.contiguous() for k, v in tiny_vae().state_dict().items()}, str(paths["vae_path"]))
    save_file(te_sd, str(paths["text_encoder_path"]))
    return {k: str(v) for k, v in paths.items()}


def make_pipeline(tiny_files, adapter=None, **model_extra) -> Nanosaur2Pipeline:
    config = {"model": {"type": "nanosaur2", "dtype": torch.float32, **tiny_files, **model_extra}}
    if adapter is not None:
        config["adapter"] = adapter
    p = Nanosaur2Pipeline(config)
    p.load_diffusion_model()
    return p


def _inputs(b=3, h=4, w=6, lengths=(5, 1, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, LATENT_CH, h, w, generator=g)
    t = torch.rand(b, generator=g) * 0.9 + 0.05
    length = max(lengths)
    context = torch.randn(b, length, TEXT_DIM, generator=g)
    weights = torch.zeros(b, length)
    for i, n in enumerate(lengths):
        weights[i, :n] = 1.0
    weights[0, 1] = 1.3  # an emphasized token
    return x, t, context, weights


def _run_layers(layers, inputs):
    out = inputs
    for layer in layers:
        out = layer(out)
    return out


def _layers(model, subset=True, pipe_parallel=False):
    n = len(model.blocks)
    layers = [InitialLayer(model, pipe_parallel=pipe_parallel)]
    for i in range(n):
        if i == 2:
            layers.append(SparseStartLayer(subset=subset))
        layers.append(TransformerLayer(model, i))
        if i == n - 3:
            layers.append(SparseEndLayer(model, subset=subset))
    layers.append(FinalLayer(model))
    return layers


# ---- DiT ----------------------------------------------------------------------------------


def test_detect_config_round_trips_the_architecture():
    model = tiny_dit()
    assert detect_config(model.state_dict()) == model.config


def test_forward_returns_velocity_of_latent_shape():
    model = tiny_dit()
    x, t, context, weights = _inputs()
    v = model(x, t, context, weights)
    assert v.shape == x.shape and v.dtype == torch.float32 and torch.isfinite(v).all()


def test_text_padding_is_invisible():
    """Padded text positions (weight 0) must not change any row's output."""
    model = tiny_dit()
    x, t, context, weights = _inputs()
    dirty = context.clone()
    dirty[1, 1:] = 1e3
    wide_ctx = torch.cat([context, torch.full_like(context[:, :2], 7.0)], dim=1)
    wide_w = torch.cat([weights, torch.zeros_like(weights[:, :2])], dim=1)
    ref = model(x, t, context, weights)
    assert torch.allclose(model(x, t, dirty, weights), ref, rtol=1e-4, atol=1e-4)
    assert torch.allclose(model(x, t, wide_ctx, wide_w), ref, rtol=1e-4, atol=1e-4)


def test_sparse_skip_rows_keep_their_dense_activation():
    """Path-drop is per row: skipping rows equals running those rows alone with the sparse
    blocks removed, and the other rows are unaffected."""
    model = tiny_dit()
    x, t, context, weights = _inputs()
    full = model(x, t, context, weights)
    mixed = model(x, t, context, weights, sparse_skip=[False, True, False])
    assert torch.allclose(mixed[[0, 2]], full[[0, 2]], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(mixed[1], full[1], atol=1e-3)
    alone = model(x[1:2], t[1:2], context[1:2], weights[1:2], sparse_skip=[True])
    assert torch.allclose(mixed[1:2], alone, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("skip", [None, [False, True, False], [True, True, True], [False, False, False]])
@pytest.mark.parametrize("mode", ["subset", "pipe_parallel"])
def test_pipeline_layers_match_monolithic_forward(skip, mode):
    model = tiny_dit()
    x, t, context, weights = _inputs()
    expected = model(x, t, context, weights, sparse_skip=skip)
    flags = torch.tensor(skip or [False] * 3)
    layers = _layers(model, subset=mode == "subset", pipe_parallel=mode == "pipe_parallel")
    with torch.no_grad():
        actual = _run_layers(layers, (x, t.view(-1, 1), context, weights, flags))
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_layers_cover_every_parameter_once():
    model = tiny_dit()
    params = {id(p) for layer in _layers(model) for p in layer.parameters()}
    assert params == {id(p) for p in model.parameters()}


def test_gradients_flow_only_through_rows_that_run_the_sparse_path():
    model = tiny_dit().train()
    x, t, context, weights = _inputs()
    layers = _layers(model)
    out = _run_layers(layers, (x, t.view(-1, 1), context, weights, torch.tensor([True, True, True])))
    out.sum().backward()
    sparse = model.blocks[2].attn.qkv_x.weight
    assert sparse.grad is None or torch.count_nonzero(sparse.grad) == 0
    assert torch.count_nonzero(model.blocks[0].attn.qkv_x.weight.grad) > 0
    model.zero_grad()
    out = _run_layers(layers, (x, t.view(-1, 1), context, weights, torch.tensor([True, False, True])))
    out.sum().backward()
    assert torch.count_nonzero(sparse.grad) > 0


# ---- text -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "caption,plain,spans",
    [
        ("a (red:1.3) fox", "a red fox", [(2, 5, 1.3)]),
        ("((cat:1.2):1.5) x", "(cat:1.2) x", [(0, 9, 1.5)]),
        ("no (weight) here (x:abc)", "no (weight) here (x:abc)", []),
        ("unbalanced (a:1.2", "unbalanced (a:1.2", []),
        ("", "", []),
    ],
)
def test_parse_prompt_emphasis(caption, plain, spans):
    assert text.parse_prompt_emphasis(caption) == (plain, spans)


def test_tokenize_with_weights_bos_emphasis_and_truncation():
    spm = text.load_sentencepiece(tiny_spm_proto())
    assert text.tokenize_with_weights(spm, "") == [(2, 1.0)]
    rows = text.tokenize_with_weights(spm, "a (red fox:1.5) sits")
    ids = [t for t, _ in rows]
    assert ids[0] == text.BOS_TOKEN and 1 not in ids  # BOS, never EOS
    assert ids[1:] == spm.encode("a red fox sits")
    pieces = [spm.id_to_piece(t) for t in ids[1:]]
    for piece, (_, weight) in zip(pieces, rows[1:]):
        emphasized = piece.strip("▁") in ("red", "fox", "r", "e", "d", "f", "o", "x", "re", "fo")
        assert weight == (1.5 if emphasized else 1.0), piece
    long = text.tokenize_with_weights(spm, "fox " * 400)
    assert len(long) == text.MAX_LENGTH


def test_encode_prompts_matches_unbatched_and_uses_penultimate_normed_state(tiny_files):
    te = loading.load_text_encoder(tiny_files["text_encoder_path"], torch.float32)
    spm = loading.load_tokenizer(tiny_files["text_encoder_path"])
    prompts = ["", "a red fox sits on the rock", "(cat:1.2) dog"]
    embeds, weights = text.encode_prompts(te, spm, prompts, "cpu")
    assert [e.shape[0] for e in embeds] == [len(text.tokenize_with_weights(spm, p)) for p in prompts]
    assert embeds[0].shape == (1, TEXT_DIM) and weights[0].tolist() == [1.0]
    assert 1.2 in [round(w, 4) for w in weights[2].tolist()]
    for i, p in enumerate(prompts):  # batching (right padding) never leaks into real tokens
        single, _ = text.encode_prompts(te, spm, [p], "cpu")
        assert torch.allclose(embeds[i], single[0], rtol=1e-4, atol=1e-4)
    ids = torch.tensor([[t for t, _ in text.tokenize_with_weights(spm, prompts[1])]])
    hidden = te(input_ids=ids, output_hidden_states=True).hidden_states
    assert torch.allclose(embeds[1], te.norm(hidden[len(te.layers) - 1])[0], rtol=1e-4, atol=1e-4)


def test_text_encoder_loader_keeps_the_checkpoint_weights(tiny_files):
    te = loading.load_text_encoder(tiny_files["text_encoder_path"], torch.float32)
    sd = load_file(tiny_files["text_encoder_path"])
    for key in ("layers.0.self_attn.q_proj.weight", "layers.1.input_layernorm.weight", "norm.weight"):
        assert torch.equal(te.state_dict()[key], sd[f"model.{key}"].float()), key
    assert all(b.device.type == "cpu" for b in te.buffers())
    assert te.embed_tokens.embed_scale.item() == pytest.approx(math.sqrt(TEXT_DIM), rel=1e-6)


# ---- VAE ------------------------------------------------------------------------------------


@pytest.mark.parametrize("hw", [(64, 64), (48, 96), (80, 48)])
def test_vae_downsamples_16_and_round_trips_shapes(hw):
    vae = tiny_vae()
    x = torch.rand(2, 3, *hw) * 2 - 1
    z = vae.encode(x)
    assert z.shape == (2, LATENT_CH, hw[0] // 16, hw[1] // 16)
    img = vae.decode(z)
    assert img.shape == x.shape and img.abs().max() <= 1.0


def test_vae_loader_detects_the_architecture(tiny_files):
    vae = loading.load_vae(tiny_files["vae_path"], torch.float32)
    ref = tiny_vae()
    assert detect_vae_config(ref.state_dict())["decoder_ch_mult"] == (1, 1, 1, 1, 1)
    x = torch.rand(1, 3, 32, 48) * 2 - 1
    assert torch.equal(vae.encode(x), ref.encode(x))


def test_loaders_reject_wrong_files(tiny_files):
    from rengu_flow.config.validation import ConfigValidationError

    with pytest.raises(ConfigValidationError, match="not a Nanosaur2 diffusion model"):
        loading.load_transformer(tiny_files["vae_path"], torch.float32)
    with pytest.raises(ConfigValidationError, match="spiece_model"):
        loading.load_tokenizer(tiny_files["vae_path"])


# ---- training recipe --------------------------------------------------------------------------


def test_timesteps_default_to_the_reference_logit_normal(tiny_files):
    p = make_pipeline(tiny_files)
    q = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.3))
    t = p.sample_timesteps(4, "cpu", timestep_quantile=0.3)
    assert torch.allclose(t, torch.sigmoid(q + LOGIT_MEAN).expand(4))
    torch.manual_seed(0)
    assert p.sample_timesteps(20000, "cpu").logit().mean().item() == pytest.approx(math.log(2), abs=0.03)
    uniform = make_pipeline(tiny_files, timestep_sample_method="uniform", shift=3.0)
    t = uniform.sample_timesteps(4, "cpu", timestep_quantile=0.5)
    assert torch.allclose(t, torch.full((4,), 0.75))  # shift 3 at t = 0.5


def test_sparse_skip_flags_follow_caption_dropout(tiny_files):
    p = make_pipeline(tiny_files, sparse_skip_prob=0.0, sparse_skip_prob_uncond=1.0)
    lengths = torch.tensor([1, 7, 1, 3])
    assert p.sparse_skip_flags(lengths).tolist() == [True, False, True, False]
    assert not p.sparse_skip_flags(lengths, timestep_quantile=0.5).any()
    torch.manual_seed(0)
    ref = make_pipeline(tiny_files)
    flags = ref.sparse_skip_flags(torch.tensor([1] * 4000 + [9] * 4000))
    assert flags[:4000].float().mean().item() == pytest.approx(0.5, abs=0.03)
    assert flags[4000:].float().mean().item() == pytest.approx(0.05, abs=0.01)


def test_loss_equals_the_reference_x0_objective(tiny_files):
    """DiTPipeline's loss over (velocity, target, weight-mask) == the train_lora.py loss
    mean(((x0_pred - x) / (t + 0.05))^2) with x0_pred = z_t - v t."""
    p = make_pipeline(tiny_files)
    p.config["model"] = p.model_config
    torch.manual_seed(0)
    latents = torch.randn(2, LATENT_CH, 4, 6)
    batch = {
        "latents": latents,
        "mask": None,
        "prompt_embeds": [torch.randn(3, TEXT_DIM), torch.randn(1, TEXT_DIM)],
        "token_weights": [torch.ones(3), torch.ones(1)],
    }
    torch.manual_seed(1)
    (zt, t, _ctx, _w, _skip), (target, weight) = p.prepare_inputs(batch)
    v = torch.randn_like(latents)
    loss = p.get_loss_fn()(v, (target, weight))
    tv = t.view(-1, 1, 1, 1)
    x0 = zt - v * tv
    reference = (((x0 - latents) / (tv + 0.05)) ** 2).mean()
    assert loss.item() == pytest.approx(reference.item(), rel=1e-5)


def test_prepare_inputs_pads_ragged_text_and_folds_masks(tiny_files):
    p = make_pipeline(tiny_files)
    batch = {
        "latents": torch.randn(2, LATENT_CH, 4, 6),
        "mask": torch.ones(2, 64, 96),
        "prompt_embeds": [torch.randn(4, TEXT_DIM), torch.randn(1, TEXT_DIM)],
        "token_weights": [torch.tensor([1.0, 1.3, 1.0, 1.0]), torch.ones(1)],
    }
    (zt, t, ctx, w, skip), (target, weight) = p.prepare_inputs(batch, timestep_quantile=0.5)
    assert ctx.shape == (2, 4, TEXT_DIM) and w.tolist()[1] == [1.0, 0.0, 0.0, 0.0]
    assert skip.dtype == torch.bool and not skip.any()
    assert weight.shape == (2, 1, 4, 6) and t.shape == (2, 1)
    trimmed = p.trim_micro_batch((zt[1:], t[1:], ctx[1:], w[1:], skip[1:]))
    assert trimmed[2].shape[1] == 1 and trimmed[3].shape[1] == 1


def test_default_caption_dropout_applies_to_the_training_dataset_only(tiny_files):
    """The 10% default rides on apply_model_training_dataset_defaults (train dataset only,
    independent of --i_know_what_i_am_doing); dataset validation never sets it, so eval
    datasets keep undropped captions; an explicit value wins."""
    from rengu_flow.data.dataset import apply_model_training_dataset_defaults

    p = make_pipeline(tiny_files)
    eval_cfg = {"directory": []}
    p.model_specific_dataset_config_validation(eval_cfg)
    assert "uncond_fraction" not in eval_cfg
    assert apply_model_training_dataset_defaults({"directory": []}, p)["uncond_fraction"] == 0.1
    assert apply_model_training_dataset_defaults({"uncond_fraction": 0.0}, p)["uncond_fraction"] == 0.0
    assert "uncond_fraction" not in apply_model_training_dataset_defaults({}, object())


def test_main_applies_the_training_default_before_building_the_train_dataset_only():
    import inspect

    import rengu_flow.main as main

    src = inspect.getsource(main)
    train = src.index("train_data = Dataset(")
    evals = src.index("eval_data = Dataset(")
    call = src.index("apply_model_training_dataset_defaults(dataset_config, model)")
    assert call < train < evals and src.count("apply_model_training_dataset_defaults(") == 1


# ---- adapters / export ------------------------------------------------------------------------


def _reference_lora_names(model) -> list[str]:
    """train_lora.py's selection."""
    return sorted(
        name
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and name.startswith(("blocks.", "text_refine_blocks."))
        and name.rsplit(".", 1)[-1] in REFERENCE_TARGETS
    )


def test_default_adapter_scope_is_the_reference_lora_scope():
    model = tiny_dit()
    names = sorted(adapter_dit._collect_target_linears(model, ADAPTER_TARGET_MODULES))
    assert names == _reference_lora_names(model)


def test_layer_groups_match_real_modules_and_cover_the_default_scope():
    from rengu_flow.networks.adapter_targets import filter_target_names
    from rengu_flow.registry.model_capabilities import get_capability

    model = tiny_dit()
    names = adapter_dit._collect_target_linears(model, ADAPTER_TARGET_MODULES)
    for group, patterns in ADAPTER_LAYER_GROUPS.items():
        assert filter_target_names(names, list(patterns), None), group
    union = filter_target_names(
        names, [p for g in ("attention", "feedforward", "text_refine") for p in ADAPTER_LAYER_GROUPS[g]], None
    )
    assert sorted(union) == sorted(names)
    cap = get_capability("nanosaur2")
    assert sorted(cap.adapter_layer_groups) == sorted(ADAPTER_LAYER_GROUPS)
    assert sorted(cap.adapter_module_roots) == sorted(n for n, _ in model.named_children())


def _export(p, tmp_path):
    for name, param in p.transformer.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                param.add_(torch.randn_like(param) * 0.01)
    state = {
        param.original_name.replace(".default", ""): param.detach().clone()
        for param in p.transformer.parameters()
        if param.requires_grad
    }
    p.save_adapter(tmp_path, state)
    return load_file(str(tmp_path / "adapter_model.safetensors"))


def test_lora_export_has_exactly_the_comfyui_keys_of_train_lora(tiny_files, tmp_path):
    p = make_pipeline(tiny_files, adapter={"type": "lora", "rank": 4, "alpha": 4, "dtype": torch.float32})
    p.configure_adapter(p.config["adapter"])
    exported = _export(p, tmp_path)
    expected = set()
    for name in _reference_lora_names(tiny_dit()):  # what train_lora.py's save() writes
        expected |= {f"diffusion_model.{name}.lora_A.weight", f"diffusion_model.{name}.lora_B.weight", f"diffusion_model.{name}.alpha"}
    assert set(exported) == expected
    assert exported["diffusion_model.blocks.0.attn.kv_y.alpha"].item() == 4.0
    assert exported["diffusion_model.blocks.0.attn.kv_y.lora_A.weight"].shape == (4, 32)
    assert exported["diffusion_model.blocks.0.attn.kv_y.lora_B.weight"].shape == (64, 4)
    # And it loads back into a fresh adapter (alpha scalars are export-only).
    q = make_pipeline(tiny_files, adapter={"type": "lora", "rank": 4, "alpha": 4, "dtype": torch.float32})
    q.configure_adapter(q.config["adapter"])
    q.load_adapter_weights(tmp_path)
    lora_a = dict(q.transformer.named_parameters())["blocks.0.attn.kv_y.lora_A.default.weight"]
    assert torch.equal(lora_a, exported["diffusion_model.blocks.0.attn.kv_y.lora_A.weight"])


def test_loading_a_lora_trained_with_another_alpha_warns(tiny_files, tmp_path):
    p = make_pipeline(tiny_files, adapter={"type": "lora", "rank": 4, "alpha": 4, "dtype": torch.float32})
    p.configure_adapter(p.config["adapter"])
    _export(p, tmp_path)
    q = make_pipeline(tiny_files, adapter={"type": "lora", "rank": 4, "alpha": 8, "dtype": torch.float32})
    q.configure_adapter(q.config["adapter"])
    with pytest.warns(UserWarning, match="differs from this run's adapter.alpha 8.0"):
        q.load_adapter_weights(tmp_path)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p.load_adapter_weights(tmp_path)  # same alpha: silent


def test_full_finetune_export_is_the_release_layout(tiny_files, tmp_path):
    p = make_pipeline(tiny_files)
    state = {param.original_name: param.detach() for param in p.transformer.parameters()}
    p.save_model(tmp_path, state)
    out = load_file(str(tmp_path / "nanosaur2_diffusion_model.safetensors"))
    assert set(out) == set(load_file(tiny_files["transformer_path"]))
    reloaded = loading.load_transformer(tmp_path / "nanosaur2_diffusion_model.safetensors", torch.float32)
    x, t, context, weights = _inputs()
    assert torch.equal(reloaded(x, t, context, weights), p.transformer(x, t, context, weights))


# ---- previews -------------------------------------------------------------------------------


def test_flow_sigmas_are_the_shift3_simple_schedule():
    sigmas = preview_sampling.flow_sigmas(30)
    assert sigmas.shape == (31,) and sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert (sigmas[:-1] > sigmas[1:]).all()
    # simple: every (1000/30)-th entry of the shift-3 table, from the top.
    t = 1.0 - int(1 * 1000 / 30) / 1000
    assert sigmas[1].item() == pytest.approx(3 * t / (1 + 2 * t), rel=1e-6)


@pytest.mark.parametrize(
    "mode,expected", [("alternate", [False, True, False, True]), ("cfg", [False] * 4), ("path_drop", [True] * 4)]
)
def test_guidance_modes(mode, expected):
    assert [preview_sampling.step_uses_path_drop(mode, i) for i in range(4)] == expected
    with pytest.raises(ValueError):
        preview_sampling.step_uses_path_drop("bogus", 0)


@pytest.mark.parametrize("path_drop", [False, True])
def test_guided_velocity_runs_the_uncond_row_without_the_sparse_path_on_path_drop_steps(path_drop):
    model = tiny_dit()
    x, t, context, weights = _inputs(b=2, lengths=(5, 2))
    sigma = torch.tensor([0.7])
    v = preview_sampling.guided_velocity(model, x[:1], sigma, context, weights, 4.0, path_drop)
    cond = model(x[:1], sigma, context[:1], weights[:1])
    uncond = model(x[:1], sigma, context[1:], weights[1:], sparse_skip=[path_drop])
    assert torch.allclose(v, uncond + 4.0 * (cond - uncond), atol=1e-4)


@pytest.mark.parametrize("mode", ["alternate", "cfg", "path_drop"])
def test_generate_preview_image_end_to_end(tiny_files, mode, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU-only test
    p = make_pipeline(tiny_files)
    cfg = {"width": 64, "height": 48, "num_inference_steps": 4, "guidance_scale": 3.0, "guidance_mode": mode,
           "negative_prompt": "oldest, low quality"}
    p.prepare_preview_memory(cfg)
    image = p.generate_preview_image(cfg, "a red fox", step=1, seed=3)
    assert image.size == (64, 48) and image.mode == "RGB"
    # Memoized: the encoder is not needed again for the same prompts.
    assert set(p._preview_embed_cache) == {"a red fox", "oldest, low quality"}
    p.restore_after_preview()


# ---- dataset -> model seam (no doubles) -----------------------------------------------------


def _seam_tree(root: Path) -> None:
    """Square, landscape (2:1) and portrait (1:2) images with captions of different lengths,
    including emphasis and an empty caption."""
    from PIL import Image

    captions = {
        "sq_a": "a red fox sits on the mossy rock at dusk",
        "sq_b": "girl (blue ears:1.4) tail",
        "wide_a": "forest sunlight",
        "wide_b": "a cat and a dog under the blue sky at dusk",
        "tall_a": "",
        "tall_b": "(red:0.5) fox",
    }
    sizes = {"sq": (64, 64), "wide": (128, 64), "tall": (64, 128)}
    root.mkdir(parents=True, exist_ok=True)
    for i, (stem, caption) in enumerate(captions.items()):
        g = torch.Generator().manual_seed(i)
        w, h = sizes[stem.split("_")[0]]
        Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()).save(root / f"{stem}.png")
        (root / f"{stem}.txt").write_text(caption, encoding="utf-8")


@pytest.mark.parametrize("accumulation", [1, 2])
def test_dataset_batches_run_through_the_layers(tmp_path, tiny_files, accumulation):
    """Real images -> real Dataset/DatasetManager cache (real VAE + Gemma3 fns, ragged
    token_weights, caption dropout through uncond_fraction) -> prepare_inputs with mixed
    path-drop -> the loader's micro-batch split + trim -> every pipeline layer -> loss and
    gradients. Buckets are square and non-square."""
    from rengu_flow.data.dataset import Dataset
    from rengu_flow.data.loader import split_batch
    from rengu_flow.data.manager import DatasetManager
    from rengu_flow.engine import select_backend

    import random

    _seam_tree(tmp_path / "images")
    # Path-drop exactly on the caption-dropped rows (BOS-only text): the "tall" bucket pairs the
    # empty caption with a real one, so its batches mix skipping and non-skipping rows.
    p = make_pipeline(tiny_files, sparse_skip_prob=0.0, sparse_skip_prob_uncond=1.0)
    p.config.update({"pipeline_stages": 1})
    cfg = {
        "resolutions": [64],
        "enable_ar_bucket": True,
        "min_ar": 0.5,
        "max_ar": 2.0,
        "num_ar_buckets": 3,
        "uncond_fraction": 0.25,
        "directory": [{"path": str(tmp_path / "images"), "num_repeats": 2}],
    }
    ds = Dataset(cfg, p, training_config={"cache_root": str(tmp_path / "cache")})
    manager = DatasetManager(p, backend=select_backend({"engine": "accelerate"}))
    manager.register(ds)
    manager.cache(unload_models=False)
    ds.post_init(0, 1, {None: 2}, 1, {None: 2})

    layers = p.to_layers()
    loss_fn = p.get_loss_fn()
    shapes, lengths_seen, skips_seen = set(), set(), set()
    random.seed(0)  # the dataset's caption dropout
    torch.manual_seed(0)
    for i in range(len(ds)):
        batch = ds[i]
        shapes.add(tuple(batch["latents"].shape[-2:]))
        features, label = p.prepare_inputs(batch)
        lengths = (features[3] > 0).sum(dim=1)
        lengths_seen |= {int(n) for n in lengths}
        assert torch.equal(features[4], lengths == 1)  # path-drop follows caption dropout
        skips_seen.add(tuple(features[4].tolist()))
        for micro_features, micro_label in split_batch((features, label), accumulation):
            micro_features = p.trim_micro_batch(micro_features)
            out = _run_layers(layers, micro_features)
            assert out.shape == micro_label[0].shape and torch.isfinite(out).all()
            loss = loss_fn(out, micro_label)
            loss.backward()
            assert torch.isfinite(loss)
    assert {(4, 4), (3, 6), (6, 3)} <= shapes, shapes  # square + both orientations, latent grid
    assert 1 in lengths_seen and max(lengths_seen) > 3  # dropped (BOS-only) and real captions
    assert any(any(s) and not all(s) for s in skips_seen), skips_seen  # mixed path-drop in a batch
    assert any(prm.grad is not None for prm in p.transformer.blocks[3].parameters())
