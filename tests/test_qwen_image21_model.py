"""CPU-only unit tests for the Qwen-Image 2.1 training pipeline: pipeline-layer parity with the
monolithic transformer, padding invariance, training targets/shift, adapter attach + export keys,
text conditioning parity against a literal copy of the reference pipeline's encode (tiny
Qwen3-VL), the preview sampler (scheduler parity with diffusers, Euler direction, KV-cache
parity), checkpoint loaders, and the lazy/streamed text-encoder wrapper."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model.dit_common.streaming import LazyStreamedEncoder
from rengu_flow.model.qwen_image21 import loading, preview_sampling
from rengu_flow.model.qwen_image21.dit import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
    build_t2i_img_mask,
    pack_latents,
    t2i_img_shapes,
    unpack_latents,
)
from rengu_flow.model.qwen_image21.layers import FinalLayer, InitialLayer, TransformerLayer
from rengu_flow.model.qwen_image21.pipeline import (
    ADAPTER_LAYER_GROUPS,
    ADAPTER_TARGET_MODULES,
    EXPORT_PREFIX,
    QUANT_LEAF_NAMES,
    QUANT_SKIP_SUBSTRINGS,
    QwenImage21Pipeline,
    calculate_shift,
)
from rengu_flow.model.qwen_image21.text import (
    PROMPT_TEMPLATE_T2I,
    SYSTEM_PROMPT,
    drop_index,
    encode_prompts,
)
from rengu_flow.networks import adapter_dit
from rengu_flow.training.block_swap import NoopOffloader

pytestmark = pytest.mark.no_ui_db

IN_CH = 8
CTX = 12
GRID = (4, 6)  # latent grid (h, w)


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


def _batch(text_lens=(5, 3), seed: int = 1):
    """Right-padded text batch: sample i has text_lens[i] valid tokens."""
    g = torch.Generator().manual_seed(seed)
    b = len(text_lens)
    latents = torch.randn(b, IN_CH, *GRID, generator=g)
    text = torch.randn(b, max(text_lens), CTX, generator=g)
    mask = torch.zeros(b, max(text_lens), dtype=torch.bool)
    for i, n in enumerate(text_lens):
        mask[i, :n] = True
        text[i, n:] = 0
    t = torch.rand(b, generator=g)
    return latents, text, mask, t


def _monolithic(model, latents, text, mask, t):
    b, _, h, w = latents.shape
    out = model(
        hidden_states=pack_latents(latents),
        encoder_hidden_states=text,
        timestep=t,
        img_shapes=t2i_img_shapes(h, w, b),
        img_mask=build_t2i_img_mask(text.shape[1], h, w, b),
        encoder_hidden_states_mask=None if mask.all() else mask,
    )
    return unpack_latents(out[:, -h * w :], h, w)


def _layers(model):
    return (
        [InitialLayer(model)]
        + [TransformerLayer(b, i, NoopOffloader()) for i, b in enumerate(model.transformer_blocks)]
        + [FinalLayer(model)]
    )


def _run_layers(model, latents, text, mask, t):
    x = (latents, t.view(-1, 1), text, mask)
    for layer in _layers(model):
        x = layer(x)
    return x


# ---- pipeline layers --------------------------------------------------------------------------


@pytest.mark.parametrize("text_lens", [(5, 3), (4, 4)], ids=["padded", "all_valid"])
def test_pipeline_layers_match_monolithic_forward(tiny_model, text_lens):
    latents, text, mask, t = _batch(text_lens)
    with torch.no_grad():
        expected = _monolithic(tiny_model, latents, text, mask, t)
        actual = _run_layers(tiny_model, latents, text, mask, t)
    assert actual.shape == latents.shape
    assert torch.allclose(actual, expected, atol=1e-5)


def test_layer_tuple_is_tensors_only_with_shape_metadata(tiny_model):
    latents, text, mask, t = _batch((5, 3))
    outputs = InitialLayer(tiny_model)((latents, t.view(-1, 1), text, mask))
    assert all(torch.is_tensor(o) for o in outputs)
    layout = outputs[5]
    assert layout.shape == (5, *GRID, 0) and layout.numel() == 0
    assert outputs[3].dtype == torch.float32 and outputs[3].shape[-1] == 2  # view_as_real RoPE
    assert outputs[4].shape == (2, 5 + GRID[0] * GRID[1])  # key_valid for the padded sample
    all_valid = InitialLayer(tiny_model)((latents, t.view(-1, 1), text[:, :3], mask[:, :3]))
    assert all_valid[4].numel() == 0  # 0-size sentinel = no padding


def test_padding_is_invisible(tiny_model):
    """Garbage in padded text slots changes nothing, and a padded sample equals itself unpadded
    (RoPE positions skip padding)."""
    latents, text, mask, t = _batch((5, 3))
    with torch.no_grad():
        out = _run_layers(tiny_model, latents, text, mask, t)
        dirty = text.clone()
        dirty[1, 3:] = 123.0
        out_dirty = _run_layers(tiny_model, latents, dirty, mask, t)
        alone = _run_layers(tiny_model, latents[1:], text[1:, :3], mask[1:, :3], t[1:])
    assert torch.allclose(out, out_dirty, atol=1e-6)
    assert torch.allclose(out[1:], alone, atol=1e-5)


def test_gradients_reach_every_parameter_through_the_layers(tiny_model):
    model = tiny_model.train()
    latents, text, mask, t = _batch((5, 3))
    _run_layers(model, latents, text, mask, t).square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum()]
    assert not missing, missing


# ---- training inputs ------------------------------------------------------------------------


def _stub_pipeline(**model_extra) -> QwenImage21Pipeline:
    p = object.__new__(QwenImage21Pipeline)
    p.config = {"model": {"dtype": torch.float32, **model_extra}}
    p.model_config = p.config["model"]
    return p


def test_prepare_inputs_targets_velocity_with_dynamic_shift():
    p = _stub_pipeline()
    latents = torch.randn(2, IN_CH, *GRID)
    embeds = [torch.randn(5, CTX), torch.randn(3, CTX)]
    masks = [torch.ones(5, dtype=torch.bool), torch.ones(3, dtype=torch.bool)]
    torch.manual_seed(0)
    (noisy, t, prompt_embeds, text_mask), (target, mask) = p.prepare_inputs(
        {"latents": latents, "mask": None, "prompt_embeds": embeds, "text_mask": masks}
    )
    assert prompt_embeds.shape == (2, 5, CTX) and text_mask.tolist()[1] == [True] * 3 + [False] * 2
    t4 = t.view(-1, 1, 1, 1)
    noise = target + latents  # target = noise - latents
    assert torch.allclose(noisy, (1 - t4) * latents + t4 * noise, atol=1e-6)
    assert mask is None

    # Same draw, fixed quantile: the timestep is the exponential shift at mu(h*w).
    q = 0.3
    (_, tq, _, _), _ = p.prepare_inputs(
        {"latents": latents, "mask": None, "prompt_embeds": embeds, "text_mask": masks}, timestep_quantile=q
    )
    raw = torch.sigmoid(torch.distributions.Normal(0, 1).icdf(torch.tensor(q)))
    mu = calculate_shift(GRID[0] * GRID[1])
    expected = math.exp(mu) / (math.exp(mu) + (1 / raw - 1))
    assert torch.allclose(tq.view(-1), expected.expand(2).float(), atol=1e-6)


def test_calculate_shift_matches_reference_scheduler_constants():
    assert calculate_shift(256) == pytest.approx(0.5)
    assert calculate_shift(8192) == pytest.approx(0.9)
    assert calculate_shift(4096) == pytest.approx(0.5 + 0.4 * (4096 - 256) / (8192 - 256))


def test_vae_fn_adds_opaque_alpha_and_normalizes():
    seen = {}

    class _Dist:
        def __init__(self, x):
            self.x = x

        def sample(self):
            return self.x

    class _Vae(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(latents_mean=[1.0] * 64, latents_std=[2.0] * 64)

        def encode(self, x):
            seen["x"] = x
            b, _, _, h, w = x.shape
            return SimpleNamespace(latent_dist=_Dist(torch.full((b, 64, 1, h // 16, w // 16), 5.0)))

    p = _stub_pipeline()
    p.vae = _Vae()
    out = p.get_call_vae_fn(p.vae)(torch.zeros(2, 3, 32, 64))["latents"]
    assert seen["x"].shape == (2, 4, 1, 32, 64)
    assert torch.all(seen["x"][:, 3] == 1.0)  # alpha = opaque in [-1, 1]
    assert out.shape == (2, 64, 2, 4) and torch.all(out == 2.0)  # (5 - 1) / 2


def test_requires_cached_text_embeddings():
    p = object.__new__(QwenImage21Pipeline)
    with pytest.raises(ConfigValidationError, match="cache_text_embeddings"):
        QwenImage21Pipeline.__init__(p, {"model": {"dtype": torch.float32, "cache_text_embeddings": False}})


def test_component_paths_resolve_from_diffusers_path_with_overrides(tmp_path):
    (tmp_path / "processor").mkdir()
    p = _stub_pipeline(diffusers_path=str(tmp_path), vae_path="custom/vae")
    assert p._component_path("transformer") == str(tmp_path / "transformer")
    assert p._component_path("text_encoder") == str(tmp_path / "text_encoder")
    assert p._component_path("vae") == "custom/vae"
    assert p._processor_path() == str(tmp_path / "processor")
    assert _stub_pipeline(diffusers_path=str(tmp_path / "nope"))._processor_path() is None
    with pytest.raises(ConfigValidationError, match="diffusers_path"):
        _stub_pipeline()._component_path("vae")


# ---- adapters ---------------------------------------------------------------------------------

ADAPTER_CONFIGS = [
    pytest.param({"type": "lora", "rank": 4, "alpha": 4, "dropout": 0.0, "dtype": torch.float32}, id="lora"),
    pytest.param(
        {
            "type": "lokr",
            "rank": 4,
            "alpha": 4,
            "factor": -1,
            "decompose_both": False,
            "full_matrix": False,
            "dtype": torch.float32,
        },
        id="lokr",
    ),
    pytest.param(
        {
            "type": "lycoris_locon",
            "rank": 4,
            "alpha": 4,
            "dropout": 0.0,
            "rank_dropout": 0.0,
            "module_dropout": 0.0,
            "train_norm": False,
            "train_conv": False,
            "use_tucker": False,
            "use_scalar": False,
            "dora_wd": False,
            "rs_lora": False,
            "wd_on_output": True,
            "dtype": torch.float32,
        },
        id="lycoris_locon",
    ),
]


@pytest.mark.parametrize("adapter_cfg", ADAPTER_CONFIGS)
def test_adapter_attach_trains_through_layers(tiny_model, adapter_cfg):
    if adapter_cfg["type"] == "lycoris_locon":
        pytest.importorskip("lycoris")
    model = copy.deepcopy(tiny_model).train()
    for name, p in model.named_parameters():
        p.original_name = name
        p.requires_grad_(False)
    adapter_dit.configure(model, adapter_cfg, targets=ADAPTER_TARGET_MODULES, layer_groups=ADAPTER_LAYER_GROUPS)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("img_mlp" in n for n in trainable) and any(".attn." in n for n in trainable)
    assert any(n.startswith(("modulation", "base_model.model.modulation")) or ".modulation." in n for n in trainable)

    latents, text, mask, t = _batch((5, 3))
    _run_layers(model, latents, text, mask, t).square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)


def test_lora_export_uses_comfy_mappable_transformer_keys(tiny_model, tmp_path):
    """ComfyUI maps 2.1 LoRAs as ``transformer.<diffusers module>`` (the gate_layer / proj halves
    of its fused gate_up included); the export must use that prefix and those module names."""
    from safetensors.torch import load_file

    model = copy.deepcopy(tiny_model)
    for name, p in model.named_parameters():
        p.original_name = name
    cfg = {"type": "lora", "rank": 4, "alpha": 4, "dropout": 0.0, "dtype": torch.float32}
    peft_config, _ = adapter_dit.configure(model, cfg, targets=ADAPTER_TARGET_MODULES)
    state = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    adapter_dit.save(tmp_path, state, cfg, peft_config, export_prefix=EXPORT_PREFIX)
    keys = list(load_file(tmp_path / "adapter_model.safetensors"))
    assert keys and all(k.startswith("transformer.") for k in keys)
    assert any(".img_mlp.gate_layer." in k for k in keys) and any(".img_mlp.proj." in k for k in keys)
    assert any(k.startswith("transformer.modulation.1.") for k in keys)


def test_quant_scope_selects_only_block_linears(tiny_model):
    from rengu_flow.training.quantize_dit import _iter_quant_targets

    names = [n for _, _, n, _ in _iter_quant_targets(tiny_model, QUANT_LEAF_NAMES, QUANT_SKIP_SUBSTRINGS)]
    per_block = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"}
    assert sorted(names) == sorted(f"transformer_blocks.{i}.{leaf}" for i in range(2) for leaf in per_block)


# ---- text conditioning -----------------------------------------------------------------------


def _bundled_tokenizer():
    return loading.load_tokenizer(None)


def _tiny_qwen3vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=64,
            vocab_size=151936,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            rope_theta=5000000,
        ),
        vision_config=dict(depth=1, hidden_size=16, num_heads=2, intermediate_size=32, out_hidden_size=32, deepstack_visual_indexes=[0]),
    )
    return Qwen3VLForConditionalGeneration(config).eval()


def _reference_prompt_embeds(text_encoder, processor, prompt):
    """Literal copy of the t2i path of ``QwenImage21Pipeline._get_qwen_prompt_embeds``
    (diffusers main 6256aa7) and its ``__init__``-derived ``_drop_idx``."""
    sys_prompt = "Comprehend and analyze the provided prompt."
    prompt_template_t2i = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    sys_message = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
    sys_tokens = processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
    _drop_idx = len(sys_tokens[0])

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [" " if not p else p for p in prompt]
    prompts = [prompt_template_t2i.format(t) for t in prompt]
    processor_kwargs = {"text": prompts, "padding": True, "padding_side": "left", "return_tensors": "pt"}
    model_inputs = processor(**processor_kwargs)
    forward_kwargs = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "output_hidden_states": True,
    }
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
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max(e.size(0) for e in split_hidden_states)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    return prompt_embeds, encoder_attention_mask, _drop_idx


def _processor(tokenizer):
    from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=tokenizer.chat_template,
    )


PROMPTS = [
    "a cat",
    "",
    "Un gato con sombrero, 日本語テキスト \"quotes\"  double  spaces\n\nnewlines",
    "A capybara wearing a wizard hat, reading a book by candlelight, oil painting",
]


def test_encode_prompts_matches_reference_pipeline():
    tokenizer = _bundled_tokenizer()
    encoder = _tiny_qwen3vl()
    expected, expected_mask, ref_drop_idx = _reference_prompt_embeds(encoder, _processor(tokenizer), PROMPTS)
    assert drop_index(tokenizer) == ref_drop_idx == 14

    # rengu loads only the text decoder; the batch is left-padded exactly like the reference.
    actual, mask = encode_prompts(encoder.model.language_model, tokenizer, PROMPTS, device="cpu")
    assert torch.equal(mask, expected_mask.bool())
    assert torch.equal(actual, expected)
    # One prompt at a time (the caching path with caching_batch_size = 1) agrees too.
    for i, prompt in enumerate(PROMPTS):
        single, single_mask = encode_prompts(encoder.model.language_model, tokenizer, [prompt], device="cpu")
        n = int(single_mask.sum())
        assert torch.allclose(single[0, :n], expected[i, :n], atol=1e-5)


def test_encode_prompts_accepts_wrapped_encoders():
    tokenizer = _bundled_tokenizer()
    encoder = _tiny_qwen3vl()
    base, _ = encode_prompts(encoder.model.language_model, tokenizer, PROMPTS[:2], device="cpu")
    full, _ = encode_prompts(encoder, tokenizer, PROMPTS[:2], device="cpu")
    lazy = LazyStreamedEncoder(lambda: encoder.model.language_model, layers_of=lambda m: m.layers)
    wrapped, _ = encode_prompts(lazy, tokenizer, PROMPTS[:2], device="cpu")
    assert torch.equal(base, full) and torch.equal(base, wrapped)


def test_template_and_drop_index_constants():
    assert SYSTEM_PROMPT in PROMPT_TEMPLATE_T2I
    assert PROMPT_TEMPLATE_T2I.format("x").endswith("<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n")


# ---- preview sampling ------------------------------------------------------------------------


@pytest.mark.parametrize("steps,seq_len", [(40, 4096), (28, 4096), (8, 1024)])
def test_sigmas_match_reference_scheduler(steps, seq_len):
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=8192,
        max_shift=0.9,
        num_train_timesteps=1000,
        shift=1.0,
        shift_terminal=0.02,
        time_shift_type="exponential",
        use_dynamic_shifting=True,
    )
    mu = calculate_shift(seq_len)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    ours = preview_sampling.shifted_sigmas(steps, seq_len)
    assert ours.shape == (steps + 1,)
    assert torch.allclose(ours, scheduler.sigmas.float(), atol=1e-6)
    assert ours[-2].item() == pytest.approx(0.02, abs=1e-6) and ours[-1].item() == 0.0


def test_euler_direction_recovers_data_with_the_training_target():
    """With the exact training target v = noise - x0 as the velocity, the preview Euler update
    x += (sigma_next - sigma) * v lands on x0: the sampler and the loss agree on the sign."""
    x0, noise = torch.randn(3, 5), torch.randn(3, 5)
    sigmas = preview_sampling.shifted_sigmas(10, 4096)
    x = noise.clone()  # sigma_0 = 1
    for i in range(10):
        x = x + (sigmas[i + 1] - sigmas[i]) * (noise - x0)
    assert torch.allclose(x, x0, atol=1e-5)


def _preview_stub(model):
    return SimpleNamespace(transformer=model, _preview_offloader=None)


def test_denoise_step_matches_forward_and_kv_cache(tiny_model):
    latents, text, mask, t = _batch((5, 3))
    b, _, h, w = latents.shape
    packed = pack_latents(latents)
    args = (text, t, t2i_img_shapes(h, w, b), build_t2i_img_mask(5, h, w, b), mask)
    with torch.no_grad():
        expected = tiny_model(packed, text, t, args[2], args[3], mask)[:, -h * w :]
        plain = preview_sampling.denoise_step(_preview_stub(tiny_model), packed, *args)
        cache = QwenImage21KVCache(2)
        first = preview_sampling.denoise_step(_preview_stub(tiny_model), packed, *args, cache, "extract")
        other = packed + 0.1
        cached = preview_sampling.denoise_step(_preview_stub(tiny_model), other, *args[:1], t * 0.5, *args[2:], cache, "cached")
        fresh = tiny_model(other, text, t * 0.5, args[2], args[3], mask)[:, -h * w :]
    assert torch.allclose(plain, expected, atol=1e-5)
    assert torch.allclose(first, expected, atol=1e-5)
    assert torch.allclose(cached, fresh, atol=1e-5)


class _PreviewPipe(QwenImage21Pipeline):
    """Tiny end-to-end preview harness: stub VAE + tiny DiT + stub prompt embeddings."""

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
                self.tiling_calls = []

            @property
            def dtype(self):
                return self.w.dtype

            def enable_tiling(self, *args):
                self.use_tiling = True
                self.tiling_calls.append(args)

            def decode(self, z):
                b, _, _, h, w = z.shape
                img = torch.tanh(z[:, :4].repeat_interleave(16, -1).repeat_interleave(16, -2))
                return SimpleNamespace(sample=img)

        self.vae = _Vae()

    def ensure_vae_for_preview(self):
        pass

    def preview_prompt_embeds(self, prompts, preview_cfg, device):
        self.encoded.append(list(prompts))
        g = torch.Generator().manual_seed(len(prompts))
        return [(torch.randn(3 + i, CTX, generator=g), 3 + i) for i in range(len(prompts))]


@pytest.mark.parametrize("guidance", [1.0, 3.0], ids=["no_cfg", "cfg"])
def test_generate_preview_image_end_to_end(tiny_model, guidance):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # the sampler runs on CUDA when present
    pipe = _PreviewPipe(tiny_model.to(device))
    cfg = {"width": 96, "height": 64, "num_inference_steps": 3, "guidance_scale": guidance, "negative_prompt": ""}
    image = preview_sampling.generate_preview_image(pipe, cfg, "a cat", step=0, seed=1)
    assert image.size == (96, 64) and image.mode == "RGB"
    assert pipe.encoded == [["a cat", ""] if guidance > 1 else ["a cat"]]
    # The decode is tiled (bounded VRAM) and the VAE's own tiling flag is restored afterwards.
    assert pipe.vae.tiling_calls == [(512, 512, 448, 448)] and pipe.vae.use_tiling is False


def test_preview_prompt_embeds_are_memoized():
    calls = []
    p = _stub_pipeline()
    p._preview_embed_cache = {}
    p.tokenizer = _bundled_tokenizer()
    p.drop_idx = 14
    encoder = _tiny_qwen3vl().model.language_model
    p.text_encoder = LazyStreamedEncoder(lambda: (calls.append(1), encoder)[1], layers_of=lambda m: m.layers)
    p.ensure_text_encoder_for_preview = lambda device: None
    p.offload_text_encoder_after_encode = lambda cfg: None
    first = p.preview_prompt_embeds(["a cat", ""], {}, "cpu")
    again = p.preview_prompt_embeds(["", "a cat"], {}, "cpu")
    assert calls == [1]
    assert torch.equal(first[0][0], again[1][0]) and first[1][1] == again[0][1]
    ref, mask = encode_prompts(encoder, p.tokenizer, ["a cat"], device="cpu")
    assert torch.equal(first[0][0], ref[0][mask[0]])


# ---- loaders ----------------------------------------------------------------------------------


def test_split_fused_mlp_matches_comfy_gate_up_layout(tiny_model):
    """ComfyUI fuses gate_layer (first half) and proj (second half) into img_mlp.gate_up."""
    sd = tiny_model.state_dict()
    comfy = {}
    for k, v in sd.items():
        if ".img_mlp.gate_layer." in k:
            comfy[k.replace(".gate_layer.", ".gate_up.")] = torch.cat([v, sd[k.replace(".gate_layer.", ".proj.")]])
        elif ".img_mlp.proj." not in k:
            comfy[k] = v
    rebuilt = tiny_transformer(seed=5)
    rebuilt.load_state_dict(loading.split_fused_mlp(comfy), strict=True)
    latents, text, mask, t = _batch((5, 3))
    with torch.no_grad():
        assert torch.equal(_monolithic(rebuilt, latents, text, mask, t), _monolithic(tiny_model, latents, text, mask, t))


def test_prequantized_dit_is_refused():
    with pytest.raises(ConfigValidationError, match="pre-quantized"):
        loading._guard_not_prequantized({"transformer_blocks.0.attn.to_q.weight": torch.zeros(2, dtype=torch.int8)}, "transformer_path")
    with pytest.raises(ConfigValidationError, match="pre-quantized"):
        loading._guard_not_prequantized({"a.weight": torch.zeros(1), "a.comfy_quant": torch.zeros(1)}, "transformer_path")


def test_comfy_layout_vae_file_is_refused_with_guidance(tmp_path):
    from safetensors.torch import save_file

    f = tmp_path / "qwen_image_2.1_vae_bf16.safetensors"
    save_file({"encoder.conv1.weight": torch.zeros(1)}, str(f))
    with pytest.raises(ConfigValidationError, match="diffusers vae/ folder"):
        loading.load_vae(f, torch.float32)


def test_missing_component_path_is_a_config_error(tmp_path):
    with pytest.raises(ConfigValidationError, match="does not exist"):
        loading.load_vae(tmp_path / "missing", torch.float32)


def test_bundled_configs_match_the_official_release():
    import json

    tcfg = json.loads(loading.TRANSFORMER_CONFIG_PATH.read_text())
    assert tcfg["num_layers"] == 32 and tcfg["in_channels"] == 64 and tcfg["causal_condition"] is True
    vcfg = json.loads(loading.VAE_CONFIG_PATH.read_text())
    assert vcfg["z_dim"] == 64 and vcfg["in_channels"] == 4 and len(vcfg["latents_mean"]) == 64
    from transformers import AutoConfig

    text = AutoConfig.from_pretrained(loading.QWEN3VL_8B_ASSETS).text_config
    assert text.hidden_size == 4096 and text.num_hidden_layers == 36


def test_text_encoder_loads_text_decoder_from_sharded_folder_and_single_file(tmp_path):
    """Folder (transformers layout, sharded + index, vision/LM-head keys present) and a
    ComfyUI-style single file (``model.layers...``) both load just the text decoder."""
    import json

    from safetensors.torch import save_file

    from rengu_flow.model.dit_common.qwen3vl import checkpoint_files, load_qwen3vl_text_model

    full = _tiny_qwen3vl()
    sd = {k: v.contiguous() for k, v in full.state_dict().items()}
    keys = sorted(sd)
    half = len(keys) // 2
    folder = tmp_path / "text_encoder"
    folder.mkdir()
    shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
    for name, ks in shards.items():
        save_file({k: sd[k] for k in ks}, str(folder / name))
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: n for n, ks in shards.items() for k in ks}})
    )
    files = checkpoint_files(folder)
    assert [f.name for f in files] == sorted(shards)

    text_config = full.config.text_config
    loaded = load_qwen3vl_text_model(files, text_config, torch.float32)
    comfy = {k.replace("model.language_model.", "model."): v for k, v in sd.items()}
    single = tmp_path / "qwen3vl_tiny.safetensors"
    save_file(comfy, str(single))
    loaded_single = load_qwen3vl_text_model(checkpoint_files(single), text_config, torch.float32)

    tokenizer = _bundled_tokenizer()
    ref, _ = encode_prompts(full.model.language_model, tokenizer, PROMPTS[:2], device="cpu")
    for model in (loaded, loaded_single):
        out, _ = encode_prompts(model, tokenizer, PROMPTS[:2], device="cpu")
        assert torch.equal(out, ref)
        assert not any(p.requires_grad for p in model.parameters())


def test_qwen3vl_remap_dequantizes_scaled_fp8_and_strips_prefixes():
    from rengu_flow.model.dit_common.qwen3vl import remap_qwen3vl_text_state_dict

    w = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
    sd = {
        "model.layers.0.mlp.down_proj.weight": w.to(torch.float8_e4m3fn),
        "model.layers.0.mlp.down_proj.weight_scale": torch.tensor(2.0),
        "model.layers.0.mlp.down_proj.comfy_quant": torch.zeros(1, dtype=torch.uint8),
        "model.language_model.norm.weight": torch.ones(2),
        "model.visual.patch_embed.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
    }
    out = remap_qwen3vl_text_state_dict(sd, torch.bfloat16)
    assert set(out) == {"layers.0.mlp.down_proj.weight", "norm.weight"}
    assert torch.allclose(out["layers.0.mlp.down_proj.weight"].float(), w * 2.0, atol=0.1)


# ---- lazy / streamed text encoder ------------------------------------------------------------


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Linear(4, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.register_buffer("scale", torch.full((8,), 0.5))

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return h * self.scale


def test_lazy_encoder_loads_on_demand_and_unloads():
    calls = []
    enc = LazyStreamedEncoder(lambda: (calls.append(1), _TinyEncoder())[1], layers_of=lambda m: m.layers)
    assert not enc.is_loaded and next(enc.parameters()).device.type == "meta"
    enc.to("cpu")  # the data layer parks idle submodels on CPU: must not trigger a load
    assert not enc.is_loaded and calls == []
    out = enc(torch.ones(1, 4))
    assert enc.is_loaded and calls == [1] and out.shape == (1, 8)
    assert next(enc.parameters()).device.type == "cpu"
    enc.to("meta")
    assert not enc.is_loaded and next(enc.parameters()).device.type == "meta"
    with pytest.raises(ValueError):
        LazyStreamedEncoder(_TinyEncoder, layers_of=lambda m: m.layers, offload="sometimes")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_encoder_matches_resident_and_keeps_layers_on_host():
    x = torch.randn(2, 4, device="cuda")
    resident = _TinyEncoder().cuda()(x)
    enc = LazyStreamedEncoder(_TinyEncoder, layers_of=lambda m: m.layers, offload="stream")
    enc.to("cuda")
    assert enc.is_streaming and next(enc.parameters()).device.type == "cuda"  # embed resident
    assert all(p.device.type == "cpu" for p in enc.module.layers.parameters())
    assert torch.allclose(enc(x), resident, atol=1e-6)
    assert all(p.device.type == "cpu" for p in enc.module.layers.parameters())  # dropped again
    enc.to("cpu")
    assert not enc.is_streaming and all(p.device.type == "cpu" for p in enc.module.parameters())
    enc.offload = "none"
    enc.to("cuda")
    assert not enc.is_streaming and all(p.device.type == "cuda" for p in enc.module.parameters())
    assert torch.allclose(enc(x), resident, atol=1e-6)
