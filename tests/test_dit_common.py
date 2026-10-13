"""rengu_flow.model.dit_common: the flow-matching math and DiTPipeline shared by DiT models.

The reference implementations below are the pre-extraction bodies of krea2/cosmos
``prepare_inputs`` — the shared helpers must reproduce them bit for bit under the same seed.
"""

from __future__ import annotations

import math

import pytest
import torch

from rengu_flow.model.dit_common import (
    DiTPipeline,
    add_flow_noise,
    calculate_shift,
    preview_compute_dtype,
    sample_timesteps,
    shift_timesteps,
    time_shift,
)


def _ref_sample(model_config, bs, device, timestep_quantile):
    method = model_config.get("timestep_sample_method", "logit_normal")
    if method == "logit_normal":
        dist = torch.distributions.normal.Normal(0, 1)
    elif method == "uniform":
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise NotImplementedError()
    if timestep_quantile is not None:
        t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
    else:
        t = dist.sample((bs,)).to(device)
    if method == "logit_normal":
        t = torch.sigmoid(t * model_config.get("sigmoid_scale", 1.0))
    return t


def _ref_krea2(model_config, latents, quantile):
    bs, _, h, w = latents.shape
    t = _ref_sample(model_config, bs, latents.device, quantile)
    if shift := model_config.get("shift", None):
        t = (t * shift) / (1 + (shift - 1) * t)
    else:
        m = (1.15 - 0.5) / (6400 - 256)
        mu = (h // 2) * (w // 2) * m + (0.5 - m * 256)
        t = math.exp(mu) / (math.exp(mu) + (1 / t - 1))
    noise = torch.randn_like(latents)
    te = t.view(-1, 1, 1, 1)
    return (1 - te) * latents + te * noise, noise - latents, t.view(-1, 1)


def _ref_cosmos(model_config, latents, quantile):
    bs, _, _, h, w = latents.shape
    t = _ref_sample(model_config, bs, latents.device, quantile)
    if shift := model_config.get("shift", None):
        t = (t * shift) / (1 + (shift - 1) * t)
    elif model_config.get("flux_shift", False):
        m = (1.15 - 0.5) / (4096 - 256)
        mu = m * ((h // 2) * (w // 2)) + (0.5 - m * 256)
        t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** 1.0)
    noise = torch.randn_like(latents)
    te = t.view(-1, 1, 1, 1, 1)
    return (1 - te) * latents + te * noise, noise - latents, t.view(-1, 1)


def _shared(model_config, latents, quantile, mu):
    t = sample_timesteps(model_config, latents.shape[0], latents.device, quantile)
    t = shift_timesteps(t, model_config.get("shift"), mu)
    return add_flow_noise(latents, t)


CONFIGS = [
    {},
    {"sigmoid_scale": 1.7},
    {"timestep_sample_method": "uniform"},
    {"shift": 3.0},
    {"shift": 0},  # falsy fixed shift falls through to the dynamic path
    {"flux_shift": True},
    {"flux_shift": True, "shift": 2.0},
]


@pytest.mark.parametrize("model_config", CONFIGS)
@pytest.mark.parametrize("quantile", [None, 0.3])
def test_matches_krea2_reference(model_config, quantile):
    latents = torch.randn(3, 16, 12, 20)
    torch.manual_seed(7)
    expected = _ref_krea2(model_config, latents, quantile)
    torch.manual_seed(7)
    mu = calculate_shift(6 * 10, 256, 6400, 0.5, 1.15)
    got = _shared(model_config, latents, quantile, mu)
    for e, g in zip(expected, got):
        assert torch.equal(e, g)


@pytest.mark.parametrize("model_config", CONFIGS)
@pytest.mark.parametrize("quantile", [None, 0.3])
def test_matches_cosmos_reference(model_config, quantile):
    latents = torch.randn(2, 16, 1, 12, 20)
    torch.manual_seed(11)
    expected = _ref_cosmos(model_config, latents, quantile)
    torch.manual_seed(11)
    mu = calculate_shift(6 * 10) if model_config.get("flux_shift") else None
    got = _shared(model_config, latents, quantile, mu)
    for e, g in zip(expected, got):
        assert torch.equal(e, g)


def test_unknown_sample_method_raises():
    with pytest.raises(NotImplementedError):
        sample_timesteps({"timestep_sample_method": "bogus"}, 2, torch.device("cpu"))


def test_time_shift_identity_at_mu_zero():
    t = torch.linspace(0.1, 0.9, 5)
    assert torch.allclose(time_shift(0.0, 1.0, t), t)


def test_model_modules_keep_their_shift_helpers():
    """krea2/cosmos keep their public shift helpers, now backed by dit_common."""
    from rengu_flow.model.cosmos_predict2 import pipeline as cosmos_pipeline
    from rengu_flow.model.cosmos_predict2 import preview_sampling as cosmos_preview
    from rengu_flow.model.krea2 import pipeline as krea2_pipeline

    assert krea2_pipeline.calculate_shift(4096) == calculate_shift(4096, 256, 6400, 0.5, 1.15)
    t = torch.tensor([0.2, 0.7])
    assert torch.equal(krea2_pipeline.time_shift(0.4, t), time_shift(0.4, 1.0, t))
    for mod in (cosmos_pipeline, cosmos_preview):
        assert mod.get_lin_function(y1=0.5, y2=1.15)(1024) == calculate_shift(1024)
        assert torch.equal(mod.time_shift(0.4, 1.0, t), time_shift(0.4, 1.0, t))


def test_preview_compute_dtype_resolves_strings():
    class P:
        model_config = {"dtype": "float16"}

    assert preview_compute_dtype(P()) is torch.float16
    P.model_config = {}
    assert preview_compute_dtype(P()) is torch.bfloat16


# ---- DiTPipeline --------------------------------------------------------------------------


class _Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


class _Pipe(DiTPipeline):
    def __init__(self) -> None:
        self.config = {}
        self.model_config = {"dtype": torch.float32}
        self.transformer = _Mod()
        self.vae = _Mod()
        self.text_encoder = _Mod()
        self.reloads = {"vae": 0, "te": 0}

    def _reload_vae_for_preview(self) -> None:
        self.reloads["vae"] += 1
        self.vae = _Mod()

    def _reload_text_encoder_for_preview(self):
        self.reloads["te"] += 1
        return _Mod()


def _dev(m):
    return next(m.parameters()).device.type


def test_preview_lifecycle_reloads_once_and_parks_on_cpu():
    p = _Pipe()
    p.vae.to("meta")
    p.text_encoder.to("meta")
    for _ in range(3):
        p.ensure_vae_for_preview()
        p.ensure_text_encoder_for_preview("cpu")
        p.offload_text_encoder_after_encode({})
        p.restore_after_preview()
        assert _dev(p.vae) == "meta", "VAE freed by caching goes back to meta"
        assert _dev(p.text_encoder) == "cpu", "TE stays on CPU between previews"
    assert p.reloads == {"vae": 3, "te": 1}


def test_restore_resumes_training_block_swap_and_retrains():
    class _Off:
        enabled = True

        def __init__(self) -> None:
            self.calls = []

        def suspend(self):
            self.calls.append("suspend")

        def resume(self):
            self.calls.append("resume")

    p = _Pipe()
    off = _Off()
    p._block_swap_offloader = off
    assert p._suspend_training_block_swap() is off
    p.transformer.eval()
    p._preview_restore_state = {"transformer_was_training": True}
    p.restore_after_preview()
    assert off.calls == ["suspend", "resume"]
    assert p.transformer.training
    assert p._preview_offloader is None and p._preview_restore_state is None


def test_offload_skipped_for_training_resident_text_encoder():
    p = _Pipe()
    p.cache_text_embeddings = False
    moves = []
    p.text_encoder.to = lambda *a, **k: moves.append(a)  # type: ignore[method-assign]
    p.offload_text_encoder_after_encode({})
    assert not moves


def test_save_adapter_forwards_export_prefix(monkeypatch):
    from rengu_flow.networks import adapter_dit

    seen = []
    monkeypatch.setattr(adapter_dit, "save", lambda *a, **k: seen.append(k))
    p = _Pipe()
    p.adapter_config = {"type": "lora"}
    p.save_adapter("x", {})
    p.adapter_export_prefix = "transformer."
    p.save_adapter("x", {})
    assert seen == [{}, {"export_prefix": "transformer."}]


def test_configure_adapter_passes_class_targets_and_groups(monkeypatch):
    from rengu_flow.model.krea2.pipeline import Krea2Pipeline
    from rengu_flow.networks import adapter_dit

    seen = {}

    def fake_configure(transformer, adapter_config, targets, layer_groups):
        seen.update(targets=targets, layer_groups=layer_groups)
        return None, adapter_config["type"]

    monkeypatch.setattr(adapter_dit, "configure", fake_configure)
    p = object.__new__(Krea2Pipeline)
    p.transformer = _Mod()
    p.configure_adapter({"type": "lokr", "dtype": torch.float32})
    assert seen["targets"] == ("Krea2Transformer2DModel",)
    assert seen["layer_groups"] is Krea2Pipeline.adapter_layer_groups
    assert p.adapter_type == "lokr"


def test_loss_fn_masks_and_averages():
    p = _Pipe()
    p.config = {}
    out = torch.ones(1, 1, 2, 2)
    target = torch.zeros(1, 1, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    loss = p.get_loss_fn()(out, (target, mask))
    assert loss.item() == pytest.approx(0.25)


def test_configure_adapter_lora_on_fp8_base_keeps_full_precision_init():
    """PEFT casts the fresh LoRA init to the fp8 base dtype; configure_adapter must undo it."""
    from rengu_flow.training.quantize_dit import Fp8TensorwiseLinear

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = Fp8TensorwiseLinear(torch.nn.Linear(256, 256, bias=False))

    class Pipe(DiTPipeline):
        adapter_target_modules = ["Block"]

    torch.manual_seed(0)
    p = object.__new__(Pipe)
    p.transformer = Block()
    assert p.transformer.proj.weight.dtype == torch.float8_e4m3fn
    p.configure_adapter({"type": "lora", "rank": 16, "alpha": 16, "dtype": torch.float32})
    layer = p.transformer.proj
    lora_a, lora_b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
    assert lora_a.dtype == torch.float32
    assert lora_a.unique().numel() > 500
    assert lora_b.dtype == torch.float32 and not lora_b.any()


# ---- shared pieces (dit_common.loading / preview_sampler / layers / text, DiTPipeline templates)
# Each is pinned against the pre-extraction body of the model code it replaces, so a model that
# migrates onto it keeps its numerics bit for bit.


def test_shifted_sigmas_match_the_krea2_and_qwen_schedules():
    from rengu_flow.model.dit_common.preview_sampler import shifted_sigmas

    def ref(n, mu, terminal=None):
        s = torch.linspace(1.0, 1.0 / n, n, dtype=torch.float64)
        s = math.exp(mu) / (math.exp(mu) + (1 / s - 1))
        if terminal is not None:  # qwen_image21: stretched so the last sigma lands on it
            one_minus = 1 - s
            s = 1 - one_minus / (one_minus[-1] / (1 - terminal))
        return torch.cat([s, s.new_zeros(1)]).float()

    for n, mu, terminal in [(8, 0.9, None), (28, 1.1, None), (28, 0.7, 0.02), (3, 0.5, 0.02)]:
        got = shifted_sigmas(n, mu, shift_terminal=terminal)
        assert got.dtype == torch.float32 and torch.equal(got, ref(n, mu, terminal))


def test_apply_cfg_keeps_each_model_formula():
    from rengu_flow.model.dit_common.preview_sampler import CFG_FROM_COND, CFG_FROM_UNCOND, apply_cfg

    g = torch.Generator().manual_seed(0)
    cond, uncond = torch.randn(2, 3, 5, generator=g)
    assert torch.equal(apply_cfg(cond, uncond, 4.5, CFG_FROM_COND), cond + 4.5 * (cond - uncond))
    assert torch.equal(apply_cfg(cond, uncond, 4.0, CFG_FROM_UNCOND), uncond + 4.0 * (cond - uncond))
    with pytest.raises(ValueError):
        apply_cfg(cond, uncond, 1.0, "both")


def test_seeded_noise_draws_on_the_generator_device():
    from rengu_flow.model.dit_common.preview_sampler import seeded_noise

    shape = (1, 4, 6, 8)
    expected = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(7), dtype=torch.float32)
    assert torch.equal(seeded_noise(shape, 7, "cpu"), expected)
    assert torch.equal(seeded_noise(shape, 7, "cpu", generator_device="cpu"), expected)
    assert seeded_noise(shape, 7, "cpu", dtype=torch.float64).dtype == torch.float64


def test_preview_size_rounds_and_floors():
    from rengu_flow.model.dit_common.preview_sampler import preview_size

    assert preview_size({}, 16) == (1024, 1024)
    assert preview_size({"width": 1000, "height": 520}, 32) == (992, 512)
    assert preview_size({"width": 3, "height": 3}, 16) == (0, 0)
    assert preview_size({"width": 3, "height": 3}, 16, minimum=16) == (16, 16)


class _RecordingOffloader:
    def __init__(self) -> None:
        self.calls = []

    def wait_for_block(self, i):
        self.calls.append(("wait", i))

    def submit_move_blocks_forward(self, i):
        self.calls.append(("submit", i))


def test_run_blocks_brackets_each_block_with_the_preview_offloader():
    from rengu_flow.model.dit_common.preview_sampler import run_blocks

    class P:
        _preview_offloader = _RecordingOffloader()

    seen = []
    out = run_blocks(P, [10, 20], 1, lambda i, block, h: seen.append((i, block, h)) or h + block)
    assert out == 31 and seen == [(0, 10, 1), (1, 20, 11)]
    assert P._preview_offloader.calls == [("wait", 0), ("submit", 0), ("wait", 1), ("submit", 1)]
    assert run_blocks(object(), [1, 2], 0, lambda i, block, h: h + block) == 3  # no offloader


def test_offloaded_block_layer_wraps_the_block_and_passes_the_tail_through():
    from rengu_flow.model.dit_common.layers import OffloadedBlockLayer

    class TransformerLayer(OffloadedBlockLayer):
        def run_block(self, inputs):
            x, y = inputs[:2]
            if self.offloader is not None:
                self.offloader.calls.append(("run", self.block_idx))
            return self.block(x).t(), y  # non-contiguous on purpose

    off = _RecordingOffloader()
    layer = TransformerLayer(torch.nn.Identity(), 3, off)
    x, y, extra = torch.randn(4, 2), torch.randn(2), torch.randn(3, 4).t()
    out = layer((x, y, extra))
    assert off.calls == [("wait", 3), ("run", 3), ("submit", 3)]
    assert len(out) == 3 and torch.equal(out[0], x.t()) and out[0].is_contiguous()
    assert out[2] is extra  # passthrough items are untouched
    assert len(TransformerLayer(torch.nn.Identity(), 0)((x, y))) == 2  # no offloader


def test_mark_inter_layer_grads_marks_carriers_unless_pipe_parallel():
    from rengu_flow.model.dit_common.layers import mark_inter_layer_grads

    def fresh():
        return torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.long), None

    outputs = fresh()
    mark_inter_layer_grads(outputs, False, outputs[:1])
    assert [getattr(t, "requires_grad", None) for t in outputs] == [True, False, False, None]
    outputs = fresh()
    mark_inter_layer_grads(outputs, True, outputs[:1])
    assert [getattr(t, "requires_grad", None) for t in outputs] == [True, True, False, None]


def test_ragged_rows_slices_every_output_by_the_mask():
    from rengu_flow.model.dit_common import ragged_rows

    embeds = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    pad = torch.tensor([[False, True, False, False], [True, False, False, False]])
    out = ragged_rows(embeds, mask, image_pad_mask=pad)
    assert [r.shape[0] for r in out["prompt_embeds"]] == [2, 3]
    assert torch.equal(out["prompt_embeds"][1], embeds[1, :3])
    assert [r.tolist() for r in out["text_mask"]] == [[True, True], [True, True, True]]
    assert [r.tolist() for r in out["image_pad_mask"]] == [[False, True], [True, False, False]]


@pytest.mark.parametrize("video", [False, True])
def test_flow_matching_batch_matches_the_krea2_and_cosmos_bodies(video):
    import torch.nn.functional as F

    from rengu_flow.model.dit_common import flow_matching_batch

    model_config = {"timestep_sample_method": "logit_normal"}
    shape = (2, 4, 1, 6, 8) if video else (2, 4, 6, 8)
    latents = torch.randn(shape)
    pixel_mask = (torch.rand(2, 48, 64) > 0.5).float()
    mu = calculate_shift(24)

    torch.manual_seed(3)
    noisy, t, target, mask = flow_matching_batch(model_config, latents, pixel_mask, None, mu=mu)

    torch.manual_seed(3)
    ref_mask = F.interpolate(pixel_mask.unsqueeze(1), size=shape[-2:], mode="nearest-exact")
    if video:
        ref_mask = ref_mask.unsqueeze(2)
    ref_t = shift_timesteps(sample_timesteps(model_config, 2, latents.device, None), None, mu)
    ref_noisy, ref_target, ref_t = add_flow_noise(latents, ref_t)
    for a, b in [(noisy, ref_noisy), (t, ref_t), (target, ref_target), (mask, ref_mask)]:
        assert torch.equal(a, b)
    assert flow_matching_batch(model_config, latents, None, 0.5)[3] is None


def test_text_padding_trimmer_is_an_opt_in_method():
    from rengu_flow.model.dit_common import text_padding_trimmer

    class P(DiTPipeline):
        trim_micro_batch = text_padding_trimmer(embeds_idx=1, mask_idx=2)

    assert not hasattr(DiTPipeline, "trim_micro_batch")
    mask = torch.tensor([[True, False, False]])
    out = object.__new__(P).trim_micro_batch((torch.zeros(1), torch.ones(1, 3, 2), mask))
    assert out[1].shape == (1, 1, 2) and out[2].tolist() == [[True]]


def test_guard_not_prequantized_message_and_markers():
    from rengu_flow.config.validation import ConfigValidationError
    from rengu_flow.model.dit_common.loading import guard_not_prequantized

    kwargs = dict(kind="int8/fp8", advice="; use the bf16 file.")
    guard_not_prequantized({"a.weight": torch.zeros(1)}, "transformer_path", **kwargs)
    with pytest.raises(ConfigValidationError) as e:
        guard_not_prequantized({"a.weight": torch.zeros(1, dtype=torch.float8_e4m3fn)}, "transformer_path", **kwargs)
    assert str(e.value) == (
        "model.transformer_path points at a pre-quantized (int8/fp8) file, which cannot be trained; use the bf16 file."
    )
    with pytest.raises(ConfigValidationError):
        guard_not_prequantized({"a.scale_weight": torch.zeros(1)}, "vae_path", **kwargs)
    with pytest.raises(ConfigValidationError):
        guard_not_prequantized(
            {"a.comfy_quant": torch.zeros(1)}, "vae_path", key_suffixes=("comfy_quant",), **kwargs
        )
    with pytest.raises(ConfigValidationError):
        guard_not_prequantized(
            {"a.weight": torch.zeros(1, dtype=torch.int8)}, "vae_path", dtypes=(torch.int8,), **kwargs
        )


def test_config_dir_or_bundled_and_load_file_or_folder(tmp_path):
    from rengu_flow.config.validation import ConfigValidationError
    from rengu_flow.model.dit_common.loading import config_dir_or_bundled, load_file_or_folder

    bundled = tmp_path / "bundled"
    folder, bare, file = tmp_path / "folder", tmp_path / "bare", tmp_path / "w.safetensors"
    folder.mkdir()
    bare.mkdir()
    (folder / "config.json").write_text("{}")
    file.write_text("x")
    assert config_dir_or_bundled(folder, bundled) == folder
    assert config_dir_or_bundled(bare, bundled) == bundled
    assert config_dir_or_bundled(file, bundled) == bundled

    loaders = dict(folder=lambda p: ("folder", p), file=lambda p: ("file", p))
    assert load_file_or_folder(str(folder), "vae_path", **loaders) == ("folder", folder)
    assert load_file_or_folder(file, "vae_path", **loaders) == ("file", file)
    with pytest.raises(ConfigValidationError, match="model.vae_path"):
        load_file_or_folder(tmp_path / "missing", "vae_path", **loaders)


def test_component_path_uses_the_spec_bundle_key(tmp_path):
    from rengu_flow.config.validation import ConfigValidationError
    from rengu_flow.model.krea2.pipeline import Krea2Pipeline

    p = object.__new__(Krea2Pipeline)
    p.model_config = {"checkpoint_path": str(tmp_path), "vae_path": "my/vae.safetensors"}
    assert p.spec.bundle_key == "checkpoint_path"
    assert p._component_path("vae") == "my/vae.safetensors"
    assert p._component_path("transformer") == str(tmp_path / "transformer")
    p.model_config = {}
    with pytest.raises(ConfigValidationError, match=r"model\.text_encoder_path is required for krea2.*model\.checkpoint_path"):
        p._component_path("text_encoder")


def test_init_pipeline_enforces_a_required_text_cache():
    from rengu_flow.config.validation import ConfigValidationError
    from rengu_flow.model.krea2.pipeline import Krea2Pipeline

    p = object.__new__(Krea2Pipeline)
    p._init_pipeline({"model": {"dtype": torch.float32}})
    assert p.cache_text_embeddings is True and p.transformer is None and p._preview_embed_cache == {}
    with pytest.raises(ConfigValidationError, match="krea2 requires cache_text_embeddings = true: the tapped"):
        p._init_pipeline({"model": {"dtype": torch.float32, "cache_text_embeddings": False}})


def test_qwen3vl_tokenizer_is_shared_and_bundled():
    from rengu_flow.model.dit_common.qwen3vl import QWEN3VL_TOKENIZER, load_qwen3vl_tokenizer

    assert (QWEN3VL_TOKENIZER / "tokenizer.json").is_file()
    tokenizer = load_qwen3vl_tokenizer(None, "tokenizer_path")
    assert tokenizer("<|im_start|>user").input_ids[0] == 151644
