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
