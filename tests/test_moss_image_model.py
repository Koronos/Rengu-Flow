"""CPU-only tests for MOSS-Image on tiny random models: the report's sigma sampler against the
closed-form formula, the text recipe against a literal copy of the reference ``encode_prompt``, the
VAE encode convention against diffusers' own Flux2 helpers, the batched padded forward against the
reference's per-sample forward, the loaders for every released layout (official folders and the
Comfy-Org single files), adapter scope, the negative-branch block skip and the export layout.

Numerical parity with the official pipeline on the real weights lives in
``scripts/parity_moss_image.py`` (needs the HF snapshot)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import NormalDist

import pytest
import torch
from safetensors.torch import load_file, save_file

from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.model.moss_image import loading, preview_sampling, testkit, timesteps
from rengu_flow.model.moss_image.dit import MOSSImageTransformer, detect_config
from rengu_flow.model.moss_image.pipeline import MOSSImagePipeline, bn_stats, patchify, unpatchify
from rengu_flow.model.moss_image.text import encode_prompts, format_prompt
from rengu_flow.model.testkit_common import ADAPTER_CONFIGS

pytestmark = pytest.mark.no_ui_db


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """CPU only: CUDA is hidden (previews would otherwise move the tiny DiT to the GPU)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    return testkit.write_bundle(tmp_path_factory.mktemp("moss_bundle"))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> MOSSImagePipeline:
    return testkit.tiny_pipeline(testkit.tiny_config(tmp_path_factory.mktemp("moss_pipe")))


# ---- the report's sigma sampler -----------------------------------------------------------------


def _reference_pmf(mu, s, a, rho, grid=1000):
    """Eq. 4 computed with plain Python (no torch): ``F(x) = Pr(1 - sigmoid(Z) <= x)``."""
    normal = NormalDist(mu, s)

    def cdf(x):
        return 0.0 if x <= 0 else 1.0 - normal.cdf(math.log((1 - x) / x))

    return [
        (1 - rho) * (cdf(a * j / grid) - cdf(a * (j - 1) / grid)) / cdf(a) + rho / grid for j in range(1, grid + 1)
    ]


@pytest.mark.parametrize(
    "mu,s,a,rho",
    [(-1.09, 0.95, 0.95, 0.01), (0.0, 1.0, 0.5, 0.0), (-1.45, 1.05, 0.95, 0.3)],
)
def test_sigma_pmf_is_equation_4(mu, s, a, rho):
    pmf = timesteps.grid_pmf(mu, s, a, rho)
    assert pmf.shape == (1000,) and math.isclose(float(pmf.sum()), 1.0, rel_tol=1e-12)
    expected = torch.tensor(_reference_pmf(mu, s, a, rho), dtype=torch.float64)
    assert torch.allclose(pmf, expected, rtol=1e-8, atol=1e-14)


def test_default_sampler_matches_the_report_final_stage():
    pmf = timesteps.grid_pmf()
    median = (int(torch.searchsorted(pmf.cumsum(0), torch.tensor(0.5, dtype=torch.float64))) + 1) / 1000
    assert 0.74 < median < 0.82  # the report's final stage (median sigma ~0.78)
    sigmas = timesteps.sample_sigmas(20000, "cpu")
    assert bool(((sigmas > 0) & (sigmas <= 1)).all())
    assert torch.allclose((sigmas * 1000).round(), sigmas * 1000, atol=1e-2)  # on the 1000-step grid
    assert abs(float(sigmas.double().mean()) - float((pmf * torch.arange(1, 1001) / 1000).sum())) < 0.01
    first = timesteps.sample_sigmas(3, "cpu", quantile=0.5)
    assert torch.equal(first, timesteps.sample_sigmas(3, "cpu", quantile=0.5)) and float(first[0]) == pytest.approx(median, abs=1e-6)


def test_uniform_mix_one_is_uniform_and_cutoff_stretches_the_grid():
    assert torch.allclose(timesteps.grid_pmf(uniform_mix=1.0), torch.full((1000,), 1e-3, dtype=torch.float64))
    with pytest.raises(ValueError):
        timesteps.grid_pmf(cutoff=1.0)


def test_pipeline_honors_the_sampler_choice(pipeline):
    cfg = pipeline.model_config
    cfg["timestep_sample_method"] = "uniform"
    try:
        uniform = pipeline.sample_timesteps(5000, "cpu")
        assert 0.45 < float(uniform.mean()) < 0.55
        cfg["timestep_sample_method"] = "moss"
        moss = pipeline.sample_timesteps(5000, "cpu")
        assert float(moss.mean()) > 0.7
        cfg["shift"] = 2.0
        shifted = pipeline.sample_timesteps(5000, "cpu", timestep_quantile=0.5)
        assert float(shifted[0]) == pytest.approx(2 * 0.781 / (1 + 0.781), abs=1e-6)
    finally:
        for key in ("timestep_sample_method", "shift"):
            cfg.pop(key, None)


# ---- text recipe --------------------------------------------------------------------------------

PROMPTS = ["", "a red fox", "girl with blue ears, standing in a forest at dusk, long wavy hair\nsecond line", "forest"]


def _reference_encode(full_model, tokenizer, prompt, max_length=4096):
    """A literal copy of the reference ``encode_prompt`` for one prompt, over the *full* ``Qwen3_5Model``."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length)
    return full_model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]


@pytest.mark.parametrize("max_length", [4096, 9])
def test_text_encoding_equals_the_reference_encode_prompt(bundle, max_length):
    full = testkit.tiny_text_encoder().eval()
    tokenizer = loading.load_tokenizer(None)
    encoder = loading.load_text_encoder(bundle / "text_encoder", torch.float32)
    with torch.no_grad():
        embeds, mask = encode_prompts(encoder, tokenizer, PROMPTS, "cpu", max_length=max_length)
        for i, prompt in enumerate(PROMPTS):
            want = _reference_encode(full, tokenizer, prompt, max_length)
            assert int(mask[i].sum()) == want.shape[0]
            assert torch.equal(embeds[i][mask[i]], want), prompt
    assert not bool(embeds[~mask].any())  # right padding only
    if max_length == 9:
        assert int(mask.sum(1).max()) == 9  # truncated like the reference
    template = format_prompt(tokenizer, "A cat")
    assert template == "<|im_start|>user\nA cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert format_prompt(tokenizer, "") == "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n"


# ---- VAE ----------------------------------------------------------------------------------------


def test_patchify_is_diffusers_flux2_and_encode_normalizes_with_bn(pipeline):
    from diffusers import Flux2Pipeline

    z = torch.randn(2, testkit.VAE_LATENT, 8, 12)
    assert torch.equal(patchify(z), Flux2Pipeline._patchify_latents(z))
    packed = patchify(z)
    assert torch.equal(unpatchify(packed), z) and torch.equal(unpatchify(packed), Flux2Pipeline._unpatchify_latents(packed))

    vae = pipeline.get_vae()
    pixels = torch.rand(1, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        mode = vae.encode(pixels).latent_dist.mode()
        want = (Flux2Pipeline._patchify_latents(mode) - vae.bn.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        )
        got = pipeline.get_call_vae_fn(vae)(pixels)["latents"]
    assert got.shape == (1, testkit.LATENT_CH, 4, 4) and torch.equal(got, want)
    mean, std = bn_stats(vae, "cpu", torch.float32)
    assert torch.allclose(got * std + mean, patchify(mode), atol=1e-5)


# ---- the batched padded forward vs the reference per-sample forward -----------------------------


def test_padded_batch_equals_the_per_sample_reference_and_masking_is_needed(pipeline):
    p = pipeline
    p.transformer.eval()
    case = testkit.cases()[0]
    batch = case.sample_batch(p, 0, lengths=(6, 1, 4, 6))  # incl. a one-token caption, and two full ones
    torch.manual_seed(0)
    features, _ = p.prepare_inputs(batch, timestep_quantile=0.5)
    layers = p.to_layers()
    with torch.no_grad():
        got = features
        for layer in layers:
            got = layer(got)
        want = testkit.monolithic_forward(p, features)
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)
        # Each sample alone gives the same prediction as inside the padded batch.
        for i in range(4):
            n = int(features[3][i].sum())
            alone = (features[0][i : i + 1], features[1][i : i + 1], features[2][i : i + 1, :n], features[3][i : i + 1, :n])
            out = alone
            for layer in layers:
                out = layer(out)
            assert torch.allclose(out, got[i : i + 1], rtol=1e-4, atol=1e-5), i
        # Without the key mask the padding leaks in: the masking is what makes the equality hold.
        unmasked = (*features[:3], torch.ones_like(features[3]))
        leaked = unmasked
        for layer in layers:
            leaked = layer(leaked)
        assert (leaked[1] - want[1]).abs().max() > 1e-4  # fp32 noise is ~1e-7


def test_an_empty_text_row_is_an_error(pipeline):
    noisy = torch.randn(1, testkit.LATENT_CH, 4, 4)
    with pytest.raises(ValueError, match="at least one valid text token"):
        pipeline.transformer.embed(noisy, torch.tensor([500.0]), torch.randn(1, 3, testkit.TEXT_DIM), torch.zeros(1, 3, dtype=torch.bool))


# ---- loaders ------------------------------------------------------------------------------------


def _state_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _fused(state: dict) -> dict:
    out = {k: v for k, v in state.items() if ".proj.q." not in k and ".proj.k." not in k and ".proj.v." not in k}
    for i in range(testkit.DIT_CONFIG["num_layers"]):
        out[f"layers.{i}.proj.qkv.weight"] = torch.cat([state[f"layers.{i}.proj.{n}.weight"] for n in "qkv"], 0)
    return out


def test_dit_loads_from_folder_official_file_and_comfy_fused_file(bundle, tmp_path):
    reference = loading.load_transformer(bundle / "transformer", torch.float32)
    ref_state = {k: v for k, v in reference.state_dict().items()}
    assert detect_config(ref_state) == {**testkit.DIT_CONFIG, "rope_base": 10000.0}
    file_state = {k: v.contiguous() for k, v in ref_state.items()}
    official = tmp_path / "official.safetensors"
    save_file(file_state, str(official))
    comfy = tmp_path / "moss_image_base_bf16.safetensors"
    save_file({k: v.contiguous() for k, v in _fused(file_state).items()}, str(comfy))
    prefixed = tmp_path / "prefixed.safetensors"
    save_file({f"diffusion_model.{k}": v.contiguous() for k, v in _fused(file_state).items()}, str(prefixed))
    for path in (official, comfy, prefixed):
        loaded = loading.load_transformer(path, torch.float32)
        assert _state_equal(loaded.state_dict(), ref_state), path.name
    assert loading.load_transformer(comfy, torch.bfloat16).layers[0].proj["q"].weight.dtype == torch.bfloat16


def test_dit_shards_with_an_index_load(bundle, tmp_path):
    reference = loading.load_transformer(bundle / "transformer", torch.float32)
    state = {k: v.contiguous() for k, v in reference.state_dict().items()}
    keys = sorted(state)
    halves = {"diffusion_pytorch_model-00001-of-00002.safetensors": keys[::2], "diffusion_pytorch_model-00002-of-00002.safetensors": keys[1::2]}
    weight_map = {}
    for name, names in halves.items():
        save_file({k: state[k] for k in names}, str(tmp_path / name))
        weight_map.update({k: name for k in names})
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "config.json").write_text(json.dumps({**testkit.DIT_CONFIG, "_class_name": "MOSSImageTransformer"}))
    assert _state_equal(loading.load_transformer(tmp_path, torch.float32).state_dict(), reference.state_dict())


@pytest.mark.parametrize("marker", ["int8", "scale"])
def test_prequantized_dit_is_refused(bundle, tmp_path, marker):
    state = {k: v.contiguous() for k, v in loading.load_transformer(bundle / "transformer", torch.float32).state_dict().items()}
    if marker == "int8":
        state["layers.0.proj.o.weight"] = torch.zeros_like(state["layers.0.proj.o.weight"], dtype=torch.int8)
    else:
        state["layers.0.proj.o.weight_scale"] = torch.ones(32, 1)
    path = tmp_path / "moss_image_base_int8_convrot.safetensors"
    save_file(state, str(path))
    with pytest.raises(ConfigValidationError, match="pre-quantized"):
        loading.load_transformer(path, torch.float32)


def test_text_encoder_loads_from_the_comfy_single_file_and_refuses_quantized(bundle, tmp_path):
    folder_state = load_file(str(bundle / "text_encoder" / "model.safetensors"))
    comfy = {f"model.{k}": v for k, v in folder_state.items()}  # model.language_model.* / model.visual.*
    comfy["lm_head.weight"] = torch.randn(8, testkit.TEXT_DIM)
    comfy["mtp.fc.weight"] = torch.randn(4, 4)
    path = tmp_path / "qwen3.5_9b_bf16.safetensors"
    save_file(comfy, str(path))
    from_folder = loading.load_text_encoder(bundle / "text_encoder", torch.float32)
    # A single file carries no config: the bundled 9B one would not fit the tiny weights, so the check
    # is on the key remap itself, then on a full load through a folder holding that file.
    assert {loading._text_key(k) for k in comfy} - {None} == set(from_folder.state_dict())
    folder = tmp_path / "te"
    folder.mkdir()
    save_file(comfy, str(folder / "model.safetensors"))
    (folder / "config.json").write_text((bundle / "text_encoder" / "config.json").read_text())
    from_comfy = loading.load_text_encoder(folder, torch.float32)
    assert _state_equal(from_comfy.state_dict(), from_folder.state_dict())
    for suffix in ("weight_codebook", "weight_scale"):
        bad = tmp_path / f"bad_{suffix}.safetensors"
        save_file({**comfy, f"model.language_model.layers.0.mlp.up_proj.{suffix}": torch.ones(2)}, str(bad))
        with pytest.raises(ConfigValidationError, match="pre-quantized"):
            loading.load_text_encoder(bad, torch.float32)


def _to_ldm(state: dict) -> dict:
    """The inverse of ``convert_ldm_vae_state_dict``, written independently: diffusers keys -> the
    original LDM layout of Comfy-Org's ``flux2-vae.safetensors``."""
    levels = 4
    out = {}
    for key, value in state.items():
        new = key
        new = re.sub(r"^quant_conv\.", "encoder.quant_conv.", new)
        new = re.sub(r"^post_quant_conv\.", "decoder.post_quant_conv.", new)
        new = re.sub(r"^(encoder|decoder)\.conv_norm_out\.", r"\1.norm_out.", new)
        new = re.sub(r"^(encoder|decoder)\.mid_block\.resnets\.(\d)\.", lambda m: f"{m[1]}.mid.block_{int(m[2]) + 1}.", new)
        m = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(group_norm|to_q|to_k|to_v|to_out\.0)\.", new)
        if m:
            name = {"group_norm": "norm", "to_q": "q", "to_k": "k", "to_v": "v", "to_out.0": "proj_out"}[m[2]]
            new = f"{m[1]}.mid.attn_1.{name}." + new[m.end() :]
            if value.ndim == 2:
                value = value[:, :, None, None]
        new = re.sub(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.down.\1.block.\2.", new)
        new = re.sub(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", new)
        new = re.sub(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"decoder.up.{levels - 1 - int(m[1])}.block.{m[2]}.", new)
        new = re.sub(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.", lambda m: f"decoder.up.{levels - 1 - int(m[1])}.upsample.", new)
        out[new.replace(".conv_shortcut.", ".nin_shortcut.")] = value
    return out


def test_vae_loads_from_the_folder_the_diffusers_file_and_the_original_layout(bundle, tmp_path):
    folder = loading.load_vae(bundle / "vae")
    ref = {k: v for k, v in folder.state_dict().items()}
    assert folder.decoder.conv_in.out_channels == testkit.VAE_CONFIG["decoder_block_out_channels"][-1]  # the small decoder
    assert not folder.training and not any(p.requires_grad for p in folder.parameters())
    diffusers_file = tmp_path / "flux2-vae_small.safetensors"
    save_file({k: v.contiguous() for k, v in ref.items()}, str(diffusers_file))
    ldm_file = tmp_path / "flux2-vae.safetensors"
    ldm = _to_ldm(ref)
    assert any(k.startswith("encoder.down.") and ".block." in k for k in ldm) and "encoder.quant_conv.weight" in ldm
    save_file({k: v.contiguous() for k, v in ldm.items()}, str(ldm_file))
    assert _state_equal(loading.convert_ldm_vae_state_dict(ldm), ref)
    pixels = torch.rand(1, 3, 64, 64) * 2 - 1
    for path in (diffusers_file, ldm_file):
        # Single files carry no config: the bundled one has the release's width, so only the tiny
        # widths of this kit need the folder config; the conversion itself is proven above.
        vae_cfg = testkit.vae_config_json()
        loaded = loading._vae_from_state(
            loading._read_vae_state([path]) if path is diffusers_file else loading.convert_ldm_vae_state_dict(load_file(str(path))),
            vae_cfg,
            torch.float32,
            "vae_path",
        )
        with torch.no_grad():
            assert torch.equal(loaded.encode(pixels).latent_dist.mode(), folder.encode(pixels).latent_dist.mode())
    with pytest.raises(ConfigValidationError, match="FLUX.2 VAE"):
        loading._vae_from_state({k: v for k, v in ref.items() if "decoder.conv_in" not in k}, testkit.vae_config_json(), torch.float32, "vae_path")


def test_missing_paths_are_config_errors(tmp_path):
    for load in (loading.load_transformer, loading.load_vae, loading.load_text_encoder):
        with pytest.raises(ConfigValidationError, match="does not exist"):
            load(tmp_path / "nope.safetensors", torch.float32)
    with pytest.raises(ConfigValidationError, match="tokenizer_path"):
        loading.load_tokenizer(tmp_path / "nope")


# ---- adapters ----------------------------------------------------------------------------------


def _adapted(tmp_path_factory, **adapter):
    from rengu_flow.model.testkit_common import prepare_adapter_config

    config = testkit.tiny_config(tmp_path_factory.mktemp("moss_adapter"), adapter={"type": "lora", "rank": 2, "alpha": 2, **adapter})
    config["adapter"] = prepare_adapter_config(config["adapter"], config["model"]["dtype"])
    p = MOSSImagePipeline(config)
    p.load_diffusion_model()
    p.configure_adapter(config["adapter"])
    return p


def _lora_modules(p):
    return sorted({n.rsplit(".lora_A", 1)[0].replace("base_model.model.", "") for n, _ in p.transformer.named_modules() if ".lora_A" in n and n.endswith("lora_A")})


@pytest.mark.parametrize(
    "adapter,expect",
    [
        ({}, lambda n: re.fullmatch(r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)", n)),
        ({"layer_groups": ["text_refiner"]}, lambda n: n.startswith("text_embedder.blocks.")),
        ({"layer_groups": ["attention"]}, lambda n: re.fullmatch(r"layers\.\d+\.proj\.[qkvo]", n)),
        ({"layer_groups": ["modulation", "image_in_out", "text_projection"]}, lambda n: not n.startswith(("layers.", "text_embedder.blocks"))),
        ({"target_include": ["layers.0.*"]}, lambda n: n.startswith("layers.0.")),
    ],
    ids=["default_blocks", "text_refiner", "attention", "rest", "explicit_include_replaces_default"],
)
def test_adapter_scope(tmp_path_factory, adapter, expect):
    modules = _lora_modules(_adapted(tmp_path_factory, **adapter))
    assert modules and all(expect(n) for n in modules), modules
    if not adapter:
        assert len(modules) == 6 * testkit.DIT_CONFIG["num_layers"]


def test_adapter_export_keys_map_through_comfyui_naming(tmp_path_factory, tmp_path):
    """Every exported module path is a module of the official checkpoint, so ComfyUI's
    ``diffusion_model.<path>`` / ``lora_unet_<path_with_underscores>`` / bare ``<path>`` all resolve."""
    p = _adapted(tmp_path_factory)
    state = {prm.original_name.replace(".default", ""): prm.detach().clone() for prm in p.transformer.parameters() if prm.requires_grad}
    p.save_adapter(tmp_path, state)
    exported = load_file(str(tmp_path / "adapter_model.safetensors"))
    official = {k[: -len(".weight")] for k in loading.load_transformer(testkit.write_bundle(tmp_path_factory.mktemp("b")) / "transformer", torch.float32).state_dict() if k.endswith(".weight")}
    assert exported and all(k.startswith("diffusion_model.") for k in exported)
    modules = {re.sub(r"\.(lora_[AB]\.weight|alpha)$", "", k)[len("diffusion_model.") :] for k in exported}
    assert modules <= official
    assert sum(k.endswith(".alpha") for k in exported) == len(modules)


# ---- previews -----------------------------------------------------------------------------------


def test_reference_sigmas_follow_the_scheduler():
    sigmas = preview_sampling.reference_sigmas(50)
    assert sigmas.dtype == torch.float32 and float(sigmas[0]) == 1.0 and float(sigmas[-1]) == 0.0
    assert bool((sigmas[1:] < sigmas[:-1]).all()) and len(sigmas) >= 45
    on_grid = sigmas[:-1] * 1000
    assert torch.allclose(on_grid, on_grid.round(), atol=2e-2)  # snapped to the 1000-step training grid
    assert len(preview_sampling.reference_sigmas(1)) == 2


def test_negative_branch_skips_the_block_and_equals_the_reference_skip(pipeline, monkeypatch):
    p = pipeline
    p.transformer.eval()
    latents = torch.randn(1, testkit.LATENT_CH, 4, 4)
    row = torch.randn(5, testkit.TEXT_DIM)
    sigma = torch.tensor(0.6)
    with torch.no_grad():
        skipped = preview_sampling.velocity(p, latents, sigma, row, skip=(1,))
        want = p.transformer(latents, sigma * 1000, row[None], None, skip_layers=(1,), return_dict=False)[0]
        assert torch.allclose(skipped, want, rtol=1e-4, atol=1e-5)
        assert not torch.allclose(skipped, preview_sampling.velocity(p, latents, sigma, row), atol=1e-4)
    calls = []
    original = preview_sampling.velocity
    monkeypatch.setattr(preview_sampling, "velocity", lambda *a, **k: (calls.append(k.get("skip", ())), original(*a, **k))[1])
    monkeypatch.setattr(preview_sampling, "UNCOND_SKIP_LAYERS", (1, 22))  # 22 does not exist in a 3-block DiT: ignored
    cfg = testkit.preview_config(True)
    p.prepare_preview_memory(cfg)
    p.generate_preview_image(cfg, "a red fox", 1, 3)
    p.restore_after_preview()
    assert calls == [(), (1,), (), (1,), (), (1,)]  # cond, then the skipping negative branch, per step


# ---- export -------------------------------------------------------------------------------------


def test_full_export_is_the_official_transformer_folder(pipeline, tmp_path):
    state = {prm.original_name: prm.detach().clone() for prm in pipeline.transformer.parameters()}
    pipeline.save_model(tmp_path, state)
    saved = load_file(str(tmp_path / "diffusion_pytorch_model.safetensors"))
    assert saved.keys() == {*state, "rope.bands"} and saved["rope.bands"].dtype == torch.float32
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["_class_name"] == "MOSSImageTransformer" and config["num_layers"] == testkit.DIT_CONFIG["num_layers"]
    reloaded = MOSSImageTransformer.from_pretrained(tmp_path)  # what the official pipeline does
    assert _state_equal({k: v for k, v in reloaded.state_dict().items() if k != "rope.bands"}, state)
