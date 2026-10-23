"""Equivalence with the *official* ``pipeline_moss_image.py`` on tiny random weights (CPU, fp32).

The reference file ships inside the Hugging Face repository; point ``MOSS_IMAGE_REFERENCE_PY`` at it, or
have a snapshot of ``OpenMOSS-Team/MOSS-Image-*`` in the default Hugging Face cache. Without it the
tests skip. They drive the reference's own classes (``MOSSImageTransformer``, ``MOSSImageScheduler``,
``MOSSImagePipeline.encode_prompt`` and ``__call__``) with the tiny kit's weights and compare them with
rengu's port: the transformer forward (per-sample and the batched padded layers), the skipped-block
negative branch, the text recipe and the whole preview sampler. Real-weights parity:
``scripts/parity_moss_image.py``."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from rengu_flow.model.moss_image import loading, preview_sampling, testkit
from rengu_flow.model.moss_image.layers import FinalLayer, InitialLayer, TransformerLayer
from rengu_flow.model.moss_image.text import encode_prompts

pytestmark = pytest.mark.no_ui_db


def _reference_path() -> Path | None:
    if os.environ.get("MOSS_IMAGE_REFERENCE_PY"):
        return Path(os.environ["MOSS_IMAGE_REFERENCE_PY"])
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    found = sorted(hub.glob("models--OpenMOSS-Team--MOSS-Image-*/snapshots/*/pipeline_moss_image.py"))
    return found[0] if found else None


@pytest.fixture(scope="module")
def ref():
    path = _reference_path()
    if path is None or not path.is_file():
        pytest.skip("official pipeline_moss_image.py not available (set MOSS_IMAGE_REFERENCE_PY)")
    spec = importlib.util.spec_from_file_location("official_pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="module")
def ours(tmp_path_factory):
    return testkit.tiny_pipeline(testkit.tiny_config(tmp_path_factory.mktemp("moss_ref")))


@pytest.fixture(scope="module")
def official(ref, ours):
    model = ref.MOSSImageTransformer(**testkit.DIT_CONFIG).eval()
    model.load_state_dict(ours.transformer.state_dict(), strict=True)
    return model


def _batch(lengths=(6, 1, 4)):
    g = torch.Generator().manual_seed(0)
    context = torch.zeros(len(lengths), max(lengths), testkit.TEXT_DIM)
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for i, n in enumerate(lengths):
        context[i, :n] = torch.randn(n, testkit.TEXT_DIM, generator=g)
        mask[i, :n] = True
    latents = torch.randn(len(lengths), testkit.LATENT_CH, 4, 6, generator=g)
    return latents, torch.tensor([350.0, 780.0, 990.0]), context, mask


def test_transformer_forward_equals_the_official_one(ours, official):
    latents, t, context, mask = _batch()
    ours.transformer.eval()
    with torch.no_grad():
        want = official(latents, t, context, mask, return_dict=False)[0]
        assert torch.equal(ours.transformer(latents, t, context, mask, return_dict=False)[0], want)
        got = (latents, (t / 1000).view(-1, 1), context, mask)
        for layer in ours.to_layers():
            got = layer(got)
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)
        for skip in [(1,), (0, 2)]:
            want = official(latents, t, context, mask, return_dict=False, skip_layers=skip)[0]
            assert torch.equal(ours.transformer(latents, t, context, mask, return_dict=False, skip_layers=skip)[0], want)


def test_official_rope_matches_the_batched_table(ours, official):
    lengths = [6, 1, 4]
    batched = ours.transformer.rope.batched(4, 6, lengths, 6, "cpu")
    for i, n in enumerate(lengths):
        one = official.rope(4, 6, n, torch.device("cpu"))
        assert torch.equal(batched[i, : 24 + n], one[0])


def test_text_recipe_and_preview_sampler_equal_the_official_pipeline(ref, ours, official):
    text_encoder = testkit.tiny_text_encoder().eval()
    tokenizer = loading.load_tokenizer(None)
    pipe = ref.MOSSImagePipeline(official, ref.MOSSImageScheduler(), text_encoder, tokenizer, ours.vae)
    prompts = ["a red fox", "", "girl with blue ears\nsecond line"]
    embeds, mask = pipe.encode_prompt(prompts)
    mine, my_mask = encode_prompts(loading.load_text_encoder(ours._component_path("text_encoder"), torch.float32), tokenizer, prompts, "cpu")
    assert torch.equal(mask, my_mask) and torch.equal(embeds, mine)

    steps = 4
    scheduler = ref.MOSSImageScheduler()
    scheduler.set_timesteps(steps)
    assert torch.equal(scheduler.sigmas, preview_sampling.reference_sigmas(steps))

    cfg = {**testkit.preview_config(True), "num_inference_steps": steps, "negative_prompt": "low quality"}
    expected = pipe(
        "a red fox",
        height=cfg["height"],
        width=cfg["width"],
        num_inference_steps=steps,
        guidance_scale=cfg["guidance_scale"],
        negative_prompt=cfg["negative_prompt"],
        uncond_skip_layers=(1,),  # the reference's block 22 does not exist in a 3-block DiT
        generator=torch.Generator().manual_seed(3),
    ).images[0]
    preview_sampling.UNCOND_SKIP_LAYERS = (1,)
    try:
        ours.prepare_preview_memory(cfg)
        image = ours.generate_preview_image(cfg, "a red fox", 1, 3)
        ours.restore_after_preview()
    finally:
        preview_sampling.UNCOND_SKIP_LAYERS = (22,)
    assert image.size == expected.size
    diff = (torch.tensor(list(image.getdata()), dtype=torch.int16) - torch.tensor(list(expected.getdata()), dtype=torch.int16)).abs()
    assert int(diff.max()) <= 1, int(diff.max())  # float rounding of the 8-bit conversion only
