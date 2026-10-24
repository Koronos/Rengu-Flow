"""Contract B: every registered model's real pipeline over its tiny kit (CPU).

Pipeline layers vs the model's own forward, the inter-layer tuple, gradients, adapter scope,
export keys and reload (DiTs and SDXL alike, through the kit's adapter hooks), layer groups /
module roots vs the real denoiser, the quantization scope, the VAE and text-encoder cache fns, the
training target, the micro-batch trim, previews (the model's own sampler and the spec-dispatched
runner), the lazy text encoder and the full-model export round trip — plus the numeric goldens
that pin them. Checks a model cannot honor are declared by its kit (``NOT_APPLICABLE``)."""

from __future__ import annotations

import re

import pytest
import torch

from contract_helpers import block_module_ids, image_summary, inter_layer_values, materialize, tensor_summary
from rengu_flow.model.testkit_common import ADAPTER_CONFIGS, SEAM_CAPTIONS, run_layers

pytestmark = pytest.mark.no_ui_db


def prediction(out):
    """The layers' prediction (a final layer may append metadata, e.g. SDXL's timesteps)."""
    return out[0] if isinstance(out, tuple) else out


def _features(pipeline, case, seed: int = 0, quantile: float | None = 0.5):
    batch = case.sample_batch(pipeline, seed)
    torch.manual_seed(seed)
    return batch, *pipeline.prepare_inputs(batch, timestep_quantile=quantile)


# ---- layers ---------------------------------------------------------------------------------


def test_layers_match_the_monolithic_forward(kit_case, golden):
    """``to_layers()`` == the model's own forward, the tuple between layers carries tensors only
    (DeepSpeed pipe), and the output is pinned by a numeric golden."""
    kit, case = kit_case
    p = kit.pipeline()
    kit.denoiser(p).eval()
    _batch, features, (target, _mask) = _features(p, case)
    assert all(torch.is_tensor(f) for f in features)
    with torch.no_grad():
        out, between = inter_layer_values(p.to_layers(), features)
        out = prediction(out)
        expected = kit.module.monolithic_forward(p, features)
    for i, value in enumerate(between):
        assert isinstance(value, tuple) and all(torch.is_tensor(v) for v in value), f"after layer {i}: {type(value)}"
    assert out.shape == expected.shape == target.shape
    golden(kit.type_id, f"layers_output/{case.name}", tensor_summary(out))
    with kit.known_gap("layers_vs_monolithic"):
        assert torch.allclose(out, expected, rtol=1e-4, atol=1e-5), (out - expected).abs().max()


def test_gradients_reach_every_trainable_parameter(kit_case, golden):
    """Full finetune: one step (random timesteps and noise, fixed seed) through the layers and the
    loss reaches every trainable parameter with a non-zero gradient; the loss is pinned by a
    numeric golden."""
    kit, case = kit_case
    p = kit.pipeline()
    trainable = {n: prm for n, prm in kit.denoiser(p).named_parameters() if prm.requires_grad}
    assert trainable
    _batch, features, label = _features(p, case, quantile=None)
    out = run_layers(p.to_layers(), features)
    loss = p.get_loss_fn()(out, label)
    loss.backward()
    assert torch.isfinite(loss)
    missing = sorted(n for n, prm in trainable.items() if prm.grad is None or not prm.grad.abs().sum())
    assert not missing, f"no (or an all-zero) gradient: {missing[:8]}"
    golden(kit.type_id, f"loss_step/{case.name}", float(loss))


def test_pipe_parallel_layers_hand_on_only_grad_carrying_floats(kit_case):
    """``pipeline_stages > 1``: DeepSpeed pipe backprops every floating tensor that crosses a stage
    boundary, so every floating item the layers hand on requires grad (a constant RoPE table or a
    float ``layout`` left out trips the pipe's backward in stage 0)."""
    kit, case = kit_case
    p = kit.pipeline(pipeline_stages=2)
    _batch, features, _label = _features(p, case)
    _out, between = inter_layer_values(p.to_layers(), features)
    for i, value in enumerate(between):
        loose = [j for j, v in enumerate(value) if torch.is_floating_point(v) and not v.requires_grad]
        assert not loose, f"after layer {i}: floating items {loose} do not require grad"


# ---- adapters -----------------------------------------------------------------------------------

ADAPTER_SUFFIX = re.compile(r"\.(lokr_\w+|lora_\w+|hada_\w+|alpha|scalar|dora_scale)(\.weight)?$")
# PEFT renames the wrapped base weight (".base_layer.") and nests the model (base_model.model.).
_PEFT_WRAP = re.compile(r"(base_model\.model\.|\.base_layer(?=\.))")


def _trainable(kit, p) -> dict:
    """``{original_name: param}`` of every trainable parameter of the adapted networks."""
    return {
        prm.original_name.replace(".default", ""): prm
        for net in kit.adapter_networks(p).values()
        for prm in net.parameters()
        if prm.requires_grad
    }


def _export(kit, p, save_dir, perturb: bool = True) -> dict:
    """``save_adapter`` of the current trainable weights (randomly moved off their init first, so
    a zero-initialized half cannot hide a wrong mapping) read back from ``save_dir``."""
    from safetensors.torch import load_file

    if perturb:
        with torch.no_grad():
            for prm in _trainable(kit, p).values():
                prm.add_(torch.randn_like(prm) * 0.01)
    state = {name: prm.detach().clone() for name, prm in _trainable(kit, p).items()}
    p.save_adapter(save_dir, state)
    (file,) = save_dir.glob("*.safetensors")
    return load_file(str(file))


@pytest.mark.parametrize("adapter", list(ADAPTER_CONFIGS))
def test_adapter_attaches_to_the_declared_scope_and_exports_golden_keys(kit, adapter, tmp_path, golden):
    """Each adapter leaves every base weight frozen, trains the denoiser through the layers, and
    exports exactly the declared scope (``kit.adapter_scope``: the DiT's ``adapter_target_modules``
    linears; SDXL's UNet blocks + text encoders) under ``spec.export_prefix`` with a pinned key set.
    An export loads back into a fresh adapter and re-exports identically."""
    kit.require("adapters")
    if adapter not in kit.spec.adapters:
        pytest.skip(f"{kit.type_id} does not offer {adapter}")
    if adapter.startswith("lycoris_"):
        pytest.importorskip("lycoris")
    base = kit.pipeline()
    scope = kit.adapter_scope(base, adapter)
    base_names = {prefix + n for prefix, net in kit.adapter_networks(base).items() for n, _ in net.named_parameters()}
    p = kit.pipeline(adapter=dict(ADAPTER_CONFIGS[adapter]))
    trainable = _trainable(kit, p)
    assert trainable, "the adapter added no trainable weight"
    unwrapped = {_PEFT_WRAP.sub("", name) for name in trainable}
    assert not base_names & unwrapped, f"base weights trainable: {sorted(base_names & unwrapped)[:4]}"

    case = kit.cases[0]
    _batch, features, label = _features(p, case, quantile=None)
    p.get_loss_fn()(run_layers(p.to_layers(), features), label).backward()
    denoiser = {id(prm) for prm in kit.denoiser(p).parameters()}
    no_grad = sorted(n for n, prm in trainable.items() if id(prm) in denoiser and prm.grad is None)
    assert not no_grad, f"no gradient: {no_grad[:4]}"

    exported = _export(kit, p, tmp_path / "export")
    prefix = kit.spec.export_prefix or ""
    assert exported and all(k.startswith(prefix) for k in exported), sorted(exported)[:4]
    golden(kit.type_id, f"export_keys/{adapter}", sorted(exported))
    modules = {ADAPTER_SUFFIX.sub("", k)[len(prefix) :] for k in exported}
    with kit.known_gap(f"adapter_scope/{adapter}"):
        assert sorted(modules) == scope, (sorted(set(scope) - modules)[:4], sorted(modules - set(scope))[:4])

    q = kit.pipeline(adapter=dict(ADAPTER_CONFIGS[adapter]))
    q.load_adapter_weights(tmp_path / "export")
    again = _export(kit, q, tmp_path / "again", perturb=False)
    assert again.keys() == exported.keys()
    for key, value in exported.items():
        assert torch.equal(again[key], value), key


def test_default_adapter_scope_follows_the_spec_default_groups(kit):
    """With no ``layer_groups`` / ``target_include`` a LoRA trains exactly the modules of the spec's
    ``adapter_default_layer_groups`` (every targeted linear when it declares none); an explicit
    ``target_include`` replaces that default."""
    from rengu_flow.networks import adapter_dit
    from rengu_flow.networks.adapter_targets import filter_target_names

    kit.require("adapters")
    p = kit.pipeline()
    if not hasattr(p, "adapter_target_modules"):
        pytest.skip(f"{kit.type_id} is not a DiT pipeline (its adapters have their own scope)")
    groups = p.adapter_layer_groups or {}
    names = adapter_dit._collect_target_linears(p.transformer, tuple(p.adapter_target_modules))
    defaults = kit.spec.adapter_default_layer_groups
    expected = filter_target_names(names, [g for d in defaults for g in groups[d]], None) if defaults else names

    def adapted(**extra):
        q = kit.pipeline(adapter={**ADAPTER_CONFIGS["lora"], **extra})
        return sorted(
            _PEFT_WRAP.sub("", n).rsplit(".lora_A", 1)[0] for n, m in q.transformer.named_modules() if n.endswith("lora_A")
        )

    assert adapted() == sorted(expected)
    if defaults:
        explicit = adapted(target_include=[sorted(names)[0]])
        assert explicit == [sorted(names)[0]]


def _break_a_factor(state: dict) -> dict:
    """``state`` without one B-side factor (``lora_B`` / ``lora_up`` / ``lokr_w2``)."""
    victim = next(k for k in sorted(state) if re.search(r"\.(lora_B|lora_up|lokr_w2)\b", k))
    return {k: v for k, v in state.items() if k != victim}


@pytest.mark.parametrize("adapter", list(ADAPTER_CONFIGS))
def test_adapter_export_passes_the_generic_export_check(kit, adapter, tmp_path):
    """The real export of each adapter is the file ``spec.adapter_export(adapter)`` names and passes
    ``export_check.check_run`` (what ``run_model_smoke.sh`` runs after a real training); a
    misnamed file, a key outside the declared roots and a missing factor each fail it."""
    from safetensors.torch import save_file

    from rengu_flow.networks.export_check import check_run

    kit.require("adapters")
    if adapter not in kit.spec.adapters:
        pytest.skip(f"{kit.type_id} does not offer {adapter}")
    if adapter.startswith("lycoris_"):
        pytest.importorskip("lycoris")
    layout = kit.spec.adapter_export(adapter)
    p = kit.pipeline(adapter=dict(ADAPTER_CONFIGS[adapter]))
    out = tmp_path / "output" / "run" / "step1"
    state = _export(kit, p, out)
    assert [f.name for f in out.glob("*.safetensors")] == [layout.file]
    config = tmp_path / "train.toml"
    config.write_text(f'[model]\ntype = "{kit.type_id}"\n[adapter]\ntype = "{adapter}"\n', encoding="utf-8")
    assert kit.type_id in check_run(config, tmp_path / "output")

    broken = {"foreign_root": {"zz_" + k: v for k, v in state.items()}, "missing_factor": _break_a_factor(state)}
    broken["misnamed_file"] = state
    for name, bad in broken.items():
        bad_out = tmp_path / name / "run" / "step1"
        bad_out.mkdir(parents=True)
        save_file(bad, str(bad_out / ("renamed.safetensors" if name == "misnamed_file" else layout.file)))
        with pytest.raises(AssertionError, match=re.escape(f"no {layout.file}") if name == "misnamed_file" else None):
            check_run(config, tmp_path / name)


def test_reentrant_checkpointing_reaches_every_adapter_weight(kit):
    """With an adapter only on the blocks (frozen embedders), reentrant activation checkpointing of
    the ``checkpointable_layers`` (as DeepSpeed / the single-device engine run it: the inter-layer
    tuple unpacked into positional args) still backprops into every trainable denoiser weight —
    the first layer must mark the carriers ``requires_grad`` (``mark_inter_layer_grads``), or the
    checkpointed segments are cut off from autograd."""
    from torch.utils.checkpoint import checkpoint

    kit.require("adapters")
    if "lora" not in kit.spec.adapters:
        pytest.skip(f"{kit.type_id} does not offer lora")
    p = kit.pipeline(adapter=dict(ADAPTER_CONFIGS["lora"]))
    checkpointable = set(p.checkpointable_layers)
    layers = list(p.to_layers())
    in_blocks = {id(prm) for layer in layers if type(layer).__name__ in checkpointable for prm in layer.parameters()}
    trainable = {}
    with torch.no_grad():
        for name, prm in _trainable(kit, p).items():
            if id(prm) in in_blocks:
                prm.add_(torch.randn_like(prm) * 0.01)  # lora_B starts at 0: lora_A would get no grad
                trainable[name] = prm
            else:
                prm.requires_grad_(False)  # the adapter only on the blocks: embedders etc. frozen
    assert trainable, f"no adapter weight inside a {sorted(checkpointable)} layer"
    _batch, features, label = _features(p, kit.cases[0], quantile=None)
    x, checkpointed = features, 0
    for layer in layers:
        if type(layer).__name__ in checkpointable and isinstance(x, tuple):
            x = checkpoint(lambda *inputs, _layer=layer: _layer(inputs), *x, use_reentrant=True)
            checkpointed += 1
        else:
            x = layer(x)
    assert checkpointed, f"no {sorted(checkpointable)} layer in to_layers()"
    loss = p.get_loss_fn()(x, label)
    with kit.known_gap("reentrant_checkpointing"):
        assert loss.requires_grad, "the checkpointed layers are cut off from autograd (loss needs no grad)"
        loss.backward()
        missing = sorted(n for n, prm in trainable.items() if prm.grad is None or not prm.grad.abs().sum())
        assert not missing, f"no gradient under reentrant checkpointing: {missing[:6]}"


def test_scope_exclusions_are_frozen_by_default_in_the_other_families(kit):
    """A module a family leaves out of its scope (``ADAPTER_SCOPE_EXCLUDES``) is not trained by
    the others either under the default config: the default trained scope is family-independent."""
    excludes = getattr(kit.module, "ADAPTER_SCOPE_EXCLUDES", {})
    if not excludes:
        pytest.skip(f"{kit.type_id}: no adapter family narrows the scope")
    prefixes = tuple({prefix for values in excludes.values() for prefix in values})
    for adapter in (a for a in ADAPTER_CONFIGS if a in kit.spec.adapters and a not in excludes):
        p = kit.pipeline(adapter=dict(ADAPTER_CONFIGS[adapter]))
        params = list(_trainable(kit, p).values())
        p.get_param_groups(params)  # applies the per-module learning rates (lr 0 = frozen)
        still = sorted(
            n for n, prm in _trainable(kit, p).items() if _PEFT_WRAP.sub("", n).startswith(prefixes)
        )
        assert not still, f"{adapter} trains {still[:4]} by default, outside the other families' scope"


def test_layer_groups_and_module_roots_match_the_real_dit(kit, monkeypatch):
    """The spec's adapter layer groups are the pipeline's and each matches real linears of the
    default scope; the torch-free module roots (preflight typo check) are real children of the
    denoiser, include every child that holds parameters, and preflight accepts each of them."""
    from rengu_flow.config.preflight import collect_preflight_issues
    from rengu_flow.networks import adapter_dit
    from rengu_flow.networks.adapter_targets import filter_target_names

    kit.require("adapters")
    p = kit.pipeline()
    net = kit.denoiser(p)
    groups = getattr(p, "adapter_layer_groups", None) or {}
    assert sorted(kit.spec.adapter_layer_groups) == sorted(groups)
    if groups:
        names = adapter_dit._collect_target_linears(net, tuple(p.adapter_target_modules))
        for group, patterns in groups.items():
            assert filter_target_names(names, list(patterns), None), group
    roots = set(kit.spec.adapter_module_roots)
    if not roots:
        return  # no torch-free typo check for this model (preflight skips it)
    children = {n for n, _ in net.named_children()}
    with_params = {n for n, m in net.named_children() if any(True for _ in m.parameters())}
    assert roots <= children, f"stale roots: {sorted(roots - children)}"
    with kit.known_gap("module_roots"):
        assert with_params <= roots, f"roots missing: {sorted(with_params - roots)}"
    monkeypatch.setenv("RENGU_PREFLIGHT", "1")  # tests/conftest.py turns host probes off
    config = {"model": {"type": kit.type_id}, "adapter": {"type": "lora", "target_include": [f"{r}.*" for r in sorted(roots)]}}
    assert not [i for i in collect_preflight_issues(config) if "module root" in i]
    config["adapter"]["target_include"] = ["no_such_root.*"]
    assert [i for i in collect_preflight_issues(config) if "no_such_root" in i]


def test_quantization_touches_only_block_linears(kit):
    """``transformer_fp8_matmul`` converts per-block linears only (every block gets some), never
    the shared embedders / projections."""
    if kit.spec.quant is None or kit.spec.quant.fp8 is None:
        pytest.skip(f"{kit.type_id} declares no fp8 quantization")
    p = kit.pipeline(transformer_fp8_matmul=True)
    quantized = {name: m for name, m in p.transformer.named_modules() if type(m).__name__.startswith("Fp8")}
    assert quantized
    inside = block_module_ids(p)
    outside = sorted(name for name, m in quantized.items() if id(m) not in inside)
    assert not outside, outside
    for i, block in enumerate(p.get_block_swap_modules()):
        assert any(type(m).__name__.startswith("Fp8") for m in block.modules()), f"block {i} not quantized"


# ---- cache fns ------------------------------------------------------------------------------------


def test_vae_fn_is_deterministic(kit):
    """A cached latent is the posterior mode: encoding the same pixels twice is identical."""
    p = kit.pipeline()
    vae = p.get_vae()
    materialize(vae)
    fn = p.get_call_vae_fn(vae)
    g = torch.Generator().manual_seed(0)
    pixels = torch.rand(kit.module.VAE_INPUT_SHAPE, generator=g) * 2 - 1
    first, second = fn(pixels), fn(pixels.clone())
    assert first.keys() == second.keys() and "latents" in first
    for key in first:
        assert torch.isfinite(first[key]).all() and torch.equal(first[key], second[key]), key


def _rows(value) -> list:
    return list(value) if isinstance(value, list) else list(value.unbind(0))


def test_text_encoder_fn_rows_are_batch_independent(kit):
    """One cached row per caption, equal to encoding that caption alone (a cache entry must not
    depend on the batch it was encoded in)."""
    p = kit.pipeline()
    encoders = p.get_text_encoders()
    if not encoders:
        pytest.skip(f"{kit.type_id} caches no text embeddings")
    captions = list(getattr(kit.module, "RAGGED_CAPTIONS", SEAM_CAPTIONS[:4]))
    for te in encoders:
        materialize(te)
        fn = p.get_call_text_encoder_fn(te)
        out = fn(captions, False)
        assert out
        for i, caption in enumerate(captions):
            alone = fn([caption], False)
            assert alone.keys() == out.keys()
            for key in out:
                rows = _rows(out[key])
                assert len(rows) == len(captions), key
                got, want = rows[i], _rows(alone[key])[0]
                assert got.shape == want.shape, (key, caption, got.shape, want.shape)
                assert torch.allclose(got.float(), want.float(), rtol=1e-4, atol=1e-4), (key, caption)


def test_text_encoder_fn_caches_ragged_rows(kit):
    """Captions of different lengths give rows of different lengths (no padded tails); every
    sequence output of one encoder is ragged the same way, the others are one vector per caption."""
    from rengu_flow.utils.cache import sequence_tensor_keys

    kit.require("ragged_text")
    p = kit.pipeline()
    captions = list(getattr(kit.module, "RAGGED_CAPTIONS", SEAM_CAPTIONS[:4]))
    for te in p.get_text_encoders():
        materialize(te)
        out = p.get_call_text_encoder_fn(te)(captions, False)
        ragged = {key: [v.shape[0] for v in value] for key, value in out.items() if isinstance(value, list)}
        assert ragged, f"no ragged output: { {k: type(v).__name__ for k, v in out.items()} }"
        # The cache stores a row ragged when it is 2-D or its key is declared (spec
        # ragged_cache_keys): an undeclared 1-D ragged key would be stacked padded.
        undeclared = [k for k in ragged if out[k][0].ndim != 2 and k not in sequence_tensor_keys()]
        assert not undeclared, f"1-D ragged keys missing from the spec's ragged_cache_keys: {undeclared}"
        lengths = next(iter(ragged.values()))
        assert len(set(lengths)) > 1, lengths
        assert all(v == lengths for v in ragged.values()), ragged
        for key, value in out.items():
            if key not in ragged:
                assert torch.is_tensor(value) and value.shape[0] == len(captions), key


# ---- training inputs ------------------------------------------------------------------------------


def test_prepare_inputs_builds_the_flow_matching_target(kit_case):
    """``noisy = x + t * target`` (rectified flow, velocity target ``noise - x``) per row, with
    ``t`` in (0, 1); a timestep quantile is deterministic."""
    kit, case = kit_case
    kit.require("flow_target")
    p = kit.pipeline()
    batch, features, (target, _mask) = _features(p, case, quantile=None)
    latents = batch["latents"].float()
    noisy, t = features[0], features[1].view(-1)
    assert noisy.shape == latents.shape == target.shape
    assert bool(((t > 0) & (t < 1)).all())
    tb = t.view(-1, *[1] * (latents.ndim - 1))
    assert torch.allclose(noisy, latents + tb * target, atol=1e-5)
    _, f1, _ = _features(p, case, seed=1, quantile=0.3)
    _, f2, _ = _features(p, case, seed=2, quantile=0.3)
    assert torch.equal(f1[1], f2[1])


def test_trim_micro_batch_drops_unused_text_lanes_without_changing_the_output(kit):
    """GAS split of mixed caption lengths: each micro-batch drops the text lanes only the other
    one used, and predicts exactly what the padded micro-batch did."""
    from rengu_flow.data.loader import split_batch

    p = kit.pipeline()
    if not callable(getattr(type(p), "trim_micro_batch", None)):
        pytest.skip(f"{kit.type_id} has no micro-batch trim")
    p.transformer.eval()
    case = kit.cases[0]
    _batch, features, label = _features(p, case)
    layers = p.to_layers()
    trimmed_any = False
    for micro, _ in split_batch((features, label), 2):
        trimmed = p.trim_micro_batch(micro)
        assert len(trimmed) == len(micro)
        trimmed_any |= any(a.shape != b.shape for a, b in zip(trimmed, micro))
        with torch.no_grad():
            assert torch.allclose(run_layers(layers, trimmed), run_layers(layers, micro), rtol=1e-4, atol=1e-5)
    assert trimmed_any, "no micro-batch had unused text lanes: the case's caption lengths do not mix"


# ---- previews ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [False, True], ids=["no_cfg", "cfg"])
def test_preview_end_to_end_memoizes_prompt_embeddings(kit, cfg, golden):
    """``prepare_preview_memory`` -> ``generate_preview_image`` -> ``restore_after_preview``:
    an RGB image of the requested size, pinned by a golden; the prompts (and the negative one with
    CFG) are encoded once — a second preview never touches the text encoder."""
    kit.require("dit_preview")
    p = kit.pipeline()
    preview_cfg = kit.module.preview_config(cfg)

    def preview():
        p.prepare_preview_memory(preview_cfg)
        image = p.generate_preview_image(preview_cfg, "a red fox", 1, 3)
        p.restore_after_preview()
        return image

    image = preview()
    assert image.size == (preview_cfg["width"], preview_cfg["height"]) and image.mode == "RGB"
    expected_keys = {"a red fox", preview_cfg["negative_prompt"]} if cfg else {"a red fox"}
    assert set(map(str, p._preview_embed_cache)) == expected_keys
    encodes = []
    original = p.ensure_text_encoder_for_preview
    p.ensure_text_encoder_for_preview = lambda *a, **k: (encodes.append(a), original(*a, **k))[1]
    again = preview()
    assert not encodes, "the second preview re-encoded memoized prompts"
    assert list(again.getdata()) == list(image.getdata())
    golden(kit.type_id, f"preview/{'cfg' if cfg else 'no_cfg'}", image_summary(image), rel=1e-3, abs_=1.0)


def test_preview_block_swap_follows_the_spec_memory_knobs(kit, monkeypatch):
    """``preview_blocks_to_swap > 0`` streams the blocks through a preview offloader only when the
    spec's ``preview_spec.memory_knobs`` has ``blocks_to_swap``; a model whose sampler runs the DiT
    whole (nanosaur2) ignores it and creates none."""
    from types import SimpleNamespace

    kit.require("dit_preview")
    p = kit.pipeline()
    made = []

    def make(*args, **kwargs):
        made.append(args)
        return SimpleNamespace(teardown=lambda: None)

    monkeypatch.setattr(p, "_make_preview_offloader", make)
    p.prepare_preview_memory({**kit.module.preview_config(False), "preview_blocks_to_swap": 1})
    offloader = p._preview_offloader
    p.restore_after_preview()
    if "blocks_to_swap" in kit.spec.preview_spec.memory_knobs:
        assert offloader is not None and made, "preview_blocks_to_swap > 0 created no preview offloader"
    else:
        assert offloader is None and not made, "a model without the blocks_to_swap knob got a preview offloader"


def test_previews_render_through_the_spec_runner(kit, golden):
    """``utils.preview.run_previews`` dispatches by ``spec.preview_spec.runner`` and logs one RGB
    image per prompt of the requested size (it swallows a failing preview, so a missing image is
    the failure); the first image is pinned by a golden."""
    from types import SimpleNamespace

    from rengu_flow.utils.preview import run_previews

    kit.require("preview")
    assert kit.spec.preview_spec is not None, f"{kit.type_id} kit declares 'preview' but its spec has no preview_spec"
    p = kit.pipeline()
    preview_cfg = {**kit.module.preview_config(True), "enabled": True, "prompts": ["a red fox", "girl with blue ears"]}
    logged = []
    sink = SimpleNamespace(image=lambda tag, image, step: logged.append((tag, image, step)))
    run_previews(p, {"preview": preview_cfg, "pipeline_stages": 1}, sink, 3)
    assert [(tag, step) for tag, _, step in logged] == [("preview/prompt_0", 3), ("preview/prompt_1", 3)]
    for _, image, _ in logged:
        assert tuple(image.shape) == (3, preview_cfg["height"], preview_cfg["width"])
        assert torch.isfinite(image).all()
    golden(kit.type_id, "preview_dispatch", tensor_summary(logged[0][1]), rel=1e-3, abs_=4e-3)


# ---- loading / export -------------------------------------------------------------------------------


def test_init_does_not_load_the_text_encoder(kit):
    """A warm text cache never needs the encoder: building the pipeline must not read it."""
    kit.require("lazy_text_encoder")
    p = kit.pipeline()
    for te in p.get_text_encoders():
        assert next(te.parameters()).device.type == "meta"


def test_save_model_round_trips_through_the_loader(kit, tmp_path):
    """A full-finetune export loads back through the model's loader with identical trained
    weights and an identical forward."""
    kit.require("save_round_trip")
    p = kit.pipeline()
    net = kit.denoiser(p)
    with torch.no_grad():
        for prm in net.parameters():
            prm.add_(torch.randn_like(prm) * 0.01)
    state = {prm.original_name: prm.detach().clone() for prm in net.parameters()}
    p.save_model(tmp_path, state)
    q = kit.reload_export(tmp_path)
    reloaded = {prm.original_name: prm for prm in kit.denoiser(q).parameters()}
    assert reloaded.keys() == state.keys()
    for name, value in state.items():
        assert torch.equal(reloaded[name].detach(), value), name
    net.eval()
    kit.denoiser(q).eval()
    case = kit.cases[0]
    _batch, features, _label = _features(p, case)
    with torch.no_grad():
        assert torch.equal(prediction(run_layers(q.to_layers(), features)), prediction(run_layers(p.to_layers(), features)))
