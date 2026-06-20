"""CPU tests for the LyCORIS library adapter types (attach/export/load/fuse + config surface)."""

import pytest

try:
    import torch
    from torch import nn

    import lycoris  # noqa: F401  # the backend under test
    from rengu_flow.networks import lycoris_attach, lycoris_sdxl
    from rengu_flow.networks.lycoris_export_check import check_export
    from rengu_flow.networks.lycoris_meta import (
        LYCORIS_ADAPTER_TYPES,
        apply_lycoris_defaults,
    )
except ImportError as e:  # pragma: no cover
    pytest.skip(f"Cannot import torch/lycoris networks: {e}", allow_module_level=True)


def _rank(adapter_type):
    return 8 if adapter_type == "lycoris_dylora" else 4


def _config(adapter_type, **overrides):
    cfg = {
        "type": adapter_type,
        "rank": _rank(adapter_type),
        "alpha": _rank(adapter_type),
        "dtype": torch.float32,
        **overrides,
    }
    apply_lycoris_defaults(cfg)
    return cfg


def _models():
    """Minimal SDXL-shaped tree: unet with down/mid/up blocks plus two text encoders.

    Each block carries an affine LayerNorm so train_norm has something to attach to
    (ignored by the Linear-only preset when train_norm is off)."""
    torch.manual_seed(0)

    def block():
        return nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.Linear(16, 16))

    unet = nn.Module()
    unet.down_blocks = block()
    unet.mid_block = block()
    unet.up_blocks = block()
    return unet, nn.Sequential(nn.Linear(8, 8)), nn.Sequential(nn.Linear(8, 8))


class _Pipe:
    def __init__(self, unet, te, te2):
        self.unet, self.text_encoder, self.text_encoder_2 = unet, te, te2


def _train_one_step(unet, te, te2):
    trainable = [p for m in (unet, te, te2) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-1)
    x16, x8 = torch.randn(2, 16), torch.randn(2, 8)
    loss = (
        unet.down_blocks(x16).sum()
        + unet.mid_block(x16).sum()
        + unet.up_blocks(x16).sum()
        + te(x8).sum()
        + te2(x8).sum()
    )
    loss.backward()
    opt.step()
    return x16


def _snapshot(unet, te, te2):
    return {
        p.original_name: p.detach().clone()
        for m in (unet, te, te2)
        for p in m.parameters()
        if p.requires_grad
    }


@pytest.mark.parametrize("adapter_type", LYCORIS_ADAPTER_TYPES)
def test_configure_train_export_check(adapter_type, tmp_path):
    """configure puts params in-tree with prefixed original_name; a step trains them;
    the export passes lycoris_export_check."""
    cfg = _config(adapter_type)
    unet, te, te2 = _models()
    base_w = unet.down_blocks[0].weight.detach().clone()

    lycoris_sdxl.configure(unet, te, te2, cfg)
    trainable = [p for m in (unet, te, te2) for p in m.parameters() if p.requires_grad]
    assert trainable
    assert {p.original_name.split(".")[0] for p in trainable} == {
        "unet",
        "text_encoder",
        "text_encoder_2",
    }
    assert all(".lycoris_adapter." in p.original_name for p in trainable)

    _train_one_step(unet, te, te2)
    assert torch.equal(unet.down_blocks[0].weight, base_w), "base weight must stay frozen"

    lycoris_sdxl.save(tmp_path, _snapshot(unet, te, te2), cfg)
    failures, n_modules = check_export(tmp_path / "adapter_model.safetensors", adapter_type)
    assert not failures, failures
    assert n_modules == 8


@pytest.mark.parametrize("adapter_type", LYCORIS_ADAPTER_TYPES)
def test_save_load_round_trip(adapter_type, tmp_path):
    """Exported weights load back into a freshly configured model exactly."""
    cfg = _config(adapter_type)
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    x16 = _train_one_step(unet, te, te2)
    snapshot = _snapshot(unet, te, te2)
    lycoris_sdxl.save(tmp_path, snapshot, cfg)

    unet2, te2_, te22 = _models()
    lycoris_sdxl.configure(unet2, te2_, te22, _config(adapter_type))
    lycoris_sdxl.load(_Pipe(unet2, te2_, te22), tmp_path)

    if adapter_type == "lycoris_dylora":
        # DyLoRA forward samples a random block subset per call: compare params.
        snap2 = _snapshot(unet2, te2_, te22)
        for key, value in snapshot.items():
            assert torch.allclose(snap2[key], value, atol=1e-6), key
    else:
        with torch.no_grad():
            expected = unet.down_blocks(x16)
            loaded = unet2.down_blocks(x16)
        assert torch.allclose(loaded, expected, atol=1e-5)


@pytest.mark.parametrize(
    "adapter_type", [t for t in LYCORIS_ADAPTER_TYPES if t != "lycoris_dylora"]
)
def test_fuse_matches_adapter_output(adapter_type, tmp_path):
    """fuse_all and fuse_weights_into both reproduce the attached-adapter output."""
    cfg = _config(adapter_type)
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    x16 = _train_one_step(unet, te, te2)
    with torch.no_grad():
        expected = unet.down_blocks(x16)
    lycoris_sdxl.save(tmp_path, _snapshot(unet, te, te2), cfg)

    lycoris_sdxl.fuse(_Pipe(unet, te, te2))
    assert not list(lycoris_attach.iter_attached_adapters(unet))
    with torch.no_grad():
        assert torch.allclose(unet.down_blocks(x16), expected, atol=1e-4)

    unet3, te3, te32 = _models()
    lycoris_sdxl.load_and_fuse(_Pipe(unet3, te3, te32), tmp_path)
    with torch.no_grad():
        assert torch.allclose(unet3.down_blocks(x16), expected, atol=1e-4)


def test_configure_twice_raises():
    cfg = _config("lycoris_locon")
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    with pytest.raises(RuntimeError, match="configure_roots called twice"):
        lycoris_sdxl.configure(unet, te, te2, cfg)


def test_load_rejects_foreign_keys(tmp_path):
    cfg = _config("lycoris_locon")
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    lycoris_sdxl.save(tmp_path, _snapshot(unet, te, te2), cfg)

    import safetensors.torch

    file = tmp_path / "adapter_model.safetensors"
    state = safetensors.torch.load_file(file)
    state["lora_unet_not_a_module.lora_up.weight"] = torch.zeros(1)
    safetensors.torch.save_file(state, file)

    unet2, te2_, te22 = _models()
    lycoris_sdxl.configure(unet2, te2_, te22, _config("lycoris_locon"))
    with pytest.raises(RuntimeError, match="match no attached module"):
        lycoris_sdxl.load(_Pipe(unet2, te2_, te22), tmp_path)


def test_export_check_rejects_wrong_algo(tmp_path):
    cfg = _config("lycoris_loha")
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    _train_one_step(unet, te, te2)
    lycoris_sdxl.save(tmp_path, _snapshot(unet, te, te2), cfg)
    failures, _ = check_export(tmp_path / "adapter_model.safetensors", "lycoris_lokr")
    assert failures


def test_export_check_flags_zero_delta_and_nonfinite(tmp_path):
    import safetensors.torch

    state = {
        "lora_unet_block.lora_up.weight": torch.zeros(16, 4),
        "lora_unet_block.lora_down.weight": torch.full((4, 16), float("nan")),
        "lora_unet_block.alpha": torch.tensor(4.0),
    }
    file = tmp_path / "adapter_model.safetensors"
    safetensors.torch.save_file(state, file)
    failures, _ = check_export(file, "lycoris_locon")
    assert any("zero training delta" in f for f in failures)
    assert any("non-finite" in f for f in failures)


def test_use_scalar_folds_into_export(tmp_path):
    """A trained use_scalar never leaks a 'scalar' key; it multiplies the up factor."""
    cfg = _config("lycoris_locon")
    cfg["use_scalar"] = True
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    # scalar starts at 0 (zero effective delta); one step trains it nonzero, after
    # which the folded lora_up.weight * scalar must carry the delta.
    _train_one_step(unet, te, te2)
    lycoris_sdxl.save(tmp_path, _snapshot(unet, te, te2), cfg)

    import safetensors.torch

    state = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")
    assert not any(k.endswith(".scalar") for k in state)
    failures, _ = check_export(tmp_path / "adapter_model.safetensors", "lycoris_locon")
    assert not failures, failures


def test_dora_exports_dora_scale(tmp_path):
    # DoRA is the dora_wd toggle on a base algo (here locon), not a standalone type.
    cfg = _config("lycoris_locon", dora_wd=True)
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    _train_one_step(unet, te, te2)
    lycoris_sdxl.save(tmp_path, _snapshot(unet, te, te2), cfg)

    import safetensors.torch

    state = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")
    modules = {k.split(".", 1)[0] for k in state}
    assert all(f"{m}.dora_scale" in state for m in modules)


# ---------------------------------------------------------------------------
# Config surface (torch-light)
# ---------------------------------------------------------------------------


def _base_config(adapter):
    return {
        "dataset": "examples/minimal_dataset.toml",
        "model": {"type": "sdxl", "dtype": "bfloat16", "checkpoint_path": "x.safetensors"},
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "adapter": adapter,
    }


def _adapter_issues(adapter):
    from rengu_flow.config.validation import collect_validation_errors

    return [
        i
        for i in collect_validation_errors(_base_config(adapter))
        if "adapter" in i.lower() or "lycoris" in i.lower() or "block_size" in i
    ]


def test_validation_accepts_all_lycoris_types():
    for adapter_type in LYCORIS_ADAPTER_TYPES:
        adapter = {"type": adapter_type, "rank": 8}
        assert not _adapter_issues(adapter), adapter_type


def test_validation_rejects_unknown_lycoris_type():
    assert any("not a known LyCORIS type" in i for i in _adapter_issues({"type": "lycoris_foo", "rank": 8}))


def test_validation_rejects_wrong_per_algo_arg():
    issues = _adapter_issues({"type": "lycoris_glora", "rank": 8, "factor": 4})
    assert any("keys not supported by lycoris_glora" in i for i in issues)


def test_validation_dylora_guards():
    assert any(
        "divisible by block_size" in i
        for i in _adapter_issues({"type": "lycoris_dylora", "rank": 6})
    )
    assert any(
        "train_conv" in i
        for i in _adapter_issues({"type": "lycoris_dylora", "rank": 8, "train_conv": True})
    )


def test_defaults_fill_per_algo_and_reject_alpha():
    from rengu_flow.config.defaults import set_config_defaults
    from rengu_flow.config.validation import ConfigValidationError

    cfg = _base_config({"type": "lycoris_lokr", "rank": 16})
    set_config_defaults(cfg)
    adapter = cfg["adapter"]
    assert adapter["alpha"] == 16
    assert adapter["factor"] == -1
    assert adapter["full_matrix"] is False
    assert adapter["dora_wd"] is False
    assert adapter["wd_on_output"] is True
    assert adapter["train_conv"] is False
    assert adapter["dtype"] == torch.bfloat16

    with pytest.raises(ConfigValidationError, match="alpha"):
        set_config_defaults(_base_config({"type": "lycoris_locon", "rank": 4, "alpha": 2}))


def test_capabilities_expose_lycoris_with_labels():
    from rengu_flow.registry.model_capabilities import ADAPTER_FIELD_TEMPLATES, capabilities_for_api

    sdxl = capabilities_for_api()["sdxl"]
    for adapter_type in LYCORIS_ADAPTER_TYPES:
        assert adapter_type in sdxl["adapters"]
        assert sdxl["adapter_labels"][adapter_type].startswith("Lycoris.")
        assert ADAPTER_FIELD_TEMPLATES.get(adapter_type), adapter_type
    assert sdxl["adapter_labels"]["lora"] == "Peft.LoRA"
    assert sdxl["adapter_labels"]["lokr"] == "LoKr"  # rengu's own → no vendor prefix


def test_install_profile_routing():
    from rengu_flow.install.manager import profiles_for_config_dict

    for adapter_type in (*LYCORIS_ADAPTER_TYPES, "lokr"):
        profiles = profiles_for_config_dict({"adapter": {"type": adapter_type}})
        assert "lycoris" in profiles, adapter_type
    assert "lycoris" not in profiles_for_config_dict({"adapter": {"type": "lora"}})


def test_validation_dylora_rejects_activation_checkpointing():
    from rengu_flow.config.validation import collect_validation_errors

    cfg = _base_config({"type": "lycoris_dylora", "rank": 8})
    cfg["activation_checkpointing"] = True
    assert any("activation_checkpointing" in i for i in collect_validation_errors(cfg))


def test_train_norm_attaches_exports_and_round_trips(tmp_path):
    """train_norm adds NormModule on the blocks' LayerNorms: exported as w_norm/b_norm
    without alpha, loadable, and fusable with output parity."""
    cfg = _config("lycoris_locon")
    cfg["train_norm"] = True
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    kinds = [type(m).__name__ for _, m in lycoris_attach.iter_attached_adapters(unet)]
    assert kinds.count("NormModule") == 3  # one LayerNorm per unet block
    x16 = _train_one_step(unet, te, te2)
    with torch.no_grad():
        expected = unet.down_blocks(x16)
    snapshot = _snapshot(unet, te, te2)
    lycoris_sdxl.save(tmp_path, snapshot, cfg)

    import safetensors.torch

    state = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")
    norm_modules = {k.rsplit(".w_norm", 1)[0] for k in state if k.endswith(".w_norm")}
    assert norm_modules
    assert all(f"{m}.b_norm" in state for m in norm_modules)
    assert all(f"{m}.alpha" not in state for m in norm_modules)
    failures, _ = check_export(tmp_path / "adapter_model.safetensors", "lycoris_locon")
    assert not failures, failures

    cfg2 = _config("lycoris_locon")
    cfg2["train_norm"] = True
    unet2, te2_, te22 = _models()
    lycoris_sdxl.configure(unet2, te2_, te22, cfg2)
    lycoris_sdxl.load(_Pipe(unet2, te2_, te22), tmp_path)
    with torch.no_grad():
        assert torch.allclose(unet2.down_blocks(x16), expected, atol=1e-5)
    lycoris_sdxl.fuse(_Pipe(unet2, te2_, te22))
    with torch.no_grad():
        assert torch.allclose(unet2.down_blocks(x16), expected, atol=1e-4)


def test_rs_lora_scale_and_export_alpha(tmp_path):
    """rs_lora trains at alpha/sqrt(r) and exports alpha*sqrt(r) so loaders
    (scale = alpha/rank) reproduce the trained strength."""
    cfg = _config("lycoris_locon")
    cfg["rs_lora"] = True
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    _, lora = next(lycoris_attach.iter_attached_adapters(unet))
    rank = cfg["rank"]
    assert lora.rs_lora is True
    assert lora.scale == pytest.approx(rank / rank**0.5)
    assert float(lora.alpha) == pytest.approx(rank * rank**0.5)

    x16 = _train_one_step(unet, te, te2)
    with torch.no_grad():
        expected = unet.down_blocks(x16)
    lycoris_sdxl.save(tmp_path, _snapshot(unet, te, te2), cfg)

    import safetensors.torch

    state = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")
    alphas = [v for k, v in state.items() if k.endswith(".alpha")]
    assert all(float(a) == pytest.approx(rank * rank**0.5) for a in alphas)
    failures, _ = check_export(tmp_path / "adapter_model.safetensors", "lycoris_locon")
    assert not failures, failures

    cfg2 = _config("lycoris_locon")
    cfg2["rs_lora"] = True
    unet2, te2_, te22 = _models()
    lycoris_sdxl.configure(unet2, te2_, te22, cfg2)
    lycoris_sdxl.load(_Pipe(unet2, te2_, te22), tmp_path)
    with torch.no_grad():
        assert torch.allclose(unet2.down_blocks(x16), expected, atol=1e-5)


def test_target_include_exclude_filtering():
    cfg = _config("lycoris_loha")
    cfg["target_include"] = ["unet.mid_block*"]
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    attached = [p for m in (unet, te, te2) for p, _ in lycoris_attach.iter_attached_adapters(m)]
    assert len(attached) == 2  # only mid_block's two Linears

    cfg = _config("lycoris_loha")
    cfg["target_exclude"] = ["*.0"]  # first Linear of every Sequential
    unet, te, te2 = _models()
    lycoris_sdxl.configure(unet, te, te2, cfg)
    n = sum(1 for m in (unet, te, te2) for _ in lycoris_attach.iter_attached_adapters(m))
    assert n == 3  # unet blocks keep only their second Linear; TEs lose theirs

    cfg = _config("lycoris_loha")
    cfg["target_include"] = ["no_such_module*"]
    unet, te, te2 = _models()
    with pytest.raises(RuntimeError, match="attached 0 modules"):
        lycoris_sdxl.configure(unet, te, te2, cfg)


def test_validation_rs_lora_and_target_lists():
    issues = _adapter_issues({"type": "lycoris_loha", "rank": 8, "rs_lora": True})
    assert any("keys not supported by lycoris_loha" in i for i in issues)
    assert not _adapter_issues({"type": "lycoris_locon", "rank": 8, "rs_lora": True})
    assert not _adapter_issues({"type": "lycoris_locon", "rank": 8, "train_norm": True})
    issues = _adapter_issues({"type": "lycoris_locon", "rank": 8, "target_include": "attn"})
    assert any("non-empty list of glob patterns" in i for i in issues)
    issues = _adapter_issues({"type": "lycoris_locon", "rank": 8, "target_exclude": []})
    assert any("non-empty list of glob patterns" in i for i in issues)
