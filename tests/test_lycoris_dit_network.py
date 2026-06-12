"""CPU tests for LyCORIS networks on DiT models (cosmos convention: dotted diffusion_model.* keys)."""

import pytest

try:
    import torch
    from torch import nn

    import lycoris  # noqa: F401
    from rengu_flow.networks import lycoris_attach, lycoris_dit
    from rengu_flow.networks.lycoris_export_check import check_export
    from rengu_flow.networks.lycoris_meta import apply_lycoris_defaults
except ImportError as e:  # pragma: no cover
    pytest.skip(f"Cannot import torch/lycoris networks: {e}", allow_module_level=True)

# The subset exposed for cosmos_predict2 (see model_capabilities).
COSMOS_TYPES = ("lycoris_locon", "lycoris_loha", "lycoris_lokr", "lycoris_dora")


class Block(nn.Module):  # class NAME is the adapter target match
    def __init__(self):
        super().__init__()
        self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(16, 16), "o_proj": nn.Linear(16, 16)})
        self.mlp = nn.ModuleDict({"up": nn.Linear(16, 32), "down": nn.Linear(32, 16)})

    def forward(self, x):
        x = self.self_attn["o_proj"](self.self_attn["q_proj"](x))
        return self.mlp["down"](self.mlp["up"](x))


class DummyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(16, 16)  # outside blocks: must NOT be adapted
        self.blocks = nn.ModuleList([Block(), Block()])
        self.final = nn.Linear(16, 16)

    def forward(self, x):
        x = self.patch_embed(x)
        for b in self.blocks:
            x = b(x)
        return self.final(x)


def _config(adapter_type):
    cfg = {"type": adapter_type, "rank": 4, "alpha": 4, "dtype": torch.float32}
    apply_lycoris_defaults(cfg)
    return cfg


def _train_one_step(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-1)
    x = torch.randn(2, 16)
    model(x).sum().backward()
    opt.step()
    return x


@pytest.mark.parametrize("adapter_type", COSMOS_TYPES)
def test_configure_targets_blocks_only(adapter_type):
    torch.manual_seed(0)
    model = DummyDiT()
    lycoris_dit.configure(model, _config(adapter_type))
    attached = [path for path, _ in lycoris_attach.iter_attached_adapters(model)]
    assert len(attached) == 8  # 4 Linears per block x 2 blocks
    assert all(path.startswith("blocks.") for path in attached)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    assert all(".lycoris_adapter." in p.original_name for p in trainable)
    # ungrouped Linears stay frozen and unadapted
    assert not model.patch_embed.weight.requires_grad
    assert not hasattr(model.patch_embed, "lycoris_adapter")


@pytest.mark.parametrize("adapter_type", COSMOS_TYPES)
def test_save_export_check_and_load_round_trip(adapter_type, tmp_path):
    torch.manual_seed(0)
    model = DummyDiT()
    lycoris_dit.configure(model, _config(adapter_type))
    x = _train_one_step(model)
    with torch.no_grad():
        expected = model(x)
    snapshot = {
        p.original_name: p.detach().clone() for p in model.parameters() if p.requires_grad
    }
    lycoris_dit.save(tmp_path, snapshot, _config(adapter_type))

    import safetensors.torch

    state = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")
    assert all(k.startswith("diffusion_model.blocks.") for k in state)
    assert not any("lycoris_adapter" in k for k in state)

    failures, n_modules = check_export(
        tmp_path / "adapter_model.safetensors", adapter_type, style="cosmos"
    )
    assert not failures, failures
    assert n_modules == 8

    torch.manual_seed(0)
    model2 = DummyDiT()
    lycoris_dit.configure(model2, _config(adapter_type))
    lycoris_dit.load(model2, tmp_path)
    with torch.no_grad():
        loaded = model2(x)
    assert torch.allclose(loaded, expected, atol=1e-5)


def test_dora_exports_dora_scale_cosmos(tmp_path):
    torch.manual_seed(0)
    model = DummyDiT()
    lycoris_dit.configure(model, _config("lycoris_dora"))
    _train_one_step(model)
    snapshot = {
        p.original_name: p.detach().clone() for p in model.parameters() if p.requires_grad
    }
    lycoris_dit.save(tmp_path, snapshot, _config("lycoris_dora"))

    import safetensors.torch

    state = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")
    modules = {k.rsplit(".dora_scale", 1)[0] for k in state if k.endswith(".dora_scale")}
    assert len(modules) == 8


def test_cosmos_export_check_rejects_kohya_style(tmp_path):
    """A kohya-flat SDXL file must not pass the cosmos-style check."""
    import safetensors.torch

    state = {
        "lora_unet_block.lora_up.weight": torch.randn(16, 4),
        "lora_unet_block.lora_down.weight": torch.randn(4, 16),
        "lora_unet_block.alpha": torch.tensor(4.0),
    }
    file = tmp_path / "adapter_model.safetensors"
    safetensors.torch.save_file(state, file)
    failures, _ = check_export(file, "lycoris_locon", style="cosmos")
    assert failures


def test_capabilities_expose_cosmos_lycoris():
    from rengu_flow.registry.model_capabilities import capabilities_for_api

    cosmos = capabilities_for_api()["cosmos_predict2"]
    for adapter_type in COSMOS_TYPES:
        assert adapter_type in cosmos["adapters"]
        assert cosmos["adapter_labels"][adapter_type].startswith("LyCORIS · ")
    assert "lycoris_dylora" not in cosmos["adapters"]
    assert "lycoris_boft" not in cosmos["adapters"]


def test_targeting_on_dit():
    torch.manual_seed(0)
    model = DummyDiT()
    cfg = _config("lycoris_locon")
    cfg["target_include"] = ["*q_proj*"]
    lycoris_dit.configure(model, cfg)
    attached = [p for p, _ in lycoris_attach.iter_attached_adapters(model)]
    assert len(attached) == 2
    assert all("q_proj" in p for p in attached)

    torch.manual_seed(0)
    model = DummyDiT()
    cfg = _config("lycoris_loha")
    cfg["target_exclude"] = ["*mlp*"]
    lycoris_dit.configure(model, cfg)
    attached = [p for p, _ in lycoris_attach.iter_attached_adapters(model)]
    assert len(attached) == 4
    assert not any("mlp" in p for p in attached)


def test_rs_lora_on_dit(tmp_path):
    torch.manual_seed(0)
    model = DummyDiT()
    cfg = _config("lycoris_dora")
    cfg["rs_lora"] = True
    lycoris_dit.configure(model, cfg)
    _, lora = next(lycoris_attach.iter_attached_adapters(model))
    assert lora.rs_lora is True
    assert float(lora.alpha) == pytest.approx(4 * 2.0)  # rank 4 -> alpha*sqrt(4)
    _train_one_step(model)
    snapshot = {
        p.original_name: p.detach().clone() for p in model.parameters() if p.requires_grad
    }
    lycoris_dit.save(tmp_path, snapshot, cfg)
    failures, _ = check_export(
        tmp_path / "adapter_model.safetensors", "lycoris_dora", style="cosmos"
    )
    assert not failures, failures

    import safetensors.torch

    state = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")
    alphas = [float(v) for k, v in state.items() if k.endswith(".alpha")]
    assert alphas and all(a == pytest.approx(8.0) for a in alphas)


def test_train_norm_rejected_on_dit_without_affine_norms():
    """The cosmos-style DiT has no affine LayerNorm/GroupNorm: train_norm must fail
    loudly instead of silently training nothing extra."""
    torch.manual_seed(0)
    model = DummyDiT()
    cfg = _config("lycoris_locon")
    cfg["train_norm"] = True
    with pytest.raises(RuntimeError, match="no trainable LayerNorm/GroupNorm"):
        lycoris_dit.configure(model, cfg)
