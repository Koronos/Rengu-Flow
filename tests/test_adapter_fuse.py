"""Tests for adapter load/fuse: infer_lokr_config_from_state, load_and_fuse_adapter errors."""

import pytest

# Networks may import torch/lycoris; skip if heavy deps fail.
try:
    from renga_flow.networks import lokr_sdxl
    import torch
except ImportError as e:
    pytest.skip(f"Cannot import networks/torch: {e}", allow_module_level=True)


def test_infer_lokr_config_from_state():
    """infer_lokr_config_from_state returns type lokr, rank from w1_b shape, defaults for rest."""
    state = {
        "unet.down_blocks.0.attentions.0.to_q.lokr_w1_b": torch.zeros(8, 64),
    }
    cfg = lokr_sdxl.infer_lokr_config_from_state(state)
    assert cfg["type"] == "lokr"
    assert cfg["rank"] == 8
    assert cfg["alpha"] == 8
    assert cfg["factor"] == -1
    assert cfg["decompose_both"] is False
    assert cfg["full_matrix"] is False
    assert "dtype" in cfg


def test_infer_lokr_config_from_state_uses_w2_a_if_no_w1_b():
    """Rank can be inferred from lokr_w2_a shape (out2, rank)."""
    state = {
        "text_encoder.encoder.layers.0.self_attn.lokr_w2_a": torch.zeros(32, 4),
    }
    cfg = lokr_sdxl.infer_lokr_config_from_state(state)
    assert cfg["rank"] == 4


def test_infer_lokr_config_from_state_default_rank_when_no_decomposed():
    """When only full matrices (lokr_w1, lokr_w2) exist, rank defaults to 4."""
    state = {
        "unet.down_blocks.0.attentions.0.to_q.lokr_w1": torch.zeros(10, 20),
    }
    cfg = lokr_sdxl.infer_lokr_config_from_state(state)
    assert cfg["rank"] == 4


def test_load_and_fuse_adapter_raises_when_no_safetensors(tmp_path):
    """load_and_fuse_adapter(path) raises RuntimeError when path has no .safetensors."""
    try:
        from renga_flow.config.defaults import set_config_defaults
        from renga_flow.model import sdxl  # noqa: F401  # register sdxl so get_model works
        from renga_flow.registry.models import get_model
    except ImportError as e:
        pytest.skip(f"Cannot import config/registry: {e}")

    config = {
        "dataset": "examples/minimal_dataset.toml",
        "model": {"type": "sdxl", "dtype": "float32", "checkpoint_path": "path/to/sdxl.safetensors"},
        "optimizer": {"type": "adamw", "lr": 1.0e-4},
    }
    set_config_defaults(config)
    model = get_model(config)
    with pytest.raises(RuntimeError, match="No .safetensors file found"):
        model.load_and_fuse_adapter(tmp_path)
