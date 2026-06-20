"""Adapter (network) form pruning: switching adapter.type drops stale keys."""

from rengu_flow_ui.adapter_form import allowed_adapter_paths, prune_adapter_form
from rengu_flow_ui.config_form import form_to_toml, parse_toml


def test_allowed_paths_per_type() -> None:
    lokr = allowed_adapter_paths("lokr")
    assert "adapter.factor" in lokr
    assert "adapter.rank" in lokr  # common
    assert "adapter.type" in lokr
    lora = allowed_adapter_paths("lora")
    assert "adapter.factor" not in lora  # lokr-only
    assert "adapter.rank" in lora


def test_prune_drops_previous_type_keys() -> None:
    form = {
        "adapter.type": "lora",
        "adapter.rank": 16,
        "adapter.dtype": "bfloat16",
        "adapter.factor": 8,  # stale lokr key
        "adapter.decompose_both": True,  # stale lokr key
    }
    out = prune_adapter_form(form)
    assert "adapter.factor" not in out
    assert "adapter.decompose_both" not in out
    assert out["adapter.rank"] == 16
    assert out["adapter.dtype"] == "bfloat16"


def test_prune_between_lycoris_types_keeps_shared_keys() -> None:
    form = {
        "adapter.type": "lycoris_locon",
        "adapter.rank": 32,
        "adapter.dropout": 0.1,
        "adapter.target_include": ["*attn*"],
        "adapter.factor": -1,  # lokr-only, stale
        "adapter.full_matrix": True,  # lokr-only, stale
    }
    out = prune_adapter_form(form)
    assert "adapter.factor" not in out
    assert "adapter.full_matrix" not in out
    assert out["adapter.dropout"] == 0.1
    assert out["adapter.target_include"] == ["*attn*"]  # lycoris-shared, kept
    assert out["adapter.rank"] == 32


def test_prune_never_touches_other_sections() -> None:
    form = {
        "adapter.type": "lora",
        "adapter.factor": 8,  # stale, dropped
        "model.type": "sdxl",
        "optimizer.type": "adamw",
        "optimizer.lr": 1e-4,
        "lr_scheduler": "cosine",
    }
    out = prune_adapter_form(form)
    assert "adapter.factor" not in out
    assert out["model.type"] == "sdxl"
    assert out["optimizer.type"] == "adamw"
    assert out["optimizer.lr"] == 1e-4
    assert out["lr_scheduler"] == "cosine"


def test_unknown_type_left_untouched() -> None:
    form = {"adapter.type": "mystery", "adapter.whatever": 1}
    assert prune_adapter_form(form) == form


def test_form_to_toml_strips_stale_adapter_keys() -> None:
    form = {
        "dataset": "d.toml",
        "model.type": "sdxl",
        "model.dtype": "bfloat16",
        "_has_adapter": True,
        "adapter.type": "lora",
        "adapter.rank": 16,
        "adapter.factor": 8,  # stale lokr key must not reach the TOML
    }
    out = form_to_toml(form)
    config = parse_toml(out)
    assert config["adapter.type"] == "lora"
    assert "adapter.factor" not in config
