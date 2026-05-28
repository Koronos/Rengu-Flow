"""Optimizer form visibility, pruning, KV extras, and custom types."""

from renga_flow_ui.config_form import form_to_config, form_to_toml, form_values_for_ui, parse_toml
from renga_flow_ui.config_schema import get_schema
from renga_flow_ui.field_visibility import field_visible
from renga_flow_ui.optim_kv_defaults import OPTIMIZER_REGISTRY_KV_DEFAULTS
from renga_flow_ui.optimizer_form import (
    collect_optimizer_betas_validation_errors,
    is_custom_optimizer_type,
    merge_optimizer_extras,
    prune_optimizer_form,
    split_optimizer_extras,
)


def test_is_custom_optimizer_type() -> None:
    assert not is_custom_optimizer_type("adamw")
    assert not is_custom_optimizer_type("AdamW")
    assert not is_custom_optimizer_type("prodigy")
    assert not is_custom_optimizer_type("Prodigy")
    assert is_custom_optimizer_type("torch.optim.AdamW")
    assert is_custom_optimizer_type("pytorch_optimizer.Prodigy")


def test_schema_no_dedicated_optimizer_param_fields() -> None:
    schema = get_schema()
    opt_sec = next(s for s in schema["sections"] if s["id"] == "optimizer")
    paths = {f["path"] for f in opt_sec["fields"]}
    assert paths == {"optimizer.type", "optimizer.extra_params"}


def test_schema_extra_params_visible_for_adamw() -> None:
    schema = get_schema()
    extra = next(
        f
        for s in schema["sections"]
        for f in s["fields"]
        if f["path"] == "optimizer.extra_params"
    )
    caps = schema["registries"]["model_capabilities"]
    assert extra["type"] == "key_value_list"
    assert field_visible(extra, {"optimizer.type": "adamw"}, caps)
    assert field_visible(extra, {"optimizer.type": "prodigy"}, caps)
    assert field_visible(extra, {"optimizer.type": "genericoptim"}, caps)


def test_collect_optimizer_betas_validation_errors() -> None:
    base = {
        "dataset": "d.toml",
        "model": {"type": "sdxl", "dtype": "bfloat16", "checkpoint_path": "/t"},
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }
    assert collect_optimizer_betas_validation_errors(base) == []
    too_many = {
        **base,
        "optimizer": {**base["optimizer"], "betas": [0.9, 0.95, 0.99]},
    }
    issues = collect_optimizer_betas_validation_errors(too_many)
    assert len(issues) == 1
    assert "exactly two" in issues[0]
    one_beta = {**base, "optimizer": {**base["optimizer"], "betas": [0.9]}}
    assert len(collect_optimizer_betas_validation_errors(one_beta)) == 1
    sgd = {**base, "optimizer": {"type": "sgd", "lr": 1e-3, "betas": [0.9, 0.95, 0.99]}}
    assert collect_optimizer_betas_validation_errors(sgd) == []


def test_prune_optimizer_form_drops_orphan_flat_keys() -> None:
    form = {
        "optimizer.type": "sgd",
        "optimizer.lr": 1e-3,
        "optimizer.betas": [0.9, 0.999],
        "optimizer.extra_params": {"lr": 1e-3, "momentum": 0.9},
    }
    pruned = prune_optimizer_form(form)
    assert "optimizer.lr" not in pruned
    assert "optimizer.betas" not in pruned
    assert pruned["optimizer.extra_params"]["momentum"] == 0.9


def test_builtin_adamw_splits_into_kv() -> None:
    toml_in = """
dataset = "x.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
lr = 1.0e-4
betas = [0.9, 0.95]
"""
    form = parse_toml(toml_in)
    assert form["optimizer.type"] == "adamw"
    assert "optimizer.lr" not in form
    assert form["optimizer.extra_params"]["lr"] == 1.0e-4
    assert form["optimizer.extra_params"]["betas"] == [0.9, 0.95]


def test_form_values_for_ui_skips_hidden_defaults() -> None:
    schema = get_schema()
    form = parse_toml(
        """
dataset = "x.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "sgd"
lr = 0.01
"""
    )
    filled = form_values_for_ui(form, schema)
    assert "optimizer.betas" not in filled
    assert filled["optimizer.extra_params"]["lr"] == 0.01


def test_custom_optimizer_extras_roundtrip() -> None:
    toml_in = """
dataset = "x.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "pytorch_optimizer.Prodigy"
lr = 1.0
decouple = true
"""
    form = parse_toml(toml_in)
    assert form["optimizer.type"] == "pytorch_optimizer.Prodigy"
    assert "optimizer.decouple" not in form
    assert form["optimizer.extra_params"] == {"lr": 1.0, "decouple": True}
    out = form_to_toml(form)
    form2 = parse_toml(out)
    assert form2["optimizer.extra_params"]["decouple"] is True
    cfg = form_to_config(form)
    assert cfg["optimizer"]["decouple"] is True


def test_adamw_to_sgd_toml_omits_betas() -> None:
    form = {
        "dataset": "d.toml",
        "model.type": "sdxl",
        "model.dtype": "bfloat16",
        "model.checkpoint_path": "/t",
        "_has_adapter": False,
        "optimizer.type": "sgd",
        "optimizer.extra_params": OPTIMIZER_REGISTRY_KV_DEFAULTS["sgd"],
    }
    out = form_to_toml(form)
    assert "betas" not in out
    assert "momentum" in out


def test_merge_optimizer_extras() -> None:
    form = {
        "optimizer.type": "prodigy",
        "optimizer.extra_params": {"lr": 1.0, "weight_decouple": True},
    }
    merged = merge_optimizer_extras(form)
    assert merged["optimizer.weight_decouple"] is True
    assert merged["optimizer.lr"] == 1.0
    assert "optimizer.extra_params" not in merged


def test_prodigy_builtin_kv_defaults() -> None:
    kv = OPTIMIZER_REGISTRY_KV_DEFAULTS["prodigy"]
    assert kv["lr"] == 1.0
    assert kv["d_coef"] == 1.0
    assert kv["safeguard_warmup"] is True
