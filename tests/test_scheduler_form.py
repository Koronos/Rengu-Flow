"""LR scheduler form visibility, pruning, KV extras, and runtime tokens."""

from renga_flow.optim.resolver import build_scheduler_runtime_values, substitute_runtime_tokens
from renga_flow_ui.config_form import form_to_config, form_to_toml, parse_toml
from renga_flow_ui.config_schema import get_schema
from renga_flow_ui.field_visibility import field_visible
from renga_flow_ui.scheduler_form import (
    is_custom_scheduler_type,
    merge_scheduler_extras,
    prune_scheduler_form,
    split_scheduler_extras,
)


def test_is_custom_scheduler_type() -> None:
    assert not is_custom_scheduler_type("cosine")
    assert not is_custom_scheduler_type("Cosine")
    assert is_custom_scheduler_type("torch.optim.lr_scheduler.CosineAnnealingLR")


def test_schema_extra_params_help_includes_runtime_token_glossary() -> None:
    from renga_flow_ui import config_field_help
    from renga_flow_ui.config_schema import get_schema

    schema = get_schema()
    extra = next(
        f
        for s in schema["sections"]
        for f in s["fields"]
        if f["path"] == "lr_scheduler_args.extra_params"
    )
    detail = config_field_help.FIELD_HELP["lr_scheduler_args.extra_params"]["detail"]
    assert "total_steps" in detail
    assert "min(total_steps" in detail
    assert extra.get("help")
    assert "total_steps" in extra["help"]
    assert "runtime token" in extra["help"].lower()


def test_schema_extra_params_visible_for_builtin_cosine() -> None:
    schema = get_schema()
    extra = next(
        f
        for s in schema["sections"]
        for f in s["fields"]
        if f["path"] == "lr_scheduler_args.extra_params"
    )
    caps = schema["registries"]["model_capabilities"]
    assert extra["type"] == "key_value_list"
    assert extra.get("runtime_tokens")
    assert "runtime_token_hints" not in extra
    assert field_visible(extra, {"lr_scheduler": "cosine"}, caps)
    assert field_visible(
        extra,
        {"lr_scheduler": "torch.optim.lr_scheduler.StepLR"},
        caps,
    )


def test_schema_scheduler_has_dedicated_warmup_field() -> None:
    schema = get_schema()
    sched_sec = next(s for s in schema["sections"] if s["id"] == "scheduler")
    paths = {f["path"] for f in sched_sec["fields"]}
    assert paths == {"lr_scheduler", "warmup_steps", "lr_scheduler_args.extra_params"}
    warmup = next(f for f in sched_sec["fields"] if f["path"] == "warmup_steps")
    assert warmup["type"] == "integer"


def test_prune_scheduler_form_drops_orphan_flat_keys() -> None:
    form = {
        "lr_scheduler": "linear",
        "warmup_steps": 10,
        "lr_scheduler_args.lr_min": 0.01,
        "lr_scheduler_args.extra_params": {"start_factor": 1.0},
    }
    pruned = prune_scheduler_form(form)
    assert "lr_scheduler_args.lr_min" not in pruned
    assert pruned["warmup_steps"] == 10
    assert pruned["lr_scheduler_args.extra_params"]["start_factor"] == 1.0


def test_builtin_cosine_splits_into_kv() -> None:
    toml_in = """
dataset = "x.toml"
lr_scheduler = "cosine"
warmup_steps = 100
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
lr = 1e-4
[lr_scheduler_args]
lr_min = 0.01
"""
    form = parse_toml(toml_in)
    assert form["lr_scheduler"] == "cosine"
    assert "lr_scheduler_args.lr_min" not in form
    assert form["warmup_steps"] == 100
    assert form["lr_scheduler_args.extra_params"]["lr_min"] == 0.01
    assert "warmup_steps" not in form["lr_scheduler_args.extra_params"]
    cfg = form_to_config(form)
    assert cfg["warmup_steps"] == 100
    assert cfg["lr_scheduler_args"]["lr_min"] == 0.01


def test_custom_scheduler_extras_roundtrip() -> None:
    toml_in = """
dataset = "x.toml"
lr_scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
lr = 1e-4
[lr_scheduler_args]
T_max = "total_steps"
eta_min = 0.0
"""
    form = parse_toml(toml_in)
    assert form["lr_scheduler"] == "torch.optim.lr_scheduler.CosineAnnealingLR"
    assert "lr_scheduler_args.T_max" not in form
    assert form["lr_scheduler_args.extra_params"]["T_max"] == "total_steps"
    cfg = form_to_config(form)
    assert cfg["lr_scheduler_args"]["T_max"] == "total_steps"
    out = form_to_toml(form)
    form2 = parse_toml(out)
    assert form2["lr_scheduler_args.extra_params"]["T_max"] == "total_steps"


def test_linear_to_none_toml_omits_warmup() -> None:
    form = {
        "dataset": "d.toml",
        "model.type": "sdxl",
        "model.dtype": "bfloat16",
        "model.checkpoint_path": "/t",
        "_has_adapter": False,
        "optimizer.type": "adamw",
        "optimizer.extra_params": {"lr": 1e-4},
        "lr_scheduler": "none",
        "lr_scheduler_args.extra_params": {},
    }
    out = form_to_toml(form)
    assert "warmup_steps" not in out


def test_merge_scheduler_extras_legacy_warmup_in_kv() -> None:
    form = {
        "lr_scheduler": "cosine",
        "lr_scheduler_args.extra_params": {"warmup_steps": 50, "lr_min": 0.0},
    }
    merged = merge_scheduler_extras(form)
    assert merged["warmup_steps"] == 50
    assert merged["lr_scheduler_args.lr_min"] == 0.0
    assert "lr_scheduler_args.extra_params" not in merged
    assert "lr_scheduler_args.warmup_steps" not in merged


def test_split_scheduler_extras_migrates_legacy_warmup_from_kv() -> None:
    form = {
        "lr_scheduler": "cosine",
        "lr_scheduler_args.extra_params": {"warmup_steps": 25, "lr_min": 0.01},
    }
    split = split_scheduler_extras(form)
    assert split["warmup_steps"] == 25
    assert "warmup_steps" not in split["lr_scheduler_args.extra_params"]


def test_build_scheduler_runtime_values_effective_total_steps() -> None:
    config = {"epochs": 10, "max_steps": 50, "gradient_accumulation_steps": 4}
    values = build_scheduler_runtime_values(config, total_steps=1000, steps_per_epoch=100)
    assert values["total_steps"] == 1000
    assert values["effective_total_steps"] == 50
    assert values["max_steps"] == 50
    assert values["gradient_accumulation_steps"] == 4
    kwargs = {"T_max": "effective_total_steps"}
    substitute_runtime_tokens(kwargs, values)
    assert kwargs["T_max"] == 50


def test_schema_scheduler_importance_tiers() -> None:
    schema = get_schema()
    sched_sec = next(s for s in schema["sections"] if s["id"] == "scheduler")
    assert "flat_optional" not in sched_sec
    by_path = {f["path"]: f for f in sched_sec["fields"]}
    assert by_path["lr_scheduler"]["importance"] == "recommended"
    assert by_path["warmup_steps"]["importance"] == "advanced"
    assert by_path["lr_scheduler_args.extra_params"]["importance"] == "advanced"


def test_parse_cosine_without_args_yields_empty_kv() -> None:
    form = parse_toml(
        """
dataset = "x.toml"
lr_scheduler = "cosine"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
lr = 1e-4
"""
    )
    assert form["lr_scheduler_args.extra_params"] == {}
