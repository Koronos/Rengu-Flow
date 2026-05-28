"""Tests for UI config form TOML conversion."""

from renga_flow.registry.model_capabilities import get_capability, normalize_model_type
from renga_flow_ui.config_form import (
    form_to_toml,
    form_values_for_ui,
    merge_form_into_config,
    parse_toml,
)
from renga_flow_ui.config_schema import get_registries, get_schema


def test_schema_registries() -> None:
    reg = get_registries()
    assert "sdxl" in reg["models"]
    assert "cosmos_predict2" in reg["models"]
    assert "anima" not in reg["models"]
    assert "lora" in reg["model_capabilities"]["sdxl"]["adapters"]
    assert "adamw" in reg["optimizers"]
    assert "cosine" in reg["schedulers"]


def test_unknown_model_type_has_no_capability() -> None:
    assert get_capability("anima") is None
    assert normalize_model_type("anima") == "anima"
    form = parse_toml(
        """
dataset = "x.toml"
[model]
type = "anima"
dtype = "bfloat16"
transformer_path = "/t"
vae_path = "/v"
"""
    )
    assert form["model.type"] == "anima"


def test_schema_training_core_importance() -> None:
    schema = get_schema()
    training = next(s for s in schema["sections"] if s["id"] == "training")
    epochs = next(f for f in training["fields"] if f["path"] == "epochs")
    micro = next(f for f in training["fields"] if f["path"] == "micro_batch_size_per_gpu")
    synthetic = next(f for f in training["fields"] if f["path"] == "synthetic_num_batches")
    assert epochs["importance"] == "recommended"
    assert micro["importance"] == "recommended"
    assert synthetic["importance"] == "advanced"


def test_schema_optimizer_scheduler_allow_custom() -> None:
    schema = get_schema()
    opt = next(
        f
        for s in schema["sections"]
        for f in s["fields"]
        if f["path"] == "optimizer.type"
    )
    sched = next(
        f
        for s in schema["sections"]
        for f in s["fields"]
        if f["path"] == "lr_scheduler"
    )
    assert opt.get("allow_custom") is True
    assert sched.get("allow_custom") is True
    assert schema["registries"].get("optimizer_allow_custom") is True


def test_schema_field_help() -> None:
    schema = get_schema()
    dataset = next(
        f
        for s in schema["sections"]
        for f in s["fields"]
        if f["path"] == "dataset"
    )
    assert dataset.get("help")
    assert "dataset-config" in dataset.get("doc_path", "")


def test_schema_train_seed_field() -> None:
    schema = get_schema()
    training = next(s for s in schema["sections"] if s["id"] == "training")
    seed_field = next(f for f in training["fields"] if f["path"] == "train_seed")
    assert seed_field["type"] == "integer"
    assert seed_field.get("default") == 42
    assert seed_field.get("help")
    assert "training-loop-and-eval" in seed_field.get("doc_path", "")


def test_schema_cosmos_model_paths_help() -> None:
    """Cosmos weight paths use plain-language help (not empty or label-only)."""
    schema = get_schema()
    paths = {f["path"]: f for s in schema["sections"] for f in s["fields"]}
    transformer = paths["model.transformer_path"]
    assert "safetensors" in transformer.get("help", "").lower()
    assert "vae" in transformer.get("help", "").lower() or "VAE" in transformer.get("help", "")
    vae = paths["model.vae_path"]
    assert "latent" in vae.get("help", "").lower() or "pixel" in vae.get("help", "").lower()


def test_schema_sections_nonempty() -> None:
    schema = get_schema()
    assert len(schema["sections"]) >= 8
    paths = [f["path"] for s in schema["sections"] for f in s["fields"]]
    assert "model.type" in paths
    assert "optimizer.type" in paths
    adapter_sec = next(s for s in schema["sections"] if s["id"] == "adapter")
    adapter_type = next(f for f in adapter_sec["fields"] if f["path"] == "adapter.type")
    assert adapter_type.get("options_from_model") is True
    eval_sec = next(s for s in schema["sections"] if s["id"] == "eval")
    assert eval_sec.get("flat_optional") is True
    mon_sec = next(s for s in schema["sections"] if s["id"] == "monitoring")
    assert mon_sec.get("flat_optional") is True


def test_all_config_fields_have_help() -> None:
    schema = get_schema()
    missing = []
    for sec in schema["sections"]:
        for field in sec["fields"]:
            path = field.get("path")
            if not path:
                continue
            if not field.get("help"):
                missing.append(path)
    assert missing == [], f"Fields without help: {missing}"


def test_parse_keeps_optimizer_betas_as_list() -> None:
    toml_in = """
dataset = "x.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
betas = [0.9, 0.95]
"""
    form = parse_toml(toml_in)
    assert form["optimizer.betas"] == [0.9, 0.95]
    out = form_to_toml(form)
    assert "[0.9, 0.95]" in out or "0.9" in out


def test_form_values_for_ui_fills_schema_defaults() -> None:
    schema = get_schema()
    form = parse_toml(
        """
dataset = "x.toml"
[model]
type = "cosmos_predict2"
dtype = "bfloat16"
transformer_path = "/t"
vae_path = "/v"
llm_path = "/l"
[optimizer]
type = "adamw"
"""
    )
    assert "model.cache_text_embeddings" not in form
    filled = form_values_for_ui(form, schema)
    assert filled["model.cache_text_embeddings"] is True
    assert filled["eval_before_first_step"] is True
    assert filled["optimizer.lr"] == 1e-4
    assert filled["optimizer.betas"] == [0.9, 0.999]
    assert filled["epochs"] == 1


def test_toml_roundtrip_without_adapter_strips_section() -> None:
    toml_in = """
dataset = "examples/minimal_dataset.toml"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"

[optimizer]
type = "adamw"
"""
    form = parse_toml(toml_in)
    assert form["_has_adapter"] is False
    out = form_to_toml(form)
    assert "[adapter]" not in out
    assert "adapter" not in parse_toml(out)


def test_toml_roundtrip_minimal() -> None:
    toml_in = """
dataset = "examples/minimal_dataset.toml"
output_dir = "output"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/tmp/x.safetensors"

[adapter]
type = "lora"
rank = 8

[optimizer]
type = "adamw"
lr = 0.0001

epochs = 2
"""
    form = parse_toml(toml_in)
    assert form["_has_adapter"] is True
    assert form["model.type"] == "sdxl"
    assert form["adapter.rank"] == 8
    out = form_to_toml(form)
    form2 = parse_toml(out)
    assert form2["model.type"] == "sdxl"
    assert form2["adapter.rank"] == 8


def test_form_to_toml_dataset_list_roundtrip() -> None:
    form = parse_toml(
        """
dataset = ["a.toml", "b.toml"]
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
"""
    )
    assert form["dataset"] == ["a.toml", "b.toml"]
    out = form_to_toml(form)
    form2 = parse_toml(out)
    assert form2["dataset"] == ["a.toml", "b.toml"]


def test_form_to_toml_single_dataset_stays_string() -> None:
    form = {"dataset": "only.toml", "model.type": "sdxl", "model.dtype": "bfloat16", "_has_adapter": False}
    out = form_to_toml(form)
    assert 'dataset = "only.toml"' in out or "dataset = 'only.toml'" in out


def test_schema_checkpoint_export_retention_fields() -> None:
    schema = get_schema()
    checkpoint = next(s for s in schema["sections"] if s["id"] == "checkpoint")
    paths = {f["path"] for f in checkpoint["fields"]}
    assert "max_model_exports_to_keep" in paths
    assert "keep_exports_from_step" in paths
    from renga_flow_ui import config_field_help

    assert "keep_exports_from_step" in config_field_help.FIELD_HELP["max_model_exports_to_keep"]["detail"].lower()


def test_form_to_toml_with_base_preserves_orphan_keys() -> None:
    """Rendering from a sparse form must not drop keys that exist only in the base TOML."""
    base_toml = """
dataset = "d.toml"
lr_scheduler = "cosine"
[lr_scheduler_args]
lr_min = 0.0
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/weights/sdxl.safetensors"
[optimizer]
type = "adamw"
"""
    sparse_form = parse_toml(
        """
dataset = "d.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
[optimizer]
type = "adamw"
"""
    )
    out = form_to_toml(sparse_form, base_content=base_toml)
    assert "lr_scheduler" in out
    assert "checkpoint_path" in out
    assert "lr_min" in out


def test_merge_form_into_config_drops_adapter_when_disabled() -> None:
    base = {
        "dataset": "d.toml",
        "adapter": {"type": "lora", "rank": 8},
        "model": {"type": "sdxl", "dtype": "bfloat16", "checkpoint_path": "/x"},
    }
    form = parse_toml(
        """
dataset = "d.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/x"
[optimizer]
type = "adamw"
"""
    )
    assert form["_has_adapter"] is False
    merged = merge_form_into_config(base, form)
    assert "adapter" not in merged
    assert merged["model"]["checkpoint_path"] == "/x"
