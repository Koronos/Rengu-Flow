"""Tests for UI config form TOML conversion."""

from rengu_flow.registry.model_capabilities import get_capability, normalize_model_type
from rengu_flow_ui.config_form import (
    form_to_toml,
    form_values_for_ui,
    merge_form_into_config,
    parse_toml,
)
from rengu_flow_ui.config_schema import get_registries, get_schema


def test_schema_registries() -> None:
    reg = get_registries()
    assert "sdxl" in reg["models"]
    assert "cosmos_predict2" in reg["models"]
    assert "anima" not in reg["models"]
    assert "lora" in reg["model_capabilities"]["sdxl"]["adapters"]
    assert "adamw" in reg["optimizers"]
    assert "cosine" in reg["schedulers"]


def test_anima_alias_resolves_to_cosmos_capability() -> None:
    assert normalize_model_type("anima") == "cosmos_predict2"
    assert get_capability("anima") is not None
    assert get_capability("anima").type_id == "cosmos_predict2"
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
    assert form["model.type"] == "cosmos_predict2"


def test_schema_training_core_importance() -> None:
    # Looked up across all sections, not within "training": the 35-field Training loop was
    # split into 7 focused sections (94073a9), which moved logging_steps and
    # synthetic_num_batches to logging_debug. What is asserted here is the importance
    # tagging, not which section a knob currently lives in.
    schema = get_schema()
    fields = {f["path"]: f for sec in schema["sections"] for f in sec["fields"]}
    assert fields["epochs"]["importance"] == "recommended"
    assert fields["micro_batch_size_per_gpu"]["importance"] == "recommended"
    assert fields["gradient_accumulation_steps"]["importance"] == "recommended"
    assert fields["logging_steps"]["importance"] == "advanced"
    assert fields["synthetic_num_batches"]["importance"] == "advanced"


def test_schema_training_tab_importance_not_over_tagged() -> None:
    """Training tab: only a few knobs are Important; optimizer extras stay visible."""
    schema = get_schema()
    training_ids = {"optimizer", "scheduler", "training", "checkpoint"}
    recommended = []
    for sec in schema["sections"]:
        if sec["id"] not in training_ids:
            continue
        for field in sec["fields"]:
            if field.get("importance") == "recommended":
                recommended.append(field["path"])
    assert set(recommended) == {
        "lr_scheduler",
        "epochs",
        "micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
    }


def test_schema_scheduler_section_has_extra_params() -> None:
    schema = get_schema()
    sched_sec = next(s for s in schema["sections"] if s["id"] == "scheduler")
    paths = {f["path"] for f in sched_sec["fields"]}
    assert "lr_scheduler_args.extra_params" in paths


def test_schema_optimizer_extra_params_advanced() -> None:
    schema = get_schema()
    opt_sec = next(s for s in schema["sections"] if s["id"] == "optimizer")
    paths = {f["path"] for f in opt_sec["fields"]}
    by_path = {f["path"]: f for f in opt_sec["fields"]}
    assert paths == {"optimizer.type", "optimizer.extra_params"}
    assert by_path["optimizer.type"]["importance"] == "required"
    assert by_path["optimizer.extra_params"]["importance"] == "advanced"
    assert "flat_optional" not in opt_sec


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
    # Component paths exist once per model type (scoped by `when`); pick the cosmos variants.
    paths = {
        f["path"]: f
        for s in schema["sections"]
        for f in s["fields"]
        if (f.get("when") or {}).get("in") == ["cosmos_predict2"] or not f.get("when")
    }
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
    tracking_sec = next(s for s in schema["sections"] if s["id"] == "tracking")
    assert "flat_optional" not in eval_sec
    assert "flat_optional" not in tracking_sec


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
    assert form["optimizer.extra_params"]["betas"] == [0.9, 0.95]
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
    from rengu_flow_ui import config_field_help

    assert "keep_exports_from_step" in config_field_help.FIELD_HELP["max_model_exports_to_keep"]["detail"].lower()


def test_scheduler_warmup_field_help_documents_trainer_wrap() -> None:
    from rengu_flow_ui import config_field_help

    kv_detail = config_field_help.FIELD_HELP["lr_scheduler_args.extra_params"]["detail"]
    warmup_detail = config_field_help.FIELD_HELP["warmup_steps"]["detail"]
    assert "warmup steps" in kv_detail.lower()
    assert "total_steps" in kv_detail
    assert "runtime token" in kv_detail.lower()
    assert "list below" not in kv_detail.lower()
    assert "trainer" in warmup_detail.lower() or "wrap" in warmup_detail.lower()
    assert "constructor" in warmup_detail.lower()


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


def test_preview_prompts_toml_roundtrip() -> None:
    # A single mixed array of a plain-string prompt and a detailed table prompt — the
    # standards-compliant way to mix the two (this is also what the UI editor produces).
    toml_in = """
dataset = "x.toml"
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
[preview]
enabled = true
prompts = [
  "a cat on a mat",
  { name = "portrait", prompt = "1woman, soft light", seed = 42, preview_every_n_steps = 500 },
]
"""
    form = parse_toml(toml_in)
    assert form["preview.prompts"][0] == "a cat on a mat"
    assert form["preview.prompts"][1]["name"] == "portrait"
    out = form_to_toml(form)
    form2 = parse_toml(out)
    assert len(form2["preview.prompts"]) == 2
    first = form2["preview.prompts"][0]
    assert first == "a cat on a mat" or first.get("prompt") == "a cat on a mat"
    assert form2["preview.prompts"][1]["seed"] == 42


def test_schema_preview_section() -> None:
    schema = get_schema()
    preview = next(s for s in schema["sections"] if s["id"] == "preview")
    assert "flat_optional" not in preview
    paths = {f["path"] for f in preview["fields"]}
    assert "preview.prompts" in paths
    prompts_field = next(f for f in preview["fields"] if f["path"] == "preview.prompts")
    assert prompts_field["type"] == "preview_entries"
    assert schema["registries"].get("preview_entry_fields")


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


def test_form_to_toml_homogenizes_mixed_numeric_arrays() -> None:
    """A betas array like [0.0, 0.999] collapses to [0, 0.999] over JSON transport; the
    rendered TOML must promote it back to all-float so the strict `toml` loader accepts it."""
    import toml

    form = {"optimizer.type": "adakaon", "optimizer.extra_params": {"betas": [0, 0.999]}}
    rendered = form_to_toml(form)
    # Round-trips through the strict loader (no "Not a homogeneous array").
    loaded = toml.loads(rendered)
    assert loaded["optimizer"]["betas"] == [0.0, 0.999]
    assert all(isinstance(x, float) for x in loaded["optimizer"]["betas"])


def test_form_to_toml_keeps_integer_arrays_integer() -> None:
    """Pure-int arrays (e.g. resolutions) must stay int — only mixed arrays are promoted."""
    import toml

    rendered = form_to_toml({"resolutions": [512, 768]})
    loaded = toml.loads(rendered)
    assert loaded["resolutions"] == [512, 768]
    assert all(isinstance(x, int) for x in loaded["resolutions"])


def test_parse_toml_tolerates_legacy_mixed_arrays() -> None:
    """Configs saved before arrays were homogenized contain `betas = [0, 0.999]`, which the
    strict `toml` lib rejects. Seeding/editing such a config must still load (via tomlkit)."""
    legacy = """
epochs = 5
[optimizer]
type = "adakaon"
betas = [0, 0.999]
"""
    form = parse_toml(legacy)
    assert form["optimizer.type"] == "adakaon"
    # Re-rendering repairs it to a homogeneous, strictly-loadable array.
    import toml

    rendered = form_to_toml(form, legacy)
    assert toml.loads(rendered)["optimizer"]["betas"] == [0.0, 0.999]


def test_parse_keeps_micro_batch_resolution_map() -> None:
    """The per-resolution micro batch dict must round-trip the form unchanged
    (TOML keys stay strings here; set_config_defaults normalizes them to int)."""
    toml_in = """
dataset = "x.toml"
micro_batch_size_per_gpu = { 512 = 2, 1024 = 1 }
[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "/t"
[optimizer]
type = "adamw"
"""
    form = parse_toml(toml_in)
    assert form["micro_batch_size_per_gpu"] == {"512": 2, "1024": 1}
    out = form_to_toml(form)
    reparsed = parse_toml(out)
    assert reparsed["micro_batch_size_per_gpu"] == {"512": 2, "1024": 1}
