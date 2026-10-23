"""Contract A: every registered model spec against everything derived from it.

Parametrized over the registered canonical types that name a pipeline (``registered_specs()``),
so a new model is covered by registering its spec. Spec-side checks: internal consistency, the
kit's declarations, docs / examples / notices / package data / README row / ``.env.example``
block, config validation and defaults, env overrides, the installer, the cache's ragged keys,
preview dispatch, and the web form (fields visible only on their model's form with help and a
doc link, preview defaults equal to the trainer's, every key the model reads shown on its form).
The "spec <-> pipeline class" checks import the pipeline (skipped when its extra is missing).

Each check is a plain function of ``cap`` (plus pytest fixtures) so the synthetic-model suite
(``test_synthetic_model_integration.py``) runs the same checks on a toy spec.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import re
from pathlib import Path

import pytest
import toml

from rengu_flow.registry import builtin_models
from rengu_flow.registry.builtin_models import builtin_specs
from rengu_flow.registry.model_capabilities import model_capability_registry
from rengu_flow.registry.model_config_rules import one_of_groups, required_model_keys

pytestmark = pytest.mark.no_ui_db

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "rengu_flow"
EXAMPLES = REPO_ROOT / "examples"


def registered_specs() -> list:
    """The registered canonical model specs (those naming a pipeline), in registration order."""
    return [cap for cap in model_capability_registry.values() if cap.pipeline]


SPECS = registered_specs()
spec_params = pytest.mark.parametrize("cap", SPECS, ids=[cap.type_id for cap in SPECS])


def _field(cap, key: str) -> dict | None:
    return next((f for f in cap.model_fields if f["path"] == f"model.{key}"), None)


def _field_keys(cap) -> list[str]:
    return [f["path"].removeprefix("model.") for f in cap.model_fields if f["path"].startswith("model.")]


def _pipeline_class(cap):
    module, _, class_name = cap.pipeline.partition(":")
    try:
        return getattr(importlib.import_module(module), class_name)
    except ModuleNotFoundError as e:  # a model extra (transformers, ...) not installed
        if e.name and e.name.split(".")[0] == "rengu_flow":
            raise
        pytest.skip(f"{cap.type_id} pipeline needs {e.name}")


def minimal_model_section(cap) -> dict:
    """The smallest ``[model]`` the validator accepts for ``cap``: every required key and the
    first member of every ``one_of`` group, as placeholder paths."""
    model = {"type": cap.type_id, "dtype": "bfloat16"}
    for key in required_model_keys(cap):
        model[key] = f"path/to/{key}"
    for group in one_of_groups(cap):
        if not any(k in model for k in group):
            model[group[0]] = f"path/to/{group[0]}"
    return model


def training_config(model: dict, **top) -> dict:
    return {
        "dataset": "examples/minimal_dataset.toml",
        "model": model,
        "optimizer": {"type": "adamw", "lr": 1e-4},
        **top,
    }


def _validation_errors(config: dict) -> list[str]:
    from rengu_flow.config.validation import collect_validation_errors

    return collect_validation_errors(copy.deepcopy(config))


def _defaults(model: dict) -> dict:
    return _defaults_with_preview(model, {})


def _defaults_with_preview(model: dict, preview: dict) -> dict:
    from rengu_flow.config.defaults import set_config_defaults

    config = {"model": dict(model), "preview": preview, "dataset": {}}
    config.update(micro_batch_size_per_gpu=1, gradient_accumulation_steps=1, epochs=1)
    set_config_defaults(config)
    return config


def examples_for(type_id: str) -> list[Path]:
    """The training examples whose ``[model] type`` is ``type_id``."""
    out = []
    for path in sorted(EXAMPLES.glob("*.toml")):
        model = toml.load(path).get("model")
        if isinstance(model, dict) and str(model.get("type", "")).lower() == type_id:
            out.append(path)
    return out


# ---- registration -------------------------------------------------------------------------------


def test_builtin_specs_are_registered_in_order():
    specs = builtin_specs()
    assert len(specs) == len(builtin_models.BUILTIN_MODEL_SPECS)
    registered = list(model_capability_registry)
    assert registered[: len(specs)] == [s.type_id for s in specs]
    for spec in specs:
        assert model_capability_registry[spec.type_id] is spec


def test_env_prefixes_unique():
    prefixes = [cap.env_prefix for cap in registered_specs()]
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*", p) for p in prefixes), prefixes
    assert len(set(prefixes)) == len(prefixes), prefixes


@spec_params
def test_spec_internal_consistency(cap):
    fields = _field_keys(cap)
    keys = [c.key for c in cap.components]
    assert keys and len(set(keys)) == len(keys)
    assert set(keys) <= set(fields), set(keys) - set(fields)
    assert all(c.expect in ("file", "dir", "any") for c in cap.components)
    if cap.bundle_key is not None:
        assert cap.bundle_key in keys
        # one_of(<component>, <bundle>) for every other component.
        expected = [[k, cap.bundle_key] for k in keys if k != cap.bundle_key]
        assert cap.model_validation.get("one_of") == expected
    assert cap.text_cache in ("optional", "required")
    assert cap.user_doc and (REPO_ROOT / cap.user_doc).is_file(), cap.user_doc
    # preview / features["preview"] derive from preview_spec.
    assert cap.preview == (cap.preview_spec is not None)
    assert cap.features.get("preview", False) == (cap.preview_spec is not None)
    if cap.preview_spec is not None:
        assert cap.preview_spec.runner in ("sdxl", "dit")
        assert cap.preview_spec.memory_knobs <= {"offload_text_encoder", "blocks_to_swap", "offload_dit_for_decode"}
        assert cap.preview_spec.edit == bool(cap.features.get("edit"))
    for key, default in cap.model_defaults.items():
        field = _field(cap, key)
        assert field is not None and field.get("default") == default, key
    if cap.quant is not None:
        assert cap.quant.fp8 in ("tensorwise", "scaled_matmul", None)
        assert (_field(cap, "transformer_4bit") is not None) == cap.quant.nf4
        assert (_field(cap, "transformer_fp8_matmul") is not None) == (cap.quant.fp8 is not None)
        assert (_field(cap, "fp8_grad_mode") is not None) == cap.quant.grad_mode
        dtype_field = _field(cap, "fp8_matmul_dtype")
        if cap.quant.fp8_dtypes:
            assert dtype_field["options"] == list(cap.quant.fp8_dtypes)
            assert dtype_field["default"] == cap.quant.fp8_dtypes[0]
        else:
            assert dtype_field is None
    else:
        assert not {"transformer_4bit", "transformer_fp8_matmul", "fp8_grad_mode"} & set(fields)
    assert cap.diffusion_dtype_sets_load_dtype in (True, False)
    assert cap.testkit and importlib.util.find_spec(cap.testkit) is not None, "every model ships a test kit"


@spec_params
def test_kit_declarations_are_consistent(cap):
    """The kit says which contract checks do not apply and why; its scope exclusions name
    adapters the model offers. Kit modules import no torch at module level."""
    from rengu_flow.model.testkit_common import ALL_CAPABILITIES

    kit = importlib.import_module(cap.testkit)
    not_applicable = getattr(kit, "NOT_APPLICABLE", None)
    assert isinstance(not_applicable, dict), f"{cap.testkit} declares no NOT_APPLICABLE"
    assert set(not_applicable) <= ALL_CAPABILITIES, set(not_applicable) - ALL_CAPABILITIES
    assert all(isinstance(r, str) and r.strip() for r in not_applicable.values())
    assert set(kit.CAPABILITIES) == ALL_CAPABILITIES - set(not_applicable)
    assert set(getattr(kit, "ADAPTER_SCOPE_EXCLUDES", {})) <= set(cap.adapters)
    assert all(isinstance(r, str) for r in getattr(kit, "KNOWN_GAPS", {}).values())
    for name in ("tiny_config", "tiny_pipeline", "monolithic_forward", "cases"):
        assert callable(getattr(kit, name, None)), f"{cap.testkit}.{name}"
    if "preview" in kit.CAPABILITIES or "dit_preview" in kit.CAPABILITIES:
        assert callable(getattr(kit, "preview_config", None))
    if "save_round_trip" in kit.CAPABILITIES:
        assert callable(getattr(kit, "reload_export", None)) or callable(getattr(kit, "export_transformer_path", None))


@spec_params
def test_pipeline_resolves_to_registered_class(cap):
    from rengu_flow.registry.models import (
        _ensure_model_imported,
        _model_modules,
        canonical_model_types,
        get_model,
        model_registry,
    )

    assert re.fullmatch(r"[\w.]+:\w+", cap.pipeline), cap.pipeline
    cls = _pipeline_class(cap)
    _ensure_model_imported(cap.type_id)  # the lazy path get_model takes, from the spec alone
    assert cap.type_id in model_registry
    assert cls.name == cap.type_id
    assert cap.type_id in canonical_model_types
    for alias in cap.aliases:
        assert model_registry[alias] is model_registry[cap.type_id]
    for key in (cap.type_id, *cap.aliases):
        assert _model_modules()[key] == cap.pipeline_module
    with pytest.raises(ValueError, match=cap.type_id):  # the "Known:" list derives from the specs
        get_model({"model": {"type": "no_such_model_type"}})


@spec_params
def test_export_prefix_matches_pipeline(cap):
    cls = _pipeline_class(cap)
    if not hasattr(cls, "adapter_export_prefix"):
        # Not a DiTPipeline: exports with its own per-adapter layout (sdxl).
        assert cap.export_prefix is None
        return
    from rengu_flow.networks import adapter_dit

    default = inspect.signature(adapter_dit.save).parameters["export_prefix"].default
    assert cap.export_prefix == (cls.adapter_export_prefix or default)


# ---- docs, examples, packaging --------------------------------------------------------------------


def _spec_package_dir(cap) -> Path | None:
    """``rengu_flow/model/<m>/`` when the spec lives in a model package (``<m>/spec.py``)."""
    module = next((m for m in builtin_models.BUILTIN_MODEL_SPECS if importlib.import_module(m).SPEC is cap), None)
    if module is None or not module.endswith(".spec"):
        return None
    return Path(importlib.util.find_spec(module).origin).parent


@spec_params
def test_examples_exist_validate_and_need_their_required_keys(cap):
    """Each model ships training examples (an adapter one when it offers adapters, a finetune
    one when it can full-finetune) that validate with their dataset in the repo; removing a
    required ``[model]`` key or a whole ``one_of`` group from one fails validation."""
    paths = examples_for(cap.type_id)
    assert paths, f"no examples/*.toml with [model] type = {cap.type_id!r}"
    configs = {p.name: toml.load(p) for p in paths}
    if cap.adapters:
        assert any(c.get("adapter") for c in configs.values()), "no adapter example"
    if cap.full_finetune:
        assert any(not c.get("adapter") for c in configs.values()), "no full-finetune example"
    for name, config in configs.items():
        assert not _validation_errors(config), (name, _validation_errors(config))
        dataset = config.get("dataset")
        if isinstance(dataset, str) and not dataset.startswith("rengu-flow-dataset:"):
            assert (REPO_ROOT / dataset).is_file(), (name, dataset)
        model = config["model"]
        for key in [k for k in required_model_keys(cap) if k in model]:
            broken = copy.deepcopy(config)
            del broken["model"][key]
            assert _validation_errors(broken), f"{name} still validates without model.{key}"
        for group in one_of_groups(cap):
            broken = copy.deepcopy(config)
            for key in group:
                broken["model"].pop(key, None)
            assert _validation_errors(broken), f"{name} still validates without any of {group}"


@spec_params
def test_validation_derives_from_required_keys_and_one_of_groups(cap):
    """The minimal section passes; dropping a required key or a whole ``one_of`` group fails
    naming the key(s); any single member of a group satisfies it; a bundle folder alone (or
    with every component on top) satisfies every group."""
    model = minimal_model_section(cap)
    assert not _validation_errors(training_config(model)), _validation_errors(training_config(model))
    for key in required_model_keys(cap):
        broken = {k: v for k, v in model.items() if k != key}
        assert any(key in e for e in _validation_errors(training_config(broken))), key
    for group in one_of_groups(cap):
        broken = {k: v for k, v in model.items() if k not in group}
        errors = _validation_errors(training_config(broken))
        assert any(all(key in e for key in group) for e in errors), (group, errors)
        for member in group:
            only = {**broken, member: f"path/to/{member}"}
            assert not _validation_errors(training_config(only)), (group, member)
    if cap.bundle_key:
        required = {k: v for k, v in model.items() if k in required_model_keys(cap)}
        bundle_only = {"type": cap.type_id, "dtype": "bfloat16", **required, cap.bundle_key: "path/to/bundle"}
        components = {c.key: f"path/to/{c.key}" for c in cap.components}
        for section in (bundle_only, {**bundle_only, **components}):
            assert not _validation_errors(training_config(section)), section


@spec_params
def test_preview_options_constrain_only_their_model(cap):
    """A ``[preview]`` key with fixed values (``preview_options``) accepts each of them and rejects
    anything else for this model; models that do not declare the key accept any value."""
    from rengu_flow.registry.model_config_rules import validate_preview_options

    model = minimal_model_section(cap)
    for key, allowed in cap.preview_options.items():
        for value in allowed:
            assert not _validation_errors(training_config(model, preview={key: value})), (key, value)
        errors = _validation_errors(training_config(model, preview={key: "not_an_option"}))
        assert any(f"preview.{key}" in e for e in errors), errors
        for other in registered_specs():
            if key not in other.preview_options:
                validate_preview_options({"preview": {key: "anything"}}, other)


@spec_params
def test_docs_notice_and_package_data(cap):
    """The user guide exists; a model package carries its NOTICE (linked from
    THIRD_PARTY_NOTICES.md) and ships its ``assets/`` as package data."""
    assert (REPO_ROOT / cap.user_doc).is_file(), cap.user_doc
    package = _spec_package_dir(cap)
    if package is None:
        return  # spec module at rengu_flow/model/ (sdxl: diffusers, nothing vendored)
    rel = package.relative_to(REPO_ROOT).as_posix()
    assert (package / "NOTICE.md").is_file(), f"{rel}/NOTICE.md"
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert f"({rel}/NOTICE.md)" in notices, f"THIRD_PARTY_NOTICES.md does not link {rel}/NOTICE.md"
    if (package / "assets").is_dir():
        pyproject = toml.load(REPO_ROOT / "pyproject.toml")
        package_data = pyproject["tool"]["setuptools"]["package-data"]
        dotted = rel.replace("/", ".")
        assert "assets/**/*" in package_data.get(dotted, []), f"package-data misses {dotted} assets"


@spec_params
def test_readme_row_and_env_example_block(cap):
    """README's model table has the model's row; ``.env.example`` lists every env override of
    its component paths (commented out or set)."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    rows = [line for line in readme.splitlines() if line.startswith("|") and f"| `{cap.type_id}` |" in line]
    assert rows, f"README.md has no model-table row for `{cap.type_id}`"
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    listed = set(re.findall(r"^#?\s*(RENGU_[A-Z0-9_]+)=", env_example, flags=re.M))
    expected = {f"RENGU_{cap.env_prefix}_{c.key.upper()}" for c in cap.components if c.env}
    assert expected <= listed, f".env.example misses {sorted(expected - listed)}"


# ---- config defaults, env, installer, cache, preview dispatch ------------------------------------


@spec_params
def test_one_of_groups_derived_from_components_and_bundle(cap):
    groups = one_of_groups(cap)
    explicit = (cap.model_validation or {}).get("one_of")
    if explicit:
        assert groups == [list(g) for g in explicit]
        if cap.bundle_key:  # the derivation alone gives the same groups
            bare = copy.copy(cap)
            bare.model_validation = {}
            assert one_of_groups(bare) == groups
    elif cap.bundle_key:
        assert groups == [[c.key, cap.bundle_key] for c in cap.components if c.key != cap.bundle_key]
    else:
        assert groups == []


@spec_params
def test_defaults_follow_the_spec(cap):
    """``set_config_defaults`` writes what the spec declares — for the type and its aliases."""
    for type_id in (cap.type_id, *cap.aliases):
        config = _defaults({"type": type_id, "dtype": "bfloat16"})
        model, preview = config["model"], config["preview"]
        assert model["cache_text_embeddings"] is True
        assert ("transformer_fp8_matmul" in model) == (cap.quant is not None)
        assert ("fp8_grad_mode" in model) == (cap.quant is not None)
        assert ("fp8_matmul_dtype" in model) == bool(cap.quant and cap.quant.fp8_dtypes)
        if cap.quant and cap.quant.fp8_dtypes:
            assert model["fp8_matmul_dtype"] == cap.quant.fp8_dtypes[0]
        for key, value in cap.model_defaults.items():
            assert model[key] == value
        others = {k for c in registered_specs() if c is not cap for k in c.model_defaults} - set(cap.model_defaults)
        assert not others & set(model), f"another model's defaults leaked in: {sorted(others & set(model))}"
        if cap.quant is not None:
            assert model["transformer_fp8_matmul"] is False and model["transformer_4bit"] is False
            assert model["fp8_grad_mode"] == "bf16"
        else:
            assert not {"transformer_4bit", "transformer_fp8_matmul", "fp8_grad_mode"} & set(model)
        spec = cap.preview_spec
        if spec is None or not spec.write_defaults:
            assert "num_inference_steps" not in preview
        else:
            assert preview["num_inference_steps"] == spec.steps
            assert preview["guidance_scale"] == spec.guidance
            assert preview["negative_prompt"] == spec.negative_prompt
            assert (preview["width"], preview["height"]) == (1024, 1024)
            assert preview["preview_offload_text_encoder"] is True
            assert ("preview_offload_dit_for_decode" in preview) == ("offload_dit_for_decode" in spec.memory_knobs)
            for key, value in spec.defaults.items():
                assert preview[key] == value
            explicit = {"num_inference_steps": spec.steps + 7, "negative_prompt": "mine"}
            explicit.update({k: f"explicit_{k}" for k in spec.defaults})
            kept = _defaults_with_preview({"type": type_id, "dtype": "bfloat16"}, dict(explicit))["preview"]
            assert {k: kept[k] for k in explicit} == explicit, "a [preview] value the user set was overwritten"
        with_dtype = _defaults({"type": type_id, "dtype": "bfloat16", "diffusion_model_dtype": "bfloat16"})["model"]
        assert ("transformer_dtype" in with_dtype) == cap.diffusion_dtype_sets_load_dtype


@spec_params
def test_quant_knobs_validate(cap):
    """fp8 and NF4 are mutually exclusive; fp8 weight format and grad mode take only their values."""
    from rengu_flow.config.validation import ConfigValidationError

    if cap.quant is None:
        return
    base = {"type": cap.type_id, "dtype": "bfloat16"}
    if cap.quant.nf4 and cap.quant.fp8:
        with pytest.raises(ConfigValidationError, match="mutually exclusive"):
            _defaults({**base, "transformer_fp8_matmul": True, "transformer_4bit": True})
    if cap.quant.fp8_dtypes:
        with pytest.raises(ConfigValidationError, match="fp8_matmul_dtype") as err:
            _defaults({**base, "fp8_matmul_dtype": "fp16"})
        assert all(f"'{d}'" in str(err.value) for d in cap.quant.fp8_dtypes), str(err.value)
    if cap.quant.grad_mode:
        with pytest.raises(ConfigValidationError, match="fp8_grad_mode"):
            _defaults({**base, "fp8_grad_mode": "nope"})


@spec_params
def test_env_overrides_fill_the_component_paths(cap, monkeypatch, tmp_path):
    """``RENGU_<PREFIX>_<KEY>`` fills every env-enabled component path (over the TOML value); with
    real files there the config has no model-path errors (the smoke prerequisites check)."""
    from rengu_flow.config import local_env

    expected = {c.key: f"RENGU_{cap.env_prefix}_{c.key.upper()}" for c in cap.components if c.env}
    assert local_env.model_path_env()[cap.type_id] == expected
    assert list(local_env.model_path_env()[cap.type_id]) == list(expected)
    assert local_env._MODEL_PATH_ENV[cap.type_id] == expected  # back-compat name
    for env in expected.values():
        monkeypatch.setenv(env, f"/env/{env}")
    config = {"model": {"type": cap.type_id, **{k: "from_toml" for k in expected}}}
    applied = local_env.apply_model_paths_from_env(config)
    assert sorted(applied) == sorted(expected.values())
    assert {k: config["model"][k] for k in expected} == {k: f"/env/{env}" for k, env in expected.items()}

    kinds = {c.key: c.expect for c in cap.components}
    for key, env in expected.items():
        target = tmp_path / key
        if kinds[key] == "dir":
            target.mkdir()
        else:
            target.write_bytes(b"x")
        monkeypatch.setenv(env, str(target))
    config = {"model": {"type": cap.type_id, "dtype": "bfloat16"}}
    local_env.apply_model_paths_from_env(config)
    for key in required_model_keys(cap):  # a required key without an env override (none today)
        if key not in config["model"]:
            (tmp_path / f"{key}_in_toml").write_bytes(b"x")
            config["model"][key] = str(tmp_path / f"{key}_in_toml")
    assert local_env.model_path_errors(config) == []


@spec_params
def test_install_profiles_follow_the_spec(cap):
    from rengu_flow.install.manager import profiles_for_config_dict
    from rengu_flow.install.profiles import PROFILE_EXTRAS

    assert set(cap.install_profiles) <= set(PROFILE_EXTRAS)
    for type_id in (cap.type_id, *cap.aliases):
        assert profiles_for_config_dict({"model": {"type": type_id}}) == list(cap.install_profiles)
    assert profiles_for_config_dict({"model": {"type": "not_a_model"}}) == []


@spec_params
def test_ragged_cache_keys_reach_the_cache(cap):
    from rengu_flow.registry.model_spec import BASE_RAGGED_CACHE_KEYS
    from rengu_flow.utils.cache import sequence_tensor_keys

    declared = set(BASE_RAGGED_CACHE_KEYS).union(*(c.ragged_cache_keys for c in model_capability_registry.values()))
    assert set(sequence_tensor_keys()) == declared
    assert BASE_RAGGED_CACHE_KEYS | cap.ragged_cache_keys <= sequence_tensor_keys()


class _Named:
    def __init__(self, name):
        self.name = name


@spec_params
def test_preview_dispatch_follows_the_spec(cap):
    from rengu_flow.utils.preview import _edit_preview_models, _preview_spec

    for type_id in (cap.type_id, *cap.aliases):
        assert _preview_spec(_Named(type_id)) is cap.preview_spec
    assert _preview_spec(_Named("not_a_model")) is None
    assert (cap.type_id in _edit_preview_models().split(" / ")) == bool(cap.preview_spec and cap.preview_spec.edit)


# ---- web form -----------------------------------------------------------------------------------


@pytest.fixture
def deepspeed_host(monkeypatch):
    """The form as a DeepSpeed host builds it (all engine-gated knobs present)."""
    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")


def _schema_fields() -> tuple[list[dict], dict]:
    from rengu_flow_ui.config_schema import get_schema

    schema = get_schema()
    return [f for s in schema["sections"] for f in s["fields"]], schema["registries"]["model_capabilities"]


def _forms(type_id: str, fields: list[dict]) -> list[dict]:
    """Form states of ``type_id`` that unlock its fields: adapter on / off, each with nothing else
    set and with every trigger set (expert ``show_if_set`` / dependent fields)."""
    base = {"model.type": type_id}
    every = {f["path"]: True for f in fields if f["path"] not in ("model.type", "_has_adapter")}
    return [{**base, "_has_adapter": on, **extra} for on in (True, False) for extra in ({}, every)]


def _visible(field: dict, type_id: str, fields: list[dict], caps: dict) -> bool:
    from rengu_flow_ui.field_visibility import field_visible

    return any(field_visible(field, form, caps) for form in _forms(type_id, fields))


@spec_params
def test_ui_features_derive_from_the_spec(cap):
    """The capability features the form gates on come from the descriptor, never a type list."""
    ps = cap.preview_spec
    f = cap.features
    assert bool(f.get("preview")) == (ps is not None)
    assert bool(f.get("preview_dit_memory")) == (ps is not None and ps.runner == "dit")
    assert bool(f.get("preview_block_swap")) == (ps is not None and "blocks_to_swap" in ps.memory_knobs)
    assert bool(f.get("preview_decode_offload")) == (ps is not None and "offload_dit_for_decode" in ps.memory_knobs)
    assert bool(f.get("edit")) == (ps is not None and ps.edit)
    assert bool(f.get("text_cache_optional")) == (cap.text_cache == "optional")
    assert bool(f.get("adapter_layer_groups")) == bool(cap.adapter_layer_groups)
    for key in (*cap.preview_options, *(ps.defaults if ps else {})):
        assert f.get(f"preview_{key}") is True, key


@spec_params
def test_model_fields_show_only_on_their_form_with_help_and_doc(cap, deepspeed_host):
    """Every form field of the spec is one schema field gated to this model: visible on its form,
    hidden on every other model's form, with help text; a field whose spec gives help links an
    existing doc (default: the spec's ``user_doc``). The model is a ``model.type`` option."""
    fields, caps = _schema_fields()
    assert cap.type_id in caps
    type_field = next(f for f in fields if f["path"] == "model.type")
    assert cap.type_id in type_field["options"]
    others = [c.type_id for c in registered_specs() if c.type_id != cap.type_id]
    for spec in cap.model_fields:
        if spec.get("ui") is False:
            continue
        own = [f for f in fields if f["path"] == spec["path"] and _visible(f, cap.type_id, fields, caps)]
        own = [f for f in own if not any(_visible(f, other, fields, caps) for other in others)] or own
        assert len(own) == 1, (spec["path"], len(own))
        field = own[0]
        for other in others:
            assert not _visible(field, other, fields, caps), f"{spec['path']} of {cap.type_id} shows on {other}"
        assert (field.get("help") or field.get("description") or "").strip(), spec["path"]
        doc = field.get("doc_path")
        if spec.get("help"):
            assert field["help"] == spec["help"], spec["path"]
        if spec.get("help") or spec.get("doc"):
            assert doc == (spec.get("doc") or cap.user_doc), (spec["path"], doc)
        if doc:
            assert (REPO_ROOT / doc.split("#")[0]).is_file(), (spec["path"], doc)


@spec_params
def test_layer_groups_field_offers_the_spec_groups(cap, deepspeed_host):
    """The form's ``adapter.layer_groups`` on this model offers exactly its spec's groups (and
    none when it declares none)."""
    fields, caps = _schema_fields()
    shown = [f for f in fields if f["path"] == "adapter.layer_groups" and _visible(f, cap.type_id, fields, caps)]
    if not cap.adapter_layer_groups:
        assert not shown
        return
    assert len(shown) == 1 and shown[0]["options"] == list(cap.adapter_layer_groups), shown


@spec_params
def test_default_layer_groups_follow_the_spec(cap, deepspeed_host):
    """A model's default adapter scope (``adapter_default_layer_groups``) names real groups, and the form's
    ``adapter.layer_groups`` placeholder and description state what empty means for that model."""
    defaults = cap.adapter_default_layer_groups
    assert set(defaults) <= set(cap.adapter_layer_groups), defaults
    fields, caps = _schema_fields()
    (field,) = [f for f in fields if f["path"] == "adapter.layer_groups" and _visible(f, cap.type_id, fields, caps)] or [None]
    if field is None:
        assert not cap.adapter_layer_groups
        return
    for text in (field["placeholder"], field["description"]):
        assert all(group in text for group in defaults), (defaults, text)
        assert defaults or "every targeted linear" in text, text


@spec_params
def test_refused_dtypes_follow_the_spec(cap, deepspeed_host):
    """A dtype the spec refuses fails config validation (``model.dtype`` and ``model.transformer_dtype``,
    naming the reason) and is missing from this model's dtype options on the form, while the
    other models keep offering it; the dtype field stays one per path."""
    model = minimal_model_section(cap)
    fields, caps = _schema_fields()
    for dtype, reason in cap.refused_dtypes.items():
        for key in ("dtype", "transformer_dtype"):
            errors = _validation_errors(training_config({**model, key: dtype}))
            assert any(f"model.{key}" in e and reason in e for e in errors), (key, errors)
        for other in registered_specs():
            if other is not cap and not other.refused_dtypes:
                assert not any("not supported" in e for e in _validation_errors(training_config({**minimal_model_section(other), "dtype": dtype})))
    for path in ("model.dtype", "model.transformer_dtype"):
        shown = [f for f in fields if f["path"] == path and _visible(f, cap.type_id, fields, caps)]
        if shown:
            assert len(shown) == 1 and not set(cap.refused_dtypes) & set(shown[0]["options"]), path
    for other in registered_specs():
        if not other.refused_dtypes:
            (field,) = [f for f in fields if f["path"] == "model.dtype" and _visible(f, other.type_id, fields, caps)]
            assert set(cap.refused_dtypes) <= set(field["options"])


@spec_params
def test_preview_memory_knobs_show_exactly_where_the_runner_honors_them(cap, deepspeed_host):
    """``preview.preview_<knob>`` is on the model's form iff its ``PreviewSpec.memory_knobs`` has
    it: no hidden knob the runner reads (Krea 2's preview OOM fix was hidden once), no dead one."""
    fields, caps = _schema_fields()
    knobs = cap.preview_spec.memory_knobs if cap.preview_spec else frozenset()
    for knob in ("offload_text_encoder", "blocks_to_swap", "offload_dit_for_decode"):
        shown = any(f["path"] == f"preview.preview_{knob}" and _visible(f, cap.type_id, fields, caps) for f in fields)
        assert shown == (knob in knobs), (knob, shown)


@spec_params
def test_tread_fields_follow_the_tread_feature(cap, deepspeed_host):
    fields, caps = _schema_fields()
    tread = [f for f in fields if f["path"].startswith("tread.")]
    assert tread
    shown = {f["path"] for f in tread if _visible(f, cap.type_id, fields, caps)}
    assert bool(shown) == bool(cap.features.get("tread")), shown


@spec_params
def test_form_shows_one_field_per_path(cap, deepspeed_host):
    """In every state of the model's form, no two visible fields share a path (two models'
    variants of a field never show together)."""
    from rengu_flow_ui.field_visibility import field_visible

    fields, caps = _schema_fields()
    for form in _forms(cap.type_id, fields):
        paths = [f["path"] for f in fields if field_visible(f, form, caps)]
        dupes = sorted({p for p in paths if paths.count(p) > 1})
        assert not dupes, (form.get("_has_adapter"), dupes)


@spec_params
def test_component_fields_follow_the_components(cap, deepspeed_host):
    """A component path field checks what its loader accepts (``ComponentPath.expect``); with a
    bundle folder no component is required on its own (the folder fills it) and the folder shows
    without a trigger. (An explicit one_of may still mark its usual member required in the form:
    cosmos asks for llm_path, t5_path is the legacy alternative.)"""
    fields, caps = _schema_fields()
    plain = _forms(cap.type_id, fields)[:1] + _forms(cap.type_id, fields)[2:3]
    in_groups = {k for g in one_of_groups(cap) for k in g} if cap.bundle_key else set()
    for component in cap.components:
        own = [f for f in fields if f["path"] == f"model.{component.key}" and _visible(f, cap.type_id, fields, caps)]
        if not own:
            continue  # a TOML-only component (ui: false)
        (field,) = own
        assert (field.get("path_expect") or "file") == component.expect, (component.key, field.get("path_expect"))
        if component.key in in_groups:
            assert not field.get("required"), component.key
    if cap.bundle_key:
        from rengu_flow_ui.field_visibility import field_visible

        bundle = [f for f in fields if f["path"] == f"model.{cap.bundle_key}" and _visible(f, cap.type_id, fields, caps)]
        assert bundle and all(field_visible(bundle[0], form, caps) for form in plain), "bundle folder hidden"


@spec_params
def test_adapter_form_fields_follow_the_spec(cap, deepspeed_host):
    """On a model with adapters the form offers ``adapter.rank`` (never its TOML alias ``dim``);
    LyCORIS conv / norm knobs show only where the adapted network has conv or affine norms."""
    from rengu_flow.registry.model_spec import LINEAR_ONLY_ADAPTERS

    if not cap.adapters:
        return
    fields, caps = _schema_fields()

    def shown(path, adapter):
        from rengu_flow_ui.field_visibility import field_visible

        form = {"model.type": cap.type_id, "_has_adapter": True, "adapter.type": adapter}
        return any(f["path"] == path and field_visible(f, form, caps) for f in fields)

    assert shown("adapter.rank", cap.adapters[0]) and not shown("adapter.dim", cap.adapters[0])
    lycoris = [a for a in cap.adapters if a == "lycoris_locon"]
    for adapter in lycoris:
        for knob in ("adapter.train_conv", "adapter.use_tucker", "adapter.train_norm"):
            assert shown(knob, adapter) == (not cap.features.get(LINEAR_ONLY_ADAPTERS)), knob


@spec_params
def test_text_cache_toggle_shows_only_when_optional(cap, deepspeed_host):
    """``model.cache_text_embeddings`` is a form toggle only where the text cache is optional (a
    "required" model rejects false)."""
    assert ("model.cache_text_embeddings" in visible_paths(cap.type_id)) == (cap.text_cache == "optional")


@spec_params
def test_preview_option_fields_follow_the_spec(cap, deepspeed_host):
    """A ``[preview]`` key with fixed values is a select of exactly those values, defaulting to
    the spec's preview default, shown only on the models that declare it."""
    fields, caps = _schema_fields()
    for key, allowed in cap.preview_options.items():
        own = [f for f in fields if f["path"] == f"preview.{key}" and _visible(f, cap.type_id, fields, caps)]
        assert len(own) == 1, key
        field = own[0]
        assert field["type"] == "select" and field["options"] == list(allowed), field
        if key in cap.preview_spec.defaults:
            assert field["default"] == cap.preview_spec.defaults[key]
        for other in registered_specs():
            if key not in other.preview_options:
                assert not _visible(field, other.type_id, fields, caps), (key, other.type_id)


@spec_params
def test_ui_preview_defaults_equal_the_trainer_defaults(cap, deepspeed_host):
    """The sampler defaults the form shows for a model are what a run uses when ``[preview]``
    omits them (``set_config_defaults``, or the runner's fallback = the spec's values)."""
    if cap.preview_spec is None:
        return
    fields, caps = _schema_fields()
    spec = cap.preview_spec
    trainer = _defaults({"type": cap.type_id, "dtype": "bfloat16"})["preview"]
    for text in spec.help.values():  # the doc a sampler help names resolves
        assert (REPO_ROOT / (text.get("doc") or cap.user_doc)).is_file()
    for key, from_spec in (("num_inference_steps", spec.steps), ("guidance_scale", spec.guidance)):
        shown = [f for f in fields if f["path"] == f"preview.{key}" and _visible(f, cap.type_id, fields, caps)]
        assert len(shown) == 1, (key, len(shown))
        expected = trainer[key] if spec.write_defaults else from_spec
        assert shown[0]["default"] == expected == from_spec, (key, shown[0]["default"], expected)
        text = spec.help.get(key) or {}
        if text.get("help"):  # model-specific sampler help links the model's doc
            assert shown[0]["help"] == text["help"]
            assert shown[0]["doc_path"] == (text.get("doc") or cap.user_doc)


def _consumed_keys(paths: list[Path]) -> set[str]:
    import test_ui_knob_coverage as knobs

    consumed: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for m in knobs._SECTIONED_RE.finditer(text):
            consumed.add(knobs._SECTION_PREFIX[m.group(1)] + (m.group(2) or m.group(3)))
        for m in knobs._BARE_RE.finditer(text):
            consumed.add(knobs._BARE_PREFIX[m.group(1)] + (m.group(2) or m.group(3)))
    return consumed


def model_source_files(cap) -> tuple[list[Path], list[Path]]:
    """``(own, shared)``: the model's own code (its pipeline package or module) and, when it
    builds on them, the shared DiT pieces (``dit_common``: quantization, preview memory, timestep
    sampling) — shared code reads every DiT knob, the model may not call all of it."""
    origin = importlib.util.find_spec(cap.pipeline_module)
    root = Path(origin.origin)
    own = sorted(root.parent.rglob("*.py")) if origin.submodule_search_locations else [root]
    uses_shared = any("dit_common" in f.read_text(encoding="utf-8") for f in own)
    shared = sorted((SRC_ROOT / "model" / "dit_common").rglob("*.py")) if uses_shared else []
    return own, shared


def declined_shared_keys(cap) -> set[str]:
    """Knobs of the shared DiT code that the spec does not offer for this model (its form hides
    them by design): quantization it does not declare, preview memory knobs it lacks."""
    quant = cap.quant
    declined = set()
    if quant is None or not quant.nf4:
        declined.add("model.transformer_4bit")
    if quant is None or quant.fp8 is None:
        declined.add("model.transformer_fp8_matmul")
    if quant is None or not quant.grad_mode:
        declined.add("model.fp8_grad_mode")
    memory = cap.preview_spec.memory_knobs if cap.preview_spec else frozenset()
    for knob in ("offload_text_encoder", "blocks_to_swap", "offload_dit_for_decode"):
        if knob not in memory:
            declined.add(f"preview.preview_{knob}")
    return declined


def visible_paths(type_id: str) -> set[str]:
    """Every field path visible on ``type_id``'s form in some state."""
    fields, caps = _schema_fields()
    return {f["path"] for f in fields if _visible(f, type_id, fields, caps)}


@spec_params
def test_every_key_the_model_reads_is_on_its_form(cap, deepspeed_host):
    """Each config key the model's code reads is visible on that model's form (gates can hide a
    field the model consumes), unless allowlisted with a reason; a shared-DiT knob the spec does
    not offer may stay hidden."""
    import test_ui_knob_coverage as knobs

    own, shared = model_source_files(cap)
    consumed = _consumed_keys(own)
    visible = visible_paths(cap.type_id)
    allowed = set(knobs.ALLOWLIST) | set(knobs.PER_MODEL_ALLOWLIST.get(cap.type_id, ()))
    if cap.text_cache == "required":
        # Always on; the pipeline raises on false, so the toggle is hidden.
        allowed.add("model.cache_text_embeddings")
    missing = sorted((consumed | (_consumed_keys(shared) - declined_shared_keys(cap))) - visible - allowed)
    assert not missing, (
        f"Config keys {cap.type_id}'s code reads but its form never shows: {missing}. Add a "
        "model_fields entry in its spec, fix the field's capability gate, or a justified "
        "PER_MODEL_ALLOWLIST entry in tests/test_ui_knob_coverage.py."
    )
    stale = sorted(set(knobs.PER_MODEL_ALLOWLIST.get(cap.type_id, ())) - consumed)
    assert not stale, f"PER_MODEL_ALLOWLIST[{cap.type_id!r}] lists keys the model no longer reads: {stale}"
