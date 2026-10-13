"""Contract D: a model added only as a package + spec + one registration line reaches every
integration surface.

``toy_model/`` (next to this file) is a Nanosaur2 under the type id ``toy_dit`` with its own env
prefix, preview defaults, install profiles, model default and ragged cache key, plus a test kit.
The ``toy`` fixture registers it the way ``registry/builtin_models.py`` does (its module appended
to ``BUILTIN_MODEL_SPECS`` and its ``SPEC`` registered) and undoes it afterwards; no other file
knows about it. Validation, defaults, env overrides, the installer, the cache's ragged keys, the
model registry, the web schema / help / form, preview dispatch, and the per-spec (A) and
per-kit (B) contract checks must all pick it up. A surface that still keeps a per-type list (or
derives one at import time) fails here.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import test_pipeline_contract as contract_b
import test_spec_contract as contract_a

pytestmark = pytest.mark.no_ui_db

TOY = "toy_dit"

# Every surface module is imported before the toy registers, as in a real process (the registry
# is filled at import, the surfaces long before a run picks a model): a surface that captured the
# registered types at import time misses the toy and fails here.
SURFACE_MODULES = (
    "rengu_flow.config.defaults",
    "rengu_flow.config.local_env",
    "rengu_flow.config.preflight",
    "rengu_flow.config.validation",
    "rengu_flow.install.manager",
    "rengu_flow.registry.model_config_rules",
    "rengu_flow.registry.models",
    "rengu_flow.utils.cache",
    "rengu_flow.utils.preview",
    "rengu_flow_ui.config_field_help",
    "rengu_flow_ui.config_schema",
    "rengu_flow_ui.field_visibility",
    "rengu_flow_ui.model_form",
    "rengu_flow_ui.preview_form",
)


@pytest.fixture
def toy(monkeypatch):
    """Register the toy spec (one line in BUILTIN_MODEL_SPECS + the registration loop); undo it,
    including the pipeline's ``@register_model`` side effects."""
    from rengu_flow.registry import builtin_models
    from rengu_flow.registry import models as model_registry_module
    from rengu_flow.registry.model_capabilities import model_capability_registry

    for module in SURFACE_MODULES:
        importlib.import_module(module)
    spec_module = "toy_model.spec"
    monkeypatch.setattr(builtin_models, "BUILTIN_MODEL_SPECS", (*builtin_models.BUILTIN_MODEL_SPECS, spec_module))
    spec = importlib.import_module(spec_module).SPEC
    monkeypatch.setitem(model_capability_registry, spec.type_id, spec)
    registry_before = dict(model_registry_module.model_registry)
    canonical_before = set(model_registry_module.canonical_model_types)
    yield spec
    model_registry_module.model_registry.clear()
    model_registry_module.model_registry.update(registry_before)
    model_registry_module.canonical_model_types.clear()
    model_registry_module.canonical_model_types.update(canonical_before)
    sys.modules.pop("toy_model.pipeline", None)  # re-registers on the next test's import


@pytest.fixture
def toy_kit(toy, tmp_path_factory):
    from conftest import Kit

    pytest.importorskip("rengu_flow.model.nanosaur2.pipeline")  # the toy's extras (not its module:
    # importing it registers the pipeline, which get_model must do on its own)
    return Kit(toy, importlib.import_module(toy.testkit), tmp_path_factory.mktemp("tiny_toy"))


def _no_golden(*_args, **_kwargs) -> None:
    """The toy has no numeric goldens (its numerics are Nanosaur2's)."""


def _toy_training_config(**top) -> dict:
    model = {
        "type": TOY,
        "dtype": "bfloat16",
        "transformer_path": "path/to/toy_dit.safetensors",
        "vae_path": "path/to/toy_vae.safetensors",
        "text_encoder_path": "path/to/toy_te.safetensors",
    }
    return contract_a.training_config(model, **top)


# ---- the toy reaches every surface --------------------------------------------------------------


def test_registration_reaches_the_registries_and_the_suites(toy, toy_kit):
    from conftest import _kit_specs

    from rengu_flow.registry.builtin_models import builtin_specs
    from rengu_flow.registry.model_capabilities import get_canonical_model_types, get_capability
    from rengu_flow.registry.models import _model_modules, get_model

    assert get_capability(TOY) is toy
    assert TOY in get_canonical_model_types()
    assert builtin_specs()[-1] is toy
    assert toy in contract_a.registered_specs()  # suite A's parameters
    assert toy in _kit_specs()  # suite B / C parameters
    assert _model_modules()[TOY] == "toy_model.pipeline"
    assert "toy_model.pipeline" not in sys.modules
    model = get_model(toy_kit.config())  # imports the pipeline module from the spec alone
    from toy_model.pipeline import ToyPipeline

    assert type(model) is ToyPipeline


def test_validation_and_defaults(toy):
    from rengu_flow.config.validation import collect_validation_errors

    config = _toy_training_config()
    assert not collect_validation_errors(config)
    del config["model"]["vae_path"]
    assert any("vae_path" in e and TOY in e for e in collect_validation_errors(config))
    bad_mode = _toy_training_config(preview={"guidance_mode": "nope"})
    assert collect_validation_errors(bad_mode)  # the spec's preview_options gate it

    defaults = contract_a._defaults({"type": TOY, "dtype": "bfloat16"})
    assert defaults["preview"]["num_inference_steps"] == 11
    assert defaults["preview"]["guidance_scale"] == 2.5
    assert defaults["preview"]["negative_prompt"] == "toy negative"
    assert defaults["model"]["sparse_skip_prob"] == 0.25


def test_env_installer_and_cache_keys(toy, monkeypatch):
    from rengu_flow.config import local_env
    from rengu_flow.install.manager import profiles_for_config_dict
    from rengu_flow.utils import cache

    monkeypatch.setenv("RENGU_TOY_DIT_VAE_PATH", "/env/vae.safetensors")
    config = {"model": {"type": TOY}}
    assert local_env.apply_model_paths_from_env(config) == ["RENGU_TOY_DIT_VAE_PATH"]
    assert config["model"]["vae_path"] == "/env/vae.safetensors"
    assert profiles_for_config_dict({"model": {"type": TOY}}) == ["cosmos", "optim"]
    assert "toy_rows" in cache.sequence_tensor_keys()
    assert cache._is_sequence_key("toy_rows", (7,))


def test_schema_form_help_and_preview_defaults(toy, monkeypatch):
    from rengu_flow_ui.config_schema import get_schema

    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")
    schema = get_schema()
    assert TOY in schema["registries"]["model_capabilities"]
    fields = [f for s in schema["sections"] for f in s["fields"]]
    caps = schema["registries"]["model_capabilities"]
    assert TOY in next(f for f in fields if f["path"] == "model.type")["options"]
    own = [f for f in fields if f["path"] == "model.transformer_path" and contract_a._visible(f, TOY, fields, caps)]
    assert len(own) == 1 and own[0]["doc_path"] == toy.user_doc
    steps = [f for f in fields if f["path"] == "preview.num_inference_steps" and contract_a._visible(f, TOY, fields, caps)]
    assert [f["default"] for f in steps] == [11]


def test_preview_dispatch(toy):
    from rengu_flow.utils.preview import _preview_spec

    class Model:
        name = TOY

    assert _preview_spec(Model()) is toy.preview_spec


# ---- suites A and B run on the toy -----------------------------------------------------------------

# Spec checks the toy passes as shipped (package + spec + kit, nothing else).
SPEC_CHECKS = (
    "test_spec_internal_consistency",
    "test_kit_declarations_are_consistent",
    "test_pipeline_resolves_to_registered_class",
    "test_export_prefix_matches_pipeline",
    "test_validation_derives_from_required_keys_and_one_of_groups",
    "test_preview_options_constrain_only_their_model",
    "test_one_of_groups_derived_from_components_and_bundle",
    "test_defaults_follow_the_spec",
    "test_quant_knobs_validate",
    "test_env_overrides_fill_the_component_paths",
    "test_install_profiles_follow_the_spec",
    "test_ragged_cache_keys_reach_the_cache",
    "test_preview_dispatch_follows_the_spec",
    "test_ui_features_derive_from_the_spec",
    "test_model_fields_show_only_on_their_form_with_help_and_doc",
    "test_layer_groups_field_offers_the_spec_groups",
    "test_form_shows_one_field_per_path",
    "test_component_fields_follow_the_components",
    "test_adapter_form_fields_follow_the_spec",
    "test_text_cache_toggle_shows_only_when_optional",
    "test_preview_option_fields_follow_the_spec",
    "test_preview_memory_knobs_show_exactly_where_the_runner_honors_them",
    "test_tread_fields_follow_the_tread_feature",
    "test_ui_preview_defaults_equal_the_trainer_defaults",
    "test_every_key_the_model_reads_is_on_its_form",
)
# Spec checks of the repo-level deliverables the toy deliberately lacks (examples, NOTICE, README
# row, .env.example block): suite A must flag each of them.
DELIVERABLE_CHECKS = (
    "test_examples_exist_validate_and_need_their_required_keys",
    "test_docs_notice_and_package_data",
    "test_readme_row_and_env_example_block",
)


def _call(check, **available):
    import inspect

    params = inspect.signature(check).parameters
    return check(**{name: available[name] for name in params})


@pytest.mark.parametrize("name", SPEC_CHECKS)
def test_spec_contract_runs_on_the_toy(toy, name, monkeypatch, tmp_path):
    if name in ("test_pipeline_resolves_to_registered_class", "test_export_prefix_matches_pipeline"):
        pytest.importorskip(toy.pipeline_module)
    monkeypatch.setenv("RENGU_ENGINE", "deepspeed")
    _call(getattr(contract_a, name), cap=toy, monkeypatch=monkeypatch, deepspeed_host=None, tmp_path=tmp_path)


@pytest.mark.parametrize("name", DELIVERABLE_CHECKS)
def test_spec_contract_flags_the_toys_missing_deliverables(toy, name, monkeypatch):
    with pytest.raises(AssertionError):
        _call(getattr(contract_a, name), cap=toy, monkeypatch=monkeypatch)


def test_pipeline_contract_runs_on_the_toy(toy_kit, tmp_path, monkeypatch):
    """Every per-kit check, on the toy's kit (CPU, no goldens)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    case = toy_kit.cases[0]
    contract_b.test_layers_match_the_monolithic_forward((toy_kit, case), _no_golden)
    contract_b.test_gradients_reach_every_trainable_parameter((toy_kit, case), _no_golden)
    contract_b.test_adapter_attaches_to_the_declared_scope_and_exports_golden_keys(
        toy_kit, "lora", tmp_path / "adapter", _no_golden
    )
    contract_b.test_layer_groups_and_module_roots_match_the_real_dit(toy_kit, monkeypatch)
    contract_b.test_vae_fn_is_deterministic(toy_kit)
    contract_b.test_text_encoder_fn_rows_are_batch_independent(toy_kit)
    contract_b.test_text_encoder_fn_caches_ragged_rows(toy_kit)
    contract_b.test_prepare_inputs_builds_the_flow_matching_target((toy_kit, case))
    contract_b.test_previews_render_through_the_spec_runner(toy_kit, _no_golden)
    contract_b.test_preview_end_to_end_memoizes_prompt_embeddings(toy_kit, True, _no_golden)
    contract_b.test_init_does_not_load_the_text_encoder(toy_kit)
    contract_b.test_save_model_round_trips_through_the_loader(toy_kit, tmp_path / "export")
