"""Model-field help lives in the model specs; ``FIELD_HELP`` keeps only generic entries."""

from __future__ import annotations

import pytest

from rengu_flow.registry import field_library
from rengu_flow.registry.model_capabilities import model_capability_registry
from rengu_flow.registry.model_spec import QuantSpec
from rengu_flow_ui.config_field_help import FIELD_HELP, enrich_schema

pytestmark = pytest.mark.no_ui_db

# [model] help entries that are not tied to one model (the type picker, the shared dtype, the
# cache toggle every model section shares).
GENERIC_MODEL_KEYS = {"model.type", "model.dtype", "model.cache_text_embeddings"}
CAPS = list(model_capability_registry.values())


def test_field_help_has_no_model_scoped_keys():
    assert not [k for k in FIELD_HELP if "@" in k]
    spec_paths = {f["path"] for cap in CAPS for f in cap.model_fields}
    model_keys = {k for k in FIELD_HELP if k.startswith("model.")}
    assert model_keys <= GENERIC_MODEL_KEYS, model_keys - GENERIC_MODEL_KEYS
    assert not model_keys & (spec_paths - GENERIC_MODEL_KEYS)


def test_enrich_prefers_field_help_over_generic_table():
    own = {"path": "model.dtype", "description": "d", "help": "own help", "doc_path": "docs/own.md"}
    bare = {"path": "model.dtype", "description": "d"}
    schema = enrich_schema({"sections": [{"fields": [own, bare]}]})
    assert own["help"] == "own help" and own["doc_path"] == "docs/own.md"
    assert bare["help"] == FIELD_HELP["model.dtype"]["detail"]
    assert bare["doc_path"] == FIELD_HELP["model.dtype"]["doc"]
    assert schema["sections"][0]["fields"] == [own, bare]


def test_enrich_falls_back_to_description():
    field = {"path": "model.no_such_field", "description": "just a description"}
    enrich_schema({"sections": [{"fields": [field]}]})
    assert field["help"] == "just a description"


@pytest.mark.parametrize(
    ("quant", "keys"),
    [
        (QuantSpec(fp8="tensorwise", nf4=True, grad_mode=True), {"nf4", "fp8", "grad_mode"}),
        (QuantSpec(fp8="scaled_matmul", nf4=True, grad_mode=False, fp8_dtypes=("e5m2", "e4m3")), {"nf4", "fp8", "fp8_dtype"}),
        (QuantSpec(fp8=None, nf4=True, grad_mode=False), {"nf4"}),
    ],
)
def test_quant_fields_follow_the_quant_spec(quant, keys):
    out = field_library.quant_fields(quant, nf4={"description": "custom"})
    assert set(out) == keys
    assert out["nf4"]["description"] == "custom"
    # fresh dicts: a consumer mutating its copy never leaks into the next call
    out["nf4"]["description"] = "mutated"
    assert field_library.quant_fields(quant)["nf4"].get("description") is None


def test_factories_return_fresh_dicts_and_drop_unset_overrides():
    a = field_library.transformer_dtype()
    a["label"] = "mutated"
    b = field_library.transformer_dtype()
    assert b["label"] == "Main model load dtype"
    entry = field_library.component_path("vae_path", "VAE", "d")
    assert set(entry) == {"path", "label", "type", "description"}
    assert field_library.bundle_folder("diffusers_path", "d")["path_expect"] == "dir"
