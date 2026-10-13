# Model capabilities, form visibility, and validation

Training configs are validated in Python (`rengu_flow.config.validation`) and edited in the web UI (`rengu_flow_ui`). Both agree on **which keys exist for which model** because both read the same per-model descriptor: the model **spec**. To add a model, follow [adding-a-model.md](adding-a-model.md); this page describes how the spec is consumed.

## One spec per model

Each model package defines `SPEC = ModelCapability(...)` in `rengu_flow/model/<m>/spec.py` (SDXL: `rengu_flow/model/sdxl_spec.py`). The descriptor types (`ModelCapability`, `ComponentPath`, `QuantSpec`, `PreviewSpec`) live in **`rengu_flow/registry/model_spec.py`**. **`rengu_flow/registry/builtin_models.py`** lists the spec modules (`BUILTIN_MODEL_SPECS`: one line per model, tuple order = registry order = schema field order). **`rengu_flow/registry/model_capabilities.py`** holds the registry (`model_capability_registry`, `get_capability`, `normalize_model_type`, `capabilities_for_api`), the adapter field templates, and the loop that registers every built-in spec at import time. The registry used to be hand-written in `model_capabilities.py`; it no longer is: do not add a `register_model_capability(...)` call there.

Specs are **torch-free** (no torch, diffusers, transformers or model code; package `__init__` files are lazy), so validation, defaults, the installer and the UI schema read them without importing a model. `tests/test_registry_lazy.py::test_model_specs_import_without_torch` guards this.

| Spec field | Consumed by |
|------------|-------------|
| `model_fields` | UI form (paths under `model.*`), required flags, per-field help and doc link |
| `components` (`ComponentPath`), `bundle_key`, `env_prefix` | `required_model_keys` / `one_of_groups` (validation), `RENGU_<PREFIX>_<KEY>` env overrides (`config/local_env.py`), `.env.example` drift test |
| `model_defaults`, `text_cache`, `quant`, `preview_spec`, `diffusion_dtype_sets_load_dtype` | `config/defaults.py` (`set_config_defaults` is a generic loop over the spec) |
| `preview_spec`, `preview_options` | preview runner dispatch (`utils/preview.py`), `[preview]` defaults and form, option validation |
| `install_profiles` | installer (`install/manager.py`) |
| `pipeline` | `registry/models.py` resolves the class lazily |
| `ragged_cache_keys`, `text_cache_keys`, `text_cache_files`, `text_cache_version` | cache layout of ragged tensors and the text-cache identity |
| `adapters`, `full_finetune`, `adapter_layer_groups`, `adapter_module_roots`, `transformer_blocks` | adapter section, layer-group options, torch-free target preflight, `[tread]` range check |
| `export_prefix`, `adapter_exports`, `testkit`, `user_doc` | export check, contract suites and smokes, default doc link of the fields' help |
| `features` | UI visibility (`capability` clauses) and training-key validation |

### Features are mostly derived

`ModelCapability.__post_init__` derives the features the UI gates on from the descriptor, so a model never lists them twice: `preview`, `preview_dit_memory`, `preview_block_swap`, `preview_decode_offload`, `edit` (from `preview_spec`), `preview_<key>` (from `preview_options` / `preview_spec.defaults`), `text_cache_optional` (from `text_cache`), `adapter_layer_groups`. Only the features that describe training capabilities are written explicitly in the spec: `block_swap`, `tread`, `video`, and `LINEAR_ONLY_ADAPTERS` (hides conv/norm-only adapter knobs).

### Field library

`rengu_flow/registry/field_library.py` has factories for the `model_fields` entries that recur (`component_path`, `bundle_folder`, `transformer_dtype`, `diffusion_model_dtype`, `shift`, `sigmoid_scale`, `timestep_sample_method`, `quant_fields`). A factory carries structure (path, label, type, options, default, visibility); the spec passes the model's own `description` / `help` / `doc` / `placeholder`. Each call returns a fresh dict.

### Help comes from the spec

The model-specific help lives in the spec's fields (and `PreviewSpec.help` for the sampler defaults). The old `path@model` entries and model-specific bare `model.*` entries of `FIELD_HELP` are gone. `enrich_schema` (`rengu_flow_ui/config_field_help.py`) applies: the field's own `help` / `doc_path` > generic `FIELD_HELP[path]` > the field's description. A field with `help` gets the spec's `user_doc` as its default doc link. Hint wording follows [documentation-conventions.md](documentation-conventions.md).

## Validation (TOML / CLI / UI Validate button)

**`rengu_flow/registry/model_config_rules.py`** — called from `validate_config()`:

1. **Canonical type** — the raw `model.type` must be a registered canonical id; aliases (`anima`) are rejected here (the UI form rewrites them before save).
2. **`required_model_keys(cap)`** — every `model_fields` entry with `required: true` and `ui` not `false`, except keys inside a `one_of` group (checked in step 3).
3. **`one_of_groups(cap)`** — at least one key per group. Derived from `components` + `bundle_key` (each other component is `one_of(<component>, <bundle_key>)`); a spec may still set `model_validation.one_of` explicitly (Cosmos: `llm_path` or `t5_path`).
4. **`model_validation.by_raw_type`** — optional stricter rules keyed by the raw `type` string (rare).
5. **`FEATURE_GATED_TRAINING_KEYS`** — top-level keys (`blocks_to_swap`, `block_swap_prefetch`, ..., `tread`) only when the matching feature is set; `[tread]` is range-checked against `transformer_blocks`.
6. **`validate_preview_options`** — `[preview]` keys constrained by `preview_options` (e.g. nanosaur2's `guidance_mode`).

SDXL needs no hand-written block in `validation.py`; `checkpoint_path` is required because the spec marks it `required: true`.

## UI visibility

**`rengu_flow_ui/field_visibility.py`** evaluates a `visibility` tree on each schema field; **`ui/web/src/lib/formUtils.ts`** implements the same clause types. The web app has no model lists: everything arrives through `GET /schema`.

| Mechanism | Purpose |
|-----------|---------|
| Per-model `model_fields` | Auto `when: { field: model.type, in: [type_id] }` |
| `ui: false` | TOML-only key (not in the form); can still be validated via `one_of` |
| `{"capability": feature}` / `when_capability` / `unless_capability` | Show or hide a field by the selected model's features (never by model id) |
| `show_if_set` / `visibility` | Expert fields once the user or TOML set a value |
| `pruneFormForModel` | Drop `model.*` keys of other models when `model.type` changes |

`ModelCapability.to_dict()` is an exact whitelist, so the descriptor fields do not change the API payload.

## Adding or changing a model

Follow [adding-a-model.md](adding-a-model.md). In short: package + `spec.py` + one line in `builtin_models.py`; no edits to `validation.py`, `config_schema.py`, `defaults.py`, `local_env.py`, the installer or Vue components. The contract suites in `tests/contract/` (A: spec, D: synthetic model with no other file touched) fail when a surface still keeps a per-type list.

## Example: Cosmos text stack

- Form shows **Qwen3** (`llm_path`, required); **T5** is not in the form (`t5_path` is TOML-only).
- Validation: `one_of: [["llm_path", "t5_path"]]` so expert T5 configs still pass.
- Unknown `model.type` values (including legacy names removed from the registry) fail validation with the list of registered types.

## Example: Block swap

- **Training** `blocks_to_swap` is gated by `features.block_swap` (sdxl, cosmos_predict2, krea2, qwen_image21).
- The validator rejects non-zero `blocks_to_swap` when the model lacks `block_swap` (nanosaur2: 670M fits, no swap).
- **Preview** block swap is separate: `preview.preview_blocks_to_swap`, enabled by `"blocks_to_swap"` in `PreviewSpec.memory_knobs`.

See also [web-ui.md](web-ui.md) (control plane) and user [web-ui.md](../user/web-ui.md) (what validators show in the UI).
