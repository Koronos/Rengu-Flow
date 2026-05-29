# Model capabilities, form visibility, and validation

Training configs are validated in Python (`rengu_flow.config.validation`) and edited in the web UI (`rengu_flow_ui`). Both should agree on **which keys exist for which model**.

## One registry per model

**`rengu_flow/registry/model_capabilities.py`** — `ModelCapability` per pipeline (`sdxl`, `cosmos_predict2`, …):

| Field | Used by |
|-------|---------|
| `model_fields` | UI form (paths under `model.*`), required flags |
| `features` | UI visibility (`when_capability`) and training-key validation |
| `model_validation` | Extra TOML rules (`one_of`, `by_raw_type`) |
| `adapters`, `full_finetune`, `preview` | Adapter section and preview tab |

Register with `register_model_capability()` in the same file (or from a model plugin module that imports it).

## Validation (TOML / CLI / UI Validate button)

**`rengu_flow/registry/model_config_rules.py`** — called from `validate_config()`:

1. **`required_model_keys(cap)`** — every `model_fields` entry with `required: true` and `ui` not `false`, except keys listed in a `one_of` group (those are checked in step 2 instead).
2. **`model_validation.one_of`** — at least one key per group (e.g. `llm_path` or `t5_path`).
3. **`model_validation.by_raw_type`** — optional stricter rules keyed by the raw `type` string in TOML (rare; prefer one canonical type per pipeline).
4. **`FEATURE_GATED_TRAINING_KEYS`** — top-level keys like `blocks_to_swap` only when `features.block_swap` is set.

SDXL does not need a hand-written block in `validation.py`; `checkpoint_path` is required because the capability marks it `required: true`.

## UI visibility

**`rengu_flow_ui/field_visibility.py`** — evaluates a `visibility` tree on each schema field (attached in `get_schema()` → `attach_visibility_to_schema()`).

**`ui/web/src/lib/formUtils.ts`** — same clause types for the SPA.

| Mechanism | Purpose |
|-----------|---------|
| Per-model `model_fields` | Auto `when: { field: model.type, in: [type_id] }` |
| `ui: false` | TOML-only key (not in form); can still be validated via `one_of` |
| `features` + `when_capability` | Show training options only for capable models |
| `show_if_set` / `visibility` | Expert fields after user or TOML already set a value |
| `pruneFormForModel` | Drop `model.*` keys from other models when `model.type` changes |

## Adding or changing a model (checklist)

1. **`@register_model`** in `rengu_flow/model/...` (training code).
2. **`register_model_capability(ModelCapability(...))`** with `model_fields`, `features`, and `model_validation` if needed.
3. **`FIELD_HELP`** in `rengu_flow_ui/config_field_help.py` for new form paths.
4. Cross-section fields in **`rengu_flow_ui/config_schema.py`**: use `when_capability="feature_name"` instead of hard-coding model IDs.
5. **Tests**: `tests/test_config_cosmos_predict2.py` (or new file), `tests/test_field_visibility.py`, `tests/test_model_config_rules.py`.
6. **User doc**: table row in the model’s `docs/user/training-*.md`.

Avoid copying model names into `validation.py` or Vue components.

## Example: Cosmos text stack

- Form shows **Qwen3** (`llm_path`, required); **T5** is not in the form (`t5_path` is TOML-only).
- Validation: `one_of: [["llm_path", "t5_path"]]` so expert T5 configs still pass.
Unknown `model.type` values (including legacy names removed from the registry) fail validation with the list of registered types.

## Example: Block swap (training — not shipped)

- **Training** `blocks_to_swap` is gated by `features.block_swap` on **sdxl** and **cosmos_predict2**.
- Validator rejects non-zero `blocks_to_swap` when the model lacks `block_swap`.
- Cosmos **preview** uses `preview.preview_blocks_to_swap` (separate code path).

See also [web-ui.md](web-ui.md) (control plane) and user [web-ui.md](../user/web-ui.md) (what validators show in the UI).
