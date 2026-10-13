# Model-spec refactor: one model = one package + one descriptor

Status: **implemented** (2026-10); phase 6 left open. Operational guide: [docs/developer/adding-a-model.md](../developer/adding-a-model.md); the code is the source of truth where this record differs (see [Implementation status](#implementation-status)). Goal: adding a DiT model = a `rengu_flow/model/<m>/` package +
one `spec.py` registered with one line + user docs/examples. Every integration surface
(validation, defaults, env vars, installer, UI schema/help/form, preview dispatch, shared
test coverage) is **derived** from the spec instead of per-type lists scattered across ~65 files.

Hard constraints (all phases):
- No existing user TOML may break; no key renamed/removed; raw alias `anima` stays rejected by
  `validate_config_model_rules` (`registry/model_config_rules.py:208`).
- Exported checkpoints/LoRA unchanged: `export_prefix`, module names, key-conversion tables,
  `save_model` layout.
- Caches stay valid: cache dir uses `model.name` (`data/cache_paths.py:108-127`,
  `data/dataset.py:2496`), text cache key from `text_cache_identity`.
- Core must not depend on the UI's `jobs.db`; specs live in core, UI only reads core.
- Explicit registration (no filesystem autodiscovery).
- Golden surface (`tests/golden/model_surface/*.json`, `tests/test_model_surface_golden.py`)
  must stay identical unless a phase says otherwise (and then the diff is reviewed by hand).

## 0. Findings that shape the design

1. The web (`ui/web/src`) has no model lists: everything comes from `GET /schema`
   (`rengu_flow_ui/app.py:1330`). Its visibility evaluator (`ui/web/src/lib/formUtils.ts:199-259`)
   already understands `{"capability": feature}` / `when_capability`, same as Python
   (`rengu_flow_ui/field_visibility.py:59-143`). Replacing `when: model.type in [...]` with
   capability clauses needs no web change.
2. `ModelCapability` (`registry/model_capabilities.py:157-240`) is already half a descriptor,
   consumed by preflight, validation, text cache, UI visibility, layer groups. **Extend it**, don't
   replace it. `to_dict()` stays an exact whitelist of current keys (`:220-240`) so the web API is
   unchanged.
3. Latent bugs the refactor removes: ad-hoc `anima` alias handling (`config/defaults.py:96` vs
   `:100/:123`, `install/manager.py:146`); `qwen_image21/loading.py:272` imports
   `krea2.loading.QWEN3VL_ASSETS`; TREAD validation duplicated (`krea2/pipeline.py:264-278` vs
   `registry/model_config_rules.py:137-174`); dead `summary` in `FIELD_HELP`
   (`config_field_help.py:1448`); bare `model.*` help keys belong to one specific model so a new
   model silently inherits wrong help; qwen loads the whole file then casts (double RAM peak,
   `qwen_image21/loading.py:96-111`) while krea2 casts per tensor (`krea2/loading.py:139-142`);
   `test_ui_knob_coverage.py:186` only parametrized on qwen_image21; stale docs
   (`.env.example` qwen smoke, `model-pipeline-contract.md:30`).
4. Import constraint: `model/krea2/__init__.py` and `model/qwen_image21/__init__.py` import the
   pipeline (torch/diffusers/transformers). A `spec.py` inside the package requires lazy package
   `__init__` (PEP 562 `__getattr__`), as `rengu_flow/model/__init__.py:16-33` already does.

## 1. Duplication map (R = real duplication, L = legitimate per-model variation)

| Concept | Where today | Type |
|---|---|---|
| Preview steps/guidance per model | `config/defaults.py:129-130`; `rengu_flow_ui/config_schema.py:312-317`; `utils/preview.py:319-320`; `krea2/preview_sampling.py:46-47`; `qwen_image21/preview_sampling.py:123-124`; help `config_field_help.py:333-359` | R |
| Which models preview and with which runner | `utils/preview.py:219-234`; `capability.preview` + `features["preview"]`; `config_schema.py:335` | R |
| DiT/edit/decode preview memory knobs | `preview_form.py:7-13`; `defaults.py:136-139`; `utils/preview.py:212-217, 378` | R |
| Text-cache optional/required | `defaults.py:96-97`; `config_schema.py:687`; `krea2/pipeline.py:101-106`; `qwen_image21/pipeline.py:188-193`; `tests/test_ui_knob_coverage.py:144-151` | R |
| Frozen-base quantization fields | `defaults.py:100-121`; `model_capabilities.py` cosmos/krea2/qwen blocks; help `config_field_help.py:134-163, 207-216, 271-299` | fields/help R, scopes L |
| Component paths, env vars, one_of with bundle folder | `config/local_env.py:17-36`; `model_validation.one_of`; `_component_path` krea2 `124-138` ≈ qwen `221-235`; `.env.example` | R (env = `RENGU_<PREFIX>_<KEY>`) |
| Install profile per model | `install/manager.py:146`; `install/profiles.py:9-56`; `rengu_flow_ui/optional_deps.py:16-20` | R |
| type → module/class map | `registry/models.py:35-41`; `model/__init__.py:5-33`; `@register_model` | R |
| Common DiT model fields (`transformer_dtype`, `diffusion_model_dtype`, `shift`, `sigmoid_scale`, `timestep_sample_method`) | 3 copies in `model_capabilities.py` | structure R, text L |
| Per-field per-model help | ~30 `path@model` entries in `config_field_help.py:89-359` + capability `description` | R |
| TREAD / video gates | `config_schema.py:864, 880, 894, 908, 1075` despite `features["tread"]` | R |
| Pipeline methods (component path, load DiT, cache hooks, VAE `.mode()`, ragged TE return, `prepare_inputs`, `to_layers` loop, block swap, fuse/save, preview lifecycle, `calculate_shift` wrappers) | krea2 vs qwen pipelines (see design §2.3) | R |
| File loading helpers | `krea2/loading.py` vs `qwen_image21/loading.py` (`_require_exists`, prequantized guard, comfy prefix strip, config dir vs bundled, tokenizer) | helpers R, rename tables L |
| Preview sampling utilities | sigmas, offloaded block loop, cfg read/size rounding, denorm+decode→PIL in krea2 vs qwen | utilities R, CFG formula L |
| Pipeline block layer | `krea2/layers.py:82-101` vs `qwen_image21/layers.py:116-142` | wrapper R, inter-layer tuple L |
| Near-duplicate tests | `test_krea2_model.py` vs `test_qwen_image21_model.py`, UI schema tests, config tests, fixed lists in `test_example_configs.py`, `test_examples_validate.py`, `test_cache.py`, `test_session_regressions.py` | R |

## 2. Target design

### 2.1 Descriptor location and registration
- `rengu_flow/model/<m>/spec.py` defines `SPEC = ModelCapability(...)` (sdxl: `rengu_flow/model/sdxl_spec.py`).
  **No torch/diffusers/transformers or model code imports in spec modules.** Package `__init__` lazy.
- `rengu_flow/registry/builtin_models.py`: `BUILTIN_MODEL_SPECS = (... module paths ...)`; tuple order =
  registry order = schema field order (preserve current order; nanosaur2 last).
- `model_capabilities.py` shrinks to the dataclass, adapter templates and the registration loop.
  Keep `model_capability_registry`, `get_capability`, `normalize_model_type`.

New fields on `ModelCapability` (shapes, final names may differ slightly):
```
ComponentPath(key, expect="file"|"dir"|"any", env=True, field={label, placeholder, description, help, doc, importance, show_if_set})
PreviewSpec(runner="sdxl"|"dit", steps, guidance, write_defaults=True,
            memory_knobs=frozenset({"offload_text_encoder","blocks_to_swap"}), edit=False, help={})
QuantSpec(fp8="tensorwise"|"scaled_matmul"|None, nf4=True, grad_mode=True)

pipeline: str            # "rengu_flow.model.krea2.pipeline:Krea2Pipeline"
env_prefix: str          # "SDXL" | "COSMOS" | "KREA2" | "QWEN_IMAGE21" | ...
components: tuple[ComponentPath, ...]
bundle_key: str | None   # one_of derived
text_cache: "optional" | "required"
quant: QuantSpec | None
preview_spec: PreviewSpec | None   # `preview` bool and features["preview"] derive from it
model_defaults: dict
diffusion_dtype_sets_load_dtype: bool
install_profiles: tuple[str, ...]
user_doc: str            # default doc_path for all its fields' help
export_prefix: str       # contract checked against the pipeline class
smoke: SmokeSpec | None
testkit: str | None      # "rengu_flow.model.krea2.testkit"
ragged_cache_keys: frozenset[str]
```
Field library `rengu_flow/registry/field_library.py` (no torch): factories `component_path`,
`bundle_folder`, `transformer_dtype`, `diffusion_model_dtype`, `shift`, `sigmoid_scale`,
`timestep_sample_method`, `quant_fields(QuantSpec)` returning `model_fields` dicts with shared
label/type/options/default/visibility and overridable description/help/doc.

### 2.2 Derivations
| Surface | Derived from |
|---|---|
| `config/defaults.py:90-139` | generic loop over spec (`model_defaults`, `text_cache`, `quant`, `preview_spec`) |
| `config/local_env.py:17-36` | `RENGU_{env_prefix}_{KEY}` for `components` with `env` |
| `model_config_rules` one_of | `components` + `bundle_key` (cosmos keeps explicit one_of) |
| `install/manager.py:146` | `install_profiles` |
| `registry/models.py`, `model/__init__.py` | `pipeline` |
| `utils/preview.py` dispatch | `preview_spec.runner` / `.edit` |
| `preview_form.py`, `config_schema.py` gates | capability features (`preview`, `preview_dit_memory`, `preview_decode_offload`, `edit`, `text_cache_optional`, `tread`, `video`) as `{"capability": ...}` clauses |
| `config_schema.py:312-329` preview defaults | grouping by `preview_spec` (same algorithm) |
| Help | precedence in `enrich_schema`: field spec help/doc > generic `FIELD_HELP[path]` > description; default doc = `user_doc`; `path@model` entries and model-specific bare `model.*` entries removed |
| `cache._SEQUENCE_TENSOR_KEYS` | base keys ∪ specs' `ragged_cache_keys` |

### 2.3 New shared pieces in `dit_common/`
- `loading.py`: `require_exists`, `strip_comfy_prefix`, `guard_not_prequantized`,
  `read_safetensors_cast` (per-tensor cast; fixes qwen RAM peak), `config_dir_or_bundled`,
  `load_file_or_folder`.
- `assets/qwen3vl_tokenizer/` shared (breaks qwen→krea2 import; update `package-data`).
- `DiTPipeline` template methods driven by class attributes (`bundle_key`,
  `requires_cached_text`, quant leaf names/skip substrings, `blocks_attr`, `support_video`,
  `preview_unload_text_encoder`) and small hooks (`_load_transformer`, `_load_vae`,
  `_encode_preview_prompts`): `_component_path`, `load_diffusion_model`, cache hooks,
  `_latent_stats`, block swap, `freeze_text_encoders`, `load_and_fuse_adapter`, `save_model`,
  memoized `preview_prompt_embeds`, `prepare_preview_memory`, `restore_after_preview`,
  `trim_micro_batch`, helper `flow_matching_batch`.
- `preview_sampler.py` (utilities only; the main loop stays per model, ~40 lines):
  `shifted_sigmas`, `apply_cfg`, `seeded_noise`, `run_blocks` (offloader-aware), `preview_size`,
  `decode_to_pil`. A full template sampler would force unifying dtype handling and change numerics.
- `layers.py`: `OffloadedBlockLayer` with a `run_block(inputs)` hook.
- `text.py`: `ragged_rows`. TREAD: `training/token_routing.parse_tread` (no torch), used by both
  `model_config_rules` and `krea2.to_layers`.

### 2.4 Contract tests (free coverage for every registered model)
- Per package `testkit.py`: `tiny_config(tmp_path, **over)`, `tiny_pipeline(config)`,
  `monolithic_forward(pipeline, features)`, `cases()` (t2i/edit/video).
- `tests/contract/` parametrized over canonical types (`importorskip` for missing extras):
  - A (spec only, no torch): internal consistency, docs/examples/NOTICE exist, package-data covers
    assets, examples validate, defaults match spec, model fields visible only in its form with
    help/doc, UI preview defaults == trainer defaults, knob coverage for all models, install
    profile, env overrides, README/.env.example rows.
  - B (tiny pipeline): `to_layers` == monolithic forward, inter-layer tuple tensors only, grads reach
    all params, lora/lokr/lycoris attach to declared scope, golden export keys with
    `spec.export_prefix`, layer groups match real modules, quant scope only block linears,
    deterministic VAE fn, ragged TE fn, `prepare_inputs` target, `trim_micro_batch`, preview e2e
    with CFG on/off memoized, `__init__` doesn't load TE, `save_model` round trip.
  - C (data↔model seam, no doubles): real images → `DatasetManager` cache with tiny VAE/TE →
    `prepare_inputs` → layers → loss/backward, all `cases()`, mixed caption lengths, GAS split.
  - D: `test_synthetic_model_integration.py` registers a toy spec via monkeypatch and checks
    validation, defaults, env, installer, schema, help, preview dispatch and suites A/B pick it up
    with no other file touched.
- Per-model tests that stay: numeric parity with the official reference only
  (`tests/models/test_<m>_parity.py`).

## 3. Migration phases
Every phase: suite green + golden identical unless stated.
- **Phase 0 · golden harness** (`scripts/dump_model_surface.py`, `tests/golden/`): no production change.
- **Phase 1 · specs + registry**: extend `ModelCapability`; move each registration to
  `model/<m>/spec.py` with all new fields filled; lazy `__init__`; `builtin_models.py`; derive
  `models.py`, `model/__init__.py`, `features["preview"]`. Test: importing
  `rengu_flow.registry.model_capabilities` in a subprocess loads no torch/diffusers/transformers.
- **Phase 2a · core derivations** (parallel with 2b/3): defaults, local_env, installer, preview
  dispatch, `_SEQUENCE_TENSOR_KEYS`, `parse_tread`.
- **Phase 2b · UI derivations**: preview sampler defaults, `WHEN_*`, capability gates. Semantic
  golden identical; raw golden re-baselined only for `when` shape (reviewed by hand).
- **Phase 2c · help + field library** (after 2b): move `@model` and model-specific bare entries
  into specs verbatim; `field_library.py`; retire the `@` mechanism. Semantic golden identical incl. help text.
- **Phase 3 · contract tests** (after 1): testkit per model, suites A–D, then delete/shrink the
  duplicated tests; assertion count must not drop without justification. Add tiny numeric goldens
  (export keys, preview hash) before Phase 4.
- **Phase 4 · `dit_common` extraction** (after 3): 4a shared pieces + krea2 + shared tokenizer;
  then 4b qwen_image21, 4c cosmos (partial), 4d nanosaur2. Verify tiny goldens, mini smokes, real
  GPU smoke if weights present (GPU-free check first), qwen RAM peak drops.
- **Phase 5 · tooling + docs**: `scripts/smoke_mini.py --model <m>` over testkits;
  `run_model_smoke.sh` default case `tests/fixtures/smoke/train_${MODEL}.toml`; drift tests for
  README, `.env.example`, package-data, AGENTS.md, THIRD_PARTY_NOTICES;
  `docs/developer/adding-a-model.md`; update `model-capabilities-and-validation.md`,
  `model-pipeline-contract.md`.
- **Phase 6 (optional)**: harmonize common field texts in the library; golden diff reviewed.

## 4. Do not unify (legitimate variation)
Vendored `dit.py` + NOTICE; text encoding recipes (templates, layer taps, drop_idx, padding
position, qwen pre-norm hook); key conversion tables; inter-layer tuple and TREAD layers; shift
constants, terminal stretch, CFG formula/threshold (expressed as data); ecosystem LoRA prefixes;
quantization scopes; SDXL (diffusers UNet: has a spec, no `dit_common`); cosmos video/LLM
adapter/param groups/decode offload; qwen edit logic; per-model user docs and VRAM numbers.

## Implementation status

Phases 0 to 5 are implemented: the golden harness, specs + `builtin_models.py`, the derivations
(defaults, env, installer, preview dispatch, UI gates and help), `field_library.py`, the contract
suites A-D (`tests/contract/`), the `dit_common` extraction (krea2, qwen_image21, nanosaur2;
cosmos partially), and the tooling/docs (`smoke_mini.py`, `run_model_smoke.sh` default fixtures,
drift tests, `adding-a-model.md`). A fifth suite, E (`tests/contract/test_smoke_mini.py`: smoke
tooling follows the registry), was added on top of the design.

**Left out**

- **Phase 6** (harmonize the common field texts of the library, reviewing the golden diff): not
  done. Each spec still carries its own wording for `transformer_dtype`, `shift`, `sigmoid_scale`
  and the quantization fields.
- Cosmos Predict2 still overrides large parts of `DiTPipeline` (video VAE, optional text cache,
  LLM adapter, own fp8 scheme) and keeps an explicit `one_of`.
- `model_capabilities.py` still holds the adapter field templates (LyCORIS groups) and the
  `ensure_default_capability` fallback; adapters have no registry (BACKLOG P0-3).
- `docs/developer/model-pipeline-contract.md` keeps its per-method status table by hand (no drift test).

**Differences between this design and the code**

| Design text | What shipped |
|-------------|--------------|
| `ComponentPath(key, expect, env, field={...})` | `ComponentPath(key, expect, env)` only; the field (label, help, doc) is a regular `model_fields` entry built with `field_library.component_path`. |
| `smoke: SmokeSpec \| None` on the spec | No `SmokeSpec`. The smoke is convention-driven: fixture `tests/fixtures/smoke/train_<type>[_lokr\|_full].toml` (or `SMOKE_FIXTURE_EXEMPT` in the testkit) and env vars derived from `components`. |
| `PreviewSpec(runner, steps, guidance, write_defaults, memory_knobs, edit, help)` | Also `negative_prompt` and `defaults` (e.g. nanosaur2's `guidance_mode`); `ModelCapability` also has `preview_options`, `text_cache_keys`, `text_cache_files`, `text_cache_version`. |
| `QuantSpec(fp8, nf4, grad_mode)` | Also `fp8_dtypes` (`fp8_matmul_dtype` options). |
| `dit_common.loading.read_safetensors_cast` | `read_safetensors(path, dtype)` (casts tensor by tensor when `dtype` is given). |
| Numeric parity tests in `tests/models/test_<m>_parity.py` | Parity against the official reference is a script, `scripts/parity_<m>.py` (needs the reference checkout and real weights, not pytest); per-model pytest files stay as `tests/test_<m>_model.py` / `tests/test_config_<m>.py`. |
| Testkit: `tiny_config`, `tiny_pipeline`, `monolithic_forward`, `cases()` | Also `CAPABILITIES` / `NOT_APPLICABLE` (checks a model cannot honor, with reasons), optional adapter/export/preview hooks and `KNOWN_GAPS`; SDXL's kit is `model/sdxl_testkit.py`. |
| `local_env._MODEL_PATH_ENV` replaced by a derived map | Still readable as a name: a module `__getattr__` derives it from the specs. |
