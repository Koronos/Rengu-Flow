# Adding or porting a model

**Normative.** This is the flow every new model follows, whether a human or an AI agent does the
work. Read it *before* writing code, follow the phases in order, and do not call a phase done
until its **exit criterion** is met with evidence you have seen (command output, a table, a
`file:line` citation). A model is its package plus one line in `rengu_flow/registry/builtin_models.py`;
every integration surface (validation, defaults, env vars, installer, UI form and help, preview
dispatch, shared tests) is **derived from the model's spec**. If something does not derive, extend
the spec machinery; never add a per-model list to config, UI or installer code.

The design record is [the model-spec refactor](../spec/model-spec-refactor.md). Companion pages:
[Model pipeline contract](model-pipeline-contract.md), [Model capabilities and
validation](model-capabilities-and-validation.md), [Testing](testing.md), [Smoke
tests](smoke-tests.md).

## The model in one picture

```
rengu_flow/model/<m>/
    __init__.py        lazy (PEP 562) exports: importing the package must not import torch
    spec.py            SPEC = ModelCapability(...)        torch-free; the single descriptor
    pipeline.py        class <M>Pipeline(DiTPipeline)     @register_model("<m>"); hooks only
    dit.py / vae.py / text.py / layers.py / loading.py / preview_sampling.py   model code
    testkit.py         tiny random model for the contract suites and smoke_mini.py
    NOTICE.md          license + provenance of vendored code
registry/builtin_models.py     + one line: "rengu_flow.model.<m>.spec"
docs/user/training-<m>.md, examples/minimal_config_<m>_*.toml, tests/fixtures/smoke/train_<m>*.toml
scripts/parity_<m>.py          numeric parity against the official reference
```

Reference implementation to read for *structure* (not to copy from): `rengu_flow/model/nanosaur2/`
(the latest model, small, complete end to end) and `krea2/spec.py` (quantization knobs, TREAD,
bundle folder). SDXL is the non-DiT exception: it has a spec (`model/sdxl_spec.py`) but no `dit_common`.

---

## Phase 0 - Mandatory reading

Read, in this order: this guide; [model-pipeline-contract.md](model-pipeline-contract.md) (what the
orchestrator and `DatasetManager` expect of a pipeline); the **seam rule** in `AGENTS.md` (a change
touching data and model needs a test with no doubles); one sibling model's package as a model of
*structure*. Then decide the **type id** (snake_case, the value of `[model] type`; the raw alias
`anima` stays rejected) and the **`env_prefix`** (`RENGU_<PREFIX>_<KEY>` env overrides).

**Exit:** type id and `env_prefix` chosen and not already used (`BUILTIN_MODEL_SPECS` and the
env prefixes of the existing specs; `test_env_prefixes_unique` enforces the latter).

## Phase 1 - Research, with the official reference code as the source of truth

Pin the reference (repo + **commit/revision**) and read it. Do not rely on a paper, a blog post or
another port when the official code exists. Answer, citing `file:line` of the reference for each:

| Topic | What you must establish |
|-------|-------------------------|
| Architecture | Blocks, widths, heads, RoPE, modulation, the forward signature. Do dims vary between released variants? (They must be **detected from the checkpoint**, see Phase 3.) |
| VAE | Is the **encoder** published? Without an encoder the model cannot be trained. Latent mean/std (or scale/shift), spatial compression, latent channels, normalization applied at encode and at decode. |
| Text encoder / tokenizer | Prompt template, which layer(s) are tapped, padding side and length, max tokens, whether the checkpoint is gated, where the tokenizer lives (separate files or inside the weights). |
| Objective | Velocity / epsilon / x0 prediction; timestep distribution (logit-normal mean/std, uniform), shift (fixed or resolution-dependent), loss and its per-timestep weighting, caption dropout. |
| Sampler | Sigma schedule, solver, CFG formula and where the negative prompt goes, default steps and guidance. |
| Ecosystem LoRA format | Key prefix and module names ComfyUI / diffusers expect, `.alpha` convention. |
| License | License of code and weights, third-party code inside (e.g. GPL in a reference repo means *do not copy code*, only the parameter-name correspondence), gating. |
| Downloads | Hugging Face direct links and sizes of the **bf16** (not pre-quantized, not distilled) files, **Comfy-Org / ComfyUI single files first** (see [Weights convention](#weights-convention)); the official diffusers repo is the alternative. |

**Exit:** every number you will put in code, a default or a doc is cited with `file:line` of the
pinned reference (keep the table in your report; the commit hash goes into `NOTICE.md`).

### Weights convention

For now we use the **Comfy-Org (ComfyUI) bf16 single files**. This is the rule for every model:

- Local weights for tests, smokes and docs are the ComfyUI / Comfy-Org bf16 files, kept in the
  git-ignored `tmp/models/<model>/` and pointed at from `.env` (`RENGU_<PREFIX>_<KEY>_PATH`). The user
  guide's files table, the examples and the smoke fixtures present them as the primary weights; the
  diffusers download is the documented alternative.
- A diffusers download of the official repo is only **temporary**, when parity must load the reference
  as-is (Phase 2). Delete its weights afterwards; keep what the tests still read (the reference code,
  configs, tokenizer, index jsons), which must be enough for the checks that need no weights.
- The loaders accept both layouts (Phase 3), and the layout of every Comfy-Org file is verified against
  its real safetensors header (an HTTP range request), never assumed.
- Documentation of all of this follows [documentation-conventions.md](documentation-conventions.md).

## Phase 2 - Numeric parity before integrating

Write `scripts/parity_<m>.py` (model: `scripts/parity_nanosaur2.py`). It is **not** a pytest: it
needs the reference checkout and real weights, and runs in an ephemeral `uv run --with ...`
overlay so rengu's environment is untouched. It compares, on real weights, the max abs error and
the relative error `max|a-b| / max|b|` of:

- text encoder outputs for several prompts (empty, emphasis, long);
- VAE encode and decode (the encode path is what training uses);
- the DiT forward in float32 against the reference, including every branch training exercises;
- the sigma / timestep schedule of the previews;
- an exported adapter loaded by the ecosystem's own loader (every key consumed).

Document the tolerances in the script header (float32 relative error `< 1e-3` is the convention).

The reference loads the official (diffusers) layout, so this is the one place a temporary diffusers
download is used; the rengu side loads the Comfy-Org files of the [Weights convention](#weights-convention)
where they exist. Checks that need no weights (schedules, adapter key maps) run from the small files alone.

**Exit:** a printed table of errors, all within tolerance, pasted in the report. If a number is
out of tolerance, fix the port; do not widen the tolerance.

## Phase 3 - Port, reusing `dit_common`

1. **Subclass `DiTPipeline`** (`rengu_flow/model/dit_common/pipeline.py`) and implement hooks, not
   copies. The template already provides construction (`_init_pipeline`, `_component_path`),
   `load_diffusion_model` (+ optional frozen-base quantization), the cache fns' skeleton, block
   swap, adapters (`configure_adapter` / `save_adapter` / `load_adapter_weights` through
   `networks.adapter_dit`), `save_model`, the masked fp32 `get_loss_fn`, and the whole preview
   memory lifecycle. The tables below list the hooks and class attributes.
2. **Dims are detected from the checkpoint** (`detect_config(state_dict)`-style), never hardcoded:
   one package must load every released variant.
3. **Module and parameter names equal the official checkpoint**, so release files load verbatim
   and exports keep the official layout (this is also what ecosystem LoRA loaders match against).
4. **If you are about to copy code from another model, stop.** Extract it first to
   `rengu_flow/model/dit_common/` and make the existing models use it. The extraction ships with:
   a numeric golden that was green *before* the move and stays green after (`tests/golden/model_numeric/`,
   suite B), and a bit-exact snapshot comparison of every model that uses the piece
   (`tests/test_dit_common.py` is the pattern). Only code that is identical across models belongs
   in common; per-model variation (text recipes, key tables, shift constants, CFG formula,
   quantization scopes, inter-layer tuples) stays in the package.
5. `__init__.py` is lazy (copy `nanosaur2/__init__.py`) so `spec.py` imports without torch.
6. Rengu never downloads or resolves repo ids at runtime: every component is a local path the user
   already has. A vendored bundle's assets (tokenizer, configs) live under the package's `assets/`
   and are declared as package data in `pyproject.toml` (`"assets/**/*"`).

**`DiTPipeline` hooks and attributes you will touch**

| You implement | Purpose |
|---------------|---------|
| `_load_transformer(dtype)`, `_load_vae()`, `_encode_text(text_encoder, prompts, device)` | Load the DiT / frozen VAE; encode captions to `(embeds, mask)` for both the cache and the previews. |
| `get_call_vae_fn`, `get_call_text_encoder_fn` | Override only if the defaults (posterior mode of a `T = 1` image VAE; ragged rows) do not fit (nanosaur2: own latent and `token_weights`). |
| `prepare_inputs(inputs, timestep_quantile=None)` | Cached batch to `(features, (target, mask))`: sample `t`, `dit_common.add_flow_noise`, build the loss weight. |
| `to_layers()` | Pipe layers over a tensors-only inter-layer tuple (use `OffloadedBlockLayer`, `mark_inter_layer_grads`). |
| `generate_preview_image(preview_cfg, prompt, step, seed)`, `preview_prompt_embeds` | The model's own ~40-line Euler loop built from `dit_common.preview_sampler` utilities. |
| `save_model(save_dir, state_dict)` | Only if the full-finetune export differs from the diffusers transformer folder. |

| You set (class attribute) | Meaning |
|---------------------------|---------|
| `name` (+ `@register_model("<m>")`) | The type id. Also names the cache dir, so never rename it. |
| `adapter_target_modules`, `adapter_layer_groups`, `adapter_export_prefix`, `adapter_export_lora_alpha` | Adapter scope, named groups (must equal the spec's `adapter_layer_groups`), export key prefix (must equal `spec.export_prefix`), `.alpha` tensors. |
| `blocks_attr`, `quant_leaf_names`, `quant_skip_substrings`, `quant_label` | Block list for swap/streaming; frozen-base quantization scope (`None` = no quantization). |
| `text_cache_required_reason`, `missing_component_message` | Error wording. |
| `preview_keep_reloaded_vae`, `preview_unload_freed_text_encoder` | Preview memory policy. |
| `trim_micro_batch = text_padding_trimmer()` | Opt in only if features carry right-padded text lanes. |
| `default_uncond_fraction`, `pixels_round_to_multiple`, `vae_spatial_compression`, `checkpointable_layers` | Data and checkpointing contract. |

Shared building blocks (`dit_common`): `flow_matching` (`sample_timesteps`, `shift_timesteps`,
`calculate_shift`, `add_flow_noise`), `text` (`compact/pad/ragged_rows`, `trim_text_padding`),
`loading` (`require_exists`, `read_safetensors`, `guard_not_prequantized`, `strip_comfy_prefix`,
`config_dir_or_bundled`, `load_file_or_folder`), `quantize` (`quantize_frozen_dit`), `layers`
(`OffloadedBlockLayer`), `preview` / `preview_sampler` (`preview_size`, `shifted_sigmas`,
`apply_cfg`, `seeded_noise`, `run_blocks`, `decode_to_pil`), `qwen3vl` (shared Qwen3-VL text
encoder + bundled tokenizer), `streaming` (`LazyStreamedEncoder` for encoders larger than the GPU).

The training graph must be **the same graph as inference**: if `to_layers()` skips or reorders
something the model's own forward does, the contract (suite B) fails, and so would your results.

**Exit:** the pipeline instantiates on the real weights, and `scripts/parity_<m>.py` still passes
against the ported code (re-run it; paste the table).

## Phase 4 - Declare the model: `spec.py` and one registration line

`spec.py` is a `ModelCapability` (`registry/model_spec.py`). It must import no torch, diffusers,
transformers or model code (`field_library`, `model_spec` and `networks.lycoris_meta` only);
`tests/test_registry_lazy.py::test_model_specs_import_without_torch` imports the registry in a subprocess and asserts torch/diffusers/transformers are not loaded.
Fill it field by field (see `nanosaur2/spec.py` as the compact example):

| Spec field | What to put there |
|------------|-------------------|
| `type_id`, `display_name`, `branding_note` | Id; UI name; which exact files to use (and which not, e.g. distilled). |
| `adapters`, `full_finetune`, `adapter_layer_groups`, `adapter_module_roots`, `transformer_blocks` | Offered training modes; named groups equal to the pipeline's `ADAPTER_LAYER_GROUPS`; top-level module names for torch-free target-typo preflight; block count for `[tread]` range checks (0 = unknown). `features={LINEAR_ONLY_ADAPTERS: True}` when the net has no Conv / affine norms. |
| `pipeline`, `testkit` | `"<module>:<Class>"` (imported lazily) and the testkit module path. |
| `env_prefix`, `components`, `bundle_key` | `ComponentPath(key, expect="file"/"dir"/"any")` per weights path; env var = `RENGU_<PREFIX>_<KEY>`; `bundle_key` names a diffusers-layout folder that fills any empty component (then `model_validation.one_of` is derived; do not write it). |
| `text_cache`, `text_cache_keys`, `text_cache_files`, `text_cache_version`, `ragged_cache_keys` | `"optional"` or `"required"` (when the encoder must not run in the training graph); every built-in declares one (left `None`, as for a pipeline registered only with `@register_model`, no `cache_text_embeddings` default is written); which `[model]` keys and files key the text cache (so swapping the encoder re-encodes); cached tensors stored as ragged rows (e.g. `token_weights`). |
| `quant`, `model_defaults`, `diffusion_dtype_sets_load_dtype` | `QuantSpec(...)` if the pipeline sets `quant_*` attributes; per-model `[model]` defaults (**each must be a field with that default**); load-dtype coupling. |
| `preview_spec`, `preview_options` | `PreviewSpec(runner="dit", steps, guidance, memory_knobs, negative_prompt, defaults, help)` with the **reference** sampler defaults; fixed-value `[preview]` keys. Features such as `preview`, `edit`, `preview_dit_memory`, `text_cache_optional` are **derived** in `ModelCapability.__post_init__`; do not list them. |
| `install_profiles`, `user_doc`, `export_prefix`, `adapter_exports` | Extras the run needs; the user guide (default doc link of every field's help); adapter export key prefix; per-adapter export layout `AdapterExport(file, style="dotted"/"kohya", roots)` when it is not the DiT default (`adapter_model.safetensors`, dotted keys under `export_prefix`) — SDXL declares its kohya `lora.safetensors`. The export check reads it (`spec.adapter_export(kind)`). |
| `model_fields` | The form and validation fields, built with `registry/field_library.py` factories (`component_path`, `bundle_folder`, `transformer_dtype`, `diffusion_model_dtype`, `shift`, `sigmoid_scale`, `timestep_sample_method`, `quant_fields`) plus plain dicts for model-specific keys. |

Rules for fields and help (also enforced by suite A):

- **The help is the model's own.** Pass `description` / `help` / `doc` to every factory; the
  factory holds structure (path, label, type, options), never wording. Never reuse a sibling's text
  and never write a `path@model` entry in `rengu_flow_ui/config_field_help.py` (that mechanism was
  removed). Precedence in `enrich_schema`: field `help`/`doc` > generic `FIELD_HELP[path]` > `description`.
  Follow the hint rules in [documentation-conventions.md](documentation-conventions.md).
- **Defaults are the official recipe** (cited in Phase 1), not "reasonable" values. A default you
  cannot cite does not go in; leave the field unset.
- Every `model_defaults` key and every key the pipeline reads from `[model]` must be a form field
  (`test_every_key_the_model_reads_is_on_its_form`).

Then add **one line** to `BUILTIN_MODEL_SPECS` in `registry/builtin_models.py` (tuple order =
registry order = schema field order; append).

**Exit:** `uv run --no-sync pytest -q tests/contract/test_spec_contract.py` is green for the new type
(it reports the missing deliverables of Phase 8 until they exist; that is expected here, but
everything else must pass).

## Phase 5 - Tests

1. **`testkit.py`** (interface documented in `rengu_flow/model/testkit_common.py`; model:
   `nanosaur2/testkit.py`). Required: `tiny_config(tmp_path, **over)` (writes tiny random weights
   *in the real release formats, through the real loaders*), `tiny_pipeline(config)`,
   `monolithic_forward(pipeline, features)`, `cases()` (`ModelCase` per t2i / edit / video, each with
   a real-media `write_dataset` unless there is no CPU-encodable media), `CAPABILITIES` and
   `NOT_APPLICABLE`. `NOT_APPLICABLE` maps each contract check the model cannot honor to a **real,
   specific reason** (`CAPABILITIES == ALL_CAPABILITIES - NOT_APPLICABLE.keys()` is checked); an empty
   dict means every check applies, which is the goal. Optional hooks: `denoiser`, `adapter_networks`,
   `adapter_scope`, `ADAPTER_SCOPE_EXCLUDES`, `reload_export` / `export_transformer_path`,
   `preview_config`, `RAGGED_CAPTIONS`, `SMOKE_FIXTURE_EXEMPT`, `KNOWN_GAPS`. Kit modules import torch
   only inside functions.
2. **Suites B and C run for free** once the spec names the kit (`tests/contract/`):
   - **A** `test_spec_contract.py`: the spec against everything derived from it (no torch).
   - **B** `test_pipeline_contract.py`: the real pipeline over the tiny kit: layers equal the model's
     own forward, gradients reach every trainable parameter, adapter scope and golden export keys,
     layer groups, quantization scope, VAE/TE cache fns, flow-matching target, previews, `save_model` round trip.
   - **C** `test_data_model_seam.py`: real images through the real `Dataset`/`DatasetManager` cache,
     the loader's micro-batch split, `prepare_inputs`, every layer, loss and backward, for every
     `cases()` entry. **No doubles.** This is the test that would have caught qwen_image21's step-3 failure.
   - **D** `test_synthetic_model_integration.py` over `tests/contract/toy_model/`: proves a model
     added as package + spec + one line reaches every surface. If you had to touch a shared file to
     make your model work, D is the place that shows a per-type list is still hiding.
   - **E** `test_smoke_mini.py`: your kit runs in `smoke_mini.py`; a smoke fixture exists or the kit
     declares `SMOKE_FIXTURE_EXEMPT`; docs use generic smoke commands.
3. **Your own tests** only for what is specific: the parity of a converter, a loader's key
   renames, a text recipe against a literal copy of the reference (`tests/test_<m>_model.py`,
   `tests/test_config_<m>.py`). Do not re-test what the contract suites parametrize.
4. **Goldens.** Numeric goldens (`tests/golden/model_numeric/<type>.json`, via the `golden` fixture)
   and surface goldens (`tests/golden/model_surface/<type>.json`, `tests/test_model_surface_golden.py`,
   dumped by `scripts/dump_model_surface.py`) are created by your first run with
   `RENGU_UPDATE_GOLDEN=1 uv run --no-sync pytest <files> -n 0`. For existing models a golden that
   moves means behavior moved: **read the diff of `tests/golden/` and justify every changed line
   before accepting**; never regenerate to make a red test green.

**Exit:** `uv run --no-sync pytest -q tests/contract tests/test_model_surface_golden.py` green for
the new type, then the whole suite (see [Testing](testing.md); needs the `dev` and `cosmos_predict2`
extras for the DiT suites).

## Phase 6 - Mini smoke (no weights)

```bash
python scripts/smoke_mini.py --model <model> --adapter lora
python scripts/smoke_mini.py --model <model> --adapter lokr
python scripts/smoke_mini.py --model <model> --adapter full
```

It trains the tiny kit a few steps down the real path (dataset, cache, layers, loss, optimizer),
renders a preview if the spec has one, exports and checks the export (the file and key layout of
`spec.adapter_export(kind)`, factor pairs, finite tensors, full-model reload). Native Windows or Linux, CPU.

**Exit:** all three exit 0; paste the summaries.

## Phase 7 - Real GPU smoke

The GPU is shared with other work. **Before every GPU run**, in its **own command**, read the
result, and only then launch:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

If another compute process is listed, or `memory.used` is more than a few hundred MB, **do not run**;
report it. Never kill a process you did not start. Under WSL2 the per-process query can return
nothing even when the GPU is busy (see [WSL/Windows workflow](wsl-windows-workflow.md)), so the
memory figure is the second half of the check. Never chain the check with the run
(`nvidia-smi ...; scripts/run_model_smoke.sh ...`): chaining is how an unread check let a full
finetune start on a busy GPU.

Then, with the Comfy-Org bf16 weights (`tmp/models/<model>/`, see the [Weights convention](#weights-convention)) in `.env` as `RENGU_<PREFIX>_<KEY>_PATH` (documented in `.env.example`):

```bash
scripts/run_model_smoke.sh <model>          # tests/fixtures/smoke/train_<model>.toml
scripts/run_model_smoke.sh <model>_lokr
scripts/run_model_smoke.sh <model>_full
```

Exit 77 = weights missing (skip). Read the peak VRAM by torch `cuda_peak` in the bench output, not
`nvidia-smi`. **Look at the preview image**: a picture of noise is a bug even when the loss is
finite. Finally load the exported adapter in the ecosystem tool (ComfyUI / diffusers) and generate.

**Exit:** each smoke exits 0, the export check passes, preview PNG looks like an image, VRAM
numbers recorded for the user doc, ecosystem load confirmed (or stated as not done).

## Phase 8 - User docs, examples, notices

All of these are checked by suite A or `tests/test_smoke_conventions.py`; none is optional.

| Deliverable | Where |
|-------------|-------|
| User guide with a files table (component, config key, `.env` variable, **direct download link**, size), recipe, measured VRAM, what not to download | `docs/user/training-<m>.md` (model: `training-nanosaur2.md`); linked from `docs/README.md` |
| Examples (adapter and finetune) and their dataset | `examples/minimal_config_<m>_{lora,lokr,finetune}.toml`, `examples/minimal_<m>_dataset.toml` |
| README model-table row | `README.md` |
| `.env.example` block with every component env var | `.env.example` |
| Provenance and license of vendored code | `rengu_flow/model/<m>/NOTICE.md` + link in `THIRD_PARTY_NOTICES.md` |
| Package data for bundled assets | `pyproject.toml` `[tool.setuptools.package-data]` |
| Smoke fixtures (no `[model]` paths) | `tests/fixtures/smoke/train_<m>.toml`, `_lokr`, `_full` |
| Developer notes for model specifics | a section in [model-pipeline-contract.md](model-pipeline-contract.md) |

**Exit:** the final checklist below is all green.

---

## Final checklist (each item names the proof)

| # | Item | Proof |
|---|------|-------|
| 1 | Spec is internally consistent (components, bundle `one_of`, defaults are fields, quant fields, kit exists) | `test_spec_contract.py::test_spec_internal_consistency` |
| 2 | Registered in order; env prefix unique | `test_builtin_specs_are_registered_in_order`, `test_env_prefixes_unique` |
| 3 | Pipeline resolves to the registered class; `export_prefix` and layer groups match it | `test_pipeline_resolves_to_registered_class`, `test_export_prefix_matches_pipeline`, `test_pipeline_contract.py::test_layer_groups_and_module_roots_match_the_real_dit` |
| 4 | Kit declares `NOT_APPLICABLE` with real reasons | `test_kit_declarations_are_consistent` |
| 5 | Examples exist, validate, fail without their required keys | `test_examples_exist_validate_and_need_their_required_keys` |
| 6 | Validation, defaults, env overrides, installer, ragged keys derive from the spec | `test_validation_derives_from_required_keys_and_one_of_groups`, `test_one_of_groups_derived_from_components_and_bundle`, `test_defaults_follow_the_spec`, `test_env_overrides_fill_the_component_paths`, `test_install_profiles_follow_the_spec`, `test_ragged_cache_keys_reach_the_cache`, `test_quant_knobs_validate` |
| 7 | Preview dispatch and UI features follow the spec; preview defaults equal the trainer's | `test_preview_dispatch_follows_the_spec`, `test_ui_features_derive_from_the_spec`, `test_ui_preview_defaults_equal_the_trainer_defaults`, `test_preview_option_fields_follow_the_spec` |
| 8 | Model fields show only on their form, with help and doc; every key the model reads is on the form | `test_model_fields_show_only_on_their_form_with_help_and_doc`, `test_every_key_the_model_reads_is_on_its_form`, `test_form_shows_one_field_per_path`, `test_component_fields_follow_the_components` |
| 9 | Docs, NOTICE (+ THIRD_PARTY link), package-data, README row, `.env.example` | `test_docs_notice_and_package_data`, `test_readme_row_and_env_example_block` |
| 10 | Training graph equals the model's forward; gradients reach everything | `test_layers_match_the_monolithic_forward`, `test_gradients_reach_every_trainable_parameter` |
| 11 | Adapters attach to the declared scope, export golden keys, reload | `test_adapter_attaches_to_the_declared_scope_and_exports_golden_keys`, `test_scope_exclusions_are_frozen_by_default_in_the_other_families` |
| 12 | Cache fns: deterministic VAE, ragged batch-independent TE; flow target; trim | `test_vae_fn_is_deterministic`, `test_text_encoder_fn_rows_are_batch_independent`, `test_text_encoder_fn_caches_ragged_rows`, `test_prepare_inputs_builds_the_flow_matching_target`, `test_trim_micro_batch_drops_unused_text_lanes_without_changing_the_output` |
| 13 | Quantization touches only block linears | `test_quantization_touches_only_block_linears` |
| 14 | Previews end to end (CFG on/off, memoized, spec runner); lazy TE; export round trip | `test_preview_end_to_end_memoizes_prompt_embeddings`, `test_previews_render_through_the_spec_runner`, `test_init_does_not_load_the_text_encoder`, `test_save_model_round_trips_through_the_loader` |
| 15 | **Seam without doubles**, every case | `test_data_model_seam.py::test_dataset_batches_train_through_the_layers` |
| 16 | Nothing per-model leaked into shared code | `test_synthetic_model_integration.py` (all tests), `test_model_surface_golden.py` diffs only in the new model's files |
| 17 | Mini smoke and smoke fixture | `scripts/smoke_mini.py --model <model>` (lora/lokr/full); `test_smoke_mini.py::test_every_testkit_model_runs_in_smoke_mini`, `test_model_has_a_smoke_fixture_or_declares_the_exemption`; `tests/test_smoke_conventions.py` |
| 18 | Spec imports no torch | `tests/test_registry_lazy.py::test_model_specs_import_without_torch` |
| 19 | Numeric parity vs the reference on real weights | `scripts/parity_<m>.py` table in the report |
| 20 | Real GPU smoke, VRAM by `cuda_peak`, preview looked at, export loaded in the ecosystem | `scripts/run_model_smoke.sh <model>` output; screenshot/description of the preview |
| 21 | Goldens changed only where intended | `git diff tests/golden` reviewed line by line |
| 22 | Whole suite green | `uv run --no-sync pytest -q` (with the `dev` and `cosmos_predict2` extras installed) |

## Anti-patterns

| Do not | Why / incident |
|--------|----------------|
| Copy-paste from another model instead of extracting to `dit_common` | Divergent copies drift (the model-specific integration was scattered over ~65 files before the refactor and krea2/qwen_image21 pipelines duplicated most methods; one cast the checkpoint whole and doubled the RAM peak, the other cast per tensor). Extract first, with a golden. |
| Hardcode dims | Breaks the other released variants. Detect from the checkpoint. |
| Invent defaults | The default is the official recipe or it is unset. Cite `file:line`. |
| Report green without parity and smoke | Unit tests with tiny random weights cannot show a wrong constant, a wrong tap layer or a noise preview. Phases 2 and 7 exist for that. |
| Edit per-type lists in config, UI or the installer | Everything derives from the spec. If something does not, extend the spec/derivation (and add the surface to suite D), never a list. The pre-refactor ad-hoc `anima` alias handling (defaults vs installer) was a latent bug of this kind. |
| `path@model` or inherited help text | Nanosaur2 nearly shipped with Cosmos/Krea2 help on its own fields (a new model silently inherits a bare `model.*` key's text). Pass the model's own `description`/`help`/`doc` to every factory. |
| Test the data-model seam with stubs | qwen_image21 edit: 2117 green tests and the first real run failed at step 3 on a data/model misalignment. Suite C feeds real batches from the real data path into the real tiny pipeline for every case. |
| `import torch` (or any model code) in `spec.py` / non-lazy package `__init__` | Config validation, the installer and the UI read specs without torch; a torch import costs seconds and breaks torch-free environments. |
| Download or resolve repo ids at runtime | Rengu only loads local paths. |
| Regenerate goldens without reading the diff | A moved golden is a behavior change. Regenerating to turn a test green hides the regression the golden exists to catch. |
| Run on a shared GPU without a checked, read, separate query | Incident (nanosaur2 smoke): the check was chained with the run, its output unread, and a full finetune started with foreign processes holding 5.4 GB. |
| Training graph different from inference | The SDXL mid-block bug: the training graph differed from the inference graph and contract B (`test_layers_match_the_monolithic_forward`) found it. Never "simplify" `to_layers()`. |
| Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, or measure VRAM with `nvidia-smi` | Crashes under WSL2/WDDM; `nvidia-smi` shows the whole GPU. Use `cuda_peak`. |
| Pre-quantized or distilled checkpoints as the training base | Cannot be trained (`guard_not_prequantized`) or train the wrong thing; the spec's `branding_note` and the doc say which file to use. |
| Copy GPL reference code | Port from the algorithm; keep only parameter-name correspondence and say so in `NOTICE.md`. |
| Commit before the checklist is green | Evidence first. |

## Where does X go?

| Concept | File |
|---------|------|
| Model descriptor (fields, components, preview, quant, defaults, env prefix) | `rengu_flow/model/<m>/spec.py` |
| Registration | `rengu_flow/registry/builtin_models.py` (one line) |
| Shared field factories (path, dtype, shift, sigmoid scale, sample method, quant fields) | `rengu_flow/registry/field_library.py` |
| Descriptor types (`ModelCapability`, `ComponentPath`, `QuantSpec`, `PreviewSpec`) | `rengu_flow/registry/model_spec.py` |
| Registry, adapter field templates, `get_capability` | `rengu_flow/registry/model_capabilities.py` |
| Cross-field validation derived from the spec | `rengu_flow/registry/model_config_rules.py` |
| Type to pipeline class resolution (`@register_model`) | `rengu_flow/registry/models.py` |
| Pipeline | `rengu_flow/model/<m>/pipeline.py` (subclass of `dit_common.DiTPipeline`) |
| Code identical across models | `rengu_flow/model/dit_common/` |
| Vendored net / VAE / text encoder | `rengu_flow/model/<m>/{dit,vae,text}.py` + `NOTICE.md` |
| Pipe layers | `rengu_flow/model/<m>/layers.py` |
| Preview sampler loop | `rengu_flow/model/<m>/preview_sampling.py` (+ `dit_common/preview_sampler.py` utilities) |
| Checkpoint key conversion, file loading | `rengu_flow/model/<m>/loading.py` (+ `dit_common/loading.py`) |
| Adapter code (LoRA / LoKr / LyCORIS) | `rengu_flow/networks/` (not per model) |
| Env overrides for paths | derived: `config/local_env.py` from `components` + `env_prefix`; list them in `.env.example` |
| Defaults applied to the config | derived: `config/defaults.py` from the spec |
| Install profiles | `install_profiles` in the spec (`install/profiles.py` defines the profiles) |
| UI form and help | the spec's `model_fields` (+ generic `FIELD_HELP` in `rengu_flow_ui/config_field_help.py` only for model-independent paths) |
| Tiny test model | `rengu_flow/model/<m>/testkit.py` (+ `model/testkit_common.py`) |
| Contract tests | `tests/contract/` (do not add per-model copies) |
| Model-specific tests | `tests/test_<m>_model.py`, `tests/test_config_<m>.py` |
| Goldens | `tests/golden/model_numeric/`, `tests/golden/model_surface/`, `tests/golden/legacy_configs/` |
| Parity script | `scripts/parity_<m>.py` |
| Smoke | fixtures `tests/fixtures/smoke/train_<m>*.toml`; runners `scripts/run_model_smoke.sh`, `scripts/smoke_mini.py` (no per-model cases) |
| User guide, examples, README row, `.env.example`, notices | see Phase 8 |
