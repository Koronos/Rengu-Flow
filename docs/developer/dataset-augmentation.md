# Dataset augmentation (developer)

This document is the **specification** for dataset diversity augmentation in Rengu Flow: **canonical `snake_case` string identifiers** for strategies (same style as preset names), typed parameters, presets, cache/seed modes, and integration hooks.

**Implementation status (MVP):** Parsing, preset merge, `apply_augmentation`, cache fingerprinting, metadata branch expansion, and UI fields are implemented under `rengu_flow/data/augmentation/`. Tier A–B strategies plus `horizontal_flip` are available; other catalogue names validate but raise `AugmentationStrategyNotImplementedError`. Video + augmentation is rejected at validation.

## Design principle: string identifiers only

The **user-facing API** uses **strings everywhere** for strategy selection:

- **`preset`** — string, e.g. `"photo_safe"`, `"easy"`.
- **`strategies`** — map from **strategy name** (string key) → parameter table. Merges on top of the preset (override or `enabled = false`).
- **`enable_strategies`** — optional list of strings (strategy names) to restrict which strategies from a preset are active; prefer explicit **`strategies`** blocks when you need parameters.

Do **not** expose numeric strategy indices in config files or user docs.

## Scope

- **Goal:** In-distribution variation (reduce memorisation) without inventing new scenes.
- **Hook point:** RGB (and masks with the same geometric transform) should be augmented inside the same path as `preprocess_media_file_fn` in `rengu_flow/data/manager.py` (`_cache_fn` → `latents_map_fn`), **before** VAE encode, unless a mode explicitly decodes latents each step (unusual).
- **Masks:** Any geometric augmentation must apply consistently to image and mask; photometric augments typically apply to image only.

## Cache and seed modes

| Mode | Behaviour | Latent cache |
|------|-----------|--------------|
| **`deterministic_per_image`** | Hash or path-derived seed; one draw per `(image_spec, augmentation_config_fingerprint)` | Compatible: same tensor every run if config unchanged. |
| **`stochastic`** | New draw each time the image is loaded for encoding | Incompatible with **fixed** precomputed latent cache for that step; requires on-the-fly encode or cache invalidation per epoch. |
| **Offline** | User stores augmented copies as separate files | Standard cache; no special RNG in pipeline. |

**Fingerprinting:** Hash the **fully merged** resolved strategy map (preset defaults + user `strategies` overrides). Include **`variant_sampling`**, per-strategy **`sampling`** (where present), and any **enumerated branch keys** implied by the resolved config, so changing between probability and enumeration invalidates cached latents. Include in `new_fingerprint_args` for `_map_and_cache` / `Cache`.

**Reference:** Kohya documents tension between `--cache_latents` and image augmentations; same principle applies here.

### Interaction with `seed_mode`

- **`deterministic_per_image`:** In **`probability`** variant resolution, one RNG draw per `(image_spec, augmentation fingerprint)` as today. In **`enumerated`** resolution, each materialised branch is a distinct row; seeds should be **stable per branch** (e.g. `(image_spec, variant_key)`) so caches are reproducible.
- **`stochastic`:** Still incompatible with a **fixed** precomputed latent cache for those steps; enumeration does not remove the need for fresh encodes each load if the pipeline stays stochastic.

See [Variant sampling and discrete branches](#variant-sampling-and-discrete-branches) for merge rules and which strategies may use **`enumerated`** per-strategy **`sampling`**.

## TOML schema (implementation contract)

### Global (optional)

```toml
[dataset.augmentation]
enabled = false
preset = "none"
```

### Per directory

Each `[[directory]]` may include an **`augmentation`** table:

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | bool | |
| `preset` | string | Preset name; `none` means no preset defaults (only `strategies` if any). |
| `seed_mode` | string | `deterministic_per_image` \| `stochastic` |
| `variant_sampling` | string | Optional. `probability` \| `enumerated` — how discrete branches become training rows (see [Variant sampling and discrete branches](#variant-sampling-and-discrete-branches)). Default: `probability`. |
| `max_branches_per_image` | int | Optional cap on the number of latent rows per `image_spec` when multiple enumerable strategies combine; if the product exceeds the cap, define **deterministic priority** or **error** (document in implementation). |
| `strategies` | map | Keys = **canonical strategy name strings** (see below); values = parameter tables. |
| `enable_strategies` | [string] | Optional allow-list of strategy names to intersect with a preset; each entry must be a known **`snake_case`** identifier. |

### Merge algorithm (implementers must follow)

1. Load **`preset`** (string) → internal default map `strategy_name → { enabled, params }` (see [Domain presets](#domain-presets-default-strategy-sets)).
2. Deep-merge user **`strategies`**: for each string key, override parameters; **`enabled: false`** removes that strategy from the active pipeline for this directory.
3. If **`preset`** is `none` / missing and **`strategies`** is non-empty → only named strategies with `enabled != false` run (defaults per strategy from schema).
4. If **`preset`** is `custom` → treat like `none` for bundling; user supplies **`strategies`** explicitly.
5. Resolve merge conflicts: user table wins. Validate unknown strategy names → error with suggestion list.
6. If **`enable_strategies`** is set (non-empty list of strings): intersect the resolved preset’s strategy **names** with this list (unknown names → error). Empty intersection → validation error or warning per product rules.
7. Resolve **`variant_sampling`** (directory-level default) with per-strategy **`sampling`** overrides on strategies that declare it (e.g. `horizontal_flip`). Per-strategy wins when set; otherwise inherit directory **`variant_sampling`** for that strategy’s discrete resolution. **`sampling = "enumerated"`** on a strategy that has **no** documented finite branch set → **validation error**.

### Variant sampling and discrete branches

This section defines the **implementation contract** for discrete augmentation branches (complements the [user doc](../user/dataset-augmentation.md#discrete-branches-probability-vs-enumeration)).

| Level | Key | Values | Role |
|-------|-----|--------|------|
| `augmentation` | **`variant_sampling`** | `probability` \| `enumerated` | Global default for how finite discrete strategies expand into rows: sample one branch vs enumerate all documented branches (subject to `max_branches_per_image`). |
| `augmentation` | **`max_branches_per_image`** | positive int, optional | Limits the product of branches across enumerable strategies per `image_spec`; overflow handling must be **documented** (priority order or error). |
| `strategies.<name>` | **`sampling`** (where supported) | `probability` \| `enumerated` | Per-strategy override; same values as **`variant_sampling`**. Only catalogue entries with a **finite** branch set may use **`enumerated`**. |

**Optional synonyms (implementations may accept, not required):** Per-strategy **`sampling = "enumerate"`** as an alias for **`"enumerated"`** (shorter); **`discrete_branch_mode`** with `sample_one` \| `enumerate_all` mapped to **`variant_sampling`** / **`sampling`** — keeps older drafts working.

**Which strategies support `enumerated`:** Only strategies with a **documented finite** discrete set in this spec (e.g. `horizontal_flip` → identity + mirror). **Continuous** parameter spaces (`color_jitter`, unbounded rotations, etc.) do **not** support **`enumerated`** unless a future spec adds an explicit discretisation grid.

**Dataset / cache implications:** The latent path (`latents_map_fn`, `SizeBucketDataset.cache_latents`, `iteration_order`) must allow **multiple rows per `image_spec`** keyed by **`(image_spec, variant_key)`** or equivalent metadata — not a single `latents_idx` per image only. Batching must reference each branch without treating two enumerated branches as duplicate “copies” unless `num_repeats` explicitly increases multiplicity.

**Merge with presets:** Presets imply **`variant_sampling = probability`** unless the merged directory table sets otherwise. Strategy-level **`sampling`** overrides only that strategy’s expansion behaviour.

## Implemented strategies (MVP code)

| Module | Role |
|--------|------|
| `rengu_flow/data/augmentation/config.py` | Merge, validate, fingerprint |
| `rengu_flow/data/augmentation/presets.py` | Preset → default strategies |
| `rengu_flow/data/augmentation/registry.py` | PIL implementations |
| `rengu_flow/data/augmentation/apply.py` | `apply_augmentation()` |
| `rengu_flow/data/augmentation/branches.py` | Enumerated variant keys |
| `rengu_flow/data/preprocess_media.py` | Hook before crop/resize |
| `rengu_flow/data/dataset.py` | Metadata rows + latent fingerprint |

**MVP strategy names:** `horizontal_flip`, `color_jitter`, `gamma`, `jpeg_simulation`, `temperature_tint`, `chromatic_aberration`, `gaussian_noise`, `crop_jitter`, `small_rotation`, `film_grain`, `lab_jitter`, `split_toning`.

**MVP presets:** `none`, `custom`, `easy`, `anime`, `anime_mixed`, `manga_mixed`, `manga_bw`, `photo_safe`, `realism_general`, `bw_photo`, `sepia`.

**Deferred presets (validation error if enabled):** `photo_cinematic`, `retro_scan`, `manga_print`.

**`max_branches_per_image`:** If the product of enumerated branches exceeds the cap, configuration fails with `AugmentationConfigError` (no silent truncation).

## Canonical strategy names and parameters

**Status:** Names not in the MVP list above are specified here but not yet implemented at runtime.

Each **`name`** is the only public identifier (TOML key under `strategies`). Default parameter values are preset-dependent unless overridden.

| name | Parameters (documented contract) |
|------|--------------------------------|
| `horizontal_flip` | `probability` ∈ [0, 1], or `enabled` bool; **`sampling`**: `probability` \| `enumerated` (two branches: identity, horizontal mirror). Per-strategy **`sampling`** overrides directory **`variant_sampling`** for this strategy (see merge step 7). |
| `vertical_flip` | `probability` |
| `color_jitter` | `brightness`, `contrast`, `saturation`, `hue` — non-negative max **delta** per op (torch-style), **not** absolute 0–255 |
| `gamma` | `gamma_min`, `gamma_max` **or** `exposure_ev_range` |
| `gaussian_noise` | `sigma` (0–255 scale) |
| `jpeg_simulation` | `quality_min`, `quality_max` |
| `gaussian_blur` | `sigma_px`, `kernel` (odd) |
| `motion_blur` | `length_px`, `angle_deg` range |
| `unsharp_mask` | `radius_px`, `amount` |
| `channel_dropout` | per-channel `probability` or `desaturate_probability` |
| `temperature_tint` | `warm_cool_range` or RGB multipliers (bounded) |
| `small_rotation` | `max_degrees`, `pad_mode` (e.g. reflect) |
| `scale_translate` | `scale_min`, `scale_max`, `translate_frac_max` |
| `shear` | `max_radians` |
| `perspective` | `strength` (small) |
| `crop_jitter` | `fraction` of shorter side |
| `random_erasing` | `area_fraction`, `count` |
| `clahe` | `clip_limit` |
| `vignette` | `strength` |
| `multiscale_jitter` | `scale_min`, `scale_max` (pre-bucket) |
| `local_tone_mapping` | `strength` |
| `exposure_bracket_fusion` | bracket count, EV spacing, merge weights |
| `bloom` | `threshold`, `radius`, `intensity` |
| `chromatic_aberration` | `shift_px` at edges |
| `lens_distortion` | `k1`, `k2` or barrel strength |
| `film_grain` | `intensity`, correlation |
| `posterize` | `bits` |
| `dithering` | `type`, `strength` |
| `halftone` | `frequency`, `angle` |
| `paper_texture` | `opacity`, noise scale |
| `moire` | `strength` |
| `vhs_analogue` | composite params |
| `crt` | scanlines, curvature |
| `bilateral` | spatial/range sigma |
| `micro_contrast` | `amount` |
| `lab_jitter` | `delta_l`, `delta_a`, `delta_b` max |
| `fft_band` | band gains |
| `split_toning` | shadow/highlight tints |
| `cross_process_lut` | `lut_id` or path |
| `orton` | blend weights |
| `dehaze` | `strength` — use with care |
| `chromatic_noise` | sensor-style |
| `rain_overlay` | `opacity`, density |
| `fog_overlay` | `opacity` |
| `lens_flare` | `intensity` |
| `bokeh_disk` | `radius_px` |
| `sensor_banding` | `strength` |
| `scan_dust` | blob params |
| `screentone` | density, angle |
| `meta_external` | not a pixel op — pipeline hook only |

Implementations should validate ranges per strategy and clamp preset defaults for `anime_mixed` / `manga_mixed` (especially `color_jitter.saturation` and `hue`).

## Priority tiers (for defaults)

| Tier | Strategies (names) | Role |
|------|----------------------|------|
| **A** | `color_jitter`, `gamma`, `jpeg_simulation` (photo), `temperature_tint`, optional mild `chromatic_aberration` | Broad defaults |
| **B** | `gaussian_noise`, `crop_jitter`, `small_rotation`, `film_grain`, `lab_jitter`, `split_toning` | Second wave |
| **C** | `vignette`, `local_tone_mapping`, `bloom`, retro, `cross_process_lut`, weather, `bokeh_disk` | Strong look |
| **D** | `dehaze`, `random_erasing`, VHS/CRT in wrong domain, artefacts | High risk |
| **E** | `meta_external` | Process-level |

## Domain presets (default strategy sets)

Presets are defined as **enabled strategy names (strings) + default parameter structs**.

| Preset | Default strategies (names) | Implementer notes |
|--------|---------------------------|-------------------|
| **`easy`** | `color_jitter` (mild), `gamma` (mild), `temperature_tint` (mild) | No flip, JPEG, or geometry. |
| **`anime_mixed`** | `color_jitter` (tight sat/hue), `gamma`, `temperature_tint`; `horizontal_flip` off or `probability ≈ 0` | Optional `lab_jitter` mild. |
| **`manga_mixed`** | same as anime_mixed, stricter `color_jitter`, no `crop_jitter`/`small_rotation` by default | |
| **`realism_general`** | `horizontal_flip`, `color_jitter`, `gamma`, `jpeg_simulation`, `temperature_tint`; optional `gaussian_noise`, `chromatic_aberration` | Emphasise `gamma` range for HDR scenes. |
| `none` | — | |
| `photo_safe` | `horizontal_flip`, `color_jitter`, `gamma`, `jpeg_simulation`, `temperature_tint` | |
| `photo_cinematic` | photo_safe + mild `local_tone_mapping`, `bloom`, `chromatic_aberration`, `split_toning`, `cross_process_lut` | |
| `anime` | optional `horizontal_flip`, `color_jitter`, `gamma`, `temperature_tint` | |
| `manga_bw` | tight `gamma`; disable strong colour strategies | |
| `bw_photo` | `horizontal_flip`, `gamma`, `temperature_tint`, optional `film_grain` | |
| `sepia` | `gamma`, `temperature_tint`, `split_toning` | |
| `retro_scan` | `paper_texture`, mild `moire`, optional `vhs_analogue`, `scan_dust` | |
| `manga_print` | `dithering`, `halftone`, `screentone` | |
| `custom` | user `strategies` only | |

## Implementation checklist

1. **Done (MVP)** Parse `augmentation` with **`strategies`** as named maps; validate keys against canonical names.
2. **Done (MVP)** Preset → default strategy map merge with user `strategies`.
3. **Done (MVP)** `apply_augmentation()` in `PreprocessMediaFile` (`rengu_flow/data/preprocess_media.py`) before crop/resize.
4. **Done (MVP)** Fingerprint resolved config in `SizeBucketDataset.cache_latents` (`aug_mvp=1` + JSON fingerprint).
5. **Done (MVP)** Metadata expansion with `image_spec` variant suffix; separate latent rows per branch.
6. **Done (MVP)** Tests in `tests/test_augmentation.py`.
7. **Future** Remaining catalogue strategies; video per-frame augmentation; deferred presets.

## References

- Kohya: [sd-scripts docs](https://github.com/kohya-ss/sd-scripts).
- OneTrainer: [GitHub](https://github.com/Nerogar/OneTrainer).
- Albumentations: [Choosing augmentations](https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/).
- FLARE: [BMVC 2024 paper PDF](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_505/paper.pdf).

See also [Dataset and cache](dataset-and-cache.md) and [user: dataset augmentation](../user/dataset-augmentation.md).
