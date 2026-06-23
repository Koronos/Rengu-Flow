# Dataset augmentation (diversity)

This page describes dataset augmentation in Rengu Flow: small in-distribution variations on training images to reduce overfitting to a single pixel–caption pair.

**Status: implemented (MVP)** — Tier A–B strategies, core presets, `deterministic_per_image` seeding. **Video is not supported** with augmentation enabled in this release (use `frame_buckets = [1]` only). Presets `photo_cinematic`, `retro_scan`, and `manga_print` are documented but not available until their extra strategies are implemented.

Augmentation applies **per directory** (each `[[directory]]` can choose a different profile). It is conceptually similar to multi-scale copies and to tools that apply mild geometric or colour jitter (see [References](#references)).

## String identifiers only

Configuration is **human-first**. All public identifiers are **strings** (same idea as preset names such as `"photo_safe"` or `"easy"`):

- Each transform has a **stable `snake_case` name** (e.g. `color_jitter`, `horizontal_flip`, `small_rotation`). There is **no** numbered strategy list in the user-facing API.
- Every strategy exposes **explicit parameters** (how strong the rotation is, how far colour may drift, JPEG quality range, etc.). Presets only set **defaults**; you override by supplying a `strategies` table.

The [developer catalogue](../developer/dataset-augmentation.md#canonical-strategy-names-and-parameters) lists the same names and parameter contracts—still **string keys**, not numeric codes.

## Relationship to latent cache

Rengu Flow encodes images to latents once during the **cache** step and trains on cached tensors. **Stochastic** augmentation (a new random transform every training step) is **not** compatible with a fixed latent cache unless latents are recomputed each time or augmentation is moved **after** decoding (not typical).

Supported **modes** (see [developer doc](../developer/dataset-augmentation.md#cache-and-seed-modes)):

| Mode | Effect on cache |
|------|-----------------|
| **`deterministic_per_image`** | One random draw per file, fixed by seed derived from file identity; latents can be cached consistently. |
| **`stochastic`** | Requires disabling fixed latent cache or re-encoding each run; training reads RGB each step (higher I/O and compute). |
| **Offline variants** | Pre-augmented files on disk; no special cache logic. |

Tools such as [Kohya sd-scripts](https://github.com/kohya-ss/sd-scripts) document that image augmentations conflict with latent caching in the same way: augmentations apply when reading RGB before the VAE.

## How augmented copies work

Each training image produces one **pristine original** plus up to **`branches_per_image`** augmented copies in the latent cache. Each copy uses a distinct deterministic seed (derived from the image path + branch index), so the same config always produces the same cache layout.

- **`branches_per_image = 0`** — only the pristine original; augmentation is effectively off for that image.
- **`branches_per_image = 1`** (default) — one augmented copy besides the original; the cache has 2 rows per image.
- **`branches_per_image = N`** — N additional augmented copies; the cache has N+1 rows per image.

Each copy is an **independent probabilistic draw**: every strategy (e.g. `horizontal_flip`, `color_jitter`) applies at its configured probability per copy. Raising `branches_per_image` increases cache disk usage proportionally and gives the model more augmented views of each image per epoch.

**`num_repeats`:** Repeats control how often an example appears in the schedule; they are **not** a substitute for more augmented branches — see [developer doc](../developer/dataset-augmentation.md#cache-and-seed-modes).

## Configuration layout

### Minimal: preset only

```toml
[[directory]]
path = "/data/photos"
num_repeats = 1
augmentation = { enabled = true, preset = "easy", seed_mode = "deterministic_per_image" }
```

### Full control: preset + named strategies with parameters

`strategies` is a table keyed by **strategy name**. Each entry can set `enabled = false` to turn off that piece of the preset, or set numeric fields to **override** strength. Omitted strategies keep the preset's defaults.

```toml
[[directory]]
path = "/data/photos"
num_repeats = 1

[directory.augmentation]
enabled = true
preset = "photo_safe"
seed_mode = "deterministic_per_image"
branches_per_image = 2

[directory.augmentation.strategies.horizontal_flip]
enabled = false

[directory.augmentation.strategies.color_jitter]
brightness = 0.05
contrast = 0.05
saturation = 0.04
hue = 0.015

[directory.augmentation.strategies.small_rotation]
enabled = true
max_degrees = 2.5
```

If your TOML loader makes nested `directory.augmentation` awkward next to `[[directory]]`, use an **inline nested table** with the same keys (see [developer doc](../developer/dataset-augmentation.md#toml-schema-implementation-contract)) or split into one dataset file per directory—the semantic model is the same: **`preset` + optional `strategies.<name>` overrides**.

Equivalent inline form (illustrative):

```toml
augmentation = {
  enabled = true,
  preset = "photo_safe",
  seed_mode = "deterministic_per_image",
  branches_per_image = 2,
  strategies = {
    horizontal_flip = { probability = 0.0 },
    color_jitter = { brightness = 0.05, contrast = 0.05, saturation = 0.04, hue = 0.015 },
    small_rotation = { enabled = true, max_degrees = 2.5 },
  },
}
```

### Custom without a bundle

Use **`preset = "none"`** or **`preset = "custom"`** and list only the strategies you want with full parameters (see [Named strategies reference](#named-strategies-reference)).

## Implemented strategies (MVP)

| Strategy | Notes |
|----------|--------|
| `color_jitter`, `gamma`, `jpeg_simulation`, `temperature_tint`, `chromatic_aberration` | Tier A (photometric) |
| `gaussian_noise`, `crop_jitter`, `small_rotation`, `film_grain`, `lab_jitter`, `split_toning` | Tier B |
| `horizontal_flip` | Geometric; flips at random per its `probability` |

Other names in the [developer catalogue](../developer/dataset-augmentation.md#canonical-strategy-names-and-parameters) are reserved; using them in TOML returns a clear error.

## Keys

### Global `[dataset.augmentation]`

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **`enabled`** | Master switch when a directory does not override. | `true` / `false` | `false` |
| **`preset`** | Default preset name. | See [Presets](#presets) | `"none"` |

### Per-directory `augmentation`

| Key | Purpose | Values | Default |
|-----|---------|--------|---------|
| **`enabled`** | Use augmentation for this folder. | `true` / `false` | Inherit or `false` |
| **`preset`** | Bundle of default enabled strategies and strengths. | See [Presets](#presets) | Inherit or `"none"` |
| **`seed_mode`** | Randomness vs cache. | `deterministic_per_image`, `stochastic` | `deterministic_per_image` |
| **`branches_per_image`** | Augmented copies cached per image besides the pristine original. 0 = original only; raises cache disk proportionally. | Integer ≥ 0 | `1` |
| **`strategies`** | Table of **named** strategy blocks; each overrides or disables part of the preset. Keys are **strings** (`snake_case`). | Map name → parameters | Empty (use preset only) |
| **`enable_strategies`** | Optional **list of strings** (strategy names) to restrict a preset, e.g. `["color_jitter", "gamma"]`. Intersects with the preset's default set. Omit if unused. | List of strings | omitted |

Prefer **`strategies`** when you need per-parameter control; use **`enable_strategies`** only as a compact filter on preset strategy names.

## Named strategies reference

Each name maps to one logical transform. Parameters are examples; exact keys and ranges are defined in the [developer parameter schema](../developer/dataset-augmentation.md#canonical-strategy-names-and-parameters).

| Strategy name | What you configure (examples) |
|---------------|-------------------------------|
| **`horizontal_flip`** | `probability`, `enabled` — flips at random per the probability per branch |
| **`vertical_flip`** | `probability` |
| **`color_jitter`** | `brightness`, `contrast`, `saturation`, `hue` (max delta per application, torch-style factors) |
| **`gamma`** | `gamma_min`, `gamma_max` or `exposure_ev_range` |
| **`gaussian_noise`** | `sigma` (0–255 scale) |
| **`jpeg_simulation`** | `quality_min`, `quality_max` |
| **`gaussian_blur`**, **`motion_blur`**, **`unsharp_mask`** | kernel / sigma / amount (per developer schema) |
| **`temperature_tint`** | warm/cool bounds |
| **`small_rotation`** | `max_degrees`, padding mode |
| **`scale_translate`**, **`crop_jitter`**, **`random_erasing`** | ranges documented per strategy |
| **`chromatic_aberration`**, **`vignette`**, **`local_tone_mapping`**, … | see developer catalogue |

## Web UI (dataset editor)

On the **Augmentation** tab, choosing a **preset** expands its default strategies in the form so you can see each transform, toggle **Enabled**, and edit parameters (same controls as per-folder customization). Overrides are stored under `strategies` in TOML; values that match the preset default are omitted on save. Changing the preset clears previous per-strategy overrides. Use **Add strategy** to attach extra transforms from the catalog (`preset = "none"` / `"custom"` starts from an empty list).

**Branches per image** sets how many augmented copies are cached per image (default 1). Set to 0 to cache only the original without any augmented copy.

## Presets vs `strategies` (merge rules)

1. Start from the **preset** definition (which strategies are on by default and their default strengths).
2. If **`enable_strategies`** is set, keep only strategies whose **name** appears in that string list (intersection with the preset).
3. Merge **`strategies`** on top: any named block **overrides** that strategy's parameters; `enabled = false` **disables** it even if the preset would enable it.
4. If **`preset`** is `none` / `custom`, only strategies listed under **`strategies`** run (each should be explicitly configured or use documented defaults for that name).

Everything is keyed by **string names**, not numeric indices.

## Presets

Presets group default **named** strategies with conservative strengths. Implementations expand a preset to a full `strategies` map internally. Details of default numbers are in the [developer doc](../developer/dataset-augmentation.md#domain-presets-default-strategy-sets).

### General "catch-all" presets (mixed content)

| Preset | Typical use |
|--------|-------------|
| **`easy`** | **Silver-bullet default** when unsure: very mild photometric variation only; no flip / JPEG / geometry by default. |
| **`anime_mixed`** | Digital anime: **B&W + colour** in one folder; tight sat/hue; flip off or very low. |
| **`manga_mixed`** | Manga art: **B&W + colour** pages/covers; photometric-first, protect line art. |
| **`realism_general`** | Photos: people, landscapes, **strong light/dark**; emphasis on exposure/gamma. |

### Other presets

| Preset | Typical use |
|--------|-------------|
| **`none`** | No augmentation. |
| **`photo_safe`** | Realistic photos: mild colour jitter, gamma, optional JPEG, optional horizontal flip. |
| **`photo_cinematic`** | Stronger cinematic look (tone mapping, bloom, split toning, LUTs—all mild). |
| **`anime`** | Mostly **colour** illustration. |
| **`manga_bw`** | **Pure** B&W line art folders. |
| **`bw_photo`** | Monochrome photography. |
| **`sepia`** | Warm vintage. |
| **`retro_scan`** | Paper / scan aesthetic (opt-in sub-effects). |
| **`manga_print`** | Halftone / screentone-oriented. |
| **`custom`** | Prefer explicit **`strategies`** only (no bundle), or `preset = "none"` + strategies. |

## When to use which preset

| Content type | Suggested starting preset | Notes |
|--------------|---------------------------|--------|
| **Unsure / first run** | **`easy`** | Then tune named parameters under `strategies` if needed. |
| Anime **B&W + colour** | **`anime_mixed`** | Override `color_jitter.saturation` / `hue` if B&W pages tint. |
| Manga **mixed** | **`manga_mixed`** | Same; disable `horizontal_flip` if text matters. |
| **Photos**, harsh lighting | **`realism_general`** or `photo_safe` | Override `gamma` / `color_jitter` for night vs day. |
| Landscapes, objects | `photo_safe` or `photo_cinematic` | |
| Faces, asymmetric characters | any preset + **`strategies.horizontal_flip.enabled = false`** | |
| Pure manga B/W | `manga_bw` | |
| Mostly colour anime | `anime` | |
| Sepia / vintage | `sepia` | |
| Scanned look | `retro_scan` | |

## References

- **Kohya sd-scripts** — `flip_aug`, `color_aug`, `random_crop`; latent cache: [train_network_advanced.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_advanced.md).
- **OneTrainer** — [Nerogar/OneTrainer](https://github.com/Nerogar/OneTrainer).
- **Albumentations** — [Choosing augmentations](https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/).
- **FLARE (BMVC 2024)** — [paper PDF](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_505/paper.pdf).

Full **strategy name** reference and implementation hooks: [Dataset augmentation (developer)](../developer/dataset-augmentation.md).
