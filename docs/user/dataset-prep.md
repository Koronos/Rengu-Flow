# Dataset Studio (`rengu prep`)

Dataset Studio prepares image datasets **outside** of training: danbooru-style
tagging, natural-language captioning, watermark cleanup, and a safe bulk tag editor.
It never runs as part of `rengu train` — each stage is its own process, launched from
the CLI or from the **Studio** section of the web UI. Studio jobs share the
single-runner queue with training, so a prep job never competes with a training run
for the GPU. Stopped/failed jobs can be **re-queued** from the job list — a re-run
resumes where it stopped, because already-processed images are skipped (unless
Overwrite is on). Finished jobs offer **Generate dataset**, which opens the dataset
editor pre-filled with the job's folder (the cleaned-copies folder for clean jobs).

Bucketing and multi-resolution handling are **not** part of prep: the trainer's
aspect-ratio buckets and per-resolution caching already cover that at cache time.

## Caption layout

Prep reads and writes the same caption formats the trainer understands:

- **Sidecar files** (default): `image.png` → `image.txt`, one caption variant per
  line. The extension is configurable (`caption_ext`). Convention: **line 1 = tags,
  line 2 = natural-language caption**; extra lines are additional variants (e.g.
  pre-baked tag-dropout variants).
- **captions.json**: one `{ "image.png": ["caption", ...] }` file per folder
  (`caption_format = "json"`).

## Stages

### Tagging — `rengu prep tag`

ONNX tagger ensemble. Default: **PixAI v0.9 + cl_tagger 1.02** (2025-26 generation —
fresher Danbooru data and far better character recall than the WD v3 line, which
remains available: `wd-eva02-large-v3`, `wd-vit-large-v3`, `wd-swinv2-v3`). Models run
one at a time (one ONNX session in VRAM); per-image probabilities merge across models
by max probability. Ratings resolve by argmax (one rating tag). Underscores become
spaces except kaomojis (`^_^`, `0_0`, …).

```bash
rengu prep tag --path /data/my_dataset                 # defaults
rengu prep tag --config prep.toml --model wd-eva02-large-v3 --overwrite
```

Images that already have a tag line are skipped unless `--overwrite`.

Output tags are **ordered by confidence** (most certain first). Confidence controls:
`general_threshold` / `character_threshold` set a global floor for every selected
model (higher = fewer but surer tags; per-model `[tag.overrides.<id>]` entries still
win), `include_character_tags = false` drops character/series name tags entirely
(taggers are weakest at character names — combine with `prepend_tags` for your own
trigger), and `include_rating = false` drops the rating tag.

### Captioning — `rengu prep caption`

Writes the natural-language caption as **line 2**, leaving the tag line untouched.
Two models, both selectable per job:

- `joycaption-beta-one` (default) — 8B LLaVA; bf16 needs ~17 GB and fits a 24 GB GPU
  because the queue guarantees exclusivity. `int8`/`nf4` quantization for smaller
  cards or bigger batches.
- `toriigate-0.5` — ~5B anime specialist; with `use_tags_as_grounding = true`
  (default) it receives the image's line-1 booru tags as context. ToriiGate is
  trained on FIXED prompt formats, so it does not take the composable instruction
  prompt: the base maps to its native format (`concise` → short, everything else →
  long), grounding/character name use its official blocks, and the modifiers ride an
  extra-requirements section. It also generates one image at a time (its hybrid
  linear-attention layers don't tolerate padded batches — that's what made captions
  start fine and then repeat/derail) and inputs are capped at ~1 Mpx (its training
  resolution). The "fast path not available" startup note is expected and harmless: the
  optional fused kernels (flash-linear-attention + causal-conv1d) were benchmarked
  (cu13torch2.10 wheel runs fine on torch 2.12) and give **no speedup** for this
  short-decode captioning workload — they target long-sequence prefill. The torch
  fallback is correct.

The model loads once per job and generates in true batches; on OOM the batch halves
and stays halved. Captions save incrementally after every batch, so a stop never
loses completed work. Oversized originals are downscaled before the VLM
(`max_image_side`, default 1536 — bucketing does the real resize later) and
thumbnails can be skipped (`min_image_side`).

**Composable prompts** (a custom `prompt` overrides the whole composition):

- **Base** (`prompt_base`, one of): `descriptive-long` (default), `concise`,
  `character-focus`, `style-focus`.
- **Modifiers** (`prompt_modifiers`, stackable):
  - `demographics` (default on) — apparent age, ethnicity/regional origin, skin
    tone when perceivable.
  - `medium_neutral` — NEVER names the medium or style (no
    photo/anime/illustration/render/realistic/…), so style isn't anchored to the
    text. This is how you train anime models on realistic data and vice versa.
  - `plain_language` — simple, direct English (no "cascading tresses"). The model
    learns to respond to the register its captions were written in: plain captions
    make plain user prompts work, no LLM prompt-embellishment needed at gen time.
  - `objective_only` — describe, never evaluate: no beautiful/stunning/masterpiece,
    so generation quality doesn't end up coupled to quality-word incantations.
  - `composition_camera` — states shot type (close-up…wide), camera angle and
    vantage: makes framing promptable.
  - `explicit_language` — direct anatomical language for NSFW datasets, no
    euphemisms.
- **Character trigger** (`character_name`) — the caption refers to the character by
  this name and never describes their inherent traits (hair, eyes, face, body):
  those get absorbed into the trigger token at training time. A deterministic
  post-pass scrubs any trait clause the (quantized) VLM leaks anyway, so absorption
  holds by construction.
- **Canonical look** (`character_canon`, optional) — for datasets with character
  VARIANTS (aged-up versions, alternate hairstyles, meme body forms). Describe the
  canon ("aqua twin-tail hair, blue eyes, slim teenage build") and the rule flips
  to: absorb what matches the canon, **describe what deviates from it** — so the
  deviation stays promptable instead of polluting the trigger. The hard scrubber is
  disabled in this mode (it cannot tell deviation from canon).
- **Caption line** (`target_line`, default 2) — write the caption to a higher line
  to ADD caption variants (rengu treats each line as one): e.g. queue two caption
  jobs, line 2 trigger-absorbed + line 3 full description.
- **Outfit policy** (`outfit`, only with a `character_name`):
  - `describe` — the outfit is captioned, so it stays swappable at generation time.
  - `omit` — the outfit is never captioned, so the default outfit is absorbed into
    the trigger (prompting the name brings the canonical look).
  - `mixed` — deterministic 50/50 per image: the dataset carries both signals, so
    the trigger retrieves the default outfit AND accepts outfit swaps. This mirrors
    the classic booru-LoRA practice of sometimes tagging the outfit, sometimes not.

Example — Miku LoRA trainable for realistic-style outputs with swappable outfit:

```toml
[caption]
prompt_base = "character-focus"
prompt_modifiers = ["medium_neutral"]
character_name = "hatsune miku"
outfit = "mixed"
```

The same absorption logic applies to the tag line: use the tagger's
`prepend_tags = ["hatsune miku"]` plus the tag editor to remove her inherent tags
(`aqua hair`, `twintails`, …) so they collapse into the trigger.

### Watermark cleanup — `rengu prep clean`

YOLO11 watermark detector (from the JoyCaption project) draws boxes → dilated mask →
LaMa inpainting (ONNX, same runtime as the taggers). Non-destructive by default:
cleaned copies go to `<dataset>/cleaned/` (or `--output-dir`). With `--in-place`,
originals are backed up under `<dataset>/.rengu_prep/cleanup_originals/<timestamp>/`
first.

### Models — `rengu prep models`

```bash
rengu prep models                    # list everything with download state
rengu prep models --stage tag --download pixai-v0.9
```

Weights land in the standard HuggingFace cache (`HF_HOME` respected). Downloads use
the `huggingface_hub` Python library bundled with the `prep` extra — the `hf` /
`huggingface-cli` binary is NOT required, and no HF account/token is needed (every
registry model is public; if you ever point a spec at a gated repo, exporting
`HF_TOKEN` is enough).

## Prep TOML

One file can describe every stage; each stage only reads its own section. CLI flags
override the file. Unknown keys are ignored (the schema can evolve freely).

```toml
path = "/data/my_dataset"
caption_format = "sidecar"      # or "json"
caption_ext = ".txt"

[tag]
models = ["pixai-v0.9", "cl-tagger-1.02"]
exclude_tags = ["realistic", "3d"]
prepend_tags = ["my_trigger_word"]
max_tags = 255
batch_size = 16
overwrite = false
# general_threshold = 0.5        # global confidence floor (omit = model defaults)
# character_threshold = 0.9
include_character_tags = true    # false: drop character/series name tags
include_rating = true            # false: drop the rating tag

[tag.overrides.pixai-v0.9]      # per-model threshold overrides
general_threshold = 0.35
character_threshold = 0.80

[caption]
model = "joycaption-beta-one"   # or "toriigate-0.5"
quantization = "bf16"           # bf16 | int8 | nf4
prompt = ""                     # empty = model default
batch_size = 4
use_tags_as_grounding = true
overwrite = false

[clean]
confidence = 0.35
mask_dilation_px = 8
in_place = false
output_dir = ""                 # empty = <path>/cleaned
```

## Tag editor (web UI → Studio → Tag editor)

Bulk tag operations over a whole folder with a **staged → diff → commit** safety
model: nothing touches disk until you commit, and every commit first snapshots all
caption files to `<dataset>/.rengu_prep/backups/<timestamp>/` (one-click restore).

- **Filters**: has all / has any / lacks — combine to select images.
- **Ops**: add tags (start/end), remove, rename (deduping), prune low-frequency tags,
  quarantine images by filter (moved to `.rengu_prep/quarantine/`, never deleted,
  restorable).
- **Line scope** per operation: line 1 only, all tag lines (default — edits propagate
  to pre-baked dropout variant lines so you never have to regenerate them), all
  lines, or one specific line. Natural-language caption lines are detected
  heuristically and left alone under "tag lines"; the diff preview is the final
  safety net before commit.
- Undo pops staged ops; closing the session without commit discards everything.

## Stopping jobs

Prep jobs honor the same signal files as training: the UI's Stop button (or touching
`save_quit` in the job dir) finishes the current batch, saves partial results, writes
`report.json`, and exits cleanly.

## Install

Dependencies live in the `prep` extra (`uv sync --extra prep`) and install on demand
the first time `rengu prep` runs — or from Maintenance → dependency profiles in the
UI. If a prep dependency ever conflicts with the training lock, isolate it with uv at
runtime (`uv run --extra prep …`) instead of downgrading.
