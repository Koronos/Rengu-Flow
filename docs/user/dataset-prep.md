# Dataset preparation (`rengu prep` / Prep section)

The prep module prepares image datasets **outside** of training: danbooru-style
tagging, natural-language captioning, watermark cleanup, and a safe bulk tag editor.
It never runs as part of `rengu train` — each stage is its own process, launched from
the CLI or from the **Prep** section of the web UI. Prep jobs share the single-runner
queue with training, so a prep job never competes with a training run for the GPU.

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

ONNX tagger ensemble. Default: **PixAI v0.9 + cl_tagger 1.01** (2025-26 generation —
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

### Captioning — `rengu prep caption`

Writes the natural-language caption as **line 2**, leaving the tag line untouched.
Two models, both selectable per job:

- `joycaption-beta-one` (default) — 8B LLaVA; bf16 needs ~17 GB and fits a 24 GB GPU
  because the queue guarantees exclusivity. `int8`/`nf4` quantization for smaller
  cards or bigger batches.
- `toriigate-0.5` — ~5B anime specialist; with `use_tags_as_grounding = true`
  (default) it receives the image's line-1 booru tags as context.

The model loads once per job and generates in true batches; on OOM the batch halves
and stays halved. Captions save incrementally after every batch, so a stop never
loses completed work.

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

Weights land in the standard HuggingFace cache (`HF_HOME` respected).

## Prep TOML

One file can describe every stage; each stage only reads its own section. CLI flags
override the file. Unknown keys are ignored (the schema can evolve freely).

```toml
path = "/data/my_dataset"
caption_format = "sidecar"      # or "json"
caption_ext = ".txt"

[tag]
models = ["pixai-v0.9", "cl-tagger-1.01"]
exclude_tags = ["realistic", "3d"]
prepend_tags = ["my_trigger_word"]
max_tags = 255
batch_size = 16
overwrite = false

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
confidence = 0.5
mask_dilation_px = 8
in_place = false
output_dir = ""                 # empty = <path>/cleaned
```

## Tag editor (web UI → Prep → Tag editor)

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
