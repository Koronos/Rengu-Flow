"""``rengu prep`` — dataset preparation stages (tag | caption | clean | quality | models).

Each stage runs in this process (the UI launches exactly this command as a
subprocess). Heavy inference deps live in the ``prep`` extra and are installed
on demand via the same profile machinery training uses.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from rengu_flow.prep.config import STAGES, PrepConfig, load_prep_config
from rengu_flow.prep.storage import prep_storage_dir


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prep", help="Dataset preparation (tagging, captioning, cleanup)")
    stage_sub = p.add_subparsers(dest="prep_stage", required=True)

    common: list[tuple[tuple, dict]] = [
        (("--config",), dict(default=None, help="Prep TOML config path")),
        (("--path",), dict(default=None, help="Dataset folder (overrides config)")),
        (
            ("--caption-format",),
            dict(default=None, choices=("sidecar", "json"), help="Caption layout"),
        ),
        (("--caption-ext",), dict(default=None, help="Sidecar extension (default .txt)")),
        (
            ("--job-dir",),
            dict(default=None, help=argparse.SUPPRESS),  # set by the UI job launcher
        ),
    ]

    t = stage_sub.add_parser("tag", help="Danbooru-style tagging (ONNX ensemble)")
    c = stage_sub.add_parser("caption", help="Natural-language captioning (VLM)")
    cl = stage_sub.add_parser("clean", help="Watermark detection + inpainting")
    q = stage_sub.add_parser("quality", help="Flag/move low-quality images (blur + resolution)")
    ix = stage_sub.add_parser("index", help="Persistent quality index (incremental, multi-model)")
    for stage_parser in (t, c, cl, q, ix):
        for args_, kwargs in common:
            stage_parser.add_argument(*args_, **kwargs)

    t.add_argument("--model", action="append", default=None,
                   help="Tagger model id (repeatable; overrides config)")
    t.add_argument("--overwrite", action="store_true", default=None,
                   help="Re-tag images that already have a tag line")
    t.add_argument("--quality-tags", action="store_true", default=None,
                   help="Prepend a deepghs aesthetic quality tag (masterpiece..worst quality)")
    t.add_argument("--underscores", action="store_true", default=None,
                   help="Keep the original danbooru form (long_hair) instead of spaces (long hair)")
    t.add_argument("--target-line", type=int, default=None,
                   help="1-based caption line to write tags to (default 1 = the tag line)")
    c.add_argument("--model", default=None, help="Caption model id (overrides config)")
    c.add_argument("--quant", default=None, choices=("bf16", "int8", "nf4"),
                   help="Quantization (overrides config)")
    c.add_argument("--overwrite", action="store_true", default=None,
                   help="Re-caption images that already have a caption line")
    c.add_argument("--engine", default=None, choices=("hf", "vllm", "gguf"),
                   help="hf: transformers (any) | vllm: JoyCaption (faster) | gguf: ToriiGate via llama.cpp (faster)")
    c.add_argument("--vllm-quant", default=None, choices=("gptq", "fp8", "awq", "none"),
                   help="vLLM quantization (gptq 4-bit fits 16 GB; fp8 ~8.5 GB; awq needs --vllm-model)")
    c.add_argument("--vllm-model", default=None,
                   help="vLLM checkpoint repo override (default resolves from --vllm-quant)")
    c.add_argument("--gguf-quant", default=None, choices=("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"),
                   help="GGUF (ToriiGate) weight quant: Q8_0 ~lossless .. Q4_K_M smallest/fastest")
    cl.add_argument("--in-place", action="store_true", default=None,
                    help="Rewrite sources (originals backed up under the app data dir)")
    cl.add_argument("--output-dir", default=None,
                    help="Destination for cleaned copies (default <path>/cleaned)")

    q.add_argument("--metric", default=None, choices=("blur", "aesthetic", "iqa"),
                   help="blur: Laplacian (dep-free) | aesthetic: deepghs booru appeal | "
                        "iqa: pyiqa technical NR-IQA (anime + photo)")
    q.add_argument("--blur-threshold", type=float, default=None,
                   help="blur: Laplacian-variance floor; below it an image is flagged blurry")
    q.add_argument("--min-side", type=int, default=None,
                   help="blur: flag images whose shorter side is below this (0 = off)")
    q.add_argument("--min-detail", type=float, default=None,
                   help="blur: flag low effective resolution (pixelated/upscaled); 0 = off")
    q.add_argument("--aesthetic-min-label", default=None,
                   choices=("worst", "low", "normal", "good", "great", "best", "masterpiece"),
                   help="aesthetic: flag images ranked below this booru label (default normal)")
    q.add_argument("--iqa-model", default=None,
                   help="iqa: pyiqa model — clipiqa/arniqa (any domain), musiq/maniqa (photos), "
                        "brisque/niqe (classic). Default clipiqa")
    q.add_argument("--iqa-threshold", type=float, default=None,
                   help="iqa: percentile cull 0-100 — flag the lowest N%% by quality in the dataset "
                        "(same behavior for every model). e.g. 10 = drop the worst 10%%")
    q.add_argument("--move", action="store_true", default=None,
                   help="Move flagged images into <path>/low_quality (default: report only)")
    q.add_argument("--output-dir", default=None,
                   help="Destination for moved files (default <path>/low_quality)")

    m = stage_sub.add_parser("models", help="List/download prep models")
    m.add_argument("--stage", default=None, choices=STAGES,
                   help="Limit listing to one stage")
    m.add_argument("--download", default=None, metavar="MODEL_ID",
                   help="Download one model (requires --stage)")

    ix.add_argument("--model", action="append", default=None,
                    help="Model id (repeatable): aesthetic, or a pyiqa model (clipiqa/niqe/...)")
    ix.add_argument("--build", action="store_true", help="Score missing (image, model) pairs")
    ix.add_argument("--worst", type=int, default=None, metavar="N",
                    help="Print the N lowest-quality images for the first --model")
    ix.add_argument("--cull", action="append", default=None, metavar="MODEL:PCT",
                    help="Preview the union cull (repeatable), e.g. --cull niqe:20 --cull clipiqa:10")


def _build_config(args: argparse.Namespace) -> PrepConfig:
    config = load_prep_config(args.config) if args.config else PrepConfig()
    if args.path:
        config.path = args.path
    if args.caption_format:
        config.caption_format = args.caption_format
    if args.caption_ext:
        config.caption_ext = args.caption_ext

    stage = args.prep_stage
    if stage == "tag":
        if args.model:
            config.tag.models = list(args.model)
        if args.overwrite is not None:
            config.tag.overwrite = args.overwrite
        if args.quality_tags is not None:
            config.tag.quality_tags = args.quality_tags
        if args.underscores is not None:
            config.tag.underscores = args.underscores
        if args.target_line is not None:
            config.tag.target_line = args.target_line
    elif stage == "caption":
        if args.model:
            config.caption.model = args.model
        if args.quant:
            config.caption.quantization = args.quant
        if args.overwrite is not None:
            config.caption.overwrite = args.overwrite
        if args.engine:
            config.caption.engine = args.engine
        if args.vllm_quant:
            config.caption.vllm_quantization = args.vllm_quant
        if args.vllm_model:
            config.caption.vllm_model = args.vllm_model
        if args.gguf_quant:
            config.caption.gguf_quantization = args.gguf_quant
    elif stage == "clean":
        if args.in_place is not None:
            config.clean.in_place = args.in_place
        if args.output_dir:
            config.clean.output_dir = args.output_dir
    elif stage == "quality":
        if args.metric:
            config.quality.metric = args.metric
        if args.blur_threshold is not None:
            config.quality.blur_threshold = args.blur_threshold
        if args.min_side is not None:
            config.quality.min_side = args.min_side
        if args.min_detail is not None:
            config.quality.min_detail = args.min_detail
        if args.aesthetic_min_label:
            config.quality.aesthetic_min_label = args.aesthetic_min_label
        if args.iqa_model:
            config.quality.iqa_model = args.iqa_model
        if args.iqa_threshold is not None:
            config.quality.iqa_threshold = args.iqa_threshold
        if args.move:
            config.quality.action = "move"
        if args.output_dir:
            config.quality.output_dir = args.output_dir
    elif stage == "index":
        if args.model:
            config.index.models = list(args.model)
    return config


def _default_job_dir(config: PrepConfig, stage: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return prep_storage_dir(config.path) / "jobs" / f"{stage}-{ts}"


def _run_models(args: argparse.Namespace) -> None:
    from rengu_flow.prep.models import ensure_model, list_models

    if args.download:
        if not args.stage:
            raise SystemExit("--download requires --stage")
        path = ensure_model(args.download, args.stage)
        print(f"downloaded: {args.download} -> {path}")
        return

    stages = (args.stage,) if args.stage else STAGES
    for stage in stages:
        print(f"[{stage}]")
        for entry in list_models(stage):
            mark = "✓" if entry.get("downloaded") else " "
            print(f"  [{mark}] {entry['id']:<22} {entry['repo_id']}")


def _run_index_query(args: argparse.Namespace) -> None:
    from rengu_flow.prep import quality_index as qi

    models = list(args.model or [])
    if args.worst:
        if not models:
            raise SystemExit("--worst requires --model")
        for w in qi.worst(args.path, models[0], args.worst):
            print(f"  {w['quality']:6.1f}  {w['path']}")
    if args.cull:
        per_model = {}
        for spec in args.cull:
            model, _, pct = spec.partition(":")
            per_model[model] = float(pct)
        cp = qi.cull_preview(args.path, per_model)
        print(f"cull union={cp['union']} of present={cp['present']}  per-model={cp['per_model']}")


def run(args: argparse.Namespace) -> None:
    if args.prep_stage == "models":
        _run_models(args)
        return
    # index is dual-mode: --worst/--cull are instant queries; otherwise it builds
    # (manual --build, or the `--config/--job-dir` a UI job passes) via run_stage.
    if args.prep_stage == "index" and (args.worst or args.cull):
        _run_index_query(args)
        return

    from rengu_flow.cli.training_extras import ensure_profiles

    ensure_profiles(["prep"], reason="dataset prep")

    from rengu_flow.prep.runner import run_stage

    config = _build_config(args)
    config.validate_for_stage(args.prep_stage)
    job_dir = Path(args.job_dir) if args.job_dir else _default_job_dir(config, args.prep_stage)
    raise SystemExit(run_stage(config, args.prep_stage, job_dir))
