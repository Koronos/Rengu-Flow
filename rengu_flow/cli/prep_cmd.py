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
    for stage_parser in (t, c, cl, q):
        for args_, kwargs in common:
            stage_parser.add_argument(*args_, **kwargs)

    t.add_argument("--model", action="append", default=None,
                   help="Tagger model id (repeatable; overrides config)")
    t.add_argument("--overwrite", action="store_true", default=None,
                   help="Re-tag images that already have a tag line")
    c.add_argument("--model", default=None, help="Caption model id (overrides config)")
    c.add_argument("--quant", default=None, choices=("bf16", "int8", "nf4"),
                   help="Quantization (overrides config)")
    c.add_argument("--overwrite", action="store_true", default=None,
                   help="Re-caption images that already have a caption line")
    cl.add_argument("--in-place", action="store_true", default=None,
                    help="Rewrite sources (originals backed up under the app data dir)")
    cl.add_argument("--output-dir", default=None,
                    help="Destination for cleaned copies (default <path>/cleaned)")

    q.add_argument("--blur-threshold", type=float, default=None,
                   help="Laplacian-variance floor; below it an image is flagged blurry")
    q.add_argument("--min-side", type=int, default=None,
                   help="Flag images whose shorter side is below this (0 = off)")
    q.add_argument("--move", action="store_true", default=None,
                   help="Move flagged images into <path>/low_quality (default: report only)")
    q.add_argument("--output-dir", default=None,
                   help="Destination for moved files (default <path>/low_quality)")

    m = stage_sub.add_parser("models", help="List/download prep models")
    m.add_argument("--stage", default=None, choices=STAGES,
                   help="Limit listing to one stage")
    m.add_argument("--download", default=None, metavar="MODEL_ID",
                   help="Download one model (requires --stage)")


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
    elif stage == "caption":
        if args.model:
            config.caption.model = args.model
        if args.quant:
            config.caption.quantization = args.quant
        if args.overwrite is not None:
            config.caption.overwrite = args.overwrite
    elif stage == "clean":
        if args.in_place is not None:
            config.clean.in_place = args.in_place
        if args.output_dir:
            config.clean.output_dir = args.output_dir
    elif stage == "quality":
        if args.blur_threshold is not None:
            config.quality.blur_threshold = args.blur_threshold
        if args.min_side is not None:
            config.quality.min_side = args.min_side
        if args.move:
            config.quality.action = "move"
        if args.output_dir:
            config.quality.output_dir = args.output_dir
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


def run(args: argparse.Namespace) -> None:
    if args.prep_stage == "models":
        _run_models(args)
        return

    from rengu_flow.cli.training_extras import ensure_profiles

    ensure_profiles(["prep"], reason="dataset prep")

    from rengu_flow.prep.runner import run_stage

    config = _build_config(args)
    config.validate_for_stage(args.prep_stage)
    job_dir = Path(args.job_dir) if args.job_dir else _default_job_dir(config, args.prep_stage)
    raise SystemExit(run_stage(config, args.prep_stage, job_dir))
