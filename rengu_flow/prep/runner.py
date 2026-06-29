"""Single-stage prep runner: the process the CLI (and therefore the UI) executes.

One process = one stage (tag | caption | clean). Emits throttled ``@@RFPROG@@``
markers so the web UI's existing live-progress plumbing works unchanged, honors the
``save_quit``/``quit`` signal files between batches (graceful partial stop), writes a
``report.json`` into the job dir, and prints the ``exits with return code = N`` line
the UI's exit-code reconciliation already parses for training jobs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rengu_flow.control.progress_stream import ProgressEmitter
from rengu_flow.prep.config import PrepConfig
from rengu_flow.utils.logging import get_logger
from rengu_flow.utils.signal_files import SIGNAL_QUIT, SIGNAL_SAVE_QUIT

logger = get_logger(__name__)


def _make_should_stop(job_dir: Path):
    signals = (job_dir / SIGNAL_SAVE_QUIT, job_dir / SIGNAL_QUIT)

    def should_stop() -> bool:
        return any(s.exists() for s in signals)

    return should_stop


def _progress_callback(emitter: ProgressEmitter, stage: str):
    def on_progress(done: int, total: int, msg: str) -> None:
        emitter.emit(
            {
                "phase": f"prep:{stage}",
                "step": done,
                "max_steps": total,
                "msg": msg,
                "percent": round(100.0 * done / total, 1) if total else 0.0,
            },
            force=(done >= total),
        )

    return on_progress


# deepghs aesthetic label -> booru-style quality tag, prepended to the tag line
# when [tag].quality_tags is on (the anime-training "masterpiece, ..." convention).
AESTHETIC_QUALITY_TAGS = {
    "masterpiece": "masterpiece",
    "best": "best quality",
    "great": "great quality",
    "good": "good quality",
    "normal": "normal quality",
    "low": "low quality",
    "worst": "worst quality",
}


def _aesthetic_quality_labels(paths, on_progress, should_stop) -> dict:
    """path string -> deepghs aesthetic label, via the uv-overlay scorer."""
    import tempfile

    from rengu_flow.prep.quality import _AESTHETIC_SCORER, _iter_scorer

    manifest = Path(tempfile.mkstemp(suffix=".txt", prefix="qtag_")[1])
    manifest.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    labels: dict[str, str] = {}
    try:
        done = 0
        for img_path, rec in _iter_scorer(_AESTHETIC_SCORER, "dghs-imgutils>=0.15", manifest, ""):
            if should_stop():
                break
            done += 1
            if "label" in rec:
                labels[str(img_path)] = rec["label"]
            on_progress(done, len(paths), f"quality:{img_path.name}")
    finally:
        manifest.unlink(missing_ok=True)
    return labels


def _run_tag(config: PrepConfig, on_progress, should_stop) -> dict:
    from rengu_flow.prep.caption_store import CaptionStore
    from rengu_flow.prep.tagger import KNOWN_TAGGERS

    cs = CaptionStore.open(config.path, fmt=config.caption_format, ext=config.caption_ext)
    stage = config.tag

    unknown = [m for m in stage.models if m not in KNOWN_TAGGERS]
    if unknown:
        raise ValueError(f"Unknown tagger model(s): {unknown}. Known: {list(KNOWN_TAGGERS)}")
    specs = [KNOWN_TAGGERS[m] for m in stage.models]

    # Global confidence/category controls become per-model overrides; explicit
    # [tag.overrides.<id>] entries from the config still win over the global values.
    global_overrides: dict = {}
    if stage.general_threshold:
        global_overrides["general_threshold"] = float(stage.general_threshold)
    if stage.character_threshold:
        global_overrides["character_threshold"] = float(stage.character_threshold)
    if stage.rating_threshold:
        global_overrides["rating_threshold"] = float(stage.rating_threshold)
    if not stage.include_character_tags:
        global_overrides["include_character"] = False
    if not stage.include_rating:
        global_overrides["include_rating"] = False
    overrides = {
        spec.id: {**global_overrides, **(stage.overrides.get(spec.id) or {})}
        for spec in specs
    } if (global_overrides or stage.overrides) else None

    from rengu_flow.prep import tag_progress
    from rengu_flow.prep.tagger import run_ensemble_chunked

    # 1-based target line -> 0-based index. Tags conventionally live on line 1 (index 0); a
    # higher target lets tags ride a different line (e.g. alongside a caption).
    target_idx = max(0, stage.target_line - 1)

    # Resume: keys this model set already finished in a prior (stopped) run are skipped, on top
    # of the overwrite rule (skip images that already have tags on the target line).
    done_prev = tag_progress.load_done(config.path, stage.models)
    to_tag = [
        key
        for key in cs.keys()
        if key not in done_prev and (stage.overwrite or not cs.get_tags(key, target_idx))
    ]
    skipped = len(cs.keys()) - len(to_tag)
    paths = [cs.images[key] for key in to_tag]
    key_by_path = {str(cs.images[key]): key for key in to_tag}

    # Optional aesthetic quality tag (one upfront pass; prepended to each tag line below).
    quality_labels = (
        _aesthetic_quality_labels(paths, on_progress, should_stop)
        if stage.quality_tags else {}
    )

    done_keys = set(done_prev)
    written: set[str] = set()
    counters = {"tagged": 0}

    def _on_chunk(_chunk_paths, merged: dict) -> None:
        # Each chunk arrives complete (all models). Write its tags to the caption files and
        # save immediately, then record the keys as done — so the work is usable and resumable.
        for path_str, line in merged.items():
            key = key_by_path.get(path_str)
            if key is None:
                continue
            if line:
                qtag = AESTHETIC_QUALITY_TAGS.get(quality_labels.get(path_str, ""))
                if qtag:  # quality tag leads, per the booru convention
                    if stage.underscores:  # match the tag line's form (best quality -> best_quality)
                        qtag = qtag.replace(" ", "_")
                    line = f"{qtag}, {line}"
                cs.set_line(key, target_idx, line)
                counters["tagged"] += 1
            done_keys.add(key)
        written.update(cs.save())
        tag_progress.save_done(config.path, stage.models, done_keys)

    run_ensemble_chunked(
        paths,
        specs,
        chunk_size=512,
        overrides=overrides,
        exclude_tags=stage.exclude_tags,
        prepend_tags=stage.prepend_tags,
        max_tags=stage.max_tags,
        batch_size=stage.batch_size,
        underscores=stage.underscores,
        on_chunk=_on_chunk,
        on_progress=on_progress,
        should_stop=should_stop,
    )

    stopped = should_stop()
    if not stopped:
        tag_progress.clear(config.path)  # finished — nothing to resume
    return {
        "tagged": counters["tagged"],
        "skipped": skipped,
        "files_written": len(written),
        "models": stage.models,
        "quality_tags": stage.quality_tags,
        "stopped": stopped,
    }


def _run_caption(config: PrepConfig, on_progress, should_stop) -> dict:
    from rengu_flow.prep.captioner import caption_folder, captioner_config_from_stage

    captioner_config = captioner_config_from_stage(config.caption)
    return caption_folder(
        config.path,
        captioner_config,
        fmt=config.caption_format,
        ext=config.caption_ext,
        on_progress=on_progress,
        should_stop=should_stop,
    )


def _run_clean(config: PrepConfig, on_progress, should_stop) -> dict:
    from rengu_flow.prep.cleanup import CleanupConfig, clean_folder

    stage = config.clean
    cleanup_config = CleanupConfig(
        confidence=stage.confidence,
        mask_dilation_px=stage.mask_dilation_px,
        in_place=stage.in_place,
        output_dir=Path(stage.output_dir) if stage.output_dir else None,
        copy_undetected=stage.copy_undetected,
    )
    return clean_folder(
        config.path,
        cleanup_config,
        on_progress=on_progress,
        should_stop=should_stop,
    )


def _run_quality(config: PrepConfig, on_progress, should_stop) -> dict:
    from rengu_flow.prep.quality import QualityConfig, filter_folder

    stage = config.quality
    quality_config = QualityConfig(
        metric=stage.metric,
        blur_threshold=stage.blur_threshold,
        min_side=stage.min_side,
        min_detail=stage.min_detail,
        aesthetic_min_label=stage.aesthetic_min_label,
        aesthetic_model=stage.aesthetic_model,
        iqa_model=stage.iqa_model,
        iqa_threshold=stage.iqa_threshold,
        action=stage.action,
        output_dir=Path(stage.output_dir) if stage.output_dir else None,
    )
    return filter_folder(
        config.path,
        quality_config,
        caption_ext=config.caption_ext,
        on_progress=on_progress,
        should_stop=should_stop,
    )


def _run_index(config: PrepConfig, on_progress, should_stop) -> dict:
    from rengu_flow.prep.quality_index import build_index, model_stats

    models = config.index.models
    if not models:
        raise ValueError("index stage needs [index].models")
    report = build_index(
        config.path, models, on_progress=on_progress, should_stop=should_stop
    )
    report["stats"] = {m: model_stats(config.path, m) for m in models}
    return report


_STAGE_RUNNERS = {
    "tag": _run_tag,
    "caption": _run_caption,
    "clean": _run_clean,
    "quality": _run_quality,
    "index": _run_index,
}


def run_stage(config: PrepConfig, stage: str, job_dir: Path) -> int:
    """Run one prep stage to completion. Returns the process exit code."""
    config.validate_for_stage(stage)
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    emitter = ProgressEmitter()
    on_progress = _progress_callback(emitter, stage)
    should_stop = _make_should_stop(job_dir)

    logger.info("prep %s: %s (%s%s)", stage, config.path, config.caption_format,
                config.caption_ext if config.caption_format == "sidecar" else "")
    emitter.emit({"phase": f"prep:{stage}", "step": 0, "max_steps": 0, "msg": "starting"},
                 force=True)

    code = 0
    try:
        report = _STAGE_RUNNERS[stage](config, on_progress, should_stop)
    except Exception as exc:
        logger.exception("prep %s failed", stage)
        report = {"error": f"{type(exc).__name__}: {exc}"}
        code = 1

    report["stage"] = stage
    report["path"] = config.path
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (job_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    emitter.emit(
        {"phase": f"prep:{stage}", "msg": "stopped" if report.get("stopped") else "done",
         "done": True},
        force=True,
    )
    # Same line DeepSpeed prints; the UI parses it to reconcile the job's exit code.
    print(f"prep {stage} exits with return code = {code}")
    return code
