"""Persistent, scalable quality index for a dataset.

Stores per-image, per-model quality scores in a SQLite database (stdlib, single
file, scales to tens of millions of rows, indexed) under the managed prep data
dir. Built incrementally: only (image, model) pairs that are missing or whose
file changed get scored, reusing the GPU scorers (pyiqa / deepghs aesthetic).

Why an index instead of a one-shot job:
  * Re-running never re-scores what's already indexed — adjust how much to cull
    without paying the model again.
  * The cull is a percentile taken over the WHOLE indexed reference (including
    images already moved out), so re-running doesn't erode further: the same
    cutoff drops the same images.
  * Each model has its own scores; the UI culls per model and unions the results.

Model ids: "aesthetic" -> deepghs booru-appeal (quality = percentile*100); any
other id -> a pyiqa NR-IQA model (quality = its normalized 1..100). Both are
higher = better, so a single "drop the lowest N%" works across models.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from rengu_flow.prep.caption_store import IMAGE_EXTENSIONS
from rengu_flow.prep.quality import _AESTHETIC_SCORER, _IQA_SCORER, _iter_scorer
from rengu_flow.prep.storage import prep_storage_dir
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id      INTEGER PRIMARY KEY,
  path    TEXT UNIQUE NOT NULL,
  mtime   INTEGER NOT NULL,
  present INTEGER NOT NULL DEFAULT 1   -- 1 = still in the dataset folder, 0 = moved/removed
);
CREATE TABLE IF NOT EXISTS scores (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  model    TEXT NOT NULL,
  raw      REAL,
  quality  REAL NOT NULL,             -- 1..100, higher = better, comparable across models
  PRIMARY KEY (image_id, model)
);
CREATE INDEX IF NOT EXISTS idx_scores_model_quality ON scores(model, quality);
"""


def index_db_path(src: str | Path) -> Path:
    return prep_storage_dir(src) / "quality_index.db"


def _connect(src: str | Path) -> sqlite3.Connection:
    path = index_db_path(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # concurrent reads while building
    conn.executescript(_SCHEMA)
    return conn


def _scan(src: Path) -> list[Path]:
    return sorted(
        p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _scorer_for(model: str) -> tuple[Path, str, str]:
    """(scorer script, uv --with package, model arg) for a model id."""
    if model == "aesthetic":
        return _AESTHETIC_SCORER, "dghs-imgutils>=0.15", ""
    return _IQA_SCORER, "pyiqa", model


def _quality_of(rec: dict) -> float | None:
    """Uniform 1..100 higher-better quality from a scorer record."""
    if "quality" in rec:  # iqa
        return float(rec["quality"])
    if "percentile" in rec:  # aesthetic (0..1)
        return round(100.0 * float(rec["percentile"]), 1)
    return None


# A scorer: given a model id and the image paths to score, yields
# (path, raw_score, quality) as results arrive. Injectable so the index logic is
# testable without the GPU; the default shells out to the uv-overlay scorers.
ScoreFn = Callable[[str, list[str], "Callable[[], bool] | None"], "Iterable[tuple[str, float | None, float]]"]


def _default_score_fn(model: str, paths: list[str], should_stop):
    scorer, pkg, model_arg = _scorer_for(model)
    manifest = Path(tempfile.mkstemp(suffix=".txt", prefix="qidx_")[1])
    manifest.write_text("\n".join(paths), encoding="utf-8")
    try:
        for img_path, rec in _iter_scorer(scorer, pkg, manifest, model_arg):
            if should_stop is not None and should_stop():
                break
            quality = _quality_of(rec)
            if quality is not None:
                yield str(img_path), rec.get("score"), quality
    finally:
        manifest.unlink(missing_ok=True)


def build_index(
    src: str | Path,
    models: Iterable[str],
    *,
    score_fn: ScoreFn | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Score every (current image, model) pair that isn't indexed yet.

    Refreshes the ``present`` flags (so moved images stay in the reference set but
    drop out of culling) and rescoring any image whose mtime changed. ``score_fn``
    is injectable for testing; it defaults to the GPU uv-overlay scorers.
    """
    score_fn = score_fn or _default_score_fn
    src = Path(src).resolve()
    models = list(models)
    conn = _connect(src)
    try:
        current = {str(p): int(p.stat().st_mtime) for p in _scan(src)}

        # Sync the image table: everything absent from the folder becomes present=0
        # (kept for the reference distribution); current files become present=1, and
        # a changed mtime drops that image's stale scores so it gets rescored.
        conn.execute("UPDATE images SET present = 0")
        for path, mtime in current.items():
            row = conn.execute("SELECT id, mtime FROM images WHERE path = ?", (path,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO images(path, mtime, present) VALUES(?, ?, 1)", (path, mtime)
                )
            elif row[1] != mtime:
                conn.execute("DELETE FROM scores WHERE image_id = ?", (row[0],))
                conn.execute("UPDATE images SET mtime = ?, present = 1 WHERE id = ?", (mtime, row[0]))
            else:
                conn.execute("UPDATE images SET present = 1 WHERE id = ?", (row[0],))
        conn.commit()

        report: dict = {"models": {}, "images": len(current)}
        for model in models:
            todo = conn.execute(
                "SELECT i.id, i.path FROM images i WHERE i.present = 1 AND NOT EXISTS "
                "(SELECT 1 FROM scores s WHERE s.image_id = i.id AND s.model = ?)",
                (model,),
            ).fetchall()
            if not todo:
                report["models"][model] = 0
                continue
            id_by_path = {path: iid for iid, path in todo}
            scored = 0
            for path, raw, quality in score_fn(model, [p for _, p in todo], should_stop):
                iid = id_by_path.get(path)
                if iid is None:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO scores(image_id, model, raw, quality) "
                    "VALUES(?, ?, ?, ?)",
                    (iid, model, raw, quality),
                )
                scored += 1
                if on_progress is not None:
                    on_progress(scored, len(todo), f"{model}:{Path(path).name}")
            conn.commit()
            report["models"][model] = scored
        return report
    finally:
        conn.close()


def model_stats(src: str | Path, model: str) -> dict:
    """Reference size, present count, and quality range for one model."""
    conn = _connect(src)
    try:
        total, lo, hi = conn.execute(
            "SELECT COUNT(*), MIN(quality), MAX(quality) FROM scores WHERE model = ?", (model,)
        ).fetchone()
        present = conn.execute(
            "SELECT COUNT(*) FROM scores s JOIN images i ON i.id = s.image_id "
            "WHERE s.model = ? AND i.present = 1",
            (model,),
        ).fetchone()[0]
        return {"model": model, "reference": total, "present": present, "min": lo, "max": hi}
    finally:
        conn.close()


def worst(src: str | Path, model: str, limit: int, offset: int = 0) -> list[dict]:
    """The lowest-quality images still in the folder, ascending — the cull preview."""
    conn = _connect(src)
    try:
        rows = conn.execute(
            "SELECT i.path, s.quality FROM scores s JOIN images i ON i.id = s.image_id "
            "WHERE s.model = ? AND i.present = 1 ORDER BY s.quality ASC LIMIT ? OFFSET ?",
            (model, limit, offset),
        ).fetchall()
        return [{"path": p, "quality": q} for p, q in rows]
    finally:
        conn.close()


def _cutoff(conn: sqlite3.Connection, model: str, percent: float) -> float | None:
    """Quality value at the ``percent`` percentile over the full reference set."""
    count = conn.execute("SELECT COUNT(*) FROM scores WHERE model = ?", (model,)).fetchone()[0]
    if not count or percent <= 0:
        return None
    offset = min(count - 1, int(percent / 100.0 * count))
    row = conn.execute(
        "SELECT quality FROM scores WHERE model = ? ORDER BY quality ASC LIMIT 1 OFFSET ?",
        (model, offset),
    ).fetchone()
    return row[0] if row else None


def cull_preview(src: str | Path, per_model: dict[str, float]) -> dict:
    """Union of the per-model culls: paths flagged by ANY model's percentile.

    ``per_model`` maps model id -> percent to drop (0..100). The cutoff is taken
    over the whole reference, so the same percent always means the same images.
    """
    conn = _connect(src)
    try:
        union: set[str] = set()
        per: dict[str, int] = {}
        for model, percent in per_model.items():
            cutoff = _cutoff(conn, model, percent)
            if cutoff is None:
                per[model] = 0
                continue
            paths = [
                r[0]
                for r in conn.execute(
                    "SELECT i.path FROM scores s JOIN images i ON i.id = s.image_id "
                    "WHERE s.model = ? AND i.present = 1 AND s.quality < ?",
                    (model, cutoff),
                ).fetchall()
            ]
            per[model] = len(paths)
            union.update(paths)
        present = conn.execute("SELECT COUNT(*) FROM images WHERE present = 1").fetchone()[0]
        return {"per_model": per, "union": len(union), "present": present, "paths": sorted(union)}
    finally:
        conn.close()


LOW_QUALITY_DIR = "low_quality"


def apply_cull(
    src: str | Path,
    per_model: dict[str, float],
    *,
    caption_ext: str = ".txt",
    output_dir: str | Path | None = None,
) -> dict:
    """Move the union cull out of the dataset and mark those images not-present.

    Their scores stay in the index (present=0) as the stable reference, so the
    same cull is idempotent on a re-run. Caption sidecars move alongside.
    """
    src = Path(src).resolve()
    preview = cull_preview(src, per_model)
    out_dir = Path(output_dir) if output_dir else src / LOW_QUALITY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = _connect(src)
    moved = 0
    try:
        for path_str in preview["paths"]:
            path = Path(path_str)
            if not path.exists():
                continue
            shutil.move(str(path), str(out_dir / path.name))
            sidecar = path.with_suffix(caption_ext)
            if sidecar.exists():
                shutil.move(str(sidecar), str(out_dir / sidecar.name))
            conn.execute("UPDATE images SET present = 0 WHERE path = ?", (path_str,))
            moved += 1
        conn.commit()
    finally:
        conn.close()
    return {"moved": moved, "per_model": preview["per_model"],
            "union": preview["union"], "output_dir": str(out_dir)}
