"""Optional per-step bench logging (memory + step time) for upgrade smokes."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch


def bench_enabled(config: dict) -> bool:
    return bool(config.get("bench", False))


def bench_init(run_dir: str) -> Path | None:
    path = Path(run_dir) / "bench_steps.csv"
    if not path.parent.exists():
        return None
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "loss",
                "iter_sec",
                "samples_per_sec",
                "cuda_alloc_gb",
                "cuda_reserved_gb",
                "cuda_peak_gb",
                "wall_sec",
                "overhead_sec",
                "wall_samples_per_sec",
            ]
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return path


def bench_record(
    csv_path: Path | None,
    *,
    step: int,
    loss: float,
    iter_sec: float,
    batch_size: int,
    wall_sec: float | None = None,
) -> None:
    if csv_path is None:
        return
    alloc = reserved = peak = 0.0
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        if os.environ.get("RENGU_BENCH_PEAK_PER_STEP") == "1":
            # Opt-in: report each step's own peak instead of the cumulative
            # high-water mark (which is dominated by the cold compile step).
            torch.cuda.reset_peak_memory_stats()
    sps = batch_size / iter_sec if iter_sec > 0 else 0.0
    wall_sec = iter_sec if wall_sec is None else wall_sec
    overhead_sec = max(0.0, wall_sec - iter_sec)
    wall_sps = batch_size / wall_sec if wall_sec > 0 else 0.0
    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                step,
                f"{loss:.6f}",
                f"{iter_sec:.4f}",
                f"{sps:.4f}",
                f"{alloc:.3f}",
                f"{reserved:.3f}",
                f"{peak:.3f}",
                f"{wall_sec:.4f}",
                f"{overhead_sec:.4f}",
                f"{wall_sps:.4f}",
            ]
        )
    print(
        f"[bench] step={step} loss={loss:.6f} iter_sec={iter_sec:.3f} "
        f"wall_sec={wall_sec:.3f} overhead_sec={overhead_sec:.3f} "
        f"samples/s={sps:.3f} wall_samples/s={wall_sps:.3f} cuda_peak_gb={peak:.2f}",
        flush=True,
    )


def bench_mean_iter_sec_after_warmup(
    csv_path: Path | str | None,
    *,
    min_step: int = 6,
) -> float | None:
    """Mean iter_sec for steps >= min_step (skip compile/warmup). Used by smoke A/B scripts."""
    if csv_path is None:
        return None
    path = Path(csv_path)
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open()))
    warm = [float(r["iter_sec"]) for r in rows if int(r["step"]) >= min_step]
    if not warm:
        return None
    return sum(warm) / len(warm)


def find_latest_bench_csv(output_dir: Path | str = "output") -> Path | None:
    """Newest bench_steps.csv under output_dir (by mtime)."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    candidates = list(root.glob("*/bench_steps.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def bench_summarize(csv_path: Path | None, label: str, run_dir: str) -> dict[str, float]:
    """Return summary stats and append one line to bench_summary.txt in run_dir."""
    out: dict[str, float] = {}
    if csv_path is None or not csv_path.is_file():
        return out
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return out
    losses = [float(r["loss"]) for r in rows]
    iters = [float(r["iter_sec"]) for r in rows]
    peaks = [float(r["cuda_peak_gb"]) for r in rows]
    sps = [float(r["samples_per_sec"]) for r in rows]
    out = {
        "steps": len(rows),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "iter_sec_mean": sum(iters) / len(iters),
        "iter_sec_p50": sorted(iters)[len(iters) // 2],
        "samples_per_sec_mean": sum(sps) / len(sps),
        "cuda_peak_gb_max": max(peaks),
    }
    if "wall_sec" in rows[0] and rows[0]["wall_sec"]:
        walls = [float(r["wall_sec"]) for r in rows]
        overheads = [float(r["overhead_sec"]) for r in rows]
        wall_sps = [float(r["wall_samples_per_sec"]) for r in rows]
        out.update(
            wall_sec_mean=sum(walls) / len(walls),
            overhead_sec_mean=sum(overheads) / len(overheads),
            wall_samples_per_sec_mean=sum(wall_sps) / len(wall_sps),
        )
    summary_path = Path(run_dir) / "bench_summary.txt"
    with summary_path.open("a") as f:
        f.write(f"\n=== {label} ===\n")
        for k, v in out.items():
            f.write(f"{k}: {v}\n")
    return out
