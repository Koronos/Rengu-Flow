"""Tests for bench CSV helpers used in smoke A/B."""

from pathlib import Path

from rengu_flow.utils.bench import (
    bench_mean_iter_sec_after_warmup,
    find_latest_bench_csv,
)


def test_bench_mean_iter_sec_after_warmup_skips_early_steps(tmp_path: Path):
    csv_path = tmp_path / "bench_steps.csv"
    csv_path.write_text(
        "step,loss,iter_sec,samples_per_sec,cuda_alloc_gb,cuda_reserved_gb,cuda_peak_gb\n"
        "1,1.0,1.0,1.0,0,0,0\n"
        "5,1.0,0.9,1.0,0,0,0\n"
        "6,1.0,0.4,1.0,0,0,0\n"
        "7,1.0,0.6,1.0,0,0,0\n",
        encoding="utf-8",
    )
    mean = bench_mean_iter_sec_after_warmup(csv_path, min_step=6)
    assert mean is not None
    assert abs(mean - 0.5) < 1e-6


def test_find_latest_bench_csv(tmp_path: Path):
    old = tmp_path / "run_old"
    new = tmp_path / "run_new"
    old.mkdir()
    new.mkdir()
    (old / "bench_steps.csv").write_text(
        "step,loss,iter_sec,samples_per_sec,cuda_alloc_gb,cuda_reserved_gb,cuda_peak_gb\n",
        encoding="utf-8",
    )
    import os
    import time

    time.sleep(0.02)
    (new / "bench_steps.csv").write_text(
        "step,loss,iter_sec,samples_per_sec,cuda_alloc_gb,cuda_reserved_gb,cuda_peak_gb\n"
        "6,1.0,0.5,1.0,0,0,0\n",
        encoding="utf-8",
    )
    os.utime(new / "bench_steps.csv", None)
    found = find_latest_bench_csv(tmp_path)
    assert found == new / "bench_steps.csv"
