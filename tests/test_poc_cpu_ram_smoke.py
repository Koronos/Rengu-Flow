"""Fast smoke for POC benchmark script (no GPU)."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_poc_cpu_ram_script_runs_quickly():
    env = {**os.environ, "POC_TMP": str(REPO / "tmp" / "poc_smoke_run")}
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "poc_cpu_ram_optimizations.py"), "--json", "tmp/poc_smoke_ci.json"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "storage_mmap_bf16_vs_pickle" in proc.stdout
