"""Capture run provenance at startup: git state, command, environment, hardware.

All functions are pure and best-effort — a missing git binary, a non-repo cwd, or an absent
GPU degrades to partial info rather than raising. This is the metadata TensorBoard can't hold;
it lands in ``run.json`` so a viewer can answer "what produced this result?". ``capture`` is
torch-free (git + command + env); ``hardware`` is split out because it imports torch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any

# Cap the stored diff so run.json stays small even with a large dirty tree.
_MAX_DIFF_CHARS = 100_000

# Packages worth pinning in the run's environment snapshot.
_ENV_PACKAGES = (
    "rengu-flow",
    "torch",
    "torchvision",
    "deepspeed",
    "diffusers",
    "transformers",
    "peft",
    "kaon",
)


def _git(args: list[str], cwd: str | None = None) -> str | None:
    git = shutil.which("git")
    if not git:
        return None
    try:
        out = subprocess.run(
            [git, *args], cwd=cwd, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_lineage(cwd: str | None = None) -> dict[str, Any]:
    """Commit, branch, dirty flag and (capped) uncommitted diff for the repo at ``cwd``."""
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"available": False}
    status = _git(["status", "--porcelain"], cwd)
    dirty = bool(status)
    diff = _git(["diff", "HEAD"], cwd) if dirty else ""
    truncated = False
    if diff and len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS]
        truncated = True
    return {
        "available": True,
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "dirty": dirty,
        "diff": diff or "",
        "diff_truncated": truncated,
    }


def command() -> dict[str, Any]:
    """The exact invocation: argv, python version, interpreter path."""
    return {
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


def environment() -> dict[str, str]:
    """Versions of the key packages (best-effort via importlib.metadata)."""
    out: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return out
    for pkg in _ENV_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def hardware() -> dict[str, Any]:
    """GPU model(s), VRAM, compute capability, CUDA/torch versions (torch required)."""
    hw: dict[str, Any] = {"available": False}
    try:
        import torch
    except ImportError:
        return hw
    hw["available"] = True
    hw["torch_version"] = torch.__version__
    hw["cuda_version"] = getattr(torch.version, "cuda", None)
    hw["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
            hw["gpu_count"] = count
            hw["gpus"] = [
                {
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                }
                for props in (torch.cuda.get_device_properties(i) for i in range(count))
            ]
        except Exception:
            pass
    return hw


def capture(cwd: str | None = None) -> dict[str, Any]:
    """Torch-free provenance bundle (git + command + environment) for the manifest lineage."""
    return {
        "git": git_lineage(cwd),
        "command": command(),
        "environment": environment(),
    }
