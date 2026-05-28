"""Dev maintenance helpers: DB reset, git submodules, dependency install commands."""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from renga_flow.install_profiles import (
    PROFILE_DESCRIPTIONS,
    PROFILE_LABELS,
    normalize_profiles,
    renga_init_command,
    uv_sync_argv,
)
from renga_flow_ui import optional_deps
from renga_flow_ui import db, settings

SCHEMA_VERSION = 1
DB_RESET_CONFIRM_TOKEN = "RESET"

# Optional dependency profiles (pyproject.toml extras).
DEP_PROFILES: dict[str, dict[str, str]] = {
    "base": {
        "label": PROFILE_LABELS["base"],
        "description": PROFILE_DESCRIPTIONS["base"],
        "command": renga_init_command(["base"]),
    },
    "ui": {
        "label": PROFILE_LABELS["ui"],
        "description": PROFILE_DESCRIPTIONS["ui"],
        "command": renga_init_command(["ui"]),
    },
    "cosmos_predict2": {
        "label": PROFILE_LABELS["cosmos_predict2"],
        "description": PROFILE_DESCRIPTIONS["cosmos_predict2"],
        "command": renga_init_command(["cosmos_predict2"]),
    },
}


@dataclass
class CommandResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    command: list[str]


def maintenance_enabled() -> bool:
    """True when destructive maintenance API/UI is allowed."""
    for key in ("RENGAFLOW_MAINTENANCE", "RENGA_FLOW_MAINTENANCE"):
        val = os.environ.get(key, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
    return False


def maintenance_allow_pip() -> bool:
    val = os.environ.get("RENGAFLOW_MAINTENANCE_ALLOW_PIP", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _run_git(args: list[str], *, timeout: int = 300) -> CommandResult:
    cmd = ["git", *args]
    proc = subprocess.run(
        cmd,
        cwd=str(settings.repo_root()),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CommandResult(
        ok=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        command=cmd,
    )


def _db_file_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "modified_at": None,
        }
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
    }


def _list_tables(path: Path) -> list[str]:
    if not path.is_file():
        return []
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def _gitmodules_entries() -> list[dict[str, str]]:
    gitmodules = settings.repo_root() / ".gitmodules"
    if not gitmodules.is_file():
        return []
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in gitmodules.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("[submodule"):
            if current.get("path"):
                entries.append(current)
            current = {}
            continue
        m = re.match(r'(\w+)\s*=\s*"?([^"]+)"?', line)
        if m and current is not None:
            current[m.group(1)] = m.group(2).strip()
    if current.get("path"):
        entries.append(current)
    return entries


def _parse_submodule_status(stdout: str) -> list[dict[str, str]]:
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    out: list[dict[str, str]] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        status = parts[0]
        path = parts[1]
        commit = parts[2] if len(parts) > 2 else ""
        out.append({"path": path, "status": status, "commit": commit, "line": line})
    return out


def get_status() -> dict[str, Any]:
    path = settings.db_path()
    tables = _list_tables(path)
    gitmodules = _gitmodules_entries()
    submodule_status: list[dict[str, str]] = []
    git_available = False
    if (settings.repo_root() / ".git").exists():
        git_available = True
        if gitmodules:
            res = _run_git(["submodule", "status", "--recursive"])
            if res.ok:
                submodule_status = _parse_submodule_status(res.stdout)
            else:
                submodule_status = [{"path": "(error)", "status": "?", "commit": "", "line": res.stderr.strip()}]

    req_files: list[dict[str, Any]] = []
    for name in ("requirements.txt", "requirements-cosmos.txt", "requirements/cosmos.txt"):
        p = settings.repo_root() / name
        req_files.append({"path": name, "exists": p.is_file(), "resolved": str(p.resolve()) if p.is_file() else None})

    profiles = []
    for key, meta in DEP_PROFILES.items():
        profiles.append(
            {
                "id": key,
                "label": meta["label"],
                "description": meta["description"],
                "command": meta["command"],
                "pip_allowed": maintenance_allow_pip(),
            }
        )

    return {
        "enabled": maintenance_enabled(),
        "schema_version": SCHEMA_VERSION,
        "database": {
            **_db_file_info(path),
            "tables": tables,
            "expected_tables": ["jobs", "training_configs", "datasets"],
        },
        "ui_data_dir": str(settings.ui_data_dir().resolve()),
        "repo_root": str(settings.repo_root().resolve()),
        "python_executable": sys.executable,
        "git": {
            "available": git_available,
            "gitmodules_exists": bool(gitmodules),
            "submodules_configured": gitmodules,
            "submodule_status": submodule_status,
        },
        "requirements_files": req_files,
        "pyproject_exists": (settings.repo_root() / "pyproject.toml").is_file(),
        "dependency_profiles": profiles,
        "pip_install_from_server": maintenance_allow_pip(),
    }


def reset_database(*, confirm: bool) -> dict[str, Any]:
    if not confirm:
        raise ValueError(
            f"Confirmation required: pass confirm=true or confirmation={DB_RESET_CONFIRM_TOKEN!r}"
        )
    before = _list_tables(settings.db_path())
    path = db.reset_ui_database()
    after = _list_tables(path)
    return {
        "ok": True,
        "path": str(path.resolve()),
        "tables_before": before,
        "tables_after": after,
        "message": "Database wiped and recreated with empty schema.",
    }


def submodule_update() -> dict[str, Any]:
    if not (settings.repo_root() / ".git").is_dir():
        return {
            "ok": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "Not a git repository.",
            "command": [],
            "message": "Submodule update requires a git checkout.",
        }
    if not _gitmodules_entries():
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "No .gitmodules in this repository (renga-flow vendors Cosmos code in-tree).\n",
            "stderr": "",
            "command": [],
            "message": "No submodules configured.",
        }
    res = _run_git(["submodule", "update", "--init", "--recursive"], timeout=600)
    return {
        "ok": res.ok,
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "command": res.command,
        "message": "Submodule update finished." if res.ok else "Submodule update failed.",
    }


def deps_install(profile: str, *, execute: bool, confirm: bool) -> dict[str, Any]:
    if profile not in DEP_PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from: {', '.join(DEP_PROFILES)}")
    meta = DEP_PROFILES[profile]
    command_str = meta["command"]
    if not execute:
        return {
            "ok": True,
            "executed": False,
            "profile": profile,
            "command": command_str,
            "message": "Copy and run at repo root, or enable RENGAFLOW_MAINTENANCE_ALLOW_PIP=1 and pass execute=true.",
        }
    if not maintenance_allow_pip():
        raise ValueError(
            "Server-side install is disabled. Set RENGAFLOW_MAINTENANCE_ALLOW_PIP=1 to allow execute=true."
        )
    if not confirm:
        raise ValueError("Pass confirm=true to run uv sync from the UI server process.")
    if profile in optional_deps.OPTIONAL_PROFILE_IDS:
        return optional_deps.install_optional_profile(
            profile,
            execute=execute,
            confirm=confirm,
        )
    profile_key = "cosmos" if profile == "cosmos_predict2" else profile
    argv = uv_sync_argv(normalize_profiles([profile_key]))
    proc = subprocess.run(
        argv,
        cwd=str(settings.repo_root()),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    command_str = " ".join(argv)
    return {
        "ok": proc.returncode == 0,
        "executed": True,
        "profile": profile,
        "command": command_str,
        "returncode": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "message": "uv sync finished." if proc.returncode == 0 else "uv sync failed.",
    }


def command_result_dict(res: CommandResult) -> dict[str, Any]:
    return {
        "ok": res.ok,
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "command": res.command,
    }
