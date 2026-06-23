"""Read and write ``rengu.local.toml`` for the UI Settings section.

Uses tomlkit so writes preserve the file's comments and formatting. Only the editable and
restart-required fields are ever written; everything else in the document is left untouched.
Binding fields (host/port/data_dir) and the toolbox execution toggle are surfaced read-only —
they only take effect at startup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomlkit

from rengu_flow.config.local_config import (
    default_local_config,
    local_config_path,
    parse_local_config_dict,
    repo_root,
)

# Editable field whitelists, by TOML section. A patch may only set these keys.
# Maintenance is intentionally not surfaced here — the section is disabled (see
# rengu_flow_ui.maintenance) and its [maintenance] TOML keys are left untouched.
_EDITABLE_TRAINING_SCALARS = ("num_gpus", "master_port", "extra_args", "engine")
_RESTART_UI = ("public", "token")
_ENGINE_VALUES = ("", "deepspeed", "accelerate")


class SettingsError(ValueError):
    """Invalid settings patch (bad value or non-editable key)."""


def config_path() -> Path:
    """Path to ``rengu.local.toml``. Indirection point so tests can target a temp file."""
    return local_config_path()


def read_settings(path: Path | None = None) -> dict[str, Any]:
    """Return the current settings as a structured dict.

    Reads ``rengu.local.toml`` when it exists; falls back to compiled defaults otherwise.
    The returned dict groups fields into ``editable``, ``restartRequired``, and ``readOnly``.
    """
    p = path or config_path()
    exists = p.is_file()
    if exists:
        cfg = parse_local_config_dict(tomlkit.parse(p.read_text(encoding="utf-8")), root=repo_root())
    else:
        cfg = default_local_config()
    from rengu_flow.engine import resolve_backend
    from rengu_flow.platform_compat import PLATFORM

    effective_engine = resolve_backend({"engine": cfg.training.engine})
    return {
        "path": str(p),
        "exists": exists,
        "editable": {
            "training": {
                "num_gpus": cfg.training.num_gpus,
                "master_port": cfg.training.master_port,
                "extra_args": cfg.training.extra_args,
                "engine": cfg.training.engine,
                "env": dict(cfg.training.env),
            },
        },
        # Host training capabilities — the UI hides multi-GPU / DeepSpeed-only settings
        # (num_gpus, master_port) when the effective engine isn't 'deepspeed'.
        "host": {
            "is_windows": PLATFORM.is_windows,
            "effective_engine": effective_engine,
            "deepspeed": effective_engine == "deepspeed",
        },
        "restartRequired": {"ui": {"public": cfg.ui.public, "token": cfg.ui.token}},
        "readOnly": {
            "ui": {"host": cfg.ui.host, "port": cfg.ui.port, "data_dir": cfg.ui.data_dir},
            "toolbox": {"enabled": cfg.toolbox.enabled},
        },
    }


def _validate_patch(patch: dict[str, Any]) -> None:
    """Raise ``SettingsError`` if *patch* contains unknown sections or invalid values."""
    allowed_sections = {"training", "ui"}
    unknown = set(patch) - allowed_sections
    if unknown:
        raise SettingsError(f"Unknown settings section(s): {', '.join(sorted(unknown))}")

    training = patch.get("training", {})
    bad = set(training) - set(_EDITABLE_TRAINING_SCALARS) - {"env"}
    if bad:
        raise SettingsError(f"Non-editable training key(s): {', '.join(sorted(bad))}")
    if "num_gpus" in training and (not isinstance(training["num_gpus"], int) or training["num_gpus"] < 1):
        raise SettingsError("num_gpus must be an integer >= 1")
    if "master_port" in training:
        v = training["master_port"]
        if not isinstance(v, int) or not (1 <= v <= 65535):
            raise SettingsError("master_port must be an integer in 1..65535")
    if "extra_args" in training and not isinstance(training["extra_args"], str):
        raise SettingsError("extra_args must be a string")
    if "engine" in training:
        v = training["engine"]
        if not isinstance(v, str) or v.strip().lower() not in _ENGINE_VALUES:
            raise SettingsError(f"engine must be one of {_ENGINE_VALUES}")
    if "env" in training:
        env = training["env"]
        if not isinstance(env, dict):
            raise SettingsError("training.env must be a table")
        for k in env:
            if not isinstance(k, str) or not k.strip():
                raise SettingsError("training.env keys must be non-empty strings")

    ui = patch.get("ui", {})
    bad_u = set(ui) - set(_RESTART_UI)
    if bad_u:
        raise SettingsError(f"Non-editable ui key(s): {', '.join(sorted(bad_u))}")
    if "public" in ui and not isinstance(ui["public"], bool):
        raise SettingsError("ui.public must be a boolean")
    if "token" in ui and ui["token"] is not None and not isinstance(ui["token"], str):
        raise SettingsError("ui.token must be a string or null")


def _table(doc: Any, name: str) -> Any:
    if name not in doc:
        doc[name] = tomlkit.table()
    return doc[name]


def write_settings(patch: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    _validate_patch(patch)
    p = path or config_path()
    if p.is_file():
        doc = tomlkit.parse(p.read_text(encoding="utf-8"))
    else:
        doc = tomlkit.document()

    if "training" in patch:
        t = _table(doc, "training")
        for key in _EDITABLE_TRAINING_SCALARS:
            if key in patch["training"]:
                t[key] = patch["training"][key]
        if "env" in patch["training"]:
            env_tbl = tomlkit.table()
            for k, v in patch["training"]["env"].items():
                env_tbl[k] = str(v)
            t["env"] = env_tbl

    if "ui" in patch:
        u = _table(doc, "ui")
        if "public" in patch["ui"]:
            u["public"] = patch["ui"]["public"]
        if "token" in patch["ui"]:
            tok = patch["ui"]["token"]
            if tok:
                u["token"] = tok
            elif "token" in u:  # falsy (None or empty string) — clear the key
                del u["token"]

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
    os.replace(tmp, p)
    return read_settings(p)
