"""Probe whether optimizer / scheduler names resolve in the current environment."""

from __future__ import annotations

import importlib
from typing import Any

from rengu_flow.optim.resolver import scheduler_registry
from rengu_flow.registry.optimizers import (
    OPTIMIZER_ALIASES,
    VENDOR_OPTIMIZER_ALIASES,
    get_optimizer_class,
    optimizer_registry,
)


def probe_optimizer(name: str) -> dict[str, Any]:
    """Try to resolve an optimizer type the same way training does."""
    if not name or not str(name).strip():
        return {"available": False, "error": "Optimizer type is empty."}
    key = str(name).strip()
    try:
        cls = get_optimizer_class(key)
    except Exception as e:
        # Known optional-dependency aliases (e.g. adakaon/muon from kaon, adamw8bit, prodigy)
        # are installed automatically by the autoinstaller when training starts (see
        # rengu_flow.install / rengu_flow.cli.training_extras). Don't surface a "not available" /
        # "please install" error for them — report them as resolvable so the UI doesn't nag.
        lower = key.lower()
        alias = OPTIMIZER_ALIASES.get(lower) or VENDOR_OPTIMIZER_ALIASES.get(lower)
        if alias is not None:
            module_path, class_name = alias
            return {
                "available": True,
                "name": key,
                "resolved_class": f"{module_path}.{class_name}",
                "source": _optimizer_source(lower),
                "deferred_install": True,
            }
        return {
            "available": False,
            "name": key,
            "error": str(e),
        }
    resolved = f"{cls.__module__}.{cls.__qualname__}"
    # A qualified path resolves any importable class, so guard that it is actually an
    # optimizer (e.g. catch a scheduler class pasted into the optimizer field).
    import torch

    if not (isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer)):
        return {
            "available": False,
            "name": key,
            "error": f"{resolved} is not a torch.optim.Optimizer.",
        }
    return {
        "available": True,
        "name": key,
        "resolved_class": resolved,
        "source": _optimizer_source(key.lower()),
    }


def probe_scheduler(name: str) -> dict[str, Any]:
    """Try to resolve a scheduler type (registry name or qualified class path)."""
    if not name or not str(name).strip():
        return {"available": False, "error": "Scheduler type is empty."}
    key = str(name).strip()
    lower = key.lower()
    if lower in scheduler_registry:
        return {
            "available": True,
            "name": key,
            "source": "registry",
            "resolved": lower,
        }
    if "." in key:
        try:
            module_path, class_name = key.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return {
                "available": True,
                "name": key,
                "source": "qualified",
                "resolved_class": f"{module_path}.{cls.__name__}",
            }
        except Exception as e:
            return {
                "available": False,
                "name": key,
                "error": str(e),
            }
    return {
        "available": False,
        "name": key,
        "error": (
            f"Unknown scheduler type '{key}'. "
            f"Registered: {sorted(scheduler_registry)}. "
            "Or use a fully-qualified class path."
        ),
    }


def probe_resolution(config: dict[str, Any]) -> dict[str, Any]:
    """Probe optimizer and lr_scheduler from a config dict (after defaults)."""
    out: dict[str, Any] = {}
    opt = config.get("optimizer") or {}
    if opt.get("type"):
        out["optimizer"] = probe_optimizer(str(opt["type"]))
    sched = config.get("lr_scheduler")
    if sched is not None and str(sched).strip():
        out["scheduler"] = probe_scheduler(str(sched))
    return out


def resolution_errors(resolution: dict[str, Any]) -> list[str]:
    """Validation issues from registry probe (scheduler only).

    Optimizer optional deps are installed automatically when training starts
    (see ``rengu_flow.cli.training_extras``); do not block save/validate here.
    """
    errors: list[str] = []
    sched = resolution.get("scheduler")
    if sched and not sched.get("available"):
        errors.append(f"LR scheduler: {sched.get('error', 'not available')}")
    return errors


def _optimizer_source(key: str) -> str:
    if key in optimizer_registry:
        return "registry"
    if key in VENDOR_OPTIMIZER_ALIASES:
        return "vendor"
    if key in OPTIMIZER_ALIASES:
        return "optional_dependency"
    if "." in key:
        return "qualified"
    return "pytorch_optimizer"
