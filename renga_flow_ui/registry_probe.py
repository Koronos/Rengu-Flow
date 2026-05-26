"""Probe whether optimizer / scheduler names resolve in the current environment."""

from __future__ import annotations

import importlib
from typing import Any

from renga_flow.optim.resolver import scheduler_registry
from renga_flow.registry.optimizers import (
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
        return {
            "available": True,
            "name": key,
            "resolved_class": f"{cls.__module__}.{cls.__qualname__}",
            "source": _optimizer_source(key.lower()),
        }
    except Exception as e:
        return {
            "available": False,
            "name": key,
            "error": str(e),
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
    errors: list[str] = []
    opt = resolution.get("optimizer")
    if opt and not opt.get("available"):
        errors.append(f"Optimizer: {opt.get('error', 'not available')}")
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
