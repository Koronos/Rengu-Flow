"""Default key-value constructor args for registry optimizers and scheduler types."""

from __future__ import annotations

from typing import Any

# Documented in docs/user/optimizer-and-scheduler.md and scheduler field help.
SCHEDULER_RUNTIME_TOKENS: tuple[str, ...] = (
    "total_steps",
    "effective_total_steps",
    "steps_per_epoch",
    "epochs",
    "max_steps",
    "gradient_accumulation_steps",
)

# One-line train-time meaning for each token (matches rengu_flow.optim.resolver).
SCHEDULER_RUNTIME_TOKEN_HINTS: dict[str, str] = {
    "total_steps": (
        "Optimizer steps for the full run (epochs × steps_per_epoch), passed when the scheduler is resolved."
    ),
    "effective_total_steps": (
        "min(total_steps, max_steps) when max_steps is a positive integer in config; otherwise total_steps."
    ),
    "steps_per_epoch": (
        "Optimizer steps in one epoch after gradient accumulation, passed when the scheduler is resolved."
    ),
    "epochs": "epochs from config (default 1 if omitted).",
    "max_steps": (
        "max_steps from config when set to a positive integer; omitted from substitution otherwise."
    ),
    "gradient_accumulation_steps": (
        "gradient_accumulation_steps from config when set; omitted from substitution otherwise."
    ),
}

# Built-in optimizer.type values -> default [optimizer] KV rows (form pre-fill on type change).
OPTIMIZER_REGISTRY_KV_DEFAULTS: dict[str, dict[str, Any]] = {
    "adamw": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
    },
    "adam": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
    },
    "sgd": {
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.0,
    },
    "adamw8bit": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
    },
    "adamw_optimi": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
    },
    "stableadamw": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
    },
    "adamw8bitkahan": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
        "kahan_buffer_offload": False,
    },
    "offload": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
    },
    "prodigy": {
        "lr": 1.0,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "d0": 1e-6,
        "d_coef": 1.0,
        "weight_decouple": True,
        "bias_correction": True,
        "safeguard_warmup": True,
    },
    "genericoptim": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
        "muon": False,
        "adamuon": False,
        "correct_bias": True,
    },
    "automagic": {
        "min_lr": 1e-7,
        "max_lr": 1e-3,
        "lr_bump": 1e-6,
    },
    # github.com/Koronos/K-Optimizers (installed on demand via the "koptim" profile).
    "adafusion": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": [1e-30, 1e-3],
        "weight_decay": 0.0,
        "clip_threshold": 1.0,
        "momentum_dtype": "bfloat16",
        "cautious": True,
        "bf16_method": "stochastic_rounding",
    },
    "muon": {
        "lr": 2e-2,
        "momentum": 0.95,
        "adamw_lr": 3e-4,
        "bf16_method": "stochastic_rounding",
    },
    # AdaMuon: Muon orthogonalized momentum + factored quantized variance.
    # NOTE: koptim's API default lr=2e-2 is Muon/LLM-scale; for diffusion use a much
    # lower lr (~1e-3, ≈ AdamW's lr ÷ 5) — that is what we pre-fill here.
    "adamuon": {
        "lr": 1e-3,
        "betas": [0.95, 0.999],
        "eps": [1e-30, 1e-3],
        "weight_decay": 0.0,
        "ns_steps": 2,
        "clip_threshold": 1.0,
        "momentum_dtype": "bfloat16",
        "cautious": True,
        "bf16_method": "stochastic_rounding",
    },
}

# Built-in lr_scheduler registry names -> default scheduler KV ([lr_scheduler_args] only).
SCHEDULER_BUILTIN_KV_DEFAULTS: dict[str, dict[str, Any]] = {
    "none": {},
    "constant": {
        "factor": 1.0,
    },
    "linear": {
        "start_factor": 1.0,
        "end_factor": 0.0,
        "total_iters": "total_steps",
    },
    "cosine": {
        "lr_min": 0.0,
    },
    "rex": {
        "lr_min": 0.0,
        "rex_d": 0.5,
    },
}

# Fully-qualified scheduler paths -> default [lr_scheduler_args] (runtime tokens as strings).
SCHEDULER_FQN_KV_DEFAULTS: dict[str, dict[str, Any]] = {
    "torch.optim.lr_scheduler.CosineAnnealingLR": {
        "T_max": "effective_total_steps",
        "eta_min": 0.0,
    },
    "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts": {
        "T_0": "steps_per_epoch",
        "T_mult": 2,
        "eta_min": 0.0,
    },
    "torch.optim.lr_scheduler.StepLR": {
        "step_size": "steps_per_epoch",
        "gamma": 0.1,
    },
    "torch.optim.lr_scheduler.MultiStepLR": {
        "milestones": [50],
        "gamma": 0.1,
    },
    "torch.optim.lr_scheduler.OneCycleLR": {
        "max_lr": 1e-4,
        "total_steps": "effective_total_steps",
        "pct_start": 0.3,
    },
    "torch.optim.lr_scheduler.ExponentialLR": {
        "gamma": 0.95,
    },
    "torch.optim.lr_scheduler.PolynomialLR": {
        "total_iters": "effective_total_steps",
        "power": 1.0,
    },
    "torch.optim.lr_scheduler.LinearLR": {
        "start_factor": 1.0,
        "end_factor": 0.0,
        "total_iters": "effective_total_steps",
    },
    "torch.optim.lr_scheduler.ConstantLR": {
        "factor": 1.0,
        "total_iters": "effective_total_steps",
    },
}

# Shown in scheduler type autocomplete (FQNs with curated defaults).
SUGGESTED_SCHEDULER_FQNS: tuple[str, ...] = tuple(SCHEDULER_FQN_KV_DEFAULTS.keys())

_NONE_SCHEDULER = frozenset({"none"})


def optimizer_extra_params_defaults(opt_type: str) -> dict[str, Any]:
    """Default optimizer KV rows for a registry name (lowercase) or empty for unknown/custom."""
    key = opt_type.strip().lower()
    return dict(OPTIMIZER_REGISTRY_KV_DEFAULTS.get(key, {}))


def scheduler_kv_defaults(sched_type: str) -> dict[str, Any]:
    """Default scheduler KV rows for built-in names, FQNs, and custom class paths."""
    name = sched_type.strip()
    if not name:
        return {}
    if "." in name:
        kv = dict(SCHEDULER_FQN_KV_DEFAULTS.get(name, {}))
    else:
        kv = dict(SCHEDULER_BUILTIN_KV_DEFAULTS.get(name.lower(), {}))
    return kv


def scheduler_extra_params_defaults(sched_type: str) -> dict[str, Any]:
    """Alias for scheduler_kv_defaults (FQN-focused name kept for callers)."""
    return scheduler_kv_defaults(sched_type)
