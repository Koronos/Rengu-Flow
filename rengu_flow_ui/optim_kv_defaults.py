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
    # github.com/Koronos/K-Optimizers (the `kaon` package; installed on demand via the "kaon" profile).
    # Adakaon: conv-aware factored optimizer (formerly "Adafusion").
    "adakaon": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": [1e-30, 1e-3],
        "weight_decay": 0.0,
        "clip_threshold": 1.0,
        "momentum_dtype": "bfloat16",
        "cautious": True,
        "fused": False,
        "bf16_method": "stochastic_rounding",
    },
    # AdaMuon: Muon orthogonalized momentum + factored quantized variance.
    # NOTE: kaon's API default lr=2e-2 is Muon/LLM-scale; for diffusion use a much
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
    # KProdigy: memory-efficient parameter-free Prodigy (D-adaptation). Train at lr=1.0 and
    # it discovers the effective LR itself; d0/d_coef are the D-adaptation knobs.
    "kprodigy": {
        "lr": 1.0,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "d0": 1e-6,
        "d_coef": 1.0,
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # Autokaon: parameter-free LR on Adakaon via a Mechanic tuner (formerly "Autofusion").
    # Train at lr=1.0; the tuner finds the scale. s_init/lr_freeze/scale_cap stay on "auto".
    "autokaon": {
        "lr": 1.0,
        "adakaon_betas": [0.0, 0.999],
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # Lion: sign-momentum (EvoLved Sign Momentum) on Adakaon's quantized-momentum backend
    # (formerly "Liofusion"; no 2nd moment). betas are a loss<->generalization dial; (0.95, 0.98)
    # (classic Lion) is the recommended small-data starting point. lr is sign-update scale (~AdamW lr x2).
    "lion": {
        "lr": 2e-4,
        "betas": [0.95, 0.98],
        "weight_decay": 0.0,
        "momentum_dtype": "bfloat16",
        "cautious": True,
        "bf16_method": "stochastic_rounding",
    },
    # AdaPNM: Adam + Positive-Negative Momentum on the factored/quantized backend. beta1 is the
    # loss<->gap dial (0.8 shipped elbow); beta0 is the PN coefficient (0.5 default, 0 = plain Adam).
    "adapnm": {
        "lr": 1e-3,
        "betas": [0.8, 0.999],
        "beta0": 0.5,
        "weight_decay": 0.0,
        "clip_threshold": 1.0,
        "cautious": True,
        "momentum_dtype": "bfloat16",
        "fused": False,
        "bf16_method": "stochastic_rounding",
    },
    # AdaBelief: Adam on the variance of the gradient residual (g - m) on the factored backend.
    # eps is deliberately tiny (1e-16) — AdaBelief's denominator is a variance, not an RMS.
    "adabelief": {
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "cautious": True,
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # AdamP: AdamW minus the radial update component on scale-invariant weights.
    # delta/wd_ratio are the projection knobs (defaults match the paper).
    "adamp": {
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "cautious": True,
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # ADOPT: modified Adam that converges with any beta2 (v-lag + normalize-then-momentum).
    # betas default to (0.9, 0.9999) — the higher beta2 is intentional and safe here.
    "adopt": {
        "lr": 1e-3,
        "betas": [0.9, 0.9999],
        "weight_decay": 0.0,
        "cautious": True,
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # ScheduleFree: Schedule-Free AdamW (iterate averaging; no LR schedule needed — pair with
    # lr_scheduler "none"). warmup_steps optionally ramps the LR over the first N steps.
    "schedulefree": {
        "lr": 2.5e-3,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "cautious": True,
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # Lookahead: k-step slow-weight averaging wrapper over Adakaon. k = sync period, alpha =
    # slow-weight step. Extra inner Adakaon kwargs (lr, betas, …) are passed through.
    "lookahead": {
        "lr": 1e-4,
        "k": 5,
        "alpha": 0.5,
        "betas": [0.9, 0.999],
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # SAM: Sharpness-Aware Minimization (two-pass flat-minima) wrapper over Adakaon. rho = the
    # neighborhood radius. Extra inner Adakaon kwargs (lr, betas, …) are passed through.
    "sam": {
        "lr": 1e-4,
        "rho": 0.05,
        "betas": [0.9, 0.999],
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # MSAM: Momentum-SAM (Becker et al. 2024) — SAM's perturbation along the stored momentum,
    # zero extra forward/backward and zero extra state (wrapper over Adakaon). rho = perturbation
    # radius (rho < 0 probes the downhill/Nesterov direction). Inner Adakaon kwargs pass through.
    # Sampling/checkpointing must bracket with opt.eval()/opt.train() — renga does this automatically.
    "msam": {
        "lr": 1e-4,
        "rho": 0.3,
        "betas": [0.9, 0.999],
        "momentum_dtype": "bfloat16",
        "bf16_method": "stochastic_rounding",
    },
    # Nekaon: Adakaon + k-step negative momentum-lookahead (the in-house flat-minima flagship);
    # gradient evaluated k steps ahead at zero extra passes/state. k = lookahead distance in steps
    # (loss<->gap dial; 0 = plain Adakaon). betas[0] is the regime knob: 0.5 default (anti-memorization
    # with margin), slide to 0.2 for small-data LoRA or 0.9 for fidelity (must be > 0). momentum_dtype
    # defaults to "4bit" (~0.56 B/param, the optimizer's deliberate default — flat-within-noise vs bf16).
    # Sampling/checkpointing must bracket with opt.eval()/opt.train() — renga does this automatically.
    "nekaon": {
        "lr": 1e-4,
        "k": 1.5,
        "betas": [0.5, 0.999],
        "weight_decay": 0.1,
        "cautious": True,
        "momentum_dtype": "4bit",
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
