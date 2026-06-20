/**
 * Default KV maps for schedulers. Keep in sync with rengu_flow_ui/optim_kv_defaults.py.
 *
 * Optimizer KV defaults are NOT mirrored here: they come from the backend via
 * `registries.optimizer_kv_defaults` (see optimizerForm.ts) so the form prefill
 * tracks the optimizer registry and cannot drift.
 */

export const SCHEDULER_RUNTIME_TOKENS = [
  "total_steps",
  "effective_total_steps",
  "steps_per_epoch",
  "epochs",
  "max_steps",
  "gradient_accumulation_steps",
] as const;

export const SCHEDULER_BUILTIN_KV_DEFAULTS: Record<string, Record<string, unknown>> = {
  none: {},
  constant: {
    factor: 1.0,
  },
  linear: {
    start_factor: 1.0,
    end_factor: 0.0,
    total_iters: "total_steps",
  },
  cosine: {
    lr_min: 0.0,
  },
  rex: {
    lr_min: 0.0,
    rex_d: 0.5,
  },
};

export const SCHEDULER_FQN_KV_DEFAULTS: Record<string, Record<string, unknown>> = {
  "torch.optim.lr_scheduler.CosineAnnealingLR": {
    T_max: "effective_total_steps",
    eta_min: 0.0,
  },
  "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts": {
    T_0: "steps_per_epoch",
    T_mult: 2,
    eta_min: 0.0,
  },
  "torch.optim.lr_scheduler.StepLR": {
    step_size: "steps_per_epoch",
    gamma: 0.1,
  },
  "torch.optim.lr_scheduler.MultiStepLR": {
    milestones: [50],
    gamma: 0.1,
  },
  "torch.optim.lr_scheduler.OneCycleLR": {
    max_lr: 1e-4,
    total_steps: "effective_total_steps",
    pct_start: 0.3,
  },
  "torch.optim.lr_scheduler.ExponentialLR": {
    gamma: 0.95,
  },
  "torch.optim.lr_scheduler.PolynomialLR": {
    total_iters: "effective_total_steps",
    power: 1.0,
  },
  "torch.optim.lr_scheduler.LinearLR": {
    start_factor: 1.0,
    end_factor: 0.0,
    total_iters: "effective_total_steps",
  },
  "torch.optim.lr_scheduler.ConstantLR": {
    factor: 1.0,
    total_iters: "effective_total_steps",
  },
};


export function schedulerKvDefaults(schedType: unknown): Record<string, unknown> {
  const name = String(schedType ?? "").trim();
  if (!name) return {};
  let kv: Record<string, unknown>;
  if (name.includes(".")) {
    kv = { ...(SCHEDULER_FQN_KV_DEFAULTS[name] ?? {}) };
  } else {
    kv = { ...(SCHEDULER_BUILTIN_KV_DEFAULTS[name.toLowerCase()] ?? {}) };
  }
  return kv;
}

/** @deprecated Use schedulerKvDefaults */
export function schedulerExtraParamsDefaults(schedType: unknown): Record<string, unknown> {
  return schedulerKvDefaults(schedType);
}
