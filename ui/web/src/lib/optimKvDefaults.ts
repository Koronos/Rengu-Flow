/**
 * Default KV maps for optimizers/schedulers. Keep in sync with rengu_flow_ui/optim_kv_defaults.py.
 */

export const SCHEDULER_RUNTIME_TOKENS = [
  "total_steps",
  "effective_total_steps",
  "steps_per_epoch",
  "epochs",
  "max_steps",
  "gradient_accumulation_steps",
] as const;

export const OPTIMIZER_REGISTRY_KV_DEFAULTS: Record<string, Record<string, unknown>> = {
  adamw: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    weight_decay: 0.01,
  },
  adam: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    weight_decay: 0.0,
  },
  sgd: {
    lr: 1e-3,
    momentum: 0.9,
    weight_decay: 0.0,
  },
  adamw8bit: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    weight_decay: 0.01,
  },
  adamw_optimi: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    weight_decay: 0.01,
  },
  stableadamw: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    weight_decay: 0.01,
  },
  adamw8bitkahan: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    weight_decay: 0.01,
    kahan_buffer_offload: false,
  },
  offload: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    weight_decay: 0.01,
  },
  prodigy: {
    lr: 1.0,
    betas: [0.9, 0.99],
    weight_decay: 0.01,
    d0: 1e-6,
    d_coef: 1.0,
    weight_decouple: true,
    bias_correction: true,
    safeguard_warmup: true,
  },
  genericoptim: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    weight_decay: 0.01,
    muon: false,
    adamuon: false,
    correct_bias: true,
  },
  automagic: {
    min_lr: 1e-7,
    max_lr: 1e-3,
    lr_bump: 1e-6,
  },
  // github.com/Koronos/K-Optimizers (installed on demand via the "koptim" profile).
  adafusion: {
    lr: 1e-4,
    betas: [0.0, 0.999],
    bf16_method: "stochastic_rounding",
    momentum_dtype: "bfloat16",
    compile: false,
  },
  muon: {
    lr: 2e-2,
    momentum: 0.95,
    adamw_lr: 3e-4,
    bf16_method: "stochastic_rounding",
  },
};

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

export function optimizerExtraParamsDefaults(optType: unknown): Record<string, unknown> {
  const key = String(optType ?? "")
    .trim()
    .toLowerCase();
  return { ...(OPTIMIZER_REGISTRY_KV_DEFAULTS[key] ?? {}) };
}

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
