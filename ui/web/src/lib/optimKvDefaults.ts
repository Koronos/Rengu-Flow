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
  // github.com/Koronos/K-Optimizers (the `kaon` package; installed on demand via the "kaon" profile).
  // Adakaon: conv-aware factored optimizer (formerly "Adafusion").
  adakaon: {
    lr: 1e-4,
    betas: [0.9, 0.999],
    eps: [1e-30, 1e-3],
    weight_decay: 0.0,
    clip_threshold: 1.0,
    momentum_dtype: "bfloat16",
    cautious: true,
    bf16_method: "stochastic_rounding",
  },
  muon: {
    lr: 2e-2,
    momentum: 0.95,
    adamw_lr: 3e-4,
    bf16_method: "stochastic_rounding",
  },
  // AdaMuon: Muon orthogonalized momentum + factored quantized variance. kaon's
  // API default lr=2e-2 is Muon/LLM-scale; for diffusion use ~1e-3 (≈ AdamW lr ÷ 5).
  adamuon: {
    lr: 1e-3,
    betas: [0.95, 0.999],
    eps: [1e-30, 1e-3],
    weight_decay: 0.0,
    ns_steps: 2,
    clip_threshold: 1.0,
    momentum_dtype: "bfloat16",
    cautious: true,
    bf16_method: "stochastic_rounding",
  },
  // KProdigy: parameter-free Prodigy (D-adaptation). Train at lr=1.0; d0/d_coef are the knobs.
  kprodigy: {
    lr: 1.0,
    betas: [0.9, 0.999],
    weight_decay: 0.0,
    d0: 1e-6,
    d_coef: 1.0,
    momentum_dtype: "bfloat16",
    bf16_method: "stochastic_rounding",
  },
  // Autokaon: parameter-free LR on Adakaon via a Mechanic tuner (formerly "Autofusion").
  autokaon: {
    lr: 1.0,
    adakaon_betas: [0.0, 0.999],
    momentum_dtype: "bfloat16",
    bf16_method: "stochastic_rounding",
  },
  // Lion: sign-momentum on Adakaon's backend (formerly "Liofusion"; no 2nd moment).
  // betas are a loss<->generalization dial; (0.95, 0.98) (classic Lion) is the small-data start.
  lion: {
    lr: 2e-4,
    betas: [0.95, 0.98],
    weight_decay: 0.0,
    momentum_dtype: "bfloat16",
    cautious: true,
    bf16_method: "stochastic_rounding",
  },
  // AdaPNM: Adam + Positive-Negative Momentum. beta1 is the loss<->gap dial; beta0 the PN coefficient.
  adapnm: {
    lr: 1e-3,
    betas: [0.8, 0.999],
    beta0: 0.5,
    weight_decay: 0.0,
    cautious: true,
    momentum_dtype: "bfloat16",
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
