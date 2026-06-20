import { describe, expect, it } from "vitest";
import {
  applyOptimizerTypeChange,
  isCustomOptimizerType,
  pruneOptimizerForm,
  type OptimizerRegistries,
} from "./optimizerForm";

// Stand-in for the backend `registries` payload. Includes a kaon optimizer
// (nekaon) that the old hardcoded frontend list used to miss.
const REG: OptimizerRegistries = {
  optimizers: ["adamw", "sgd", "adam", "genericoptim", "adakaon", "adamuon", "prodigy", "nekaon"],
  optimizerKvDefaults: {
    adamw: { lr: 1e-4, betas: [0.9, 0.999], weight_decay: 0.01 },
    sgd: { lr: 1e-3, momentum: 0.9 },
    genericoptim: { lr: 1e-4, muon: false },
    adakaon: { lr: 1e-4, cautious: true },
    adamuon: { lr: 1e-3, ns_steps: 2 },
    prodigy: { lr: 1.0, d0: 1e-6 },
    nekaon: { lr: 1e-4, k: 1.5 },
  },
};

describe("optimizerForm", () => {
  it("applies KV defaults when switching to genericoptim", () => {
    const next = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "genericoptim", REG);
    expect(next["optimizer.extra_params"]).toEqual(REG.optimizerKvDefaults.genericoptim);
  });

  it("detects custom optimizer types against the registry", () => {
    expect(isCustomOptimizerType("adamw", REG)).toBe(false);
    expect(isCustomOptimizerType("prodigy", REG)).toBe(false);
    expect(isCustomOptimizerType("adakaon", REG)).toBe(false);
    // not in the registry list -> custom (no defaults to prefill)
    expect(isCustomOptimizerType("torch.optim.SGD", REG)).toBe(true);
    expect(isCustomOptimizerType("pytorch_optimizer.Prodigy", REG)).toBe(true);
    expect(isCustomOptimizerType("notareal_optimizer", REG)).toBe(true);
  });

  it("prefills defaults for a kaon optimizer the old hardcoded list missed", () => {
    const next = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "nekaon", REG);
    expect(next["optimizer.type"]).toBe("nekaon");
    expect(next["optimizer.extra_params"]).toEqual(REG.optimizerKvDefaults.nekaon);
  });

  it("normalizes mixed-case builtin values to the registry's lowercase value", () => {
    const next = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "AdaMuon", REG);
    expect(next["optimizer.type"]).toBe("adamuon");
    expect(next["optimizer.extra_params"]).toEqual(REG.optimizerKvDefaults.adamuon);
  });

  it("replaces KV when switching builtin types", () => {
    const form = {
      "optimizer.type": "adamw",
      "optimizer.extra_params": { lr: 1e-3, betas: [0.9, 0.95] },
    };
    const next = applyOptimizerTypeChange(form, "sgd", REG);
    expect(next["optimizer.type"]).toBe("sgd");
    expect(next["optimizer.extra_params"]).toEqual(REG.optimizerKvDefaults.sgd);
  });

  it("pruneOptimizerForm drops orphan flat keys", () => {
    const pruned = pruneOptimizerForm({
      "optimizer.type": "genericoptim",
      "optimizer.lr": 1e-4,
      "optimizer.momentum": 0.9,
      "optimizer.extra_params": REG.optimizerKvDefaults.genericoptim,
    });
    expect(pruned["optimizer.lr"]).toBeUndefined();
    expect(pruned["optimizer.momentum"]).toBeUndefined();
    expect(pruned["optimizer.extra_params"]).toEqual(REG.optimizerKvDefaults.genericoptim);
  });

  it("custom optimizer gets empty KV", () => {
    const next = applyOptimizerTypeChange(
      { "optimizer.type": "adamw", "optimizer.extra_params": REG.optimizerKvDefaults.adamw },
      "pytorch_optimizer.Prodigy",
      REG
    );
    expect(next["optimizer.extra_params"]).toEqual({});
  });

  it("treats everything as custom when registries are empty (degraded, no crash)", () => {
    const next = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "adamw", {
      optimizers: [],
      optimizerKvDefaults: {},
    });
    expect(next["optimizer.extra_params"]).toEqual({});
  });
});
