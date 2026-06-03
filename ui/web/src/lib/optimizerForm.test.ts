import { describe, expect, it } from "vitest";
import {
  applyOptimizerTypeChange,
  isCustomOptimizerType,
  pruneOptimizerForm,
} from "./optimizerForm";
import { OPTIMIZER_REGISTRY_KV_DEFAULTS } from "./optimKvDefaults";

describe("optimizerForm", () => {
  it("applies KV defaults when switching to genericoptim", () => {
    const next = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "genericoptim");
    expect(next["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.genericoptim);
  });

  it("detects custom optimizer types", () => {
    expect(isCustomOptimizerType("adamw")).toBe(false);
    expect(isCustomOptimizerType("prodigy")).toBe(false);
    // koptim aliases are builtin (auto-installed), not custom — so their params populate.
    expect(isCustomOptimizerType("adafusion")).toBe(false);
    expect(isCustomOptimizerType("muon")).toBe(false);
    expect(isCustomOptimizerType("torch.optim.SGD")).toBe(true);
    expect(isCustomOptimizerType("pytorch_optimizer.Prodigy")).toBe(true);
  });

  it("prefills adafusion/muon KV defaults (koptim optimizers show their params)", () => {
    const adafusion = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "adafusion");
    expect(adafusion["optimizer.type"]).toBe("adafusion");
    expect(adafusion["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.adafusion);
    const muon = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "muon");
    expect(muon["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.muon);
  });

  it("replaces KV when switching builtin types", () => {
    const form = {
      "optimizer.type": "adamw",
      "optimizer.extra_params": { lr: 1e-3, betas: [0.9, 0.95] },
    };
    const next = applyOptimizerTypeChange(form, "sgd");
    expect(next["optimizer.type"]).toBe("sgd");
    expect(next["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.sgd);
  });

  it("pruneOptimizerForm drops orphan flat keys", () => {
    const pruned = pruneOptimizerForm({
      "optimizer.type": "genericoptim",
      "optimizer.lr": 1e-4,
      "optimizer.betas": [0.9, 0.999],
      "optimizer.momentum": 0.9,
      "optimizer.extra_params": OPTIMIZER_REGISTRY_KV_DEFAULTS.genericoptim,
    });
    expect(pruned["optimizer.lr"]).toBeUndefined();
    expect(pruned["optimizer.momentum"]).toBeUndefined();
    expect(pruned["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.genericoptim);
  });

  it("prefills adamw defaults including lr and betas", () => {
    const next = applyOptimizerTypeChange({ "optimizer.type": "sgd" }, "adamw");
    expect(next["optimizer.extra_params"]).toMatchObject({
      lr: 1e-4,
      betas: [0.9, 0.999],
    });
  });

  it("prefills prodigy KV defaults", () => {
    const next = applyOptimizerTypeChange(
      {
        "optimizer.type": "adamw",
        "optimizer.extra_params": OPTIMIZER_REGISTRY_KV_DEFAULTS.adamw,
      },
      "prodigy"
    );
    expect(next["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.prodigy);
  });

  it("custom optimizer gets empty KV", () => {
    const next = applyOptimizerTypeChange(
      {
        "optimizer.type": "adamw",
        "optimizer.extra_params": OPTIMIZER_REGISTRY_KV_DEFAULTS.adamw,
      },
      "pytorch_optimizer.Prodigy"
    );
    expect(next["optimizer.extra_params"]).toEqual({});
  });
});
