import { describe, expect, it } from "vitest";
import {
  applyOptimizerTypeChange,
  isCustomOptimizerType,
  KNOWN_BUILTIN_OPTIMIZER_TYPES,
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
    // kaon aliases are builtin (auto-installed), not custom — so their params populate.
    expect(isCustomOptimizerType("adakaon")).toBe(false);
    expect(isCustomOptimizerType("muon")).toBe(false);
    expect(isCustomOptimizerType("adamuon")).toBe(false);
    expect(isCustomOptimizerType("AdaMuon")).toBe(false);
    expect(isCustomOptimizerType("torch.optim.SGD")).toBe(true);
    expect(isCustomOptimizerType("pytorch_optimizer.Prodigy")).toBe(true);
  });

  it("prefills adakaon/muon/adamuon KV defaults (kaon optimizers show their params)", () => {
    const adakaon = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "adakaon");
    expect(adakaon["optimizer.type"]).toBe("adakaon");
    expect(adakaon["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.adakaon);
    const muon = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "muon");
    expect(muon["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.muon);
    // AdaMuon: the option value from the registry select is "AdaMuon" (mixed case).
    const adamuon = applyOptimizerTypeChange({ "optimizer.type": "adamw" }, "AdaMuon");
    expect(adamuon["optimizer.type"]).toBe("adamuon");
    expect(adamuon["optimizer.extra_params"]).toEqual(OPTIMIZER_REGISTRY_KV_DEFAULTS.adamuon);
    expect(adamuon["optimizer.extra_params"]).toMatchObject({ lr: 1e-3, ns_steps: 2 });
  });

  it("every known builtin optimizer type has KV defaults (guards against mirror drift)", () => {
    for (const type of KNOWN_BUILTIN_OPTIMIZER_TYPES) {
      expect(
        Object.keys(OPTIMIZER_REGISTRY_KV_DEFAULTS[type] ?? {}).length,
        `missing KV defaults for builtin optimizer "${type}"`
      ).toBeGreaterThan(0);
    }
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
