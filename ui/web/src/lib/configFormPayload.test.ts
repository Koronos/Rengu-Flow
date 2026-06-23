import { describe, expect, it } from "vitest";
import { reactive } from "vue";
import { sanitizeConfigForm } from "./configFormPayload";

describe("sanitizeConfigForm", () => {
  it("returns a plain cloned form for ordinary input", () => {
    const result = sanitizeConfigForm({
      "optimizer.type": "adamw",
      "optimizer.extra_params": { lr: 1e-4 },
    });
    expect(result).not.toBeNull();
    expect(result?.["optimizer.type"]).toBe("adamw");
    expect(result?.["optimizer.extra_params"]).toEqual({ lr: 1e-4 });
  });

  it("preserves values that carry Vue reactive proxies (registry-sourced KV defaults)", () => {
    // Regression: applyOptimizerTypeChange seeds optimizer.extra_params from the reactive
    // schema registry via a shallow spread, so nested arrays (e.g. betas) stay reactive
    // proxies. structuredClone throws on those, sanitize returned null, and setForm then
    // silently dropped the whole update — the optimizer picker appeared frozen.
    const reactiveKv = reactive({ lr: 2e-4, betas: [0.95, 0.98] });
    const result = sanitizeConfigForm({
      "optimizer.type": "lion",
      "optimizer.extra_params": { ...reactiveKv },
    });
    expect(result).not.toBeNull();
    expect(result?.["optimizer.type"]).toBe("lion");
    expect(result?.["optimizer.extra_params"]).toEqual({ lr: 2e-4, betas: [0.95, 0.98] });
  });
});
