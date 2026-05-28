import { describe, expect, it } from "vitest";
import {
  applySchedulerTypeChange,
  isCustomSchedulerType,
  pruneSchedulerForm,
} from "./schedulerForm";
import { SCHEDULER_BUILTIN_KV_DEFAULTS } from "./optimKvDefaults";

describe("schedulerForm", () => {
  it("detects custom scheduler types", () => {
    expect(isCustomSchedulerType("cosine")).toBe(false);
    expect(isCustomSchedulerType("torch.optim.lr_scheduler.CosineAnnealingLR")).toBe(true);
  });

  it("replaces KV when switching builtin types", () => {
    const form = {
      lr_scheduler: "cosine",
      warmup_steps: 100,
      "lr_scheduler_args.extra_params": { lr_min: 0.01 },
    };
    const next = applySchedulerTypeChange(form, "linear");
    expect(next.lr_scheduler).toBe("linear");
    expect(next.warmup_steps).toBe(100);
    expect(next["lr_scheduler_args.extra_params"]).toEqual(SCHEDULER_BUILTIN_KV_DEFAULTS.linear);
  });

  it("prunes orphan flat scheduler keys but keeps warmup_steps", () => {
    const pruned = pruneSchedulerForm({
      lr_scheduler: "cosine",
      warmup_steps: 50,
      "lr_scheduler_args.lr_min": 0.0,
      "lr_scheduler_args.extra_params": { lr_min: 0.0 },
    });
    expect(pruned.warmup_steps).toBe(50);
    expect(pruned["lr_scheduler_args.lr_min"]).toBeUndefined();
    expect(pruned["lr_scheduler_args.extra_params"]).toEqual({ lr_min: 0.0 });
  });

  it("prefills extra_params for suggested FQN schedulers", () => {
    const next = applySchedulerTypeChange(
      { lr_scheduler: "cosine" },
      "torch.optim.lr_scheduler.CosineAnnealingLR"
    );
    expect(next["lr_scheduler_args.extra_params"]).toMatchObject({
      T_max: "effective_total_steps",
    });
    expect(next["lr_scheduler_args.extra_params"]).not.toHaveProperty("warmup_steps");
  });

  it("prefills builtin cosine defaults", () => {
    const next = applySchedulerTypeChange({ lr_scheduler: "constant" }, "cosine");
    expect(next["lr_scheduler_args.extra_params"]).toMatchObject({
      lr_min: 0.0,
    });
    expect(next["lr_scheduler_args.extra_params"]).not.toHaveProperty("warmup_steps");
  });

  it("none scheduler gets empty KV and drops warmup field", () => {
    const next = applySchedulerTypeChange(
      {
        lr_scheduler: "cosine",
        warmup_steps: 5,
        "lr_scheduler_args.extra_params": { lr_min: 0.1 },
      },
      "none"
    );
    expect(next["lr_scheduler_args.extra_params"]).toEqual({});
    expect(next.warmup_steps).toBeUndefined();
  });
});
