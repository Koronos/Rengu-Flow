import { describe, expect, it } from "vitest";
import { emaSmooth } from "./smoothing";

const pts = (values: number[]) => values.map((value, step) => ({ step, value }));

describe("emaSmooth", () => {
  it("returns the series unchanged when weight is 0", () => {
    const input = pts([1, 5, 2, 8]);
    expect(emaSmooth(input, 0)).toBe(input);
  });

  it("keeps a constant series constant (debiased)", () => {
    const out = emaSmooth(pts([5, 5, 5, 5]), 0.9);
    for (const p of out) expect(p.value).toBeCloseTo(5, 6);
  });

  it("starts at the first value", () => {
    const out = emaSmooth(pts([2, 10, 3, 7]), 0.8);
    expect(out[0].value).toBeCloseTo(2, 6);
  });

  it("reduces variance vs the raw series", () => {
    const raw = [0, 10, 0, 10, 0, 10];
    const out = emaSmooth(pts(raw), 0.9).map((p) => p.value);
    const spread = Math.max(...out) - Math.min(...out);
    expect(spread).toBeLessThan(10);
  });

  it("preserves steps and length", () => {
    const out = emaSmooth(pts([1, 2, 3]), 0.5);
    expect(out.map((p) => p.step)).toEqual([0, 1, 2]);
  });
});
