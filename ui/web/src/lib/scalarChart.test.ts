import { describe, expect, it } from "vitest";
import {
  formatScalarValue,
  formatWallTime,
  nearestPointIndex,
  resolveStepJumpTag,
} from "./scalarChart";

describe("nearestPointIndex", () => {
  const rect = { left: 0, width: 100 } as DOMRect;

  it("returns -1 for empty series", () => {
    expect(nearestPointIndex(50, rect, 0)).toBe(-1);
  });

  it("maps x position to nearest index", () => {
    expect(nearestPointIndex(0, rect, 5)).toBe(0);
    expect(nearestPointIndex(100, rect, 5)).toBe(4);
    expect(nearestPointIndex(50, rect, 5)).toBe(2);
  });
});

describe("formatScalarValue", () => {
  it("formats small numbers with fixed decimals", () => {
    expect(formatScalarValue(0.123456)).toBe("0.123456");
  });
});

describe("formatWallTime", () => {
  it("formats unix seconds", () => {
    const s = formatWallTime(1_700_000_000);
    expect(s).toBeTruthy();
    expect(s).toMatch(/2023|2024|2025|2026/);
  });

  it("returns null for missing time", () => {
    expect(formatWallTime(undefined)).toBeNull();
  });
});

describe("resolveStepJumpTag", () => {
  it("prefers grad_norm when present", () => {
    expect(
      resolveStepJumpTag({
        "train/grad_norm": [{ step: 1, value: 1 }],
        "train/prodigy_d": [{ step: 1, value: 2 }],
      })
    ).toBe("train/grad_norm");
  });

  it("falls back to prodigy_d", () => {
    expect(
      resolveStepJumpTag({
        "train/prodigy_d": [{ step: 1, value: 2 }],
      })
    ).toBe("train/prodigy_d");
  });
});
