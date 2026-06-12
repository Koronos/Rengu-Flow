import { describe, expect, it } from "vitest";
import { RUN_COLOR_PALETTE, buildRunColorMap, colorForRun } from "./runColors";

describe("runColors", () => {
  it("cycles the palette and wraps", () => {
    expect(colorForRun(0)).toBe(RUN_COLOR_PALETTE[0]);
    expect(colorForRun(RUN_COLOR_PALETTE.length)).toBe(RUN_COLOR_PALETTE[0]);
  });

  it("assigns a stable color per run by position", () => {
    const map = buildRunColorMap(["a", "b", "c"]);
    expect(map.a).toBe(colorForRun(0));
    expect(map.b).toBe(colorForRun(1));
    expect(map.c).toBe(colorForRun(2));
    // A run keeps its color as long as the list order is stable.
    expect(buildRunColorMap(["a", "b", "c"]).b).toBe(map.b);
  });
});
