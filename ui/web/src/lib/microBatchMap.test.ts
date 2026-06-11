import { describe, expect, it } from "vitest";
import { microBatchIssues, parseMicroBatch, serializeMicroBatch } from "./microBatchMap";

describe("parseMicroBatch", () => {
  it("treats integers as uniform mode", () => {
    expect(parseMicroBatch(2)).toEqual({ mode: "uniform", uniform: 2, rows: [] });
  });

  it("treats unset/garbage as empty uniform", () => {
    for (const v of [undefined, null, "", "abc", -1, 0]) {
      expect(parseMicroBatch(v).mode).toBe("uniform");
      expect(parseMicroBatch(v).uniform).toBeUndefined();
    }
  });

  it("parses a resolution map (string keys, sorted ascending)", () => {
    const m = parseMicroBatch({ "1024": 1, "512": 2 });
    expect(m.mode).toBe("per_resolution");
    expect(m.rows).toEqual([
      { resolution: 512, batch: 2 },
      { resolution: 1024, batch: 1 },
    ]);
  });
});

describe("serializeMicroBatch", () => {
  it("round-trips uniform", () => {
    expect(serializeMicroBatch(parseMicroBatch(3))).toBe(3);
  });

  it("round-trips a map (keys back to strings)", () => {
    const value = { "512": 2, "1024": 1 };
    expect(serializeMicroBatch(parseMicroBatch(value))).toEqual(value);
  });

  it("drops incomplete rows and returns undefined when nothing is usable", () => {
    const m = parseMicroBatch({ "512": 2 });
    m.rows.push({ resolution: undefined, batch: 4 });
    expect(serializeMicroBatch(m)).toEqual({ "512": 2 });
    expect(
      serializeMicroBatch({ mode: "per_resolution", uniform: undefined, rows: [] }),
    ).toBeUndefined();
    expect(serializeMicroBatch({ mode: "uniform", uniform: undefined, rows: [] })).toBeUndefined();
  });
});

describe("microBatchIssues", () => {
  it("flags incomplete rows and duplicate resolutions", () => {
    const m = parseMicroBatch({ "512": 2 });
    expect(microBatchIssues(m)).toEqual([]);
    m.rows.push({ resolution: 512, batch: 1 });
    expect(microBatchIssues(m).join(" ")).toContain("Duplicate resolution 512");
    m.rows.push({ resolution: undefined, batch: undefined });
    expect(microBatchIssues(m).join(" ")).toContain("needs a resolution");
  });

  it("uniform mode never has issues", () => {
    expect(microBatchIssues(parseMicroBatch(5))).toEqual([]);
  });
});
