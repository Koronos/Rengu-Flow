import { describe, expect, it } from "vitest";
import {
  dictToKvRows,
  formatKvValue,
  kvModelMatchesRows,
  kvRowsToDict,
  parseKvValue,
} from "./keyValueList";

describe("keyValueList", () => {
  it("round-trips dict through rows", () => {
    const src = { T_max: "effective_total_steps", eta_min: 0.0, decouple: true };
    const rows = dictToKvRows(src);
    expect(rows).toHaveLength(3);
    expect(kvRowsToDict(rows)).toEqual(src);
  });

  it("parses booleans and numbers from strings", () => {
    expect(parseKvValue("true")).toBe(true);
    expect(parseKvValue("0.1")).toBe(0.1);
    expect(parseKvValue("total_steps")).toBe("total_steps");
  });

  it("formats values for display", () => {
    expect(formatKvValue(false)).toBe("false");
    expect(formatKvValue([50, 100])).toBe("[50,100]");
    expect(formatKvValue(1e-4)).toBe("1e-4");
    expect(formatKvValue(1e-6)).toBe("1e-6");
  });

  it("round-trips scientific KV strings", () => {
    const rows = [{ key: "lr", value: "1e-4" }, { key: "d0", value: "1e-6" }];
    expect(kvRowsToDict(rows)).toEqual({ lr: 1e-4, d0: 1e-6 });
  });

  it("parses JSON array values", () => {
    const rows = [{ key: "milestones", value: "[50, 100]" }];
    expect(kvRowsToDict(rows)).toEqual({ milestones: [50, 100] });
  });

  it("kvModelMatchesRows keeps draft rows when parent is empty", () => {
    const draft = [
      { key: "warmup_steps", value: "" },
      { key: "", value: "0.01" },
    ];
    expect(kvModelMatchesRows("", draft)).toBe(true);
    expect(kvModelMatchesRows({}, draft)).toBe(true);
  });

  it("kvModelMatchesRows detects external model changes", () => {
    const rows = [{ key: "lr_min", value: "0.01" }];
    expect(kvModelMatchesRows({ lr_min: 0.01 }, rows)).toBe(true);
    expect(kvModelMatchesRows({ lr_min: 0.02 }, rows)).toBe(false);
  });
});
