import { describe, expect, it } from "vitest";
import {
  normalizeSizeBucket,
  parseSizeBuckets,
  sizeBucketsFormValue,
  sizeBucketsNeedJsonEditor,
  validateSizeBucketsJson,
} from "./sizeBuckets";

describe("sizeBuckets", () => {
  it("parseSizeBuckets accepts arrays and JSON strings", () => {
    expect(parseSizeBuckets([[512, 512, 1], [768, 768, 1]])).toEqual([
      [512, 512, 1],
      [768, 768, 1],
    ]);
    expect(parseSizeBuckets("[[512, 512, 1], [768, 768, 1]]")).toEqual([
      [512, 512, 1],
      [768, 768, 1],
    ]);
    expect(parseSizeBuckets("")).toEqual([]);
  });

  it("normalizeSizeBucket rejects invalid rows", () => {
    expect(normalizeSizeBucket([512, 512])).toBeNull();
    expect(normalizeSizeBucket([0, 512, 1])).toBeNull();
    expect(normalizeSizeBucket([512.5, 512, 1])).toBeNull();
    expect(normalizeSizeBucket([512, 512, 1])).toEqual([512, 512, 1]);
  });

  it("sizeBucketsFormValue serializes empty as blank", () => {
    expect(sizeBucketsFormValue([])).toBe("");
    expect(sizeBucketsFormValue([[512, 512, 1]])).toEqual([[512, 512, 1]]);
  });

  it("validateSizeBucketsJson reports parse and shape errors", () => {
    expect(validateSizeBucketsJson("")).toBeNull();
    expect(validateSizeBucketsJson("not-json")).toBe("Invalid JSON");
    expect(validateSizeBucketsJson("{}")).toBe(
      "Expected an array of [width, height, frames] rows"
    );
    expect(validateSizeBucketsJson("[[512, 512]]")).toBe(
      "Row 1: expected [width, height, frames]"
    );
    expect(validateSizeBucketsJson("[[512, 512, 1]]")).toBeNull();
  });

  it("sizeBucketsNeedJsonEditor detects malformed values", () => {
    expect(sizeBucketsNeedJsonEditor([[512, 512, 1]])).toBe(false);
    expect(sizeBucketsNeedJsonEditor("[[512, 512, 1]]")).toBe(false);
    expect(sizeBucketsNeedJsonEditor("[[512, 512]]")).toBe(true);
    expect(sizeBucketsNeedJsonEditor("{bad")).toBe(true);
  });
});
