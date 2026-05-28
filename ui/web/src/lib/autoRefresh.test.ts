import { describe, expect, it } from "vitest";
import {
  DEFAULT_AUTO_REFRESH_SEC,
  effectiveRefreshMs,
  parseStoredInterval,
} from "./autoRefresh";

describe("parseStoredInterval", () => {
  it("defaults to 10s for missing or invalid values", () => {
    expect(parseStoredInterval(null)).toBe(DEFAULT_AUTO_REFRESH_SEC);
    expect(parseStoredInterval("99")).toBe(DEFAULT_AUTO_REFRESH_SEC);
  });

  it("accepts configured options", () => {
    expect(parseStoredInterval("0")).toBe(0);
    expect(parseStoredInterval("5")).toBe(5);
    expect(parseStoredInterval("30")).toBe(30);
    expect(parseStoredInterval("60")).toBe(60);
    expect(parseStoredInterval("300")).toBe(300);
  });
});

describe("effectiveRefreshMs", () => {
  it("returns 0 when off", () => {
    expect(effectiveRefreshMs(0, true)).toBe(0);
  });

  it("uses selected interval when active", () => {
    expect(effectiveRefreshMs(10, true)).toBe(10_000);
  });

  it("does not poll when inactive", () => {
    expect(effectiveRefreshMs(5, false)).toBe(0);
    expect(effectiveRefreshMs(10, false)).toBe(0);
    expect(effectiveRefreshMs(30, false)).toBe(0);
  });

  it("uses selected interval for long options when active", () => {
    expect(effectiveRefreshMs(60, true)).toBe(60_000);
    expect(effectiveRefreshMs(300, true)).toBe(300_000);
  });
});
