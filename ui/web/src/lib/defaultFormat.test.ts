import { describe, expect, it } from "vitest";
import {
  formatDefaultNumber,
  formatDefaultValue,
  formatScientific,
} from "./defaultFormat";

describe("defaultFormat", () => {
  it("formats small learning rates as scientific", () => {
    expect(formatDefaultNumber(1e-4)).toBe("1e-4");
    expect(formatDefaultNumber(1e-6)).toBe("1e-6");
    expect(formatDefaultNumber(1e-7)).toBe("1e-7");
  });

  it("keeps short decimals in fixed notation", () => {
    expect(formatDefaultNumber(0.01)).toBe("0.01");
    expect(formatDefaultNumber(0.5)).toBe("0.5");
    expect(formatDefaultNumber(7)).toBe("7");
    expect(formatDefaultNumber(0.999)).toBe("0.999");
  });

  it("uses scientific for more than three fractional digits", () => {
    expect(formatDefaultNumber(0.00001)).toBe("1e-5");
  });

  it("formats large magnitudes as scientific", () => {
    expect(formatDefaultNumber(100_000)).toBe("1e+5");
  });

  it("formatDefaultValue handles booleans and arrays", () => {
    expect(formatDefaultValue(false)).toBe("false");
    expect(formatDefaultValue([0.9, 0.999])).toBe("[0.9,0.999]");
  });

  it("formatScientific matches TOML-friendly strings", () => {
    expect(formatScientific(1.5e-3)).toBe("1.5e-3");
    expect(formatScientific(0)).toBe("0");
  });
});
