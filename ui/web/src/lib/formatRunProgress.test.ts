import { describe, expect, it } from "vitest";
import { formatRunProgressHint } from "./formatRunProgress";

describe("formatRunProgressHint", () => {
  it("prefers the EMA-smoothed s/it over the instant step time", () => {
    const hint = formatRunProgressHint({
      step_time_sec: 9.9,
      step_time_sec_ema: 2.5,
      eta: "5m",
    });
    expect(hint).toBe("2.50 s/it · ETA 5m");
  });

  it("falls back to the instant step time when no EMA is present", () => {
    const hint = formatRunProgressHint({ step_time_sec: 3, eta: "1m" });
    expect(hint).toBe("3.00 s/it · ETA 1m");
  });

  it("derives s/it from steps_per_second when no step time is given", () => {
    const hint = formatRunProgressHint({ steps_per_second_ema: 0.5 });
    expect(hint).toBe("2.00 s/it");
  });

  it("returns an empty string for no progress", () => {
    expect(formatRunProgressHint(null)).toBe("");
  });
});
