import { describe, expect, it } from "vitest";
import {
  normalizeResolutions,
  normalizeStage,
  parseResolutionSchedule,
  scheduleFormValue,
  scheduleNeedsJsonEditor,
  stageEffectivePercent,
  validateResolutionScheduleJson,
} from "./resolutionSchedule";

describe("resolutionSchedule", () => {
  it("parses objects and JSON strings, accepting the `stage` or `stages` key", () => {
    const obj = {
      enabled: true,
      stage: [
        { resolutions: [512], fraction: 0.33 },
        { resolutions: [768, 1024], fraction: 0.67 },
      ],
    };
    expect(parseResolutionSchedule(obj)).toEqual({
      enabled: true,
      stages: [
        { resolutions: [512], fraction: 0.33 },
        { resolutions: [768, 1024], fraction: 0.67 },
      ],
    });
    expect(parseResolutionSchedule(JSON.stringify(obj)).stages).toHaveLength(2);
    expect(
      parseResolutionSchedule({ enabled: true, stages: [{ resolution: 512, fraction: 1 }] })
    ).toEqual({ enabled: true, stages: [{ resolutions: [512], fraction: 1 }] });
    expect(parseResolutionSchedule("")).toEqual({ enabled: false, stages: [] });
    expect(parseResolutionSchedule("not-json")).toEqual({ enabled: false, stages: [] });
  });

  it("normalizeStage rejects empty resolutions or non-positive fractions", () => {
    expect(normalizeStage({ resolutions: [], fraction: 1 })).toBeNull();
    expect(normalizeStage({ resolutions: [512], fraction: 0 })).toBeNull();
    expect(normalizeStage({ resolutions: [512], fraction: -1 })).toBeNull();
    expect(normalizeStage({ resolutions: [512.4, 0, "x"], fraction: 0.5 })).toEqual({
      resolutions: [512],
      fraction: 0.5,
    });
  });

  it("normalizeResolutions coerces scalars and drops invalid entries", () => {
    expect(normalizeResolutions(768)).toEqual([768]);
    expect(normalizeResolutions([512, "768", -1, 0, 1024.6])).toEqual([512, 768, 1025]);
    expect(normalizeResolutions("")).toEqual([]);
  });

  it("scheduleFormValue emits TOML-shaped JSON and blanks when empty+disabled", () => {
    expect(scheduleFormValue({ enabled: false, stages: [] })).toBe("");
    const value = scheduleFormValue({
      enabled: true,
      stages: [{ resolutions: [512], fraction: 0.33 }],
    });
    expect(JSON.parse(value as string)).toEqual({
      enabled: true,
      stage: [{ resolutions: [512], fraction: 0.33 }],
    });
    // Invalid stages are dropped on serialize.
    expect(
      JSON.parse(
        scheduleFormValue({
          enabled: true,
          stages: [{ resolutions: [], fraction: 1 }],
        }) as string
      )
    ).toEqual({ enabled: true, stage: [] });
    // Disabled but with stages -> keep them, mark inactive.
    expect(
      JSON.parse(
        scheduleFormValue({
          enabled: false,
          stages: [{ resolutions: [512], fraction: 1 }],
        }) as string
      )
    ).toEqual({ enabled: false, stage: [{ resolutions: [512], fraction: 1 }] });
  });

  it("stageEffectivePercent normalizes fractions to 100%", () => {
    expect(
      stageEffectivePercent([
        { resolutions: [512], fraction: 1 },
        { resolutions: [768], fraction: 1 },
        { resolutions: [1024], fraction: 2 },
      ])
    ).toEqual([25, 25, 50]);
    expect(stageEffectivePercent([])).toEqual([]);
  });

  it("validateResolutionScheduleJson reports shape errors", () => {
    expect(validateResolutionScheduleJson("")).toBeNull();
    expect(validateResolutionScheduleJson("nope")).toBe("Invalid JSON");
    expect(validateResolutionScheduleJson("[]")).toBe(
      "Expected an object like { enabled, stage: [...] }"
    );
    expect(
      validateResolutionScheduleJson('{"enabled":true,"stage":[{"resolutions":[],"fraction":1}]}')
    ).toMatch(/Stage 1: 'resolutions'/);
    expect(
      validateResolutionScheduleJson('{"enabled":true,"stage":[{"resolutions":[512],"fraction":0}]}')
    ).toMatch(/Stage 1: 'fraction'/);
    expect(
      validateResolutionScheduleJson('{"enabled":true,"stage":[{"resolutions":[512],"fraction":0.5}]}')
    ).toBeNull();
  });

  it("scheduleNeedsJsonEditor flags only unrepresentable values", () => {
    expect(scheduleNeedsJsonEditor("")).toBe(false);
    expect(scheduleNeedsJsonEditor({ enabled: true, stage: [{ resolutions: [512], fraction: 1 }] })).toBe(
      false
    );
    expect(scheduleNeedsJsonEditor("broken")).toBe(true);
    expect(scheduleNeedsJsonEditor({ enabled: true, stage: { not: "array" } })).toBe(true);
    expect(scheduleNeedsJsonEditor({ enabled: true, stage: [{ resolutions: [], fraction: 1 }] })).toBe(
      true
    );
  });
});
