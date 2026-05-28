import { describe, expect, it } from "vitest";
import {
  emptyAugmentationConfig,
  parseAugmentationConfig,
  parseGlobalAugmentation,
  serializeDirectoryAugmentation,
  serializeGlobalAugmentation,
  setStrategyOverride,
  shouldWriteDirectoryAugmentation,
  shouldWriteGlobalAugmentation,
} from "./datasetAugmentation";

describe("datasetAugmentation", () => {
  it("parses global augmentation JSON", () => {
    const config = parseGlobalAugmentation({
      _dataset_augmentation: '{"enabled": true, "preset": "easy"}',
    });
    expect(config?.enabled).toBe(true);
    expect(config?.preset).toBe("easy");
  });

  it("parses directory augmentation with strategy overrides", () => {
    const config = parseAugmentationConfig({
      enabled: true,
      preset: "photo_safe",
      strategies: {
        horizontal_flip: { enabled: false },
      },
    });
    expect(config?.strategies?.horizontal_flip?.enabled).toBe(false);
  });

  it("serializes directory augmentation for form rows", () => {
    const row = serializeDirectoryAugmentation({
      enabled: true,
      preset: "easy",
      strategies: { color_jitter: { brightness: 0.04 } },
    });
    expect(row?.enabled).toBe(true);
    expect(typeof row?.strategies).toBe("string");
  });

  it("serializes global augmentation as JSON text", () => {
    const text = serializeGlobalAugmentation({ enabled: false, preset: "none" });
    expect(text).toContain('"enabled": false');
    expect(text).toContain('"preset": "none"');
  });

  it("setStrategyOverride removes empty strategy entries", () => {
    const base = {
      enabled: true,
      preset: "easy",
      strategies: { gamma: { enabled: false } },
    };
    const next = setStrategyOverride(base, "gamma", null);
    expect(next.strategies).toBeUndefined();
  });

  it("shouldWriteGlobalAugmentation omits default-off config", () => {
    expect(shouldWriteGlobalAugmentation(emptyAugmentationConfig())).toBe(false);
    expect(shouldWriteGlobalAugmentation({ enabled: true, preset: "easy" })).toBe(true);
  });

  it("shouldWriteDirectoryAugmentation keeps strategy-only overrides", () => {
    const global = { enabled: true, preset: "easy" };
    expect(
      shouldWriteDirectoryAugmentation(
        { strategies: { horizontal_flip: { enabled: false } } },
        global
      )
    ).toBe(true);
    expect(shouldWriteDirectoryAugmentation({ enabled: true, preset: "easy" }, global)).toBe(
      false
    );
  });

  it("serializeDirectoryAugmentation strips inherited keys", () => {
    const global = { enabled: true, preset: "easy" };
    const row = serializeDirectoryAugmentation(
      {
        enabled: true,
        preset: "easy",
        strategies: { color_jitter: { brightness: 0.04 } },
      },
      { global }
    );
    expect(row?.enabled).toBeUndefined();
    expect(row?.preset).toBeUndefined();
    expect(typeof row?.strategies).toBe("string");
  });
});
