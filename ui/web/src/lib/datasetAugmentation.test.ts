import { describe, expect, it } from "vitest";
import {
  applyPresetChange,
  compactStrategiesForStorage,
  effectiveStrategyMap,
  effectiveStrategyNames,
  emptyAugmentationConfig,
  parseAugmentationConfig,
  parseGlobalAugmentation,
  presetStrategyDefaults,
  serializeDirectoryAugmentation,
  serializeGlobalAugmentation,
  setStrategyOverride,
  shouldWriteDirectoryAugmentation,
  shouldWriteGlobalAugmentation,
  strategyParamsDiff,
  type AugmentationCatalog,
} from "./datasetAugmentation";

const mockCatalog: AugmentationCatalog = {
  presets: [
    {
      name: "easy",
      label: "Easy",
      available: true,
      strategies: ["color_jitter", "gamma"],
      strategy_defaults: {
        color_jitter: {
          enabled: true,
          brightness: 0.03,
          contrast: 0.03,
          saturation: 0.03,
          hue: 0.01,
        },
        gamma: { enabled: true, gamma_min: 0.97, gamma_max: 1.03 },
      },
    },
    { name: "none", label: "None", available: true, strategies: [], strategy_defaults: {} },
  ],
  strategies: [
    {
      name: "color_jitter",
      label: "Color jitter",
      category: "photometric",
      implemented: true,
      parameters: [{ path: "brightness", label: "Brightness", type: "number", default: 0.05 }],
    },
  ],
};

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

  it("effectiveStrategyMap merges preset defaults with overrides", () => {
    const config = {
      enabled: true,
      preset: "easy",
      strategies: { color_jitter: { brightness: 0.04 } },
    };
    const map = effectiveStrategyMap(config, mockCatalog);
    expect(map.color_jitter.brightness).toBe(0.04);
    expect(map.color_jitter.contrast).toBe(0.03);
    expect(effectiveStrategyNames(config, mockCatalog)).toEqual(["color_jitter", "gamma"]);
  });

  it("applyPresetChange clears strategy overrides", () => {
    const next = applyPresetChange(
      { enabled: true, preset: "easy", strategies: { gamma: { gamma_min: 0.9 } } },
      "photo_safe"
    );
    expect(next.preset).toBe("photo_safe");
    expect(next.strategies).toBeUndefined();
  });

  it("strategyParamsDiff omits unchanged preset fields", () => {
    const defaults = presetStrategyDefaults(mockCatalog, "easy").color_jitter;
    expect(strategyParamsDiff(defaults, { ...defaults, brightness: 0.04 })).toEqual({
      brightness: 0.04,
    });
    expect(strategyParamsDiff(defaults, defaults)).toBeNull();
  });

  it("compactStrategiesForStorage drops redundant overrides", () => {
    const config = {
      enabled: true,
      preset: "easy",
      strategies: {
        color_jitter: {
          enabled: true,
          brightness: 0.03,
          contrast: 0.03,
          saturation: 0.03,
          hue: 0.01,
        },
      },
    };
    const compact = compactStrategiesForStorage(config, mockCatalog);
    expect(compact.strategies).toBeUndefined();
  });
});
