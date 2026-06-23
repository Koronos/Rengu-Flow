import { describe, expect, it } from "vitest";
import { reactive } from "vue";
import { sanitizeDatasetForm } from "./datasetFormPayload";

describe("sanitizeDatasetForm", () => {
  it("returns a plain cloned form for ordinary input", () => {
    const result = sanitizeDatasetForm({ name: "set", resolution: [512, 1024] });
    expect(result).not.toBeNull();
    expect(result?.name).toBe("set");
    expect(result?.resolution).toEqual([512, 1024]);
  });

  it("preserves values that carry Vue reactive proxies (registry-sourced defaults)", () => {
    // Same regression class as sanitizeConfigForm: nested reactive proxies (e.g. an array
    // copied from the reactive schema registry) make structuredClone throw, sanitize return
    // null, and setForm silently drop the update.
    const reactiveDefaults = reactive({ buckets: [256, 512] });
    const result = sanitizeDatasetForm({ name: "set", _meta: { ...reactiveDefaults } });
    expect(result).not.toBeNull();
    expect(result?.name).toBe("set");
    expect((result?._meta as { buckets: number[] }).buckets).toEqual([256, 512]);
  });
});
