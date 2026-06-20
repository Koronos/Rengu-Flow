import { describe, expect, it } from "vitest";
import { applyAdapterTypeChange, pruneAdapterForm, adapterFieldsFromSchema } from "./adapterForm";
import type { SchemaField } from "../types/forms";

const HAS_ADAPTER = { field: "_has_adapter", equals: true };
const forTypes = (...kinds: string[]) => ({
  all: [HAS_ADAPTER, { field: "adapter.type", in: kinds }],
});

const SCHEMA = {
  sections: [
    {
      id: "adapter",
      fields: [
        { path: "adapter.type", type: "select", visibility: HAS_ADAPTER },
        { path: "adapter.rank", type: "integer", default: 16, visibility: HAS_ADAPTER },
        { path: "adapter.dtype", type: "select", visibility: HAS_ADAPTER },
        { path: "adapter.dropout", type: "number", default: 0, visibility: forTypes("lora", "lokr", "lycoris_locon") },
        { path: "adapter.factor", type: "integer", default: -1, visibility: forTypes("lokr") },
        { path: "adapter.decompose_both", type: "boolean", default: false, visibility: forTypes("lokr") },
        { path: "adapter.wd_on_output", type: "boolean", default: true, visibility: forTypes("lycoris_locon") },
      ] as SchemaField[],
    },
    { id: "model", fields: [{ path: "model.type", type: "select" }] as SchemaField[] },
  ],
};

describe("adapterForm", () => {
  it("extracts only adapter.* fields from the schema", () => {
    const paths = adapterFieldsFromSchema(SCHEMA).map((f) => f.path);
    expect(paths).toContain("adapter.factor");
    expect(paths).not.toContain("model.type");
  });

  it("drops previous-type keys when switching lokr -> lora", () => {
    const next = applyAdapterTypeChange(
      {
        _has_adapter: true,
        "adapter.type": "lokr",
        "adapter.rank": 16,
        "adapter.factor": 8,
        "adapter.decompose_both": true,
        "model.type": "sdxl",
        "optimizer.type": "adamw",
      },
      "lora",
      SCHEMA,
      {}
    );
    expect(next["adapter.type"]).toBe("lora");
    expect(next["adapter.rank"]).toBe(16);
    expect("adapter.factor" in next).toBe(false);
    expect("adapter.decompose_both" in next).toBe(false);
    // other sections untouched
    expect(next["model.type"]).toBe("sdxl");
    expect(next["optimizer.type"]).toBe("adamw");
  });

  it("seeds meaningful defaults for the new type, skipping falsy ones", () => {
    const next = applyAdapterTypeChange(
      { _has_adapter: true, "adapter.type": "lora", "adapter.rank": 16 },
      "lokr",
      SCHEMA,
      {}
    );
    expect(next["adapter.factor"]).toBe(-1); // non-falsy default seeded
    expect("adapter.decompose_both" in next).toBe(false); // false default not seeded
  });

  it("does not seed wd_on_output=true noise but seeds for the right type", () => {
    const next = applyAdapterTypeChange(
      { _has_adapter: true, "adapter.type": "lora" },
      "lycoris_locon",
      SCHEMA,
      {}
    );
    expect(next["adapter.wd_on_output"]).toBe(true); // meaningful (true) default seeded
    expect("adapter.factor" in next).toBe(false); // lokr-only, not applicable
  });

  it("keeps the user's existing common value instead of resetting to default", () => {
    const next = applyAdapterTypeChange(
      { _has_adapter: true, "adapter.type": "lora", "adapter.rank": 64 },
      "lokr",
      SCHEMA,
      {}
    );
    expect(next["adapter.rank"]).toBe(64);
  });

  it("pruneAdapterForm leaves the form alone when type is empty", () => {
    const form = { "adapter.factor": 8 };
    expect(pruneAdapterForm(form, adapterFieldsFromSchema(SCHEMA), {})).toBe(form);
  });
});
