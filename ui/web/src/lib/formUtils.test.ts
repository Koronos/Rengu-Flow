import { describe, expect, it } from "vitest";
import { adapterOptionsForModel, fieldVisible, isTruthyFormValue } from "./formUtils";
import type { FormValues, ModelCapabilities, SchemaField } from "../types/forms";

describe("isTruthyFormValue", () => {
  it("treats explicit falsy values (incl. KV strings) as false", () => {
    for (const v of [undefined, null, false, 0, "", "false", "False", "0", "no", "off", "none"]) {
      expect(isTruthyFormValue(v)).toBe(false);
    }
  });

  it("treats real values (incl. string 'true') as true", () => {
    for (const v of [true, 1, "true", "True", "yes", "1"]) {
      expect(isTruthyFormValue(v)).toBe(true);
    }
  });
});

describe("fieldVisible with form_map_truthy", () => {
  // Mirrors the block_swap_prefetch gate: capability + blocks_to_swap>0 + gradient_release.
  const field: SchemaField = {
    path: "block_swap_prefetch",
    label: "Block-swap prefetch",
    type: "boolean",
    visibility: {
      all: [
        { capability: "block_swap" },
        { form_nonempty: "blocks_to_swap", exclude_zero: true },
        { form_map_truthy: { path: "optimizer.extra_params", key: "gradient_release" } },
      ],
    },
  };
  const caps = { sdxl: { type_id: "sdxl", features: { block_swap: true } } };
  const base: FormValues = { "model.type": "sdxl", blocks_to_swap: 6 };

  it("hidden without gradient_release", () => {
    expect(fieldVisible(field, base, caps)).toBe(false);
    expect(
      fieldVisible(field, { ...base, "optimizer.extra_params": { gradient_release: false } }, caps)
    ).toBe(false);
  });

  it("hidden when blocks_to_swap is 0", () => {
    expect(
      fieldVisible(
        field,
        { "model.type": "sdxl", blocks_to_swap: 0, "optimizer.extra_params": { gradient_release: true } },
        caps
      )
    ).toBe(false);
  });

  it("shown when blocks swapped and gradient_release truthy (bool or string)", () => {
    expect(
      fieldVisible(field, { ...base, "optimizer.extra_params": { gradient_release: true } }, caps)
    ).toBe(true);
    expect(
      fieldVisible(field, { ...base, "optimizer.extra_params": { gradient_release: "true" } }, caps)
    ).toBe(true);
  });

  it("hidden when the model lacks the block_swap capability", () => {
    expect(
      fieldVisible(
        field,
        { "model.type": "other", blocks_to_swap: 6, "optimizer.extra_params": { gradient_release: true } },
        { other: { type_id: "other", features: {} } }
      )
    ).toBe(false);
  });
});

describe("adapterOptionsForModel", () => {
  const caps: ModelCapabilities = {
    sdxl: {
      type_id: "sdxl",
      adapters: ["lora", "lokr", "lycoris_locon"],
      adapter_labels: { lora: "LoRA (PEFT)", lokr: "LoKr", lycoris_locon: "LyCORIS · LoCon" },
    },
    other: {
      type_id: "other",
      adapters: ["lora", "novelkind"],
    },
  };

  it("returns labeled options using adapter_labels", () => {
    const opts = adapterOptionsForModel(caps, "sdxl");
    expect(opts).toEqual([
      { value: "lora", label: "LoRA (PEFT)" },
      { value: "lokr", label: "LoKr" },
      { value: "lycoris_locon", label: "LyCORIS · LoCon" },
    ]);
  });

  it("falls back to the raw kind when no label exists", () => {
    const opts = adapterOptionsForModel(caps, "other");
    expect(opts).toEqual([
      { value: "lora", label: "lora" },
      { value: "novelkind", label: "novelkind" },
    ]);
  });

  it("returns empty array for unknown model type", () => {
    expect(adapterOptionsForModel(caps, "nonexistent")).toEqual([]);
  });

  it("returns empty array for null capabilities", () => {
    expect(adapterOptionsForModel(null, "sdxl")).toEqual([]);
  });
});
