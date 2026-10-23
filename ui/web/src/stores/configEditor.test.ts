import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../api", () => ({ api: { renderToml: vi.fn(async () => ({ ok: true, content: "" })), parseToml: vi.fn() } }));

import { useConfigEditorStore } from "./configEditor";
import type { SchemaField } from "../types/forms";

const forModels = (...ids: string[]) => ({ field: "model.type", in: ids });
const sampler = (path: string, def: number, ...ids: string[]): SchemaField => ({
  path,
  type: "number",
  default: def,
  visibility: forModels(...ids),
});

// The sampler fields as rengu_flow_ui.config_schema emits them: one per model, gated by model.type.
const SCHEMA = {
  registries: {
    model_capabilities: {
      sdxl: { type_id: "sdxl", adapters: ["lora"], full_finetune: true, model_fields: [] },
      moss_image: { type_id: "moss_image", adapters: ["lora"], full_finetune: true, model_fields: [] },
    },
  },
  sections: [
    {
      id: "preview",
      fields: [
        sampler("preview.num_inference_steps", 20, "sdxl"),
        sampler("preview.guidance_scale", 7, "sdxl"),
        sampler("preview.num_inference_steps", 50, "moss_image"),
        sampler("preview.guidance_scale", 3, "moss_image"),
      ],
    },
  ],
};

describe("configEditor store: switching model.type", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setActivePinia(createPinia());
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  function storeOn(form: Record<string, unknown>) {
    const store = useConfigEditorStore();
    store.schema = SCHEMA;
    store.form = form;
    return store;
  }

  it("does not pin the previous model's preview defaults (sdxl -> moss_image)", () => {
    const store = storeOn({
      "model.type": "sdxl",
      "preview.num_inference_steps": 20,
      "preview.guidance_scale": 7,
    });
    store.patchFormField("model.type", "moss_image");
    expect(store.form?.["model.type"]).toBe("moss_image");
    expect(store.form?.["preview.num_inference_steps"]).toBeUndefined();
    expect(store.form?.["preview.guidance_scale"]).toBeUndefined();
  });

  it("keeps a value the user edited and releases the untouched one", () => {
    const store = storeOn({
      "model.type": "sdxl",
      "preview.num_inference_steps": 33,
      "preview.guidance_scale": 7,
    });
    store.patchFormField("model.type", "moss_image");
    expect(store.form?.["preview.num_inference_steps"]).toBe(33);
    expect(store.form?.["preview.guidance_scale"]).toBeUndefined();
  });
});
