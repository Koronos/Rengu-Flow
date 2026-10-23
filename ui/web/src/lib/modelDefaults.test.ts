import { describe, expect, it } from "vitest";
import { allSchemaFields, releaseStaleModelDefaults } from "./modelDefaults";
import type { SchemaField } from "../types/forms";

const forModels = (...ids: string[]) => ({ field: "model.type", in: ids });
const preview = (path: string, def: number, ...ids: string[]): SchemaField => ({
  path,
  type: "number",
  default: def,
  visibility: forModels(...ids),
});

// One field per model group for the sampler defaults, as rengu_flow_ui.config_schema emits them.
const SCHEMA = {
  sections: [
    {
      id: "preview",
      fields: [
        preview("preview.num_inference_steps", 20, "sdxl"),
        preview("preview.guidance_scale", 7, "sdxl"),
        preview("preview.num_inference_steps", 28, "qwen_image21"),
        preview("preview.guidance_scale", 1, "qwen_image21"),
        preview("preview.num_inference_steps", 50, "moss_image"),
        preview("preview.guidance_scale", 3, "moss_image"),
        { path: "preview.width", type: "integer", default: 1024 },
      ] as SchemaField[],
    },
    { id: "model", fields: [{ path: "model.type", type: "select" }] as SchemaField[] },
  ],
};
const FIELDS = allSchemaFields(SCHEMA);

const switchTo = (prev: Record<string, unknown>, type: string) =>
  releaseStaleModelDefaults(prev, { ...prev, "model.type": type }, FIELDS, {});

describe("releaseStaleModelDefaults", () => {
  it("lists the fields of every section", () => {
    expect(FIELDS).toHaveLength(8);
  });

  it("releases values still at the previous model's default (sdxl -> moss_image)", () => {
    const prev = { "model.type": "sdxl", "preview.num_inference_steps": 20, "preview.guidance_scale": 7 };
    const next = switchTo(prev, "moss_image");
    expect(next["model.type"]).toBe("moss_image");
    expect("preview.num_inference_steps" in next).toBe(false);
    expect("preview.guidance_scale" in next).toBe(false);
  });

  it("keeps what the user edited and releases only what was left at the default", () => {
    const prev = { "model.type": "sdxl", "preview.num_inference_steps": 33, "preview.guidance_scale": 7 };
    const next = switchTo(prev, "qwen_image21");
    expect(next["preview.num_inference_steps"]).toBe(33);
    expect("preview.guidance_scale" in next).toBe(false);
  });

  it("leaves unset fields unset and fields without per-model variants alone", () => {
    const prev = { "model.type": "sdxl", "preview.width": 1024, "output_dir": "out" };
    const next = switchTo(prev, "moss_image");
    expect(next).toEqual({ ...prev, "model.type": "moss_image" });
  });

  it("follows the defaults across consecutive switches", () => {
    let form: Record<string, unknown> = { "model.type": "sdxl", "preview.num_inference_steps": 20 };
    form = switchTo(form, "moss_image");
    form = { ...form, "preview.num_inference_steps": 50 }; // a control re-emits the shown default
    form = switchTo(form, "qwen_image21");
    expect("preview.num_inference_steps" in form).toBe(false);
  });

  it("keeps a value the new model does not show a default for", () => {
    const prev = { "model.type": "sdxl", "preview.num_inference_steps": 20 };
    const next = switchTo(prev, "cosmos_predict2"); // no variant visible for it
    expect(next["preview.num_inference_steps"]).toBe(20);
  });
});
