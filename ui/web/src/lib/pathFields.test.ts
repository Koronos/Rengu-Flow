import { describe, expect, it } from "vitest";
import { isPathField, pathFieldExpect } from "./pathFields";
import type { SchemaField } from "../types/forms";

function field(path: string, extra: Partial<SchemaField> = {}): SchemaField {
  return { path, label: path, type: "string", ...extra } as SchemaField;
}

describe("pathFieldExpect", () => {
  it("treats an undeclared path field as a file", () => {
    expect(pathFieldExpect(field("model.checkpoint_path", { type: "path" }))).toBe("file");
    expect(pathFieldExpect(field("output_dir"))).toBe("dir");
  });

  it("honors the schema's path_expect over the path heuristics", () => {
    // qwen_image21: the diffusers folder / processor folder are directories…
    expect(pathFieldExpect(field("model.diffusers_path", { type: "path", path_expect: "dir" }))).toBe("dir");
    expect(pathFieldExpect(field("model.processor_path", { type: "path", path_expect: "dir" }))).toBe("dir");
    // …and component overrides take a folder or one .safetensors: existence-only check.
    expect(pathFieldExpect(field("model.transformer_path", { type: "path", path_expect: "any" }))).toBeNull();
    expect(pathFieldExpect(field("some_dir", { path_expect: "file" }))).toBe("file");
  });

  it("keeps path_expect fields validated as paths", () => {
    expect(isPathField(field("model.diffusers_path", { type: "path", path_expect: "dir" }))).toBe(true);
    expect(isPathField(field("model.vae_path", { type: "path", path_expect: "any" }))).toBe(true);
  });
});
