import { describe, expect, it } from "vitest";
import {
  ADAPTER_MODE_PATH,
  PINNED_TOP_FIELD_PATH,
  PREVIEW_PROMPTS_PATH,
  adapterModeField,
  fieldImportance,
  isPreviewListField,
  partitionSectionFields,
  sectionAttentionCount,
  sectionHasVisibleFields,
  unfilledRequiredCount,
} from "./configFormSectionLogic";
import type { ConfigSchemaSection } from "./configFormSections";
import type { SchemaField } from "../types/forms";

function field(path: string, extra: Partial<SchemaField> = {}): SchemaField {
  return { path, label: path, type: "string", ...extra } as SchemaField;
}

describe("fieldImportance", () => {
  it("forces output_dir to advanced", () => {
    expect(fieldImportance(field("output_dir", { required: true }))).toBe("advanced");
  });
  it("honors explicit importance, then required/recommended flags, else advanced", () => {
    expect(fieldImportance(field("a", { importance: "recommended" }))).toBe("recommended");
    expect(fieldImportance(field("b", { required: true }))).toBe("required");
    expect(fieldImportance(field("c", { recommended: true }))).toBe("recommended");
    expect(fieldImportance(field("d"))).toBe("advanced");
  });
});

describe("isPreviewListField", () => {
  it("matches by path or type", () => {
    expect(isPreviewListField(field(PREVIEW_PROMPTS_PATH))).toBe(true);
    expect(isPreviewListField(field("x", { type: "preview_entries" }))).toBe(true);
    expect(isPreviewListField(field("x"))).toBe(false);
  });
});

describe("partitionSectionFields", () => {
  it("splits by tier and excludes pinned / preview-list fields", () => {
    const section: ConfigSchemaSection = {
      id: "general",
      fields: [
        field("req", { required: true }),
        field("rec", { recommended: true }),
        field("adv"),
        field(PINNED_TOP_FIELD_PATH, { required: true }),
        field(PREVIEW_PROMPTS_PATH),
      ],
    };
    const p = partitionSectionFields(section, {}, {});
    expect(p.required.map((f) => f.path)).toEqual(["req"]);
    expect(p.recommended.map((f) => f.path)).toEqual(["rec"]);
    expect(p.advanced.map((f) => f.path)).toEqual(["adv"]);
  });
});

describe("adapterModeField", () => {
  it("returns the adapter-mode field only for the adapter section", () => {
    const adapterSec: ConfigSchemaSection = {
      id: "adapter",
      fields: [field(ADAPTER_MODE_PATH), field("rank")],
    };
    expect(adapterModeField(adapterSec, {}, {})?.path).toBe(ADAPTER_MODE_PATH);
    expect(adapterModeField({ id: "general", fields: [field(ADAPTER_MODE_PATH)] }, {}, {})).toBeNull();
  });
});

describe("attention counts", () => {
  it("counts empty required fields", () => {
    const section: ConfigSchemaSection = {
      id: "general",
      fields: [field("req", { required: true }), field("set", { required: true })],
    };
    expect(unfilledRequiredCount(section, { set: "value" }, {})).toBe(1);
  });
  it("adds one for an empty preview prompt list", () => {
    const preview: ConfigSchemaSection = { id: "preview", fields: [] };
    expect(sectionAttentionCount(preview, {}, {})).toBe(1);
    expect(sectionAttentionCount(preview, { [PREVIEW_PROMPTS_PATH]: [{}] }, {})).toBe(0);
  });
});

describe("sectionHasVisibleFields", () => {
  it("is true for preview and adapter-mode sections, false for empty sections", () => {
    expect(sectionHasVisibleFields({ id: "preview", fields: [] }, {}, {})).toBe(true);
    expect(sectionHasVisibleFields({ id: "general", fields: [] }, {}, {})).toBe(false);
    expect(
      sectionHasVisibleFields({ id: "general", fields: [field("a", { required: true })] }, {}, {})
    ).toBe(true);
  });
});
