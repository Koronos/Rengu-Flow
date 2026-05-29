/**
 * Pure per-section logic for the config form editor: field partitioning,
 * importance tiers, visibility, and attention counts. Kept framework-free so
 * both the section card (rendering) and the editor (tab badges) share one
 * source of truth, and so the rules are unit-testable.
 *
 * Section ids must match rengu_flow_ui/config_schema.py get_sections().
 */
import type { ConfigSchemaSection } from "./configFormSections";
import { fieldIsFilled, fieldVisible } from "./formUtils";
import type { FormValues, ModelCapabilities, SchemaField } from "../types/forms";

export const ADAPTER_MODE_PATH = "_has_adapter";
/** Shown at the top of the config editor page, not inside the Setup tab. */
export const PINNED_TOP_FIELD_PATH = "run_name";
export const PREVIEW_PROMPTS_PATH = "preview.prompts";

export type FieldTier = "required" | "recommended" | "advanced";

export interface SectionPartition {
  required: SchemaField[];
  recommended: SchemaField[];
  advanced: SchemaField[];
}

export function fieldImportance(field: SchemaField): FieldTier {
  if (field.path === "output_dir") return "advanced";
  if (
    field.importance === "required" ||
    field.importance === "recommended" ||
    field.importance === "advanced"
  ) {
    return field.importance;
  }
  if (field.required) return "required";
  if (field.recommended) return "recommended";
  return "advanced";
}

export function isPreviewListField(field: SchemaField): boolean {
  return field.path === PREVIEW_PROMPTS_PATH || field.type === "preview_entries";
}

function isPinnedAdapterField(section: ConfigSchemaSection, field: SchemaField): boolean {
  return section.id === "adapter" && field.path === ADAPTER_MODE_PATH;
}

export function adapterModeField(
  section: ConfigSchemaSection,
  values: FormValues,
  caps: ModelCapabilities
): SchemaField | null {
  if (section.id !== "adapter") return null;
  const field = (section.fields || []).find((f) => f.path === ADAPTER_MODE_PATH);
  if (!field || !fieldVisible(field, values, caps)) return null;
  return field;
}

export function partitionSectionFields(
  section: ConfigSchemaSection,
  values: FormValues,
  caps: ModelCapabilities
): SectionPartition {
  const visible = (section.fields || []).filter(
    (f) =>
      fieldVisible(f, values, caps) &&
      !isPinnedAdapterField(section, f) &&
      f.path !== PINNED_TOP_FIELD_PATH &&
      !isPreviewListField(f)
  );
  return {
    required: visible.filter((f) => fieldImportance(f) === "required"),
    recommended: visible.filter((f) => fieldImportance(f) === "recommended"),
    advanced: visible.filter((f) => fieldImportance(f) === "advanced"),
  };
}

export function sectionHasVisibleFields(
  section: ConfigSchemaSection,
  values: FormValues,
  caps: ModelCapabilities
): boolean {
  if (adapterModeField(section, values, caps)) return true;
  if (section.id === "preview") return true;
  const p = partitionSectionFields(section, values, caps);
  return p.required.length + p.recommended.length + p.advanced.length > 0;
}

export function unfilledRequiredCount(
  section: ConfigSchemaSection,
  values: FormValues,
  caps: ModelCapabilities
): number {
  return partitionSectionFields(section, values, caps).required.filter(
    (f) => !fieldIsFilled(f, values)
  ).length;
}

export function sectionAttentionCount(
  section: ConfigSchemaSection,
  values: FormValues,
  caps: ModelCapabilities
): number {
  let extra = 0;
  if (section.id === "preview") {
    const prompts = values[PREVIEW_PROMPTS_PATH];
    if (!Array.isArray(prompts) || prompts.length === 0) extra = 1;
  }
  return unfilledRequiredCount(section, values, caps) + extra;
}
