/**
 * Fewer form tabs: each tab merges related schema sections.
 * Section ids must match renga_flow_ui/config_schema.py get_sections().
 */
import type { SchemaField } from "../types/forms";

export interface ConfigSchemaSection {
  id: string;
  title?: string;
  description?: string;
  fields?: SchemaField[];
  flat_optional?: boolean;
}

export interface ConfigFormTabGroup {
  id: string;
  label: string;
  description: string;
  sectionIds: string[];
}

export interface ConfigFormTab extends ConfigFormTabGroup {
  sections: ConfigSchemaSection[];
}

export const CONFIG_FORM_TAB_GROUPS: ConfigFormTabGroup[] = [
  {
    id: "setup",
    label: "Setup",
    description: "Dataset, output folder, model checkpoint, and adapter.",
    sectionIds: ["general", "model", "adapter"],
  },
  {
    id: "training",
    label: "Training",
    description: "Optimizer, learning-rate schedule, loop, and checkpoints.",
    sectionIds: ["optimizer", "scheduler", "training", "checkpoint"],
  },
  {
    id: "extras",
    label: "Eval & more",
    description: "Evaluation, sample previews, and experiment tracking.",
    sectionIds: ["eval", "preview", "monitoring"],
  },
];

export function buildConfigFormTabs(
  schemaSections: ConfigSchemaSection[] = [],
  isSectionVisible: (section: ConfigSchemaSection) => boolean
): ConfigFormTab[] {
  const byId = Object.fromEntries(schemaSections.map((s) => [s.id, s]));
  return CONFIG_FORM_TAB_GROUPS.map((tab) => ({
    ...tab,
    sections: tab.sectionIds
      .map((id) => byId[id])
      .filter(Boolean)
      .filter((sec) => isSectionVisible(sec as ConfigSchemaSection)),
  })).filter((tab) => tab.sections.length > 0);
}

export { schemaFieldColSpan as configFieldColSpan } from "./schemaFieldLayout";
