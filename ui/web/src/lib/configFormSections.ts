/**
 * Fewer form tabs: each tab merges related schema sections.
 * Section ids must match rengu_flow_ui/config_schema.py get_sections().
 */
import type { SchemaField } from "../types/forms";

export interface ConfigSchemaSection {
  id: string;
  title?: string;
  description?: string;
  fields?: SchemaField[];
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
    description: "Optimizer, schedule, cache, loop, loss, compile, DataLoader, TREAD, OOM recovery, and memory.",
    sectionIds: [
      "optimizer", "scheduler", "te_cache",
      "training", "loss", "compile", "dataloader", "tread", "oom_recovery", "logging_debug",
      "memory", "deepspeed",
    ],
  },
  {
    id: "checkpoints",
    label: "Checkpoints",
    description: "Checkpoint cadence, retention, and model export.",
    sectionIds: ["checkpoint"],
  },
  {
    id: "previews",
    label: "Sampling",
    description: "Sample images during training: global defaults + per-prompt sampling rows.",
    sectionIds: ["preview"],
  },
  {
    id: "extras",
    label: "Eval & more",
    description: "Evaluation datasets and experiment tracking.",
    sectionIds: ["eval", "monitoring"],
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
