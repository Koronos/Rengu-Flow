/**
 * Fewer form tabs: each tab merges related schema sections.
 * Section ids must match renga_flow_ui/config_schema.py get_sections().
 */
export const CONFIG_FORM_TAB_GROUPS = [
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

/** @param {object[]} schemaSections */
export function buildConfigFormTabs(schemaSections, isSectionVisible) {
  const byId = Object.fromEntries((schemaSections || []).map((s) => [s.id, s]));
  return CONFIG_FORM_TAB_GROUPS.map((tab) => ({
    ...tab,
    sections: tab.sectionIds
      .map((id) => byId[id])
      .filter(Boolean)
      .filter((sec) => isSectionVisible(sec)),
  })).filter((tab) => tab.sections.length > 0);
}

export function configFieldColSpan(field) {
  const path = field.path || "";
  if (
    field.type === "json" ||
    field.type === "integer_list" ||
    field.type === "number_list"
  ) {
    return 24;
  }
  if (field.type === "boolean") {
    return 12;
  }
  if (
    path === "dataset" ||
    path === "eval_datasets" ||
    path === "output_dir" ||
    path.includes("path") ||
    path.endsWith("_dir")
  ) {
    return 24;
  }
  if (field.type === "string" && (field.allow_custom || (field.options?.length ?? 0) > 6)) {
    return 24;
  }
  return 12;
}
