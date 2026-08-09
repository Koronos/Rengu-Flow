/** Shared helpers for the extracted prep stage forms. */

/** Wrap plain help text as the `SchemaField`-ish shape `FieldHelpIcon` expects. */
export function help(text: string) {
  return { path: "", type: "string", help: text, doc_path: "docs/user/dataset-prep.md" };
}

/** Copy only the keys the target form already knows; unknown/stale keys are dropped. */
export function copyKnown(target: Record<string, unknown>, src: unknown): void {
  if (!src || typeof src !== "object") return;
  const s = src as Record<string, unknown>;
  for (const key of Object.keys(target)) {
    if (s[key] !== undefined && s[key] !== null) target[key] = s[key];
  }
}
