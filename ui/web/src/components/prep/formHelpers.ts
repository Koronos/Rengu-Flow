/** Shared helpers for the extracted prep stage forms. */

/** Wrap plain help text as the `SchemaField`-ish shape `FieldHelpIcon` expects. */
export function help(text: string) {
  return { path: "", type: "string", help: text, doc_path: "docs/user/dataset-prep.md" };
}

/**
 * Copy only the keys the target form already knows; unknown/stale keys are dropped.
 *
 * `null` is skipped (an older config's gap keeps the form default) except for the keys in
 * `nullable`, where it is a saved choice — e.g. a cleared `temperature` meaning "the model's
 * recommended value". Skipping it there would put the default back and, in the workflow drawer,
 * autosave that default over what the user cleared.
 */
export function copyKnown(
  target: Record<string, unknown>,
  src: unknown,
  nullable: readonly string[] = [],
): void {
  if (!src || typeof src !== "object") return;
  const s = src as Record<string, unknown>;
  for (const key of Object.keys(target)) {
    if (s[key] === undefined) continue;
    if (s[key] === null && !nullable.includes(key)) continue;
    target[key] = s[key];
  }
}

/** Sampling overrides where `null` means "use the model's recommended value". */
export const NULLABLE_SAMPLING = ["temperature", "top_p"] as const;
