/** Normalize training config form before API render/parse. */

import { clonePlain } from "./clonePlain";
import { pruneFormForModel } from "./formUtils";
import { pruneOptimizerForm } from "./optimizerForm";
import { pruneSchedulerForm } from "./schedulerForm";
import type { FormValues, ModelCapabilities } from "../types/forms";

export function sanitizeConfigForm(
  raw: unknown,
  capabilities: ModelCapabilities | null = {}
): FormValues | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  let form: FormValues;
  try {
    form = clonePlain(raw) as FormValues;
  } catch {
    return null;
  }
  for (const key of Object.keys(form)) {
    if (form[key] === undefined) {
      delete form[key];
    }
  }
  return pruneSchedulerForm(pruneOptimizerForm(pruneFormForModel(form, capabilities)));
}
