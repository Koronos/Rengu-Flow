/** Normalize training config form before API render/parse. */

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
    // JSON round-trip rather than structuredClone: the form holds JSON-data bound for TOML,
    // and values copied from the reactive schema registry (e.g. optimizer KV defaults with a
    // nested `betas` array) are Vue reactive proxies that structuredClone refuses to clone.
    // That threw here, sanitize returned null, and setForm then silently dropped the update —
    // the optimizer picker looked frozen. JSON reads through proxies and yields plain data.
    form = JSON.parse(JSON.stringify(raw)) as FormValues;
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
