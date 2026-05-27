/** Normalize training config form before API render/parse. */

import { pruneFormForModel } from "./formUtils";

function clonePlain(raw) {
  try {
    return structuredClone(raw);
  } catch {
    return JSON.parse(JSON.stringify(raw));
  }
}

/**
 * @param {unknown} raw
 * @param {Record<string, unknown>} capabilities
 * @returns {Record<string, unknown> | null}
 */
export function sanitizeConfigForm(raw, capabilities = {}) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  let form;
  try {
    form = clonePlain(raw);
  } catch {
    return null;
  }
  for (const key of Object.keys(form)) {
    if (form[key] === undefined) {
      delete form[key];
    }
  }
  return pruneFormForModel(form, capabilities);
}
