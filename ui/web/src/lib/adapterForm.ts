/**
 * Adapter (network) form helpers. Keep in sync with rengu_flow_ui/adapter_form.py.
 *
 * When the adapter `type` changes, keys belonging to the previous network must be
 * dropped — otherwise they linger in the form/TOML and only surface as errors when
 * the validator runs — and the new type's meaningful defaults seeded. Mirrors the
 * optimizer/scheduler/model type-change handlers. Only `adapter.*` keys are touched;
 * other sections are never modified.
 */
import { fieldVisible } from "./formUtils";
import type { FormValues, ModelCapabilities, SchemaField } from "../types/forms";

const ADAPTER_TYPE_PATH = "adapter.type";

/** Adapter-section fields declared in the schema (paths under `adapter.`). */
export function adapterFieldsFromSchema(schema: Record<string, unknown> | null): SchemaField[] {
  const sections =
    (schema?.sections as { id?: string; fields?: SchemaField[] }[] | undefined) ?? [];
  const adapter = sections.find((s) => s.id === "adapter");
  return (adapter?.fields ?? []).filter(
    (f) => typeof f.path === "string" && f.path.startsWith("adapter.")
  );
}

/** A default worth writing: skip false / 0 / "" / null / empty (absent ≡ that value). */
function isMeaningfulDefault(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (value === false) return false;
  if (typeof value === "number" && value === 0) return false;
  if (typeof value === "string" && value.trim() === "") return false;
  if (Array.isArray(value) && value.length === 0) return false;
  return true;
}

/** Drop `adapter.*` keys not valid for the form's current `adapter.type`. */
export function pruneAdapterForm(
  form: FormValues,
  fields: SchemaField[],
  caps: ModelCapabilities | null = null
): FormValues {
  const type = form[ADAPTER_TYPE_PATH];
  if (type === undefined || type === null || type === "") return form;
  const allowed = new Set<string>([ADAPTER_TYPE_PATH]);
  for (const field of fields) {
    if (fieldVisible(field, form, caps || {})) allowed.add(field.path);
  }
  const next = { ...form };
  for (const key of Object.keys(next)) {
    if (key.startsWith("adapter.") && !allowed.has(key)) delete next[key];
  }
  return next;
}

/** Switch `adapter.type`: prune the old type's keys, seed the new type's defaults. */
export function applyAdapterTypeChange(
  form: FormValues,
  nextType: unknown,
  schema: Record<string, unknown> | null,
  caps: ModelCapabilities | null = null
): FormValues {
  const fields = adapterFieldsFromSchema(schema);
  let next: FormValues = { ...form, [ADAPTER_TYPE_PATH]: nextType };
  next = pruneAdapterForm(next, fields, caps);
  for (const field of fields) {
    if (field.path === ADAPTER_TYPE_PATH) continue;
    if (field.path in next) continue;
    if (!fieldVisible(field, next, caps || {})) continue;
    if (isMeaningfulDefault(field.default)) next[field.path] = field.default;
  }
  return next;
}
