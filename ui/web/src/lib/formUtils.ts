/**
 * Config form helpers. Field visibility must match renga_flow_ui/field_visibility.py
 * (schema fields include a normalized `visibility` tree from the server).
 */

import { parseIntegerList } from "./integerList";
import { parseNumberList } from "./numberList";
import { parseStringList, stringListNeedsJsonEditor } from "./stringList";
import type {
  FormValues,
  ModelCapabilities,
  ModelCapability,
  RawListInput,
  SchemaField,
  VisibilityClause,
} from "../types/forms";

export function getFormValue(form: FormValues, path: string): unknown {
  if (path in form) return form[path];
  return undefined;
}

export function setFormValue(form: FormValues, path: string, value: unknown): void {
  form[path] = value;
}

export function isFormValueFilled(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") return value.trim() !== "";
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value as object).length > 0;
  return true;
}

/** Value shown in controls: explicit form/TOML value, else schema default. */
export function fieldEffectiveValue(field: SchemaField, form: FormValues): unknown {
  if (field.path in form) {
    return form[field.path];
  }
  if ("default" in field) {
    return field.default;
  }
  if (field.type === "boolean") {
    return false;
  }
  return undefined;
}

/** Dataset-only visibility (flat keys, no model.type). */
export function datasetFieldVisible(field: SchemaField, form: FormValues): boolean {
  const path = field.path;
  if (field.show_if_set && Object.prototype.hasOwnProperty.call(form, path)) {
    return true;
  }
  if (field.show_if_set && isFormValueFilled(form[path])) {
    return true;
  }
  if (field.show_when_field) {
    if (Object.prototype.hasOwnProperty.call(form, path)) {
      return true;
    }
    if (isFormValueFilled(form[path])) {
      return true;
    }
    if (form[field.show_when_field]) {
      return true;
    }
    return false;
  }
  if (field.show_if_set) {
    return false;
  }
  return true;
}

function listFieldIsFilled(
  field: SchemaField,
  form: FormValues,
  parseList: (value: RawListInput) => unknown[]
): boolean {
  const raw = (field.path in form
    ? form[field.path]
    : "default" in field
      ? field.default
      : []) as RawListInput;
  if (field.type === "string_list" && stringListNeedsJsonEditor(raw)) {
    return isFormValueFilled(raw);
  }
  return parseList(raw).length > 0;
}

/** True when the field has a user/TOML value or a schema default will apply at train time. */
export function fieldIsFilled(field: SchemaField, form: FormValues): boolean {
  if (field.type === "integer_list") {
    return listFieldIsFilled(field, form, parseIntegerList);
  }
  if (field.type === "number_list") {
    return listFieldIsFilled(field, form, parseNumberList);
  }
  if (field.type === "string_list") {
    return listFieldIsFilled(field, form, parseStringList);
  }
  if (field.path in form) {
    return isFormValueFilled(form[field.path]);
  }
  if (!("default" in field)) {
    return false;
  }
  const def = field.default;
  if (def === undefined || def === null || def === "") {
    return false;
  }
  return true;
}

export function normalizeModelType(
  modelType: unknown,
  capabilities: ModelCapabilities | null
): string | undefined {
  if (!modelType || !capabilities) return modelType ? String(modelType) : undefined;
  const key = String(modelType).toLowerCase();
  if (capabilities[key]) return capabilities[key].type_id || key;
  for (const cap of Object.values(capabilities)) {
    const aliases = (cap.aliases || []).map((a) => a.toLowerCase());
    if (aliases.includes(key)) return cap.type_id;
  }
  return key;
}

export function getModelCapability(
  capabilities: ModelCapabilities | null,
  modelType: unknown
): ModelCapability | null {
  if (!capabilities || !modelType) return null;
  const canonical = normalizeModelType(modelType, capabilities);
  return canonical ? capabilities[canonical] || null : null;
}

export function capabilityHasFeature(
  capabilities: ModelCapabilities | null,
  modelType: unknown,
  feature: string
): boolean {
  const cap = getModelCapability(capabilities, modelType);
  if (!cap || !feature) return false;
  return !!(cap.features && cap.features[feature]);
}

export function modelSupportsAdapters(cap: ModelCapability | null): boolean {
  return !!(cap && Array.isArray(cap.adapters) && cap.adapters.length > 0);
}

function evalVisibilityClause(
  clause: VisibilityClause | undefined,
  form: FormValues,
  capabilities: ModelCapabilities | null
): boolean {
  if (!clause) return true;

  if (clause.all) {
    return clause.all.every((c) => evalVisibilityClause(c, form, capabilities));
  }
  if (clause.any) {
    return clause.any.some((c) => evalVisibilityClause(c, form, capabilities));
  }
  if (clause.not) {
    return !evalVisibilityClause(clause.not, form, capabilities);
  }

  if (clause.capability !== undefined) {
    const feature = clause.capability;
    const want = clause.equals !== undefined ? clause.equals : true;
    const has = capabilityHasFeature(
      capabilities,
      getFormValue(form, "model.type"),
      feature
    );
    return want ? has : !has;
  }

  if (clause.form_nonempty !== undefined) {
    const val = getFormValue(form, clause.form_nonempty);
    if (!isFormValueFilled(val)) return false;
    if (clause.exclude_zero && (val === 0 || val === 0.0 || val === "0")) return false;
    return true;
  }

  if (clause.when_model_has_adapter) {
    const cap = getModelCapability(capabilities, getFormValue(form, "model.type"));
    if (!modelSupportsAdapters(cap)) return false;
    return getFormValue(form, "_has_adapter") === true;
  }

  const field = clause.field;
  if (field !== undefined) {
    const val = getFormValue(form, field);
    if (Object.prototype.hasOwnProperty.call(clause, "equals")) {
      return val === clause.equals;
    }
    if (clause.in) {
      return clause.in.includes(val);
    }
  }
  return true;
}

function legacyVisibility(
  field: SchemaField,
  _form: FormValues,
  _capabilities: ModelCapabilities | null
): VisibilityClause | null {
  const clauses: VisibilityClause[] = [];
  if (field.when_model_has_adapter) {
    clauses.push({ when_model_has_adapter: true });
  }
  if (field.when) clauses.push(field.when);
  if (field.when_capability) {
    clauses.push({ capability: field.when_capability });
  }
  if (field.show_if_set && field.path) {
    const entry: VisibilityClause = { form_nonempty: field.path };
    if (field.show_if_set_exclude_zero) entry.exclude_zero = true;
    clauses.push(entry);
  }
  if (!clauses.length) return null;
  if (clauses.length === 1) return clauses[0];
  return { all: clauses };
}

export function fieldVisible(
  field: SchemaField,
  form: FormValues,
  capabilities: ModelCapabilities | null = null
): boolean {
  const vis = field.visibility || legacyVisibility(field, form, capabilities);
  if (!vis) return true;
  return evalVisibilityClause(vis, form, capabilities || {});
}

export function modelSpecificPaths(capabilities: ModelCapabilities | null): Record<string, Set<string>> {
  const out: Record<string, Set<string>> = {};
  if (!capabilities) return out;
  for (const cap of Object.values(capabilities)) {
    const id = cap.type_id;
    const paths = new Set<string>();
    for (const spec of cap.model_fields || []) {
      if (spec.path && spec.ui !== false) paths.add(spec.path);
    }
    out[id] = paths;
  }
  return out;
}

export function pruneFormForModel(
  form: FormValues,
  capabilities: ModelCapabilities | null
): FormValues {
  const modelType = normalizeModelType(form["model.type"], capabilities);
  if (!modelType) return form;
  const owned = modelSpecificPaths(capabilities);
  const allowed = new Set([...(owned[modelType] || []), "model.type", "model.dtype"]);
  const all = new Set<string>();
  for (const paths of Object.values(owned)) {
    for (const p of paths) all.add(p);
  }
  const next = { ...form };
  for (const path of all) {
    if (!allowed.has(path)) delete next[path];
  }
  return next;
}

export function adapterOptionsForModel(
  capabilities: ModelCapabilities | null,
  modelType: unknown
): string[] {
  const cap = getModelCapability(capabilities, modelType);
  if (!cap?.adapters) return [];
  return cap.adapters;
}

export function trainingModesLabel(cap: ModelCapability | null): string {
  if (!cap) return "";
  const parts: string[] = [];
  if (cap.full_finetune) parts.push("full finetune");
  if (cap.adapters?.length) parts.push(...cap.adapters.map((a) => a.toUpperCase()));
  return parts.join(", ");
}

export function jsonStringify(value: unknown): string {
  if (value === undefined || value === null || value === "") return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
