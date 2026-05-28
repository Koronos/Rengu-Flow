import { fieldEffectiveValue, isFormValueFilled } from "./formUtils";
import type { FormValues, SchemaField } from "../types/forms";

export type DirectoryOverrideStatus = "inherited" | "overridden";

export const DIRECTORY_PRIMARY_PATHS = new Set(["path", "num_repeats", "directory_caption"]);

export type DirectoryFormRow = FormValues & {
  path: string;
  num_repeats: number;
  [key: string]: unknown;
};

interface DatasetSchemaWithDirectoryFields {
  directory_fields?: SchemaField[];
}

export function emptyDirectoryRow(): DirectoryFormRow {
  return { path: "", num_repeats: 1 };
}

export function primaryDirectoryFields(schema: DatasetSchemaWithDirectoryFields | null): SchemaField[] {
  const fields = schema?.directory_fields || [];
  return fields.filter((f) => DIRECTORY_PRIMARY_PATHS.has(f.path));
}

export function overrideDirectoryFields(schema: DatasetSchemaWithDirectoryFields | null): SchemaField[] {
  const fields = schema?.directory_fields || [];
  return fields.filter((f) => !DIRECTORY_PRIMARY_PATHS.has(f.path));
}

/** Initial value when the user turns on an optional dataset/directory field. */
export function initialValueForOptionalField(field: SchemaField): unknown {
  if (
    "default" in field &&
    field.default !== undefined &&
    field.default !== null
  ) {
    return field.default;
  }
  switch (field.type) {
    case "boolean":
      if (field.path === "enable_ar_bucket") return true;
      return false;
    case "integer":
      return field.min ?? 0;
    case "number":
      return 0;
    case "integer_list":
    case "number_list":
      return [];
    case "json":
      if (field.path === "_dataset_augmentation") {
        return JSON.stringify({ enabled: false, preset: "none" }, null, 2);
      }
      if (field.path === "size_buckets") {
        return [[512, 512, 1]];
      }
      return "{}";
    case "string":
      return "";
    default:
      return "";
  }
}

/** Every optional [[directory]] override (non-primary row identity) gets Inherited/Override. */
export function needsDirectoryOverrideToggle(field: SchemaField): boolean {
  return !DIRECTORY_PRIMARY_PATHS.has(field.path);
}

/** Whether this key is stored on the [[directory]] row (vs inherited from TOML root). */
export function directoryFieldWritesToToml(
  field: SchemaField,
  entry: FormValues | null
): boolean {
  if (!entry || !field.path) return false;
  return Object.prototype.hasOwnProperty.call(entry, field.path);
}

export function directoryFieldOverrideStatus(
  field: SchemaField,
  entry: FormValues | null
): DirectoryOverrideStatus {
  return directoryFieldWritesToToml(field, entry) ? "overridden" : "inherited";
}

/** Per-directory fields that are not optional root overrides (e.g. shuffle_tags). */
export function explicitDirectoryOverrideFields(fields: SchemaField[]): SchemaField[] {
  return fields.filter((f) => !f.show_if_set && !f.show_when_field);
}

/** Optional keys that replace a root-level dataset default when enabled. */
export function optionalRootOverrideFields(fields: SchemaField[]): SchemaField[] {
  return fields.filter((f) => !!f.show_if_set && !f.show_when_field);
}

/** Fields shown when a parent override is enabled (e.g. cache_shuffle_num). */
export function conditionalDirectoryOverrideFields(fields: SchemaField[]): SchemaField[] {
  return fields.filter((f) => !!f.show_when_field);
}

function formatHintValue(value: unknown): string {
  if (typeof value === "boolean") return value ? "on" : "off";
  if (Array.isArray(value)) {
    if (!value.length) return "empty";
    return value.map(String).join(", ");
  }
  if (typeof value === "object" && value !== null) return "configured";
  return String(value);
}

/** Short summary of the dataset-default value for inherited directory fields. */
export function globalFieldDisplayHint(
  field: SchemaField,
  globalForm: FormValues | null
): string | null {
  if (!globalForm) return null;
  const value = fieldEffectiveValue(field, globalForm);
  if (value === undefined || value === null || value === "") return null;
  return formatHintValue(value);
}

export const isOverrideEnabled = directoryFieldWritesToToml;

function directoryParentValue(
  parentPath: string,
  entry: FormValues,
  globalForm: FormValues | null
): unknown {
  if (Object.prototype.hasOwnProperty.call(entry, parentPath)) {
    return entry[parentPath];
  }
  if (globalForm && Object.prototype.hasOwnProperty.call(globalForm, parentPath)) {
    return globalForm[parentPath];
  }
  return undefined;
}

/** Whether a dependent override block should appear (parent on this row or dataset default). */
export function directoryOverrideBlockVisible(
  field: SchemaField,
  entry: FormValues | null,
  globalForm: FormValues | null
): boolean {
  const parent = field.show_when_field;
  if (!parent) return true;
  if (!entry) return false;
  return !!directoryParentValue(parent, entry, globalForm);
}

export function setOverrideEnabled(
  field: SchemaField,
  entry: FormValues | null,
  enabled: boolean
): DirectoryFormRow {
  const next = { ...(entry ?? {}) } as DirectoryFormRow;
  if (enabled) {
    if (!Object.prototype.hasOwnProperty.call(next, field.path)) {
      next[field.path] = initialValueForOptionalField(field);
    }
  } else {
    delete next[field.path];
  }
  return next;
}

export function countDirectoryOverrides(entry: FormValues | null | undefined): number {
  if (!entry || typeof entry !== "object") return 0;
  let n = 0;
  for (const key of Object.keys(entry)) {
    if (DIRECTORY_PRIMARY_PATHS.has(key)) continue;
    if (key === "augmentation") {
      if (isFormValueFilled(entry[key])) n += 1;
      continue;
    }
    if (isFormValueFilled(entry[key])) n += 1;
  }
  return n;
}

export function basenameFromPath(path: string | null | undefined): string {
  const p = (path || "").trim().replace(/\/+$/, "");
  if (!p) return "";
  const parts = p.split(/[/\\]/);
  return parts[parts.length - 1] || p;
}
