import { isFormValueFilled } from "./formUtils";

export const DIRECTORY_PRIMARY_PATHS = new Set(["path", "num_repeats", "directory_caption"]);

export function emptyDirectoryRow() {
  return { path: "", num_repeats: 1 };
}

export function primaryDirectoryFields(schema) {
  const fields = schema?.directory_fields || [];
  return fields.filter((f) => DIRECTORY_PRIMARY_PATHS.has(f.path));
}

export function overrideDirectoryFields(schema) {
  const fields = schema?.directory_fields || [];
  return fields.filter((f) => !DIRECTORY_PRIMARY_PATHS.has(f.path));
}

/** Initial value when the user turns on an optional dataset/directory field. */
export function initialValueForOptionalField(field) {
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
        return JSON.stringify([[512, 512, 1]], null, 2);
      }
      return "{}";
    case "string":
      return "";
    default:
      return "";
  }
}

export function isOverrideEnabled(field, entry) {
  if (!field.show_if_set && !field.show_when_field) return true;
  return Object.prototype.hasOwnProperty.call(entry, field.path);
}

export function setOverrideEnabled(field, entry, enabled) {
  const next = { ...entry };
  if (enabled) {
    if (!Object.prototype.hasOwnProperty.call(next, field.path)) {
      next[field.path] = initialValueForOptionalField(field);
    }
  } else {
    delete next[field.path];
  }
  return next;
}

export function countDirectoryOverrides(entry) {
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

export function basenameFromPath(path) {
  const p = (path || "").trim().replace(/\/+$/, "");
  if (!p) return "";
  const parts = p.split(/[/\\]/);
  return parts[parts.length - 1] || p;
}
