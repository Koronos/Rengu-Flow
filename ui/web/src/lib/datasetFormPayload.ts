/** Normalize dataset editor form before API render/parse (plain JSON-safe objects). */

import type { FormValues } from "../types/forms";

const INTEGER_LIST_KEYS = new Set(["resolutions", "frame_buckets"]);
const NUMBER_LIST_KEYS = new Set(["ar_buckets"]);
const JSON_KEYS = new Set(["size_buckets", "_dataset_augmentation"]);

type DirectoryRow = FormValues & { path: string; num_repeats: number };
type DatasetFormValues = FormValues & { _directories?: DirectoryRow[] | string };

function cleanListFieldValue(key: string, value: unknown): unknown {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  if (Array.isArray(value) && value.length === 0) {
    return undefined;
  }
  if (
    (INTEGER_LIST_KEYS.has(key) || NUMBER_LIST_KEYS.has(key) || JSON_KEYS.has(key)) &&
    typeof value === "string" &&
    !value.trim()
  ) {
    return undefined;
  }
  return value;
}

function cleanDirectoryRow(entry: unknown): DirectoryRow {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
    return { path: "", num_repeats: 1 };
  }
  const row = { ...(entry as FormValues) } as DirectoryRow;
  row.path = typeof row.path === "string" ? row.path : "";
  row.num_repeats = Number(row.num_repeats) || 1;
  for (const key of Object.keys(row)) {
    const cleaned = cleanListFieldValue(key, row[key]);
    if (cleaned === undefined) {
      delete row[key];
    } else {
      row[key] = cleaned;
    }
  }
  return row;
}

export function sanitizeDatasetForm(raw: unknown): FormValues | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  let form: DatasetFormValues;
  try {
    form = structuredClone(raw) as DatasetFormValues;
  } catch {
    return null;
  }

  let directories: unknown = form._directories;
  if (directories === undefined) {
    form._directories = [];
  } else if (typeof directories === "string") {
    try {
      directories = JSON.parse(directories) as unknown;
    } catch {
      directories = [];
    }
    form._directories = Array.isArray(directories) ? directories.map(cleanDirectoryRow) : [];
  } else if (!Array.isArray(directories)) {
    form._directories = [];
  } else {
    form._directories = directories.map(cleanDirectoryRow);
  }

  for (const key of Object.keys(form)) {
    if (key === "_directories") continue;
    const cleaned = cleanListFieldValue(key, form[key]);
    if (cleaned === undefined) {
      delete form[key];
    } else {
      form[key] = cleaned;
    }
  }

  return form;
}
