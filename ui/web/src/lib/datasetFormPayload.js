/** Normalize dataset editor form before API render/parse (plain JSON-safe objects). */

const INTEGER_LIST_KEYS = new Set(["resolutions", "frame_buckets"]);
const NUMBER_LIST_KEYS = new Set(["ar_buckets"]);
const JSON_KEYS = new Set(["size_buckets", "_dataset_augmentation"]);

function clonePlain(raw) {
  try {
    return structuredClone(raw);
  } catch {
    return JSON.parse(JSON.stringify(raw));
  }
}

function cleanListFieldValue(key, value) {
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

function cleanDirectoryRow(entry) {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
    return { path: "", num_repeats: 1 };
  }
  const row = { ...entry };
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

/**
 * @param {unknown} raw
 * @returns {Record<string, unknown> | null}
 */
export function sanitizeDatasetForm(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  let form;
  try {
    form = clonePlain(raw);
  } catch {
    return null;
  }

  let directories = form._directories;
  if (directories === undefined) {
    form._directories = [];
  } else if (typeof directories === "string") {
    try {
      directories = JSON.parse(directories);
    } catch {
      directories = [];
    }
    form._directories = Array.isArray(directories) ? directories : [];
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
