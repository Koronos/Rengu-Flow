/** Training-config `rengu-flow-dataset:` refs (optional `:label` suffix for TOML readability). */

import type { DatasetLibraryRefParts } from "../types/forms";

export const DATASET_REF_PREFIX = "rengu-flow-dataset:";

export function parseDatasetLibraryRef(value: unknown): DatasetLibraryRefParts {
  const s = String(value ?? "").trim();
  if (!s.startsWith(DATASET_REF_PREFIX)) {
    return { isRef: false, id: null, label: null, canonical: s };
  }
  const rest = s.slice(DATASET_REF_PREFIX.length).trim();
  if (!rest) {
    return { isRef: true, id: null, label: null, canonical: DATASET_REF_PREFIX };
  }
  const colon = rest.indexOf(":");
  if (colon === -1) {
    const id = rest;
    return {
      isRef: true,
      id,
      label: null,
      canonical: `${DATASET_REF_PREFIX}${id}`,
    };
  }
  const id = rest.slice(0, colon).trim();
  const label = rest.slice(colon + 1).trim();
  return {
    isRef: true,
    id,
    label: label || null,
    canonical: `${DATASET_REF_PREFIX}${id}`,
  };
}

/** Alias aligned with Python `dataset_library_ref()`. */
export function formatDatasetLibraryRef(id: string | number, label?: string | null): string {
  const base = `${DATASET_REF_PREFIX}${id}`;
  const text = (label ?? "").trim();
  return text ? `${base}:${text}` : base;
}

export function canonicalDatasetRef(value: unknown): string {
  const p = parseDatasetLibraryRef(value);
  return p.isRef ? p.canonical : String(value ?? "").trim();
}

/** Tag / table label: display suffix, else library id, else full path. */
export function datasetRefDisplayLabel(value: unknown): string {
  const p = parseDatasetLibraryRef(value);
  if (p.isRef) {
    return p.label || p.id || String(value);
  }
  return String(value);
}

export function isLibraryDatasetRef(value: unknown): boolean {
  return parseDatasetLibraryRef(value).isRef;
}

/** Numeric library id from a ref string, or null if not a valid library ref. */
export function libraryDatasetIdFromRef(value: unknown): string | null {
  const p = parseDatasetLibraryRef(value);
  if (!p.isRef || !p.id || !/^\d+$/.test(p.id)) {
    return null;
  }
  return p.id;
}

/**
 * Normalize training `dataset` form value to a list of path/ref strings.
 * Recovers values corrupted by `String(array)` (comma-joined refs).
 */
export function coerceTrainingDatasetEntries(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .filter((e): e is string => typeof e === "string" && e.trim().length > 0)
      .map((e) => e.trim());
  }
  if (typeof value === "string") {
    const s = value.trim();
    if (!s) return [];
    const splitMerged = s.split(
      new RegExp(`,(?=${DATASET_REF_PREFIX.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`)
    );
    if (splitMerged.length > 1) {
      return splitMerged.map((p) => p.trim()).filter(Boolean);
    }
    return [s];
  }
  return [];
}

/** Form model for TrainingDatasetsField: "" | one path | several paths. */
export function trainingDatasetFormValue(
  entries: string[]
): string | string[] {
  if (entries.length === 0) return "";
  if (entries.length === 1) return entries[0];
  return entries;
}

/** Append paths not already present (by canonical ref). */
export function appendUniqueDatasetPaths<T extends string>(
  existing: T[],
  added: string[]
): T[] {
  const seen = new Set(existing.map(canonicalDatasetRef));
  const next = [...existing];
  for (const p of added) {
    const key = canonicalDatasetRef(p);
    if (!seen.has(key)) {
      next.push(p as T);
      seen.add(key);
    }
  }
  return next;
}
